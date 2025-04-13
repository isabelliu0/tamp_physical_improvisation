"""Test script for goal-conditioned RL."""

from pathlib import Path

import pytest

from tamp_improv.approaches.improvisational.base import ImprovisationalTAMPApproach
from tamp_improv.approaches.improvisational.graph_training import (
    collect_goal_conditioned_training_data,
)
from tamp_improv.approaches.improvisational.policies.base import (
    GoalConditionedTrainingData,
    PolicyContext,
)
from tamp_improv.approaches.improvisational.policies.goal_rl import (
    GoalConditionedRLConfig,
    GoalConditionedRLPolicy,
)
from tamp_improv.approaches.improvisational.policies.pushing import PushingPolicy
from tamp_improv.approaches.improvisational.training import (
    TrainingConfig,
    train_and_evaluate_goal_conditioned,
)
from tamp_improv.benchmarks.blocks2d import Blocks2DTAMPSystem
from tamp_improv.benchmarks.goal_wrapper import GoalConditionedWrapper


def test_goal_conditioned_data_collection():
    """Test collecting goal-conditioned data."""
    print("\n=== Testing Goal-Conditioned Data Collection ===")
    config = {
        "seed": 42,
        "max_steps": 50,
        "render": True,
        "collect_episodes": 1,
        "force_collect": True,
        "training_data_dir": "training_data/test_goal",
    }
    system = Blocks2DTAMPSystem.create_default(
        seed=config["seed"], render_mode="rgb_array" if config["render"] else None
    )
    policy = PushingPolicy(seed=config["seed"])
    approach = ImprovisationalTAMPApproach(system, policy, seed=config["seed"])
    train_data = collect_goal_conditioned_training_data(
        system,
        approach,
        config,
    )

    assert hasattr(train_data, "node_states"), "No node states in training data"
    assert hasattr(train_data, "valid_shortcuts"), "No valid shortcuts in training data"
    assert hasattr(train_data, "node_preimages"), "No node preimages in training data"

    print(f"Collected states for {len(train_data.node_states)} nodes")
    print(f"Found {len(train_data.valid_shortcuts)} valid shortcuts")
    print(f"Collected {len(train_data.node_preimages)} node preimages")

    save_dir = Path(config["training_data_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    train_data.save(save_dir)

    loaded_data = GoalConditionedTrainingData.load(save_dir)
    assert len(loaded_data.node_states) == len(
        train_data.node_states
    ), "Node states not preserved"
    assert len(loaded_data.valid_shortcuts) == len(
        train_data.valid_shortcuts
    ), "Valid shortcuts not preserved"
    assert len(loaded_data.node_preimages) == len(
        train_data.node_preimages
    ), "Node preimages not preserved"

    return train_data


def test_goal_wrapper():
    """Test the goal-conditioned wrapper."""
    print("\n=== Testing Goal-Conditioned Wrapper ===")
    train_data = test_goal_conditioned_data_collection()
    system = Blocks2DTAMPSystem.create_default(seed=42, render_mode=None)
    goal_env = GoalConditionedWrapper(
        env=system.env,
        node_states=train_data.node_states,
        valid_shortcuts=train_data.valid_shortcuts,
        perceiver=system.perceiver,
        node_preimages=train_data.node_preimages,
        max_preimage_size=14,
        use_preimages=True,
        success_threshold=0.01,
        success_reward=10.0,
        step_penalty=-0.5,
        max_episode_steps=50,
    )

    obs, _ = goal_env.reset()

    assert isinstance(obs, dict), "Observation should be a Dict"
    assert "observation" in obs, "Observation should have 'observation' key"
    assert "achieved_goal" in obs, "Observation should have 'achieved_goal' key"
    assert "desired_goal" in obs, "Observation should have 'desired_goal' key"
    assert obs["observation"].shape == goal_env.observation_space["observation"].shape

    action = goal_env.action_space.sample()
    next_obs, reward, _, _, step_info = goal_env.step(action)
    assert isinstance(next_obs, dict), "Step should return a Dict observation"
    print(
        f"Step reward: {reward}, preimage distance: {step_info['preimage_distance']:.3f}"
    )

    return goal_env


@pytest.mark.parametrize("algorithm", ["SAC"])
def test_goal_conditioned_rl(algorithm):
    """Test goal-conditioned RL policy training."""
    print(f"\n=== Testing Goal-Conditioned RL Policy ({algorithm}) ===")
    train_data = test_goal_conditioned_data_collection()
    system = Blocks2DTAMPSystem.create_default(seed=42, render_mode=None)
    goal_env = GoalConditionedWrapper(
        env=system.env,
        node_states=train_data.node_states,
        valid_shortcuts=train_data.valid_shortcuts,
        perceiver=system.perceiver,
        node_preimages=train_data.node_preimages,
        max_preimage_size=14,
        use_preimages=True,
        success_threshold=0.01,
        success_reward=10.0,
        step_penalty=-0.5,
        max_episode_steps=50,
    )
    config = GoalConditionedRLConfig(
        algorithm=algorithm,
        batch_size=64,  # Small batch for testing
        buffer_size=1000,  # Small buffer for testing
    )
    policy = GoalConditionedRLPolicy(seed=42, config=config)

    # Set node states and valid shortcuts
    policy.node_states = train_data.node_states
    policy.valid_shortcuts = train_data.valid_shortcuts
    policy.node_preimages = train_data.node_preimages

    # Train the policy
    train_data.config["max_steps"] = 50
    policy.train(goal_env, train_data)

    # Test action prediction
    if train_data.valid_shortcuts:
        source_id, target_id = train_data.valid_shortcuts[0]
        context = PolicyContext(
            preimage=set(),
            current_atoms=set(),
            info={"source_node_id": source_id, "target_node_id": target_id},
        )
        policy.configure_context(context)

        obs, _ = goal_env.reset(
            options={"source_node_id": source_id, "goal_node_id": target_id}
        )
        print(f"Initial observation: {obs}")
        action = policy.get_action(obs)
        assert (
            action.shape == goal_env.action_space.shape
        ), f"Wrong action shape: {action.shape}"

    # Test save/load
    save_dir = Path("test_data")
    save_dir.mkdir(exist_ok=True)
    save_path = str(save_dir / "test_goal_policy")

    policy.save(save_path)

    # Load policy
    new_policy = GoalConditionedRLPolicy(seed=42, config=config)
    new_policy.load(save_path)

    assert len(new_policy.node_states) == len(
        policy.node_states
    ), "Node states not preserved"

    return policy


@pytest.mark.skip("TODO: Enable using different graphs and node-states for HER")
@pytest.mark.parametrize("algorithm", ["SAC"])
def test_goal_conditioned_training_pipeline(algorithm):
    """Test the full goal-conditioned training and evaluation pipeline."""
    print(f"\n=== Testing Goal-Conditioned Training Pipeline ({algorithm}) ===")
    config = TrainingConfig(
        seed=42,
        num_episodes=1,
        max_steps=50,
        collect_episodes=1,
        episodes_per_scenario=250,
        force_collect=True,
        render=True,
        record_training=False,
        training_record_interval=100,
        training_data_dir="training_data/test_goal_pipeline",
        save_dir="trained_policies/test_goal_pipeline",
        max_preimage_size=14,
        success_threshold=0.01,
        success_reward=10.0,
        step_penalty=-0.5,
    )

    # Create system
    system = Blocks2DTAMPSystem.create_default(
        seed=config.seed, render_mode="rgb_array" if config.render else None
    )

    # Create policy factory
    def policy_factory(seed):
        return GoalConditionedRLPolicy(
            seed=seed,
            config=GoalConditionedRLConfig(
                algorithm=algorithm,
                batch_size=64,
                buffer_size=1000,
                n_sampled_goal=4,
            ),
        )

    metrics = train_and_evaluate_goal_conditioned(
        system,
        policy_factory,
        config,
        policy_name=f"Test_Goal_{algorithm}",
        use_preimages=True,
    )
    print(f"Success rate: {metrics.success_rate:.2%}")
    print(f"Average episode length: {metrics.avg_episode_length:.2f}")

    return metrics


@pytest.mark.skip("TODO: Enable using different graphs and node-states for HER")
@pytest.mark.parametrize("algorithm", ["SAC"])
def test_goal_conditioned_rl_rollouts(
    algorithm,
    use_random_rollouts=True,
    num_rollouts_per_node=1000,
    max_steps_per_rollout=100,
    shortcut_success_threshold=1,
):
    """Test the full goal-conditioned RL with random rollout shortcut
    selection."""
    print("\n=== Testing Goal-Conditioned RL with Random Rollout Selection ===")
    config = TrainingConfig(
        seed=42,
        num_episodes=1,
        max_steps=50,
        collect_episodes=1,
        episodes_per_scenario=200,
        force_collect=False,
        render=True,
        record_training=True,
        training_record_interval=100,
        training_data_dir="training_data/test_goal_rollouts",
        save_dir="trained_policies/test_goal_rollouts",
        max_preimage_size=14,
        success_threshold=0.01,
        success_reward=10.0,
        step_penalty=-0.5,
    )

    # Create system
    system = Blocks2DTAMPSystem.create_default(
        seed=config.seed, render_mode="rgb_array" if config.render else None
    )

    # Create policy factory
    def policy_factory(seed):
        return GoalConditionedRLPolicy(
            seed=seed,
            config=GoalConditionedRLConfig(
                algorithm=algorithm,
                batch_size=64,
                buffer_size=1000,
                n_sampled_goal=4,
            ),
        )

    metrics = train_and_evaluate_goal_conditioned(
        system,
        policy_factory,
        config,
        policy_name=f"Goal_{algorithm}_rollouts",
        use_preimages=True,
        use_random_rollouts=use_random_rollouts,
        num_rollouts_per_node=num_rollouts_per_node,
        max_steps_per_rollout=max_steps_per_rollout,
        shortcut_success_threshold=shortcut_success_threshold,
    )
    print(f"Success rate: {metrics.success_rate:.2%}")
    print(f"Average episode length: {metrics.avg_episode_length:.2f}")

    return metrics
