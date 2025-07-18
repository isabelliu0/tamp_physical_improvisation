"""Test script for goal-conditioned RL."""

import pytest

from tamp_improv.approaches.improvisational.policies.goal_rl import (
    GoalConditionedRLConfig,
    GoalConditionedRLPolicy,
)
from tamp_improv.approaches.improvisational.training import (
    TrainingConfig,
    train_and_evaluate_goal_conditioned,
)
from tamp_improv.benchmarks.obstacle2d import Obstacle2DTAMPSystem
from tamp_improv.benchmarks.pybullet_cluttered_drawer import ClutteredDrawerTAMPSystem
from tamp_improv.benchmarks.pybullet_obstacle_tower_graph import (
    GraphObstacleTowerTAMPSystem,
)


@pytest.mark.parametrize("algorithm", ["SAC"])
def test_goal_conditioned_training_pipeline(algorithm):
    """Test the full goal-conditioned training and evaluation pipeline."""
    print(f"\n=== Testing Goal-Conditioned Training Pipeline ({algorithm}) ===")
    config = TrainingConfig(
        seed=42,
        num_episodes=1,
        max_steps=50,
        collect_episodes=5,
        episodes_per_scenario=250,
        force_collect=True,
        render=True,
        record_training=False,
        training_record_interval=100,
        training_data_dir="training_data/test_goal_pipeline",
        save_dir="trained_policies/test_goal_pipeline",
        max_atom_size=14,
        success_threshold=0.01,
        success_reward=10.0,
        step_penalty=-0.5,
    )

    # Create system
    system = Obstacle2DTAMPSystem.create_default(
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
        use_atom_as_obs=False,
        use_random_rollouts=True,
        num_rollouts_per_node=1000,
        max_steps_per_rollout=100,
        shortcut_success_threshold=1,
    )
    print(f"Success rate: {metrics.success_rate:.2%}")
    print(f"Average episode length: {metrics.avg_episode_length:.2f}")

    return metrics


@pytest.mark.skip(reason="Taking too long as unit tests")
@pytest.mark.parametrize(
    "system_cls,max_atom_size",
    [
        (GraphObstacleTowerTAMPSystem, 42),
        (ClutteredDrawerTAMPSystem, 72),
    ],
)
def test_goal_conditioned_pybullet(system_cls, max_atom_size):
    """Test the goal-conditioned training and evaluation on pybullet
    environments."""
    print(f"\n=== Testing Goal-Conditioned Training on {system_cls.__name__} ===")
    config = TrainingConfig(
        seed=42,
        num_episodes=1,
        max_steps=50,
        collect_episodes=1,
        episodes_per_scenario=50,
        force_collect=True,
        render=True,
        record_training=False,
        training_record_interval=100,
        training_data_dir="training_data/test_goal_pipeline",
        save_dir="trained_policies/test_goal_pipeline",
        max_atom_size=max_atom_size,
        success_threshold=0.01,
        success_reward=10.0,
        step_penalty=-0.5,
    )

    # Create system
    system = system_cls.create_default(
        seed=config.seed, render_mode="rgb_array" if config.render else None
    )

    # Create policy factory
    def policy_factory(seed):
        return GoalConditionedRLPolicy(
            seed=seed,
            config=GoalConditionedRLConfig(
                algorithm="SAC",
                batch_size=64,
                buffer_size=1000,
                n_sampled_goal=4,
            ),
        )

    metrics = train_and_evaluate_goal_conditioned(
        system,
        policy_factory,
        config,
        policy_name="Test_Goal",
        use_atom_as_obs=False,
        use_random_rollouts=True,
        num_rollouts_per_node=100,
        max_steps_per_rollout=300,
        shortcut_success_threshold=5,
    )
    print(f"Success rate: {metrics.success_rate:.2%}")
    print(f"Average episode length: {metrics.avg_episode_length:.2f}")

    return metrics
