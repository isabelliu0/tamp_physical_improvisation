"""Test script for graph-based training data collection."""

from pathlib import Path

import numpy as np
import pytest
import torch

from tamp_improv.approaches.improvisational.base import ImprovisationalTAMPApproach
from tamp_improv.approaches.improvisational.graph_training import (
    collect_graph_based_training_data,
)
from tamp_improv.approaches.improvisational.policies.mpc import MPCConfig, MPCPolicy
from tamp_improv.approaches.improvisational.policies.multi_rl import MultiRLPolicy
from tamp_improv.approaches.improvisational.policies.pushing_pybullet import (
    PybulletPushingPolicy,
)
from tamp_improv.approaches.improvisational.policies.rl import RLConfig
from tamp_improv.approaches.improvisational.policies.rl2mpc import (
    RL2MPCConfig,
    RL2MPCPolicy,
)
from tamp_improv.approaches.improvisational.training import (
    TrainingConfig,
    train_and_evaluate,
)
from tamp_improv.benchmarks.blocks2d import Blocks2DTAMPSystem
from tamp_improv.benchmarks.blocks2d_graph import GraphBlocks2DTAMPSystem
from tamp_improv.benchmarks.pybullet_clear_and_place import ClearAndPlaceTAMPSystem
from tamp_improv.benchmarks.pybullet_clear_and_place_graph import (
    GraphClearAndPlaceTAMPSystem,
)
from tamp_improv.benchmarks.pybullet_cluttered_drawer import ClutteredDrawerTAMPSystem


def run_multi_seed_experiment(system_cls, use_context_wrapper, seeds):
    """Run the experiment with multiple seeds and return aggregated results."""
    all_metrics = []

    # RL configuration
    rl_config = RLConfig(
        learning_rate=3e-4,
        batch_size=32,
        n_epochs=10,
        gamma=0.99,
        ent_coef=0.01,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    for seed in seeds:
        print(f"\n\n=== Running experiment with seed {seed} ===")

        # Configuration with current seed
        config = TrainingConfig(
            seed=seed,
            num_episodes=1,
            max_steps=300,
            max_training_steps_per_shortcut=100,
            collect_episodes=1,
            episodes_per_scenario=1500,
            force_collect=False,
            render=False,
            record_training=False,
            training_record_interval=100,
            # training_data_dir=f"training_data/multi_rl{'_context' if use_context_wrapper else ''}/seed_{seed}",  # pylint: disable=line-too-long
            training_data_dir="training_data/graph_training_data",
            save_dir=f"trained_policies/multi_rl{'_context' if use_context_wrapper else ''}/seed_{seed}",  # pylint: disable=line-too-long
            batch_size=32,
            max_atom_size=14,
            action_scale=0.015,
        )

        print(f"\n1. Creating system for seed {seed}...")
        system = system_cls.create_default(seed=config.seed, render_mode=None)

        print(f"\n2. Training and evaluating policy for seed {seed}...")

        # Define policy factory
        def policy_factory(seed_val: int) -> MultiRLPolicy:
            return MultiRLPolicy(seed=seed_val, config=rl_config)

        # Train and evaluate with graph-based collection
        metrics = train_and_evaluate(
            system,
            policy_factory,
            config,
            policy_name=f"MultiRL{'_Context' if use_context_wrapper else ''}_Seed{seed}",
            use_context_wrapper=use_context_wrapper,
            use_random_rollouts=True,
            num_rollouts_per_node=100,
            max_steps_per_rollout=300,
            shortcut_success_threshold=5,
        )

        all_metrics.append(metrics)

    # Calculate aggregate statistics
    success_rates = [m.success_rate for m in all_metrics]
    episode_lengths = [m.avg_episode_length for m in all_metrics]
    rewards = [m.avg_reward for m in all_metrics]
    training_times = [m.training_time for m in all_metrics]

    print("\n\n=== Aggregated Results Across All Seeds ===")
    print(
        f"Success Rate: Mean = {np.mean(success_rates):.2%}, Std = {np.std(success_rates):.2%}"  # pylint: disable=line-too-long
    )
    print(
        f"Average Episode Length: Mean = {np.mean(episode_lengths):.2f}, Std = {np.std(episode_lengths):.2f}"  # pylint: disable=line-too-long
    )
    print(f"Average Reward: Mean = {np.mean(rewards):.2f}, Std = {np.std(rewards):.2f}")
    print(
        f"Training Time: Mean = {np.mean(training_times):.2f}s, Std = {np.std(training_times):.2f}s"  # pylint: disable=line-too-long
    )

    # Per-seed details
    print("\n=== Per-Seed Results ===")
    for seed, metrics in zip(seeds, all_metrics):
        print(f"Seed {seed}:")
        print(f"  Success Rate: {metrics.success_rate:.2%}")
        print(f"  Average Episode Length: {metrics.avg_episode_length:.2f}")
        print(f"  Average Reward: {metrics.avg_reward:.2f}")
        print(f"  Training Time: {metrics.training_time:.2f} seconds")

    return {
        "all_metrics": all_metrics,
        "summary": {
            "success_rate": {
                "mean": np.mean(success_rates),
                "std": np.std(success_rates),
            },
            "episode_length": {
                "mean": np.mean(episode_lengths),
                "std": np.std(episode_lengths),
            },
            "reward": {"mean": np.mean(rewards), "std": np.std(rewards)},
            "training_time": {
                "mean": np.mean(training_times),
                "std": np.std(training_times),
            },
        },
    }


@pytest.mark.skip("Takes too long to run.")
def test_pybullet_graph_training_collection():
    """Test collecting graph-based training data."""
    print("\n=== Testing Graph-Based Training Data Collection ===")

    config = {
        "seed": 42,
        "num_episodes": 1,
        "max_steps": 300,
        "max_training_steps_per_shortcut": 100,
        "render": True,
        "collect_episodes": 1,
        "force_collect": True,
        "training_data_dir": "training_data/graph_training_data",
    }

    print("\n1. Creating system...")
    system = ClearAndPlaceTAMPSystem.create_default(
        seed=config["seed"], render_mode="rgb_array" if config["render"] else None
    )

    print("\n2. Creating approach...")
    policy = PybulletPushingPolicy(seed=config["seed"])
    approach = ImprovisationalTAMPApproach(system, policy, seed=config["seed"])

    print("\n3. Collecting training data...")
    train_data, _ = collect_graph_based_training_data(
        system,
        approach,
        config,
        use_random_rollouts=True,
        num_rollouts_per_node=100,
        max_steps_per_rollout=300,
        shortcut_success_threshold=5,
        action_scale=0.015,
    )

    print("\n=== Training Data Statistics ===")
    print(f"Collected {len(train_data)} training examples")

    # Print details of each collected shortcut
    for i in range(len(train_data)):
        print(f"\nShortcut Example {i+1}:")
        print("Source atoms:")
        for atom in sorted(train_data.current_atoms[i], key=str):
            print(f"  - {atom}")

        print("Target atoms:")
        for atom in sorted(train_data.goal_atoms[i], key=str):
            print(f"  - {atom}")

    save_dir = Path(config["training_data_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "pybullet_all_shortcuts_2"
    train_data.save(save_path)
    print(f"\nSaved training data to {save_path}")

    return train_data


@pytest.mark.skip("Takes too long to run.")
@pytest.mark.parametrize(
    "system_cls,use_context_wrapper",
    [(GraphBlocks2DTAMPSystem, False), (Blocks2DTAMPSystem, False)],
)
def test_multi_rl_blocks2d_pipeline(system_cls, use_context_wrapper):
    """Test the multi-policy RL training and evaluation pipeline."""
    print("\n=== Testing Multi-Policy RL Pipeline ===")

    # Configuration
    config = TrainingConfig(
        seed=42,
        num_episodes=5,
        max_steps=50,
        max_training_steps_per_shortcut=50,
        collect_episodes=3,
        episodes_per_scenario=1000,
        force_collect=False,
        render=True,
        record_training=False,
        training_record_interval=125,
        training_data_dir=f"training_data/multi_rl{'_context' if use_context_wrapper else ''}",  # pylint: disable=line-too-long
        save_dir=f"trained_policies/multi_rl{'_context' if use_context_wrapper else ''}",  # pylint: disable=line-too-long
        batch_size=32,
        max_atom_size=14,
    )

    # RL configuration
    rl_config = RLConfig(
        learning_rate=3e-4,
        batch_size=32,
        n_epochs=10,
        gamma=0.99,
        ent_coef=0.05,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    print("\n1. Creating system...")
    system = system_cls.create_default(
        n_blocks=2, seed=config.seed, render_mode="rgb_array" if config.render else None
    )

    print("\n2. Training and evaluating policy...")

    # Define policy factory
    def policy_factory(seed: int) -> MultiRLPolicy:
        return MultiRLPolicy(seed=seed, config=rl_config)

    # Train and evaluate with graph-based collection
    metrics = train_and_evaluate(
        system,
        policy_factory,
        config,
        policy_name=f"MultiRL{'_Context' if use_context_wrapper else ''}",
        use_context_wrapper=use_context_wrapper,
        use_random_rollouts=True,
        num_rollouts_per_node=1000,
        max_steps_per_rollout=100,
        shortcut_success_threshold=1,
    )

    print("\n=== Results ===")
    print(f"Success Rate: {metrics.success_rate:.2%}")
    print(f"Average Episode Length: {metrics.avg_episode_length:.2f}")
    print(f"Average Reward: {metrics.avg_reward:.2f}")
    print(f"Training Time: {metrics.training_time:.2f} seconds")
    print(f"Total Time: {metrics.total_time:.2f} seconds")

    return metrics


@pytest.mark.skip("Takes too long to run.")
def test_multi_rl_blocks2d_multi_seed():
    """Test the multi-policy RL training and evaluation pipeline with multiple
    seeds."""
    print("\n=== Testing Multi-Policy RL Pipeline with Multiple Seeds ===")
    results = run_multi_seed_experiment(
        system_cls=GraphBlocks2DTAMPSystem,
        use_context_wrapper=False,
        seeds=[42, 43, 44, 45, 46],
    )
    return results


@pytest.mark.parametrize("system_cls", [GraphBlocks2DTAMPSystem, Blocks2DTAMPSystem])
def test_multi_rl_blocks2d_loaded(system_cls):
    """Test MultiRL on Blocks2D with loaded policies."""
    policy_dir = Path("trained_policies/multi_rl")
    policy_dir.mkdir(parents=True, exist_ok=True)

    # Configuration
    config = TrainingConfig(
        seed=42,
        num_episodes=5,
        max_steps=50,
        render=True,
        collect_episodes=0,
        episodes_per_scenario=0,
        force_collect=False,
        record_training=False,
        training_data_dir="training_data",
        save_dir="trained_policies",
    )

    # RL configuration
    rl_config = RLConfig(
        learning_rate=3e-4,
        batch_size=32,
        n_epochs=10,
        gamma=0.99,
        ent_coef=0.05,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    print(f"\n=== Testing MultiRL Loaded Policies on {system_cls.__name__} ===")
    system = system_cls.create_default(
        n_blocks=3, seed=42, render_mode="rgb_array" if config.render else None
    )

    policy_name = "MultiRL"
    policy_path = policy_dir / f"{system_cls.__name__}_{policy_name}"

    if not policy_path.exists():
        pytest.skip(f"Policy directory not found at {policy_path}")

    def multi_policy_factory(seed: int) -> MultiRLPolicy:
        policy: MultiRLPolicy = MultiRLPolicy(seed=seed, config=rl_config)
        policy.load(str(policy_path))
        return policy

    metrics = train_and_evaluate(
        system,
        multi_policy_factory,
        config,
        policy_name=f"{policy_name}_Loaded",
        use_context_wrapper=False,
        select_random_goal=False,
    )

    print("\nMultiRL Loaded Policies Results:")
    print(f"Success Rate: {metrics.success_rate:.2%}")
    print(f"Average Episode Length: {metrics.avg_episode_length:.2f}")
    print(f"Average Reward: {metrics.avg_reward:.2f}")


@pytest.mark.skip("Takes too long to run")
@pytest.mark.parametrize(
    "system_cls,use_context_wrapper",
    [(GraphClearAndPlaceTAMPSystem, False), (ClearAndPlaceTAMPSystem, False)],
)
def test_multi_rl_pybullet_pipeline(system_cls, use_context_wrapper):
    """Test the multi-policy RL training and evaluation pipeline."""
    print("\n=== Testing Multi-Policy RL Pipeline ===")

    # Configuration
    config = TrainingConfig(
        seed=42,
        num_episodes=1,
        max_steps=300,
        max_training_steps_per_shortcut=100,
        collect_episodes=1,
        episodes_per_scenario=3000,
        force_collect=False,
        render=True,
        record_training=False,
        training_record_interval=100,
        training_data_dir="training_data/graph_training_data",
        save_dir=f"trained_policies/multi_rl{'_context' if use_context_wrapper else ''}",  # pylint: disable=line-too-long
        batch_size=16,
        max_atom_size=14,
        action_scale=0.015,
    )

    # RL configuration
    rl_config = RLConfig(
        learning_rate=3e-4,
        batch_size=16,
        n_epochs=10,
        gamma=0.99,
        ent_coef=0.01,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    print("\n1. Creating system...")
    system = system_cls.create_default(
        seed=config.seed, render_mode="rgb_array" if config.render else None
    )

    print("\n2. Training and evaluating policy...")

    # Define policy factory
    def policy_factory(seed: int) -> MultiRLPolicy:
        return MultiRLPolicy(seed=seed, config=rl_config)

    # Train and evaluate with graph-based collection
    metrics = train_and_evaluate(
        system,
        policy_factory,
        config,
        policy_name=f"MultiRL{'_Context' if use_context_wrapper else ''}",
        use_context_wrapper=use_context_wrapper,
        use_random_rollouts=True,
        num_rollouts_per_node=100,
        max_steps_per_rollout=300,
        shortcut_success_threshold=5,
    )

    print("\n=== Results ===")
    print(f"Success Rate: {metrics.success_rate:.2%}")
    print(f"Average Episode Length: {metrics.avg_episode_length:.2f}")
    print(f"Average Reward: {metrics.avg_reward:.2f}")
    print(f"Training Time: {metrics.training_time:.2f} seconds")
    print(f"Total Time: {metrics.total_time:.2f} seconds")

    return metrics


@pytest.mark.skip("Takes too long to run.")
def test_multi_rl_pybullet_multi_seed():
    """Test the multi-policy RL training and evaluation pipeline with multiple
    seeds."""
    print("\n=== Testing Multi-Policy RL Pipeline with Multiple Seeds ===")
    results = run_multi_seed_experiment(
        system_cls=GraphClearAndPlaceTAMPSystem,
        use_context_wrapper=False,
        seeds=[44, 45, 46],
    )
    return results


def test_multi_rl_pybullet_loaded(system_cls=GraphClearAndPlaceTAMPSystem):
    """Test MultiRL on Pybullet ClearAndPlace with loaded policies."""
    policy_dir = Path("trained_policies/multi_rl")
    policy_dir.mkdir(parents=True, exist_ok=True)

    # Configuration
    config = TrainingConfig(
        seed=42,
        num_episodes=1,
        max_steps=300,
        render=True,
        collect_episodes=0,
        episodes_per_scenario=0,
        force_collect=False,
        record_training=False,
        training_data_dir="training_data/graph_training_data",
        save_dir="trained_policies/multi_rl",
    )

    # RL configuration
    rl_config = RLConfig(
        learning_rate=3e-4,
        batch_size=16,
        n_epochs=10,
        gamma=0.99,
        ent_coef=0.01,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    print(f"\n=== Testing MultiRL Loaded Policies on {system_cls.__name__} ===")
    system = system_cls.create_default(
        seed=42,
        render_mode="rgb_array" if config.render else None,
        num_obstacle_blocks=3,
    )

    policy_name = "MultiRL"
    policy_path = policy_dir / f"{system_cls.__name__}_{policy_name}"

    if not policy_path.exists():
        pytest.skip(f"Policy directory not found at {policy_path}")

    def multi_policy_factory(seed: int) -> MultiRLPolicy:
        policy: MultiRLPolicy = MultiRLPolicy(seed=seed, config=rl_config)
        policy.load(str(policy_path))
        return policy

    metrics = train_and_evaluate(
        system,
        multi_policy_factory,
        config,
        policy_name=f"{policy_name}_Loaded",
        use_context_wrapper=False,
        select_random_goal=False,
    )

    print("\nMultiRL Loaded Policies Results:")
    print(f"Success Rate: {metrics.success_rate:.2%}")
    print(f"Average Episode Length: {metrics.avg_episode_length:.2f}")
    print(f"Average Reward: {metrics.avg_reward:.2f}")


# @pytest.mark.skip("Takes too long to run.")
@pytest.mark.parametrize(
    "system_cls,use_context_wrapper",
    [(ClutteredDrawerTAMPSystem, False)],
)
def test_multi_rl_cluttered_drawer_pipeline(system_cls, use_context_wrapper):
    """Test the multi-policy RL training and evaluation pipeline."""
    print("\n=== Testing Multi-Policy RL Pipeline ===")

    # Configuration
    config = TrainingConfig(
        seed=42,
        num_episodes=1,
        max_steps=500,
        max_training_steps_per_shortcut=100,
        collect_episodes=1,
        episodes_per_scenario=1300,
        force_collect=False,
        render=True,
        record_training=True,
        training_record_interval=100,
        training_data_dir="training_data/graph_training_data",
        save_dir=f"trained_policies/multi_rl{'_context' if use_context_wrapper else ''}",  # pylint: disable=line-too-long
        batch_size=16,
        max_atom_size=14,
        action_scale=0.005,
    )

    # RL configuration
    rl_config = RLConfig(
        learning_rate=3e-4,
        batch_size=16,
        n_epochs=10,
        gamma=0.99,
        ent_coef=0.01,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    print("\n1. Creating system...")
    system = system_cls.create_default(
        seed=config.seed, render_mode="rgb_array" if config.render else None
    )

    print("\n2. Training and evaluating policy...")

    # Define policy factory
    def policy_factory(seed: int) -> MultiRLPolicy:
        return MultiRLPolicy(seed=seed, config=rl_config)

    # Train and evaluate with graph-based collection
    metrics = train_and_evaluate(
        system,
        policy_factory,
        config,
        policy_name=f"MultiRL{'_Context' if use_context_wrapper else ''}",
        use_context_wrapper=use_context_wrapper,
        use_random_rollouts=True,
        num_rollouts_per_node=100,
        max_steps_per_rollout=300,
        shortcut_success_threshold=1,
    )

    print("\n=== Results ===")
    print(f"Success Rate: {metrics.success_rate:.2%}")
    print(f"Average Episode Length: {metrics.avg_episode_length:.2f}")
    print(f"Average Reward: {metrics.avg_reward:.2f}")
    print(f"Training Time: {metrics.training_time:.2f} seconds")
    print(f"Total Time: {metrics.total_time:.2f} seconds")

    return metrics


@pytest.mark.skip()
def test_graph_mpc_pipeline(use_context_wrapper=True):
    """Test MPC policy with specific target shortcuts."""
    print("\n=== Testing MPC Policy with Specific Target Shortcuts ===")

    # Create config
    config = TrainingConfig(
        seed=42,
        num_episodes=1,
        max_steps=50,
        render=True,
    )

    # Create system
    system = Blocks2DTAMPSystem.create_default(
        seed=config.seed, render_mode="rgb_array" if config.render else None
    )

    # Configure MPC policy with specific shortcuts
    target_shortcuts = [(0, 3), (1, 4)]

    def policy_factory(seed: int) -> MPCPolicy:
        policy: MPCPolicy = MPCPolicy(
            seed=seed,
            config=MPCConfig(
                num_rollouts=50,
                horizon=20,
                num_control_points=10,
                noise_scale=0.25,
            ),
        )

        # Add target shortcuts
        for source_id, target_id in target_shortcuts:
            policy.add_target_shortcut(source_id, target_id)

        return policy

    # Run evaluation
    print("\nRunning evaluation with target shortcuts...")
    metrics = train_and_evaluate(
        system,
        policy_factory,
        config,
        policy_name=f"GraphMPC{'_Context' if use_context_wrapper else ''}",
        use_context_wrapper=use_context_wrapper,
    )

    # Print results
    print("\nEvaluation Results:")
    print(f"Success Rate: {metrics.success_rate:.2%}")
    print(f"Average Episode Length: {metrics.avg_episode_length:.2f}")
    print(f"Average Reward: {metrics.avg_reward:.2f}")

    return metrics


@pytest.mark.skip()
def test_graph_rl2mpc_pipeline(use_context_wrapper=True):
    """Test RL2MPC policy with the training and evaluation pipeline."""
    print("\n=== Testing RL2MPC with Training Pipeline ===")

    # Create training config
    config = TrainingConfig(
        seed=42,
        num_episodes=1,
        max_steps=50,
        render=True,
        collect_episodes=2,
        episodes_per_scenario=100,
        force_collect=False,
        record_training=False,
        training_record_interval=25,
        training_data_dir=f"training_data/graph_rl{'_context' if use_context_wrapper else ''}",  # pylint: disable=line-too-long
        save_dir=f"trained_policies/graph_rl{'_context' if use_context_wrapper else ''}",
        batch_size=32,
    )

    # Create system
    system = Blocks2DTAMPSystem.create_default(
        seed=config.seed, render_mode="rgb_array" if config.render else None
    )

    # Create RL2MPC policy
    def policy_factory(seed: int) -> RL2MPCPolicy:
        return RL2MPCPolicy(
            seed=seed,
            config=RL2MPCConfig(
                rl_config=RLConfig(
                    learning_rate=3e-4,
                    batch_size=32,
                    n_epochs=10,
                    gamma=0.99,
                ),
                mpc_config=MPCConfig(
                    num_rollouts=50,
                    horizon=20,
                    num_control_points=10,
                    noise_scale=0.05,
                ),
                reward_threshold=-15.0,
                window_size=10,
            ),
        )

    # Run train_and_evaluate
    metrics = train_and_evaluate(
        system,
        policy_factory,
        config,
        policy_name=f"GraphRL2MPC{'_Context' if use_context_wrapper else ''}",
        use_context_wrapper=use_context_wrapper,
    )

    # Print results
    print("\nEvaluation Results:")
    print(f"Success Rate: {metrics.success_rate:.2%}")
    print(f"Average Episode Length: {metrics.avg_episode_length:.2f}")
    print(f"Average Reward: {metrics.avg_reward:.2f}")

    return metrics
