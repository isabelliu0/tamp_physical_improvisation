"""Script for running pybullet experiments with configurable parameters."""

import argparse
from pathlib import Path

import numpy as np
import torch

from tamp_improv.approaches.improvisational.policies.multi_rl import MultiRLPolicy
from tamp_improv.approaches.improvisational.policies.rl import RLConfig
from tamp_improv.approaches.improvisational.training import (
    TrainingConfig,
    train_and_evaluate,
)
from tamp_improv.benchmarks.pybullet_clear_and_place_graph import (
    GraphClearAndPlaceTAMPSystem,
)


def run_multi_seed_experiment(
    system_cls, use_context_wrapper, seeds, episodes_per_scenario
):
    """Run the experiment with multiple seeds and return aggregated results."""
    all_metrics = []

    # RL configuration
    rl_config = RLConfig(
        learning_rate=3e-4,
        batch_size=16,
        n_epochs=10,
        gamma=0.99,
        ent_coef=0.01,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    for seed in seeds:
        print(f"\n\n=== Running experiment with seed {seed} ===")

        # Configuration with current seed and specified episodes_per_scenario
        config = TrainingConfig(
            seed=seed,
            num_episodes=1,
            max_steps=300,
            max_training_steps_per_shortcut=100,
            collect_episodes=1,
            episodes_per_scenario=episodes_per_scenario,  # Use the passed parameter
            force_collect=False,
            render=False,
            record_training=False,
            training_record_interval=100,
            training_data_dir="training_data/graph_training_data",
            save_dir=f"trained_policies/multi_rl_eps_{episodes_per_scenario}/seed_{seed}",  # pylint: disable=line-too-long
            batch_size=16,
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
            policy_name=f"MultiRL_EPS{episodes_per_scenario}_Seed{seed}",
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

    # Save results summary to a file
    results_dir = Path(f"results/multi_rl_eps_{episodes_per_scenario}")
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write(f"Episodes per scenario: {episodes_per_scenario}\n")
        f.write(
            f"Success Rate: Mean = {np.mean(success_rates):.2%}, Std = {np.std(success_rates):.2%}\n"  # pylint: disable=line-too-long
        )
        f.write(
            f"Average Episode Length: Mean = {np.mean(episode_lengths):.2f}, Std = {np.std(episode_lengths):.2f}\n"  # pylint: disable=line-too-long
        )
        f.write(
            f"Average Reward: Mean = {np.mean(rewards):.2f}, Std = {np.std(rewards):.2f}\n"  # pylint: disable=line-too-long
        )
        f.write(
            f"Training Time: Mean = {np.mean(training_times):.2f}s, Std = {np.std(training_times):.2f}s\n"  # pylint: disable=line-too-long
        )

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


def test_multi_rl_pybullet_multi_seed(episodes_per_scenario):
    """Test the multi-policy RL training and evaluation pipeline with multiple
    seeds."""
    print("\n=== Testing Multi-Policy RL Pipeline with Multiple Seeds ===")
    print(f"Episodes per scenario: {episodes_per_scenario}")

    results = run_multi_seed_experiment(
        system_cls=GraphClearAndPlaceTAMPSystem,
        use_context_wrapper=False,
        seeds=[42, 43, 44, 45, 46],
        episodes_per_scenario=episodes_per_scenario,
    )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run PyBullet multi-seed experiments with configurable parameters"
    )
    parser.add_argument(
        "--episodes", type=int, required=True, help="Number of episodes per scenario"
    )
    args = parser.parse_args()
    test_multi_rl_pybullet_multi_seed(args.episodes)  # type: ignore[no-untyped-call]
