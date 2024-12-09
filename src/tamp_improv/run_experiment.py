"""Run experiments with different TAMP systems and approaches."""

from dataclasses import dataclass
from pathlib import Path
from typing import Type

from tabulate import tabulate

from tamp_improv.approaches.improvisational.base import ImprovisationalTAMPApproach
from tamp_improv.approaches.improvisational.policies.mpc import MPCConfig, MPCPolicy
from tamp_improv.approaches.improvisational.policies.rl import RLPolicy
from tamp_improv.approaches.improvisational.training import (
    TrainingConfig,
    train_and_evaluate,
)
from tamp_improv.benchmarks.blocks2d import Blocks2DTAMPSystem
from tamp_improv.benchmarks.number import NumberTAMPSystem


@dataclass
class SystemConfig:
    """Configuration for a specific system."""

    system_cls: Type[NumberTAMPSystem] | Type[Blocks2DTAMPSystem]
    mpc_config: MPCConfig


@dataclass
class ExperimentConfig:
    """Configuration for running experiments."""

    # General settings
    seed: int = 42
    num_episodes: int = 5
    max_steps: int = 50
    render: bool = True

    # Training settings
    collect_episodes: int = 50
    episodes_per_scenario: int = 5
    force_collect: bool = False
    record_training: bool = True
    training_record_interval: int = 50
    save_dir: str = "trained_policies"
    training_data_dir: str = "training_data"


def get_system_configs() -> list[SystemConfig]:
    """Get configurations for all systems."""
    return [
        SystemConfig(
            system_cls=Blocks2DTAMPSystem,
            mpc_config=MPCConfig(
                num_rollouts=100, horizon=35, num_control_points=5, noise_scale=1.0
            ),
        ),
        SystemConfig(
            system_cls=NumberTAMPSystem,
            mpc_config=MPCConfig(
                num_rollouts=20, horizon=10, num_control_points=3, noise_scale=0.5
            ),
        ),
    ]


def evaluate_system(
    system_config: SystemConfig,
    exp_config: ExperimentConfig,
) -> list[list[str]]:
    """Evaluate all approaches on a system."""
    results = []

    # Base config for MPC
    base_config = TrainingConfig(
        seed=exp_config.seed,
        num_episodes=exp_config.num_episodes,
        max_steps=exp_config.max_steps,
        render=exp_config.render,
    )

    # Test MPC approach
    print(f"\n=== Testing MPC on {system_config.system_cls.__name__} ===")
    system = system_config.system_cls.create_default(
        seed=exp_config.seed, render_mode="rgb_array" if exp_config.render else None
    )
    policy = MPCPolicy(seed=exp_config.seed, config=system_config.mpc_config)
    _ = ImprovisationalTAMPApproach(system, policy, seed=exp_config.seed)
    metrics = train_and_evaluate(system, type(policy), base_config)
    results.append(
        [
            system.name,
            "MPC",
            "Fresh",
            f"{metrics.success_rate:.1%}",
            f"{metrics.avg_episode_length:.1f}",
            f"{metrics.avg_reward:.2f}",
        ]
    )

    # Create RL-specific config
    rl_config = TrainingConfig(
        # Basic settings from base config
        seed=exp_config.seed,
        num_episodes=exp_config.num_episodes,
        max_steps=exp_config.max_steps,
        render=exp_config.render,
        # RL-specific settings
        collect_episodes=exp_config.collect_episodes,
        episodes_per_scenario=exp_config.episodes_per_scenario,
        force_collect=exp_config.force_collect,
        record_training=exp_config.record_training,
        training_record_interval=exp_config.training_record_interval,
        training_data_dir=exp_config.training_data_dir,
        save_dir=exp_config.save_dir,
    )

    # Test RL approach
    print(f"\n=== Testing RL on {system_config.system_cls.__name__} ===")
    # Fresh training
    system = system_config.system_cls.create_default(
        seed=exp_config.seed, render_mode="rgb_array" if exp_config.render else None
    )
    policy = RLPolicy(seed=exp_config.seed)
    _ = ImprovisationalTAMPApproach(system, policy, seed=exp_config.seed)
    metrics = train_and_evaluate(
        system, type(policy), rl_config
    )  # Use RL-specific config
    results.append(
        [
            system.name,
            "RL",
            "Fresh",
            f"{metrics.success_rate:.1%}",
            f"{metrics.avg_episode_length:.1f}",
            f"{metrics.avg_reward:.2f}",
        ]
    )

    # Test loaded RL policy
    policy_file = Path(exp_config.save_dir) / f"{system.name}_RLPolicy"
    if policy_file.exists():
        print(f"\n=== Testing Loaded RL on {system_config.system_cls.__name__} ===")
        system = system_config.system_cls.create_default(
            seed=exp_config.seed, render_mode="rgb_array" if exp_config.render else None
        )
        loaded_policy = RLPolicy(seed=exp_config.seed)
        loaded_policy.initialize(system.wrapped_env)
        loaded_policy.load(str(policy_file))
        _ = ImprovisationalTAMPApproach(system, loaded_policy, seed=exp_config.seed)

        metrics = train_and_evaluate(
            system,
            type(loaded_policy),
            rl_config,  # Use RL-specific config
            is_loaded_policy=True,
            loaded_policy=loaded_policy,
        )
        results.append(
            [
                system.name,
                "RL",
                "Loaded",
                f"{metrics.success_rate:.1%}",
                f"{metrics.avg_episode_length:.1f}",
                f"{metrics.avg_reward:.2f}",
            ]
        )
    else:
        print(
            f"\nSkipping loaded policy evaluation - no saved policy found at {policy_file}"
        )

    return results


def run_experiments(config: ExperimentConfig) -> None:
    """Run all experiments."""
    # Create results directory
    results_dir = Path(config.save_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Get all system configurations
    system_configs = get_system_configs()

    # Store all results
    all_results = []

    # Run experiments for each system
    for system_config in system_configs:
        print(
            f"\n{'='*20} Testing system: {system_config.system_cls.__name__} {'='*20}"
        )
        results = evaluate_system(system_config, config)
        all_results.extend(results)

    # Print results table
    headers = ["System", "Approach", "Type", "Success Rate", "Avg Length", "Avg Reward"]
    print("\n\nFinal Results:")
    print(tabulate(all_results, headers=headers, tablefmt="grid"))

    # Save results
    results_file = results_dir / "results.txt"
    with open(results_file, "w") as f:
        f.write(tabulate(all_results, headers=headers, tablefmt="grid"))
    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    config = ExperimentConfig()
    run_experiments(config)
