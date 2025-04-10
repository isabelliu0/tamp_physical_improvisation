"""Script for running systematic experiments across environments using random
rollouts and multi-policy RL."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from tabulate import tabulate

from tamp_improv.approaches.improvisational.policies.base import Policy
from tamp_improv.approaches.improvisational.policies.multi_rl import MultiRLPolicy
from tamp_improv.approaches.improvisational.policies.rl import RLConfig
from tamp_improv.approaches.improvisational.training import (
    Metrics,
    TrainingConfig,
    train_and_evaluate,
)
from tamp_improv.benchmarks.base import ImprovisationalTAMPSystem
from tamp_improv.benchmarks.blocks2d import Blocks2DTAMPSystem
from tamp_improv.benchmarks.pybullet_clear_and_place import ClearAndPlaceTAMPSystem


@dataclass
class EnvironmentParams:
    """Parameters specific to each environment."""

    max_steps: int
    max_training_steps_per_shortcut: int
    episodes_per_scenario: int
    training_record_interval: int
    num_rollouts_per_node: int
    max_steps_per_rollout: int
    shortcut_success_threshold: int
    action_scale: float


@dataclass
class ExperimentConfig:
    """Configuration for running experiments."""

    # General settings
    seed: int = 42
    render: bool = True

    # Collection and training settings
    eval_episodes: int = 1
    collect_episodes: int = 1
    max_steps: int = 50
    max_training_steps_per_shortcut: int = 50
    episodes_per_scenario: int = 5
    force_collect: bool = False
    record_training: bool = False
    training_record_interval: int = 50
    batch_size: int = 32
    max_preimage_size: int = 12
    action_scale: float = 1.0

    # Output settings
    results_dir: Path = Path("results")
    training_data_dir: Path = Path("training_data")
    save_dir: Path = Path("trained_policies")

    # Policy-specific configs
    rl_config: RLConfig = field(
        default_factory=lambda: RLConfig(
            learning_rate=3e-4,
            batch_size=32,
            n_epochs=10,
            gamma=0.99,
            ent_coef=0.01,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
    )

    # Environment-specific parameters
    env_params: dict[str, EnvironmentParams] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize environment-specific parameters."""
        self.env_params["Blocks2DTAMPSystem"] = EnvironmentParams(
            max_steps=50,
            max_training_steps_per_shortcut=50,
            episodes_per_scenario=200,
            training_record_interval=25,
            num_rollouts_per_node=100,
            max_steps_per_rollout=100,
            shortcut_success_threshold=1,
            action_scale=1.0,
        )
        self.env_params["ClearAndPlaceTAMPSystem"] = EnvironmentParams(
            max_steps=500,
            max_training_steps_per_shortcut=100,
            episodes_per_scenario=1500,
            training_record_interval=100,
            num_rollouts_per_node=100,
            max_steps_per_rollout=500,
            shortcut_success_threshold=5,
            action_scale=0.015,
        )

    def get_training_config(self, env_name: str) -> TrainingConfig:
        """Get the training configuration."""
        env_params = self.env_params.get(env_name)
        assert env_params is not None
        return TrainingConfig(
            seed=self.seed,
            num_episodes=self.eval_episodes,
            max_steps=env_params.max_steps,
            max_training_steps_per_shortcut=env_params.max_training_steps_per_shortcut,
            render=self.render,
            collect_episodes=self.collect_episodes,
            episodes_per_scenario=env_params.episodes_per_scenario,
            force_collect=self.force_collect,
            record_training=self.record_training,
            training_record_interval=env_params.training_record_interval,
            training_data_dir=str(self.training_data_dir),
            save_dir=str(self.save_dir),
            batch_size=self.batch_size,
            max_preimage_size=self.max_preimage_size,
            action_scale=env_params.action_scale,
        )


class PolicyFactory:
    """Factory for creating policies with system-specific configurations."""

    def __init__(self, config: ExperimentConfig):
        """Initialize factory with experiment configuration."""
        self.config = config
        self._current_system: str = ""

    @property
    def current_system(self) -> str:
        """Get current system name."""
        return self._current_system

    @current_system.setter
    def current_system(self, system_name: str) -> None:
        """Set current system name."""
        self._current_system = system_name

    def create_multi_rl_policy(self, seed: int) -> Policy:
        """Create multi-policy RL."""
        return MultiRLPolicy(seed=seed, config=self.config.rl_config)


def get_available_systems() -> list[type[ImprovisationalTAMPSystem[Any, Any]]]:
    """Get list of available TAMP systems."""
    return [
        Blocks2DTAMPSystem,
        ClearAndPlaceTAMPSystem,
    ]


def run_experiments(config: ExperimentConfig) -> dict[tuple[str, str], Metrics]:
    """Run experiments for all systems and approaches."""
    config.results_dir.mkdir(parents=True, exist_ok=True)
    config.training_data_dir.mkdir(parents=True, exist_ok=True)
    config.save_dir.mkdir(parents=True, exist_ok=True)

    systems = get_available_systems()
    factory = PolicyFactory(config)
    approaches = {"MultiRL": factory.create_multi_rl_policy}
    results: dict[tuple[str, str], Metrics] = {}

    for system_cls in systems:
        system_name = system_cls.__name__
        print(f"\n{'='*20} Testing {system_name} {'='*20}")

        factory.current_system = system_name
        env_params = config.env_params.get(system_name)
        assert env_params is not None

        for approach_name, policy_creator in approaches.items():
            print(f"\n{'-'*10} Testing {approach_name} {'-'*10}")

            try:
                system = system_cls.create_default(  # type: ignore
                    seed=config.seed,
                    render_mode="rgb_array" if config.render else None,
                )
                train_config = config.get_training_config(system_name)

                metrics = train_and_evaluate(
                    system=system,
                    policy_factory=policy_creator,
                    config=train_config,
                    policy_name=approach_name,
                    use_context_wrapper=False,
                    use_random_rollouts=True,
                    num_rollouts_per_node=env_params.num_rollouts_per_node,
                    max_steps_per_rollout=env_params.max_steps_per_rollout,
                    shortcut_success_threshold=env_params.shortcut_success_threshold,
                )

                results[(system_cls.__name__, approach_name)] = metrics
                print(f"\nResults for {system_cls.__name__} with {approach_name}:")
                print(f"Success Rate: {metrics.success_rate:.2%}")
                print(f"Average Episode Length: {metrics.avg_episode_length:.2f}")
                print(f"Average Reward: {metrics.avg_reward:.2f}")

            except FileNotFoundError as e:
                print(f"Error running {approach_name} on {system_cls.__name__}: {e}")
                continue
            except Exception as e:
                print(f"Error running {approach_name} on {system_cls.__name__}: {e}")
                print(f"Full error: {str(e)}")
                continue

    return results


def save_results(
    results: dict[tuple[str, str], Metrics], config: ExperimentConfig
) -> None:
    """Save experiment results."""
    table_data = []
    headers = [
        "System",
        "Approach",
        "Success Rate",
        "Avg Length",
        "Avg Reward",
        "Train Time (s)",
        "Total Time (s)",
    ]

    for (system_name, approach_name), metrics in sorted(results.items()):
        table_data.append(
            [
                system_name,
                approach_name,
                f"{metrics.success_rate:.2%}",
                f"{metrics.avg_episode_length:.2f}",
                f"{metrics.avg_reward:.2f}",
                f"{metrics.training_time:.2f}",
                f"{metrics.total_time:.2f}",
            ]
        )

    table = tabulate(table_data, headers=headers, tablefmt="grid")
    results_file = config.results_dir / "experiment_results.txt"
    with open(results_file, "w", encoding="utf-8") as f:
        f.write(table)

    print(f"\nResults saved to {results_file}")


def main() -> None:
    """Run experiments with default configuration."""
    config = ExperimentConfig()
    results = run_experiments(config)
    save_results(results, config)


if __name__ == "__main__":
    main()
