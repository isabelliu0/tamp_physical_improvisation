"""Script for running systematic experiments across environments and
approaches."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tabulate import tabulate

from tamp_improv.approaches.improvisational.policies.base import Policy
from tamp_improv.approaches.improvisational.policies.mpc import MPCConfig, MPCPolicy
from tamp_improv.approaches.improvisational.policies.rl import RLConfig, RLPolicy
from tamp_improv.approaches.improvisational.policies.rl2mpc import (
    RL2MPCConfig,
    RL2MPCPolicy,
)
from tamp_improv.approaches.improvisational.training import (
    Metrics,
    TrainingConfig,
    train_and_evaluate,
)
from tamp_improv.benchmarks.base import ImprovisationalTAMPSystem
from tamp_improv.benchmarks.blocks2d import Blocks2DTAMPSystem
from tamp_improv.benchmarks.number import NumberTAMPSystem


@dataclass
class ExperimentConfig:
    """Configuration for running experiments."""

    # General settings
    seed: int = 42
    render: bool = False

    # Collection and training settings
    collect_episodes: int = 50
    max_steps: int = 50
    episodes_per_scenario: int = 5
    force_collect: bool = False
    record_training: bool = False
    training_record_interval: int = 50

    # Evaluation settings
    eval_episodes: int = 5

    # Output settings
    results_dir: Path = Path("results")
    training_data_dir: Path = Path("training_data")
    save_dir: Path = Path("trained_policies")

    # Policy-specific configs
    mpc_configs: dict[str, MPCConfig] = field(
        default_factory=lambda: {
            "Blocks2DTAMPSystem": MPCConfig(
                num_rollouts=100,
                horizon=35,
                num_control_points=5,
                noise_scale=1.0,
            ),
            "NumberTAMPSystem": MPCConfig(
                num_rollouts=20,
                horizon=10,
                num_control_points=3,
                noise_scale=0.5,
            ),
        }
    )

    rl_config: RLConfig = field(
        default_factory=lambda: RLConfig(
            learning_rate=1e-4,
            batch_size=64,
            n_epochs=5,
            gamma=0.99,
        )
    )

    rl2mpc_configs: dict[str, RL2MPCConfig] = field(
        default_factory=lambda: {
            "Blocks2DTAMPSystem": RL2MPCConfig(
                rl_config=RLConfig(
                    learning_rate=1e-4,
                    batch_size=64,
                    n_epochs=5,
                    gamma=0.99,
                ),
                mpc_config=MPCConfig(
                    num_rollouts=100,
                    horizon=35,
                    num_control_points=5,
                    noise_scale=1.0,
                ),
                reward_threshold=-5.0,
                window_size=50,
            ),
            "NumberTAMPSystem": RL2MPCConfig(
                rl_config=RLConfig(
                    learning_rate=1e-4,
                    batch_size=64,
                    n_epochs=5,
                    gamma=0.99,
                ),
                mpc_config=MPCConfig(
                    num_rollouts=20,
                    horizon=10,
                    num_control_points=3,
                    noise_scale=0.5,
                ),
                reward_threshold=-0.5,
                window_size=5,
            ),
        }
    )

    def get_training_config(self) -> TrainingConfig:
        """Get training configuration for specific phase."""
        return TrainingConfig(
            seed=self.seed,
            num_episodes=self.eval_episodes,
            max_steps=self.max_steps,
            render=self.render,
            collect_episodes=self.collect_episodes,
            episodes_per_scenario=self.episodes_per_scenario,
            force_collect=self.force_collect,
            record_training=self.record_training,
            training_record_interval=self.training_record_interval,
            save_dir=str(self.save_dir),
            training_data_dir=str(self.training_data_dir),
        )


class PolicyFactory:
    """Factory for creating policies with system-specific configurations."""

    def __init__(self, config: ExperimentConfig):
        """Initialize factory with experiment config."""
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

    def create_mpc_policy(self, seed: int) -> Policy:
        """Create MPC policy for current system."""
        mpc_config = self.config.mpc_configs[self.current_system]
        return MPCPolicy(seed=seed, config=mpc_config)

    def create_rl_policy(self, seed: int) -> Policy:
        """Create RL policy."""
        return RLPolicy(seed=seed, config=self.config.rl_config)

    def create_loaded_rl_policy(self, seed: int) -> Policy:
        """Create pre-trained RL policy."""
        policy_path = Path(self.config.save_dir) / f"{self._current_system}_RL.zip"

        if not policy_path.exists():
            raise FileNotFoundError(
                f"No trained policy found at {policy_path}. "
                "Please run training with RL policy first."
            )

        policy: RLPolicy = RLPolicy(seed=seed, config=self.config.rl_config)
        policy.load(str(policy_path))
        return policy

    def create_rl2mpc_policy(self, seed: int) -> Policy:
        """Create RL2MPC policy for current system."""
        rl2mpc_config = self.config.rl2mpc_configs[self.current_system]
        return RL2MPCPolicy(seed=seed, config=rl2mpc_config)


def get_available_systems() -> list[type[ImprovisationalTAMPSystem[Any, Any]]]:
    """Get list of available TAMP systems."""
    return [
        Blocks2DTAMPSystem,
        NumberTAMPSystem,
    ]


def run_experiments(config: ExperimentConfig) -> dict[tuple[str, str], Metrics]:
    """Run experiments for all systems and approaches."""
    config.results_dir.mkdir(parents=True, exist_ok=True)
    config.training_data_dir.mkdir(parents=True, exist_ok=True)
    config.save_dir.mkdir(parents=True, exist_ok=True)

    systems = get_available_systems()

    factory = PolicyFactory(config)

    approaches = {
        "MPC": factory.create_mpc_policy,
        "RL": factory.create_rl_policy,
        "RL_Loaded": factory.create_loaded_rl_policy,
        "RL2MPC": factory.create_rl2mpc_policy,
    }

    results: dict[tuple[str, str], Metrics] = {}

    for system_cls in systems:
        print(f"\n{'='*20} Testing {system_cls.__name__} {'='*20}")

        factory.current_system = system_cls.__name__

        for approach_name, policy_creator in approaches.items():
            print(f"\n{'-'*10} Testing {approach_name} {'-'*10}")

            try:
                system = system_cls.create_default(  # type: ignore
                    seed=config.seed,
                    render_mode="rgb_array" if config.render else None,
                )
                train_config = config.get_training_config()

                metrics = train_and_evaluate(
                    system=system,
                    policy_factory=policy_creator,
                    config=train_config,
                    policy_name=approach_name,
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
    headers = ["System", "Approach", "Success Rate", "Avg Length", "Avg Reward"]

    for (system_name, approach_name), metrics in sorted(results.items()):
        table_data.append(
            [
                system_name,
                approach_name,
                f"{metrics.success_rate:.2%}",
                f"{metrics.avg_episode_length:.2f}",
                f"{metrics.avg_reward:.2f}",
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
