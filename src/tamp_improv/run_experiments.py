"""Script to run experiments."""

from tamp_improv.approaches.base import BaseApproach
from tamp_improv.approaches.mpc import MPCApproach, MPCConfig
from tamp_improv.approaches.rl import RLApproach, RLConfig
from tamp_improv.benchmarks.base import BaseEnvironment
from tamp_improv.benchmarks.blocks2d import Blocks2DEnvironment
from tamp_improv.benchmarks.number import NumberEnvironment


def run_experiments(
    envs: list[BaseEnvironment],
    approaches: list[BaseApproach],
    num_episodes: int = 100,
) -> dict[tuple[str, str], float]:
    """Run experiments for all environment-approach combinations."""
    metrics = {}

    for env in envs:
        for approach in approaches:
            print(
                f"\nRunning {approach.__class__.__name__} on {env.__class__.__name__}"
            )

            # Run episodes
            successes = 0
            for episode in range(num_episodes):
                obs, info = env.reset()
                action = approach.reset(obs, info)
                terminated = truncated = False

                while not (terminated or truncated):
                    obs, reward, terminated, truncated, info = env.step(action)
                    action = approach.step(obs, reward, terminated, truncated, info)

                if terminated and reward > 0:
                    successes += 1

                if (episode + 1) % 10 == 0:
                    print(
                        f"Episode {episode + 1}: {successes/(episode + 1):.2%} success rate"
                    )

            metrics[(env.__class__.__name__, approach.__class__.__name__)] = (
                successes / num_episodes
            )

    return metrics


def main() -> None:
    """Run main experiments."""
    envs = [
        NumberEnvironment(seed=42),
        Blocks2DEnvironment(seed=42),
    ]

    approaches = [
        RLApproach(
            envs[0],
            seed=42,
            config=RLConfig(
                policy_path="policies/number_policy.zip",
                train_online=False,
            ),
        ),
        MPCApproach(
            envs[0],
            seed=42,
            config=MPCConfig(
                num_rollouts=100,
                horizon=35,
            ),
        ),
    ]

    metrics = run_experiments(envs, approaches)

    print("\nFinal Results:")
    print("=" * 50)
    for (env_name, approach_name), success_rate in metrics.items():
        print(f"{env_name} with {approach_name}: {success_rate:.2%} success rate")


if __name__ == "__main__":
    main()
