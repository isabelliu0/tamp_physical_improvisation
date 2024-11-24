"""Script for training the number transition policy."""

from pathlib import Path

from tamp_improv.approaches.rl_improvisational_policy import RLImprovisationalPolicy
from tamp_improv.benchmarks.number_env_old import NumberEnv
from tamp_improv.benchmarks.number_env_wrapper_old import make_number_env_wrapper


def train_number_policy(
    total_timesteps: int = 10_000,
    seed: int = 42,
    save_path: str = "trained_policies/number_policy",
) -> None:
    """Train and save a policy for the 1->2 transition.

    Args:
        total_timesteps: Number of timesteps to train for
        seed: Random seed
        save_path: Where to save the trained policy
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    # Create environments
    base_env = NumberEnv()
    env = make_number_env_wrapper(base_env, seed=seed)

    # Create and train policy
    policy: RLImprovisationalPolicy[int, int] = RLImprovisationalPolicy(env)
    print(f"Training policy for {total_timesteps} timesteps...")
    policy.train(total_timesteps=total_timesteps, seed=seed)

    # Save trained policy
    policy.save(save_path)
    print(f"Saved trained policy to {save_path}")


if __name__ == "__main__":
    train_number_policy(
        total_timesteps=10_000,
        seed=42,
        save_path="trained_policies/number_policy",
    )
