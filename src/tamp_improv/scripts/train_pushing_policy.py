"""Script for training the pushing policy."""

from pathlib import Path

from tamp_improv.approaches.rl_improvisational_policy import RLImprovisationalPolicy
from tamp_improv.benchmarks.blocks2d_env import Blocks2DEnv
from tamp_improv.benchmarks.pushing_env import make_pushing_env


def train_pushing_policy(
    total_timesteps: int = 1_000_000,
    seed: int = 42,
    save_path: str = "trained_policies/pushing_policy",
) -> None:
    """Train and save a pushing policy.

    Args:
        total_timesteps: Number of timesteps to train for
        seed: Random seed
        save_path: Where to save the trained policy
    """
    # Create save directory if it doesn't exist
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    # Create environments
    base_env = Blocks2DEnv()
    env = make_pushing_env(base_env, seed=seed)

    # Create and train policy
    policy = RLImprovisationalPolicy(env)
    print(f"Training policy for {total_timesteps} timesteps...")
    policy.train(total_timesteps=total_timesteps, seed=seed)

    # Save trained policy
    policy.save(save_path)
    print(f"Saved trained policy to {save_path}")