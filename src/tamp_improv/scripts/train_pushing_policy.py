"""Script for training the pushing policy."""

import argparse
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


def main():
    """Parse arguments and train pushing policy."""

    parser = argparse.ArgumentParser(description="Train a pushing policy.")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=1_000_000,
        help="Number of timesteps to train for",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--save-path",
        type=str,
        default="trained_policies/pushing_policy",
        help="Where to save the trained policy",
    )
    args = parser.parse_args()

    train_pushing_policy(args.timesteps, args.seed, args.save_path)
