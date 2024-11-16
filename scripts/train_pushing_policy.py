"""Script for training the pushing policy."""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from tamp_improv.approaches.rl_improvisational_policy import RLImprovisationalPolicy
from tamp_improv.benchmarks.blocks2d_env import Blocks2DEnv
from tamp_improv.benchmarks.blocks2d_env_wrapper import make_pushing_env


def train_pushing_policy(
    total_timesteps: int = 1_000_000,
    seed: int = 42,
    save_path: str = "trained_policies/pushing_policy",
    render: bool = False,
) -> None:
    """Train and save a pushing policy.

    Args:
        total_timesteps: Number of timesteps to train for
        seed: Random seed
        save_path: Where to save the trained policy
        render: Whether to render the environment
    """
    # Create save directory if it doesn't exist
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    # Create environments
    base_env = Blocks2DEnv(render_mode="rgb_array" if render else None)
    env = make_pushing_env(base_env, seed=seed)

    # # Uncomment to watch a video.
    # from gymnasium.wrappers import RecordVideo

    # if render:
    #     video_folder = "videos/blocks2d-rl-improvisational-training"
    #     Path(video_folder).mkdir(parents=True, exist_ok=True)
    #     env = RecordVideo(
    #         env,
    #         video_folder,
    #         episode_trigger=lambda x: x % 500 == 0,  # Record every 500th episode
    #         name_prefix="training",
    #     )

    # Create and train policy
    policy: RLImprovisationalPolicy[NDArray[np.float32], NDArray[np.float32]] = (
        RLImprovisationalPolicy(env)
    )
    print(f"Training policy for {total_timesteps} timesteps...")
    policy.train(total_timesteps=total_timesteps, seed=seed)

    # Save trained policy
    policy.save(save_path)
    print(f"Saved trained policy to {save_path}")


if __name__ == "__main__":
    train_pushing_policy(
        total_timesteps=500_000,
        seed=42,
        save_path="trained_policies/pushing_policy",
        render=True,
    )
