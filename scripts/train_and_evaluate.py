"""Training and evaluation scripts for improvisational policies."""

from pathlib import Path
from typing import TypeVar

import numpy as np

from tamp_improv.approaches.rl import RLImprovisationalPolicy, RLPolicyConfig
from tamp_improv.benchmarks.base import BaseTAMPSystem
from tamp_improv.benchmarks.blocks2d import Blocks2DTAMPSystem
from tamp_improv.benchmarks.number import NumberTAMPSystem

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


def train_policy(
    system: BaseTAMPSystem[ObsType, ActType],
    total_timesteps: int,
    save_path: str,
    seed: int = 42,
    render: bool = False,
) -> None:
    """Train policy for a system."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    if render:
        system.env = type(system.env)(render_mode="rgb_array")
        video_folder = f"videos/{system.name}-training"
        Path(video_folder).mkdir(parents=True, exist_ok=True)
        from gymnasium.wrappers import RecordVideo

        system.env = RecordVideo(system.env, video_folder)

    policy = RLImprovisationalPolicy(RLPolicyConfig(seed=seed))
    print(f"Training policy for {total_timesteps} timesteps...")
    policy.train(system.wrapped_env, total_timesteps=total_timesteps, seed=seed)
    policy.save(save_path)
    print(f"Saved trained policy to {save_path}")


def evaluate_policy(
    system: BaseTAMPSystem[ObsType, ActType],
    policy_path: str,
    num_episodes: int = 50,
    seed: int = 42,
    render: bool = False,
) -> None:
    """Evaluate trained policy."""
    if render:
        system.env = type(system.env)(render_mode="rgb_array")
        video_folder = f"videos/{system.name}-evaluation"
        Path(video_folder).mkdir(parents=True, exist_ok=True)
        from gymnasium.wrappers import RecordVideo

        system.env = RecordVideo(system.env, video_folder)

    policy = RLImprovisationalPolicy(RLPolicyConfig(seed=seed))
    policy.load(policy_path)

    success_count = 0
    episode_lengths = []
    rewards_history = []

    for episode in range(num_episodes):
        obs, _ = system.reset()
        total_reward = 0.0
        step = 0

        while True:
            action = policy.get_action(obs)
            obs, reward, terminated, truncated, _ = system.env.step(action)
            total_reward += reward
            step += 1

            if terminated or truncated:
                if terminated:
                    success_count += 1
                episode_lengths.append(step)
                rewards_history.append(total_reward)
                print(f"Episode {episode + 1}: {'Success' if terminated else 'Failed'}")
                break

    print(f"\nSuccess Rate: {success_count/num_episodes:.1%}")
    print(f"Average Length: {np.mean(episode_lengths):.1f}")
    print(f"Average Reward: {np.mean(rewards_history):.2f}")


if __name__ == "__main__":
    # Train and evaluate Blocks2D policy
    blocks_system = Blocks2DTAMPSystem.create_default(seed=42)
    train_policy(
        blocks_system,
        total_timesteps=500_000,
        save_path="trained_policies/blocks2d_policy",
        render=True,
    )
    evaluate_policy(
        blocks_system, policy_path="trained_policies/blocks2d_policy", render=True
    )

    # Train and evaluate Number policy
    number_system = NumberTAMPSystem.create_default(seed=42)
    train_policy(
        number_system,
        total_timesteps=10_000,
        save_path="trained_policies/number_policy",
    )
    evaluate_policy(number_system, policy_path="trained_policies/number_policy")
