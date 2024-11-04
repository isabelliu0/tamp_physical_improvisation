"""Script to evaluate trained pushing policy."""

import numpy as np

from tamp_improv.approaches.rl_improvisational_policy import RLImprovisationalPolicy
from tamp_improv.benchmarks.blocks2d_env import Blocks2DEnv
from tamp_improv.benchmarks.pushing_env import make_pushing_env


def evaluate_pushing_policy(
    policy_path: str = "trained_policies/pushing_policy",
    num_episodes: int = 50,
    seed: int = 42,
    render: bool = False,
) -> None:
    """Evaluate a trained pushing policy.

    Args:
        policy_path: Path to trained policy
        num_episodes: Number of episodes to run
        seed: Random seed
        render: Whether to render the environment
        debug: Whether to print debug information
    """
    # Create environment with evaluation seeds
    base_env = Blocks2DEnv(render_mode="rgb_array" if render else None)
    env = make_pushing_env(
        base_env,
        seed=seed,
    )

    # # Uncomment to watch a video.
    # from pathlib import Path

    # from gymnasium.wrappers import RecordVideo

    # if render:
    #     video_folder = "videos/blocks2d-rl-improvisational-evaluation"
    #     Path(video_folder).mkdir(parents=True, exist_ok=True)
    #     env = RecordVideo(
    #         env,
    #         video_folder,
    #         episode_trigger=lambda x: x % 10 == 0,  # Record every 10th episode
    #         name_prefix="evaluation",
    #     )

    # Load policy
    policy = RLImprovisationalPolicy(env)
    policy.load(policy_path)

    success_count = 0
    episode_lengths = []
    rewards_history = []

    # Run episodes
    for episode in range(num_episodes):
        obs, _ = env.reset()

        episode_length = 0.0
        episode_reward = 0.0
        step = 0

        while True:
            # Get action from policy
            action = policy.get_action(obs)

            obs, reward, terminated, truncated, _ = env.step(action)

            episode_length += 1
            episode_reward += reward
            step += 1

            if terminated or truncated:
                if terminated:  # Successfully cleared area
                    success_count += 1
                episode_lengths.append(episode_length)
                rewards_history.append(episode_reward)

                print(
                    f"Episode {episode + 1}: "
                    f"{'Succeed' if terminated else 'Fail'} in {episode_length} steps."
                    f"Total reward: {episode_reward:.1f}"
                )
                break

    print("\nEvaluation Summary:")
    print(f"Success Rate: {(success_count / num_episodes):.1%}")
    print(f"Average Episode Length: {(np.mean(episode_lengths)):.1f}")
    print(f"Success Count: {success_count}/{num_episodes}")


if __name__ == "__main__":
    evaluate_pushing_policy(
        policy_path="trained_policies/pushing_policy",
        num_episodes=50,
        seed=42,
        render=True,
    )
