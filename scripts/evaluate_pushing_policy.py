"""Script to evaluate trained pushing policy."""

from typing import Dict, Optional

import numpy as np

from tamp_improv.approaches.rl_improvisational_policy import RLImprovisationalPolicy
from tamp_improv.benchmarks.blocks2d_env import Blocks2DEnv
from tamp_improv.benchmarks.pushing_env import make_pushing_env


def evaluate_pushing_policy(
    policy_path: str = "trained_policies/pushing_policy",
    num_episodes: int = 50,
    seed: int = 42,
    render: bool = False,
    debug: bool = False,
) -> Optional[Dict]:
    """Evaluate a trained pushing policy.

    Args:
        policy_path: Path to trained policy
        num_episodes: Number of episodes to run
        seed: Random seed
        render: Whether to render the environment
        debug: (for testing)

    Returns: Dict of results
    """
    # Setup
    base_env = Blocks2DEnv(render_mode="rgb_array" if render else None)
    env = make_pushing_env(base_env, seed=seed)

    # # Uncomment to watch a video.
    # from pathlib import Path

    # from gymnasium.wrappers import RecordVideo

    # if render:
    #     video_folder = "videos/blocks2d-rl-improvisational-evaluation"
    #     Path(video_folder).mkdir(parents=True, exist_ok=True)
    #     env = RecordVideo(
    #         env,
    #         video_folder,
    #         episode_trigger=lambda _: True,  # Record all episodes
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
        # obs, _ = env.reset(seed=seed + episode if seed is not None else None)
        obs, _ = env.reset(seed=seed if seed is not None else None)

        if debug:
            print(f"\nEpisode {episode + 1}:")
            print(f"Initial robot position: ({obs[0]:.3f}, {obs[1]:.3f})")
            print(f"Initial block 2 position: ({obs[6]:.3f}, {obs[7]:.3f})")
            print(f"Target area: ({obs[11]:.3f}, {obs[12]:.3f})")

        episode_length = 0.0
        episode_reward = 0.0

        while True:
            # Get action from policy
            action = policy.get_action(obs)

            if debug:
                print(f"\nStep {episode_length + 1}:")
                print(f"Action: {action}")

            obs, reward, terminated, truncated, _ = env.step(action)

            if debug:
                print(f"New robot position: ({obs[0]:.3f}, {obs[1]:.3f})")
                print(f"New block 2 position: ({obs[6]:.3f}, {obs[7]:.3f})")
                print(f"Reward: {reward:.3f}")

            episode_length += 1
            episode_reward += reward

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

    # Calculate and return results
    results = {
        "success_rate": success_count / num_episodes,
        "avg_episode_length": np.mean(episode_lengths),
        "avg_reward": np.mean(rewards_history),
        "min_length": min(episode_lengths),
        "max_length": max(episode_lengths),
        "successes": success_count,
        "total_episodes": num_episodes,
    }

    print("\nEvaluation Summary:")
    print(f"Success Rate: {results['success_rate']:.1%}")
    print(f"Average Episode Length: {results['avg_episode_length']:.1f}")
    print(f"Success Count: {success_count}/{num_episodes}")

    return results


def test_evaluate_pushing_policy():
    """Test trained pushing policy through evaluation."""

    results = evaluate_pushing_policy(
        policy_path="trained_policies/pushing_policy_test",
        num_episodes=1,
        seed=42,
        render=True,
        debug=False,
    )

    assert (
        results["success_rate"] >= 0.9
    ), "Policy should succeed at least 90% of the time"
    assert results["avg_episode_length"] < 50, "Episodes should complete efficiently"