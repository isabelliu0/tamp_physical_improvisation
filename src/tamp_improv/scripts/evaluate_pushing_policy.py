"""Script to visualize and evaluate trained pushing policy."""

import argparse
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from typing import Optional, Dict

from tamp_improv.benchmarks.blocks2d_env import Blocks2DEnv
from tamp_improv.benchmarks.pushing_env import make_pushing_env
from tamp_improv.approaches.rl_improvisational_policy import RLImprovisationalPolicy


def evaluate_policy(
    policy_path: str = "trained_policies/pushing_policy",
    num_episodes: int = 50,
    seed: int = 42,
    render: bool = True,
    delay: float = 0.1,  # seconds between rendered frames
    save_video: bool = False,
    debug: bool = False,
) -> Optional[Dict]:
    """Evaluate a trained pushing policy.
    
    Args:
        policy_path: Path to trained policy
        num_episodes: Number of episodes to run
        seed: Random seed
        render: Whether to render the environment
        delay: Delay between frames when rendering
        save_video: Whether to save rendered frames
        debug: (for testing)
        
    Returns: Dict of results
    """
    env = Blocks2DEnv(render_mode="rgb_array" if render else None)
    pushing_env = make_pushing_env(env, seed=seed)

    if debug:
        print("\nInitializing policy evaluation...")
        print(f"Policy path: {policy_path}")
    
    policy = RLImprovisationalPolicy(pushing_env)
    try:
        policy.load(policy_path)
        if debug:
            print("Successfully loaded policy")
    except Exception as e:
        print(f"Error loading policy: {e}")
        raise

    if save_video:
        video_dir = Path("videos/pushing_policy_eval")
        video_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0
    episode_lengths = []
    rewards_history = []
        
    for episode in range(num_episodes):
        obs, _ = pushing_env.reset(seed=seed + episode) # Different seed for each episode
        if debug:
            print(f"\nEpisode {episode + 1}:")
            print(f"Initial block 2 position (x,y): ({obs[6]:.3f}, {obs[7]:.3f})")
            print(f"Target area (x,y,w,h): ({obs[11]:.3f}, {obs[12]:.3f}, {obs[13]:.3f}, {obs[14]:.3f})")

        episode_length = 0
        episode_reward = 0
        is_blocked_initial = pushing_env.is_target_area_blocked(obs)
        
        if debug and not is_blocked_initial:
            print(f"Warning: Initial state not blocked!")

        frames = []
        
        while True:
            if render:
                frame = env.render()
                if frame is not None:
                    if save_video:
                        frames.append(frame)
                    plt.imshow(frame)
                    plt.axis('off')
                    plt.pause(delay)
                    plt.clf()

            # Get action from policy
            action = policy.get_action(obs)
            if debug and episode_length == 0:
                print(f"First action: {action}")

            obs, reward, terminated, truncated, _ = pushing_env.step(action)
            
            episode_length += 1
            episode_reward += reward
            
            if terminated or truncated:
                if terminated:  # Successfully cleared area
                    success_count += 1
                episode_lengths.append(episode_length)
                rewards_history.append(episode_reward)
                
                if save_video and terminated:  # Save successful episodes
                    episode_dir = video_dir / f"episode_{episode}"
                    episode_dir.mkdir(exist_ok=True)
                    for i, frame in enumerate(frames):
                        plt.imsave(
                            episode_dir / f"frame_{i:03d}.png",
                            frame
                        )
                
                print(f"Episode {episode + 1}: "
                        f"{'Success' if terminated else 'Failure'} in {episode_length} steps. "
                        f"Total reward: {episode_reward:.1f}")
                break

    success_rate = success_count / num_episodes
    avg_length = np.mean(episode_lengths)
    avg_reward = np.mean(rewards_history)

    print("\nEvaluation Summary:")
    print(f"Success Rate: {success_rate:.1%}")
    print(f"Average Episode Length: {avg_length:.1f} steps")
    print(f"Shortest Episode: {min(episode_lengths)} steps")
    print(f"Longest Episode: {max(episode_lengths)} steps")
    print(f"Average Reward: {avg_reward:.1f}")

    if render:
        plt.close()
        
    return {
        'success_rate': success_rate,
        'avg_episode_length': avg_length,
        'avg_reward': avg_reward,
        'min_length': min(episode_lengths),
        'max_length': max(episode_lengths),
        'all_lengths': episode_lengths,
        'all_rewards': rewards_history,
        'successes': success_count,
        'total_episodes': num_episodes
    }
