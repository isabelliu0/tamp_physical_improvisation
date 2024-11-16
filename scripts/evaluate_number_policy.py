"""Script to evaluate trained number transition policy."""

import numpy as np

from tamp_improv.approaches.rl_improvisational_policy import RLImprovisationalPolicy
from tamp_improv.benchmarks.number_env import NumberEnv
from tamp_improv.benchmarks.number_env_wrapper import make_number_env_wrapper


def evaluate_number_policy(
    policy_path: str = "trained_policies/number_policy",
    num_episodes: int = 50,
    seed: int = 42,
) -> None:
    """Evaluate a trained number transition policy.

    Args:
        policy_path: Path to trained policy
        num_episodes: Number of episodes to run
        seed: Random seed
    """
    # Create environment
    base_env = NumberEnv()
    env = make_number_env_wrapper(base_env, seed=seed)

    # Load policy
    policy: RLImprovisationalPolicy[int, int] = RLImprovisationalPolicy(env)
    policy.load(policy_path)

    success_count = 0
    episode_lengths = []
    rewards_history = []

    # Run episodes
    for episode in range(num_episodes):
        obs, _ = env.reset()

        episode_length = 0
        episode_reward = 0.0
        step = 0

        while True:
            # Get action from policy
            action = policy.get_action(obs)

            # Take step
            obs, reward, terminated, truncated, _ = env.step(action)

            episode_length += 1
            episode_reward += reward
            step += 1

            if terminated or truncated:
                if terminated:  # Successfully reached state 2
                    success_count += 1
                episode_lengths.append(episode_length)
                rewards_history.append(episode_reward)

                print(
                    f"Episode {episode + 1}: "
                    f"{'Success' if terminated else 'Failed'} in {episode_length} steps."
                    f"State: {obs}, Total reward: {episode_reward:.1f}"
                )
                break

    print("\nEvaluation Summary:")
    print(f"Success Rate: {(success_count / num_episodes):.1%}")
    print(f"Average Episode Length: {np.mean(episode_lengths):.1f}")
    print(f"Success Count: {success_count}/{num_episodes}")
    print(f"Average Reward: {np.mean(rewards_history):.2f}")


if __name__ == "__main__":
    evaluate_number_policy(
        policy_path="trained_policies/number_policy",
        num_episodes=50,
        seed=42,
    )
