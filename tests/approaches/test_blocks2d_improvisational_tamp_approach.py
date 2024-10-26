"""Test blocks2d_improvisational_tamp_approach.py."""

from gymnasium.wrappers import TimeLimit

from tamp_improv.approaches.blocks2d_improvisational_tamp_approach import (
    Blocks2DImprovisationalTAMPApproach,
)
from tamp_improv.benchmarks.blocks2d_env import Blocks2DEnv


def test_blocks2d_improvisational_tamp_approach():
    """Tests for Blocks2DImprovisationalTAMPApproach()."""

    env = Blocks2DEnv(render_mode="rgb_array")
    env = TimeLimit(env, max_episode_steps=100)

    # # Uncomment to watch a video.
    # from gymnasium.wrappers import RecordVideo

    # env = RecordVideo(env, "videos/blocks2d-improvisational-tamp-test")

    approach = Blocks2DImprovisationalTAMPApproach(
        env.observation_space, env.action_space, seed=123
    )

    obs, info = env.reset()
    approach.reset(obs, info)

    total_reward = 0
    for step in range(100):  # should terminate earlier
        action = approach.step(
            obs, 0, False, False, info
        )  # Get action before stepping the environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        print(f"Step {step + 1}: Action: {action}, Reward: {reward}")

        if terminated or truncated:
            print(f"Episode finished after {step + 1} steps")
            print(f"Total reward: {total_reward}")
            break
    else:
        print("Episode didn't finish within 100 steps")

    env.close()
