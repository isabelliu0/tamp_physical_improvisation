"""Number environment wrapper for learning the 1->2 transition."""

from typing import Any

import gymnasium as gym

from tamp_improv.benchmarks.number_env import NumberEnv


class NumberEnvWrapper(gym.Env):
    """Wrapper for learning the 1->2 transition."""

    def __init__(self, base_env: NumberEnv) -> None:
        self.env = base_env
        self.max_episode_steps = 10

        self.observation_space = base_env.observation_space
        self.action_space = base_env.action_space
        self.steps = 0

        self.render_mode = self.env.render_mode

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        self.steps = 0
        return self.env.reset(seed=seed)

    def step(self, action: int) -> tuple[int, float, bool, bool, dict[str, Any]]:
        obs, _, _, truncated, info = self.env.step(action)
        self.steps += 1

        # Success is reaching state 2
        success = obs == 2
        reward = 1.0 if success else 0.0

        terminated = success
        truncated = truncated or self.steps >= self.max_episode_steps

        return obs, reward, terminated, truncated, info

    def render(self) -> None:
        self.env.render()


def make_number_env_wrapper(
    env: NumberEnv, max_episode_steps: int = 10, seed: int | None = None
) -> NumberEnvWrapper:
    """Create a transition learning environment."""
    wrapped_env = NumberEnvWrapper(env)
    wrapped_env.max_episode_steps = max_episode_steps
    if seed is not None:
        wrapped_env.reset(seed=seed)
    return wrapped_env
