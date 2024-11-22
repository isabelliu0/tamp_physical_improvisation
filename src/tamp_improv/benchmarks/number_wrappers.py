"""Wrappers for number environment."""

from typing import Any

import gymnasium as gym


class NumberEnvWrapper(gym.Env):
    """Wrapped environment for learning the 1->2 transition."""

    def __init__(self, base_env: gym.Env) -> None:
        self.env = base_env
        self.action_space = base_env.action_space
        self.observation_space = base_env.observation_space
        self.max_episode_steps = 10
        self.steps = 0
        self.render_mode = base_env.render_mode

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
