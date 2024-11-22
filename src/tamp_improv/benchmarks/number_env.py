"""Core number environment."""

from typing import Any

import gymnasium as gym
from gymnasium.spaces import Discrete


class NumberEnv(gym.Env):
    """Number environment with states {0,1,2}.

    States: 0,1,2
    Legal transitions: 0->1, 1->2
    Initial state: 0
    Goal state: 2
    Action: Single integer
    """

    def __init__(self, render_mode: str | None = None) -> None:
        self.action_space = Discrete(2)
        self.observation_space = Discrete(3)
        self.state = 0
        self.render_mode = render_mode

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        super().reset(seed=seed)
        self.state = 0
        return self.state, {}

    def step(self, action: int) -> tuple[int, float, bool, bool, dict[str, Any]]:
        if action == 1 and self.state < 2:
            self.state += 1
        reward = float(self.state == 2)
        terminated = self.state == 2
        return self.state, reward, terminated, False, {}

    def render(self) -> None:
        print(f"Current state: {self.state}")
