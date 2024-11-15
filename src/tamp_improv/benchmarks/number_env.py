"""A simple environment with integer state transitions."""

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym


class NumberEnv(gym.Env):
    """Simple number environment with states {0,1,2,3}.

    States: 0,1,2,3
    Legal transitions: 0->1, 1->2, 2->3
    Initial state: 0
    Goal state: 3
    Action: Single integer
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 4}

    def __init__(self, render_mode: str | None = None) -> None:
        # Single integer action in {0,1}
        self.action_space = gym.spaces.Discrete(2)

        # Single integer observation (the state)
        self.observation_space = gym.spaces.Discrete(4)

        self.state = 0  # Start at state 0

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[int, Dict[str, Any]]:
        super().reset(seed=seed)

        self.state = 0
        return self.state, {}

    def step(self, action: int) -> Tuple[int, float, bool, bool, Dict[str, Any]]:
        current_state = self.state

        if action == 1:
            if current_state < 3:
                self.state += 1

        # Reward is 1 only if we reach the goal state, 0 otherwise
        reward = float(self.state == 3)

        terminated = self.state == 3
        truncated = False

        return self.state, reward, terminated, truncated, {}

    def render(self) -> None:
        """Render the environment.

        For this simple environment, we just print the current state.
        """
        print(f"Current state: {self.state}")
