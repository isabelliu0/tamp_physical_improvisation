"""Core number environment."""

from typing import Any, NamedTuple, Union

import gymnasium as gym
import numpy as np
from gymnasium.spaces import MultiDiscrete
from numpy.typing import NDArray


class NumberState(NamedTuple):
    """State of the number environment."""

    state: int
    light_switch: int


class NumberEnv(gym.Env):
    """Number environment in 1D.

    Observations are 2D:
    - Current state {0, 1, 2}
    - Binary light switch {0, 1}

    Actions are 2D:
    - Stay or move forward {0, 1}
    - Toggle light switch {0, 1}
    """

    def __init__(self, render_mode: str | None = None) -> None:
        self.action_space = MultiDiscrete([2, 2])
        self.observation_space = MultiDiscrete([3, 2])
        self.state = 0
        self.light_switch = 0
        self.render_mode = render_mode

    def reset_from_state(
        self,
        state: Union[NumberState, NDArray[np.int32]],
        *,
        seed: int | None = None,
    ) -> tuple[NDArray[np.int32], dict[str, Any]]:
        """Reset environment to specific state."""
        super().reset(seed=seed)

        if isinstance(state, np.ndarray):
            if state.shape != (2,):
                raise ValueError(f"Expected state shape (2,), got {state.shape}")
            new_state = int(state[0])
            new_light = int(state[1])
        elif isinstance(state, NumberState):
            new_state = state.state
            new_light = state.light_switch
        else:
            raise ValueError(f"Unsupported state type: {type(state)}")

        # Validate states
        if not 0 <= new_state <= 2:
            raise ValueError(f"Invalid state: {new_state}")
        if new_light not in {0, 1}:
            raise ValueError(f"Invalid light switch: {new_light}")

        self.state = new_state
        self.light_switch = new_light
        return self._get_obs(), {}

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[NDArray[np.int32], dict[str, Any]]:
        """Reset environment to initial state."""
        super().reset(seed=seed)
        self.state = 0
        self.light_switch = 0
        return self._get_obs(), {}

    def _get_obs(self) -> NDArray[np.int32]:
        """Get current observation."""
        return np.array([self.state, self.light_switch])

    def step(
        self, action: NDArray[np.int32]
    ) -> tuple[NDArray[np.int32], float, bool, bool, dict[str, Any]]:
        """Take environment step."""
        movement, toggle_switch = action

        self.light_switch = toggle_switch
        if movement == 1 and self.state < 2:
            self.state += 1

        obs = self._get_obs()
        reward = float(self.state == 2)
        terminated = self.state == 2

        return obs, reward, terminated, False, {}

    def render(self) -> None:
        """Render the environment."""
        print(f"state: {self.state}, light switch: {self.light_switch}")
