"""A block environment in 2D."""

from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from tomsgeoms2d.structs import Rectangle
from tomsutils.utils import fig2data


class Blocks2DEnv(gym.Env[NDArray[np.float32], NDArray[np.float32]]):
    """A block environment in 2D.

    Observations are 5D:
        - 2D for the x, y position of the robot
        - 2D for the x, y position of the block
        - 1D for the gripper "activation"

    Actions are 3D:
        - 2D for dx, dy for the robot
        - 1D for activating / deactivating the gripper

    The environment has boundaries x=0 to x=1 and y=0 to y=1.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 4}

    def __init__(self, render_mode: str | None = None) -> None:
        self.observation_space = spaces.Box(low=0, high=1, shape=(5,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.array([-0.1, -0.1, -1.0]),
            high=np.array([0.1, 0.1, 1.0]),
            dtype=np.float32,
        )

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # Define constant values for the robot and blocks.
        self._robot_width = 0.2
        self._robot_height = 0.2
        self._block_width = 0.2
        self._block_height = 0.2

        # Set up initial values for robot and block.
        self._robot_position = (0.5, 1.0)  # this is the center of the robot
        self._block_position = (0.0, 0.0)  # this is the center of the block
        self._gripper_status = 0.0  # the gripper is deactivated

    def _get_obs(self) -> NDArray[np.float32]:
        return np.array(
            [
                self._robot_position[0],
                self._robot_position[1],
                self._block_position[0],
                self._block_position[1],
                self._gripper_status,
            ],
            dtype=np.float32,
        )

    def _get_info(self) -> dict[str, Any]:
        return {}

    def step(
        self,
        action: NDArray[np.float32],
    ) -> tuple[NDArray[np.float32], SupportsFloat, bool, bool, dict[str, Any]]:

        # Update the position of the robot.
        dx, dy, gripper_activation_change = action

        # TODO clip the values so that the robot stays in bounds.
        self._robot_position = (
            self._robot_position[0] + dx,
            self._robot_position[1] + dy,
        )

        terminated = False
        reward = 0
        observation = self._get_obs()
        info = self._get_info()
        return observation, reward, terminated, False, info

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        super().reset(seed=seed)

        self._robot_position = (0.5, 1.0)
        self._block_position = (0.0, 0.0)
        self._gripper_status = 0.0

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def render(self) -> NDArray[np.uint8]:  # type: ignore
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.set_xlim(
            (
                0.0 - max(self._robot_width / 2, self._block_width / 2),
                1.0 + max(self._robot_width / 2, self._block_width / 2),
            )
        )
        ax.set_ylim(
            (
                0.0 - max(self._robot_height / 2, self._block_height / 2),
                1.0 + max(self._robot_height / 2, self._robot_height / 2),
            )
        )

        # Draw the robot.
        robot_rect = Rectangle.from_center(
            self._robot_position[0],
            self._robot_position[1],
            self._robot_width,
            self._robot_height,
            0.0,
        )
        robot_rect.plot(ax, facecolor="purple", edgecolor="black")

        # Draw the block.
        block_rect = Rectangle.from_center(
            self._block_position[0],
            self._block_position[1],
            self._block_width,
            self._block_height,
            0.0,
        )
        block_rect.plot(ax, facecolor="blue", edgecolor="black")

        img = fig2data(fig)
        return img
