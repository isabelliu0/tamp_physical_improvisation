"""A block environment in 2D."""

from typing import Any, SupportsFloat, Tuple
from typing import Any, SupportsFloat, Tuple

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
        self._robot_position = np.array(
            [0.5, 1.0], dtype=np.float32
        )  # center of the robot
        self._block_position = np.array(
            [0.0, 0.0], dtype=np.float32
        )  # center of the block
        self._gripper_status = np.float32(0.0)  # the gripper is deactivated

        # Set up the target block position.
        self._target_block_position = np.array([0.5, 0.0], dtype=np.float32)

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

    def _calculate_distance_to_block(self) -> float:
        return float(
            np.linalg.norm(
                np.array(self._robot_position) - np.array(self._block_position)
            )
        )

    def _get_info(self) -> dict[str, Any]:
        return {"distance": self._calculate_distance_to_block()}

    def step(
        self,
        action: NDArray[np.float32],
    ) -> Tuple[NDArray[np.float32], SupportsFloat, bool, bool, dict[str, Any]]:

        # Update the position of the robot.
        dx, dy, new_gripper_status = action

        # Clip the values so that the robot stays in bounds.
        new_x = np.clip(self._robot_position[0] + dx, 0.0, 1.0).astype(np.float32)
        new_y = np.clip(self._robot_position[1] + dy, 0.0, 1.0).astype(np.float32)

        # Update the position and the gripper status of the robot.
        self._robot_position = np.array([new_x, new_y], dtype=np.float32)
        self._gripper_status = new_gripper_status.astype(np.float32)

        distance_to_block = self._calculate_distance_to_block()

        # Check if the robot has collided with the block (when the gripper is deactivated).
        if np.isclose(self._gripper_status, 0.0, atol=1e-6):
            collision_threshold = (self._robot_width + self._block_width) / 2
            collision = distance_to_block < (
                collision_threshold - 1e-5
            )  # Margin for floating point errors
            if collision:
                reward = np.float32(-1.0)
                observation = self._get_obs()
                info = self._get_info()
                return observation, reward, False, True, info

        # Update the position of the block if the gripper suffices the conditions to move the block.
        if self._gripper_status > 0.0 and distance_to_block <= 0.3:
            self._block_position = self._robot_position.copy()
        elif self._gripper_status < 0.0 and distance_to_block <= 0.1:
            self._block_position = np.array(
                [self._robot_position[0], 0.0], dtype=np.float32
            )

        # Check if the robot has reached the goal and if the gripper is deactivated.
        goal_reached = np.array_equal(self._block_position, self._target_block_position)

        # Calculate reward
        reward = np.float32(1.0) if goal_reached else np.float32(0.0)

        terminated = goal_reached
        truncated = False  # False for now since we are using gym's TimeLimit wrapper

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> Tuple[NDArray[np.float32], dict[str, Any]]:
        super().reset(seed=seed)

        self._robot_position = np.array([0.5, 1.0], dtype=np.float32)
        self._block_position = np.array([0.0, 0.0], dtype=np.float32)
        self._gripper_status = np.float32(0.0)

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
                1.0 + max(self._robot_height / 2, self._block_height / 2),
            )
        )

        # Draw the target block.
        target_block_rect = Rectangle.from_center(
            self._target_block_position[0],
            self._target_block_position[1],
            self._block_width,
            self._block_height,
            0.0,
        )
        target_block_rect.plot(ax, facecolor="green", edgecolor="red")

        # Draw the robot.
        robot_rect = Rectangle.from_center(
            self._robot_position[0],
            self._robot_position[1],
            self._robot_width,
            self._robot_height,
            0.0,
        )
        robot_rect.plot(ax, facecolor="silver", edgecolor="black")

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
        plt.close(fig)
        return img
