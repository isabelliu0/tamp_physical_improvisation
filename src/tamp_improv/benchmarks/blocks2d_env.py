"""Core blocks2d environment."""

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from tomsgeoms2d.structs import Rectangle
from tomsutils.utils import fig2data


def is_block_1_in_target_area(
    block_x: float,
    block_y: float,
    block_width: float,
    block_height: float,
    target_x: float,
    target_y: float,
    target_width: float,
    target_height: float,
) -> bool:
    """Check if block is completely in target area -- goal."""
    target_left = target_x - target_width / 2
    target_right = target_x + target_width / 2
    target_bottom = target_y - target_height / 2
    target_top = target_y + target_height / 2

    block_left = block_x - block_width / 2
    block_right = block_x + block_width / 2
    block_bottom = block_y - block_height / 2
    block_top = block_y + block_height / 2

    return (
        target_left <= block_left
        and block_right <= target_right
        and target_bottom <= block_bottom
        and block_top <= target_top
    )


class Blocks2DEnv(gym.Env):
    """A block environment in 2D.

    Observations are 15D:
        - 4D for the x, y position (center), the width, and the height of
        the robot
        - 2D for the x, y position (center) of block 1 (the target block)
        - 2D for the x, y position (center) of block 2 (the other block)
        - 2D for the width and the height of the blocks
        - 1D for the gripper "activation"
        - 4D for the x, y position (center), the width, and the height of
        the target area

    Actions are 3D:
        - 2D for dx, dy for the robot
        - 1D for activating / deactivating the gripper

    The environment has boundaries x=0 to x=1 and y=0 to y=1.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 4}

    def __init__(self, render_mode: str | None = None) -> None:
        self.observation_space = Box(low=0, high=1, shape=(15,), dtype=np.float32)
        self.action_space = Box(
            low=np.array([-0.1, -0.1, -1.0]),
            high=np.array([0.1, 0.1, 1.0]),
            dtype=np.float32,
        )

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # set constants
        self._robot_width = 0.2
        self._robot_height = 0.2
        self._block_width = 0.2
        self._block_height = 0.2
        self._target_area = {"x": 0.5, "y": 0.0, "width": 0.2, "height": 0.2}

        # Initialize positions and status
        self.robot_position = np.array([0.5, 1.0], dtype=np.float32)
        self.block_1_position = np.array([0.0, 0.0], dtype=np.float32)
        self.block_2_position = np.array([0.5, 0.0], dtype=np.float32)
        self.gripper_status = np.float32(0.0)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        super().reset(seed=seed)
        if options is None:
            options = {}

        self.robot_position = options.get(
            "robot_pos", np.array([0.5, 1.0], dtype=np.float32)
        ).copy()
        self.block_1_position = options.get(
            "block_1_pos", np.array([0.0, 0.0], dtype=np.float32)
        ).copy()
        self.block_2_position = options.get(
            "block_2_pos", np.array([0.5, 0.0], dtype=np.float32)
        ).copy()
        self.gripper_status = np.float32(0.0)

        return self._get_obs(), self._get_info()

    def _get_obs(self) -> NDArray[np.float32]:
        return np.array(
            [
                self.robot_position[0],
                self.robot_position[1],
                self._robot_width,
                self._robot_height,
                self.block_1_position[0],
                self.block_1_position[1],
                self.block_2_position[0],
                self.block_2_position[1],
                self._block_width,
                self._block_height,
                self.gripper_status,
                self._target_area["x"],
                self._target_area["y"],
                self._target_area["width"],
                self._target_area["height"],
            ],
            dtype=np.float32,
        )

    def _get_info(self) -> dict[str, Any]:
        return {
            "distance_to_block1": np.linalg.norm(
                self.robot_position - self.block_1_position
            ),
            "distance_to_block2": np.linalg.norm(
                self.robot_position - self.block_2_position
            ),
        }

    def step(
        self,
        action: NDArray[np.float32],
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        dx, dy, gripper_action = action

        # Save previous positions
        robot_position_prev = self.robot_position.copy()
        prevgripper_status = self.gripper_status

        # Update robot position and gripper
        self.robot_position[0] = np.clip(self.robot_position[0] + dx, 0.0, 1.0).astype(
            np.float32
        )
        self.robot_position[1] = np.clip(self.robot_position[1] + dy, 0.0, 1.0).astype(
            np.float32
        )
        self.gripper_status = gripper_action.astype(np.float32)

        # Handle block 2 pushing
        if self._is_adjacent(robot_position_prev, self.block_2_position):
            relative_pos = robot_position_prev[0] - self.block_2_position[0]
            if relative_pos * dx < 0.0:  # Push
                self.block_2_position[0] = np.clip(
                    self.block_2_position[0] + dx, 0.0, 1.0
                ).astype(np.float32)

        # Handle block 1 interactions (pick/drop)
        distance = np.linalg.norm(self.robot_position - self.block_1_position)

        # Case 1: Robot is picking up the block
        if (
            self.gripper_status > 0.0
            and distance <= ((self._robot_width + self._block_width) / 2) + 1e-3
        ):
            self.block_1_position = self.robot_position.copy()

        # Case 2: Robot was holding the block and gripper is deactivated
        elif (
            0.0 < prevgripper_status
            and self.gripper_status <= 0.0
            and np.allclose(self.block_1_position, robot_position_prev, atol=1e-3)
        ):
            self.block_1_position = np.array(
                [self.robot_position[0], 0.0], dtype=np.float32
            )

        # Case 3: Robot is holding the block (continue moving it)
        elif self.gripper_status > 0.0 and np.allclose(
            self.block_1_position, robot_position_prev, atol=1e-3
        ):
            self.block_1_position = self.robot_position.copy()

        # Check for collision between all pairs
        obs = self._get_obs()
        info = self._get_info()

        # Robot-Block1 collision
        if self._check_collisions(self.robot_position, self.block_1_position):
            if np.isclose(self.gripper_status, 0.0, atol=1e-3):
                return obs, -1.0, False, True, info

        # Robot-Block2 collision
        if self._check_collisions(self.robot_position, self.block_2_position):
            return obs, -1.0, False, True, info

        # Block1-Block2 collision
        if self._check_collisions(self.block_1_position, self.block_2_position):
            return obs, -1.0, False, True, info

        # Check if the robot has reached the goal
        goal_reached = is_block_1_in_target_area(
            self.block_1_position[0],
            self.block_1_position[1],
            self._block_width,
            self._block_height,
            self._target_area["x"],
            self._target_area["y"],
            self._target_area["width"],
            self._target_area["height"],
        )

        reward = 1.0 if goal_reached else 0.0
        terminated = goal_reached

        return obs, reward, terminated, False, info

    def _check_collisions(
        self,
        pos1: NDArray[np.float32],
        pos2: NDArray[np.float32],
    ) -> bool:
        dx = abs(pos1[0] - pos2[0])
        dy = abs(pos1[1] - pos2[1])

        # Use block dimensions for both blocks, and robot dimensions
        # when robot is involved
        width_sum = self._block_width - 1e-3
        height_sum = self._block_height - 1e-3

        if np.array_equal(pos1, self.robot_position) or np.array_equal(
            pos2, self.robot_position
        ):
            width_sum = (self._robot_width + self._block_width) / 2 - 1e-3
            height_sum = (self._robot_height + self._block_height) / 2 - 1e-3

        return dx < width_sum and dy < height_sum

    def _is_adjacent(
        self,
        robot_position: NDArray[np.float32],
        block_position: NDArray[np.float32],
    ) -> bool:
        vertical_aligned = (
            np.abs(robot_position[1] - block_position[1])
            < (self._robot_height + self._block_height) / 4
        )
        horizontal_adjacent = np.isclose(
            np.abs(robot_position[0] - block_position[0]),
            (self._robot_width + self._block_width) / 2,
            atol=2e-2,  # tolerance to make the task easier for RL agents
        )
        return vertical_aligned and horizontal_adjacent

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

        # Draw the target area
        target_rect = Rectangle.from_center(
            self._target_area["x"],
            self._target_area["y"],
            self._target_area["width"],
            self._target_area["height"],
            0.0,
        )
        target_rect.plot(ax, facecolor="green", edgecolor="red")

        # Draw the robot.
        robot_rect = Rectangle.from_center(
            self.robot_position[0],
            self.robot_position[1],
            self._robot_width,
            self._robot_height,
            0.0,
        )
        robot_rect.plot(ax, facecolor="silver", edgecolor="black")

        # Draw the blocks.
        for block_position in [self.block_1_position, self.block_2_position]:
            block_rect = Rectangle.from_center(
                block_position[0],
                block_position[1],
                self._block_width,
                self._block_height,
                0.0,
            )
            block_rect.plot(ax, facecolor="blue", edgecolor="black")

        img = fig2data(fig)
        plt.close(fig)
        return img
