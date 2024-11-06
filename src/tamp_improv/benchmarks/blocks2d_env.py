"""A block environment in 2D."""

from typing import Any, SupportsFloat, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from tomsgeoms2d.structs import Rectangle
from tomsutils.utils import fig2data


def is_block_in_target_area(
    block_x: float,
    block_y: float,
    block_width: float,
    block_height: float,
    target_x: float,
    target_y: float,
    target_width: float,
    target_height: float,
) -> bool:
    """Checks if the block 1 is in the target area."""
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


class Blocks2DEnv(gym.Env[NDArray[np.float32], NDArray[np.float32]]):
    """A block environment in 2D.

    Observations are 15D:
        - 4D for the x, y position (center), the width, and the height of the robot
        - 2D for the x, y position (center) of block 1 (the target block)
        - 2D for the x, y position (center) of block 2 (the other block)
        - 2D for the width and the height of the blocks
        - 1D for the gripper "activation"
        - 4D for the x, y position (center), the width, and the height of the target area

    Actions are 3D:
        - 2D for dx, dy for the robot
        - 1D for activating / deactivating the gripper

    The environment has boundaries x=0 to x=1 and y=0 to y=1.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 4}

    def __init__(self, render_mode: str | None = None) -> None:
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(15,), dtype=np.float32
        )
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
        self._block_1_position = np.array(
            [0.0, 0.0], dtype=np.float32
        )  # center of block 1
        self._block_2_position = np.array(
            [0.5, 0.0], dtype=np.float32
        )  # center of block 2
        self._gripper_status = np.float32(0.0)  # the gripper is deactivated

        # Set up the target area. (x, y) is the center of the area.
        self._target_area = {"x": 0.5, "y": 0.0, "width": 0.2, "height": 0.2}

    def _get_obs(self) -> NDArray[np.float32]:
        return np.array(
            [
                self._robot_position[0],
                self._robot_position[1],
                self._robot_width,
                self._robot_height,
                self._block_1_position[0],
                self._block_1_position[1],
                self._block_2_position[0],
                self._block_2_position[1],
                self._block_width,
                self._block_height,
                self._gripper_status,
                self._target_area["x"],
                self._target_area["y"],
                self._target_area["width"],
                self._target_area["height"],
            ],
            dtype=np.float32,
        )

    def _calculate_distance_to_block(
        self, block_position: NDArray[np.float32]
    ) -> float:
        return float(np.linalg.norm(self._robot_position - block_position))

    def _get_info(self) -> dict[str, Any]:
        return {
            "distance_to_block1": self._calculate_distance_to_block(
                self._block_1_position
            ),
            "distance_to_block2": self._calculate_distance_to_block(
                self._block_2_position
            ),
            "collision_occurred": False,
        }

    def step(
        self,
        action: NDArray[np.float32],
    ) -> Tuple[NDArray[np.float32], SupportsFloat, bool, bool, dict[str, Any]]:

        # Update the position of the robot.
        dx, dy, new_gripper_status = action

        # Save the previous robot position for push/pull interactions
        robot_position_prev = self._robot_position.copy()
        prev_gripper_status = self._gripper_status

        # Update the position and the gripper status of the robot
        new_robot_x = np.clip(self._robot_position[0] + dx, 0.0, 1.0).astype(np.float32)
        new_robot_y = np.clip(self._robot_position[1] + dy, 0.0, 1.0).astype(np.float32)
        self._robot_position = np.array([new_robot_x, new_robot_y], dtype=np.float32)
        self._gripper_status = new_gripper_status.astype(np.float32)

        # Handle block 2 interactions (push)
        if self._is_adjacent(robot_position_prev, self._block_2_position):
            relative_pos = robot_position_prev[0] - self._block_2_position[0]
            if relative_pos * dx < 0.0:  # Push
                self._block_2_position[0] = np.clip(
                    self._block_2_position[0] + dx, 0.0, 1.0
                ).astype(np.float32)

        # Handle block 1 interactions (pick/drop)
        distance = self._calculate_distance_to_block(self._block_1_position)

        # Case 1: Robot is picking up the block
        if (
            self._gripper_status > 0.0
            and distance <= ((self._robot_width + self._block_width) / 2) + 1e-3
        ):
            self._block_1_position = self._robot_position.copy()

        # Case 2: Robot was holding the block and gripper is deactivated
        elif (
            0.0 < prev_gripper_status
            and self._gripper_status <= 0.0
            and np.allclose(self._block_1_position, robot_position_prev, atol=1e-3)
        ):
            self._block_1_position = np.array(
                [self._robot_position[0], 0.0], dtype=np.float32
            )

        # Case 3: Robot is holding the block (continue moving it)
        elif self._gripper_status > 0.0 and np.allclose(
            self._block_1_position, robot_position_prev, atol=1e-3
        ):
            self._block_1_position = self._robot_position.copy()

        # Check for collision between all pairs
        observation = self._get_obs()
        info = self._get_info()

        # Robot-Block1 collision
        if self._check_collisions(self._robot_position, self._block_1_position):
            if np.isclose(self._gripper_status, 0.0, atol=1e-3):
                return observation, float(-1.0), False, True, info

        # Robot-Block2 collision
        if self._check_collisions(self._robot_position, self._block_2_position):
            return observation, float(-1.0), False, True, info

        # Block1-Block2 collision
        if self._check_collisions(self._block_1_position, self._block_2_position):
            return observation, float(-1.0), False, True, info

        # Check if the robot has reached the goal
        goal_reached = is_block_in_target_area(
            self._block_1_position[0],
            self._block_1_position[1],
            self._block_width,
            self._block_height,
            self._target_area["x"],
            self._target_area["y"],
            self._target_area["width"],
            self._target_area["height"],
        )

        # Calculate reward
        reward = float(1.0) if goal_reached else float(0.0)

        terminated = goal_reached
        truncated = False

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def _check_collisions(
        self, pos1: NDArray[np.float32], pos2: NDArray[np.float32]
    ) -> bool:
        """Check if two objects are colliding."""
        dx = abs(pos1[0] - pos2[0])
        dy = abs(pos1[1] - pos2[1])

        # Use block dimensions for both blocks, and robot dimensions when robot
        # is involved
        width_sum = self._block_width - 1e-3
        height_sum = self._block_height - 1e-3

        if np.array_equal(pos1, self._robot_position) or np.array_equal(
            pos2, self._robot_position
        ):
            width_sum = (self._robot_width + self._block_width) / 2 - 1e-3
            height_sum = (self._robot_height + self._block_height) / 2 - 1e-3

        return dx < width_sum and dy < height_sum

    def _is_adjacent(
        self, robot_position: NDArray[np.float32], block_position: NDArray[np.float32]
    ) -> bool:
        """Check if robot is adjacent to a block (for pushing/pulling)."""
        vertical_aligned = (
            np.abs(robot_position[1] - block_position[1])
            < (self._robot_height + self._block_height) / 4
        )
        horizontal_adjacent = np.isclose(
            np.abs(robot_position[0] - block_position[0]),
            (self._robot_width + self._block_width) / 2,
            atol=1e-2,  # tolerance to make the task easier for RL agents
        )
        return vertical_aligned and horizontal_adjacent

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> Tuple[NDArray[np.float32], dict[str, Any]]:
        super().reset(seed=seed)

        if options is None:
            options = {}

        # Set positions from options if provided, or use defaults
        self._robot_position = options.get(
            "robot_pos", np.array([0.5, 1.0], dtype=np.float32)
        )
        self._block_1_position = options.get(
            "block_1_pos", np.array([0.0, 0.0], dtype=np.float32)
        )
        self._block_2_position = options.get(
            "block_2_pos", np.array([0.5, 0.0], dtype=np.float32)
        )
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
            self._robot_position[0],
            self._robot_position[1],
            self._robot_width,
            self._robot_height,
            0.0,
        )
        robot_rect.plot(ax, facecolor="silver", edgecolor="black")

        # Draw the blocks.
        for block_position in [self._block_1_position, self._block_2_position]:
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
