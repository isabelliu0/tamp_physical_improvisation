"""A block environment in 2D."""

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
        self.observation_space = spaces.Box(low=0, high=1, shape=(15,), dtype=np.float32)
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

    def _calculate_distance_to_block(self, block_position: NDArray[np.float32]) -> float:
        return float(np.linalg.norm(self._robot_position - block_position))

    def _get_info(self) -> dict[str, Any]:
        return {
            "distance_to_block1": self._calculate_distance_to_block(self._block_1_position),
            "distance_to_block2": self._calculate_distance_to_block(self._block_2_position),
        }

    def step(
        self,
        action: NDArray[np.float32],
    ) -> Tuple[NDArray[np.float32], SupportsFloat, bool, bool, dict[str, Any]]:

        # Update the position of the robot.
        dx, dy, new_gripper_status = action

        # Clip the values so that the robot stays in bounds.
        new_robot_x = np.clip(self._robot_position[0] + dx, 0.0, 1.0).astype(np.float32)
        new_robot_y = np.clip(self._robot_position[1] + dy, 0.0, 1.0).astype(np.float32)

        # Push the block if the robot is adjacent to the block.
        if self._is_adjacent(self._robot_position, self._block_2_position):
            self._push_block(new_robot_x)

        # Update the position and the gripper status of the robot.
        self._robot_position = np.array([new_robot_x, new_robot_y], dtype=np.float32)
        self._gripper_status = new_gripper_status.astype(np.float32)

        # Check for collisions with the blocks
        self._check_collisions(self._block_1_position)
        self._check_collisions(self._block_2_position)
        
        # Update block 1's position if the gripper suffices the conditions to move the block.
        distance = self._calculate_distance_to_block(self._block_1_position)
        if self._gripper_status > 0.0 and distance <= (self._robot_width + self._block_width) / 2:
            # Robot fetches and holds the block.
            self._block_position = self._robot_position.copy()
        elif self._gripper_status < 0.0 and np.isclose(distance, 0.0, atol=1e-3):
            # Robot drops the block.
            self._block_position = np.array(
                [self._robot_position[0], 0.0], dtype=np.float32
            )

        # Check if the robot has reached the goal
        goal_reached = self.is_block_in_target_area(self._block_1_position[0], self._block_1_position[1], self._block_width, self._block_height, self._target_area["x"], self._target_area["y"], self._target_area["width"], self._target_area["height"])

        # Calculate reward
        reward = np.float32(1.0) if goal_reached else np.float32(0.0)

        terminated = goal_reached
        truncated = False  # False for now since we are using gym's TimeLimit wrapper

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def _check_collisions(self, block_position: NDArray[np.float32]) -> None:
        distance = self._calculate_distance_to_block(block_position)

        collision_threshold = (self._robot_width + self._block_width) / 2
        collision = distance < (collision_threshold - 1e-5)  # Margin for floating point errors
        if collision:
            reward = np.float32(-1.0)
            observation = self._get_obs()
            info = self._get_info()
            return observation, reward, False, True, info
        
    def _is_adjacent(self, robot_position: NDArray[np.float32], block_position: NDArray[np.float32]) -> bool:
        return (np.abs(robot_position[1] - block_position[1]) <= self._block_height / 2) and np.isclose(np.abs(robot_position[0] - block_position[0]), (self._robot_width + self._block_width) / 2, atol=1e-3)
    
    def _push_block(self, new_robot_x: float):
        push_dx = new_robot_x - self._robot_position[0]
        new_block_x = np.clip(self._block_2_position[0] + push_dx, 0.0, 1.0).astype(np.float32)
        self._block_2_position[0] = new_block_x
    
    def is_block_in_target_area(self, block_x: float, block_y: float, block_width: float, block_height: float, target_x: float, target_y: float, target_width: float, target_height: float) -> bool:
        target_left = target_x - target_width / 2
        target_right = target_x + target_width / 2
        target_bottom = target_y - target_height / 2
        target_top = target_y + target_height / 2
        
        block_left = block_x - block_width / 2
        block_right = block_x + block_width / 2
        block_bottom = block_y - block_height / 2
        block_top = block_y + block_height / 2
        
        return (target_left <= block_left and block_right <= target_right
                and target_bottom <= block_bottom and block_top <= target_top)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> Tuple[NDArray[np.float32], dict[str, Any]]:
        super().reset(seed=seed)

        self._robot_position = np.array([0.5, 1.0], dtype=np.float32)
        self._block_1_position = np.array([0.0, 0.0], dtype=np.float32)
        self._block_2_position = np.array([0.5, 0.0], dtype=np.float32)
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
