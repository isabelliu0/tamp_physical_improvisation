"""Core blocks2d environment."""

from __future__ import annotations

from typing import Any, NamedTuple

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from tomsgeoms2d.structs import Rectangle
from tomsutils.utils import fig2data


class Blocks2DState(NamedTuple):
    """State of the blocks2d environment."""

    robot_position: NDArray[np.float32]
    block_1_position: NDArray[np.float32]
    block_2_position: NDArray[np.float32]
    gripper_status: float


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
    """Check if block is completely in target area."""
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

        # Set constants
        self._robot_width = 0.2
        self._robot_height = 0.2
        self._block_width = 0.2
        self._block_height = 0.2
        self._target_area = {"x": 0.5, "y": 0.0, "width": 0.2, "height": 0.2}

        # Initialize state
        self.state = self._get_default_state()

    @property
    def robot_position(self) -> NDArray[np.float32]:
        """Get robot position."""
        return self.state.robot_position

    @property
    def block_1_position(self) -> NDArray[np.float32]:
        """Get block 1 position."""
        return self.state.block_1_position

    @property
    def block_2_position(self) -> NDArray[np.float32]:
        """Get block 2 position."""
        return self.state.block_2_position

    @property
    def gripper_status(self) -> float:
        """Get gripper status."""
        return self.state.gripper_status

    def _get_default_state(self) -> Blocks2DState:
        """Get default initial state with randomized block 2 position."""
        # Target area properties are constant
        target_x = self._target_area["x"]
        target_width = self._target_area["width"]

        # Calculate target area bounds
        target_left = target_x - target_width / 2
        target_right = target_x + target_width / 2

        # Calculate valid range for block 2's x position to ensure it blocks target
        min_overlap = 0  # adjustable
        block_2_left_bound = target_left - (self._block_width / 2) + min_overlap
        block_2_right_bound = target_right + (self._block_width / 2) - min_overlap

        # Randomly position block 2 within valid range
        _block_2_x = self.np_random.uniform(block_2_left_bound, block_2_right_bound)

        return Blocks2DState(
            robot_position=np.array([0.5, 1.0], dtype=np.float32),
            block_1_position=np.array([0.0, 0.0], dtype=np.float32),
            block_2_position=np.array(
                [0.5, 0.0], dtype=np.float32
            ),  # fixed block 2 position for testing
            gripper_status=0.0,
        )

    def reset_from_state(
        self,
        state: Blocks2DState | NDArray[np.float32],
        *,
        seed: int | None = None,
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        """Reset environment to specific state."""
        super().reset(seed=seed)

        if isinstance(state, np.ndarray):
            # Convert array to state
            self.state = Blocks2DState(
                robot_position=state[0:2].copy(),
                block_1_position=state[4:6].copy(),
                block_2_position=state[6:8].copy(),
                gripper_status=float(state[10]),
            )
        else:
            self.state = state

        return self._get_obs(), self._get_info()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        """Reset environment to default state."""
        super().reset(seed=seed)

        if options is None:
            self.state = self._get_default_state()
        else:
            self.state = Blocks2DState(
                robot_position=options.get(
                    "robot_pos", np.array([0.5, 1.0], dtype=np.float32)
                ).copy(),
                block_1_position=options.get(
                    "block_1_pos", np.array([0.0, 0.0], dtype=np.float32)
                ).copy(),
                block_2_position=options.get(
                    "block_2_pos", np.array([0.5, 0.0], dtype=np.float32)
                ).copy(),
                gripper_status=0.0,
            )

        return self._get_obs(), self._get_info()

    def _get_obs(self) -> NDArray[np.float32]:
        """Get observation from current state."""
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
        """Get info from current state."""
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
        """Take environment step."""
        dx, dy, gripper_action = action

        # Save previous state
        prev_state = self.state

        # Update robot position
        new_robot_position = np.array(
            [
                np.clip(self.robot_position[0] + dx, 0.0, 1.0),
                np.clip(self.robot_position[1] + dy, 0.0, 1.0),
            ],
            dtype=np.float32,
        )

        # Update block positions and handle interactions
        new_block_1_position = self.block_1_position.copy()
        new_block_2_position = self.block_2_position.copy()
        new_gripper_status = float(gripper_action)

        # Handle block 2 pushing
        if self._is_adjacent(self.robot_position, self.block_2_position):
            relative_pos = self.robot_position[0] - self.block_2_position[0]
            if relative_pos * dx < 0.0:  # Push
                new_block_2_position[0] = np.clip(
                    self.block_2_position[0] + dx, 0.0, 1.0
                ).astype(np.float32)

        # Check if already holding a block
        block1_held = np.allclose(self.block_1_position, self.robot_position, atol=1e-3)
        block2_held = np.allclose(self.block_2_position, self.robot_position, atol=1e-3)
        block_held = block1_held or block2_held

        # Handle picking up and placing down a block
        picking_up_block1 = False
        picking_up_block2 = False
        if self.gripper_status > 0.0 and block_held:  # Already holding a block
            if new_gripper_status <= 0.0:  # Releasing grip
                if block1_held:
                    new_block_1_position = np.array(
                        [self.block_1_position[0], 0.0], dtype=np.float32
                    )
                elif block2_held:
                    new_block_2_position = np.array(
                        [self.block_2_position[0], 0.0], dtype=np.float32
                    )
            else:  # Continuing to hold
                if block1_held:
                    new_block_1_position = new_robot_position.copy()
                elif block2_held:
                    new_block_2_position = new_robot_position.copy()
        elif new_gripper_status > 0.0:  # Attempting to pick up
            # Calculate distances to blocks
            dist_to_block1 = np.linalg.norm(new_robot_position - self.block_1_position)
            dist_to_block2 = np.linalg.norm(new_robot_position - self.block_2_position)
            # Pick up a block that's close enough
            threshold = ((self._robot_width + self._block_width) / 2) + 1e-3
            if dist_to_block1 <= threshold:
                new_block_1_position = new_robot_position.copy()
                picking_up_block1 = True
            elif dist_to_block2 <= threshold:
                new_block_2_position = new_robot_position.copy()
                picking_up_block2 = True

        # Update state
        self.state = Blocks2DState(
            robot_position=new_robot_position,
            block_1_position=new_block_1_position,
            block_2_position=new_block_2_position,
            gripper_status=new_gripper_status,
        )

        # Check for collisions - revert to previous state if collision
        if self._check_collisions(
            block1_held, block2_held, picking_up_block1, picking_up_block2
        ):
            self.state = prev_state
            obs = self._get_obs()
            info = self._get_info()
            return obs, -0.1, False, False, info

        # Get observation
        obs = self._get_obs()
        info = self._get_info()

        # Check if the robot has reached the goal
        goal_reached = is_block_in_target_area(
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
        block1_held: bool,
        block2_held: bool,
        picking_up_block1: bool,
        picking_up_block2: bool,
    ) -> bool:
        """Check for collisions between objects."""
        if (
            self._check_collision_between(self.robot_position, self.block_1_position)
            and not picking_up_block1
            and not block1_held
        ):
            return True
        if (
            self._check_collision_between(self.robot_position, self.block_2_position)
            and not picking_up_block2
            and not block2_held
        ):
            return True
        if self._check_collision_between(self.block_1_position, self.block_2_position):
            return True
        return False

    def _check_collision_between(
        self,
        pos1: NDArray[np.float32],
        pos2: NDArray[np.float32],
    ) -> bool:
        """Check collision between two positions."""
        dx = abs(pos1[0] - pos2[0])
        dy = abs(pos1[1] - pos2[1])

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
        """Render the environment."""
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

    def clone(self) -> Blocks2DEnv:
        """Clone the environment."""
        clone_env = Blocks2DEnv(self.render_mode)
        clone_env.reset_from_state(self.state)
        return clone_env
