"""Core blocks2d environment."""

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
        """Get default initial state."""
        return Blocks2DState(
            robot_position=np.array([0.5, 1.0], dtype=np.float32),
            block_1_position=np.array([0.0, 0.0], dtype=np.float32),
            block_2_position=np.array([0.5, 0.0], dtype=np.float32),
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

        # Handle block 2 pushing
        if self._is_adjacent(prev_state.robot_position, self.block_2_position):
            relative_pos = prev_state.robot_position[0] - self.block_2_position[0]
            if relative_pos * dx < 0.0:  # Push
                new_block_2_position[0] = np.clip(
                    self.block_2_position[0] + dx, 0.0, 1.0
                ).astype(np.float32)

        # Handle block 1 interactions (pick/drop)
        distance = np.linalg.norm(new_robot_position - self.block_1_position)
        new_gripper_status = float(gripper_action)

        # Case 1: Robot is picking up the block
        if (
            new_gripper_status > 0.0
            and distance <= ((self._robot_width + self._block_width) / 2) + 1e-3
        ):
            new_block_1_position = new_robot_position.copy()

        # Case 2: Robot was holding the block and gripper is deactivated
        elif (
            self.gripper_status > 0.0
            and new_gripper_status <= 0.0
            and np.allclose(self.block_1_position, prev_state.robot_position, atol=1e-3)
        ):
            new_block_1_position = np.array(
                [new_robot_position[0], 0.0], dtype=np.float32
            )

        # Case 3: Robot is holding the block (continue moving it)
        elif new_gripper_status > 0.0 and np.allclose(
            self.block_1_position, prev_state.robot_position, atol=1e-3
        ):
            new_block_1_position = new_robot_position.copy()

        # Update state
        self.state = Blocks2DState(
            robot_position=new_robot_position,
            block_1_position=new_block_1_position,
            block_2_position=new_block_2_position,
            gripper_status=new_gripper_status,
        )

        # Get observation
        obs = self._get_obs()
        info = self._get_info()

        # Check for collisions
        if self._check_collisions():
            return obs, -0.1, False, False, info

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

    def _check_collisions(self) -> bool:
        """Check for collisions between objects."""
        # Robot-Block1 collision with empty gripper
        if self._check_collision_between(
            self.robot_position, self.block_1_position
        ) and np.isclose(self.gripper_status, 0.0, atol=1e-3):
            return True

        # Robot-Block2 collision
        if self._check_collision_between(self.robot_position, self.block_2_position):
            return True

        # Block1-Block2 collision
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
