"""Blocks2D environment implementation."""

from dataclasses import dataclass
from typing import Any, Sequence, Union, cast

import gymnasium as gym
import numpy as np
from gymnasium.core import RenderFrame
from gymnasium.spaces import Box
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from relational_structs import (
    GroundAtom,
    LiftedAtom,
    LiftedOperator,
    Object,
    Predicate,
    Type,
    Variable,
)
from task_then_motion_planning.structs import LiftedOperatorSkill, Perceiver, Skill
from tomsgeoms2d.structs import Rectangle
from tomsutils.utils import fig2data

from tamp_improv.benchmarks.base import BaseEnvironment, PlanningComponents


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


def is_target_area_blocked(
    block_x: float,
    block_width: float,
    target_x: float,
    target_width: float,
) -> bool:
    """Check if block 2 blocks the target area -- impediment.

    Block 2 is considered blocking if it overlaps with the target area
    enough that another block cannot fit in the remaining space.
    """
    target_left = target_x - target_width / 2
    target_right = target_x + target_width / 2
    block_left = block_x - block_width / 2
    block_right = block_x + block_width / 2

    # If no horizontal overlap, not blocking
    if block_right <= target_left or block_left >= target_right:
        return False

    # Calculate remaining free width
    overlap_width = min(block_right, target_right) - max(block_left, target_left)
    free_width = target_width - overlap_width

    # Block needs at least its width to fit
    return free_width < block_width


@dataclass
class Blocks2DPredicates:
    """Container for blocks2d predicates."""

    block_in_target_area: Predicate
    block_not_in_target_area: Predicate
    holding: Predicate
    gripper_empty: Predicate
    target_area_clear: Predicate | None = None
    target_area_blocked: Predicate | None = None

    def __getitem__(self, key: str) -> Predicate:
        """Get predicate by name."""
        if key == "BlockInTargetArea":
            return self.block_in_target_area
        if key == "BlockNotInTargetArea":
            return self.block_not_in_target_area
        if key == "Holding":
            return self.holding
        if key == "GripperEmpty":
            return self.gripper_empty
        if key == "TargetAreaClear" and self.target_area_clear is not None:
            return self.target_area_clear
        if key == "TargetAreaBlocked" and self.target_area_blocked is not None:
            return self.target_area_blocked
        raise KeyError(f"Unknown predicate: {key}")

    def as_set(self) -> set[Predicate]:
        """Convert to set of predicates."""
        predicates = {
            self.block_in_target_area,
            self.block_not_in_target_area,
            self.holding,
            self.gripper_empty,
        }
        if self.target_area_clear is not None:
            predicates.add(self.target_area_clear)
        if self.target_area_blocked is not None:
            predicates.add(self.target_area_blocked)
        return predicates


class BaseBlocks2DSkill(LiftedOperatorSkill[NDArray[np.float32], NDArray[np.float32]]):
    """Base class for blocks2d environment skills."""

    def __init__(self, env: "Blocks2DEnvironment") -> None:
        """Initialize skill."""
        super().__init__()
        self._env = env


class ClearTargetAreaSkill(BaseBlocks2DSkill):
    """Skill for clearing target area."""

    def _get_lifted_operator(self) -> LiftedOperator:
        return next(op for op in self._env.operators if op.name == "ClearTargetArea")

    def _get_action_given_objects(
        self,
        objects: Sequence[Object],
        obs: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        robot_x, robot_y, robot_width = obs[0:3]
        block_1_x = obs[4]
        block_2_x, block_2_y = obs[6:8]
        block_width = obs[8]
        target_x = obs[11]
        target_width = obs[13]

        # Determine push direction
        space_on_left = (
            abs((target_x - target_width / 2) - (block_1_x + block_width / 2))
            if block_1_x < target_x
            else abs(target_x - target_width / 2)
        )
        space_on_right = (
            abs((block_1_x - block_width / 2) - (target_x + target_width / 2))
            if block_1_x > target_x
            else abs(1.0 - (target_x + target_width / 2))
        )

        # Push in direction with more space
        push_direction = -0.1 if space_on_left > space_on_right else 0.1

        # Calculate target position for pushing
        target_x_offset = (robot_width + block_width) / 2  # robot_width + block_width
        target_x_offset *= -np.sign(push_direction)
        target_robot_x = block_2_x + target_x_offset

        # Calculate distance to pushing position
        dist_to_target = np.hypot(target_robot_x - robot_x, block_2_y - robot_y)

        if dist_to_target > 0.1:
            # Move to pushing position
            dx = np.clip(target_robot_x - robot_x, -0.1, 0.1)
            dy = np.clip(block_2_y - robot_y, -0.1, 0.1)
            return np.array([dx, dy, 1.0])

        # Push
        return np.array([push_direction, 0.0, 1.0])


class PickUpSkill(BaseBlocks2DSkill):
    """Skill for picking up blocks."""

    def _get_lifted_operator(self) -> LiftedOperator:
        return next(op for op in self._env.operators if op.name == "PickUp")

    def _get_action_given_objects(
        self,
        objects: Sequence[Object],
        obs: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        robot_x, robot_y = obs[0:2]
        robot_height = obs[3]
        block_x, block_y = obs[4:6]
        block_height = obs[9]
        gripper_status = obs[10]

        # Target position above block
        target_y = block_y + block_height / 2 + robot_height / 2

        # Calculate distance to block
        dist_to_block = np.hypot(block_x - robot_x, target_y - robot_y)

        if dist_to_block > 0.15:
            # If we're far from the block, move towards it using combined motion
            dx = np.clip(block_x - robot_x, -0.1, 0.1)
            dy = np.clip(target_y - robot_y, -0.1, 0.1)
            return np.array([dx, dy, 0.0])

        if not np.isclose(robot_y, target_y, atol=1e-3):
            # Fine positioning: align vertically first
            dy = np.clip(target_y - robot_y, -0.1, 0.1)
            return np.array([0.0, dy, 0.0])

        if not np.isclose(robot_x, block_x, atol=1e-3):
            # Then align horizontally
            dx = np.clip(block_x - robot_x, -0.1, 0.1)
            return np.array([dx, 0.0, 0.0])

        # If aligned and gripper is open, close it
        if gripper_status <= 0.0:
            return np.array([0.0, 0.0, 1.0])

        return np.array([0.0, 0.0, 0.0])


class PutDownSkill(BaseBlocks2DSkill):
    """Skill for putting down blocks."""

    def _get_lifted_operator(self) -> LiftedOperator:
        return next(op for op in self._env.operators if op.name == "PutDown")

    def _get_action_given_objects(
        self,
        objects: Sequence[Object],
        obs: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        robot_x, robot_y = obs[0:2]
        robot_height = obs[3]
        block_height = obs[9]
        gripper_status = obs[10]
        target_x, target_y = obs[11:13]

        # Target position above target area
        target_y = target_y + block_height / 2 + robot_height / 2

        # Calculate distance to target
        dist_to_target = np.hypot(target_x - robot_x, target_y - robot_y)

        if dist_to_target > 0.15:
            # If we're far from the target, use combined motion
            dx = np.clip(target_x - robot_x, -0.1, 0.1)
            dy = np.clip(target_y - robot_y, -0.1, 0.1)
            return np.array([dx, dy, gripper_status])

        if not np.isclose(robot_x, target_x, atol=1e-3):
            # Fine positioning: align horizontally first
            dx = np.clip(target_x - robot_x, -0.1, 0.1)
            return np.array([dx, 0.0, gripper_status])

        if robot_y - target_y > 0.0:
            # Then align vertically
            dy = np.clip(target_y - robot_y, -0.1, 0.1)
            return np.array([0.0, dy, gripper_status])

        # If aligned and holding block, release it
        if gripper_status > 0.0:
            return np.array([0.0, 0.0, -1.0])

        return np.array([0.0, 0.0, 0.0])


class BaseBlocks2DPerceiver(Perceiver[NDArray[np.float32]]):
    """Base class for Blocks2D environment perceiver."""

    def __init__(
        self, robot_type: Type, block_type: Type, include_pushing_models: bool
    ) -> None:
        """Initialize with required types."""
        self._robot = Object("robot", robot_type)
        self._block_1 = Object("block1", block_type)
        self._block_2 = Object("block2", block_type)
        self._include_pushing_models = include_pushing_models
        self._predicates: Blocks2DPredicates | None = None

    def initialize(self, predicates: Blocks2DPredicates) -> None:
        """Initialize predicates after environment creation."""
        self._predicates = predicates

    @property
    def predicates(self) -> Blocks2DPredicates:
        """Get predicates, ensuring they're initialized."""
        if self._predicates is None:
            raise RuntimeError("Predicates not initialized. Call initialize() first.")
        return self._predicates


class Blocks2DPerceiver(BaseBlocks2DPerceiver):
    """Perceiver for blocks2d environment."""

    def reset(
        self,
        obs: NDArray[np.float32],
        _info: dict[str, Any],
    ) -> tuple[set[Object], set[GroundAtom], set[GroundAtom]]:
        """Reset perceiver with observation and info."""
        objects = {self._robot, self._block_1, self._block_2}
        atoms = self._get_atoms(obs)
        goal = {
            self.predicates["BlockInTargetArea"]([self._block_1]),
            self.predicates["GripperEmpty"]([self._robot]),
        }
        return objects, atoms, goal

    def step(self, obs: NDArray[np.float32]) -> set[GroundAtom]:
        """Step perceiver with observation."""
        return self._get_atoms(obs)

    def _get_atoms(self, obs: NDArray[np.float32]) -> set[GroundAtom]:
        atoms = set()

        # Get positions from observation
        robot_x, robot_y = obs[0:2]
        block_1_x, block_1_y, block_2_x = obs[4:7]
        block_width, block_height = obs[8:10]
        gripper_status = obs[10]
        target_x, target_y, target_width, target_height = obs[11:15]

        # Check block 1 target area status
        if is_block_1_in_target_area(
            block_1_x,
            block_1_y,
            block_width,
            block_height,
            target_x,
            target_y,
            target_width,
            target_height,
        ):
            atoms.add(self.predicates["BlockInTargetArea"]([self._block_1]))
        else:
            atoms.add(self.predicates["BlockNotInTargetArea"]([self._block_1]))

        # Check gripper status
        if (
            gripper_status > 0.0
            and np.isclose(block_1_x, robot_x, atol=1e-3)
            and np.isclose(block_1_y, robot_y, atol=1e-3)
        ):
            atoms.add(self.predicates["Holding"]([self._robot, self._block_1]))
        else:
            atoms.add(self.predicates["GripperEmpty"]([self._robot]))

        # Add pushing-related predicates only if pushing models included
        if self._include_pushing_models:
            if is_target_area_blocked(block_2_x, block_width, target_x, target_width):
                atoms.add(GroundAtom(self.predicates["TargetAreaBlocked"], []))
            else:
                atoms.add(GroundAtom(self.predicates["TargetAreaClear"], []))

        print("CURRENT ATOMS:", atoms)

        return atoms


class Blocks2DEnvironment(BaseEnvironment[NDArray[np.float32], NDArray[np.float32]]):
    """2D blocks environment."""

    def __init__(
        self,
        seed: int | None = None,
        include_pushing_models: bool = True,
    ) -> None:
        """Initialize environment.

        Args:
            seed: Random seed
            include_pushing_models: Whether to include pushing-related
            predicates/operators
        """
        self._robot_type = Type("robot")
        self._block_type = Type("block")
        self._perceiver = Blocks2DPerceiver(
            self._robot_type, self._block_type, include_pushing_models
        )

        super().__init__(
            seed=seed,
            include_pushing_models=include_pushing_models,
        )

    def _create_env(self) -> gym.Env:
        """Create base environment."""

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
                self.observation_space = Box(
                    low=0, high=1, shape=(15,), dtype=np.float32
                )
                self.action_space = Box(
                    low=np.array([-0.1, -0.1, -1.0]),
                    high=np.array([0.1, 0.1, 1.0]),
                    dtype=np.float32,
                )

                assert (
                    render_mode is None or render_mode in self.metadata["render_modes"]
                )
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
                )
                self.block_1_position = options.get(
                    "block_1_pos", np.array([0.0, 0.0], dtype=np.float32)
                )
                self.block_2_position = options.get(
                    "block_2_pos", np.array([0.5, 0.0], dtype=np.float32)
                )
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
                self.robot_position[0] = np.clip(
                    self.robot_position[0] + dx, 0.0, 1.0
                ).astype(np.float32)
                self.robot_position[1] = np.clip(
                    self.robot_position[1] + dy, 0.0, 1.0
                ).astype(np.float32)
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
                    and np.allclose(
                        self.block_1_position, robot_position_prev, atol=1e-3
                    )
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

        return Blocks2DEnv(render_mode="rgb_array")

    def _create_wrapped_env(self) -> gym.Env:
        """Create wrapped environment for training."""

        class PushingEnvWrapper(gym.Env):
            """Environment wrapper for learning the pushing policy while
            maintaining operator preconditions in Blocks2D environment."""

            def __init__(self, base_env: gym.Env) -> None:
                self.env = base_env
                self.observation_space = base_env.observation_space
                self.action_space = base_env.action_space
                self.max_episode_steps = 100
                self.steps = 0
                self.prev_distance_to_block2 = None

                # Precondition tracking
                self.current_operator: LiftedOperator | None = None
                self.preconditions_to_maintain: set[GroundAtom] = set()

                # Reset options
                self._custom_reset_options = {
                    "robot_pos": np.array([0.5, 1.0], dtype=np.float32),
                    "ensure_blocking": True,
                }

                self.render_mode = self.env.render_mode

            def update_preconditions(
                self, operator: LiftedOperator, preconditions: set[GroundAtom]
            ) -> None:
                """Update the preconditions that should be maintained."""
                self.current_operator = operator
                self.preconditions_to_maintain = preconditions

            def _check_preconditions(self, obs: NDArray[np.float32]) -> bool:
                """Check if all maintained preconditions are still
                satisfied."""
                if not self.preconditions_to_maintain:
                    return True
                current_atoms = cast(
                    Blocks2DEnvironment, self.env
                ).components.perceiver.step(obs)
                return self.preconditions_to_maintain.issubset(current_atoms)

            def _calculate_precondition_violation_penalty(
                self, obs: NDArray[np.float32]
            ) -> float:
                """Calculate penalty for violating operator preconditions."""
                if not self._check_preconditions(obs):
                    return -1.0
                return 0.0

            def reset(
                self,
                *,
                seed: int | None = None,
                options: dict[str, Any] | None = None,
            ) -> tuple[NDArray[np.float32], dict[str, Any]]:
                self.steps = 0
                self.prev_distance_to_block2 = None

                # Initialize RNG
                if seed is not None:
                    rng = np.random.default_rng(seed)
                else:
                    rng = np.random.default_rng()

                # Get initial observation to access dimensions
                obs, _ = self.env.reset(seed=seed)

                # Generate random block 2 position that blocks target
                block_2_x = rng.uniform(
                    obs[11] - obs[13] / 2,  # target_x - target_width/2
                    obs[11] + obs[13] / 2,  # target_x + target_width/2
                )

                reset_options = {
                    "robot_pos": self._custom_reset_options["robot_pos"],
                    "block_1_pos": self._custom_reset_options[
                        "robot_pos"
                    ],  # Block 1 starts at robot
                    "block_2_pos": np.array([block_2_x, 0.0], dtype=np.float32),
                }

                if options:
                    reset_options.update(options)

                obs, info = self.env.reset(options=reset_options)

                # Initialize distance tracking
                robot_pos = obs[0:2]
                block2_pos = obs[6:8]
                robot_width = obs[2]
                block_width = obs[8]
                self.prev_distance_to_block2 = (
                    np.linalg.norm(robot_pos - block2_pos)
                    - (robot_width + block_width) / 2
                )

                # Verify block 2 is blocking if required
                if self._custom_reset_options[
                    "ensure_blocking"
                ] and not is_target_area_blocked(
                    block2_pos[0],
                    block_width,
                    obs[11],  # target_x
                    obs[13],  # target_width
                ):
                    return self.reset(seed=seed, options=options)

                return obs, info

            def step(
                self,
                action: NDArray[np.float32],
            ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
                obs, _, _, _, info = self.env.step(action)
                self.steps += 1

                # Check target area blocking and preconditions
                is_blocked = is_target_area_blocked(
                    obs[6],  # block2_x
                    obs[8],  # block_width
                    obs[11],  # target_x
                    obs[13],  # target_width
                )
                precondition_penalty = self._calculate_precondition_violation_penalty(
                    obs
                )

                # Calculate reward components
                distance_reward = self._calculate_distance_reward(obs)
                collision_penalty = -0.1 if self._check_collision(obs) else 0.0

                # Success case: cleared area while maintaining preconditions
                if not is_blocked and precondition_penalty == 0.0:
                    reward = 10.0
                else:
                    reward = (
                        -0.1  # Base step penalty
                        + distance_reward
                        + collision_penalty
                        + precondition_penalty
                    )

                terminated = not is_blocked  # Terminate when area is cleared
                truncated = self.steps >= self.max_episode_steps

                return obs, reward, terminated, truncated, info

            def _calculate_distance_reward(self, obs: NDArray[np.float32]) -> float:
                """Calculate reward based on distance to block 2."""
                robot_pos = obs[0:2]
                block2_pos = obs[6:8]
                robot_width = obs[2]
                block_width = obs[8]

                current_distance = (
                    np.linalg.norm(robot_pos - block2_pos)
                    - (robot_width + block_width) / 2
                )

                # Reward for moving closer
                distance_delta = self.prev_distance_to_block2 - current_distance
                reward = 0.5 * distance_delta

                # Bonus for being close
                if np.isclose(current_distance, 0.0, atol=2e-2):
                    reward += 0.5

                self.prev_distance_to_block2 = current_distance
                return reward

            def _check_collision(self, obs: NDArray[np.float32]) -> bool:
                """Check for collisions between robot and block 2."""
                robot_pos = obs[0:2]
                block2_pos = obs[6:8]
                robot_width, robot_height = obs[2:4]
                block_width, block_height = obs[8:10]

                dx = abs(robot_pos[0] - block2_pos[0])
                dy = abs(robot_pos[1] - block2_pos[1])

                width_sum = (robot_width + block_width) / 2 - 1e-3
                height_sum = (robot_height + block_height) / 2 - 1e-3

                return dx < width_sum and dy < height_sum

            def render(self) -> Union[RenderFrame, list[RenderFrame], None]:
                """Render the environment."""
                rendered: Union[RenderFrame, list[RenderFrame], None] = (
                    self.env.render()
                )
                return rendered

        return PushingEnvWrapper(self.env)

    def _create_planning_components(
        self,
        include_pushing_models: bool = True,
        **_kwargs: Any,
    ) -> PlanningComponents[NDArray[np.float32]]:
        """Create all planning components."""
        # Create types
        types = {self._robot_type, self._block_type}

        # Create predicates
        predicates = Blocks2DPredicates(
            block_in_target_area=Predicate("BlockInTargetArea", [self._block_type]),
            block_not_in_target_area=Predicate(
                "BlockNotInTargetArea", [self._block_type]
            ),
            holding=Predicate("Holding", [self._robot_type, self._block_type]),
            gripper_empty=Predicate("GripperEmpty", [self._robot_type]),
            target_area_clear=(
                Predicate("TargetAreaClear", []) if include_pushing_models else None
            ),
            target_area_blocked=(
                Predicate("TargetAreaBlocked", []) if include_pushing_models else None
            ),
        )

        # Initialize perceiver with predicates
        self._perceiver.initialize(predicates)

        # Create variables for operators
        robot = Variable("?robot", self._robot_type)
        block = Variable("?block", self._block_type)

        # Create operators
        operators = {
            LiftedOperator(
                "PickUp",
                [robot, block],
                preconditions={
                    predicates["GripperEmpty"]([robot]),
                    predicates["BlockNotInTargetArea"]([block]),
                },
                add_effects={predicates["Holding"]([robot, block])},
                delete_effects={predicates["GripperEmpty"]([robot])},
            )
        }

        putdown_preconditions = {predicates["Holding"]([robot, block])}
        if include_pushing_models:
            putdown_preconditions.add(LiftedAtom(predicates["TargetAreaClear"], []))

        operators.add(
            LiftedOperator(
                "PutDown",
                [robot, block],
                preconditions=putdown_preconditions,
                add_effects={
                    predicates["BlockInTargetArea"]([block]),
                    predicates["GripperEmpty"]([robot]),
                },
                delete_effects={predicates["Holding"]([robot, block])},
            )
        )

        # Add pushing operator only if pushing models included
        if include_pushing_models:
            operators.add(
                LiftedOperator(
                    "ClearTargetArea",
                    [robot, block],
                    preconditions={
                        LiftedAtom(predicates["TargetAreaBlocked"], []),
                        predicates["Holding"]([robot, block]),
                    },
                    add_effects={LiftedAtom(predicates["TargetAreaClear"], [])},
                    delete_effects={LiftedAtom(predicates["TargetAreaBlocked"], [])},
                )
            )

        # Create skills based on operators
        skills: set[Skill[Any, Any]] = {
            cast(Skill[Any, Any], PickUpSkill(self)),
            cast(Skill[Any, Any], PutDownSkill(self)),
        }
        if include_pushing_models:
            skills.add(cast(Skill[Any, Any], ClearTargetAreaSkill(self)))

        return PlanningComponents(
            types=types,
            predicate_container=predicates,
            operators=operators,
            skills=skills,
            perceiver=self._perceiver,
        )

    def _get_domain_name(self) -> str:
        """Get domain name."""
        return "blocks2d-domain"
