"""Complete implementation of Blocks2D environment."""

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box
from numpy.typing import NDArray
from relational_structs import (
    GroundAtom,
    LiftedOperator,
    Object,
    Predicate,
    Type,
    Variable,
)
from task_then_motion_planning.structs import LiftedOperatorSkill, Perceiver, Skill

from tamp_improv.benchmarks.base import BaseEnvironment


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
    """Check if block is in target area."""
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


class Blocks2DEnvironment(BaseEnvironment[NDArray[np.float32], NDArray[np.float32]]):
    """2D blocks environment."""

    def _create_env(self) -> gym.Env:
        """Create base environment."""

        class Blocks2DEnv(gym.Env):
            def __init__(self) -> None:
                self.observation_space = Box(
                    low=0, high=1, shape=(15,), dtype=np.float32
                )
                self.action_space = Box(
                    low=np.array([-0.1, -0.1, -1.0]),
                    high=np.array([0.1, 0.1, 1.0]),
                    dtype=np.float32,
                )

                # set constants
                self._robot_width = 0.2
                self._robot_height = 0.2
                self._block_width = 0.2
                self._block_height = 0.2
                self._target_area = {"x": 0.5, "y": 0.0, "width": 0.2, "height": 0.2}

                self.reset()

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

                # Handle block 1 interactions
                distance = np.linalg.norm(self.robot_position - self.block_1_position)

                # Pick up block
                if (
                    self.gripper_status > 0.0
                    and distance <= ((self._robot_width + self._block_width) / 2) + 1e-3
                ):
                    self.block_1_position = self.robot_position.copy()

                # Drop block
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

                # Move held block
                elif self.gripper_status > 0.0 and np.allclose(
                    self.block_1_position, robot_position_prev, atol=1e-3
                ):
                    self.block_1_position = self.robot_position.copy()

                # Check collisions
                obs = self._get_obs()
                info = self._get_info()

                if self._check_collisions(self.robot_position, self.block_1_position):
                    if np.isclose(self.gripper_status, 0.0, atol=1e-3):
                        return obs, -1.0, False, True, info

                if self._check_collisions(self.robot_position, self.block_2_position):
                    return obs, -1.0, False, True, info

                if self._check_collisions(self.block_1_position, self.block_2_position):
                    return obs, -1.0, False, True, info

                # Check goal
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
                pos1: NDArray[np.float32],
                pos2: NDArray[np.float32],
            ) -> bool:
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
                    atol=2e-2,
                )
                return vertical_aligned and horizontal_adjacent

            # TODO: render()

        return Blocks2DEnv()

    def _create_wrapped_env(self) -> gym.Env:
        """Create wrapped environment for training."""

        class PushingEnvWrapper(gym.Env):
            def __init__(self, base_env: gym.Env) -> None:
                self.env = base_env
                self.observation_space = base_env.observation_space
                self.action_space = base_env.action_space
                self.max_episode_steps = 100
                self.steps = 0
                self.prev_distance_to_block2 = None

            def reset(
                self,
                *,
                seed: int | None = None,
                options: dict[str, Any] | None = None,
            ) -> tuple[NDArray[np.float32], dict[str, Any]]:
                self.steps = 0

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
                    "robot_pos": np.array([0.5, 1.0], dtype=np.float32),
                    "block_1_pos": np.array([0.5, 1.0], dtype=np.float32),
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

                return obs, info

            def step(
                self,
                action: NDArray[np.float32],
            ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
                obs, _, _, _, info = self.env.reset(action)
                self.steps += 1

                # Calculate current distance and reward
                robot_pos = obs[0:2]
                block2_pos = obs[6:8]
                robot_width = obs[2]
                block_width = obs[8]
                current_distance = (
                    np.linalg.norm(robot_pos - block2_pos)
                    - (robot_width + block_width) / 2
                )

                distance_delta = self.prev_distance_to_block2 - current_distance
                reward = 0.5 * distance_delta

                if np.isclose(current_distance, 0.0, atol=2e-2):
                    reward += 0.5

                self.prev_distance_to_block2 = current_distance

                # Check if block 2 still blocks target
                is_blocking = self._is_blocking(obs)
                terminated = not is_blocking
                truncated = self.steps >= self.max_episode_steps

                if terminated and not truncated:
                    reward = 10.0

                return obs, reward, terminated, truncated, info

            def _is_blocking(self, obs: NDArray[np.float32]) -> bool:
                block_2_x = obs[6]
                block_width = obs[8]
                target_x = obs[11]
                target_width = obs[13]

                target_left = target_x - target_width / 2
                target_right = target_x + target_width / 2
                block_left = block_2_x - block_width / 2
                block_right = block_2_x + block_width / 2

                if block_right <= target_left or block_left >= target_right:
                    return False

                overlap_width = min(block_right, target_right) - max(
                    block_left, target_left
                )
                free_width = target_width - overlap_width

                return free_width < block_width

            # TODO: more complicated reward/penalty settings

        return PushingEnvWrapper(self.env)

    def _create_types(self) -> set[Type]:
        """Create PDDL types."""
        return {Type("robot"), Type("block")}

    def _create_predicates(self) -> set[Predicate]:
        """Create PDDL predicates."""
        robot_type = next(t for t in self.types if t.name == "robot")
        block_type = next(t for t in self.types if t.name == "block")
        predicates = {
            Predicate("BlockInTargetArea", [block_type]),
            Predicate("BlockNotInTargetArea", [block_type]),
            Predicate("Holding", [robot_type, block_type]),
            Predicate("GripperEmpty", [robot_type]),
            Predicate("TargetAreaClear", []),
            Predicate("TargetAreaBlocked", []),
        }
        return predicates

    def _create_operators(self) -> set[LiftedOperator]:
        """Create PDDL operators."""
        robot = Variable("?robot", next(t for t in self.types if t.name == "robot"))
        block = Variable("?block", next(t for t in self.types if t.name == "block"))

        operators = set()

        # ClearTargetArea operator
        operators.add(
            LiftedOperator(
                "ClearTargetArea",
                [robot, block],
                {
                    self.predicates["TargetAreaBlocked"]([]),
                    self.predicates["Holding"]([robot, block]),
                },
                {self.predicates["TargetAreaClear"]([])},
                {self.predicates["TargetAreaBlocked"]([])},
            )
        )

        # PickUp operator
        operators.add(
            LiftedOperator(
                "PickUp",
                [robot, block],
                {
                    self.predicates["GripperEmpty"]([robot]),
                    self.predicates["BlockNotInTargetArea"]([block]),
                },
                {self.predicates["Holding"]([robot, block])},
                {self.predicates["GripperEmpty"]([robot])},
            )
        )

        # PutDown operator
        operators.add(
            LiftedOperator(
                "PutDown",
                [robot, block],
                {
                    self.predicates["Holding"]([robot, block]),
                    self.predicates["TargetAreaClear"]([]),
                },
                {
                    self.predicates["BlockInTargetArea"]([block]),
                    self.predicates["GripperEmpty"]([robot]),
                },
                {self.predicates["Holding"]([robot, block])},
            )
        )

        return operators

    def _create_perceiver(self) -> Perceiver[NDArray[np.float32]]:
        """Create state perceiver."""

        class Blocks2DPerceiver(Perceiver[NDArray[np.float32]]):
            def __init__(self, env: Blocks2DEnvironment) -> None:
                self._env = env
                self._robot = Object(
                    next(t for t in env.types if t.name == "robot"), "robot"
                )
                self._block_1 = Object(
                    next(t for t in env.types if t.name == "block"), "block1"
                )
                self._block_2 = Object(
                    next(t for t in env.types if t.name == "block"), "block2"
                )

            def reset(
                self, obs: NDArray[np.float32], info: dict[str, Any]
            ) -> tuple[set[Object], set[GroundAtom], set[GroundAtom]]:
                objects = {self._robot, self._block_1, self._block_2}
                atoms = self._get_atoms(obs)
                goal = {
                    self._env.predicates["BlockInTargetArea"]([self._block_1]),
                    self._env.predicates["GripperEmpty"]([self._robot]),
                }
                return objects, atoms, goal

            def step(self, obs: NDArray[np.float32]) -> set[GroundAtom]:
                return self._get_atoms(obs)

            def _get_atoms(self, obs: NDArray[np.float32]) -> set[GroundAtom]:
                atoms = set()

                # Get positions from observation
                robot_x, robot_y = obs[0:2]
                block_1_x, block_1_y = obs[4:6]
                block_2_x, block_2_y = obs[6:8]
                block_width, block_height = obs[8:10]
                gripper_status = obs[10]
                target_x, target_y, target_width, target_height = obs[11:15]

                # Check block 1 target area status
                if is_block_in_target_area(
                    block_1_x,
                    block_1_y,
                    block_width,
                    block_height,
                    target_x,
                    target_y,
                    target_width,
                    target_height,
                ):
                    atoms.add(
                        self._env.predicates["BlockInTargetArea"]([self._block_1])
                    )
                else:
                    atoms.add(
                        self._env.predicates["BlockNotInTargetArea"]([self._block_1])
                    )

                # Check gripper status
                if (
                    gripper_status > 0.0
                    and np.isclose(block_1_x, robot_x, atol=1e-3)
                    and np.isclose(block_1_y, robot_y, atol=1e-3)
                ):
                    atoms.add(
                        self._env.predicates["Holding"]([self._robot, self._block_1])
                    )
                else:
                    atoms.add(self._env.predicates["GripperEmpty"]([self._robot]))

                # Check if target area is blocked by block 2
                target_left = target_x - target_width / 2
                target_right = target_x + target_width / 2
                target_top = target_y + target_height / 2
                target_bottom = target_y - target_height / 2

                block_left = block_2_x - block_width / 2
                block_right = block_2_x + block_width / 2
                block_top = block_2_y + block_height / 2
                block_bottom = block_2_y - block_height / 2

                overlap = (
                    block_left < target_right
                    and block_right > target_left
                    and block_bottom < target_top
                    and block_top > target_bottom
                )

                if overlap:
                    free_width = target_width - max(
                        0, min(block_right, target_right) - max(block_left, target_left)
                    )
                    free_height = target_height - max(
                        0, min(block_top, target_top) - max(block_bottom, target_bottom)
                    )
                    if free_width < block_width or free_height < block_height:
                        atoms.add(self._env.predicates["TargetAreaBlocked"]([]))
                    else:
                        atoms.add(self._env.predicates["TargetAreaClear"]([]))
                else:
                    atoms.add(self._env.predicates["TargetAreaClear"]([]))

                return atoms

        return Blocks2DPerceiver(self)

    def _create_skills(self) -> set[Skill]:
        """Create skills for operators."""

        class ClearTargetAreaSkill(
            LiftedOperatorSkill[NDArray[np.float32], NDArray[np.float32]]
        ):
            def _get_lifted_operator(self) -> LiftedOperator:
                return next(
                    op for op in self._env.operators if op.name == "ClearTargetArea"
                )

            def _get_action_given_objects(
                self, objects: list[Object], obs: NDArray[np.float32]
            ) -> NDArray[np.float32]:
                robot_x, robot_y = obs[0:2]
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

                # Choose direction with more space
                push_direction = -0.1 if space_on_left > space_on_right else 0.1

                # Calculate target position for pushing
                target_x_offset = (
                    obs[2] + block_width
                ) / 2  # robot_width + block_width
                target_x_offset *= -np.sign(push_direction)
                target_robot_x = block_2_x + target_x_offset

                # Calculate distances
                dist_to_target = np.hypot(target_robot_x - robot_x, block_2_y - robot_y)

                if dist_to_target > 0.1:
                    # Move to pushing position
                    dx = np.clip(target_robot_x - robot_x, -0.1, 0.1)
                    dy = np.clip(block_2_y - robot_y, -0.1, 0.1)
                    return np.array([dx, dy, 1.0])

                # Push
                return np.array([push_direction, 0.0, 1.0])

        class PickUpSkill(
            LiftedOperatorSkill[NDArray[np.float32], NDArray[np.float32]]
        ):
            def _get_lifted_operator(self) -> LiftedOperator:
                return next(op for op in self._env.operators if op.name == "PickUp")

            def _get_action_given_objects(
                self, objects: list[Object], obs: NDArray[np.float32]
            ) -> NDArray[np.float32]:
                robot_x, robot_y = obs[0:2]
                robot_height = obs[3]
                block_x, block_y = obs[4:6]
                block_height = obs[9]
                gripper_status = obs[10]

                # Target position above block
                target_y = block_y + block_height / 2 + robot_height / 2

                # Calculate distances
                dist_to_block = np.hypot(block_x - robot_x, target_y - robot_y)

                if dist_to_block > 0.15:
                    # Move towards block
                    dx = np.clip(block_x - robot_x, -0.1, 0.1)
                    dy = np.clip(target_y - robot_y, -0.1, 0.1)
                    return np.array([dx, dy, 0.0])

                if not np.isclose(robot_y, target_y, atol=1e-3):
                    # Align vertically
                    dy = np.clip(target_y - robot_y, -0.1, 0.1)
                    return np.array([0.0, dy, 0.0])

                if not np.isclose(robot_x, block_x, atol=1e-3):
                    # Align horizontally
                    dx = np.clip(block_x - robot_x, -0.1, 0.1)
                    return np.array([dx, 0.0, 0.0])

                # Close gripper
                if gripper_status <= 0.0:
                    return np.array([0.0, 0.0, 1.0])

                return np.array([0.0, 0.0, 0.0])

        class PutDownSkill(
            LiftedOperatorSkill[NDArray[np.float32], NDArray[np.float32]]
        ):
            def _get_lifted_operator(self) -> LiftedOperator:
                return next(op for op in self._env.operators if op.name == "PutDown")

            def _get_action_given_objects(
                self, objects: list[Object], obs: NDArray[np.float32]
            ) -> NDArray[np.float32]:
                robot_x, robot_y = obs[0:2]
                robot_height = obs[3]
                block_height = obs[9]
                gripper_status = obs[10]
                target_x, target_y = obs[11:13]

                # Target position above target area
                target_y = target_y + block_height / 2 + robot_height / 2

                # Calculate distance
                dist_to_target = np.hypot(target_x - robot_x, target_y - robot_y)

                if dist_to_target > 0.15:
                    # Move towards target
                    dx = np.clip(target_x - robot_x, -0.1, 0.1)
                    dy = np.clip(target_y - robot_y, -0.1, 0.1)
                    return np.array([dx, dy, gripper_status])

                if not np.isclose(robot_x, target_x, atol=1e-3):
                    # Align horizontally
                    dx = np.clip(target_x - robot_x, -0.1, 0.1)
                    return np.array([dx, 0.0, gripper_status])

                if robot_y - target_y > 0.0:
                    # Move down
                    dy = np.clip(target_y - robot_y, -0.1, 0.1)
                    return np.array([0.0, dy, gripper_status])

                # Open gripper
                if gripper_status > 0.0:
                    return np.array([0.0, 0.0, -1.0])

                return np.array([0.0, 0.0, 0.0])

        return {ClearTargetAreaSkill(), PickUpSkill(), PutDownSkill()}

    def _get_domain_name(self) -> str:
        """Get domain name."""
        return "blocks2d-domain"
