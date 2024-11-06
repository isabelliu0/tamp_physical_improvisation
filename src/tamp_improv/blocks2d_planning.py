"""Planning components for the Blocks2D environment."""

from typing import Any, Sequence, Set, Tuple

import numpy as np
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

from tamp_improv.benchmarks.blocks2d_env import Blocks2DEnv, is_block_in_target_area


def create_blocks2d_planning_models(
    include_pushing_models: bool = True,
) -> tuple[set[Type], set[Predicate], Perceiver, set[LiftedOperator], set[Skill]]:
    """Create types, predicates, perceiver, operators, and skills for the
    Blocks2D environment."""

    # Create types and predicates
    robot_type = Type("robot")
    block_type = Type("block")
    types = {robot_type, block_type}

    BlockInTargetArea = Predicate("BlockInTargetArea", [block_type])
    BlockNotInTargetArea = Predicate("BlockNotInTargetArea", [block_type])
    Holding = Predicate("Holding", [robot_type, block_type])
    GripperEmpty = Predicate("GripperEmpty", [robot_type])
    TargetAreaClear = Predicate("TargetAreaClear", [])
    TargetAreaBlocked = Predicate("TargetAreaBlocked", [])
    predicates = {
        BlockInTargetArea,
        BlockNotInTargetArea,
        Holding,
        GripperEmpty,
    }
    if include_pushing_models:
        predicates |= {TargetAreaClear, TargetAreaBlocked}

    # Create operators
    robot = Variable("?robot", robot_type)
    block = Variable("?block", block_type)

    # Only include this operator if include_pushing_models is True.
    ClearTargetAreaOperator = LiftedOperator(
        "ClearTargetArea",
        [robot, block],
        preconditions={LiftedAtom(TargetAreaBlocked, []), Holding([robot, block])},
        add_effects={LiftedAtom(TargetAreaClear, [])},
        delete_effects={LiftedAtom(TargetAreaBlocked, [])},
    )

    pick_up_operator_preconditions = {
        GripperEmpty([robot]),
        LiftedAtom(BlockNotInTargetArea, [block]),
    }
    # if include_pushing_models:
    #     pick_up_operator_preconditions.add(LiftedAtom(TargetAreaClear, []))

    PickUpOperator = LiftedOperator(
        "PickUp",
        [robot, block],
        preconditions=pick_up_operator_preconditions,
        add_effects={Holding([robot, block])},
        delete_effects={GripperEmpty([robot])},
    )

    put_down_operator_preconditions = {
        Holding([robot, block]),
    }
    if include_pushing_models:
        put_down_operator_preconditions.add(LiftedAtom(TargetAreaClear, []))

    PutDownOperator = LiftedOperator(
        "PutDown",
        [robot, block],
        preconditions=put_down_operator_preconditions,
        add_effects={
            LiftedAtom(BlockInTargetArea, [block]),
            LiftedAtom(GripperEmpty, [robot]),
        },
        delete_effects={Holding([robot, block])},
    )

    operators = {PickUpOperator, PutDownOperator}
    if include_pushing_models:
        operators.add(ClearTargetAreaOperator)

    # Create perceiver
    class Blocks2DPerceiver(Perceiver[np.ndarray]):
        """Perceiver for the Blocks2D env."""

        def __init__(self, env: "Blocks2DEnv"):
            self.env = env
            self._robot = robot_type("robot")
            self._block_1 = block_type("block1")
            self._block_2 = block_type("block2")

        def reset(
            self, obs: np.ndarray, info: dict[str, Any]
        ) -> Tuple[Set[Object], Set[GroundAtom], Set[GroundAtom]]:
            objects = {self._robot, self._block_1, self._block_2}
            atoms = self._get_atoms(obs)
            goal: Set[GroundAtom] = {
                BlockInTargetArea([self._block_1]),
                GripperEmpty([self._robot]),
            }
            return objects, atoms, goal  # type: ignore

        def step(self, obs: np.ndarray) -> set[GroundAtom]:
            return self._get_atoms(obs)

        def _get_atoms(self, obs: np.ndarray) -> set[GroundAtom]:
            (
                robot_x,
                robot_y,
                _,
                _,
                block_1_x,
                block_1_y,
                block_2_x,
                block_2_y,
                block_width,
                block_height,
                gripper_status,
                target_x,
                target_y,
                target_width,
                target_height,
            ) = obs

            atoms = set()

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
                atoms.add(BlockInTargetArea([self._block_1]))
            else:
                atoms.add(BlockNotInTargetArea([self._block_1]))

            if (
                gripper_status > 0.0
                and np.isclose(block_1_x, robot_x, atol=1e-3)
                and np.isclose(block_1_y, robot_y, atol=1e-3)
            ):
                atoms.add(Holding([self._robot, self._block_1]))
            else:
                atoms.add(GripperEmpty([self._robot]))

            if include_pushing_models:
                if self._is_target_area_blocked(
                    block_2_x,
                    block_2_y,
                    block_width,
                    block_height,
                    target_x,
                    target_y,
                    target_width,
                    target_height,
                ):
                    atoms.add(GroundAtom(TargetAreaBlocked, []))
                else:
                    atoms.add(GroundAtom(TargetAreaClear, []))

            print("CURRENT ATOMS:", atoms)

            return atoms

        def _is_target_area_blocked(
            self,
            block_x: float,
            block_y: float,
            block_width: float,
            block_height: float,
            target_x: float,
            target_y: float,
            target_width: float,
            target_height: float,
        ) -> bool:
            target_left = target_x - target_width / 2
            target_right = target_x + target_width / 2
            target_top = target_y + target_height / 2
            target_bottom = target_y - target_height / 2

            block_left = block_x - block_width / 2
            block_right = block_x + block_width / 2
            block_top = block_y + block_height / 2
            block_bottom = block_y - block_height / 2

            # Check if there's any overlap
            overlap = (
                block_left < target_right
                and block_right > target_left
                and block_bottom < target_top
                and block_top > target_bottom
            )

            if overlap:
                # Calculate the remaining free width/height in the target area
                free_width = target_width - max(
                    0, min(block_right, target_right) - max(block_left, target_left)
                )
                free_height = target_height - max(
                    0, min(block_top, target_top) - max(block_bottom, target_bottom)
                )
                # If the free width/height is less than the width/height of block_1,
                # it's blocking.
                return (free_width < block_width) or (free_height < block_height)

            return False

    # Update the skills
    class ClearTargetAreaSkill(LiftedOperatorSkill[np.ndarray, np.ndarray]):
        """Skill for clearing the target area by pushing or pulling block 2."""

        def _get_lifted_operator(self) -> LiftedOperator:
            return ClearTargetAreaOperator

        def _get_action_given_objects(
            self, objects: Sequence[Object], obs: np.ndarray
        ) -> np.ndarray:
            (
                robot_x,
                robot_y,
                robot_width,
                _,
                block_1_x,
                _,
                block_2_x,
                block_2_y,
                block_width,
                _,
                _,
                target_x,
                _,
                target_width,
                _,
            ) = obs
            print("CLEAR TARGET AREA SKILL")

            # First, determine in which direction to move block 2
            move_direction = self._get_movement_direction(
                block_1_x,
                target_x,
                block_width,
                target_width,
            )

            # Calculate target position to push from
            target_x_offset = (robot_width + block_width) / 2
            target_x_offset *= -np.sign(move_direction)
            target_robot_x = block_2_x + target_x_offset

            # Calculate distances
            dist_to_target = np.hypot(target_robot_x - robot_x, block_2_y - robot_y)

            if dist_to_target > 0.1:  # If we're far from pushing position
                # Move towards the target position using a combined motion
                dx = np.clip(target_robot_x - robot_x, -0.1, 0.1)
                dy = np.clip(block_2_y - robot_y, -0.1, 0.1)
                return np.array([dx, dy, 1.0])

            # We're in position to push
            return np.array([np.clip(move_direction, -0.1, 0.1), 0.0, 1.0])

        def _get_movement_direction(
            self,
            block_1_x: float,
            target_x: float,
            block_width: float,
            target_width: float,
        ) -> float:
            """Determines the direction to move block 2.

            Returns:
                float: -0.1 for left, 0.1 for right
            """
            # Check if there's enough space on either side
            space_on_left = (
                abs((target_x - target_width / 2) - (block_1_x + block_width / 2))
                if block_1_x < target_x
                else abs(target_x - target_width / 2)
            )
            space_on_right = (
                abs((block_1_x - block_width / 2) - (target_x + target_width / 2))
                if block_1_x > target_x
                else abs(float(1.0) - (target_x + target_width / 2))
            )

            # Move in direction with more space
            if space_on_left > space_on_right:
                return -0.1  # Push left
            return 0.1  # Push right

    class PickUpSkill(LiftedOperatorSkill[np.ndarray, np.ndarray]):
        """Skill for picking up a block."""

        def _get_lifted_operator(self) -> LiftedOperator:
            return PickUpOperator

        def _get_action_given_objects(
            self, objects: Sequence[Object], obs: np.ndarray
        ) -> np.ndarray:
            (
                robot_x,
                robot_y,
                _,
                robot_height,
                block_x,
                block_y,
                _,
                _,
                _,
                block_height,
                gripper_status,
                _,
                _,
                _,
                _,
            ) = obs
            print("PICK UP SKILL")

            # Calculate target position above block
            target_y = block_y + block_height / 2 + robot_height / 2

            # Calculate distances
            dist_to_block = np.hypot(block_x - robot_x, target_y - robot_y)

            # If we're far from the block, move towards it using combined motion
            if dist_to_block > 0.15:
                dx = np.clip(block_x - robot_x, -0.1, 0.1)
                dy = np.clip(target_y - robot_y, -0.1, 0.1)
                return np.array([dx, dy, 0.0])

            # Fine positioning: align vertically first
            if not np.isclose(robot_y, target_y, atol=1e-3):
                dy = np.clip(target_y - robot_y, -0.1, 0.1)
                return np.array([0.0, dy, 0.0])

            # Then align horizontally
            if not np.isclose(robot_x, block_x, atol=1e-3):
                dx = np.clip(block_x - robot_x, -0.1, 0.1)
                return np.array([dx, 0.0, 0.0])

            # If aligned and gripper is open, close it
            if gripper_status <= 0.0:
                return np.array([0.0, 0.0, 1.0])

            return np.array([0.0, 0.0, 0.0])

    class PutDownSkill(LiftedOperatorSkill[np.ndarray, np.ndarray]):
        """Skill for putting down a block."""

        def _get_lifted_operator(self) -> LiftedOperator:
            return PutDownOperator

        def _get_action_given_objects(
            self, objects: Sequence[Object], obs: np.ndarray
        ) -> np.ndarray:
            (
                robot_x,
                robot_y,
                _,
                robot_height,
                _,
                _,
                _,
                _,
                _,
                block_height,
                gripper_status,
                target_x,
                target_y,
                _,
                _,
            ) = obs
            print("PUT DOWN SKILL")

            # Calculate target position
            target_y = target_y + block_height / 2 + robot_height / 2

            # Calculate distance to target
            dist_to_target = np.hypot(target_x - robot_x, target_y - robot_y)

            # If we're far from the target, use combined motion
            if dist_to_target > 0.15:
                dx = np.clip(target_x - robot_x, -0.1, 0.1)
                dy = np.clip(target_y - robot_y, -0.1, 0.1)
                return np.array([dx, dy, gripper_status])

            # Fine positioning: align vertically first
            if not np.isclose(robot_y, target_y, atol=1e-3):
                dy = np.clip(target_y - robot_y, -0.1, 0.1)
                return np.array([0.0, dy, gripper_status])

            # Then align horizontally
            if not np.isclose(robot_x, target_x, atol=1e-3):
                dx = np.clip(target_x - robot_x, -0.1, 0.1)
                return np.array([dx, 0.0, gripper_status])

            # If aligned and holding block, release it
            if gripper_status > 0.0:
                return np.array([0.0, 0.0, -1.0])

            return np.array([0.0, 0.0, 0.0])

    skills = {PickUpSkill(), PutDownSkill()}
    if include_pushing_models:
        skills.add(ClearTargetAreaSkill())

    perceiver = Blocks2DPerceiver(Blocks2DEnv())

    return types, predicates, perceiver, operators, skills  # type: ignore
