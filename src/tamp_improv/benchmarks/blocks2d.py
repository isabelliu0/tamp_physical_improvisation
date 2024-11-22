"""Blocks2D environment implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import gymnasium as gym
import numpy as np
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
from task_then_motion_planning.structs import LiftedOperatorSkill, Perceiver

from tamp_improv.benchmarks.base import BaseSkillLearningSys, PlanningComponents
from tamp_improv.benchmarks.blocks2d_env import Blocks2DEnv, is_block_1_in_target_area
from tamp_improv.benchmarks.blocks2d_wrappers import (
    Blocks2DEnvWrapper,
    is_target_area_blocked,
)


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
        return next(p for p in self.as_set() if p.name == key)

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

    def __init__(self, components: PlanningComponents[NDArray[np.float32]]) -> None:
        """Initialize skill."""
        super().__init__()
        self._components = components


class ClearTargetAreaSkill(BaseBlocks2DSkill):
    """Skill for clearing target area."""

    def _get_lifted_operator(self) -> LiftedOperator:
        return next(
            op for op in self._components.operators if op.name == "ClearTargetArea"
        )

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
        return next(op for op in self._components.operators if op.name == "PickUp")

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
        return next(op for op in self._components.operators if op.name == "PutDown")

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


class Blocks2DPerceiver(Perceiver[NDArray[np.float32]]):
    """Perceiver for blocks2d environment."""

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


class Blocks2DTAMPSystem(
    BaseSkillLearningSys[NDArray[np.float32], NDArray[np.float32]]
):
    """TAMP system for 2D blocks environment."""

    def _create_env(self) -> gym.Env:
        """Create base environment."""
        return Blocks2DEnv()

    def _create_wrapped_env(
        self, components: PlanningComponents[NDArray[np.float32]]
    ) -> gym.Env:
        """Create wrapped environment for training."""
        return Blocks2DEnvWrapper(self.env, perceiver=components.perceiver)

    def _get_domain_name(self) -> str:
        """Get domain name."""
        return "blocks2d-domain"

    @staticmethod
    def create_default(
        seed: int | None = None,
        include_pushing_models: bool = True,
    ) -> Blocks2DTAMPSystem:
        """Factory method for creating system with default components."""
        # Create types
        robot_type = Type("robot")
        block_type = Type("block")
        types = {robot_type, block_type}

        # Create predicates
        predicates = Blocks2DPredicates(
            block_in_target_area=Predicate("BlockInTargetArea", [block_type]),
            block_not_in_target_area=Predicate("BlockNotInTargetArea", [block_type]),
            holding=Predicate("Holding", [robot_type, block_type]),
            gripper_empty=Predicate("GripperEmpty", [robot_type]),
            target_area_clear=(
                Predicate("TargetAreaClear", []) if include_pushing_models else None
            ),
            target_area_blocked=(
                Predicate("TargetAreaBlocked", []) if include_pushing_models else None
            ),
        )

        # Create perceiver
        perceiver = Blocks2DPerceiver(robot_type, block_type, include_pushing_models)
        perceiver.initialize(predicates)

        # Create variables for operators
        robot = Variable("?robot", robot_type)
        block = Variable("?block", block_type)

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

        # Create system
        system = Blocks2DTAMPSystem(
            PlanningComponents(
                types=types,
                predicate_container=predicates,
                operators=operators,
                skills=set(),
                perceiver=perceiver,
            ),
            seed=seed,
        )

        # Create skills with reference to components
        skills = {
            PickUpSkill(system.components),
            PutDownSkill(system.components),
        }
        if include_pushing_models:
            skills.add(ClearTargetAreaSkill(system.components))

        # Update components with skills
        system.components.skills.update(skills)

        return system
