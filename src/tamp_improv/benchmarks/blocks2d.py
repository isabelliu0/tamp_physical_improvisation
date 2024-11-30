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
    PDDLDomain,
    Predicate,
    Type,
    Variable,
)
from task_then_motion_planning.structs import LiftedOperatorSkill, Perceiver

from tamp_improv.benchmarks.base import (
    BaseTAMPSystem,
    ImprovisationalTAMPSystem,
    PlanningComponents,
)
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
        self._lifted_operator = self._get_lifted_operator()

    def _get_lifted_operator(self) -> LiftedOperator:
        """Get the operator this skill implements."""
        return next(
            op
            for op in self._components.operators
            if op.name == self._get_operator_name()
        )

    def _get_operator_name(self) -> str:
        """Get the name of the operator this skill implements."""
        raise NotImplementedError


class ClearTargetAreaSkill(BaseBlocks2DSkill):
    """Skill for clearing target area."""

    def _get_operator_name(self) -> str:
        return "ClearTargetArea"

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
            return np.array([dx, dy, 0.0])

        # Push
        return np.array([push_direction, 0.0, 0.0])


class PickUpSkill(BaseBlocks2DSkill):
    """Skill for picking up blocks."""

    def _get_operator_name(self) -> str:
        return "PickUp"

    def _get_action_given_objects(
        self,
        objects: Sequence[Object],
        obs: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        robot_x, robot_y = obs[0:2]
        robot_height = obs[3]
        block_x, block_y = obs[4:6]
        block_height = obs[9]

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

        # If aligned and not holding block, pick it up
        return np.array([0.0, 0.0, 1.0])


class PutDownSkill(BaseBlocks2DSkill):
    """Skill for putting down blocks."""

    def _get_operator_name(self) -> str:
        return "PutDown"

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

    def __init__(self, robot_type: Type, block_type: Type) -> None:
        """Initialize with required types."""
        self._robot = Object("robot", robot_type)
        self._block_1 = Object("block1", block_type)
        self._block_2 = Object("block2", block_type)
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

        # Check target area blocking
        if is_target_area_blocked(block_2_x, block_width, target_x, target_width):
            atoms.add(GroundAtom(self.predicates["TargetAreaBlocked"], []))
        else:
            atoms.add(GroundAtom(self.predicates["TargetAreaClear"], []))

        return atoms


class BaseBlocks2DTAMPSystem(BaseTAMPSystem[NDArray[np.float32], NDArray[np.float32]]):
    """Base TAMP system for 2D blocks environment."""

    def __init__(
        self,
        planning_components: PlanningComponents[NDArray[np.float32]],
        seed: int | None = None,
        render_mode: str | None = None,
    ) -> None:
        """Initialize blocks2d TAMP system."""
        super().__init__(planning_components, name="Blocks2DTAMPSystem", seed=seed)
        self._render_mode = render_mode

    def _create_env(self) -> gym.Env:
        """Create base environment."""
        return Blocks2DEnv(render_mode=self._render_mode)

    def _get_domain_name(self) -> str:
        """Get domain name."""
        return "blocks2d-domain"

    def get_domain(self, include_extra_preconditions: bool = True) -> PDDLDomain:
        """Get domain with or without pushing preconditions.

        Args:
            include_extra_preconditions: If True, include pushing models/preconditions.
                                        If False, use base operators.
        """
        # Create planning models with appropriate flags
        if include_extra_preconditions:
            return PDDLDomain(
                self._get_domain_name(),
                self.components.full_operators,
                self.components.predicate_container.as_set(),
                self.components.types,
            )
        return PDDLDomain(
            self._get_domain_name(),
            self.components.base_operators,
            self.components.predicate_container.as_set(),
            self.components.types,
        )

    @staticmethod
    def create_default(
        seed: int | None = None,
        include_pushing_models: bool = False,
        render_mode: str | None = None,
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
            target_area_clear=Predicate("TargetAreaClear", []),
            target_area_blocked=Predicate("TargetAreaBlocked", []),
        )

        # Create perceiver
        perceiver = Blocks2DPerceiver(robot_type, block_type)
        perceiver.initialize(predicates)

        # Create variables for operators
        robot = Variable("?robot", robot_type)
        block = Variable("?block", block_type)

        # Create base operators (without pushing)
        base_operators = {
            LiftedOperator(
                "PickUp",
                [robot, block],
                preconditions={
                    predicates["GripperEmpty"]([robot]),
                    predicates["BlockNotInTargetArea"]([block]),
                },
                add_effects={predicates["Holding"]([robot, block])},
                delete_effects={predicates["GripperEmpty"]([robot])},
            ),
            LiftedOperator(
                "PutDown",
                [robot, block],
                preconditions={predicates["Holding"]([robot, block])},
                add_effects={
                    predicates["BlockInTargetArea"]([block]),
                    predicates["GripperEmpty"]([robot]),
                },
                delete_effects={predicates["Holding"]([robot, block])},
            ),
        }

        # Create full operators (with pushing)
        full_operators = {
            LiftedOperator(
                "PickUp",
                [robot, block],
                preconditions={
                    predicates["GripperEmpty"]([robot]),
                    predicates["BlockNotInTargetArea"]([block]),
                },
                add_effects={predicates["Holding"]([robot, block])},
                delete_effects={predicates["GripperEmpty"]([robot])},
            ),
            LiftedOperator(
                "PutDown",
                [robot, block],
                preconditions={
                    predicates["Holding"]([robot, block]),
                    LiftedAtom(predicates["TargetAreaClear"], []),
                },
                add_effects={
                    predicates["BlockInTargetArea"]([block]),
                    predicates["GripperEmpty"]([robot]),
                },
                delete_effects={predicates["Holding"]([robot, block])},
            ),
            LiftedOperator(
                "ClearTargetArea",
                [robot, block],
                preconditions={
                    LiftedAtom(predicates["TargetAreaBlocked"], []),
                    predicates["Holding"]([robot, block]),
                },
                add_effects={LiftedAtom(predicates["TargetAreaClear"], [])},
                delete_effects={LiftedAtom(predicates["TargetAreaBlocked"], [])},
            ),
        }

        # Select current operators based on flag
        operators = full_operators if include_pushing_models else base_operators

        # Create system
        system = Blocks2DTAMPSystem(
            PlanningComponents(
                types=types,
                predicate_container=predicates,
                base_operators=base_operators,
                full_operators=full_operators,
                operators=operators,
                skills=set(),
                perceiver=perceiver,
            ),
            seed=seed,
            render_mode=render_mode,
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


class Blocks2DTAMPSystem(
    ImprovisationalTAMPSystem[NDArray[np.float32], NDArray[np.float32]],
    BaseBlocks2DTAMPSystem,
):
    """TAMP system for 2D blocks environment with improvisational policy
    learning enabled."""

    def _create_wrapped_env(
        self, components: PlanningComponents[NDArray[np.float32]]
    ) -> gym.Env:
        """Create wrapped environment for training."""
        return Blocks2DEnvWrapper(self.env, perceiver=components.perceiver)
