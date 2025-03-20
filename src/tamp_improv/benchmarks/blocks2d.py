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
from tamp_improv.benchmarks.blocks2d_env import (
    Blocks2DEnv,
    check_collisions,
    is_block_in_target_area,
)
from tamp_improv.benchmarks.wrappers import ImprovWrapper


@dataclass
class Blocks2DTypes:
    """Container for blocks2d types."""

    def __init__(self) -> None:
        """Initialize types."""
        self.robot = Type("robot")
        self.block = Type("block")
        self.surface = Type("surface")

    def as_set(self) -> set[Type]:
        """Convert to set of types."""
        return {self.robot, self.block, self.surface}


@dataclass
class Blocks2DPredicates:
    """Container for blocks2d predicates."""

    def __init__(self, types: Blocks2DTypes) -> None:
        """Initialize predicates."""
        self.on = Predicate("On", [types.block, types.surface])
        self.holding = Predicate("Holding", [types.robot, types.block])
        self.gripper_empty = Predicate("GripperEmpty", [types.robot])
        self.clear = Predicate("Clear", [types.surface])
        self.is_target = Predicate("IsTarget", [types.surface])
        self.not_is_target = Predicate("NotIsTarget", [types.surface])
        self.no_collision = Predicate("NoCollision", [])

    def __getitem__(self, key: str) -> Predicate:
        """Get predicate by name."""
        return next(p for p in self.as_set() if p.name == key)

    def as_set(self) -> set[Predicate]:
        """Convert to set of predicates."""
        return {
            self.on,
            self.holding,
            self.gripper_empty,
            self.clear,
            self.is_target,
            self.not_is_target,
            self.no_collision,
        }


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


class PickUpSkill(BaseBlocks2DSkill):
    """Skill for picking up blocks from non-target surfaces."""

    def _get_operator_name(self) -> str:
        return "PickUp"

    def _get_action_given_objects(
        self,
        objects: Sequence[Object],
        obs: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        robot_x, robot_y = obs[0:2]
        robot_height = obs[3]

        # Get which block to pick up and its position
        block_obj = objects[1]
        if block_obj.name == "block1":
            block_x, block_y = obs[4:6]
        else:
            block_x, block_y = obs[6:8]
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


class PickUpFromTargetSkill(PickUpSkill):
    """Skill for picking up blocks from target area."""

    def _get_operator_name(self) -> str:
        return "PickUpFromTarget"


class PutDownSkill(BaseBlocks2DSkill):
    """Skill for putting down blocks on non-target surfaces."""

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

        # Get which surface to place on and its position
        surface_obj = objects[2]
        if surface_obj.name == "target_area":
            target_x, target_y = obs[11:13]
        else:
            # Put on table (right of target area)
            target_x = 0.7
            target_y = 0.0

        # Target position above drop location
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


class PutDownOnTargetSkill(PutDownSkill):
    """Skill for putting down blocks in target area."""

    def _get_operator_name(self) -> str:
        return "PutDownOnTarget"


class Blocks2DPerceiver(Perceiver[NDArray[np.float32]]):
    """Perceiver for blocks2d environment."""

    def __init__(self, types: Blocks2DTypes) -> None:
        """Initialize with required types."""
        self._robot = Object("robot", types.robot)
        self._block_1 = Object("block1", types.block)
        self._block_2 = Object("block2", types.block)
        self._table = Object("table", types.surface)
        self._target_area = Object("target_area", types.surface)
        self._predicates: Blocks2DPredicates | None = None
        self._types = types

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
        objects = {
            self._robot,
            self._block_1,
            self._block_2,
            self._table,
            self._target_area,
        }
        atoms = self._get_atoms(obs)
        goal = {
            self.predicates["On"]([self._block_1, self._target_area]),
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
        robot_width, robot_height = obs[2:4]
        block_1_x, block_1_y = obs[4:6]
        block_2_x, block_2_y = obs[6:8]
        block_width, block_height = obs[8:10]
        gripper_status = obs[10]
        target_x, target_y, target_width, target_height = obs[11:15]

        # Add target identification predicates
        atoms.add(self.predicates["IsTarget"]([self._target_area]))
        atoms.add(self.predicates["NotIsTarget"]([self._table]))

        # Check for collisions
        if not check_collisions(
            np.array([robot_x, robot_y], dtype=np.float32),
            np.array([block_1_x, block_1_y], dtype=np.float32),
            np.array([block_2_x, block_2_y], dtype=np.float32),
            gripper_status,
            robot_width,
            robot_height,
            block_width,
            block_height,
        ):
            atoms.add(GroundAtom(self.predicates["NoCollision"], []))

        # Check gripper status
        block1_held = False
        block2_held = False
        if gripper_status > 0.0:
            if np.isclose(block_1_x, robot_x, atol=1e-3) and np.isclose(
                block_1_y, robot_y, atol=1e-3
            ):
                atoms.add(self.predicates["Holding"]([self._robot, self._block_1]))
                block1_held = True
            elif np.isclose(block_2_x, robot_x, atol=1e-3) and np.isclose(
                block_2_y, robot_y, atol=1e-3
            ):
                atoms.add(self.predicates["Holding"]([self._robot, self._block_2]))
                block2_held = True
            else:
                atoms.add(self.predicates["GripperEmpty"]([self._robot]))
        else:
            atoms.add(self.predicates["GripperEmpty"]([self._robot]))

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
            atoms.add(self.predicates["On"]([self._block_1, self._target_area]))
        elif not block1_held:
            atoms.add(self.predicates["On"]([self._block_1, self._table]))

        # Check block 2 target area status
        if is_block_in_target_area(
            block_2_x,
            block_2_y,
            block_width,
            block_height,
            target_x,
            target_y,
            target_width,
            target_height,
        ):
            atoms.add(self.predicates["On"]([self._block_2, self._target_area]))
        elif not block2_held:
            atoms.add(self.predicates["On"]([self._block_2, self._table]))

        # Check if surface is clear: Target area is blocked by block 2 if overlapped
        is_target_clear = block2_held or not self._is_target_area_blocked(
            block_2_x, block_width, target_x, target_width
        )
        if is_target_clear:
            atoms.add(self.predicates["Clear"]([self._target_area]))

        # Table is always "clear" since we can place things on it
        atoms.add(self.predicates["Clear"]([self._table]))

        return atoms

    def _is_target_area_blocked(
        self,
        block_x: float,
        block_width: float,
        target_x: float,
        target_width: float,
    ) -> bool:
        """Check if block 2 blocks the target area.

        Block 2 is considered blocking if it overlaps with the target
        area enough that another block cannot fit in the remaining
        space.
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
        free_width = target_width - overlap_width + 1e-6

        # Block needs at least its width to fit
        return free_width < block_width


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

    def get_domain(self) -> PDDLDomain:
        """Get domain with or without pushing preconditions.

        Args:
            include_extra_preconditions: If True, include pushing models/preconditions.
                                        If False, use base operators.
        """
        return PDDLDomain(
            self._get_domain_name(),
            self.components.operators,
            self.components.predicate_container.as_set(),
            self.components.types,
        )

    @staticmethod
    def create_default(
        seed: int | None = None,
        render_mode: str | None = None,
    ) -> Blocks2DTAMPSystem:
        """Factory method for creating system with default components."""
        # Create types
        types_container = Blocks2DTypes()
        types_set = types_container.as_set()

        # Create predicates
        predicates = Blocks2DPredicates(types_container)

        # Create perceiver
        perceiver = Blocks2DPerceiver(types_container)
        perceiver.initialize(predicates)

        # Create variables for operators
        robot = Variable("?robot", types_container.robot)
        block = Variable("?block", types_container.block)
        surface = Variable("?surface", types_container.surface)

        # Create operators
        operators = {
            LiftedOperator(
                "PickUp",
                [robot, block, surface],
                preconditions={
                    predicates["GripperEmpty"]([robot]),
                    predicates["On"]([block, surface]),
                    predicates["NotIsTarget"]([surface]),
                    LiftedAtom(predicates["NoCollision"], []),
                },
                add_effects={
                    predicates["Holding"]([robot, block]),
                    predicates["Clear"]([surface]),
                },
                delete_effects={
                    predicates["GripperEmpty"]([robot]),
                    predicates["On"]([block, surface]),
                },
            ),
            LiftedOperator(
                "PickUpFromTarget",
                [robot, block, surface],
                preconditions={
                    predicates["GripperEmpty"]([robot]),
                    predicates["On"]([block, surface]),
                    predicates["IsTarget"]([surface]),
                    LiftedAtom(predicates["NoCollision"], []),
                },
                add_effects={
                    predicates["Holding"]([robot, block]),
                    predicates["Clear"]([surface]),
                },
                delete_effects={
                    predicates["GripperEmpty"]([robot]),
                    predicates["On"]([block, surface]),
                },
            ),
            LiftedOperator(
                "PutDown",
                [robot, block, surface],
                preconditions={
                    predicates["Holding"]([robot, block]),
                    predicates["Clear"]([surface]),
                    predicates["NotIsTarget"]([surface]),
                    LiftedAtom(predicates["NoCollision"], []),
                },
                add_effects={
                    predicates["On"]([block, surface]),
                    predicates["GripperEmpty"]([robot]),
                },
                delete_effects={predicates["Holding"]([robot, block])},
            ),
            LiftedOperator(
                "PutDownOnTarget",
                [robot, block, surface],
                preconditions={
                    predicates["Holding"]([robot, block]),
                    predicates["Clear"]([surface]),
                    predicates["IsTarget"]([surface]),
                    LiftedAtom(predicates["NoCollision"], []),
                },
                add_effects={
                    predicates["On"]([block, surface]),
                    predicates["GripperEmpty"]([robot]),
                },
                delete_effects={
                    predicates["Holding"]([robot, block]),
                    predicates["Clear"]([surface]),
                },
            ),
        }

        # Create system
        system = Blocks2DTAMPSystem(
            PlanningComponents(
                types=types_set,
                predicate_container=predicates,
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
            PickUpFromTargetSkill(system.components),
            PutDownSkill(system.components),
            PutDownOnTargetSkill(system.components),
        }

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
        return ImprovWrapper(
            base_env=self.env,
            perceiver=components.perceiver,
            step_penalty=-0.5,
            achievement_bonus=10.0,
        )
