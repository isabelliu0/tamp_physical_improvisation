"""ClearAndPlace environment implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray
from pybullet_blocks.envs.clear_and_place_env import (
    ClearAndPlacePyBulletBlocksEnv,
    ClearAndPlaceSceneDescription,
)
from pybullet_blocks.planning_models.action import get_active_operators_and_skills
from pybullet_blocks.planning_models.perception import (
    PREDICATES,
    TYPES,
    ClearAndPlacePyBulletBlocksPerceiver,
)
from relational_structs import PDDLDomain
from task_then_motion_planning.structs import Skill

from tamp_improv.benchmarks.base import (
    ImprovisationalTAMPSystem,
    PlanningComponents,
)
from tamp_improv.benchmarks.wrappers import ImprovWrapper


@dataclass
class ClearAndPlacePredicates:
    """Container for ClearAndPlace predicates."""

    def __getitem__(self, key: str) -> Any:
        """Get predicate by name."""
        return next(p for p in PREDICATES if p.name == key)

    def as_set(self) -> set:
        """Convert to set of predicates."""
        return set(PREDICATES)


class ClearAndPlaceTAMPSystem(
    ImprovisationalTAMPSystem[NDArray[np.float32], NDArray[np.float32]]
):
    """TAMP system for ClearAndPlace environment."""

    @staticmethod
    def create_default(
        seed: int | None = None,
        include_improvisational_models: bool = False,
        render_mode: str | None = None,
    ) -> ClearAndPlaceTAMPSystem:
        """Factory method for creating system with default components."""
        # Create scene description
        scene_description = ClearAndPlaceSceneDescription(
            num_obstacle_blocks=3,
            stack_blocks=True,
        )

        # Create sim for perceiver
        sim = ClearAndPlacePyBulletBlocksEnv(
            scene_description=scene_description,
            render_mode=render_mode,
            use_gui=False,
        )

        # Get operators and skills from imported planning models
        base_operators, skill_types = get_active_operators_and_skills(
            include_improvisational_models=False
        )
        full_operators, _ = get_active_operators_and_skills(
            include_improvisational_models=True
        )
        pybullet_skills = {s(sim, max_motion_planning_time=0.1) for s in skill_types}
        skills: set[Skill[NDArray[np.float32], NDArray[np.float32]]] = cast(
            set[Skill[NDArray[np.float32], NDArray[np.float32]]], pybullet_skills
        )

        # Create perceiver
        perceiver = ClearAndPlacePyBulletBlocksPerceiver(sim)

        # Create predicates container
        predicates = ClearAndPlacePredicates()

        # Create system
        system = ClearAndPlaceTAMPSystem(
            PlanningComponents(
                types=set(TYPES),
                predicate_container=predicates,
                base_operators=base_operators,
                full_operators=full_operators,
                full_operators_active=include_improvisational_models,
                skills=skills,
                perceiver=perceiver,
            ),
            seed=seed,
            render_mode=render_mode,
        )

        return system

    def _create_env(self) -> gym.Env:
        """Create base environment."""
        scene_description = ClearAndPlaceSceneDescription(
            num_obstacle_blocks=3,
            stack_blocks=True,
        )
        return ClearAndPlacePyBulletBlocksEnv(
            scene_description=scene_description,
            render_mode=self._render_mode,
            use_gui=False,
        )

    def _create_wrapped_env(
        self, components: PlanningComponents[NDArray[np.float32]]
    ) -> gym.Env:
        """Create wrapped environment for training."""
        return ImprovWrapper(
            base_env=self.env,
            perceiver=components.perceiver,
            step_penalty=-0.5,
            precondition_violation_penalty=-0.5,
            achievement_bonus=10.0,
        )

    def _get_domain_name(self) -> str:
        """Get domain name."""
        return "clear-and-place-domain"

    def get_domain(self) -> PDDLDomain:
        """Get PDDL domain."""
        return PDDLDomain(
            self._get_domain_name(),
            self.components.operators,
            self.components.predicate_container.as_set(),
            self.components.types,
        )
