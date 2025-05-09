"""Graph-based ClearAndPlace environment implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import gymnasium as gym
import numpy as np
from gymnasium.spaces import GraphInstance
from numpy.typing import NDArray
from pybullet_blocks.envs.clear_and_place_env import (
    ClearAndPlaceSceneDescription,
    GraphClearAndPlacePyBulletBlocksEnv,
)
from pybullet_blocks.planning_models.action import OPERATORS, SKILLS
from pybullet_blocks.planning_models.perception import (
    PREDICATES,
    TYPES,
    GraphClearAndPlacePyBulletBlocksPerceiver,
)
from relational_structs import PDDLDomain, Predicate
from task_then_motion_planning.structs import Skill

from tamp_improv.benchmarks.base import (
    BaseTAMPSystem,
    ImprovisationalTAMPSystem,
    PlanningComponents,
)
from tamp_improv.benchmarks.wrappers import ImprovWrapper


@dataclass(frozen=True)
class GraphClearAndPlacePredicates:
    """Container for GraphClearAndPlace predicates."""

    def __getitem__(self, key: str) -> Any:
        """Get predicate by name."""
        return next(p for p in PREDICATES if p.name == key)

    def as_set(self) -> set[Predicate]:
        """Convert to set of predicates."""
        return set(PREDICATES)


class BaseGraphClearAndPlaceTAMPSystem(
    BaseTAMPSystem[GraphInstance, NDArray[np.float32]]
):
    """Base TAMP system for graph-based ClearAndPlace environment."""

    def __init__(
        self,
        planning_components: PlanningComponents[GraphInstance],
        seed: int | None = None,
        render_mode: str | None = None,
        num_obstacle_blocks: int = 3,
    ) -> None:
        """Initialize graph-based ClearAndPlace TAMP system."""
        self._render_mode = render_mode
        self._num_obstacle_blocks = num_obstacle_blocks
        super().__init__(
            planning_components, name="GraphClearAndPlaceTAMPSystem", seed=seed
        )

    def _create_env(self) -> gym.Env:
        """Create base environment."""
        scene_description = ClearAndPlaceSceneDescription(
            num_obstacle_blocks=self._num_obstacle_blocks,
            stack_blocks=True,
        )
        return GraphClearAndPlacePyBulletBlocksEnv(
            scene_description=scene_description,
            render_mode=self._render_mode,
            use_gui=False,
        )

    def _get_domain_name(self) -> str:
        """Get domain name."""
        return "graph-clear-and-place-domain"

    def get_domain(self) -> PDDLDomain:
        """Get PDDL domain."""
        return PDDLDomain(
            self._get_domain_name(),
            self.components.operators,
            self.components.predicate_container.as_set(),
            self.components.types,
        )

    @classmethod
    def create_default(
        cls,
        seed: int | None = None,
        render_mode: str | None = None,
        num_obstacle_blocks: int = 3,
    ) -> BaseGraphClearAndPlaceTAMPSystem:
        """Factory method for creating system with default components."""
        scene_description = ClearAndPlaceSceneDescription(
            num_obstacle_blocks=num_obstacle_blocks,
            stack_blocks=True,
        )
        sim = GraphClearAndPlacePyBulletBlocksEnv(
            scene_description=scene_description,
            render_mode=render_mode,
            use_gui=False,
        )
        pybullet_skills = {
            s(sim, max_motion_planning_time=0.1)  # type:ignore[abstract]
            for s in SKILLS
        }
        skills: set[Skill[GraphInstance, NDArray[np.float32]]] = cast(
            set[Skill[GraphInstance, NDArray[np.float32]]], pybullet_skills
        )
        perceiver = GraphClearAndPlacePyBulletBlocksPerceiver(sim)
        predicates = GraphClearAndPlacePredicates()
        system = cls(
            PlanningComponents(
                types=set(TYPES),
                predicate_container=predicates,
                operators=set(OPERATORS),
                skills=skills,
                perceiver=perceiver,
            ),
            seed=seed,
            render_mode=render_mode,
            num_obstacle_blocks=num_obstacle_blocks,
        )
        return system


class GraphClearAndPlaceTAMPSystem(
    ImprovisationalTAMPSystem[GraphInstance, NDArray[np.float32]],
    BaseGraphClearAndPlaceTAMPSystem,
):
    """TAMP system for graph-based ClearAndPlace environment with
    improvisational policy learning enabled."""

    def __init__(
        self,
        planning_components: PlanningComponents[GraphInstance],
        seed: int | None = None,
        render_mode: str | None = None,
        num_obstacle_blocks: int = 3,
    ) -> None:
        """Initialize graph-based ClearAndPlace TAMP system."""
        self._render_mode = render_mode
        self._num_obstacle_blocks = num_obstacle_blocks
        ImprovisationalTAMPSystem.__init__(
            self,
            planning_components,
            seed=seed,
            render_mode=render_mode,
        )
        BaseGraphClearAndPlaceTAMPSystem.__init__(
            self,
            planning_components,
            seed=seed,
            render_mode=render_mode,
            num_obstacle_blocks=num_obstacle_blocks,
        )

    def _create_wrapped_env(
        self, components: PlanningComponents[GraphInstance]
    ) -> gym.Env:
        """Create wrapped environment for training."""
        return ImprovWrapper(
            base_env=self.env,
            perceiver=components.perceiver,
            step_penalty=-1.0,
            achievement_bonus=100.0,
            action_scale=0.015,
        )

    @classmethod
    def create_default(
        cls,
        seed: int | None = None,
        render_mode: str | None = None,
        num_obstacle_blocks: int = 3,
    ) -> GraphClearAndPlaceTAMPSystem:
        """Factory method for creating system with default components."""
        scene_description = ClearAndPlaceSceneDescription(
            num_obstacle_blocks=num_obstacle_blocks,
            stack_blocks=True,
        )
        sim = GraphClearAndPlacePyBulletBlocksEnv(
            scene_description=scene_description,
            render_mode=render_mode,
            use_gui=False,
        )
        pybullet_skills = {
            s(sim, max_motion_planning_time=0.1)  # type:ignore[abstract]
            for s in SKILLS
        }
        skills: set[Skill[GraphInstance, NDArray[np.float32]]] = cast(
            set[Skill[GraphInstance, NDArray[np.float32]]], pybullet_skills
        )
        perceiver = GraphClearAndPlacePyBulletBlocksPerceiver(sim)
        predicates = GraphClearAndPlacePredicates()
        system = cls(
            PlanningComponents(
                types=set(TYPES),
                predicate_container=predicates,
                operators=set(OPERATORS),
                skills=skills,
                perceiver=perceiver,
            ),
            seed=seed,
            render_mode=render_mode,
            num_obstacle_blocks=num_obstacle_blocks,
        )
        return system
