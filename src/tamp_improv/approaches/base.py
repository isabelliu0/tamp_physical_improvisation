"""Base approach interface."""

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from relational_structs import GroundAtom, GroundOperator
from task_then_motion_planning.structs import Skill

from tamp_improv.benchmarks.base import BaseEnvironment

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


class BaseApproach(Generic[ObsType, ActType], ABC):
    """Base class for approaches."""

    def __init__(
        self,
        env: BaseEnvironment[ObsType, ActType],
        seed: int,
        planner_id: str = "pyperplan",
    ) -> None:
        """Initialize approach.

        Args:
            env: Environment
            seed: Random seed
            planner_id: PDDL planner ID
        """
        self.env = env
        self._seed = seed
        self._planner_id = planner_id

        # Initialize planning components
        self._domain = env.get_domain()
        self._perceiver = env.perceiver
        self._skills = env.skills

        # Initialize policy
        self._policy = self._create_policy()
        self._policy_active = False
        self._target_atoms: set[GroundAtom] = set()
        self._current_task_plan: list[GroundOperator] = []
        self._current_operator: GroundOperator | None = None
        self._current_skill: Skill | None = None

    @abstractmethod
    def _create_policy(self) -> Any:
        """Create improvisational policy."""

    @abstractmethod
    def reset(self, obs: ObsType, info: dict[str, Any]) -> ActType:
        """Reset approach."""

    @abstractmethod
    def step(
        self,
        obs: ObsType,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ) -> ActType:
        """Step approach."""
