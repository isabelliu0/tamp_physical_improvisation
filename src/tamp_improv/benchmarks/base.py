"""Base environment interface."""

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

import gymnasium as gym
from relational_structs import LiftedOperator, PDDLDomain, Predicate, Type
from task_then_motion_planning.structs import Perceiver, Skill

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


class BaseEnvironment(Generic[ObsType, ActType], ABC):
    """Base class for environments."""

    def __init__(self, seed: int | None = None) -> None:
        """Initialize environment.

        Args:
            seed: Random seed
        """
        self.env = self._create_env()
        self.wrapped_env = self._create_wrapped_env()
        self.types = self._create_types()
        self.predicates = self._create_predicates()
        self.operators = self._create_operators()
        self.perceiver = self._create_perceiver()
        self.skills = self._create_skills()

        if seed is not None:
            self.env.reset(seed=seed)
            self.wrapped_env.reset(seed=seed)

    @abstractmethod
    def _create_env(self) -> gym.Env:
        """Create the base environment."""

    @abstractmethod
    def _create_wrapped_env(self) -> gym.Env:
        """Create the wrapped environment for training."""

    @abstractmethod
    def _create_types(self) -> set[Type]:
        """Create PDDL types."""

    @abstractmethod
    def _create_predicates(self) -> set[Predicate]:
        """Create PDDL predicates."""

    @abstractmethod
    def _create_operators(self) -> set[LiftedOperator]:
        """Create PDDL operators."""

    @abstractmethod
    def _create_perceiver(self) -> Perceiver[ObsType]:
        """Create state perceiver."""

    @abstractmethod
    def _create_skills(self) -> set[Skill]:
        """Create skills for operators."""

    def get_domain(self) -> PDDLDomain:
        """Get PDDL domain."""
        return PDDLDomain(
            self._get_domain_name(), self.operators, self.predicates, self.types
        )

    @abstractmethod
    def _get_domain_name(self) -> str:
        """Get domain name."""

    def reset(self, seed: int | None = None) -> tuple[ObsType, dict[str, Any]]:
        """Reset environment."""
        return self.env.reset(seed=seed)
