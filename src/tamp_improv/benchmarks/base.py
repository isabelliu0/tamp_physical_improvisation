"""Base environment interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeVar

import gymnasium as gym
from relational_structs import LiftedOperator, PDDLDomain, Predicate, Type
from task_then_motion_planning.structs import Perceiver, Skill

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


class PredicateContainer(Protocol):
    """Protocol for predicate containers."""

    def __getitem__(self, key: str) -> Predicate:
        """Get predicate by name."""

    def as_set(self) -> set[Predicate]:
        """Convert to set of predicates."""


@dataclass
class PlanningComponents(Generic[ObsType]):
    """Container for all planning-related components."""

    types: set[Type]
    predicate_container: PredicateContainer
    operators: set[LiftedOperator]
    skills: set[Skill]
    perceiver: Perceiver[ObsType]


class BaseEnvironment(Generic[ObsType, ActType], ABC):
    """Base class for environments."""

    def __init__(self, seed: int | None = None, **kwargs: Any) -> None:
        """Initialize environment.

        Args:
            seed: Random seed
            **kwargs: Additional arguments for subclasses
        """
        # Create environments
        self.env = self._create_env()
        self.wrapped_env = self._create_wrapped_env()

        # Create planning components
        self.components = self._create_planning_components(**kwargs)

        # Initialize environments
        if seed is not None:
            self.env.reset(seed=seed)
            self.wrapped_env.reset(seed=seed)

    @property
    def types(self) -> set[Type]:
        """Get types."""
        return self.components.types

    @property
    def predicates(self) -> set[Predicate]:
        """Get PDDL predicates."""
        return self.components.predicate_container.as_set()

    @property
    def operators(self) -> set[LiftedOperator]:
        """Get PDDL operators."""
        return self.components.operators

    @property
    def perceiver(self) -> Perceiver[ObsType]:
        """Get state perceiver."""
        return self.components.perceiver

    @property
    def skills(self) -> set[Skill]:
        """Get skills."""
        return self.components.skills

    @abstractmethod
    def _create_env(self) -> gym.Env:
        """Create the base environment."""

    @abstractmethod
    def _create_wrapped_env(self) -> gym.Env:
        """Create the wrapped environment for training."""

    @abstractmethod
    def _create_planning_components(self, **kwargs: Any) -> PlanningComponents[ObsType]:
        """Create all planning-related components."""

    @abstractmethod
    def _get_domain_name(self) -> str:
        """Get domain name."""

    def get_domain(self) -> PDDLDomain:
        """Get PDDL domain."""
        return PDDLDomain(
            self._get_domain_name(), self.operators, self.predicates, self.types
        )

    def reset(self, seed: int | None = None) -> tuple[ObsType, dict[str, Any]]:
        """Reset environment."""
        return self.env.reset(seed=seed)
