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
    base_operators: set[LiftedOperator]
    full_operators: set[LiftedOperator]
    full_operators_active: bool
    skills: set[Skill]
    perceiver: Perceiver[ObsType]

    @property
    def operators(self) -> set[LiftedOperator]:
        """Get the currently active operator set."""
        return (
            self.full_operators if self.full_operators_active else self.base_operators
        )


class BaseTAMPSystem(Generic[ObsType, ActType], ABC):
    """Base class for Task-and-Motion Planning (TAMP) systems.

    This class combines:
    1. The actual environment (gym.Env) that represents the physical world
    2. The agent's planning components (types, predicates, operators, etc.)
       that represent the agent's abstract model of the world for planning
    """

    def __init__(
        self,
        planning_components: PlanningComponents[ObsType],
        name: str = "TAMPSystem",
        seed: int | None = None,
    ) -> None:
        """Initialize TAMP system.

        Args:
            planning_components: The agent's planning model/components
            seed: Random seed for environment
        """
        self.name = name
        self.components = planning_components
        self.env = self._create_env()
        if seed is not None:
            self.env.reset(seed=seed)

    @property
    def types(self) -> set[Type]:
        """Get types."""
        return self.components.types

    @property
    def predicates(self) -> set[Predicate]:
        """Get PDDL predicates."""
        return self.components.predicate_container.as_set()

    @property
    def base_operators(self) -> set[LiftedOperator]:
        """Get PDDL operators without extra preconditions."""
        return self.components.base_operators

    @property
    def full_operators(self) -> set[LiftedOperator]:
        """Get PDDL operators with extra preconditions."""
        return self.components.full_operators

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
    def _get_domain_name(self) -> str:
        """Get domain name."""

    @abstractmethod
    def get_domain(self) -> PDDLDomain:
        """Get PDDL domain with or without extra preconditions for skill
        learning."""

    def reset(self, seed: int | None = None) -> tuple[ObsType, dict[str, Any]]:
        """Reset environment."""
        return self.env.reset(seed=seed)


class ImprovisationalTAMPSystem(BaseTAMPSystem[ObsType, ActType], ABC):
    """Base class for systems that support improvisational policies."""

    def __init__(
        self,
        planning_components: PlanningComponents[ObsType],
        seed: int | None = None,
        render_mode: str | None = None,
    ) -> None:
        """Initialize improvisational TAMP system."""
        self._render_mode = render_mode
        super().__init__(planning_components, seed=seed)
        self.wrapped_env = self._create_wrapped_env(planning_components)
        if seed is not None:
            self.wrapped_env.reset(seed=seed)

    @abstractmethod
    def _create_wrapped_env(self, components: PlanningComponents[ObsType]) -> gym.Env:
        """Create the wrapped environment for training."""
