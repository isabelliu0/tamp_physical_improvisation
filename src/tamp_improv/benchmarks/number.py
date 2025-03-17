"""Number environment implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray
from relational_structs import (
    GroundAtom,
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
from tamp_improv.benchmarks.number_env import NumberEnv
from tamp_improv.benchmarks.wrappers import ImprovWrapper


@dataclass
class NumberPredicates:
    """Container for predicates to allow dictionary-like access."""

    at_state0: Predicate
    at_state1: Predicate
    at_state2: Predicate
    can_progress: Predicate | None = None

    def __getitem__(self, key: str) -> Predicate:
        """Get predicate by name."""
        return next(p for p in self.as_set() if p.name == key)

    def as_set(self) -> set[Predicate]:
        """Convert to set of predicates."""
        predicates = {self.at_state0, self.at_state1, self.at_state2}
        if self.can_progress is not None:
            predicates.add(self.can_progress)
        return predicates


class BaseNumberSkill(LiftedOperatorSkill[NDArray[np.int32], NDArray[np.int32]]):
    """Base class for number environment skills."""

    def __init__(self, components: PlanningComponents[NDArray[np.int32]]) -> None:
        """Initialize skill."""
        super().__init__()
        self._components = components

    def _get_action_given_objects(
        self, objects: Sequence[Object], obs: NDArray[np.int32]
    ) -> NDArray[np.int32]:
        """Return action for movement without touching light switch."""
        return np.array([1, 0])


class ZeroToOneSkill(BaseNumberSkill):
    """Skill for transitioning from state 0 to 1."""

    def _get_lifted_operator(self) -> LiftedOperator:
        """Get lifted operator."""
        return next(op for op in self._components.operators if op.name == "ZeroToOne")


class OneToTwoSkill(BaseNumberSkill):
    """Skill for transitioning from state 1 to 2."""

    def _get_lifted_operator(self) -> LiftedOperator:
        """Get lifted operator."""
        return next(op for op in self._components.operators if op.name == "OneToTwo")


class NumberPerceiver(Perceiver[NDArray[np.int32]]):
    """Perceiver for number environment."""

    def __init__(self, state_type: Type) -> None:
        """Initialize with state type."""
        self._state = Object("state1", state_type)
        self._predicates: NumberPredicates | None = None

    def initialize(self, predicates: NumberPredicates) -> None:
        """Initialize predicates after environment creation."""
        self._predicates = predicates

    @property
    def predicates(self) -> NumberPredicates:
        """Get predicates, ensuring they're initialized."""
        if self._predicates is None:
            raise RuntimeError("Predicates not initialized. Call initialize() first.")
        return self._predicates

    def reset(
        self, obs: NDArray[np.int32], _info: dict[str, Any]
    ) -> tuple[set[Object], set[GroundAtom], set[GroundAtom]]:
        """Reset perceiver with observation."""
        objects = {self._state}
        atoms = self._get_atoms(obs)
        goal = {self.predicates["AtState2"]([self._state])}
        return objects, atoms, goal

    def step(self, obs: NDArray[np.int32]) -> set[GroundAtom]:
        """Update perceiver with new observation."""
        return self._get_atoms(obs)

    def _get_atoms(self, obs: NDArray[np.int32]) -> set[GroundAtom]:
        """Get atoms from observation."""
        state, light_switch = obs
        atoms = set()

        state_to_predicate = {
            0: "AtState0",
            1: "AtState1",
            2: "AtState2",
        }
        atoms.add(self.predicates[state_to_predicate[state]]([self._state]))

        if light_switch == 1:
            atoms.add(self.predicates["CanProgress"]([self._state]))

        return atoms


class BaseNumberTAMPSystem(BaseTAMPSystem[NDArray[np.int32], NDArray[np.int32]]):
    """Base TAMP system for number environment."""

    def __init__(
        self,
        planning_components: PlanningComponents[NDArray[np.int32]],
        seed: int | None = None,
        render_mode: str | None = None,
    ) -> None:
        """Initialize number TAMP system."""
        super().__init__(planning_components, name="NumberTAMPSystem", seed=seed)
        self._render_mode = render_mode

    def _create_env(self) -> gym.Env:
        """Create base number environment."""
        return NumberEnv()

    def _get_domain_name(self) -> str:
        """Get domain name."""
        return "number-domain"

    def get_domain(self) -> PDDLDomain:
        """Get domain with or without transition preconditions.

        Args:
            include_extra_preconditions: If True, include the CanProgress precondition
            to switch off improvisational models. If False, improvisational models for
            TAMP are enabled.
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
    ) -> NumberTAMPSystem:
        """Factory method for creating system with default components."""
        # Create state type
        state_type = Type("state")
        types = {state_type}

        # Create predicates
        predicates = NumberPredicates(
            at_state0=Predicate("AtState0", [state_type]),
            at_state1=Predicate("AtState1", [state_type]),
            at_state2=Predicate("AtState2", [state_type]),
            can_progress=Predicate("CanProgress", [state_type]),
        )

        # Create perceiver with state type
        perceiver = NumberPerceiver(state_type)
        perceiver.initialize(predicates)

        # Create variables for operators
        state = Variable("?state", state_type)

        # Create base operators (without transition preconditions)
        base_operators = {
            LiftedOperator(
                "ZeroToOne",
                [state],
                preconditions={predicates["AtState0"]([state])},
                add_effects={predicates["AtState1"]([state])},
                delete_effects={predicates["AtState0"]([state])},
            ),
            LiftedOperator(
                "OneToTwo",
                [state],
                preconditions={predicates["AtState1"]([state])},
                add_effects={predicates["AtState2"]([state])},
                delete_effects={predicates["AtState1"]([state])},
            ),
        }

        # Create full operators (with transition preconditions)
        _ = {
            LiftedOperator(
                "ZeroToOne",
                [state],
                preconditions={predicates["AtState0"]([state])},
                add_effects={predicates["AtState1"]([state])},
                delete_effects={predicates["AtState0"]([state])},
            ),
            LiftedOperator(
                "OneToTwo",
                [state],
                preconditions={
                    predicates["AtState1"]([state]),
                    predicates["CanProgress"]([state]),
                },
                add_effects={predicates["AtState2"]([state])},
                delete_effects={predicates["AtState1"]([state])},
            ),
        }

        # Create system
        system = NumberTAMPSystem(
            PlanningComponents(
                types=types,
                predicate_container=predicates,
                operators=base_operators,
                skills=set(),
                perceiver=perceiver,
            ),
            seed=seed,
            render_mode=render_mode,
        )

        # Create skills with reference to components
        skills = {
            ZeroToOneSkill(system.components),
            OneToTwoSkill(system.components),
        }

        # Update components with skills
        system.components.skills.update(skills)

        return system


class NumberTAMPSystem(
    ImprovisationalTAMPSystem[NDArray[np.int32], NDArray[np.int32]],
    BaseNumberTAMPSystem,
):
    """TAMP system for the number environment with improvisational policy
    learning enabled."""

    def _create_wrapped_env(
        self, components: PlanningComponents[NDArray[np.int32]]
    ) -> gym.Env:
        """Create wrapped environment for training."""
        return ImprovWrapper(
            self.env, perceiver=components.perceiver, max_episode_steps=10
        )
