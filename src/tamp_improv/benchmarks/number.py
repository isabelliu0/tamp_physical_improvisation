"""Number environment implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import gymnasium as gym
from relational_structs import (
    GroundAtom,
    LiftedOperator,
    Object,
    Predicate,
    Type,
    Variable,
)
from task_then_motion_planning.structs import LiftedOperatorSkill, Perceiver

from tamp_improv.benchmarks.base import BaseSkillLearningSys, PlanningComponents
from tamp_improv.benchmarks.number_env import NumberEnv
from tamp_improv.benchmarks.number_wrappers import NumberEnvWrapper


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


class BaseNumberSkill(LiftedOperatorSkill[int, int]):
    """Base class for number environment skills."""

    def __init__(self, components: PlanningComponents[int]) -> None:
        """Initialize skill."""
        super().__init__()
        self._components = components

    def _get_action_given_objects(self, objects: Sequence[Object], obs: int) -> int:
        """All skills in this environment just return 1."""
        return 1


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


class NumberPerceiver(Perceiver[int]):
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
        self, obs: int, info: dict[str, Any]
    ) -> tuple[set[Object], set[GroundAtom], set[GroundAtom]]:
        objects = {self._state}
        atoms = self._get_atoms(obs)
        goal = {self.predicates["AtState2"]([self._state])}
        return objects, atoms, goal

    def step(self, obs: int) -> set[GroundAtom]:
        return self._get_atoms(obs)

    def _get_atoms(self, obs: int) -> set[GroundAtom]:
        atoms = set()
        state_to_predicate = {
            0: "AtState0",
            1: "AtState1",
            2: "AtState2",
        }
        atoms.add(self.predicates[state_to_predicate[obs]]([self._state]))
        return atoms


class NumberTAMPSystem(BaseSkillLearningSys[int, int]):
    """TAMP system for the number environment."""

    def _create_env(self) -> gym.Env:
        """Create base number environment."""
        return NumberEnv()

    def _create_wrapped_env(self, components: PlanningComponents[int]) -> gym.Env:
        """Create wrapped environment for training."""
        return NumberEnvWrapper(self.env)

    def _get_domain_name(self) -> str:
        """Get domain name."""
        return "number-domain"

    @staticmethod
    def create_default(
        seed: int | None = None,
        switch_off_improvisational_models: bool = False,
    ) -> NumberTAMPSystem:
        """Factory method for creating system with default components."""
        # Create state type
        state_type = Type("state")

        # Create predicates
        predicates = NumberPredicates(
            at_state0=Predicate("AtState0", [state_type]),
            at_state1=Predicate("AtState1", [state_type]),
            at_state2=Predicate("AtState2", [state_type]),
            can_progress=(
                Predicate("CanProgress", [state_type])
                if switch_off_improvisational_models
                else None
            ),
        )

        # Create perceiver with state type
        perceiver = NumberPerceiver(state_type)
        perceiver.initialize(predicates)

        # Create operators
        state = Variable("?state", state_type)
        operators = {
            LiftedOperator(
                "ZeroToOne",
                [state],
                preconditions={predicates["AtState0"]([state])},
                add_effects={predicates["AtState1"]([state])},
                delete_effects={predicates["AtState0"]([state])},
            ),
        }

        # One to Two operator (preconditions depend on improvisation setting)
        one_to_two_preconditions = {predicates["AtState1"]([state])}
        if switch_off_improvisational_models:
            one_to_two_preconditions.add(predicates["CanProgress"]([state]))

        operators.add(
            LiftedOperator(
                "OneToTwo",
                [state],
                preconditions=one_to_two_preconditions,
                add_effects={predicates["AtState2"]([state])},
                delete_effects={predicates["AtState1"]([state])},
            ),
        )

        # Create system
        system = NumberTAMPSystem(
            PlanningComponents(
                types={state_type},
                predicate_container=predicates,
                operators=operators,
                skills=set(),
                perceiver=perceiver,
            ),
            seed=seed,
        )

        # Create skills with reference to components
        skills = {
            ZeroToOneSkill(system.components),
            OneToTwoSkill(system.components),
        }

        # Update components with skills
        system.components.skills.update(skills)

        return system
