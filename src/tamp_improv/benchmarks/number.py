"""Number environment implementation."""

from dataclasses import dataclass
from typing import Any, Sequence

import gymnasium as gym
from gymnasium.spaces import Discrete
from relational_structs import (
    GroundAtom,
    LiftedOperator,
    Object,
    Predicate,
    Type,
    Variable,
)
from task_then_motion_planning.structs import LiftedOperatorSkill, Perceiver

from tamp_improv.benchmarks.base import BaseEnvironment, PlanningComponents


@dataclass
class NumberPredicates:
    """Container for predicates to allow dictionary-like access."""

    at_state0: Predicate
    at_state1: Predicate
    at_state2: Predicate
    can_progress: Predicate | None = None

    def __getitem__(self, key: str) -> Predicate:
        """Get predicate by name."""
        if key == "AtState0":
            return self.at_state0
        if key == "AtState1":
            return self.at_state1
        if key == "AtState2":
            return self.at_state2
        if key == "CanProgress" and self.can_progress is not None:
            return self.can_progress
        raise KeyError(f"Unknown predicate: {key}")

    def as_set(self) -> set[Predicate]:
        """Convert to set of predicates."""
        predicates = {self.at_state0, self.at_state1, self.at_state2}
        if self.can_progress is not None:
            predicates.add(self.can_progress)
        return predicates


class BaseNumberSkill(LiftedOperatorSkill[int, int]):
    """Base class for number environment skills."""

    def __init__(self, env: "NumberEnvironment") -> None:
        """Initialize skill."""
        super().__init__()
        self._env = env

    def _get_action_given_objects(self, objects: Sequence[Object], obs: int) -> int:
        """All skills in this environment just return 1."""
        return 1


class ZeroToOneSkill(BaseNumberSkill):
    """Skill for transitioning from state 0 to 1."""

    def _get_lifted_operator(self) -> LiftedOperator:
        """Get lifted operator."""
        return next(op for op in self._env.operators if op.name == "ZeroToOne")


class OneToTwoSkill(BaseNumberSkill):
    """Skill for transitioning from state 1 to 2."""

    def _get_lifted_operator(self) -> LiftedOperator:
        """Get lifted operator."""
        return next(op for op in self._env.operators if op.name == "OneToTwo")


class BaseNumberPerceiver(Perceiver[int]):
    """Base class for number environment perceiver."""

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


class NumberPerceiver(BaseNumberPerceiver):
    """Perceiver for number environment."""

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


class NumberEnvironment(BaseEnvironment[int, int]):
    """Number environment with states {0,1,2}."""

    def __init__(
        self,
        seed: int | None = None,
        switch_off_improvisational_models: bool = False,
    ) -> None:
        """Initialize number environment."""
        self._state_type = Type("state")
        self._perceiver = NumberPerceiver(self._state_type)

        super().__init__(
            seed=seed,
            switch_off_improvisational_models=switch_off_improvisational_models,
        )

    def _create_env(self) -> gym.Env:
        """Create base number environment."""

        class NumberEnv(gym.Env):
            """Number environment with states {0,1,2}.

            States: 0,1,2
            Legal transitions: 0->1, 1->2
            Initial state: 0
            Goal state: 2
            Action: Single integer
            """

            def __init__(self) -> None:
                self.action_space = Discrete(2)
                self.observation_space = Discrete(3)
                self.state = 0

            def reset(
                self,
                *,
                seed: int | None = None,
                options: dict[str, Any] | None = None,
            ) -> tuple[int, dict[str, Any]]:
                super().reset(seed=seed)
                self.state = 0
                return self.state, {}

            def step(
                self, action: int
            ) -> tuple[int, float, bool, bool, dict[str, Any]]:
                if action == 1 and self.state < 2:
                    self.state += 1
                reward = float(self.state == 2)
                terminated = self.state == 2
                return self.state, reward, terminated, False, {}

            def render(self) -> None:
                print(f"Current state: {self.state}")

        return NumberEnv()

    def _create_wrapped_env(self) -> gym.Env:
        """Create wrapped environment for training."""

        class NumberEnvWrapper(gym.Env):
            """Wrapped environment for learning the 1->2 transition."""

            def __init__(self, base_env: gym.Env) -> None:
                self.env = base_env
                self.action_space = base_env.action_space
                self.observation_space = base_env.observation_space
                self.max_episode_steps = 10
                self.steps = 0

            def reset(
                self,
                *,
                seed: int | None = None,
                options: dict[str, Any] | None = None,
            ) -> tuple[int, dict[str, Any]]:
                self.steps = 0
                return self.env.reset(seed=seed)

            def step(
                self, action: int
            ) -> tuple[int, float, bool, bool, dict[str, Any]]:
                obs, _, _, truncated, info = self.env.step(action)
                self.steps += 1

                # Success is reaching state 2
                success = obs == 2
                reward = 1.0 if success else 0.0
                terminated = success
                truncated = truncated or self.steps >= self.max_episode_steps
                return obs, reward, terminated, truncated, info

            def render(self) -> None:
                self.env.render()

        return NumberEnvWrapper(self.env)

    def _create_planning_components(
        self,
        switch_off_improvisational_models: bool = False,
        **kwargs: Any,
    ) -> PlanningComponents[int]:
        """Create all planning components."""
        # Create predicates
        predicates = NumberPredicates(
            at_state0=Predicate("AtState0", [self._state_type]),
            at_state1=Predicate("AtState1", [self._state_type]),
            at_state2=Predicate("AtState2", [self._state_type]),
            can_progress=(
                Predicate("CanProgress", [self._state_type])
                if switch_off_improvisational_models
                else None
            ),
        )

        # Initialize perceiver with predicates
        self._perceiver.initialize(predicates)

        # Create operators
        state = Variable("?state", self._state_type)
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

        return PlanningComponents(
            types={self._state_type},
            predicate_container=predicates,
            operators=operators,
            skills={ZeroToOneSkill(self), OneToTwoSkill(self)},
            perceiver=self._perceiver,
        )

    def _get_domain_name(self) -> str:
        """Get domain name."""
        return "number-domain"
