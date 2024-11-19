"""Number environment implementation."""

from typing import Any

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
from task_then_motion_planning.structs import LiftedOperatorSkill, Perceiver, Skill

from tamp_improv.benchmarks.base import BaseEnvironment


class NumberEnvironment(BaseEnvironment[int, int]):
    """Number environment with states {0,1,2}."""

    def _create_env(self) -> gym.Env:
        """Create base environment."""

        class NumberEnv(gym.Env):
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

            # TODO: render()

        return NumberEnv()

    def _create_wrapped_env(self) -> gym.Env:
        """Create wrapped environment for training."""

        class NumberEnvWrapper(gym.Env):
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
                success = obs == 2
                reward = 1.0 if success else 0.0
                terminated = success
                truncated = truncated or self.steps >= self.max_episode_steps
                return obs, reward, terminated, truncated, info

        return NumberEnvWrapper(self.env)

    def _create_types(self) -> set[Type]:
        """Create PDDL types."""
        return {Type("state")}

    def _create_predicates(self) -> set[Predicate]:
        """Create PDDL predicates."""
        state_type = next(iter(self.types))  # Get the state type
        # TODO: CanProgress is added only if switch_off_improvisational_models is True
        predicates = {
            Predicate("AtState0", [state_type]),
            Predicate("AtState1", [state_type]),
            Predicate("AtState2", [state_type]),
            Predicate("CanProgress", [state_type]),  # For full preconditions
        }
        return predicates

    def _create_operators(self) -> set[LiftedOperator]:
        """Create PDDL operators."""
        state_type = next(iter(self.types))  # Get the state type
        state = Variable("?state", state_type)

        # Create operators
        ops = set()

        # Zero to One operator
        ops.add(
            LiftedOperator(
                "ZeroToOne",
                [state],
                preconditions={self.predicates["AtState0"]([state])},
                add_effects={self.predicates["AtState1"]([state])},
                delete_effects={self.predicates["AtState0"]([state])},
            )
        )

        # One to Two operator
        ops.add(
            LiftedOperator(
                "OneToTwo",
                [state],
                preconditions={
                    self.predicates["AtState1"]([state]),
                    self.predicates["CanProgress"]([state]),
                },
                add_effects={self.predicates["AtState2"]([state])},
                delete_effects={self.predicates["AtState1"]([state])},
            )
        )

        return ops

    def _create_perceiver(self) -> Perceiver[int]:
        """Create state perceiver."""

        class NumberPerceiver(Perceiver[int]):
            def __init__(self, env: NumberEnvironment) -> None:
                self._env = env
                self._state = Object(env.types["state"], "state1")

            def reset(
                self, obs: int, info: dict[str, Any]
            ) -> tuple[set[Object], set[GroundAtom], set[GroundAtom]]:
                objects = {self._state}
                atoms = self._get_atoms(obs)
                goal = {self._env.predicates["AtState2"]([self._state])}
                return objects, atoms, goal

            def step(self, obs: int) -> set[GroundAtom]:
                return self._get_atoms(obs)

            def _get_atoms(self, obs: int) -> set[GroundAtom]:
                atoms = set()
                state_to_predicate = {
                    0: self._env.predicates["AtState0"],
                    1: self._env.predicates["AtState1"],
                    2: self._env.predicates["AtState2"],
                }
                atoms.add(state_to_predicate[obs]([self._state]))
                return atoms

        return NumberPerceiver(self)

    def _create_skills(self) -> set[Skill]:
        """Create skills for operators."""

        class ZeroToOneSkill(LiftedOperatorSkill[int, int]):
            def _get_lifted_operator(self) -> LiftedOperator:
                return next(op for op in self._env.operators if op.name == "ZeroToOne")

            def _get_action_given_objects(self, objects: list[Object], obs: int) -> int:
                return 1

        class OneToTwoSkill(LiftedOperatorSkill[int, int]):
            def _get_lifted_operator(self) -> LiftedOperator:
                return next(op for op in self._env.operators if op.name == "OneToTwo")

            def _get_action_given_objects(self, objects: list[Object], obs: int) -> int:
                return 1

        return {ZeroToOneSkill(), OneToTwoSkill()}

    def _get_domain_name(self) -> str:
        """Get domain name."""
        return "number-domain"
