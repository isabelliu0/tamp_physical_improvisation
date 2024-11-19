"""Base framework for improvisational TAMP approach."""

from typing import Any, Generic, Set

import gymnasium as gym
from gymnasium.core import ActType, ObsType
from relational_structs import (
    GroundAtom,
    GroundOperator,
    LiftedOperator,
    PDDLDomain,
    PDDLProblem,
)
from relational_structs.utils import parse_pddl_plan
from task_then_motion_planning.planning import TaskThenMotionPlanningFailure
from task_then_motion_planning.structs import Perceiver, Skill, _Action, _Observation
from tomsutils.pddl_planning import run_pddl_planner

from tamp_improv.approaches.base_approach import BaseApproach


class ImprovisationalPolicy(Generic[ObsType, ActType]):
    """Abstract base class for improvisational policies."""

    def __init__(self, env: gym.Env) -> None:
        """Initialize policy."""
        self.env = env

    def get_action(self, obs: ObsType) -> ActType:
        """Get action from policy based on current observation."""
        raise NotImplementedError

    def train(self, total_timesteps: int, seed: int | None = None) -> None:
        """Train the policy."""
        raise NotImplementedError

    def save(self, path: str) -> None:
        """Save the policy."""
        raise NotImplementedError

    def load(self, path: str) -> None:
        """Load the policy."""
        raise NotImplementedError


class ImprovisationalTAMPApproach(BaseApproach[ObsType, ActType]):
    """Generic improvisational TAMP approach.

    This class provides the core framework for combining TAMP with
    learned policies that handle situations where preconditions aren't
    met. Specific environment implementations should subclass this and
    provide their own policy implementation.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        seed: int,
        policy: ImprovisationalPolicy[ObsType, ActType],
        planner_id: str = "pyperplan",
        domain_name: str = "custom-domain",
    ) -> None:
        super().__init__(observation_space, action_space, seed)

        # Initialize planning components
        self._domain_name = domain_name
        self._planner_id = planner_id
        self._current_problem: PDDLProblem | None = None
        self._current_task_plan: list[GroundOperator] = []
        self._current_operator: GroundOperator | None = None
        self._current_skill: Skill | None = None

        # Initialize policy components
        self._policy = policy
        self._policy_active: bool = False
        self._target_atoms: Set[GroundAtom] = set()
        self._goal: Set[GroundAtom] = set()

        # These need to be set by subclass
        self._domain: PDDLDomain | None = None
        self._operator_to_full_precondition_operator: dict[
            LiftedOperator, LiftedOperator
        ] = {}
        self._perceiver: Perceiver[ObsType] | None = None

    def _replan(self, obs: ObsType, info: dict[str, Any]) -> None:
        """Create a new plan from the current state."""
        assert self._domain is not None and self._perceiver is not None

        objects, atoms, _ = self._perceiver.reset(obs, info)
        self._current_problem = PDDLProblem(
            self._domain_name, self._domain_name, objects, atoms, self._goal
        )
        plan_str = run_pddl_planner(
            str(self._domain), str(self._current_problem), planner=self._planner_id
        )
        assert plan_str is not None
        self._current_task_plan = parse_pddl_plan(
            plan_str, self._domain, self._current_problem
        )
        self._current_operator = None
        self._current_skill = None

    def reset(self, obs: ObsType, info: dict[str, Any]) -> ActType:
        assert self._domain is not None and self._perceiver is not None

        objects, atoms, goal = self._perceiver.reset(obs, info)
        self._goal = goal
        self._current_problem = PDDLProblem(
            self._domain_name, self._domain_name, objects, atoms, goal
        )
        plan_str = run_pddl_planner(
            str(self._domain), str(self._current_problem), planner=self._planner_id
        )
        assert plan_str is not None
        self._current_task_plan = parse_pddl_plan(
            plan_str, self._domain, self._current_problem
        )
        self._current_operator = None
        self._current_skill = None
        self._policy_active = False
        self._target_atoms = set()

        return self.step(obs, 0.0, False, False, info)

    def step(
        self,
        obs: ObsType,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ) -> ActType:
        assert self._perceiver is not None
        atoms = self._perceiver.step(obs)

        # Check if the policy achieved its goal
        if self._policy_active:
            if self._target_atoms.issubset(atoms):
                print("Policy successfully achieved target atoms!")
                self._policy_active = False
                self._target_atoms = set()
                self._replan(obs, info)  # replan from current state
            else:
                return self._policy.get_action(obs)

        # If the current operator is None or terminated, get the next one.
        if self._current_operator is None or (
            self._current_operator.add_effects.issubset(atoms)
            and not (self._current_operator.delete_effects & atoms)
        ):
            # If there is no more plan to execute, fail.
            if not self._current_task_plan:
                raise TaskThenMotionPlanningFailure("Empty task plan")

            self._current_operator = self._current_task_plan.pop(0)

            # Check preconditions of the FULL operator. If any are not satisfied,
            # call the improvisational policy to get us back to a state where
            # the preconditions are satisfied.
            full_lifted_operator = self._operator_to_full_precondition_operator[
                self._current_operator.parent
            ]
            full_ground_operator = full_lifted_operator.ground(
                tuple(self._current_operator.parameters)
            )

            if not full_ground_operator.preconditions.issubset(atoms):
                print("Preconditions not met, activating policy")
                self._policy_active = True
                self._target_atoms = full_ground_operator.preconditions
                return self._policy.get_action(obs)

            # Get a skill that can execute this operator.
            self._current_skill = self._get_skill_for_operator(self._current_operator)
            self._current_skill.reset(self._current_operator)

        assert self._current_skill is not None
        return self._current_skill.get_action(obs)

    def _get_skill_for_operator(
        self, operator: GroundOperator
    ) -> Skill[_Observation, _Action]:
        raise NotImplementedError("Subclasses must implement _get_skill_for_operator")
