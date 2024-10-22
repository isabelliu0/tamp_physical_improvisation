"""Improvisational TAMP approach."""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from tamp_improv.approaches.base_approach import BaseApproach
from tamp_improv.benchmarks.blocks2d_env import Blocks2DEnv
from tamp_improv.blocks2d_planning import create_blocks2d_planning_models

from relational_structs import (
    GroundOperator,
    LiftedOperator,
    PDDLDomain,
    PDDLProblem,
    Predicate,
    Type,
)
from relational_structs.utils import parse_pddl_plan
from tomsutils.pddl_planning import run_pddl_planner
from task_then_motion_planning.structs import Perceiver, Skill, _Action, _Observation
from task_then_motion_planning.planning import TaskThenMotionPlanningFailure


class ImprovisationalTAMPApproach(
    BaseApproach[NDArray[np.float32], NDArray[np.float32]]
):
    """Improvisational TAMP approach."""

    def __init__(self, observation_space, action_space, seed: int,
                 planner_id: str = "pyperplan",
                 domain_name: str = "custom-domain") -> None:
        super().__init__(observation_space, action_space, seed)
        self.env_name = "blocks2d"

        types, predicates, _, operators, skills = create_blocks2d_planning_models(include_pushing_models=False)

        self._types = types
        self._predicates = predicates
        self._operators = operators
        self._skills = skills
        self._planner_id = planner_id
        self._domain_name = domain_name
        self._domain = PDDLDomain(
            self._domain_name, self._operators, self._predicates, self._types
        )

        # Create operators with "full" preconditions.
        _, _, perceiver, full_precondition_operators, _ = create_blocks2d_planning_models(include_pushing_models=True)
        self._operator_to_full_precondition_operator: dict[LiftedOperator, LiftedOperator] = {}
        operator_name_to_operator = {o.name: o for o in self._operators}
        for full_precondition_operator in full_precondition_operators:
            if full_precondition_operator.name not in operator_name_to_operator:
                continue
            operator = operator_name_to_operator[full_precondition_operator.name]
            self._operator_to_full_precondition_operator[operator] = full_precondition_operator
        self._perceiver = perceiver

        self._current_problem: PDDLProblem | None = None
        self._current_task_plan: list[GroundOperator] = []
        self._current_operator: GroundOperator | None = None
        self._current_skill: Skill | None = None

    def reset(
        self, obs: NDArray[np.float32], info: dict[str, Any]
    ) -> NDArray[np.float32]:
        objects, atoms, goal = self._perceiver.reset(obs, info)
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

    def step(
        self,
        obs: NDArray[np.float32],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ) -> NDArray[np.float32]:
        # Get the current atoms.
        atoms = self._perceiver.step(obs)

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
            full_lifted_operator = self._operator_to_full_precondition_operator[self._current_operator.parent]
            full_ground_operator = full_lifted_operator.ground(tuple(self._current_operator.parameters))
            if not (full_ground_operator.preconditions.issubset(atoms)):
                # The preconditions don't hold, so we need to call policy!
                import ipdb; ipdb.set_trace()

            # Get a skill that can execute this operator.
            self._current_skill = self._get_skill_for_operator(self._current_operator)
            self._current_skill.reset(self._current_operator)

        assert self._current_skill is not None
        return self._current_skill.get_action(obs)

    def _get_skill_for_operator(
        self, operator: GroundOperator
    ) -> Skill[_Observation, _Action]:
        applicable_skills = [s for s in self._skills if s.can_execute(operator)]
        if not applicable_skills:
            raise TaskThenMotionPlanningFailure("No skill can execute operator")
        assert len(applicable_skills) == 1, "Multiple operators per skill not supported"
        return applicable_skills[0]
