"""MPC-based implementation of improvisational TAMP for Blocks2D
environment."""

import numpy as np
from numpy.typing import NDArray
from relational_structs import GroundOperator, PDDLDomain
from task_then_motion_planning.planning import TaskThenMotionPlanningFailure
from task_then_motion_planning.structs import Perceiver, Skill, _Action, _Observation

from tamp_improv.approaches.base_improvisational_tamp_approach import (
    ImprovisationalTAMPApproach,
)
from tamp_improv.approaches.mpc_improvisational_policy import (
    MPCImprovisationalPolicy,
    PredictiveSamplingConfig,
)
from tamp_improv.benchmarks.blocks2d_env_old import Blocks2DEnv
from tamp_improv.benchmarks.blocks2d_env_wrapper_old import make_pushing_env
from tamp_improv.blocks2d_planning import create_blocks2d_planning_models


class MPCBlocks2DImprovisationalTAMPApproach(ImprovisationalTAMPApproach):
    """Blocks2D improvisational TAMP approach using MPC policy."""

    def __init__(
        self,
        observation_space,
        action_space,
        seed: int,
        config: PredictiveSamplingConfig | None = None,
        planner_id: str = "pyperplan",
        domain_name: str = "custom-domain",
    ) -> None:
        # Create base env and wrapped env for the MPC policy
        base_env = Blocks2DEnv()
        training_env = make_pushing_env(base_env, seed)

        # Initialize MPC policy
        policy: MPCImprovisationalPolicy[NDArray[np.float32], NDArray[np.float32]] = (
            MPCImprovisationalPolicy(training_env, seed=seed, config=config)
        )

        # Initialize base class
        super().__init__(
            observation_space, action_space, seed, policy, planner_id, domain_name
        )

        self.env_name = "blocks2d"

        # Initialize Blocks2D-specific planning components
        types, predicates, _, operators, skills = create_blocks2d_planning_models(
            include_pushing_models=False
        )
        _, _, perceiver, full_precondition_operators, _ = (
            create_blocks2d_planning_models(include_pushing_models=True)
        )

        self._types = types
        self._predicates = predicates
        self._operators = operators
        self._skills = skills
        self._domain = PDDLDomain(
            self._domain_name, self._operators, self._predicates, self._types
        )
        self._perceiver: Perceiver[NDArray[np.float32]] = perceiver

        # Set up operator mapping
        operator_name_to_operator = {o.name: o for o in self._operators}
        for full_precondition_operator in full_precondition_operators:
            if full_precondition_operator.name not in operator_name_to_operator:
                continue
            operator = operator_name_to_operator[full_precondition_operator.name]
            self._operator_to_full_precondition_operator[operator] = (
                full_precondition_operator
            )

    def _get_skill_for_operator(
        self, operator: GroundOperator
    ) -> Skill[_Observation, _Action]:
        """Get the appropriate skill for executing an operator."""
        applicable_skills = [s for s in self._skills if s.can_execute(operator)]
        if not applicable_skills:
            raise TaskThenMotionPlanningFailure("No skill can execute operator")
        assert len(applicable_skills) == 1
        return applicable_skills[0]
