"""RL-based implementation of improvisational TAMP for Blocks2D environment."""

import numpy as np
from numpy.typing import NDArray
from relational_structs import (
    GroundOperator,
    PDDLDomain,
)
from task_then_motion_planning.planning import TaskThenMotionPlanningFailure
from task_then_motion_planning.structs import Perceiver, Skill, _Action, _Observation

from tamp_improv.approaches.base_improvisational_tamp_approach import (
    ImprovisationalTAMPApproach,
)
from tamp_improv.approaches.rl_improvisational_policy import RLImprovisationalPolicy
from tamp_improv.benchmarks.blocks2d_env import Blocks2DEnv
from tamp_improv.benchmarks.pushing_env import make_pushing_env
from tamp_improv.blocks2d_planning import create_blocks2d_planning_models


class RLBlocks2DImprovisationalTAMPApproach(ImprovisationalTAMPApproach):
    """Blocks2D improvisational TAMP approach using learned RL policy.

    This approach uses a trained RL policy to handle situations where
    preconditions aren't met (specifically when the target area is
    blocked), and a TAMP planner for high-level task planning. The RL
    policy is trained separately to learn pushing behaviors.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        seed: int,
        policy_path: str,
        planner_id: str = "pyperplan",
        domain_name: str = "custom-domain",
    ) -> None:
        """Initialize the approach.

        Args:
            observation_space: Observation space from the environment
            action_space: Action space from the environment
            seed: Random seed
            policy_path: Path to saved RL policy
            planner_id: ID of PDDL planner to use
            domain_name: Name of PDDL domain
        """
        # Create base env and wrapped env for the RL policy
        base_env = Blocks2DEnv()
        pushing_env = make_pushing_env(base_env, seed=seed)

        # Initialize RL policy and load weights
        policy = RLImprovisationalPolicy(pushing_env)
        policy.load(policy_path)

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
        self._domain_name = domain_name
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
        """Get the appropriate skill for executing an operator.

        Args:
            operator: The ground operator to execute

        Returns:
            Skill that can execute the operator

        Raises:
            TaskThenMotionPlanningFailure: If no skill can execute the operator
        """
        applicable_skills = [s for s in self._skills if s.can_execute(operator)]
        if not applicable_skills:
            raise TaskThenMotionPlanningFailure("No skill can execute operator")
        assert len(applicable_skills) == 1, "Multiple operators per skill not supported"
        return applicable_skills[0]
