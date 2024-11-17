"""RL-based implementation of improvisational TAMP for Number environment."""

from relational_structs import (
    GroundOperator,
    PDDLDomain,
)
from task_then_motion_planning.planning import TaskThenMotionPlanningFailure
from task_then_motion_planning.structs import Skill, _Action, _Observation

from tamp_improv.approaches.base_improvisational_tamp_approach import (
    ImprovisationalTAMPApproach,
)
from tamp_improv.approaches.rl_improvisational_policy import RLImprovisationalPolicy
from tamp_improv.benchmarks.number_env import NumberEnv
from tamp_improv.benchmarks.number_env_wrapper import make_number_env_wrapper
from tamp_improv.number_planning import create_number_planning_models


class RLNumberImprovisationalTAMPApproach(ImprovisationalTAMPApproach):
    """Number improvisational TAMP approach using learned RL policy."""

    def __init__(
        self,
        observation_space,
        action_space,
        seed: int,
        policy_path: str,
        planner_id: str = "pyperplan",
        domain_name: str = "simple-domain",
    ) -> None:
        # Create base env and wrapped env for the RL policy
        base_env = NumberEnv()
        training_env = make_number_env_wrapper(base_env, seed=seed)

        # Initialize RL policy and load weights
        policy: RLImprovisationalPolicy[int, int] = RLImprovisationalPolicy(
            training_env
        )
        policy.load(policy_path)

        # Initialize base class
        super().__init__(
            observation_space, action_space, seed, policy, planner_id, domain_name
        )

        self.env_name = "number"

        # Initialize Number-specific planning components
        types, predicates, _, operators, skills = create_number_planning_models(
            switch_off_improvisational_models=False
        )
        _, _, perceiver, full_precondition_operators, _ = create_number_planning_models(
            switch_off_improvisational_models=True
        )

        self._types = types
        self._predicates = predicates
        self._operators = operators
        self._skills = skills
        self._domain = PDDLDomain(
            self._domain_name, self._operators, self._predicates, self._types
        )
        self._perceiver = perceiver

        # Set up operator mapping
        operator_name_to_operator = {o.name: o for o in self._operators}
        for full_operator in full_precondition_operators:
            if full_operator.name not in operator_name_to_operator:
                continue
            operator = operator_name_to_operator[full_operator.name]
            self._operator_to_full_precondition_operator[operator] = full_operator

    def _get_skill_for_operator(
        self, operator: GroundOperator
    ) -> Skill[_Observation, _Action]:
        applicable_skills = [s for s in self._skills if s.can_execute(operator)]
        if not applicable_skills:
            raise TaskThenMotionPlanningFailure("No skill can execute operator")
        assert len(applicable_skills) == 1
        return applicable_skills[0]
