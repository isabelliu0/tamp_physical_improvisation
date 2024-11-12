"""MPC-based implementation of improvisational TAMP for Blocks2D
environment."""

from dataclasses import dataclass

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
from tamp_improv.approaches.predictive_sampling_policy import (
    PredictiveSamplingHyperparameters,
    PredictiveSamplingImprovisationalPolicy,
)
from tamp_improv.benchmarks.blocks2d_env import Blocks2DEnv
from tamp_improv.benchmarks.pushing_env import make_pushing_env
from tamp_improv.blocks2d_planning import create_blocks2d_planning_models


@dataclass(frozen=True)
class MPCBlocks2DConfig:
    """Configuration for MPC Blocks2D approach."""

    horizon: int = 20
    num_rollouts: int = 100
    noise_scale: float = 1.0
    num_control_points: int = 10
    dt: float = 0.5
    warm_start: bool = True


class MPCBlocks2DImprovisationalTAMPApproach(ImprovisationalTAMPApproach):
    """Blocks2D improvisational TAMP approach using MPC."""

    def __init__(
        self,
        observation_space,
        action_space,
        seed: int,
        config: MPCBlocks2DConfig | None = None,
        planner_id: str = "pyperplan",
        domain_name: str = "custom-domain",
    ) -> None:
        """Initialize the approach.

        Args:
            observation_space: Observation space from environment
            action_space: Action space from environment
            seed: Random seed
            config: Configuration for MPC approach
            planner_id: ID of PDDL planner to use
            domain_name: Name of PDDL domain
        """
        # Initialize configuration
        self._config = config or MPCBlocks2DConfig()

        # Create base env and wrapped env for MPC policy
        base_env = Blocks2DEnv()
        pushing_env = make_pushing_env(base_env, seed=seed)

        # Create MPC policy
        policy = PredictiveSamplingImprovisationalPolicy(
            pushing_env,
            seed=seed,
            config=PredictiveSamplingHyperparameters(
                horizon=self._config.horizon,
                num_rollouts=self._config.num_rollouts,
                noise_scale=self._config.noise_scale,
                num_control_points=self._config.num_control_points,
                dt=self._config.dt,
            ),
            warm_start=self._config.warm_start,
        )

        # Initialize base class
        super().__init__(
            observation_space, action_space, seed, policy, planner_id, domain_name
        )

        self.env_name = "blocks2d"

        # Initialize planning components
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
        assert len(applicable_skills) == 1, "Multiple skills per operator not supported"
        return applicable_skills[0]
