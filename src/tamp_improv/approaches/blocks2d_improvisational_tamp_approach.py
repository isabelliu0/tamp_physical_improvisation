"""Blocks2D-specific implementation of improvisational TAMP approach."""

import math

import numpy as np
from numpy.typing import NDArray
from relational_structs import (
    GroundOperator,
    PDDLDomain,
)
from task_then_motion_planning.planning import TaskThenMotionPlanningFailure
from task_then_motion_planning.structs import Perceiver, Skill, _Action, _Observation

from tamp_improv.approaches.base_improvisational_tamp_approach import (
    ImprovisationalPolicy,
    ImprovisationalTAMPApproach,
)
from tamp_improv.blocks2d_planning import create_blocks2d_planning_models


class Blocks2DImprovisationalPolicy(
    ImprovisationalPolicy[NDArray[np.float32], NDArray[np.float32]]
):
    """Policy for handling situations where TAMP preconditions aren't met in
    Blocks2D.

    The observation space is 15D:
        [0:2] - robot x,y position
        [2:4] - robot width, height
        [4:6] - block 1 (target block) x,y position
        [6:8] - block 2 (obstacle block) x,y position
        [8:10] - block width, height
        [10] - gripper status
        [11:15] - target area x,y,width,height

        The action space is 3D:
        [0:2] - robot dx, dy movement
        [2] - gripper activation
    """

    def _get_push_direction(
        self,
        block_1_x: float,
        block_2_x: float,
        target_x: float,
        block_width: float,
        target_width: float,
    ) -> float:
        left_margin = (
            (block_2_x - block_width / 2) - (target_x - target_width / 2) + 0.1
        )  # Add a small margin
        right_margin = (
            (target_x + target_width / 2) - (block_2_x + block_width / 2) + 0.1
        )

        if block_1_x < block_2_x:
            left_margin *= 2  # Discourage pushing left if block_1 is on the left
        else:
            right_margin *= 2

        if left_margin < right_margin:
            return -0.1  # Push left
        return 0.1  # Push right

    def get_action(self, obs: NDArray[np.float32]) -> NDArray[np.float32]:
        """Get action from policy based on current observation."""
        (
            robot_x,
            robot_y,
            robot_width,
            robot_height,
            block_1_x,
            _,
            block_2_x,
            block_2_y,
            block_width,
            block_height,
            _,
            target_x,
            _,
            target_width,
            _,
        ) = obs

        # Determine the best direction to push block 2
        push_direction = self._get_push_direction(
            block_1_x, block_2_x, target_x, block_width, target_width
        )

        # Calculate distances
        distance = np.linalg.norm([robot_x - block_2_x, robot_y - block_2_y])
        vertical_distance = np.abs(robot_y - block_2_y)

        if distance > ((robot_width + block_width) / 2) * math.sqrt(2):
            dx = np.clip(
                block_2_x - robot_x + (robot_width + block_width) / 2, -0.1, 0.1
            )
            dy = np.clip(
                block_2_y - robot_y + (robot_height + block_height) / 2, -0.1, 0.1
            )
            return np.array([dx, dy, 0.0])
        if vertical_distance > 0.01:
            # Move towards the y-level of the block
            dy = np.clip(block_2_y - robot_y, -0.1, 0.1)
            return np.array([0.0, dy, 0.0])
        # Push the block horizontally
        dx = np.clip(push_direction, -0.1, 0.1)
        return np.array([dx, 0.0, 0.0])


class Blocks2DImprovisationalTAMPApproach(
    ImprovisationalTAMPApproach[NDArray[np.float32], NDArray[np.float32]]
):
    """Blocks2D-specific improvisational TAMP approach."""

    def __init__(
        self,
        observation_space,
        action_space,
        seed: int,
        planner_id: str = "pyperplan",
        domain_name: str = "custom-domain",
    ) -> None:
        policy = Blocks2DImprovisationalPolicy()

        # Initialize base class
        super().__init__(
            observation_space, action_space, seed, policy, planner_id, domain_name
        )

        self.env_name = "blocks2d"

        # Initialize Blocks2D-specific components
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
        applicable_skills = [s for s in self._skills if s.can_execute(operator)]
        if not applicable_skills:
            raise TaskThenMotionPlanningFailure("No skill can execute operator")
        assert len(applicable_skills) == 1, "Multiple operators per skill not supported"
        return applicable_skills[0]
