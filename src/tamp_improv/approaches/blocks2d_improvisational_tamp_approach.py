"""Blocks2D-specific implementation of improvisational TAMP approach."""

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

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        (
            robot_x,
            robot_y,
            robot_width,
            _,
            block_1_x,
            _,
            block_2_x,
            block_2_y,
            block_width,
            _,
            _,
            target_x,
            _,
            target_width,
            _,
        ) = obs

        # First, determine in which direction to move block 2
        move_direction = self._get_movement_direction(
            block_1_x,
            target_x,
            block_width,
            target_width,
        )

        # Calculate target position to push from
        target_x_offset = (robot_width + block_width) / 2
        target_x_offset *= -np.sign(move_direction)
        target_robot_x = block_2_x + target_x_offset

        # Calculate distances
        dist_to_target = np.hypot(target_robot_x - robot_x, block_2_y - robot_y)

        if dist_to_target > 0.1:  # If we're far from pushing position
            # Move towards the target position using a combined motion
            dx = np.clip(target_robot_x - robot_x, -0.1, 0.1)
            dy = np.clip(block_2_y - robot_y, -0.1, 0.1)
            return np.array([dx, dy, 1.0])

        # We're in position to push
        return np.array([np.clip(move_direction, -0.1, 0.1), 0.0, 1.0])

    def _get_movement_direction(
        self,
        block_1_x: float,
        target_x: float,
        block_width: float,
        target_width: float,
    ) -> float:
        """Determines the direction to move block 2.

        Returns:
            float: -0.1 for left, 0.1 for right
        """
        # Check if there's enough space on either side
        space_on_left = (
            abs((target_x - target_width / 2) - (block_1_x + block_width / 2))
            if block_1_x < target_x
            else abs(target_x - target_width / 2)
        )
        space_on_right = (
            abs((block_1_x - block_width / 2) - (target_x + target_width / 2))
            if block_1_x > target_x
            else abs(float(1.0) - (target_x + target_width / 2))
        )

        # Move in direction with more space
        if space_on_left > space_on_right:
            return -0.1  # Push left
        return 0.1  # Push right


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
