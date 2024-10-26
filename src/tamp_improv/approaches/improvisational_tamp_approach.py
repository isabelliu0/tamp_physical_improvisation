"""Improvisational TAMP approach."""

import math
from typing import Any, Set

import numpy as np
from numpy.typing import NDArray
from relational_structs import (
    GroundAtom,
    GroundOperator,
    LiftedOperator,
    PDDLDomain,
    PDDLProblem,
)
from relational_structs.utils import parse_pddl_plan
from task_then_motion_planning.planning import TaskThenMotionPlanningFailure
from task_then_motion_planning.structs import Skill, _Action, _Observation
from tomsutils.pddl_planning import run_pddl_planner

from tamp_improv.approaches.base_approach import BaseApproach
from tamp_improv.blocks2d_planning import create_blocks2d_planning_models


class ImprovisationalPolicy:
    """Policy for handling situations where TAMP preconditions aren't met."""

    def __init__(self) -> None:
        """Initialize policy.

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
        """Get action from policy based on current observation.

        Implements the same pushing behavior as ClearTargetAreaSkill
        from blocks2d_planning.py.
        """
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


class ImprovisationalTAMPApproach(
    BaseApproach[NDArray[np.float32], NDArray[np.float32]]
):
    """Improvisational TAMP approach."""

    def __init__(
        self,
        observation_space,
        action_space,
        seed: int,
        planner_id: str = "pyperplan",
        domain_name: str = "custom-domain",
    ) -> None:
        super().__init__(observation_space, action_space, seed)
        self.env_name = "blocks2d"

        types, predicates, _, operators, skills = create_blocks2d_planning_models(
            include_pushing_models=False
        )

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
        _, _, perceiver, full_precondition_operators, _ = (
            create_blocks2d_planning_models(include_pushing_models=True)
        )
        self._operator_to_full_precondition_operator: dict[
            LiftedOperator, LiftedOperator
        ] = {}
        operator_name_to_operator = {o.name: o for o in self._operators}
        for full_precondition_operator in full_precondition_operators:
            if full_precondition_operator.name not in operator_name_to_operator:
                continue
            operator = operator_name_to_operator[full_precondition_operator.name]
            self._operator_to_full_precondition_operator[operator] = (
                full_precondition_operator
            )
        self._perceiver = perceiver

        # Initialize planning state.
        self._current_problem: PDDLProblem | None = None
        self._current_task_plan: list[GroundOperator] = []
        self._current_operator: GroundOperator | None = None
        self._current_skill: Skill | None = None

        # Initialize RL components.
        self._policy = ImprovisationalPolicy()
        self._policy_active: bool = False
        self._target_atoms: Set[GroundAtom] = (
            set()
        )  # Atoms that the policy should try to achieve
        self._goal: Set[GroundAtom] = set()

    def _replan(self, obs: NDArray[np.float32], info: dict[str, Any]) -> None:
        """Create a new plan from the current state."""
        objects, atoms, _ = self._perceiver.reset(
            obs, info
        )  # Get current objects and atoms
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

    def reset(
        self, obs: NDArray[np.float32], info: dict[str, Any]
    ) -> NDArray[np.float32]:
        objects, atoms, goal = self._perceiver.reset(obs, info)
        self._goal = goal  # Store goal for later replanning
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

        # Get the first action
        first_action = self.step(obs, 0.0, False, False, info)
        return first_action

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
                # The preconditions don't hold, so we need to call policy!
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
        applicable_skills = [s for s in self._skills if s.can_execute(operator)]
        if not applicable_skills:
            raise TaskThenMotionPlanningFailure("No skill can execute operator")
        assert len(applicable_skills) == 1, "Multiple operators per skill not supported"
        return applicable_skills[0]
