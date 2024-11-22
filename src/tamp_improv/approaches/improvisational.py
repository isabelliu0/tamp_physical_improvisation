"""Core improvisational TAMP approach."""

from typing import Any

from relational_structs import GroundAtom, GroundOperator, Object, PDDLProblem
from relational_structs.utils import parse_pddl_plan
from task_then_motion_planning.planning import TaskThenMotionPlanningFailure
from task_then_motion_planning.structs import Skill
from tomsutils.pddl_planning import run_pddl_planner

from tamp_improv.approaches.base import (
    ActType,
    BaseApproach,
    ImprovisationalPolicy,
    ObsType,
)


class ImprovisationalTAMPApproach(BaseApproach[ObsType, ActType]):
    """Base class for improvisational TAMP approaches.

    This approach combines task-and-motion planning with learned
    policies for handling situations where operator preconditions aren't
    met.
    """

    def __init__(
        self,
        system,
        policy: ImprovisationalPolicy[ObsType, ActType],
        seed: int,
        planner_id: str = "pyperplan",
    ) -> None:
        """Initialize approach.

        Args:
            system: The TAMP system to use
            policy: The improvisational policy to use
            seed: Random seed
            planner_id: ID of PDDL planner to use
        """
        super().__init__(system, seed)
        self.policy = policy
        self.planner_id = planner_id

        # Initialize planning state
        self._current_task_plan: list[GroundOperator] = []
        self._current_operator: GroundOperator | None = None
        self._current_skill: Skill | None = None
        self._policy_active = False
        self._target_atoms: set[GroundAtom] = set()
        self._goal: set[GroundAtom] = set()

    def reset(self, obs: ObsType, info: dict[str, Any]) -> ActType:
        """Reset approach with initial observation."""
        # Get initial state
        objects, atoms, goal = self.system.perceiver.reset(obs, info)
        self._goal = goal

        # Create initial plan
        self._current_task_plan = self._create_task_plan(objects, atoms, self._goal)
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
        """Step approach with new observation."""
        # Get current state
        atoms = self.system.perceiver.step(obs)

        # Check if policy achieved its goal
        if self._policy_active:
            if self._target_atoms.issubset(atoms):
                print("Policy successfully achieved target atoms!")
                self._policy_active = False
                self._target_atoms = set()
                self._replan(obs, info)  # replan from current state
            else:
                return self.policy.get_action(obs)

        # Get new operator if needed
        if self._current_operator is None or self._operator_completed(
            self._current_operator, atoms
        ):
            if not self._current_task_plan:
                raise TaskThenMotionPlanningFailure("Empty task plan")

            self._current_operator = self._current_task_plan.pop(0)

            # Check preconditions
            if not self._check_preconditions(self._current_operator, atoms):
                self._policy_active = True
                self._target_atoms = self._get_full_preconditions(
                    self._current_operator
                )
                return self.policy.get_action(obs)

            # Get skill for operator
            self._current_skill = self._get_skill(self._current_operator)
            self._current_skill.reset(self._current_operator)

        assert self._current_skill is not None
        return self._current_skill.get_action(obs)

    def _create_task_plan(
        self,
        objects: set[Object],
        init_atoms: set[GroundAtom],
        goal: set[GroundAtom],
    ) -> list[GroundOperator]:
        """Create task plan to achieve goal."""
        domain = self.system.get_domain()
        problem = PDDLProblem(domain.name, domain.name, objects, init_atoms, goal)
        plan_str = run_pddl_planner(str(domain), str(problem), planner=self.planner_id)
        if plan_str is None:
            raise TaskThenMotionPlanningFailure("No plan found")
        return parse_pddl_plan(plan_str, domain, problem)

    def _check_preconditions(
        self, operator: GroundOperator, atoms: set[GroundAtom]
    ) -> bool:
        """Check if operator preconditions are satisfied."""
        full_preconditions = self._get_full_preconditions(operator)
        return full_preconditions.issubset(atoms)

    def _get_full_preconditions(self, operator: GroundOperator) -> set[GroundAtom]:
        """Get full preconditions for operator including improvisation
        cases."""
        # Override in subclasses if needed
        return operator.preconditions

    def _operator_completed(
        self, operator: GroundOperator, atoms: set[GroundAtom]
    ) -> bool:
        """Check if operator effects have been achieved."""
        return operator.add_effects.issubset(atoms) and not (
            operator.delete_effects & atoms
        )

    def _get_skill(self, operator: GroundOperator) -> Skill:
        """Get skill that can execute operator."""
        skills = [s for s in self.system.skills if s.can_execute(operator)]
        if not skills:
            raise TaskThenMotionPlanningFailure(
                f"No skill found for operator {operator.name}"
            )
        return skills[0]

    def _replan(self, obs: ObsType, info: dict[str, Any]) -> None:
        """Create new plan from current state."""
        objects, atoms, _ = self.system.perceiver.reset(obs, info)
        self._current_task_plan = self._create_task_plan(objects, atoms, self._goal)
        self._current_operator = None
        self._current_skill = None
