"""Base improvisational TAMP approach."""

from typing import Any

from relational_structs import GroundAtom, GroundOperator, Object, PDDLProblem
from relational_structs.utils import parse_pddl_plan
from task_then_motion_planning.planning import TaskThenMotionPlanningFailure
from task_then_motion_planning.structs import Skill
from tomsutils.pddl_planning import run_pddl_planner

from tamp_improv.approaches.base import (
    ActType,
    BaseApproach,
    ImprovisationalTAMPSystem,
    ObsType,
)
from tamp_improv.approaches.improvisational.policies.base import Policy


class ImprovisationalTAMPApproach(BaseApproach[ObsType, ActType]):
    """General improvisational TAMP approach.

    This approach combines task-and-motion planning with learned
    policies for handling situations where operator preconditions aren't
    met.
    """

    def __init__(
        self,
        system: ImprovisationalTAMPSystem[ObsType, ActType],
        policy: Policy[ObsType, ActType],
        seed: int,
        planner_id: str = "pyperplan",
    ) -> None:
        """Initialize approach."""
        super().__init__(system, seed)
        self.policy = policy
        self.planner_id = planner_id

        # Initialize policy with wrapped environment
        policy.initialize(system.wrapped_env)

        # Get base and full domains
        self.base_domain = system.get_domain(include_extra_preconditions=False)
        self.full_domain = system.get_domain(include_extra_preconditions=True)

        # Map operators
        self._operator_to_full = {}
        for base_op in self.base_domain.operators:
            for full_op in self.full_domain.operators:
                if base_op.name == full_op.name:
                    self._operator_to_full[base_op] = full_op

        # Initialize planning state
        self._current_task_plan: list[GroundOperator] = []
        self._current_operator: GroundOperator | None = None
        self._current_skill: Skill | None = None
        self._policy_active = False
        self._target_atoms: set[GroundAtom] = set()
        self._currently_satisfied: set[GroundAtom] = set()
        self._goal: set[GroundAtom] = set()

    def reset(self, obs: ObsType, info: dict[str, Any]) -> ActType:
        """Reset approach with initial observation."""
        objects, atoms, goal = self.system.perceiver.reset(obs, info)
        self._goal = goal

        # Create initial plan
        self._current_task_plan = self._create_task_plan(objects, atoms, goal)
        self._current_operator = None
        self._current_skill = None
        self._policy_active = False
        self._target_atoms = set()
        self._currently_satisfied = set()

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
        atoms = self.system.perceiver.step(obs)

        # Check if policy achieved its goal
        if self._policy_active:
            if self._target_atoms.issubset(
                atoms
            ) and self._currently_satisfied.issubset(atoms):
                print("Policy successfully achieved target atoms!")
                print(f"Current atoms before replan: {atoms}")
                self._policy_active = False
                self._target_atoms = set()
                self._currently_satisfied = set()
                self._replan(obs, info)
                print(f"After replan - task plan: {self._current_task_plan}")
                print(
                    f"After replan - first operator: {self._current_task_plan[0] if self._current_task_plan else None}"
                )
                return self.step(obs, reward, terminated, truncated, info)
            return self.policy.get_action(obs)

        # Get new operator if needed
        if self._current_operator is None or self._operator_completed(
            self._current_operator, atoms
        ):
            if not self._current_task_plan:
                raise TaskThenMotionPlanningFailure("Empty task plan")

            # Get next operator from base domain plan
            self._current_operator = self._current_task_plan.pop(0)
            print(f"\nAttempting to execute operator: {self._current_operator}")
            print(f"Operator type: {self._current_operator.parent}")

            # Check full preconditions
            full_op = self._operator_to_full[self._current_operator.parent]
            full_ground_op = full_op.ground(tuple(self._current_operator.parameters))
            print(f"Full operator: {full_ground_op}")
            print(f"Full preconditions: {full_ground_op.preconditions}")

            # Check preconditions
            if not full_ground_op.preconditions.issubset(atoms):
                print("Preconditions not met, activating policy")
                self._policy_active = True

                # Track satisfied vs target preconditions
                self._currently_satisfied = full_ground_op.preconditions & atoms
                self._target_atoms = full_ground_op.preconditions - atoms

                # Configure env with preconditions to maintain
                if hasattr(self.system.wrapped_env, "update_preconditions"):
                    self.system.wrapped_env.update_preconditions(
                        full_op, self._currently_satisfied
                    )
                return self.policy.get_action(obs)

            # Get skill for the operator from the base domain
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
        problem = PDDLProblem(
            self.base_domain.name, self.base_domain.name, objects, init_atoms, goal
        )
        plan_str = run_pddl_planner(
            str(self.base_domain), str(problem), planner=self.planner_id
        )
        if plan_str is None:
            raise TaskThenMotionPlanningFailure("No plan found")
        return parse_pddl_plan(plan_str, self.base_domain, problem)

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
        print(f"Available skills: {self.system.skills}")
        print(f"Operator to find skill for: {operator}")
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