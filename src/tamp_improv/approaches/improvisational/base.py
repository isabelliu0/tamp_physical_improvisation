"""Base improvisational TAMP approach."""

from typing import Any

from relational_structs import GroundAtom, GroundOperator, Object, PDDLProblem
from relational_structs.utils import parse_pddl_plan
from task_then_motion_planning.planning import TaskThenMotionPlanningFailure
from task_then_motion_planning.structs import Skill
from tomsutils.pddl_planning import run_pddl_planner

from tamp_improv.approaches.base import (
    ActType,
    ApproachStepResult,
    BaseApproach,
    ImprovisationalTAMPSystem,
    ObsType,
)
from tamp_improv.approaches.improvisational.graph import (
    PlanningGraph,
    PlanningGraphEdge,
)
from tamp_improv.approaches.improvisational.policies.base import Policy, PolicyContext


class ImprovisationalTAMPApproach(BaseApproach[ObsType, ActType]):
    """General improvisational TAMP approach.

    This approach combines task-and-motion planning with learned
    policies for creating shortcuts between non-adjacent nodes in the
    plan.
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

        # Get domain
        self.domain = system.get_domain()

        # Initialize planning state
        self._current_task_plan: list[GroundOperator] = []
        self._current_operator: GroundOperator | None = None
        self._current_skill: Skill | None = None
        self._goal: set[GroundAtom] = set()

        # Graph-based planning state
        self._planning_graph: PlanningGraph | None = None
        self._current_path: list[PlanningGraphEdge] = []
        self._current_edge: PlanningGraphEdge | None = None
        self._current_preimage: set[GroundAtom] = set()
        self.policy_active = False

    def reset(self, obs: ObsType, info: dict[str, Any]) -> ApproachStepResult[ActType]:
        """Reset approach with initial observation."""
        objects, atoms, goal = self.system.perceiver.reset(obs, info)
        self._goal = goal

        # Create initial plan
        self._current_task_plan = self._create_task_plan(objects, atoms, goal)

        # Create planning graph
        self._planning_graph = self._create_planning_graph(
            objects, atoms, self._current_task_plan
        )

        # Compute preimages
        if self._planning_graph:
            self._planning_graph.compute_preimages()

            # Try to add shortcuts (initially just for pushing)
            self._try_add_shortcuts(self._planning_graph)

            # Find shortest path
            self._current_path = self._planning_graph.find_shortest_path()
        else:
            self._current_path = []

        # Reset state
        self._current_operator = None
        self._current_skill = None
        self._current_edge = None
        self._current_preimage = set()
        self.policy_active = False

        return self.step(obs, 0.0, False, False, info)

    def step(
        self,
        obs: ObsType,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ) -> ApproachStepResult[ActType]:
        """Step approach with new observation."""
        atoms = self.system.perceiver.step(obs)

        # Check if policy achieved its goal
        if self.policy_active and self._planning_graph:
            current_node = self._current_edge.source if self._current_edge else None
            if current_node and self._current_preimage.issubset(atoms):
                print("Policy successfully achieved preimage!")
                self.policy_active = False
                self._current_preimage = set()
                return self.step(obs, reward, terminated, truncated, info)
            return ApproachStepResult(action=self.policy.get_action(obs))

        # Get next edge if needed
        if not self._current_edge and self._current_path:
            self._current_edge = self._current_path.pop(0)

            if self._current_edge.is_shortcut and self._planning_graph:
                print("Using shortcut edge")
                self.policy_active = True

                # Get preimage for the target node
                target_node = self._current_edge.target
                if target_node in self._planning_graph.preimages:
                    self._current_preimage = self._planning_graph.preimages[target_node]
                else:
                    # Fallback to target node atoms if preimage not found
                    print(
                        "Preimage not found for target node, using target node atoms..."
                    )
                    self._current_preimage = set(target_node.atoms)

                # Configure policy with new context
                self.policy.configure_context(
                    PolicyContext(
                        preimage=self._current_preimage,
                        current_atoms=atoms,
                    )
                )

                # If in training mode, collect this state and terminate
                if self.training_mode:
                    return ApproachStepResult(
                        action=self.policy.get_action(obs),
                        terminate=True,
                        info={
                            "training_state": obs,
                            "current_atoms": atoms,
                            "preimage": self._current_preimage,
                        },
                    )

                return ApproachStepResult(action=self.policy.get_action(obs))

            # Regular edge - use operator skill
            self._current_operator = self._current_edge.operator

            if not self._current_operator:
                raise TaskThenMotionPlanningFailure("Edge has no operator")

            # Get skill for the operator
            self._current_skill = self._get_skill(self._current_operator)
            self._current_skill.reset(self._current_operator)

        # Check if current edge's target state is achieved
        if self._current_edge and set(self._current_edge.target.atoms).issubset(atoms):
            print("Edge target achieved")
            self._current_edge = None
            return self.step(obs, reward, terminated, truncated, info)

        # Execute current skill
        if not self._current_skill:
            raise TaskThenMotionPlanningFailure("No current skill")

        return ApproachStepResult(action=self._current_skill.get_action(obs))

    def _create_task_plan(
        self,
        objects: set[Object],
        init_atoms: set[GroundAtom],
        goal: set[GroundAtom],
    ) -> list[GroundOperator]:
        """Create task plan to achieve goal."""
        problem = PDDLProblem(
            self.domain.name, self.domain.name, objects, init_atoms, goal
        )
        plan_str = run_pddl_planner(
            str(self.domain), str(problem), planner=self.planner_id
        )
        if plan_str is None:
            raise TaskThenMotionPlanningFailure("No plan found")
        return parse_pddl_plan(plan_str, self.domain, problem)

    def _get_skill(self, operator: GroundOperator) -> Skill:
        """Get skill that can execute operator."""
        skills = [s for s in self.system.skills if s.can_execute(operator)]
        if not skills:
            raise TaskThenMotionPlanningFailure(
                f"No skill found for operator {operator.name}"
            )
        return skills[0]

    def _create_planning_graph(
        self,
        _objects: set[Object],
        init_atoms: set[GroundAtom],
        task_plan: list[GroundOperator],
    ) -> PlanningGraph:
        """Create planning graph from task plan."""
        graph = PlanningGraph()

        current_atoms = init_atoms.copy()
        current_node = graph.add_node(current_atoms, 0)

        # Create nodes and edges for each step in the plan
        for i, operator in enumerate(task_plan):
            # Apply operator effects to get new state
            next_atoms = current_atoms.copy()
            next_atoms.difference_update(operator.delete_effects)
            next_atoms.update(operator.add_effects)

            next_node = graph.add_node(next_atoms, i + 1)
            graph.add_edge(current_node, next_node, operator, cost=1.0)

            current_atoms = next_atoms
            current_node = next_node

        return graph

    def _try_add_shortcuts(self, graph: PlanningGraph) -> None:
        """Try to add shortcut edges to the graph.

        For now, manually detect pushing opportunity in the blocks2d
        scenario.
        """
        if not graph.nodes or len(graph.nodes) < 3:
            print("Graph too small for shortcuts")
            return

        # For the initial version, we'll just add the shortcut for pushing block 2
        # Start --> Push block 2 out of target area --> Pick up block 1

        # Find initial node
        initial_node = min(graph.nodes, key=lambda n: n.index)

        # Target node should be the node right before "Pick up block 1" operation
        # In our 5-step plan, it should be node with index 2
        # (after "Put down block 2 on table" but before "Pick up block 1")
        target_node_candidates = [n for n in graph.nodes if n.index == 2]

        if not target_node_candidates:
            print("No suitable target node found for shortcut")
            return

        target_node = target_node_candidates[0]

        # Check if this is a potential pushing scenario
        # In new predicate structure:
        # - Initial state: target_area is NOT clear (block 2 is blocking it)
        # - Target state: target_area IS clear (block 2 has been moved)

        # Find target_area object
        target_area_atom = next(
            (
                atom
                for atom in target_node.atoms
                if atom.predicate.name == "Clear"
                and len(atom.objects) == 1
                and atom.objects[0].name == "target_area"
            ),
            None,
        )

        if not target_area_atom:
            print("Could not find target_area object in node atoms")
            return

        initial_atoms_str = str(initial_node.atoms)
        target_atoms_str = str(target_node.atoms)
        print(f"Initial node atoms: {initial_atoms_str}")
        print(f"Target node atoms: {target_atoms_str}")

        # Check if target_area is not clear in initial state but is clear in target state
        target_clear_in_initial = any(
            atom.predicate.name == "Clear" and atom.objects[0].name == "target_area"
            for atom in initial_node.atoms
        )

        target_clear_in_target = any(
            atom.predicate.name == "Clear" and atom.objects[0].name == "target_area"
            for atom in target_node.atoms
        )

        # Block 2 should be on table in target state (moved from target area)
        block2_on_table_in_target = any(
            atom.predicate.name == "On"
            and atom.objects[0].name == "block2"
            and atom.objects[1].name == "table"
            for atom in target_node.atoms
        )

        if (
            (not target_clear_in_initial)
            and target_clear_in_target
            and block2_on_table_in_target
        ):
            print(
                f"Adding pushing shortcut: {initial_node.index} to {target_node.index}"
            )
            # Add shortcut with lower cost (0.5 * standard path length)
            cost = 0.5 * (target_node.index - initial_node.index)
            graph.add_edge(initial_node, target_node, None, cost, is_shortcut=True)

            # Update preimages if they've been computed
            if graph.preimages:
                # Make sure initial node has the preimage for target node
                if target_node in graph.preimages:
                    if initial_node in graph.preimages:
                        graph.preimages[initial_node].update(
                            graph.preimages[target_node]
                        )
                    else:
                        graph.preimages[initial_node] = graph.preimages[
                            target_node
                        ].copy()
        else:
            print("No pushing shortcut opportunity detected:")
            print(f"  - Target clear in initial node: {target_clear_in_initial}")
            print(f"  - Target clear in target node: {target_clear_in_target}")
            print(f"  - Block2 on table in target node: {block2_on_table_in_target}")
