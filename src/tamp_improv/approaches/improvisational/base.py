"""Base improvisational TAMP approach."""

import itertools
from collections import deque
from typing import Any

from relational_structs import (
    GroundAtom,
    GroundOperator,
    LiftedOperator,
    Object,
    PDDLProblem,
)
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
    PlanningGraphNode,
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
        objects: set[Object],
        init_atoms: set[GroundAtom],
        _task_plan: list[GroundOperator] | None = None,  # For compatibility
    ) -> PlanningGraph:
        """Create a tree-based planning graph by exploring possible action
        sequences.

        This builds a graph representing multiple possible plans rather
        than just following a single task plan. This method is domain-
        agnostic.
        """
        graph = PlanningGraph()
        initial_node = graph.add_node(init_atoms, 0)
        visited_states = {frozenset(init_atoms): initial_node}
        queue = deque([(initial_node, 0)])  # Queue for BFS: [(node, depth)]
        node_count = 0
        max_nodes = 100
        print(f"Building planning graph with max {max_nodes} nodes...")

        # Breadth-first search to build the graph
        while queue and node_count < max_nodes:
            current_node, depth = queue.popleft()
            node_count += 1

            print(f"\n--- Node {node_count} at depth {depth} ---")
            print(f"Contains {len(current_node.atoms)} atoms")

            # Check if this is a goal state
            if self._goal and self._goal.issubset(current_node.atoms):
                print(f"Found goal state at depth {depth}, not expanding further...")
                continue

            # Find applicable ground operators using the domain's operators
            applicable_ops = self._find_applicable_operators(
                set(current_node.atoms), objects
            )
            print(f"  Found {len(applicable_ops)} applicable operators")

            # Apply each applicable operator to generate new states
            for i, op in enumerate(applicable_ops):
                print(f"\n  Applying operator {i+1}/{len(applicable_ops)}: {op.name}")
                print(f"    Parameters: {[p.name for p in op.parameters]}")

                # Apply operator effects to get next state
                next_atoms = set(current_node.atoms)
                next_atoms.difference_update(op.delete_effects)
                next_atoms.update(op.add_effects)

                # Summarize effects
                if op.delete_effects:
                    print(f"    Delete effects: {len(op.delete_effects)} atoms")
                if op.add_effects:
                    print(f"    Add effects: {len(op.add_effects)} atoms")

                # Check if we've seen this state before
                next_atoms_frozen = frozenset(next_atoms)
                if next_atoms_frozen in visited_states:
                    # Create edge to existing node
                    next_nodes = visited_states[next_atoms_frozen]
                    print(
                        f"State already visited (Node {next_node.index}), creating edge"
                    )
                    graph.add_edge(current_node, next_nodes, op, cost=1.0)
                else:
                    # Create new node and edge
                    next_node: PlanningGraphNode = graph.add_node(next_atoms, depth + 1)
                    print(f"    Created new Node {next_node.index}")
                    visited_states[next_atoms_frozen] = next_node
                    graph.add_edge(current_node, next_node, op, cost=1.0)
                    queue.append((next_node, depth + 1))

        print(
            f"Planning graph with {len(graph.nodes)} nodes and {len(graph.edges)} edges"
        )

        # Print detailed graph structure
        print("\nDetailed Graph Structure:")
        for node in sorted(graph.nodes, key=lambda n: n.index):
            print(f"\nNode {node.index}:")
            print("  Key atoms:")
            count = 0
            for atom in node.atoms:
                print(f"  - {atom}")
                count += 1

        print("\nGraph Edges:")
        for edge in graph.edges:
            op_str = f"{edge.operator.name}" if edge.operator else "SHORTCUT"
            print(
                f"  Node {edge.source.index} --[{op_str}]--> Node {edge.target.index}"
            )
        return graph

    def _find_applicable_operators(
        self, current_atoms: set[GroundAtom], objects: set[Object]
    ) -> list[GroundOperator]:
        """Find all ground operators that are applicable in the current
        state."""
        applicable_ops = []
        domain_operators = self.domain.operators
        print(
            f"Find applicable operators for {len(domain_operators)} domain operators..."
        )

        for lifted_op in domain_operators:
            # Find valid groundings using parameter types
            valid_groundings = self._find_valid_groundings(lifted_op, objects)
            print(
                f"Operator {lifted_op.name} has {len(valid_groundings)} valid groundings"
            )

            for grounding in valid_groundings:
                ground_op = lifted_op.ground(grounding)

                # Check if preconditions are satisfied
                if ground_op.preconditions.issubset(current_atoms):
                    applicable_ops.append(ground_op)

        return applicable_ops

    def _find_valid_groundings(
        self, lifted_op: LiftedOperator, objects: set[Object]
    ) -> list[tuple[Object, ...]]:
        """Find all valid groundings for a lifted operator.

        Args:
            lifted_op: The lifted operator to ground
            objects: Available objects in the domain

        Returns:
            List of valid parameter tuples for grounding
        """
        # Group objects by type
        objects_by_type: dict[Any, list[Object]] = {}
        for obj in objects:
            if obj.type not in objects_by_type:
                objects_by_type[obj.type] = []
            objects_by_type[obj.type].append(obj)

        # Print the parameter requirements for debugging
        param_types = []
        for param in lifted_op.parameters:
            param_types.append(f"{param.name} ({param.type.name})")
        print(f"    Parameters needed: {', '.join(param_types)}")

        # For each parameter, find objects of the right type
        param_objects = []
        for param in lifted_op.parameters:
            if param.type in objects_by_type:
                param_objects.append(objects_by_type[param.type])
                len_avail = len(objects_by_type[param.type])
                print(f"- For {param.name}: {len_avail} objects available")
            else:
                # If no objects of this type, operator can't be grounded
                print(
                    f"- For {param.name}: No objects of type {param.type.name} available"
                )
                return []

        # Generate all possible groundings
        groundings = list(itertools.product(*param_objects))

        return groundings

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
