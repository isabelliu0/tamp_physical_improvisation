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
        max_skill_steps: int = 1_000,
    ) -> None:
        """Initialize approach."""
        super().__init__(system, seed)
        self.policy = policy
        self.planner_id = planner_id
        self._max_skill_steps = max_skill_steps

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
        self.planning_graph: PlanningGraph | None = None
        self._current_path: list[PlanningGraphEdge] = []
        self._current_edge: PlanningGraphEdge | None = None
        self._current_preimage: set[GroundAtom] = set()
        self.policy_active = False
        self.observed_states: dict[int, Any] = {}  # Maps node_id -> state

    def reset(self, obs: ObsType, info: dict[str, Any]) -> ApproachStepResult[ActType]:
        """Reset approach with initial observation."""
        objects, atoms, goal = self.system.perceiver.reset(obs, info)
        self._goal = goal

        # Create initial plan
        self._current_task_plan = self._create_task_plan(objects, atoms, goal)

        # Create planning graph and store observed state for initial node
        self.planning_graph = self._create_planning_graph(
            objects, atoms, self._current_task_plan
        )
        if self.planning_graph:
            initial_node = next(iter(self.planning_graph.nodes))
            self.observed_states[initial_node.id] = obs

        # Compute preimages
        self.planning_graph.compute_preimages(goal)

        # Try to add shortcuts
        self._try_add_shortcuts(self.planning_graph)

        # Compute edge costs
        self._compute_planning_graph_edge_costs(obs, info)

        # Find shortest path
        self._current_path = self.planning_graph.find_shortest_path(atoms, goal)

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
        if self.policy_active and self.planning_graph:
            current_node = self._current_edge.source if self._current_edge else None
            if current_node and self._current_preimage.issubset(atoms):
                print("Policy successfully achieved preimage!")
                self.policy_active = False
                self._current_preimage = set()
                return self.step(obs, reward, terminated, truncated, info)
            return ApproachStepResult(action=self.policy.get_action(obs))

        # Get next edge if needed
        assert self.planning_graph is not None
        if not self._current_edge and self._current_path:
            self._current_edge = self._current_path.pop(0)

            if self._current_edge.is_shortcut:
                print("Using shortcut edge")
                self.policy_active = True

                # Get preimage for the target node
                target_node = self._current_edge.target
                if target_node in self.planning_graph.preimages:
                    self._current_preimage = self.planning_graph.preimages[target_node]
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

            # Store observed state for target node
            target_node = self._current_edge.target
            self.observed_states[target_node.id] = obs

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
        initial_node = graph.add_node(init_atoms)
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
                    next_node = visited_states[next_atoms_frozen]
                    print(f"State already visited (Node {next_node.id}), creating edge")
                    graph.add_edge(current_node, next_node, op)
                else:
                    # Create new node and edge
                    next_node = graph.add_node(next_atoms)
                    print(f"    Created new Node {next_node.id}")
                    visited_states[next_atoms_frozen] = next_node
                    graph.add_edge(current_node, next_node, op)
                    queue.append((next_node, depth + 1))

        print(
            f"Planning graph with {len(graph.nodes)} nodes and {len(graph.edges)} edges"
        )

        # Print detailed graph structure
        print("\nDetailed Graph Structure:")
        for node in sorted(graph.nodes, key=lambda n: n.id):
            print(f"\nNode {node.id}:")
            print("  Key atoms:")
            count = 0
            for atom in node.atoms:
                print(f"  - {atom}")
                count += 1

        print("\nGraph Edges:")
        for edge in graph.edges:
            op_str = f"{edge.operator.name}" if edge.operator else "SHORTCUT"
            print(f"  Node {edge.source.id} --[{op_str}]--> Node {edge.target.id}")
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
        """Try to add shortcut edges to the graph."""

        # Consider all pairs of initial nodes and preimages and check if the
        # policy can be initiated given that context.
        for source_node in graph.nodes:
            for target_node in graph.nodes:
                if source_node == target_node:
                    continue
                # Don't bother trying to take a shortcut if we already have an
                # operator for source -> target.
                if target_node in graph.node_to_outgoing_edges[source_node]:
                    continue
                self.policy.configure_context(
                    PolicyContext(
                        preimage=graph.preimages[target_node],
                        current_atoms=set(source_node.atoms),
                    )
                )
                if self.policy.can_initiate():
                    print(f"Adding shortcut: {source_node.id} to {target_node.id}")
                    graph.add_edge(source_node, target_node, None, is_shortcut=True)

    def _compute_planning_graph_edge_costs(
        self, obs: ObsType, info: dict[str, Any]
    ) -> None:
        """Add edge costs to the current planning graph."""

        assert self.planning_graph is not None

        _, init_atoms, _ = self.system.perceiver.reset(obs, info)
        initial_node = self.planning_graph.node_map[frozenset(init_atoms)]
        node_to_obs_info: dict[PlanningGraphNode, tuple[ObsType, dict]] = {}
        node_to_obs_info[initial_node] = (obs, info)

        sim = self.system.wrapped_env  # issue #31

        queue = [initial_node]
        while queue:
            node = queue.pop()
            for edge in self.planning_graph.node_to_outgoing_edges[node]:
                obs, info = node_to_obs_info[node]
                sim.reset_from_state(obs)  # type: ignore
                _, init_atoms, _ = self.system.perceiver.reset(obs, info)
                preimage = self.planning_graph.preimages[edge.target]
                if edge.is_shortcut:
                    self.policy.configure_context(
                        PolicyContext(
                            preimage=preimage,
                            current_atoms=init_atoms,
                        )
                    )
                    skill: Policy | Skill = self.policy
                else:
                    assert edge.operator is not None
                    skill = self._get_skill(edge.operator)
                    skill.reset(edge.operator)
                num_steps = 0
                for _ in range(self._max_skill_steps):
                    act = skill.get_action(obs)
                    obs, _, _, _, info = sim.step(act)
                    num_steps += 1
                    atoms = self.system.perceiver.step(obs)
                    if preimage.issubset(atoms):
                        break  # success
                else:
                    # Edge expansion failed.
                    continue
                assert edge.cost == float("inf")
                edge.cost = num_steps
                # NOTE: we are making the strong assumption that it does not
                # matter which low-level state you are in within the abstract
                # state, so if there are multiple observations for a node, we
                # just keep one arbitrarily.
                if edge.target not in node_to_obs_info:
                    node_to_obs_info[edge.target] = (obs, info)
                    queue.append(edge.target)

    def get_shortcut_distance(self, source_node, target_node):
        """Compute the distance (in regular steps) between two nodes."""
        # Quick check: if there's a direct edge, distance is 1
        for edge in self.planning_graph.node_to_outgoing_edges.get(source_node, []):
            if edge.target == target_node and not edge.is_shortcut:
                return 1

        # BFS to find shortest path
        queue = deque([(source_node, 0)])
        visited = {source_node}

        while queue:
            current, distance = queue.popleft()

            for edge in self.planning_graph.node_to_outgoing_edges.get(current, []):
                if edge.is_shortcut:
                    continue

                next_node = edge.target

                # Found path to target
                if next_node == target_node:
                    return distance + 1

                # Add to queue if not visited
                if next_node not in visited:
                    visited.add(next_node)
                    queue.append((next_node, distance + 1))

        # No path found
        return float("inf")
