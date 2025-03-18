"""Base improvisational TAMP approach."""

from __future__ import annotations

import itertools
from collections import deque
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
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
from tamp_improv.benchmarks.context_wrapper import ContextAwareWrapper


@dataclass
class ShortcutSignature:
    """Domain-agnostic signature of a shortcut for matching purposes."""

    source_predicates: set[str]
    target_predicates: set[str]
    source_types: set[str]
    target_types: set[str]

    @classmethod
    def from_context(
        cls, source_atoms: set[GroundAtom], target_preimage: set[GroundAtom]
    ) -> ShortcutSignature:
        """Create signature from context."""
        source_preds = {atom.predicate.name for atom in source_atoms}
        target_preds = {atom.predicate.name for atom in target_preimage}

        source_types = set()
        for atom in source_atoms:
            for obj in atom.objects:
                source_types.add(obj.type.name)

        target_types = set()
        for atom in target_preimage:
            for obj in atom.objects:
                target_types.add(obj.type.name)

        return cls(source_preds, target_preds, source_types, target_types)

    def similarity(self, other: ShortcutSignature) -> float:
        """Calculate similarity score between signatures."""
        # Predicate similarity (Jaccard)
        source_pred_sim = len(self.source_predicates & other.source_predicates) / max(
            len(self.source_predicates | other.source_predicates), 1
        )
        target_pred_sim = len(self.target_predicates & other.target_predicates) / max(
            len(self.target_predicates | other.target_predicates), 1
        )

        # Object type similarity (Jaccard)
        source_type_sim = len(self.source_types & other.source_types) / max(
            len(self.source_types | other.source_types), 1
        )
        target_type_sim = len(self.target_types & other.target_types) / max(
            len(self.target_types | other.target_types), 1
        )

        # Overall similarity - weighted average
        return (
            0.3 * source_pred_sim
            + 0.3 * target_pred_sim
            + 0.2 * source_type_sim
            + 0.2 * target_type_sim
        )


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
        max_preimage_size: int = 11,
    ) -> None:
        """Initialize approach."""
        super().__init__(system, seed)
        self.policy = policy
        self.planner_id = planner_id
        self._max_skill_steps = max_skill_steps

        # Create context-aware wrapper
        if not isinstance(system.wrapped_env, ContextAwareWrapper):
            self.context_env = ContextAwareWrapper(
                system.wrapped_env,
                system.perceiver,
                max_preimage_size=max_preimage_size,
            )
            system.wrapped_env = self.context_env
        else:
            self.context_env = system.wrapped_env

        # Initialize policy with (context-aware) wrapped environment
        policy.initialize(self.context_env)

        # Get domain
        self.domain = system.get_domain()

        # Initialize planning state
        self._current_operator: GroundOperator | None = None
        self._current_skill: Skill | None = None
        self._goal: set[GroundAtom] = set()

        # Graph-based planning state
        self.planning_graph: PlanningGraph | None = None
        self._current_path: list[PlanningGraphEdge] = []
        self._current_edge: PlanningGraphEdge | None = None
        self._current_preimage: set[GroundAtom] = set()
        self.policy_active = False

        # Shortcut signatures for similarity matching
        self.trained_signatures: list[ShortcutSignature] = []

    def reset(self, obs: ObsType, info: dict[str, Any]) -> ApproachStepResult[ActType]:
        """Reset approach with initial observation."""
        objects, atoms, goal = self.system.perceiver.reset(obs, info)
        self._goal = goal

        # Create planning graph
        self.planning_graph = self._create_planning_graph(objects, atoms)

        # Compute preimages
        self.planning_graph.compute_preimages(goal)

        # Try to add shortcuts
        if not self.training_mode:
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
            self.context_env.set_context(atoms, self._current_preimage)
            aug_obs = self.context_env.augment_observation(obs)
            return ApproachStepResult(action=self.policy.get_action(aug_obs))

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

                # Configure policy and context wrapper
                self.context_env.set_context(atoms, self._current_preimage)
                self.policy.configure_context(
                    PolicyContext(
                        preimage=self._current_preimage,
                        current_atoms=atoms,
                        info={
                            "source_node_id": self._current_edge.source.id,
                            "target_node_id": target_node.id,
                        },
                    )
                )

                # If in training mode, collect this state and terminate
                if self.training_mode:
                    aug_obs = self.context_env.augment_observation(obs)
                    return ApproachStepResult(
                        action=self.policy.get_action(aug_obs),
                        terminate=True,
                        info={
                            "training_state": obs,
                            "current_atoms": atoms,
                            "preimage": self._current_preimage,
                        },
                    )

                aug_obs = self.context_env.augment_observation(obs)
                return ApproachStepResult(action=self.policy.get_action(aug_obs))

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

            print(f"\n--- Node {node_count-1} at depth {depth} ---")
            print(f"Contains {len(current_node.atoms)} atoms: {current_node.atoms}")

            # Check if this is a goal state
            if self._goal and self._goal.issubset(current_node.atoms):
                continue

            # Find applicable ground operators using the domain's operators
            applicable_ops = self._find_applicable_operators(
                set(current_node.atoms), objects
            )

            # Apply each applicable operator to generate new states
            for op in applicable_ops:
                # Apply operator effects to get next state
                next_atoms = set(current_node.atoms)
                next_atoms.difference_update(op.delete_effects)
                next_atoms.update(op.add_effects)

                # Check if we've seen this state before
                next_atoms_frozen = frozenset(next_atoms)
                if next_atoms_frozen in visited_states:
                    # Create edge to existing node
                    next_node = visited_states[next_atoms_frozen]
                    graph.add_edge(current_node, next_node, op)
                else:
                    # Create new node and edge
                    next_node = graph.add_node(next_atoms)
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

        for lifted_op in domain_operators:
            # Find valid groundings using parameter types
            valid_groundings = self._find_valid_groundings(lifted_op, objects)

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

        # For each parameter, find objects of the right type
        param_objects = []
        for param in lifted_op.parameters:
            if param.type in objects_by_type:
                param_objects.append(objects_by_type[param.type])
            else:
                return []

        # Generate all possible groundings
        groundings = list(itertools.product(*param_objects))

        return groundings

    def _try_add_shortcuts(self, graph: PlanningGraph) -> None:
        """Try to add shortcut edges to the graph."""
        for source_node in graph.nodes:
            for target_node in graph.nodes:
                # Skip same node and existing edges
                if source_node == target_node:
                    continue
                if any(
                    edge.target == target_node
                    for edge in graph.node_to_outgoing_edges.get(source_node, [])
                ):
                    continue

                source_atoms = set(source_node.atoms)
                target_preimage = graph.preimages.get(target_node, set())
                if not target_preimage:
                    continue

                # Check if this is similar to a trained shortcut
                if self.trained_signatures:
                    current_sig = ShortcutSignature.from_context(
                        source_atoms, target_preimage
                    )
                    can_handle = False
                    best_similarity = 0.0

                    for trained_sig in self.trained_signatures:
                        similarity = current_sig.similarity(trained_sig)
                        if similarity > 0.9:  # Threshold to be tuned
                            can_handle = True
                            best_similarity = max(best_similarity, similarity)

                    if not can_handle:
                        continue

                    print(f"Found similar shortcut with similarity {best_similarity}")

                # Configure context for environment and policy
                self.context_env.set_context(source_atoms, target_preimage)
                self.policy.configure_context(
                    PolicyContext(
                        preimage=target_preimage,
                        current_atoms=source_atoms,
                        info={
                            "source_node_id": source_node.id,
                            "target_node_id": target_node.id,
                        },
                    )
                )
                if self.policy.can_initiate():
                    print(
                        f"Trying to add shortcut: {source_node.id} to {target_node.id}"
                    )
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

        raw_env = self._get_base_env(self.system.wrapped_env)

        queue = [initial_node]
        while queue:
            node = queue.pop()
            for edge in self.planning_graph.node_to_outgoing_edges[node]:
                raw_obs, info = node_to_obs_info[node]
                raw_env.reset_from_state(raw_obs)  # type: ignore
                _, init_atoms, _ = self.system.perceiver.reset(raw_obs, info)
                preimage = self.planning_graph.preimages[edge.target]
                if edge.is_shortcut:
                    self.policy.configure_context(
                        PolicyContext(
                            preimage=preimage,
                            current_atoms=init_atoms,
                            info={
                                "source_node_id": node.id,
                                "target_node_id": edge.target.id,
                            },
                        )
                    )
                    self.context_env.set_context(init_atoms, preimage)
                    aug_obs = self.context_env.augment_observation(raw_obs)
                    skill: Policy | Skill = self.policy
                else:
                    assert edge.operator is not None
                    skill = self._get_skill(edge.operator)
                    skill.reset(edge.operator)
                    aug_obs = raw_obs

                num_steps = 0
                curr_raw_obs = raw_obs
                curr_aug_obs = aug_obs
                for _ in range(self._max_skill_steps):
                    act = skill.get_action(curr_aug_obs)
                    next_raw_obs, _, _, _, info = raw_env.step(act)
                    curr_raw_obs = next_raw_obs
                    atoms = self.system.perceiver.step(curr_raw_obs)
                    if edge.is_shortcut:
                        self.context_env.set_context(atoms, preimage)
                        curr_aug_obs = self.context_env.augment_observation(
                            curr_raw_obs
                        )
                    else:
                        curr_aug_obs = curr_raw_obs

                    num_steps += 1

                    if preimage.issubset(atoms):
                        print(
                            f"Added shortcut edge {edge.source.id} -> {edge.target.id} cost: {num_steps}. Is shortcut? {edge.is_shortcut}"  # pylint: disable=line-too-long
                        )
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
                    node_to_obs_info[edge.target] = (curr_raw_obs, info)
                    queue.append(edge.target)

    def _get_base_env(self, env: gym.Env) -> gym.Env:
        """Get the base environment by unwrapping wrappers if needed."""
        current_env = env
        while hasattr(current_env, "env"):
            if isinstance(current_env, ContextAwareWrapper):
                current_env = current_env.env
                continue
            if hasattr(current_env, "reset_from_state"):
                return current_env
            current_env = current_env.env

        if hasattr(current_env, "reset_from_state"):
            return current_env

        raise AttributeError("Could not find environment with reset_from_state method")
