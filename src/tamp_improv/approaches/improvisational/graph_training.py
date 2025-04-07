"""Graph-based training data collection for improvisational TAMP."""

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, TypeVar

import numpy as np
from relational_structs import GroundAtom
from tomsutils.utils import sample_seed_from_rng

from tamp_improv.approaches.improvisational.base import (
    ImprovisationalTAMPApproach,
    ShortcutSignature,
)
from tamp_improv.approaches.improvisational.graph import (
    PlanningGraph,
    PlanningGraphEdge,
    PlanningGraphNode,
)
from tamp_improv.approaches.improvisational.policies.base import (
    GoalConditionedTrainingData,
    TrainingData,
)
from tamp_improv.benchmarks.base import ImprovisationalTAMPSystem

ObsType = TypeVar("ObsType")


@dataclass
class ShortcutCandidate:
    """Represents a potential shortcut in the planning graph."""

    source_node: PlanningGraphNode
    target_node: PlanningGraphNode
    source_atoms: set[GroundAtom]
    target_preimage: set[GroundAtom]
    source_state: Any  # The actual environment state at the source node


def collect_states_for_all_nodes(
    system, planning_graph: PlanningGraph, max_attempts: int = 10
) -> dict[int, ObsType]:
    """Collect observed states for all nodes in the planning graph.

    This function systematically visits each node in the planning graph by:
    1. Resetting the environment
    2. Finding a path to the target node
    3. Executing the path
    4. Storing the resulting observation
    """
    print("\n=== Collecting States for All Nodes ===")

    observed_states: dict[int, ObsType] = {}

    initial_node = None
    if planning_graph.nodes:
        initial_node = planning_graph.nodes[0]
    assert initial_node is not None

    # Collect state for initial node
    obs, info = system.reset()
    observed_states[initial_node.id] = obs
    print(f"Collected state for initial node {initial_node.id}")

    # For each other node, try to reach it and collect its state
    remaining_nodes = [n for n in planning_graph.nodes if n.id != initial_node.id]
    print(f"Attempting to collect states for {len(remaining_nodes)} additional nodes")

    for target_node in remaining_nodes:
        print(f"\nTargeting node {target_node.id}...")
        # Find path from initial node to target node
        path = find_path_to_node(planning_graph, initial_node, target_node)

        if not path:
            print(f"No path found to node {target_node.id}, skipping")
            continue

        print(f"Found path of length {len(path)} to node {target_node.id}")

        # Try to execute the path and collect the state
        for attempt in range(max_attempts):
            obs, info = system.reset()
            _ = system.perceiver.reset(obs, info)

            print(f"Attempt {attempt+1}/{max_attempts} to reach node {target_node.id}")

            # Execute each step in the path
            success = True
            for i, edge in enumerate(path):
                print(f"  Step {i+1}/{len(path)}: {edge.source.id} -> {edge.target.id}")

                # Execute the operator for this edge
                if not edge.operator:
                    print("  No operator for this edge, skipping")
                    success = False
                    break

                # Get the skill for this operator
                skill = None
                for s in system.skills:
                    if s.can_execute(edge.operator):
                        skill = s
                        break

                if not skill:
                    print(
                        f"  No skill found for operator {edge.operator.name}, skipping"
                    )
                    success = False
                    break

                # Reset the skill with the operator
                skill.reset(edge.operator)

                # Execute the skill until complete
                max_steps = 50
                for step in range(max_steps):
                    action = skill.get_action(obs)
                    obs, _, term, trunc, info = system.env.step(action)
                    atoms = system.perceiver.step(obs)

                    if set(edge.target.atoms) == atoms:
                        print(f"  Reached state for node {edge.target.id}")
                        break

                    if term or trunc:
                        print("  Episode terminated unexpectedly")
                        success = False
                        break

                    if step == max_steps - 1:
                        success = False

                if not success:
                    break

            # If we successfully executed the path, store the state
            if success:
                observed_states[target_node.id] = obs
                print(f"Successfully collected state for node {target_node.id}")
                break

            if attempt == max_attempts - 1:
                print(f"Failed to collect state for node {target_node.id}")

    print(
        f"\nFinal collection: {len(observed_states)}/{len(planning_graph.nodes)} nodes"
    )
    return observed_states


def collect_node_states_for_shortcuts(
    system, planning_graph, max_attempts: int = 3
) -> tuple[dict[int, ObsType], list[tuple[int, int]]]:
    """Collect node states for valid shortcuts in the planning graph."""
    print("\n=== Collecting States for Goal-Conditioned Learning ===")
    node_states: dict[int, ObsType] = collect_states_for_all_nodes(
        system, planning_graph, max_attempts
    )

    # Generate valid shortcuts
    valid_shortcuts = []
    node_ids = sorted(list(node_states.keys()))
    for i, source_id in enumerate(node_ids):
        source_node = next((n for n in planning_graph.nodes if n.id == source_id), None)
        if not source_node:
            continue

        for target_id in node_ids[i + 1 :]:
            target_node = next(
                (n for n in planning_graph.nodes if n.id == target_id), None
            )
            if not target_node:
                continue

            # Skip if there's already a direct edge
            has_direct_edge = False
            for edge in planning_graph.node_to_outgoing_edges.get(source_node, []):
                if edge.target == target_node and not edge.is_shortcut:
                    has_direct_edge = True
                    break
            if has_direct_edge:
                continue

            # Only include shortcuts where states are available
            if source_id in node_states and target_id in node_states:
                valid_shortcuts.append((source_id, target_id))

    print(f"Collected states for {len(node_states)} nodes")
    print(f"Identified {len(valid_shortcuts)} valid shortcuts")
    return node_states, valid_shortcuts


def find_path_to_node(
    planning_graph: PlanningGraph,
    start_node: PlanningGraphNode,
    target_node: PlanningGraphNode,
) -> list[PlanningGraphEdge]:
    """Find a path from start_node to target_node in the planning graph."""
    queue: deque[tuple[PlanningGraphNode, list[PlanningGraphEdge]]]
    queue = deque([(start_node, [])])
    visited = {start_node}

    while queue:
        current, path = queue.popleft()

        if current == target_node:
            return path

        for edge in planning_graph.node_to_outgoing_edges.get(current, []):
            if edge.is_shortcut:
                continue

            next_node = edge.target

            if next_node not in visited:
                visited.add(next_node)
                queue.append((next_node, path + [edge]))

    return []


def collect_graph_based_training_data(
    system: ImprovisationalTAMPSystem,
    approach: ImprovisationalTAMPApproach,
    config: dict[str, Any],
    max_shortcuts_per_graph: int = 100,
    target_specific_shortcuts: bool = False,
    use_random_rollouts: bool = False,
    num_rollouts_per_node: int = 50,
    max_steps_per_rollout: int = 50,
    shortcut_success_threshold: int = 1,
    rng: np.random.Generator | None = None,
) -> tuple[TrainingData, dict[int, ObsType]]:
    """Collect training data by exploring the planning graph.

    This actively identifies potential shortcuts between nodes in the
    planning graph and collects the low-level states and goal preimages
    for these shortcuts.
    """
    print("\n=== Collecting Training Data by Exploring Planning Graphs ===")
    approach.training_mode = True

    training_states = []
    current_atoms_list = []
    preimages_list = []
    shortcut_info = []

    # settings from config
    collect_episodes = config.get("collect_episodes", 10)
    seed = config.get("seed", 42)
    rng = np.random.default_rng(seed)

    # Keep track of shortcuts we want to find
    found_target_shortcuts = []

    for episode in range(collect_episodes):
        print(f"\n=== Building planning graph for episode {episode + 1} ===")
        episode_seed = sample_seed_from_rng(rng)
        obs, info = system.reset()
        _ = approach.reset(obs, info)

        assert (
            hasattr(approach, "planning_graph") and approach.planning_graph is not None
        )
        planning_graph = approach.planning_graph
        context_env = approach.context_env

        # Proactively collect states for all nodes
        observed_states: dict[int, Any] = collect_states_for_all_nodes(
            system, planning_graph, max_attempts=3
        )

        # Find potential shortcuts using the collected states
        print(f"\nIdentifying shortcuts using {len(observed_states)} observed states")

        if use_random_rollouts:
            shortcut_candidates = identify_promising_shortcuts_with_rollouts(
                system,
                planning_graph,
                observed_states,
                num_rollouts_per_node,
                max_steps_per_rollout,
                shortcut_success_threshold,
                random_seed=episode_seed,
            )
        else:
            shortcut_candidates = identify_shortcut_candidates(
                planning_graph,
                observed_states,
            )

        print(f"Found {len(shortcut_candidates)} potential shortcut candidates")

        # If targeting specific shortcuts, filter and prioritize them
        if target_specific_shortcuts:
            target_candidates = []

            for candidate in shortcut_candidates:
                # Check if this is one of our target shortcuts
                if is_target_shortcut_1(candidate) and 1 not in found_target_shortcuts:
                    print("Found TARGET SHORTCUT 1!")
                    target_candidates.append(candidate)
                    found_target_shortcuts.append(1)
                elif (
                    is_target_shortcut_2(candidate) and 2 not in found_target_shortcuts
                ):
                    print("Found TARGET SHORTCUT 2!")
                    target_candidates.append(candidate)
                    found_target_shortcuts.append(2)

            if target_candidates:
                print(f"Using {len(target_candidates)} targeted shortcuts")
                selected_candidates = target_candidates
            else:
                print("No target shortcuts found, using regular selection")
                selected_candidates = select_random_shortcuts(
                    shortcut_candidates,
                    max_shortcuts_per_graph,
                    rng,
                )
        else:
            selected_candidates = select_random_shortcuts(
                shortcut_candidates,
                max_shortcuts_per_graph,
                rng,
            )

        print(f"Selected {len(selected_candidates)} shortcuts for training")

        # Collect data (with augmented observations) for each selected shortcut
        for candidate in selected_candidates:
            if approach.use_context_wrapper and context_env is not None:
                context_env.set_context(
                    candidate.source_atoms, candidate.target_preimage
                )
                augmented_obs = context_env.augment_observation(candidate.source_state)
                training_states.append(augmented_obs)
            else:
                training_states.append(candidate.source_state)
            current_atoms_list.append(candidate.source_atoms)
            preimages_list.append(candidate.target_preimage)

            # Record shortcut signature in the approach
            signature = ShortcutSignature.from_context(
                candidate.source_atoms,
                candidate.target_preimage,
            )
            if signature not in approach.trained_signatures:
                approach.trained_signatures.append(signature)
                print(
                    f"Recorded shortcut signature with predicates: {signature.source_predicates} -> {signature.target_predicates}"  # pylint: disable=line-too-long
                )

            # Store shortcut info
            shortcut_info.append(
                {
                    "source_node_id": candidate.source_node.id,
                    "target_node_id": candidate.target_node.id,
                    "source_atoms_count": len(candidate.source_atoms),
                    "target_preimage_count": len(candidate.target_preimage),
                }
            )

            print(
                f"\nAdded shortcut to training data collection: Node {candidate.source_node.id} -> Node {candidate.target_node.id}"  # pylint: disable=line-too-long
            )
            print(f"  Source atoms: {len(candidate.source_atoms)}")
            print(f"  Target preimage: {len(candidate.target_preimage)}")

        # If we've found both target shortcuts, we can stop collecting
        if (
            target_specific_shortcuts
            and 1 in found_target_shortcuts
            and 2 in found_target_shortcuts
        ):
            print("\nFound both target shortcuts, ending collection")
            break

    print("\n=== Training Collection Summary ===")
    print(f"Collected {len(training_states)} examples from {collect_episodes} episodes")

    if target_specific_shortcuts:
        print("Target shortcuts found:")
        if 1 in found_target_shortcuts:
            print("  - 1: Pushing block2 away from target area while holding block1")
        if 2 in found_target_shortcuts:
            print("  - 2: Pushing block2 away from target area with empty gripper")
        if not found_target_shortcuts:
            print("  - No target shortcuts found")

    approach.training_mode = False

    # Get the atom-to-index mapping from the context environment
    atom_to_index = {}
    if (
        approach.use_context_wrapper
        and context_env is not None
        and hasattr(context_env, "get_atom_index_mapping")
    ):
        atom_to_index = context_env.get_atom_index_mapping()

    return (
        TrainingData(
            states=training_states,
            current_atoms=current_atoms_list,
            preimages=preimages_list,
            config={
                **config,
                "shortcut_info": shortcut_info,
                "atom_to_index": atom_to_index,
                "using_context_wrapper": approach.use_context_wrapper,
                "use_random_rollouts": use_random_rollouts,
                "num_rollouts_per_node": num_rollouts_per_node,
                "max_steps_per_rollout": max_steps_per_rollout,
                "shortcut_success_threshold": shortcut_success_threshold,
            },
        ),
        observed_states,
    )


def collect_goal_conditioned_training_data(
    system: ImprovisationalTAMPSystem,
    approach: ImprovisationalTAMPApproach,
    config: dict[str, Any],
    use_random_rollouts: bool = False,
    num_rollouts_per_node: int = 50,
    max_steps_per_rollout: int = 50,
    shortcut_success_threshold: int = 1,
    rng: np.random.Generator | None = None,
) -> GoalConditionedTrainingData:
    """Collect training data for goal-conditioned learning."""
    print("\n=== Collecting Training Data for Goal-Conditioned Learning ===")
    node_states: dict[int, Any] = {}
    valid_shortcuts: list[tuple[int, int]] = []
    node_preimages: dict[int, set[GroundAtom]] = {}

    train_data, node_states = collect_graph_based_training_data(
        system,
        approach,
        config,
        use_random_rollouts=use_random_rollouts,
        num_rollouts_per_node=num_rollouts_per_node,
        max_steps_per_rollout=max_steps_per_rollout,
        shortcut_success_threshold=shortcut_success_threshold,
        rng=rng,
    )
    shortcut_info = train_data.config.get("shortcut_info", [])
    planning_graph = approach.planning_graph
    assert planning_graph is not None
    for info in shortcut_info:
        source_id = info.get("source_node_id")
        target_id = info.get("target_node_id")
        assert source_id is not None and target_id is not None
        valid_shortcuts.append((source_id, target_id))
    for node in planning_graph.nodes:
        if node.id in node_states and node in planning_graph.preimages:
            node_preimages[node.id] = planning_graph.preimages[node]

    # Create goal-conditioned training data
    goal_train_data = GoalConditionedTrainingData(
        states=train_data.states,
        current_atoms=train_data.current_atoms,
        preimages=train_data.preimages,
        config={
            **train_data.config,
            "node_state_count": len(node_states),
            "valid_shortcut_count": len(valid_shortcuts),
        },
        node_states=node_states,
        valid_shortcuts=valid_shortcuts,
        node_preimages=node_preimages,
    )
    return goal_train_data


def identify_shortcut_candidates(
    planning_graph: PlanningGraph,
    observed_states: dict[int, ObsType],
) -> list[ShortcutCandidate]:
    """Identify potential shortcuts in the planning graph.

    A shortcut candidate is a pair of nodes (source, target) where:
    1. target is not directly reachable from source with a single action
    2. target is at least min_distance steps away from source
    3. there is a viable path from source to target
    4. we have an observed state for the source node
    """
    nodes = list(planning_graph.nodes)
    shortcut_candidates = []

    # Check all pairs of nodes
    for source_node in nodes:
        # Skip nodes we don't have an observed state for
        if source_node.id not in observed_states:
            continue

        source_state = observed_states[source_node.id]

        for target_node in nodes:
            if source_node == target_node:
                continue
            if target_node not in planning_graph.preimages:
                continue
            if target_node.id <= source_node.id:
                continue

            # Check if there's already a direct edge from source to target
            has_direct_edge = False
            for edge in planning_graph.node_to_outgoing_edges.get(source_node, []):
                if edge.target == target_node and not edge.is_shortcut:
                    has_direct_edge = True
                    break

            if has_direct_edge:
                continue

            # Check if there's already a direct shortcut edge
            has_shortcut_edge = False
            for edge in planning_graph.node_to_outgoing_edges.get(source_node, []):
                if edge.target == target_node and edge.is_shortcut:
                    has_shortcut_edge = True
                    break

            if has_shortcut_edge:
                continue

            shortcut_candidates.append(
                ShortcutCandidate(
                    source_node=source_node,
                    target_node=target_node,
                    source_atoms=set(source_node.atoms),
                    target_preimage=planning_graph.preimages.get(target_node, set()),
                    source_state=source_state,
                )
            )

    return shortcut_candidates


def identify_promising_shortcuts_with_rollouts(
    system,
    planning_graph,
    observed_states: dict[int, Any],
    num_rollouts_per_node: int = 50,
    max_steps_per_rollout: int = 30,
    shortcut_success_threshold: int = 1,
    random_seed: int = 42,
) -> list[ShortcutCandidate]:
    """Identify promising shortcuts by performing random rollouts from each
    node.

    This function runs random rollouts from each source node in the planning graph
    and tracks which preimages/nodes are reached during exploration. Shortcuts that
    are reached at least `shortcut_success_threshold` times through random exploration
    are considered promising candidates for training.
    """
    print("\n=== Identifying Promising Shortcuts with Random Rollouts ===")
    shortcut_success_counts: defaultdict[tuple[int, int], int] = defaultdict(
        int
    )  # (source_node_id, target_node_id) -> count
    promising_candidates = []

    # Get the base environment for running rollouts
    raw_env = system.env
    raw_env.action_space.seed(random_seed)

    # For each node with an observed state, perform random rollouts
    for source_node_id, source_state in observed_states.items():
        source_node = next(
            (n for n in planning_graph.nodes if n.id == source_node_id), None
        )
        assert source_node is not None
        source_atoms = set(source_node.atoms)
        print(
            f"\nPerforming {num_rollouts_per_node} rollouts from node {source_node_id}"
        )

        # Track preimages reached from this source node
        reached_preimages: defaultdict[int, int] = defaultdict(
            int
        )  # target_node_id -> count

        # Perform random rollouts
        for rollout_idx in range(num_rollouts_per_node):
            if rollout_idx > 0 and rollout_idx % 100 == 0:
                print(f"  Completed {rollout_idx}/{num_rollouts_per_node} rollouts")

            # Reset the environment to source state
            raw_env.reset_from_state(source_state)
            curr_atoms = source_atoms.copy()

            # Execute random actions
            for _ in range(max_steps_per_rollout):
                action = raw_env.action_space.sample()
                obs, _, terminated, truncated, _ = raw_env.step(action)
                curr_atoms = system.perceiver.step(obs)

                # Check if any preimage is reached
                for target_node in planning_graph.nodes:
                    if target_node.id <= source_node_id:
                        continue

                    has_direct_edge = False
                    for edge in planning_graph.node_to_outgoing_edges.get(
                        source_node, []
                    ):
                        if edge.target == target_node and not edge.is_shortcut:
                            has_direct_edge = True
                            break
                    if has_direct_edge:
                        continue

                    # Note: no need to stop this rollout when we reach a preimage
                    # since we want to explore all reachable preimages
                    if target_node in planning_graph.preimages:
                        preimage = planning_graph.preimages[target_node]
                        if preimage and preimage == curr_atoms:
                            reached_preimages[target_node.id] += 1
                            shortcut_success_counts[
                                (source_node_id, target_node.id)
                            ] += 1

                if terminated or truncated:
                    break

        if reached_preimages:
            print(f"  Preimages reached from node {source_node_id}:")
            for target_id, count in sorted(
                reached_preimages.items(), key=lambda x: -x[1]
            ):
                print(f"    → Node {target_id}: {count}/{num_rollouts_per_node} times")
        else:
            print(f"  No preimages reached from node {source_node_id}")

    # Collect promising shortcut candidates
    print("\nShortcuts reaching success threshold:")
    for (source_id, target_id), count in sorted(
        shortcut_success_counts.items(), key=lambda x: -x[1]
    ):
        if count >= shortcut_success_threshold:
            source_node = next(
                (n for n in planning_graph.nodes if n.id == source_id), None
            )
            target_node = next(
                (n for n in planning_graph.nodes if n.id == target_id), None
            )

            assert source_node is not None and target_node is not None
            print(f"  Node {source_id} → Node {target_id}: {count} successes")
            candidate = ShortcutCandidate(
                source_node=source_node,
                target_node=target_node,
                source_atoms=set(source_node.atoms),
                target_preimage=planning_graph.preimages.get(target_node, set()),
                source_state=observed_states[source_id],
            )
            promising_candidates.append(candidate)

    print(f"\nFound {len(promising_candidates)} promising shortcut candidates")
    return promising_candidates


def is_target_shortcut_1(candidate: ShortcutCandidate) -> bool:
    """Check if candidate matches target shortcut 1:
    Pushing block2 away from target area while holding block1
    """
    source_atoms = candidate.source_atoms
    target_atoms = candidate.target_preimage

    # Check if source has the required atoms
    has_holding_block1 = any(
        atom.predicate.name == "Holding"
        and len(atom.objects) == 2
        and atom.objects[0].name == "robot"
        and atom.objects[1].name == "block1"
        for atom in source_atoms
    )
    has_block2_on_target = any(
        atom.predicate.name == "On"
        and len(atom.objects) == 2
        and atom.objects[0].name == "block2"
        and atom.objects[1].name == "target_area"
        for atom in source_atoms
    )
    has_clear_table = any(
        atom.predicate.name == "Clear"
        and len(atom.objects) == 1
        and atom.objects[0].name == "table"
        for atom in source_atoms
    )

    # Check if target has the required atoms
    has_target_clear_target = any(
        atom.predicate.name == "Clear"
        and len(atom.objects) == 1
        and atom.objects[0].name == "target_area"
        for atom in target_atoms
    )
    has_target_holding_block1 = any(
        atom.predicate.name == "Holding"
        and len(atom.objects) == 2
        and atom.objects[0].name == "robot"
        and atom.objects[1].name == "block1"
        for atom in target_atoms
    )

    return (
        has_holding_block1
        and has_block2_on_target
        and has_clear_table
        and has_target_clear_target
        and has_target_holding_block1
    )


def is_target_shortcut_2(candidate: ShortcutCandidate) -> bool:
    """Check if candidate matches target shortcut 2:
    Pushing block2 away from target area with empty gripper
    """
    source_atoms = candidate.source_atoms
    target_atoms = candidate.target_preimage

    # Check if source has the required atoms
    has_block1_on_table = any(
        atom.predicate.name == "On"
        and len(atom.objects) == 2
        and atom.objects[0].name == "block1"
        and atom.objects[1].name == "table"
        for atom in source_atoms
    )
    has_block2_on_target = any(
        atom.predicate.name == "On"
        and len(atom.objects) == 2
        and atom.objects[0].name == "block2"
        and atom.objects[1].name == "target_area"
        for atom in source_atoms
    )
    has_gripper_empty = any(
        atom.predicate.name == "GripperEmpty"
        and len(atom.objects) == 1
        and atom.objects[0].name == "robot"
        for atom in source_atoms
    )
    has_clear_table = any(
        atom.predicate.name == "Clear"
        and len(atom.objects) == 1
        and atom.objects[0].name == "table"
        for atom in source_atoms
    )

    # Check if target has the required atoms
    has_target_clear_target = any(
        atom.predicate.name == "Clear"
        and len(atom.objects) == 1
        and atom.objects[0].name == "target_area"
        for atom in target_atoms
    )
    has_target_gripper_empty = any(
        atom.predicate.name == "GripperEmpty"
        and len(atom.objects) == 1
        and atom.objects[0].name == "robot"
        for atom in target_atoms
    )

    return (
        has_block1_on_table
        and has_block2_on_target
        and has_gripper_empty
        and has_clear_table
        and has_target_clear_target
        and has_target_gripper_empty
    )


def select_random_shortcuts(
    candidates: list[ShortcutCandidate],
    max_shortcuts: int,
    rng: np.random.Generator | None = None,
) -> list[ShortcutCandidate]:
    """Select a random subset of shortcut candidates."""
    rng = rng or np.random.default_rng()
    if len(candidates) <= max_shortcuts:
        return candidates
    indices = np.arange(len(candidates))
    selected_indices = rng.choice(indices, size=max_shortcuts, replace=False)
    return [candidates[i] for i in selected_indices]
