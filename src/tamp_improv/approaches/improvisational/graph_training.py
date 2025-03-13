"""Graph-based training data collection for improvisational TAMP."""

from dataclasses import dataclass
from typing import Any

import numpy as np
from relational_structs import GroundAtom

from tamp_improv.approaches.improvisational.base import ImprovisationalTAMPApproach
from tamp_improv.approaches.improvisational.graph import (
    PlanningGraph,
    PlanningGraphNode,
)
from tamp_improv.approaches.improvisational.policies.base import TrainingData
from tamp_improv.benchmarks.base import ImprovisationalTAMPSystem


@dataclass
class ShortcutCandidate:
    """Represents a potential shortcut in the planning graph."""

    source_node: PlanningGraphNode
    target_node: PlanningGraphNode
    source_atoms: set[GroundAtom]
    target_preimage: set[GroundAtom]
    source_state: Any  # The actual environment state at the source node
    distance: int  # How many regular steps this shortcut would skip


def collect_graph_based_training_data(
    system: ImprovisationalTAMPSystem,
    approach: ImprovisationalTAMPApproach,
    config: dict[str, Any],
    max_shortcuts_per_graph: int = 5,
    min_shortcut_distance: int = 2,  # Minimum number of regular steps to skip
    target_specific_shortcuts: bool = True,  # Whether to prioritize specific shortcuts
) -> TrainingData:
    """Collect training data by exploring the planning graph.

    This actively identifies potential shortcuts between nodes in the
    planning graph and collects the low-level states and goal preimages
    for these shortcuts.
    """
    print("\n=== Collecting Training Data by Exploring Planning Graphs ===")

    training_states = []
    current_atoms_list = []
    preimages_list = []
    shortcut_info = []

    # For backwards compatibility
    preconditions_to_maintain = []
    preconditions_to_achieve = []

    # settings from config
    collect_episodes = config.get("collect_episodes", 10)
    _ = config.get("max_steps", 100)
    seed = config.get("seed", 42)

    np.random.seed(seed)

    # Keep track of shortcuts we want to find
    found_target_shortcuts = []

    for episode in range(collect_episodes):
        print(f"\n=== Building planning graph for episode {episode + 1} ===")
        obs, info = system.reset()
        _ = approach.reset(obs, info)

        assert (
            hasattr(approach, "planning_graph") and approach.planning_graph is not None
        )
        planning_graph = approach.planning_graph

        observed_states = {}
        assert hasattr(approach, "observed_states")
        observed_states = approach.observed_states

        # Find potential shortcuts
        shortcut_candidates = identify_shortcut_candidates(
            planning_graph,
            approach,
            observed_states,
            min_distance=min_shortcut_distance,
        )

        print(f"Found {len(shortcut_candidates)} potential shortcut candidates")

        # If targeting specific shortcuts, filter and prioritize them
        if target_specific_shortcuts:
            target_candidates = []

            for candidate in shortcut_candidates:
                print_shortcut_atoms(candidate)

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
                selected_candidates = select_diverse_shortcuts(
                    shortcut_candidates, max_shortcuts_per_graph
                )
        else:
            selected_candidates = select_diverse_shortcuts(
                shortcut_candidates, max_shortcuts_per_graph
            )

        print(f"Selected {len(selected_candidates)} shortcuts for training")

        # Collect data for each selected shortcut
        for candidate in selected_candidates:
            training_states.append(candidate.source_state)
            current_atoms_list.append(candidate.source_atoms)
            preimages_list.append(candidate.target_preimage)

            # Store shortcut info
            shortcut_info.append(
                {
                    "source_node_id": candidate.source_node.id,
                    "target_node_id": candidate.target_node.id,
                    "distance": candidate.distance,
                    "source_atoms_count": len(candidate.source_atoms),
                    "target_preimage_count": len(candidate.target_preimage),
                }
            )

            # For backwards compatibility
            preconditions_to_maintain.append(candidate.source_atoms)
            preconditions_to_achieve.append(candidate.target_preimage)

            print(
                f"\nAdded shortcut: Node {candidate.source_node.id} -> Node {candidate.target_node.id}"  # pylint: disable=line-too-long
            )
            print(f"  Distance: {candidate.distance} steps")
            print(f"  Source atoms: {len(candidate.source_atoms)}")
            print(f"  Target preimage: {len(candidate.target_preimage)}")

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

    return TrainingData(
        states=training_states,
        current_atoms=current_atoms_list,
        preimages=preimages_list,
        preconditions_to_maintain=preconditions_to_maintain,
        preconditions_to_achieve=preconditions_to_achieve,
        config={
            **config,
            "shortcut_info": shortcut_info,
        },
    )


def identify_shortcut_candidates(
    planning_graph: PlanningGraph,
    approach: ImprovisationalTAMPApproach,
    observed_states: dict[int, Any],
    min_distance: int = 2,
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
            print(f"Skipping node without observed state: Node {source_node.id}")
            continue

        source_state = observed_states[source_node.id]

        for target_node in nodes:
            # Skip self-connections
            if source_node == target_node:
                print(
                    f"Skipping self-connection: Node {source_node.id} -> Node {target_node.id}"  # pylint: disable=line-too-long
                )
                continue

            # Skip if we don't have a preimage for the target
            if target_node not in planning_graph.preimages:
                print(f"Skipping node without preimage: Node {target_node.id}")
                continue

            # Compute the distance between nodes
            try:
                if hasattr(approach, "get_shortcut_distance"):
                    distance = approach.get_shortcut_distance(source_node, target_node)  # type: ignore # pylint: disable=line-too-long
                    print("Using custom distance function")
                else:
                    # Fallback to simple distance check
                    distance = 0
                    # Check if there's a direct edge
                    has_direct_edge = False
                    for edge in planning_graph.node_to_outgoing_edges.get(
                        source_node, []
                    ):
                        if edge.target == target_node and not edge.is_shortcut:
                            has_direct_edge = True
                            break

                    if has_direct_edge:
                        distance = 1
                    else:
                        distance = min_distance
            except Exception as e:
                print(f"Error computing distance: {e}")
                distance = min_distance  # Assume it's a potential shortcut

            # Skip if shortcut doesn't save enough steps
            if distance < min_distance:
                print(
                    f"Skipping shortcut {source_node.id} -> {target_node.id}: Distance {distance} < {min_distance}"  # pylint: disable=line-too-long
                )
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
                    distance=distance,
                )
            )

    return shortcut_candidates


def print_shortcut_atoms(candidate: ShortcutCandidate) -> None:
    """Print the atoms involved in a shortcut candidate."""
    print(
        f"\nShortcut: Node {candidate.source_node.id} -> Node {candidate.target_node.id} (distance: {candidate.distance})"  # pylint: disable=line-too-long
    )
    print("Source atoms:")
    for atom in sorted(candidate.source_atoms, key=str):
        print(f"  - {atom}")
    print("Target preimage:")
    for atom in sorted(candidate.target_preimage, key=str):
        print(f"  - {atom}")


def is_target_shortcut_1(candidate: ShortcutCandidate) -> bool:
    """Check if candidate matches target shortcut 1:
    Pushing block2 away from target area while holding block1
    """
    # Convert atoms to strings for easier matching
    source_atoms = [str(atom) for atom in candidate.source_atoms]
    target_atoms = [str(atom) for atom in candidate.target_preimage]

    # Check if source has the required atoms
    has_holding_block1 = any("Holding(robot, block1)" in a for a in source_atoms)
    has_block2_on_target = any("On(block2, target_area)" in a for a in source_atoms)
    has_clear_table = any("Clear(table)" in a for a in source_atoms)

    # Check if target has the required atoms
    has_target_clear_target = any("Clear(target_area)" in a for a in target_atoms)
    has_target_block2_on_table = any("On(block2, table)" in a for a in target_atoms)
    has_target_holding_block1 = any("Holding(robot, block1)" in a for a in target_atoms)
    has_target_clear_table = any("Clear(table)" in a for a in target_atoms)

    # Check if this matches our target shortcut
    return (
        has_holding_block1
        and has_block2_on_target
        and has_clear_table
        and has_target_clear_target
        and has_target_block2_on_table
        and has_target_holding_block1
        and has_target_clear_table
    )


def is_target_shortcut_2(candidate: ShortcutCandidate) -> bool:
    """Check if candidate matches target shortcut 2:
    Pushing block2 away from target area with empty gripper
    """
    # Convert atoms to strings for easier matching
    source_atoms = [str(atom) for atom in candidate.source_atoms]
    target_atoms = [str(atom) for atom in candidate.target_preimage]

    # Check if source has the required atoms
    has_block1_on_table = any("On(block1, table)" in a for a in source_atoms)
    has_block2_on_target = any("On(block2, target_area)" in a for a in source_atoms)
    has_gripper_empty = any("GripperEmpty(robot)" in a for a in source_atoms)
    has_clear_table = any("Clear(table)" in a for a in source_atoms)

    # Check if target has the required atoms
    has_target_clear_target = any("Clear(target_area)" in a for a in target_atoms)
    has_target_block2_on_table = any("On(block2, table)" in a for a in target_atoms)
    has_target_block1_on_table = any("On(block1, table)" in a for a in target_atoms)
    has_target_gripper_empty = any("GripperEmpty(robot)" in a for a in target_atoms)
    has_target_clear_table = any("Clear(table)" in a for a in target_atoms)

    # Check if this matches our target shortcut
    return (
        has_block1_on_table
        and has_block2_on_target
        and has_gripper_empty
        and has_clear_table
        and has_target_clear_target
        and has_target_block2_on_table
        and has_target_block1_on_table
        and has_target_gripper_empty
        and has_target_clear_table
    )


def select_diverse_shortcuts(
    candidates: list[ShortcutCandidate], max_shortcuts: int
) -> list[ShortcutCandidate]:
    """Select a diverse subset of shortcut candidates."""
    if len(candidates) <= max_shortcuts:
        return candidates

    # Sort by distance (prefer longer shortcuts)
    candidates = sorted(candidates, key=lambda c: c.distance, reverse=True)
    return candidates[:max_shortcuts]
