"""Visualization test for the planning graph."""

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pytest

from tamp_improv.approaches.improvisational.base import ImprovisationalTAMPApproach
from tamp_improv.approaches.improvisational.policies.pushing import PushingPolicy
from tamp_improv.approaches.improvisational.policies.pushing_pybullet import (
    PybulletPushingPolicy,
)
from tamp_improv.benchmarks.blocks2d import Blocks2DTAMPSystem
from tamp_improv.benchmarks.pybullet_clear_and_place import ClearAndPlaceTAMPSystem


def visualize_graph(graph, output_path=None):
    """Visualize a planning graph."""
    G = nx.DiGraph()

    for node in graph.nodes:
        atoms_str = []
        count = 0
        for atom in node.atoms:
            atom_str = str(atom)
            atoms_str.append(atom_str)
            count += 1

        # label = f"Node {node.id}\n" + "\n".join(atoms_str)
        label = f"Node {node.id}"
        G.add_node(node, label=label)

    for edge in graph.edges:
        edge_label = ""
        if edge.operator:
            edge_label = edge.operator.name
            if hasattr(edge.operator, "parameters") and edge.operator.parameters:
                param_names = [p.name for p in edge.operator.parameters]
                if len(param_names) <= 2:  # Avoid long labels
                    edge_label += f"\n({', '.join(param_names)})"
        else:
            edge_label = "Shortcut" if edge.is_shortcut else ""

        G.add_edge(edge.source, edge.target, label=edge_label)

    plt.figure(figsize=(14, 10))

    if len(G.nodes) <= 10:
        pos = nx.spring_layout(G, seed=42, k=1.0)
    else:
        pos = nx.kamada_kawai_layout(G)

    nx.draw_networkx_nodes(G, pos, node_size=2000, node_color="lightblue", alpha=0.8)
    nx.draw_networkx_edges(
        G,
        pos,
        width=1.5,
        arrowsize=20,
        connectionstyle="arc3,rad=0.2",
        arrowstyle="->",
        alpha=0.7,
    )
    nx.draw_networkx_labels(
        G,
        pos,
        labels=nx.get_node_attributes(G, "label"),
        font_size=9,
        font_family="sans-serif",
    )

    edge_labels = {(u, v): d["label"] for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(
        G, pos, connectionstyle="arc3,rad=0.2", edge_labels=edge_labels, font_size=8
    )

    plt.title("Planning Graph Visualization", fontsize=16, pad=20)
    plt.axis("off")
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Graph visualization saved to {output_path}")


@pytest.mark.parametrize(
    "system_cls,policy_cls,env_name",
    [
        (Blocks2DTAMPSystem, PushingPolicy, "blocks2d"),
        (ClearAndPlaceTAMPSystem, PybulletPushingPolicy, "pybullet"),
    ],
)
def test_planning_graph_visualization(system_cls, policy_cls, env_name):
    """Test building and visualizing the planning graphs."""
    print(f"\n=== Testing {env_name} Planning Graph Visualization ===")

    # Create system and approach
    system = system_cls.create_default(seed=42)
    approach = ImprovisationalTAMPApproach(
        system=system,
        policy=policy_cls(seed=42),
        seed=42,
    )

    # Reset system and approach
    print("Resetting system and approach...")
    obs, info = system.reset()
    objects, init_atoms, goal_atoms = system.perceiver.reset(obs, info)

    # Save initial state information
    output_dir = Path("results/planning_graph")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / f"{env_name}_planning_info.txt", "w", encoding="utf-8") as f:
        f.write(f"=== {env_name} Planning Problem Information ===\n\n")

        f.write("Objects:\n")
        for obj in objects:
            f.write(f"  {obj.name} (Type: {obj.type.name})\n")

        f.write("\nInitial Atoms:\n")
        for atom in init_atoms:
            f.write(f"  {atom}\n")

        f.write("\nGoal Atoms:\n")
        for atom in goal_atoms:
            f.write(f"  {atom}\n")

    approach._goal = goal_atoms  # pylint: disable=protected-access
    graph = approach._create_planning_graph(  # pylint: disable=protected-access
        objects, init_atoms
    )

    print("\nVisualizing planning graph...")
    visualize_graph(graph, output_dir / f"{env_name}_planning_graph.png")

    return graph
