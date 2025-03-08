"""Planning graph representation for improvisational TAMP."""

from dataclasses import dataclass

from relational_structs import GroundAtom, GroundOperator


@dataclass
class PlanningGraphNode:
    """Node in the planning graph representing a set of atoms."""

    atoms: frozenset[GroundAtom]
    index: int  # Step index in the plan

    def __hash__(self) -> int:
        return hash(self.atoms)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PlanningGraphNode):
            return False
        return self.atoms == other.atoms


@dataclass
class PlanningGraphEdge:
    """Edge in the planning graph representing a transition."""

    source: PlanningGraphNode
    target: PlanningGraphNode
    operator: GroundOperator | None = None
    cost: float = float("inf")
    is_shortcut: bool = False  # Whether this is a learned shortcut


class PlanningGraph:
    """Graph representation of a task plan."""

    def __init__(self) -> None:
        self.nodes: list[PlanningGraphNode] = []
        self.edges: list[PlanningGraphEdge] = []
        self._node_map: dict[frozenset[GroundAtom], PlanningGraphNode] = {}
        self.preimages: dict[PlanningGraphNode, set[GroundAtom]] = {}

    def add_node(self, atoms: set[GroundAtom], index: int) -> PlanningGraphNode:
        """Add a node to the graph."""
        frozen_atoms = frozenset(atoms)
        if frozen_atoms in self._node_map:
            return self._node_map[frozen_atoms]

        node = PlanningGraphNode(frozen_atoms, index)
        self.nodes.append(node)
        self._node_map[frozen_atoms] = node
        return node

    def add_edge(
        self,
        source: PlanningGraphNode,
        target: PlanningGraphNode,
        operator: GroundOperator | None = None,
        cost: float = 1.0,
        is_shortcut: bool = False,
    ) -> PlanningGraphEdge:
        """Add an edge to the graph."""
        edge = PlanningGraphEdge(source, target, operator, cost, is_shortcut)
        self.edges.append(edge)
        return edge

    def compute_preimages(self) -> dict[PlanningGraphNode, set[GroundAtom]]:
        """Compute preimages for all nodes.

        Preimage(j) := Preimage(j+1) + op(j)[preconditions] - op(j)[add
        effects]
        """
        self.preimages = {}
        # Start from the goal node (last node)
        if not self.nodes:
            return self.preimages

        goal_node = max(self.nodes, key=lambda n: n.index)
        self.preimages[goal_node] = set(goal_node.atoms)

        # Work backwards
        for node in sorted(self.nodes, key=lambda n: n.index, reverse=True):
            if node not in self.preimages:
                continue

            incoming_edges = [edge for edge in self.edges if edge.target == node]
            for edge in incoming_edges:
                if edge.operator:
                    source_preimage = self.preimages[node].copy()
                    source_preimage.difference_update(edge.operator.add_effects)
                    source_preimage.update(edge.operator.preconditions)

                    if edge.source in self.preimages:
                        self.preimages[edge.source].update(source_preimage)
                    else:
                        self.preimages[edge.source] = source_preimage
                elif edge.is_shortcut:
                    # For shortcuts, the preimage is the same as the target
                    if edge.source in self.preimages:
                        self.preimages[edge.source].update(self.preimages[node])
                    else:
                        self.preimages[edge.source] = self.preimages[node].copy()

        return self.preimages

    def find_shortest_path(self) -> list[PlanningGraphEdge]:
        """Find shortest path from initial node to goal node."""
        if not self.nodes:
            return []

        initial_node = min(self.nodes, key=lambda n: n.index)
        goal_node = max(self.nodes, key=lambda n: n.index)

        # Dijkstra's algorithm
        distances = {node: float("inf") for node in self.nodes}
        distances[initial_node] = 0
        previous: dict[
            PlanningGraphNode, tuple[PlanningGraphNode, PlanningGraphEdge] | None
        ] = {node: None for node in self.nodes}
        unvisited = set(self.nodes)

        while unvisited:
            current = min(unvisited, key=lambda n: distances[n])

            if current == goal_node:
                break

            unvisited.remove(current)

            outgoing_edges = [edge for edge in self.edges if edge.source == current]
            for edge in outgoing_edges:
                distance = distances[current] + edge.cost
                if distance < distances[edge.target]:
                    distances[edge.target] = distance
                    previous[edge.target] = (current, edge)

        # Reconstruct path
        path = []
        current = goal_node
        while current != initial_node:
            prev_entry = previous[current]
            if prev_entry is None:
                return []  # No path found
            prev_node, edge = prev_entry
            path.append(edge)
            current = prev_node

        path.reverse()
        return path
