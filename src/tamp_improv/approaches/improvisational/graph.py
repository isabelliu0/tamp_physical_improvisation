"""Planning graph representation for improvisational TAMP."""

from dataclasses import dataclass

from relational_structs import GroundAtom, GroundOperator


@dataclass
class PlanningGraphNode:
    """Node in the planning graph representing a set of atoms."""

    atoms: frozenset[GroundAtom]
    id: int

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
        self.node_map: dict[frozenset[GroundAtom], PlanningGraphNode] = {}
        self.preimages: dict[PlanningGraphNode, set[GroundAtom]] = {}

    def add_node(self, atoms: set[GroundAtom]) -> PlanningGraphNode:
        """Add a node to the graph."""
        frozen_atoms = frozenset(atoms)
        assert frozen_atoms not in self.node_map
        node_id = len(self.nodes)
        node = PlanningGraphNode(frozen_atoms, node_id)
        self.nodes.append(node)
        self.node_map[frozen_atoms] = node
        return node

    def add_edge(
        self,
        source: PlanningGraphNode,
        target: PlanningGraphNode,
        operator: GroundOperator | None = None,
        cost: float = float("inf"),
        is_shortcut: bool = False,
    ) -> PlanningGraphEdge:
        """Add an edge to the graph."""
        edge = PlanningGraphEdge(source, target, operator, cost, is_shortcut)
        self.edges.append(edge)
        return edge

    def compute_preimages(self, goal: set[GroundAtom]) -> None:
        """Compute self.preimages for all nodes.

        Preimage(j) := Preimage(j+1) + op(j)[preconditions] - op(j)[add
        effects]
        """
        self.preimages = {}
        goal_nodes = [node for node in self.nodes if goal.issubset(node.atoms)]
        assert len(goal_nodes) == 1, "Figure out how to handle this"
        goal_node = goal_nodes[0]
        self.preimages[goal_node] = set(goal)

        node_to_in_edges: dict[PlanningGraphNode, list[PlanningGraphEdge]] = {
            n: [] for n in self.nodes
        }
        for edge in self.edges:
            node_to_in_edges[edge.target].append(edge)

        # Work backwards from the goal in (reverse) breadth-first order
        queue = [goal_node]
        while queue:
            node = queue.pop(0)  # breadth-first
            assert node in self.preimages
            incoming_edges = node_to_in_edges[node]
            for edge in incoming_edges:
                assert edge.operator and not edge.is_shortcut
                source_preimage = self.preimages[node].copy()
                source_preimage.difference_update(edge.operator.add_effects)
                source_preimage.update(edge.operator.preconditions)
                # If there is more than one path to get to the goal, prefer
                # the one that is closer to the goal -- this should happen
                # because we are expanding in reverse breadth-first order.
                # It's possible that there are ties, in which case we are
                # tiebreaking arbitrarily.
                if edge.source not in self.preimages:
                    self.preimages[edge.source] = source_preimage
                    queue.append(edge.source)

    def find_shortest_path(
        self, init_atoms: set[GroundAtom], goal: set[GroundAtom]
    ) -> list[PlanningGraphEdge]:
        """Find shortest path from initial node to goal node."""
        if not self.nodes:
            return []

        initial_node = self.node_map[frozenset(init_atoms)]
        goal_nodes = [node for node in self.nodes if goal.issubset(node.atoms)]
        assert len(goal_nodes) == 1, "Figure out how to handle this"
        goal_node = goal_nodes[0]

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
