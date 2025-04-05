"""Planning graph representation for improvisational TAMP."""

import heapq
import itertools
from dataclasses import dataclass, field

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
    is_shortcut: bool = False

    # Store path-dependent costs: (path, source_node_id) -> cost
    # where path is a tuple of node IDs
    costs: dict[tuple[tuple[int, ...], int], float] = field(default_factory=dict)

    def __hash__(self) -> int:
        return hash((self.source, self.target, self.operator))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PlanningGraphEdge):
            return False
        return (
            self.source == other.source
            and self.target == other.target
            and self.operator == other.operator
        )

    def get_cost(self, path: tuple[int, ...]) -> float:
        """Get the cost of this edge when coming via the specified path."""
        if not self.costs:
            return self.cost

        # Try to find exact path match
        for (p, _), cost in self.costs.items():
            if p == path:
                return cost

        # If no exact match, look for a path ending with the same node
        for (p, node_id), cost in self.costs.items():
            if p and p[-1] == self.source.id and node_id == self.source.id:
                return cost

        # Default to the minimum cost if no matching path is found
        return self.cost


class PlanningGraph:
    """Graph representation of a task plan."""

    def __init__(self) -> None:
        self.nodes: list[PlanningGraphNode] = []
        self.edges: list[PlanningGraphEdge] = []
        self.node_to_incoming_edges: dict[
            PlanningGraphNode, list[PlanningGraphEdge]
        ] = {}
        self.node_to_outgoing_edges: dict[
            PlanningGraphNode, list[PlanningGraphEdge]
        ] = {}
        self.node_map: dict[frozenset[GroundAtom], PlanningGraphNode] = {}
        self.preimages: dict[PlanningGraphNode, set[GroundAtom]] = {}
        self.goal_nodes: list[PlanningGraphNode] = []

    def add_node(self, atoms: set[GroundAtom]) -> PlanningGraphNode:
        """Add a node to the graph."""
        frozen_atoms = frozenset(atoms)
        assert frozen_atoms not in self.node_map
        node_id = len(self.nodes)
        node = PlanningGraphNode(frozen_atoms, node_id)
        self.nodes.append(node)
        self.node_map[frozen_atoms] = node
        self.node_to_incoming_edges[node] = []
        self.node_to_outgoing_edges[node] = []
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
        self.node_to_incoming_edges[edge.target].append(edge)
        self.node_to_outgoing_edges[edge.source].append(edge)
        return edge

    def compute_preimages(self, goal: set[GroundAtom]) -> None:
        """Compute self.preimages for all nodes.

        Preimage(j) := Preimage(j+1) + op(j)[preconditions] - op(j)[add
        effects]
        """
        self.preimages = {}
        self.goal_nodes = [node for node in self.nodes if goal.issubset(node.atoms)]
        assert self.goal_nodes, "No goal node found"
        print(f"Found {len(self.goal_nodes)} goal nodes")
        for goal_node in self.goal_nodes:
            self.preimages[goal_node] = set(goal)

        # Initialize queue with all goal nodes
        queue = []
        for goal_node in self.goal_nodes:
            self.preimages[goal_node] = set(goal)
            queue.append(goal_node)

        # Work backwards from the goals in (reverse) breadth-first order
        while queue:
            node = queue.pop(0)
            assert node in self.preimages
            incoming_edges = self.node_to_incoming_edges[node]
            for edge in incoming_edges:
                assert edge.operator and not edge.is_shortcut
                source_preimage = self.preimages[node].copy()
                source_preimage.difference_update(edge.operator.add_effects)
                source_preimage.update(edge.operator.preconditions)
                # NOTE: If several branches start from the same source node, we only
                # define the preimage of the source node once when we first encounter
                # it in reverse BFS.
                if edge.source not in self.preimages:
                    self.preimages[edge.source] = source_preimage
                    queue.append(edge.source)

    def find_shortest_path(
        self, init_atoms: set[GroundAtom], goal: set[GroundAtom]
    ) -> list[PlanningGraphEdge]:
        """Find shortest path from initial node to goal node using path-aware
        costs."""
        if not self.nodes:
            return []

        initial_node = self.node_map[frozenset(init_atoms)]
        goal_nodes = [node for node in self.nodes if goal.issubset(node.atoms)]
        assert goal_nodes, "No goal node found"
        assert len(goal_nodes) == 1, "No support for multiple goal nodes"
        goal_node = goal_nodes[0]

        # Modified Dijkstra's algorithm that considers the path taken
        distances: dict[tuple[PlanningGraphNode, tuple[int, ...]], float] = {}
        previous: dict[
            tuple[PlanningGraphNode, tuple[int, ...]],
            tuple[tuple[PlanningGraphNode, tuple[int, ...]], PlanningGraphEdge] | None,
        ] = {}

        # Initialize with empty path for initial node
        empty_path: tuple[int, ...] = tuple()
        start_state = (initial_node, empty_path)
        distances[start_state] = 0
        previous[start_state] = None

        # Priority queue for Dijkstra's algorithm
        # (use a counter to break ties and avoid comparing non-comparable objects)
        counter = itertools.count()
        queue: list[tuple[float, int, tuple[PlanningGraphNode, tuple[int, ...]]]] = [
            (0, next(counter), start_state)
        ]  # (distance, counter, (node, path))

        # Track visited states to avoid cycles
        visited = set()

        while queue:
            # Get state with smallest distance
            current_dist, _, current_state = heapq.heappop(queue)
            current_node, current_path = current_state

            if current_state in visited:
                continue
            visited.add(current_state)

            if current_node == goal_node:
                break

            # Check all outgoing edges
            for edge in [e for e in self.edges if e.source == current_node]:
                edge_cost = edge.get_cost(current_path)
                new_dist = current_dist + edge_cost
                new_path = current_path + (current_node.id,)
                new_state = (edge.target, new_path)

                # If we found a better path, update
                if new_state not in distances or new_dist < distances.get(
                    new_state, float("inf")
                ):
                    distances[new_state] = float(new_dist)
                    previous[new_state] = (current_state, edge)
                    heapq.heappush(queue, (new_dist, next(counter), new_state))

        # Find the best goal state
        goal_states = [(n, p) for (n, p) in distances if n == goal_node]
        assert goal_states, "No goal state found"
        best_goal_state = min(goal_states, key=lambda s: distances.get(s, float("inf")))

        # Reconstruct path
        path = []
        current_state = best_goal_state
        while current_state != start_state:
            prev_entry = previous.get(current_state)
            if prev_entry is None:
                raise ValueError("No valid path found")
            prev_state, edge = prev_entry
            path.append(edge)
            current_state = prev_state
        path.reverse()

        # Print detailed path information
        total_cost = distances[best_goal_state]
        print(f"Shortest path's cost: {total_cost}")
        path_details = []
        for edge in path:
            if edge.costs:
                cost_details = []
                for (p, _), cost in edge.costs.items():
                    path_str = "-".join(str(node_id) for node_id in p) if p else "start"
                    cost_details.append(f"via {path_str}: {cost}")
                path_details.append(
                    f"{edge.source.id}->{edge.target.id} [{', '.join(cost_details)}]"
                )
            else:
                path_details.append(
                    f"{edge.source.id}->{edge.target.id} [cost: {edge.cost}]"
                )
        print(f"Path details: {' -> '.join(path_details)}")

        return path
