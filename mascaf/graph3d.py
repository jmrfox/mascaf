from __future__ import annotations

import logging
from typing import Optional, Set

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class Graph3D(nx.Graph):
    """Undirected graph whose vertices are embedded in 3D."""

    position_attr = "xyz"

    def get_node_position(self, node: int) -> np.ndarray:
        """Return the 3D position of a node."""
        return np.asarray(self.nodes[node][self.position_attr], dtype=float)

    def set_node_position(self, node: int, pos: np.ndarray) -> None:
        """Set the 3D position of a node."""
        self.nodes[node][self.position_attr] = np.asarray(pos, dtype=float)

    def get_all_positions(self) -> np.ndarray:
        """Return positions of all nodes ordered by sorted node ID."""
        if self.number_of_nodes() == 0:
            return np.zeros((0, 3), dtype=float)
        return np.array(
            [self.get_node_position(node) for node in sorted(self.nodes())],
            dtype=float,
        )

    def set_all_positions(self, positions: np.ndarray) -> None:
        """Set positions of all nodes from an `(N, 3)` array."""
        if positions.shape[0] != self.number_of_nodes():
            raise ValueError(
                f"Position array has {positions.shape[0]} rows but graph has "
                f"{self.number_of_nodes()} nodes"
            )
        for idx, node in enumerate(sorted(self.nodes())):
            self.set_node_position(node, positions[idx])

    def get_terminal_nodes(self) -> Set[int]:
        """Return nodes of degree 1."""
        return {node for node in self.nodes() if self.degree(node) == 1}

    def get_branch_nodes(self) -> Set[int]:
        """Return nodes of degree at least 3."""
        return {node for node in self.nodes() if self.degree(node) >= 3}

    def get_continuation_nodes(self) -> Set[int]:
        """Return nodes of degree 2."""
        return {node for node in self.nodes() if self.degree(node) == 2}

    def is_terminal_node(self, node: int) -> bool:
        """Return whether a node is terminal."""
        return self.degree(node) == 1

    def is_branch_node(self, node: int) -> bool:
        """Return whether a node is a branch node."""
        return self.degree(node) >= 3

    def is_continuation_node(self, node: int) -> bool:
        """Return whether a node is a continuation node."""
        return self.degree(node) == 2

    def bounds(self) -> Optional[dict]:
        """Return axis-aligned bounds of the vertex set."""
        if self.number_of_nodes() == 0:
            return None
        if self.__class__.__name__ == "MorphologyGraph":
            logger.info(
                "MorphologyGraph bounds refer to the vertex set, "
                "not the volumetric model"
            )
        positions = self.get_all_positions()
        lo = positions.min(axis=0)
        hi = positions.max(axis=0)
        return {
            "x": (float(lo[0]), float(hi[0])),
            "y": (float(lo[1]), float(hi[1])),
            "z": (float(lo[2]), float(hi[2])),
        }

    def midpoint(self) -> Optional[np.ndarray]:
        """Return the mean position of the graph's vertex set."""
        if self.number_of_nodes() == 0:
            return None
        return self.get_all_positions().mean(axis=0)

    def get_total_length(self) -> float:
        """Return the sum of edge lengths.

        Missing edge lengths are computed from node geometry.
        """
        total = 0.0
        for u, v, data in self.edges(data=True):
            length = data.get("length")
            if length is None:
                pu = self.get_node_position(u)
                pv = self.get_node_position(v)
                length = float(np.linalg.norm(pv - pu))
            total += float(length)
        return total

    def cyclomatic_number(self) -> int:
        """Return the cyclomatic number (independent cycle count) of the graph.

        For a graph with ``c`` connected components this is
        ``|E| - |V| + c`` (zero for forests).
        """
        if self.number_of_nodes() == 0:
            return 0
        return (
            self.number_of_edges()
            - self.number_of_nodes()
            + nx.number_connected_components(self)
        )

    def consolidation_changes_cycle_count(self, u: int, v: int) -> bool:
        """Return whether merging ``(u, v)`` would change :meth:`cyclomatic_number`."""
        if u not in self or v not in self:
            raise KeyError(f"Both nodes must exist; got u={u}, v={v}")
        if not self.has_edge(u, v):
            raise ValueError(f"Nodes {u} and {v} are not connected by an edge")
        before = self.cyclomatic_number()
        after_graph = self._topology_after_consolidation(u, v)
        after = (
            after_graph.number_of_edges()
            - after_graph.number_of_nodes()
            + nx.number_connected_components(after_graph)
        )
        return after != before

    def _topology_after_consolidation(self, u: int, v: int) -> nx.Graph:
        """Topology-only preview of :meth:`consolidate_nodes` (for cycle checks)."""
        g = nx.Graph(self)
        keep, drop = (u, v) if u < v else (v, u)
        for nbr in list(g.neighbors(drop)):
            if nbr != keep:
                g.add_edge(keep, nbr)
        g.remove_node(drop)
        return g

    def consolidate_nodes(self, u: int, v: int) -> int:
        """Merge an edge ``(u, v)`` into one node at the midpoint.

        The lower node id is kept. Neighbors of both endpoints (except each
        other) are reattached to the survivor; if they share a neighbor, that
        becomes a single edge (no multi-edges). For a simple edge with no
        common neighbors, the survivor degree is ``deg(u) + deg(v) - 2``.

        Returns
        -------
        int
            The surviving node id.
        """
        if u not in self or v not in self:
            raise KeyError(f"Both nodes must exist; got u={u}, v={v}")
        if u == v:
            raise ValueError("Cannot consolidate a node with itself")
        if not self.has_edge(u, v):
            raise ValueError(f"Nodes {u} and {v} are not connected by an edge")

        keep, drop = (u, v) if u < v else (v, u)
        pos_keep = self.get_node_position(keep)
        pos_drop = self.get_node_position(drop)
        mid = 0.5 * (pos_keep + pos_drop)

        drop_neighbors = [n for n in self.neighbors(drop) if n != keep]
        for nbr in drop_neighbors:
            if not self.has_edge(keep, nbr):
                self.add_edge(keep, nbr)

        self.remove_node(drop)
        self.set_node_position(keep, mid)
        # Midpoint move invalidates lengths on every incident edge.
        for nbr in list(self.neighbors(keep)):
            self.edges[keep, nbr]["length"] = float(
                np.linalg.norm(self.get_node_position(nbr) - mid)
            )
        return keep

    def next_node_id(self) -> int:
        """Return an unused integer node id (``max(nodes)+1``, or ``0`` if empty)."""
        if self.number_of_nodes() == 0:
            return 0
        return int(max(self.nodes())) + 1

    def bisect_edge(self, u: int, v: int) -> int:
        """Split edge ``(u, v)`` by inserting a new node at the midpoint.

        Removes ``(u, v)`` and adds ``(u, mid)`` and ``(mid, v)`` with
        geometric edge lengths. The new node degree is 2.

        Returns
        -------
        int
            The newly inserted node id.
        """
        if u not in self or v not in self:
            raise KeyError(f"Both nodes must exist; got u={u}, v={v}")
        if u == v:
            raise ValueError("Cannot bisect a self-loop")
        if not self.has_edge(u, v):
            raise ValueError(f"Nodes {u} and {v} are not connected by an edge")

        pos_u = self.get_node_position(u)
        pos_v = self.get_node_position(v)
        mid = 0.5 * (pos_u + pos_v)
        new_id = self.next_node_id()
        self.add_node(new_id, **{self.position_attr: mid})
        self.remove_edge(u, v)
        half_u = float(np.linalg.norm(mid - pos_u))
        half_v = float(np.linalg.norm(pos_v - mid))
        self.add_edge(u, new_id, length=half_u)
        self.add_edge(new_id, v, length=half_v)
        return new_id
