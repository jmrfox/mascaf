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
