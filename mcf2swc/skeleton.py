"""
Graph-based skeleton handler.

Provides a `SkeletonGraph` class that inherits from networkx.Graph to:
- Represent skeleton as a graph with xyz coordinates on each node
- Load from polylines array or polylines text format: `N x1 y1 z1 x2 y2 z2 ...`
- Identify terminal nodes (degree 1) and branch nodes (degree 3+)
- Every point from input polylines becomes a node in the graph

Note: This class represents skeleton topology as a graph where:
- Nodes have 'pos' attribute with (x, y, z) coordinates
- Edges connect consecutive nodes along polylines
- Terminal nodes have degree 1
- Branch nodes have degree 3+
- Continuation nodes have degree 2
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Set

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class SkeletonGraph(nx.Graph):
    """
    Graph-based skeleton representation with xyz coordinates on nodes.

    Inherits from networkx.Graph. Each node has a 'pos' attribute storing
    (x, y, z) coordinates. Edges represent connections between consecutive
    points along polylines.

    Every point from input polylines becomes a node.
    Endpoints within tolerance are merged into single nodes.

    Terminal nodes (degree 1) are isolated endpoints.
    Branch nodes (degree 3+) are where multiple branches meet.
    Continuation nodes (degree 2) are intermediate points along branches.
    """

    def __init__(self, tolerance: float = 1e-6, **attr):
        """
        Initialize a SkeletonGraph.

        Args:
            tolerance: Distance threshold for merging nearby endpoints
            **attr: Additional graph attributes
        """
        super().__init__(**attr)
        self.graph["tolerance"] = tolerance
        self._next_node_id = 0

    @classmethod
    def from_polylines(
        cls,
        polylines: Sequence[np.ndarray],
        tolerance: float = 1e-6,
    ) -> "SkeletonGraph":
        """
        Create a SkeletonGraph from a list of polyline arrays.

        Every point in the polylines becomes a node in the graph.
        Consecutive points are connected by edges.
        Endpoints within tolerance are merged into a single node.

        Args:
            polylines: List of (N_i, 3) arrays representing polylines
            tolerance: Distance threshold for merging nearby endpoints

        Returns:
            SkeletonGraph instance
        """
        graph = cls(tolerance=tolerance)

        if not polylines:
            return graph

        # Step 1: Create nodes for all points in all polylines
        point_to_node = {}  # Maps (poly_idx, point_idx) -> node_id
        endpoints = []  # List of (node_id, poly_idx, point_idx, coord)

        for poly_idx, pl in enumerate(polylines):
            if len(pl) == 0:
                continue

            for point_idx, coord in enumerate(pl):
                node_id = graph._get_next_node_id()
                graph.add_node(node_id, pos=np.array(coord, dtype=float))
                point_to_node[(poly_idx, point_idx)] = node_id

                # Track endpoints (first and last points of each polyline)
                if point_idx == 0 or point_idx == len(pl) - 1:
                    endpoints.append((node_id, poly_idx, point_idx, coord))

        # Step 2: Merge endpoints that are within tolerance
        endpoint_groups = []
        used = set()

        for i, (node_i, poly_i, pt_i, coord_i) in enumerate(endpoints):
            if i in used:
                continue

            # Find all endpoints close to this one
            group = [(node_i, poly_i, pt_i, coord_i)]
            for j, (node_j, poly_j, pt_j, coord_j) in enumerate(endpoints):
                if i != j and j not in used:
                    dist = np.linalg.norm(coord_i - coord_j)
                    if dist < tolerance:
                        group.append((node_j, poly_j, pt_j, coord_j))

            # Mark all as used
            for idx in range(len(endpoints)):
                if any(endpoints[idx][0] == node_id for node_id, _, _, _ in group):
                    used.add(idx)

            endpoint_groups.append(group)

        # Merge nodes in each group
        for group in endpoint_groups:
            if len(group) == 1:
                continue

            # Use the first node as the merged node
            merged_node = group[0][0]

            # Compute centroid position for the merged node
            coords = np.array([item[3] for item in group])
            merged_pos = coords.mean(axis=0)
            graph.nodes[merged_node]["pos"] = merged_pos

            # Map all other nodes to the merged node
            for node_id, poly_idx, point_idx, _ in group[1:]:
                point_to_node[(poly_idx, point_idx)] = merged_node
                # Remove the redundant node
                if node_id in graph:
                    graph.remove_node(node_id)

        # Step 3: Add edges connecting consecutive points in each polyline
        for poly_idx, pl in enumerate(polylines):
            if len(pl) < 2:
                continue

            for point_idx in range(len(pl) - 1):
                node_u = point_to_node.get((poly_idx, point_idx))
                node_v = point_to_node.get((poly_idx, point_idx + 1))

                if node_u is not None and node_v is not None and node_u != node_v:
                    # Compute edge length
                    pos_u = graph.nodes[node_u]["pos"]
                    pos_v = graph.nodes[node_v]["pos"]
                    length = float(np.linalg.norm(pos_v - pos_u))

                    graph.add_edge(
                        node_u,
                        node_v,
                        polyline_idx=poly_idx,
                        segment_idx=point_idx,
                        length=length,
                    )

        return graph

    @classmethod
    def from_txt(cls, path: str, tolerance: float = 1e-6) -> "SkeletonGraph":
        """
        Load a SkeletonGraph from a `.polylines.txt` file.

        File format: Each line is `N x1 y1 z1 x2 y2 z2 ... xN yN zN`

        Args:
            path: Path to the polylines text file
            tolerance: Distance threshold for merging nearby endpoints

        Returns:
            SkeletonGraph instance
        """
        polylines: List[np.ndarray] = []

        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                s = line.strip()
                if not s:
                    continue

                parts = s.split()
                try:
                    n = int(float(parts[0]))
                except Exception as e:
                    raise ValueError(f"Invalid header count on line {line_no}") from e

                coords = parts[1:]
                if len(coords) != 3 * n:
                    raise ValueError(
                        f"Line {line_no}: expected {3*n} coordinate values, got {len(coords)}"
                    )

                vals = np.array([float(c) for c in coords], dtype=float)
                pl = vals.reshape(n, 3)
                polylines.append(pl)

        return cls.from_polylines(polylines, tolerance=tolerance)

    def _get_next_node_id(self) -> int:
        """Get the next available node ID."""
        node_id = self._next_node_id
        self._next_node_id += 1
        return node_id

    # ---------------------------------------------------------------------
    # Node classification
    # ---------------------------------------------------------------------
    def get_terminal_nodes(self) -> Set[int]:
        """
        Get all terminal nodes (degree 1 - isolated endpoints).

        Returns:
            Set of node IDs that are terminal nodes
        """
        return {node for node in self.nodes() if self.degree(node) == 1}

    def get_branch_nodes(self) -> Set[int]:
        """
        Get all branch nodes (degree 3+ - where multiple branches meet).

        Returns:
            Set of node IDs that are branch nodes
        """
        return {node for node in self.nodes() if self.degree(node) >= 3}

    def get_continuation_nodes(self) -> Set[int]:
        """
        Get all continuation nodes (degree 2 - intermediate points).

        Returns:
            Set of node IDs that are continuation nodes
        """
        return {node for node in self.nodes() if self.degree(node) == 2}

    def is_terminal_node(self, node: int) -> bool:
        """Check if a node is a terminal node (degree 1)."""
        return self.degree(node) == 1

    def is_branch_node(self, node: int) -> bool:
        """Check if a node is a branch node (degree 3+)."""
        return self.degree(node) >= 3

    def is_continuation_node(self, node: int) -> bool:
        """Check if a node is a continuation node (degree 2)."""
        return self.degree(node) == 2

    # ---------------------------------------------------------------------
    # Coordinate access and manipulation
    # ---------------------------------------------------------------------
    def get_node_position(self, node: int) -> np.ndarray:
        """
        Get the (x, y, z) position of a node.

        Args:
            node: Node ID

        Returns:
            (3,) array with xyz coordinates
        """
        return np.array(self.nodes[node]["pos"])

    def set_node_position(self, node: int, pos: np.ndarray) -> None:
        """
        Set the (x, y, z) position of a node.

        Args:
            node: Node ID
            pos: (3,) array with xyz coordinates
        """
        self.nodes[node]["pos"] = np.array(pos, dtype=float)

    def get_all_positions(self) -> np.ndarray:
        """
        Get positions of all nodes as an array.

        Returns:
            (N, 3) array where N is the number of nodes
        """
        if self.number_of_nodes() == 0:
            return np.zeros((0, 3), dtype=float)

        positions = []
        for node in sorted(self.nodes()):
            positions.append(self.get_node_position(node))

        return np.array(positions)

    def set_all_positions(self, positions: np.ndarray) -> None:
        """
        Set positions of all nodes from an array.

        Args:
            positions: (N, 3) array where N is the number of nodes
        """
        if positions.shape[0] != self.number_of_nodes():
            raise ValueError(
                f"Position array has {positions.shape[0]} rows but graph has "
                f"{self.number_of_nodes()} nodes"
            )

        for i, node in enumerate(sorted(self.nodes())):
            self.set_node_position(node, positions[i])

    # ---------------------------------------------------------------------
    # Basic properties
    # ---------------------------------------------------------------------
    def total_points(self) -> int:
        """
        Get total number of points in the skeleton.

        Since every point is a node, this is just the number of nodes.

        Returns:
            Total number of points in the skeleton
        """
        return self.number_of_nodes()

    def bounds(self) -> Optional[dict]:
        """
        Get bounding box of all node positions.

        Returns:
            Dictionary with 'x', 'y', 'z' keys, each containing (min, max) tuple,
            or None if graph is empty
        """
        if self.number_of_nodes() == 0:
            return None

        positions = self.get_all_positions()
        lo = positions.min(axis=0)
        hi = positions.max(axis=0)

        return {
            "x": (float(lo[0]), float(hi[0])),
            "y": (float(lo[1]), float(hi[1])),
            "z": (float(lo[2]), float(hi[2])),
        }

    def centroid(self) -> Optional[np.ndarray]:
        """
        Get centroid of all node positions.

        Returns:
            (3,) array with centroid coordinates, or None if graph is empty
        """
        if self.number_of_nodes() == 0:
            return None

        positions = self.get_all_positions()
        return positions.mean(axis=0)

    # ---------------------------------------------------------------------
    # Conversion
    # ---------------------------------------------------------------------
    def to_polylines(self) -> List[np.ndarray]:
        """
        Convert the graph back to a list of polyline arrays.

        Reconstructs polylines by grouping edges with the same polyline_idx
        and ordering them by segment_idx.

        Returns:
            List of (N_i, 3) arrays representing polylines
        """
        if self.number_of_edges() == 0:
            return []

        # Group edges by polyline_idx
        polyline_edges = {}  # Maps polyline_idx -> list of (segment_idx, u, v)

        for u, v, data in self.edges(data=True):
            poly_idx = data.get("polyline_idx")
            seg_idx = data.get("segment_idx")

            if poly_idx is not None and seg_idx is not None:
                if poly_idx not in polyline_edges:
                    polyline_edges[poly_idx] = []
                polyline_edges[poly_idx].append((seg_idx, u, v))

        # Reconstruct each polyline
        polylines = []

        for poly_idx in sorted(polyline_edges.keys()):
            edges = sorted(polyline_edges[poly_idx], key=lambda x: x[0])

            if not edges:
                continue

            # Build polyline from ordered edges
            points = []

            # Add first point of first edge
            _, u0, v0 = edges[0]
            points.append(self.get_node_position(u0))

            # Add second point of each edge
            for _, u, v in edges:
                points.append(self.get_node_position(v))

            polylines.append(np.array(points))

        return polylines

    def to_txt(self, path: str) -> None:
        """
        Save the skeleton to a `.polylines.txt` file.

        Args:
            path: Output file path
        """
        polylines = self.to_polylines()

        with open(path, "w", encoding="utf-8") as f:
            for pl in polylines:
                n = int(pl.shape[0])
                flat = " ".join(f"{v:.17g}" for v in pl.reshape(-1))
                f.write(f"{n} {flat}\n")

    # ---------------------------------------------------------------------
    # Copy
    # ---------------------------------------------------------------------
    def copy_skeleton(self) -> "SkeletonGraph":
        """
        Create a deep copy of the skeleton graph.

        Returns:
            New SkeletonGraph instance with copied data
        """
        # Create new graph with same tolerance
        new_graph = SkeletonGraph(tolerance=self.graph.get("tolerance", 1e-6))

        # Copy nodes with positions
        for node, data in self.nodes(data=True):
            pos = np.array(data["pos"])
            new_graph.add_node(node, pos=pos.copy())

        # Copy edges with data
        for u, v, data in self.edges(data=True):
            edge_data = dict(data)
            new_graph.add_edge(u, v, **edge_data)

        # Update node ID counter
        new_graph._next_node_id = self._next_node_id

        return new_graph

    # ---------------------------------------------------------------------
    # Statistics
    # ---------------------------------------------------------------------
    def get_statistics(self) -> dict:
        """
        Get statistics about the skeleton graph.

        Returns:
            Dictionary with various statistics
        """
        terminal_nodes = self.get_terminal_nodes()
        branch_nodes = self.get_branch_nodes()
        continuation_nodes = self.get_continuation_nodes()

        edge_lengths = [data.get("length", 0.0) for _, _, data in self.edges(data=True)]

        stats = {
            "num_nodes": self.number_of_nodes(),
            "num_edges": self.number_of_edges(),
            "num_terminal_nodes": len(terminal_nodes),
            "num_branch_nodes": len(branch_nodes),
            "num_continuation_nodes": len(continuation_nodes),
            "total_points": self.total_points(),
        }

        if edge_lengths:
            stats["total_length"] = sum(edge_lengths)
            stats["mean_edge_length"] = np.mean(edge_lengths)
            stats["min_edge_length"] = min(edge_lengths)
            stats["max_edge_length"] = max(edge_lengths)

        return stats

    def __repr__(self) -> str:
        """String representation of the skeleton graph."""
        stats = self.get_statistics()
        return (
            f"SkeletonGraph(nodes={stats['num_nodes']}, "
            f"edges={stats['num_edges']}, "
            f"terminals={stats['num_terminal_nodes']}, "
            f"branches={stats['num_branch_nodes']})"
        )

    # ---------------------------------------------------------------------
    # Resampling
    # ---------------------------------------------------------------------
    def resample(self, spacing: float) -> "SkeletonGraph":
        """
        Resample the skeleton to have approximately uniform spacing between nodes.

        Converts to polylines, resamples each polyline, then reconstructs graph.

        Args:
            spacing: Target distance between consecutive nodes

        Returns:
            New SkeletonGraph with resampled nodes
        """
        # Convert to polylines
        polylines = self.to_polylines()

        # Resample each polyline
        resampled_polylines = []
        for pl in polylines:
            resampled = _resample_polyline(pl, float(spacing))
            resampled_polylines.append(resampled)

        # Create new graph from resampled polylines
        return SkeletonGraph.from_polylines(
            resampled_polylines, tolerance=self.graph.get("tolerance", 1e-6)
        )

    # ---------------------------------------------------------------------
    # Mesh surface projection
    # ---------------------------------------------------------------------
    def snap_to_mesh_surface(
        self,
        mesh,
        project_outside_only: bool = True,
        max_distance: Optional[float] = None,
    ) -> tuple:
        """
        Project node positions to the nearest surface point on mesh.

        Args:
            mesh: trimesh.Trimesh object
            project_outside_only: If True, only project points outside the mesh
            max_distance: If provided, only move points beyond this distance from surface

        Returns:
            (n_moved, mean_move_distance) tuple
        """
        if mesh is None or len(getattr(mesh, "vertices", [])) == 0:
            return 0, 0.0

        if self.number_of_nodes() == 0:
            return 0, 0.0

        # Get all node positions
        positions = self.get_all_positions()

        # Determine which points to project
        use_mask = None
        if project_outside_only:
            try:
                from trimesh.proximity import signed_distance

                d = signed_distance(mesh, positions)
                use_mask = d > 0  # outside
            except Exception:
                use_mask = None

        # Find closest points on mesh
        try:
            from trimesh.proximity import closest_point

            closest_positions, distances, _ = closest_point(mesh, positions)
        except Exception:
            # Fallback: KDTree over vertices only
            vertices = np.asarray(mesh.vertices, dtype=float)
            if vertices.size == 0:
                return 0, 0.0
            from scipy.spatial import cKDTree

            tree = cKDTree(vertices)
            distances, idx = tree.query(positions, k=1)
            closest_positions = vertices[idx]

        # Apply masks
        if use_mask is None:
            mask = np.ones(positions.shape[0], dtype=bool)
        else:
            mask = use_mask

        if max_distance is not None:
            mask = mask & (distances >= float(max_distance))

        # Update positions
        moved = 0
        total_move = 0.0

        for i, node in enumerate(sorted(self.nodes())):
            if mask[i]:
                self.set_node_position(node, closest_positions[i])
                moved += 1
                total_move += distances[i]

        mean_move = (total_move / moved) if moved > 0 else 0.0
        return moved, mean_move

    # ---------------------------------------------------------------------
    # Branch length computation
    # ---------------------------------------------------------------------
    def compute_branch_lengths(self) -> dict:
        """
        Compute the length of each branch (path between terminal/branch nodes).

        Returns:
            Dictionary mapping (start_node, end_node) -> length
        """
        branch_lengths = {}

        # Get terminal and branch nodes
        terminal_nodes = self.get_terminal_nodes()
        branch_nodes = self.get_branch_nodes()
        special_nodes = terminal_nodes | branch_nodes

        # For each special node, trace paths to other special nodes
        for start_node in special_nodes:
            # Use BFS to find paths to other special nodes
            visited = {start_node}
            queue = [(start_node, [start_node], 0.0)]  # (node, path, length)

            while queue:
                current, path, length = queue.pop(0)

                for neighbor in self.neighbors(current):
                    if neighbor in visited:
                        continue

                    # Get edge length
                    edge_data = self.get_edge_data(current, neighbor)
                    edge_length = edge_data.get("length", 0.0) if edge_data else 0.0
                    new_length = length + edge_length
                    new_path = path + [neighbor]

                    # If we reached another special node, record the branch
                    if neighbor in special_nodes and neighbor != start_node:
                        key = tuple(sorted([start_node, neighbor]))
                        if key not in branch_lengths:
                            branch_lengths[key] = new_length
                    else:
                        # Continue searching
                        visited.add(neighbor)
                        queue.append((neighbor, new_path, new_length))

        return branch_lengths

    def get_total_length(self) -> float:
        """
        Get the total length of all edges in the skeleton.

        Returns:
            Total length
        """
        return sum(data.get("length", 0.0) for _, _, data in self.edges(data=True))


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------
def _resample_polyline(pl: np.ndarray, spacing: float) -> np.ndarray:
    """
    Resample a polyline at approximately constant arc-length spacing.

    Includes the first and last vertex; inserts intermediate points every
    multiple of `spacing` along cumulative arclength.

    Args:
        pl: (N, 3) array of points
        spacing: Target spacing between points

    Returns:
        (M, 3) array of resampled points
    """
    P = np.asarray(pl, dtype=float)
    if P.ndim != 2 or P.shape[1] != 3 or P.shape[0] == 0:
        return np.zeros((0, 3), dtype=float)
    if P.shape[0] == 1:
        return P.copy()

    seg = np.linalg.norm(P[1:] - P[:-1], axis=1)
    L = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(L[-1])

    if total <= 0.0:
        return P[[0], :].copy()

    step = float(max(spacing, 1e-12))
    # Always include start and end
    targets = list(np.arange(0.0, total, step))
    if targets[-1] != total:
        targets.append(total)

    out: List[np.ndarray] = []
    si = 0  # segment index
    for t in targets:
        # advance si until L[si] <= t <= L[si+1]
        while si < len(seg) and L[si + 1] < t:
            si += 1
        if si >= len(seg):
            out.append(P[-1])
            continue
        t0 = L[si]
        t1 = L[si + 1]
        if t1 <= t0:
            out.append(P[si])
            continue
        alpha = (t - t0) / (t1 - t0)
        Q = (1.0 - alpha) * P[si] + alpha * P[si + 1]
        out.append(Q)

    return np.vstack(out) if out else P[[0], :].copy()
