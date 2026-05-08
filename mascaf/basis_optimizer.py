"""MorphologyGraph basis optimization prior to radius fitting."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import trimesh

from .graph3d import Graph3D
from .morphology_graph import MorphologyGraph

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@dataclass
class BasisOptimizerOptions:
    """Configuration for morphology-basis optimization."""

    do_pruning: bool = False
    pruning_min_length: Optional[float] = None
    pruning_min_length_percentile: Optional[float] = None
    pruning_iterative: bool = True
    do_snapping: bool = True
    do_forcing: bool = True
    snap_distance_multiplier: float = 1.1
    max_iterations: int = 100
    step_size: float = 0.1
    convergence_threshold: float = 1e-4
    preserve_terminal_nodes: bool = True
    preserve_branch_nodes: bool = False
    n_rays: int = 6
    fallback_distance: float = 10.0
    smoothing_weight: float = 0.5
    verbose: bool = False


class BasisOptimizer:
    """Optimize a downsampled MorphologyGraph basis against a target mesh."""

    def __init__(
        self,
        graph: MorphologyGraph,
        mesh: trimesh.Trimesh,
        options: Optional[BasisOptimizerOptions] = None,
    ):
        self.graph = graph.copy()
        self.mesh = mesh
        self.options = options or BasisOptimizerOptions()

        self._surface_crossing_detected = False
        self._outside_nodes: list[int] = []
        logger.debug(
            "Initialized BasisOptimizer with %d nodes, %d edges, do_pruning=%s, "
            "do_snapping=%s, do_forcing=%s",
            self.graph.number_of_nodes(),
            self.graph.number_of_edges(),
            self.options.do_pruning,
            self.options.do_snapping,
            self.options.do_forcing,
        )

    def get_outside_nodes(self) -> Tuple[list[int], bool, int, float]:
        """Identify graph nodes that lie outside the mesh surface."""
        if self.graph.number_of_nodes() == 0:
            self._outside_nodes = []
            self._surface_crossing_detected = False
            logger.debug("Outside-node query on empty morphology basis")
            return [], False, 0, 0.0

        node_ids = list(self.graph.nodes())
        all_pts = self.graph.get_all_positions()
        logger.debug(
            "Checking %d basis nodes against mesh containment",
            len(node_ids),
        )

        try:
            inside_mask = self.mesh.contains(all_pts)
            outside_mask = ~inside_mask
            num_outside = int(np.sum(outside_mask))
            outside_node_ids = [
                node_ids[i] for i, is_out in enumerate(outside_mask) if is_out
            ]

            max_dist = 0.0
            if num_outside > 0:
                from trimesh.proximity import closest_point

                outside_pts = all_pts[outside_mask]
                _, distances, _ = closest_point(self.mesh, outside_pts)
                max_dist = float(np.max(distances))

            has_crossing = num_outside > 0
            self._surface_crossing_detected = has_crossing
            self._outside_nodes = outside_node_ids
            logger.debug(
                (
                    "Outside-node detection complete: "
                    "has_crossing=%s, outside_nodes=%s"
                ),
                has_crossing,
                outside_node_ids,
            )
            return outside_node_ids, has_crossing, num_outside, max_dist

        except Exception as exc:
            logger.warning("Failed to identify outside basis nodes: %s", exc)
            self._outside_nodes = []
            return [], False, 0, 0.0

    def optimize(self) -> MorphologyGraph:
        """Optimize the morphology basis via pruning, snapping, and forcing."""
        logger.info("Starting basis optimization...")
        logger.info("  Nodes: %d", self.graph.number_of_nodes())

        if self.options.do_pruning:
            self._run_pruning_phase()
        else:
            logger.debug("Skipping pruning phase because do_pruning is False")

        if self.options.do_snapping:
            self._run_snapping_phase()
        else:
            logger.debug("Skipping snapping phase because do_snapping is False")

        if self.options.do_forcing:
            self._run_forcing_phase()
        else:
            logger.debug("Skipping forcing phase because do_forcing is False")

        self._update_edge_lengths()
        self.get_outside_nodes()
        logger.info("Basis optimization complete")
        return self.graph

    def _run_pruning_phase(self) -> None:
        """Prune short terminal branches before geometric optimization."""
        logger.info("Phase 0 - Pruning")
        if self.graph.number_of_nodes() == 0:
            logger.debug("Skipping pruning because basis graph is empty")
            return

        threshold = self._resolve_pruning_threshold()
        if threshold is None:
            logger.debug("Skipping pruning because no pruning threshold was configured")
            return

        logger.info("  Removing branches with length < %.4f", threshold)
        current = self.graph.copy()
        while True:
            terminal_nodes = sorted(current.get_terminal_nodes())
            nodes_to_remove: set[int] = set()
            visited_terminals: set[int] = set()

            for terminal in terminal_nodes:
                if terminal in visited_terminals or terminal not in current:
                    continue

                end, path, length = self._trace_from_terminal(current, terminal)
                if len(path) <= 1:
                    visited_terminals.add(terminal)
                    continue

                visited_terminals.add(terminal)
                if end != terminal and current.degree(end) == 1:
                    visited_terminals.add(end)

                is_isolated = end != terminal and current.degree(end) == 1
                ends_at_branch = end != terminal and current.degree(end) >= 3

                should_remove = is_isolated or (ends_at_branch and length < threshold)
                if not should_remove:
                    continue

                if ends_at_branch:
                    nodes_to_remove.update(path[:-1])
                else:
                    nodes_to_remove.update(path)

            if not nodes_to_remove:
                break

            logger.debug(
                "Pruning %d nodes from short branches",
                len(nodes_to_remove),
            )
            current.remove_nodes_from([n for n in nodes_to_remove if n in current])
            if not self.options.pruning_iterative:
                break

        self.graph = current

    def _resolve_pruning_threshold(self) -> Optional[float]:
        """Resolve the branch-pruning threshold from absolute or percentile input."""
        if self.options.pruning_min_length is not None:
            return float(self.options.pruning_min_length)

        percentile = self.options.pruning_min_length_percentile
        if percentile is None:
            return None

        branch_lengths = list(self._compute_branch_lengths(self.graph).values())
        if not branch_lengths:
            return None
        return float(np.percentile(branch_lengths, float(percentile)))

    def _run_snapping_phase(self) -> None:
        """Snap outside basis nodes back into the mesh."""
        outside_node_ids, has_crossing, num_outside, _ = self.get_outside_nodes()
        logger.info("Phase 1 - Snapping: %d nodes outside mesh", num_outside)
        logger.debug("Snapping candidate nodes: %s", outside_node_ids)

        if not has_crossing:
            logger.debug("No snapping required because all nodes are inside the mesh")
            return

        for node in outside_node_ids:
            pos = self.graph.get_node_position(node)
            direction, dist = self._compute_snap_direction(pos)
            if dist < 1e-10:
                logger.debug(
                    "Skipping snap for node %s because distance is negligible",
                    node,
                )
                continue
            displacement = direction * dist * self.options.snap_distance_multiplier
            logger.debug(
                "Snapping node %s from %s with distance %.6f and displacement %s",
                node,
                pos,
                dist,
                displacement,
            )
            self.graph.set_node_position(node, pos + displacement)

    def _run_forcing_phase(self) -> None:
        """Iteratively move basis nodes toward the medial axis."""
        logger.info("Phase 2 - Forcing: max %d iterations", self.options.max_iterations)
        terminal_nodes = (
            self.graph.get_terminal_nodes()
            if self.options.preserve_terminal_nodes
            else set()
        )
        branch_nodes = (
            self.graph.get_branch_nodes()
            if self.options.preserve_branch_nodes
            else set()
        )

        for iteration in range(self.options.max_iterations):
            old_positions = self.graph.get_all_positions()
            if old_positions.size == 0:
                logger.info("Phase 2 - Forcing skipped because basis graph is empty")
                break

            for node in self.graph.nodes():
                if node in terminal_nodes or node in branch_nodes:
                    continue

                pos = self.graph.get_node_position(node)
                direction = self._compute_centering_direction(pos)

                smoothing_direction = np.zeros(3)
                if self.options.smoothing_weight > 0:
                    smoothing_direction = self._compute_smoothing_direction_for_node(
                        node
                    )

                total_direction = (
                    1.0 - self.options.smoothing_weight
                ) * direction + self.options.smoothing_weight * smoothing_direction
                self.graph.set_node_position(
                    node,
                    pos + self.options.step_size * total_direction,
                )

            movement = self._average_movement(
                old_positions,
                self.graph.get_all_positions(),
            )
            logger.info("  Iteration %d: avg movement = %.6f", iteration, movement)
            if movement < self.options.convergence_threshold:
                logger.info("  Converged at iteration %d", iteration)
                break

    def _update_edge_lengths(self) -> None:
        """Update edge lengths after node positions have changed."""
        for u, v in self.graph.edges():
            pos_u = self.graph.get_node_position(u)
            pos_v = self.graph.get_node_position(v)
            self.graph.edges[u, v]["length"] = float(np.linalg.norm(pos_v - pos_u))

    def _compute_smoothing_direction_for_node(self, node: int) -> np.ndarray:
        """Compute a unit smoothing direction from the node's neighbors."""
        neighbors = list(self.graph.neighbors(node))
        if not neighbors:
            return np.zeros(3)

        pos = self.graph.get_node_position(node)
        neighbor_positions = np.array(
            [self.graph.get_node_position(n) for n in neighbors],
            dtype=float,
        )
        direction = neighbor_positions.mean(axis=0) - pos
        norm = np.linalg.norm(direction)
        if norm > 1e-10:
            return direction / norm
        return np.zeros(3)

    def _compute_centering_direction(self, point: np.ndarray) -> np.ndarray:
        """Compute a direction that moves a point toward the medial axis."""
        is_inside = self.mesh.contains(point.reshape(1, 3))[0]
        if not is_inside:
            return self._compute_closest_point_direction(point)

        try:
            directions = self._get_uniform_sphere_directions(self.options.n_rays)
            force = np.zeros(3)
            for direction in directions:
                distance = self._ray_distance_to_surface(point, direction)
                if distance > 1e-6:
                    force -= direction / distance
            force_mag = np.linalg.norm(force)
            if force_mag > 1e-10:
                return force / force_mag
            return np.zeros(3)
        except Exception as exc:
            logger.warning("Failed to compute centering direction: %s", exc)
            return self._compute_closest_point_direction(point)

    def _compute_snap_direction(self, point: np.ndarray) -> Tuple[np.ndarray, float]:
        """Return the direction and distance to the nearest mesh point."""
        try:
            from trimesh.proximity import closest_point

            cp, _, _ = closest_point(self.mesh, point.reshape(1, 3))
            surface_point = cp[0]
            to_surface = surface_point - point
            dist = float(np.linalg.norm(to_surface))
            if dist < 1e-10:
                return np.zeros(3), 0.0
            return to_surface / dist, dist
        except Exception as exc:
            logger.warning("Failed to compute snap direction: %s", exc)
            return np.zeros(3), 0.0

    def _compute_closest_point_direction(self, point: np.ndarray) -> np.ndarray:
        """Fallback for outside points: move toward the closest mesh point."""
        direction, _ = self._compute_snap_direction(point)
        return direction

    def _get_uniform_sphere_directions(self, n_points: int) -> np.ndarray:
        """Generate approximately uniform directions on the unit sphere."""
        if n_points == 6:
            return np.array(
                [
                    [1.0, 0.0, 0.0],
                    [-1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, -1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, -1.0],
                ]
            )

        indices = np.arange(0, n_points, dtype=float) + 0.5
        phi = (1 + np.sqrt(5)) / 2
        theta = 2 * np.pi * indices / phi
        z = 1 - (2 * indices / n_points)
        radius = np.sqrt(1 - z * z)
        directions = np.column_stack(
            [radius * np.cos(theta), radius * np.sin(theta), z]
        )
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        return directions / (norms + 1e-10)

    def _ray_distance_to_surface(
        self, point: np.ndarray, direction: np.ndarray
    ) -> float:
        """Compute distance from a point to the mesh surface along a ray."""
        try:
            locations, _, _ = self.mesh.ray.intersects_location(
                ray_origins=point.reshape(1, 3),
                ray_directions=direction.reshape(1, 3),
            )
            if len(locations) == 0:
                return self.options.fallback_distance
            return float(np.min(np.linalg.norm(locations - point, axis=1)))
        except Exception as exc:
            logger.warning("Ray tracing failed, using fallback distance: %s", exc)
            return self.options.fallback_distance

    def get_optimization_stats(self) -> dict:
        """Return summary statistics for the optimized basis graph."""
        _, _, num_outside, max_dist = self.get_outside_nodes()
        return {
            "surface_crossing_detected": self._surface_crossing_detected,
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "num_terminal_nodes": len(self.graph.get_terminal_nodes()),
            "num_branch_nodes": len(self.graph.get_branch_nodes()),
            "total_length": self.graph.get_total_length(),
            "nodes_outside_mesh": num_outside,
            "max_distance_outside": max_dist,
        }

    def _compute_branch_lengths(self, graph: Graph3D) -> dict[tuple[int, int], float]:
        """Compute terminal-to-branch lengths for pruning thresholding."""
        branch_lengths: dict[tuple[int, int], float] = {}
        for terminal in graph.get_terminal_nodes():
            end, _, length = self._trace_from_terminal(graph, terminal)
            if end != terminal:
                branch_lengths[(terminal, end)] = length
        return branch_lengths

    def _trace_from_terminal(
        self,
        graph: Graph3D,
        start: int,
    ) -> tuple[int, list[int], float]:
        """Trace from a terminal node until degree differs from 2."""
        if start not in graph or graph.degree(start) != 1:
            return start, [start], 0.0

        path = [start]
        prev = None
        current = start
        length = 0.0

        while True:
            nbrs = list(graph.neighbors(current))
            if prev is not None:
                nbrs = [n for n in nbrs if n != prev]
            if not nbrs:
                break
            nxt = nbrs[0]
            length += self._edge_length(graph, current, nxt)
            prev, current = current, nxt
            path.append(current)
            if graph.degree(current) != 2:
                break

        return current, path, length

    def _edge_length(self, graph: Graph3D, u: int, v: int) -> float:
        """Return an edge length, computing it from geometry if needed."""
        data = graph.get_edge_data(u, v) or {}
        length = data.get("length")
        if length is not None:
            return float(length)
        return float(
            np.linalg.norm(graph.get_node_position(v) - graph.get_node_position(u))
        )

    def _average_movement(
        self,
        old_positions: np.ndarray,
        new_positions: np.ndarray,
    ) -> float:
        """Return the mean per-node displacement between two position arrays."""
        if old_positions.size == 0 or new_positions.size == 0:
            return 0.0
        return float(np.linalg.norm(new_positions - old_positions, axis=1).mean())
