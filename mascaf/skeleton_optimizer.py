"""
Skeleton optimization for MCF-generated polylines.

This module provides optimization for skeleton polylines produced by mean-curvature
flow (MCF) calculations. MCF skeletons reproduce general topology well but can deviate
from the true medial axis and sometimes clip through the mesh surface, especially in
regions with holes or high curvature.

The SkeletonOptimizer gently pushes skeleton points toward the center of the mesh
volume while preserving the overall topology and structure.

Key features:
- Detection of skeleton points that cross the mesh surface
- Optimization to push points toward the medial axis
- Preservation of skeleton topology and connectivity
- Configurable optimization parameters
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import trimesh

from .skeleton import SkeletonGraph

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@dataclass
class SkeletonOptimizerOptions:
    """
    Configuration for skeleton optimization.

    Attributes:
        do_snapping: If True, run phase 1 which snaps nodes outside the mesh
            surface back inside. Default: True
        do_forcing: If True, run phase 2 which iteratively pushes nodes toward
            the medial axis. Default: True
        snap_distance_multiplier: Multiplier applied to the displacement vector
            during snapping. Values > 1.0 overshoot slightly to ensure the node
            lands inside the mesh. Default: 1.1
        max_iterations: Maximum number of optimization iterations. Default: 100
        step_size: Step size for moving points toward the center. Smaller values
            are more conservative. Default: 0.1
        convergence_threshold: Stop optimization when average point movement
            is below this threshold. Default: 1e-4
        preserve_terminal_nodes: If True, do not move terminal nodes (degree 1).
            Default: True
        preserve_branch_nodes: If True, do not move branch nodes (degree 3+).
            Default: False
        n_rays: Number of evenly spaced rays to cast in 3D for distance sampling.
            Uses Fibonacci sphere algorithm for uniform distribution. If set to 6,
            uses axis-aligned rays (+/- x, y, z) for simpler debugging. Default: 6
        fallback_distance: Distance to use when ray tracing fails to find an
            intersection with the mesh. Default: 10.0
        smoothing_weight: Weight for smoothing regularization to maintain
            skeleton smoothness (0 = no smoothing, 1 = strong smoothing).
            Default: 0.5
    """

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


class SkeletonOptimizer:
    """
    Optimizer for skeleton graphs to push nodes toward the mesh medial axis.

    This class takes a skeleton graph produced by MCF skeletonization
    and a mesh, then optimizes the node positions to better approximate the
    medial axis of the mesh volume.

    Example:
        >>> from mascaf.skeleton import SkeletonGraph
        >>> from mascaf import MeshManager, SkeletonOptimizer
        >>> skeleton = SkeletonGraph.from_txt("skeleton.polylines.txt")
        >>> mesh_mgr = MeshManager(mesh_path="mesh.obj")
        >>> optimizer = SkeletonOptimizer(skeleton, mesh_mgr.mesh)
        >>> optimized_skeleton = optimizer.optimize()
    """

    def __init__(
        self,
        skeleton: SkeletonGraph,
        mesh: trimesh.Trimesh,
        options: Optional[SkeletonOptimizerOptions] = None,
    ):
        """
        Initialize the skeleton optimizer.

        Args:
            skeleton: Input skeleton graph from MCF calculation
            mesh: Target mesh that the skeleton should approximate
            options: Optimization configuration options
        """
        self.skeleton = skeleton.copy_skeleton()
        self.mesh = mesh
        self.options = options or SkeletonOptimizerOptions()

        self._surface_crossing_detected = False
        self._outside_nodes: list = []
        self._optimization_history = []
        logger.debug(
            "Initialized SkeletonOptimizer with %d nodes, %d edges, "
            "do_snapping=%s, do_forcing=%s",
            self.skeleton.number_of_nodes(),
            self.skeleton.number_of_edges(),
            self.options.do_snapping,
            self.options.do_forcing,
        )

    def get_outside_nodes(self) -> Tuple[list, bool, int, float]:
        """
        Identify skeleton nodes that are outside the mesh surface.

        Returns:
            Tuple of
            (outside_node_ids, has_crossing, num_outside_nodes, max_distance)
            - outside_node_ids: List of node IDs outside the mesh
            - has_crossing: True if any nodes are outside the mesh
            - num_outside_nodes: Number of nodes outside the mesh
            - max_distance: Maximum distance to surface for outside nodes
        """
        if self.skeleton.number_of_nodes() == 0:
            self._outside_nodes = []
            self._surface_crossing_detected = False
            logger.debug("Outside-node query on empty skeleton")
            return [], False, 0, 0.0

        node_ids = list(self.skeleton.nodes())
        all_pts = self.skeleton.get_all_positions()
        logger.debug(
            "Checking %d skeleton nodes against mesh containment", len(node_ids)
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
                cp, distances, _ = closest_point(self.mesh, outside_pts)
                max_dist = float(np.max(distances))

            has_crossing = num_outside > 0
            self._surface_crossing_detected = has_crossing
            self._outside_nodes = outside_node_ids
            logger.debug(
                "Outside-node detection complete: has_crossing=%s, outside_nodes=%s",
                has_crossing,
                outside_node_ids,
            )

            if has_crossing:
                logger.info(
                    "Surface crossing detected: %d/%d nodes outside mesh (max distance: %.4f)",
                    num_outside,
                    len(all_pts),
                    max_dist,
                )
            else:
                logger.info("No surface crossing detected - all nodes inside mesh")

            return outside_node_ids, has_crossing, num_outside, max_dist

        except Exception as e:
            logger.warning("Failed to identify outside nodes: %s", e)
            self._outside_nodes = []
            return [], False, 0, 0.0

    def optimize(self) -> SkeletonGraph:
        """
        Optimize the skeleton in two phases.

        Phase 1 (snapping): Move any nodes outside the mesh surface back inside
        by displacing them toward the nearest surface point.

        Phase 2 (forcing): Iteratively push all movable nodes toward the medial
        axis using ray-based centering and smoothing forces.

        Returns:
            Optimized skeleton graph
        """
        logger.info("Starting skeleton optimization...")
        logger.info("  Nodes: %d", self.skeleton.number_of_nodes())
        logger.debug(
            "Optimization options: snap_distance_multiplier=%.4f, "
            "max_iterations=%d, step_size=%.4f, convergence_threshold=%.6g, "
            "preserve_terminal_nodes=%s, preserve_branch_nodes=%s, n_rays=%d, "
            "fallback_distance=%.4f, smoothing_weight=%.4f",
            self.options.snap_distance_multiplier,
            self.options.max_iterations,
            self.options.step_size,
            self.options.convergence_threshold,
            self.options.preserve_terminal_nodes,
            self.options.preserve_branch_nodes,
            self.options.n_rays,
            self.options.fallback_distance,
            self.options.smoothing_weight,
        )

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

        logger.info("Optimization complete")

        return self.skeleton

    def _run_snapping_phase(self) -> None:
        """
        Phase 1: Snap nodes that are outside the mesh surface back inside.

        For each outside node, the displacement vector from the node to the
        nearest surface point is scaled by snap_distance_multiplier and applied
        to move the node to (or slightly past) the surface.
        """
        outside_node_ids, has_crossing, num_outside, _ = self.get_outside_nodes()

        logger.info("Phase 1 - Snapping: %d nodes outside mesh", num_outside)
        logger.debug("Snapping candidate nodes: %s", outside_node_ids)

        if not has_crossing:
            logger.debug("No snapping required because all nodes are inside the mesh")
            return

        for node in outside_node_ids:
            pos = self.skeleton.get_node_position(node)
            direction, dist = self._compute_snap_direction(pos)
            if dist < 1e-10:
                logger.debug(
                    "Skipping snap for node %s because distance is negligible", node
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
            self.skeleton.set_node_position(node, pos + displacement)

        logger.info("Phase 1 - Snapping complete")

    def _run_forcing_phase(self) -> None:
        """
        Phase 2: Iteratively push nodes toward the medial axis.

        Each iteration visits every movable node and applies a centering force
        (and optional smoothing force) to nudge it toward the medial axis.
        """
        logger.info("Phase 2 - Forcing: max %d iterations", self.options.max_iterations)
        logger.info("  Step size: %.4f", self.options.step_size)
        logger.info("  Smoothing weight: %.4f", self.options.smoothing_weight)

        terminal_nodes = (
            self.skeleton.get_terminal_nodes()
            if self.options.preserve_terminal_nodes
            else set()
        )
        branch_nodes = (
            self.skeleton.get_branch_nodes()
            if self.options.preserve_branch_nodes
            else set()
        )
        logger.debug(
            "Protected nodes for forcing: %d terminal, %d branch",
            len(terminal_nodes),
            len(branch_nodes),
        )

        for iteration in range(self.options.max_iterations):
            old_positions = self.skeleton.get_all_positions()
            logger.debug("Starting forcing iteration %d", iteration)

            if old_positions.size == 0:
                logger.info("Phase 2 - Forcing skipped because skeleton is empty")
                break

            for node in self.skeleton.nodes():
                if node in terminal_nodes:
                    logger.debug("Skipping terminal node %s during forcing", node)
                    continue
                if node in branch_nodes:
                    logger.debug("Skipping branch node %s during forcing", node)
                    continue

                pos = self.skeleton.get_node_position(node)

                direction = self._compute_centering_direction(pos)

                smoothing_direction = np.zeros(3)
                if self.options.smoothing_weight > 0:
                    smoothing_direction = self._compute_smoothing_direction_for_node(
                        node
                    )

                total_direction = (
                    1.0 - self.options.smoothing_weight
                ) * direction + self.options.smoothing_weight * smoothing_direction

                new_pos = pos + self.options.step_size * total_direction
                logger.debug(
                    "Iteration %d node %s: pos=%s center=%s smooth=%s total=%s new_pos=%s",
                    iteration,
                    node,
                    pos,
                    direction,
                    smoothing_direction,
                    total_direction,
                    new_pos,
                )
                self.skeleton.set_node_position(node, new_pos)

            new_positions = self.skeleton.get_all_positions()
            if new_positions.size == 0:
                movement = 0.0
            else:
                movement = float(
                    np.linalg.norm(
                        new_positions - old_positions,
                        axis=1,
                    ).mean()
                )

            logger.info("  Iteration %d: avg movement = %.6f", iteration, movement)
            logger.debug(
                "Completed forcing iteration %d with movement %.6f", iteration, movement
            )

            if movement < self.options.convergence_threshold:
                logger.info("  Converged at iteration %d", iteration)
                logger.debug(
                    "Stopping forcing because movement %.6f is below threshold %.6f",
                    movement,
                    self.options.convergence_threshold,
                )
                break

        logger.info("Phase 2 - Forcing complete")

    def _update_edge_lengths(self) -> None:
        """Update edge lengths after node positions have changed."""
        logger.debug("Updating lengths for %d edges", self.skeleton.number_of_edges())
        for u, v in self.skeleton.edges():
            pos_u = self.skeleton.get_node_position(u)
            pos_v = self.skeleton.get_node_position(v)
            length = float(np.linalg.norm(pos_v - pos_u))
            self.skeleton.edges[u, v]["length"] = length
            logger.debug("Updated edge (%s, %s) length to %.6f", u, v, length)

    def _compute_smoothing_direction_for_node(self, node: int) -> np.ndarray:
        """
        Compute smoothing direction for a node based on its neighbors.

        The smoothing direction pulls the node toward the average position
        of its neighbors, helping maintain smooth skeleton structure.

        Args:
            node: Node ID

        Returns:
            (3,) array representing the smoothing direction (unit vector)
        """
        neighbors = list(self.skeleton.neighbors(node))
        logger.debug(
            "Computing smoothing direction for node %s with neighbors %s",
            node,
            neighbors,
        )

        if len(neighbors) == 0:
            logger.debug("Node %s has no neighbors; smoothing direction is zero", node)
            return np.zeros(3)

        # Get current position
        pos = self.skeleton.get_node_position(node)

        # Compute average neighbor position
        neighbor_positions = np.array(
            [self.skeleton.get_node_position(n) for n in neighbors]
        )
        avg_neighbor_pos = neighbor_positions.mean(axis=0)

        # Direction toward average neighbor position
        direction = avg_neighbor_pos - pos
        norm = np.linalg.norm(direction)

        if norm > 1e-10:
            logger.debug(
                "Smoothing direction for node %s: avg_neighbor_pos=%s, raw=%s, norm=%.6f",
                node,
                avg_neighbor_pos,
                direction,
                norm,
            )
            return direction / norm
        else:
            logger.debug(
                "Node %s already matches neighbor average; smoothing direction is zero",
                node,
            )
            return np.zeros(3)

    def _compute_centering_direction(self, point: np.ndarray) -> np.ndarray:
        """
        Compute the direction to move a point toward the medial axis.

        Uses uniform 3D ray sampling to find distances to the mesh surface
        in all directions, then computes a force that equalizes these distances.

        Args:
            point: (3,) array representing a single point

        Returns:
            (3,) array representing the direction to move (unit vector)
        """
        # Check if point is inside the mesh
        is_inside = self.mesh.contains(point.reshape(1, 3))[0]
        logger.debug(
            "Computing centering direction at point %s (inside=%s)", point, is_inside
        )
        if not is_inside:
            # Point is outside - move toward closest surface point
            logger.debug(
                "Point %s is outside mesh; using closest-point direction", point
            )
            return self._compute_closest_point_direction(point)

        try:
            # Get uniformly distributed ray directions
            directions = self._get_uniform_sphere_directions(self.options.n_rays)

            # Compute force based on distance imbalance
            force = np.zeros(3)
            for direction in directions:
                # Distance to surface in this direction
                d = self._ray_distance_to_surface(point, direction)

                # Force is inversely proportional to distance
                # Points should move away from closer surfaces
                # We use 1/d as the "pressure" from that direction
                if d > 1e-6:
                    force -= direction / d

            # Normalize to unit vector
            force_mag = np.linalg.norm(force)
            if force_mag > 1e-10:
                logger.debug(
                    "Centering force at point %s: raw_force=%s, magnitude=%.6f",
                    point,
                    force,
                    force_mag,
                )
                return force / force_mag
            else:
                logger.debug("Centering force at point %s is negligible", point)
                return np.zeros(3)

        except Exception as e:
            logger.warning("Failed to compute centering direction: %s", e)
            return self._compute_closest_point_direction(point)

    def _compute_snap_direction(self, point: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Compute the displacement direction and distance from an outside node
        to the nearest point on the mesh surface.

        This is the canonical snap direction used by the snapping phase.
        Replace this method to use a different snapping strategy.

        Args:
            point: (3,) array representing a node position outside the mesh

        Returns:
            Tuple of (direction, distance)
            - direction: (3,) unit vector pointing from the node toward the
              nearest surface point
            - distance: Euclidean distance to that surface point
        """
        try:
            from trimesh.proximity import closest_point

            cp, _, _ = closest_point(self.mesh, point.reshape(1, 3))
            surface_point = cp[0]

            to_surface = surface_point - point
            dist = float(np.linalg.norm(to_surface))

            if dist < 1e-10:
                logger.debug("Snap direction at point %s is zero-length", point)
                return np.zeros(3), 0.0

            logger.debug(
                "Snap direction at point %s: surface_point=%s, distance=%.6f",
                point,
                surface_point,
                dist,
            )
            return to_surface / dist, dist

        except Exception as e:
            logger.warning("Failed to compute snap direction: %s", e)
            return np.zeros(3), 0.0

    def _compute_closest_point_direction(self, point: np.ndarray) -> np.ndarray:
        """
        Fallback for points outside the mesh: move toward closest surface point.

        Args:
            point: (3,) array representing a single point

        Returns:
            (3,) array representing the direction to move (unit vector)
        """
        direction, _ = self._compute_snap_direction(point)
        logger.debug(
            "Closest-point fallback direction at point %s: %s", point, direction
        )
        return direction

    def _get_uniform_sphere_directions(self, n_points: int) -> np.ndarray:
        """
        Generate uniformly distributed points on a unit sphere using Fibonacci spiral.

        This gives approximately evenly spaced directions in 3D space.
        Special case: if n_points=6, uses exact axis-aligned rays (+/- x, y, z).

        Args:
            n_points: Number of points to generate

        Returns:
            (n_points, 3) array of unit direction vectors
        """
        # Special case: axis-aligned rays for debugging
        if n_points == 6:
            logger.debug("Using axis-aligned sphere directions for n_points=6")
            return np.array(
                [
                    [1.0, 0.0, 0.0],  # +X
                    [-1.0, 0.0, 0.0],  # -X
                    [0.0, 1.0, 0.0],  # +Y
                    [0.0, -1.0, 0.0],  # -Y
                    [0.0, 0.0, 1.0],  # +Z
                    [0.0, 0.0, -1.0],  # -Z
                ]
            )

        indices = np.arange(0, n_points, dtype=float) + 0.5

        # Golden ratio
        phi = (1 + np.sqrt(5)) / 2

        # Fibonacci sphere algorithm
        theta = 2 * np.pi * indices / phi
        z = 1 - (2 * indices / n_points)
        radius = np.sqrt(1 - z * z)

        x = radius * np.cos(theta)
        y = radius * np.sin(theta)

        directions = np.column_stack([x, y, z])

        # Normalize to ensure unit vectors (should already be, but for numerical stability)
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        directions = directions / (norms + 1e-10)

        logger.debug("Generated %d Fibonacci-sphere directions", n_points)

        return directions

    def _ray_distance_to_surface(
        self, point: np.ndarray, direction: np.ndarray
    ) -> float:
        """
        Compute distance from a point to the mesh surface along a ray direction.

        Uses ray tracing to find the exact intersection with the mesh surface.

        Args:
            point: (3,) array representing the ray origin
            direction: (3,) array representing the ray direction (should be normalized)

        Returns:
            Distance to the surface along the ray direction. Returns probe_distance
            if no intersection is found.
        """
        logger.debug("Tracing ray from point %s in direction %s", point, direction)
        try:
            # Cast a ray from the point in the given direction
            ray_origins = point.reshape(1, 3)
            ray_directions = direction.reshape(1, 3)

            # Find intersections with the mesh
            locations, index_ray, index_tri = self.mesh.ray.intersects_location(
                ray_origins=ray_origins, ray_directions=ray_directions
            )

            if len(locations) == 0:
                # No intersection found - use fallback distance
                logger.debug("Ray found no intersection, using fallback distance")
                return self.options.fallback_distance

            # Find the closest intersection point
            distances = np.linalg.norm(locations - point, axis=1)
            min_dist = float(np.min(distances))

            logger.debug(
                "Ray found %d intersections, using closest (%.4f)",
                len(locations),
                min_dist,
            )

            return min_dist

        except Exception as e:
            logger.warning("Ray tracing failed, using fallback distance: %s", e)
            return self.options.fallback_distance

    def get_optimization_stats(self) -> dict:
        """
        Get statistics about the optimization process.

        Returns:
            Dictionary containing optimization statistics
        """
        logger.debug("Collecting optimization statistics")
        stats = {
            "surface_crossing_detected": self._surface_crossing_detected,
            "num_nodes": self.skeleton.number_of_nodes(),
            "num_edges": self.skeleton.number_of_edges(),
            "num_terminal_nodes": len(self.skeleton.get_terminal_nodes()),
            "num_branch_nodes": len(self.skeleton.get_branch_nodes()),
            "total_length": self.skeleton.get_total_length(),
        }

        _, _, num_outside, max_dist = self.get_outside_nodes()
        stats["nodes_outside_mesh"] = num_outside
        stats["max_distance_outside"] = max_dist

        logger.debug("Optimization statistics: %s", stats)

        return stats
