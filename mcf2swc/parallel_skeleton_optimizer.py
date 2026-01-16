"""
Parallel skeleton optimization for MCF-generated polylines.

This module provides parallelized optimization for skeleton polylines produced by
mean-curvature flow (MCF) calculations, focusing on speeding up the most
computationally expensive operations through multithreading and multiprocessing.

Key optimizations:
- Parallel ray tracing for centering direction computation
- Batch processing of multiple nodes
- Concurrent surface crossing detection
- Memory-efficient parallel processing
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass
from typing import Optional, Tuple, List
import multiprocessing as mp

import numpy as np
import trimesh

from .skeleton import SkeletonGraph
from .skeleton_optimizer import SkeletonOptimizerOptions

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@dataclass
class ParallelOptimizerOptions:
    """
    Configuration for parallel skeleton optimization.

    Attributes:
        max_workers: Maximum number of worker threads/processes.
                    If None, uses CPU count. Default: None
        use_processes: If True, use ProcessPoolExecutor instead of ThreadPoolExecutor.
                      Better for CPU-bound tasks. Default: True
        batch_size: Number of nodes to process in each batch. Default: 10
        ray_batch_size: Number of rays to process in parallel for each node.
                       Default: 6 (all rays at once)
        enable_ray_parallel: Enable parallel ray tracing. Default: True
        enable_node_parallel: Enable parallel node processing. Default: True
        chunk_size: Chunk size for parallel operations. Default: 1
    """

    max_workers: Optional[int] = None
    use_processes: bool = True
    batch_size: int = 10
    ray_batch_size: int = 6
    enable_ray_parallel: bool = True
    enable_node_parallel: bool = True
    chunk_size: int = 1


class ParallelSkeletonOptimizer:
    """
    Parallel optimizer for skeleton graphs to push nodes toward the mesh medial axis.

    This class provides the same functionality as SkeletonOptimizer but with
    parallel processing capabilities to significantly speed up computation on
    multi-core systems.
    """

    def __init__(
        self,
        skeleton: SkeletonGraph,
        mesh: trimesh.Trimesh,
        options: Optional[SkeletonOptimizerOptions] = None,
        parallel_options: Optional[ParallelOptimizerOptions] = None,
    ):
        """
        Initialize the parallel skeleton optimizer.

        Args:
            skeleton: Input skeleton graph from MCF calculation
            mesh: Target mesh that the skeleton should approximate
            options: Optimization configuration options
            parallel_options: Parallel processing configuration options
        """
        self.skeleton = skeleton.copy_skeleton()
        self.mesh = mesh
        self.options = options or SkeletonOptimizerOptions()
        self.parallel_options = parallel_options or ParallelOptimizerOptions()

        # Determine number of workers
        if self.parallel_options.max_workers is None:
            self.parallel_options.max_workers = mp.cpu_count()

        self._surface_crossing_detected = False
        self._optimization_history = []

    def check_surface_crossing(self) -> Tuple[bool, int, float]:
        """
        Parallel check if any skeleton nodes are outside the mesh surface.

        Returns:
            Tuple of (has_crossing, num_outside_nodes, max_distance)
        """
        if self.skeleton.number_of_nodes() == 0:
            return False, 0, 0.0

        all_pts = self.skeleton.get_all_positions()

        if (
            self.parallel_options.enable_node_parallel
            and len(all_pts) > self.parallel_options.batch_size
        ):
            return self._check_surface_crossing_parallel(all_pts)
        else:
            return self._check_surface_crossing_sequential(all_pts)

    def _check_surface_crossing_sequential(
        self, all_pts: np.ndarray
    ) -> Tuple[bool, int, float]:
        """Sequential surface crossing check (fallback)."""
        try:
            inside_mask = self.mesh.contains(all_pts)
            outside_mask = ~inside_mask
            num_outside = int(np.sum(outside_mask))

            max_dist = 0.0
            if num_outside > 0:
                from trimesh.proximity import closest_point

                outside_pts = all_pts[outside_mask]
                cp, distances, _ = closest_point(self.mesh, outside_pts)
                max_dist = float(np.max(distances))

            has_crossing = num_outside > 0
            self._surface_crossing_detected = has_crossing

            if self.options.verbose:
                if has_crossing:
                    logger.info(
                        "Surface crossing detected: %d/%d nodes outside mesh (max distance: %.4f)",
                        num_outside,
                        len(all_pts),
                        max_dist,
                    )
                else:
                    logger.info("No surface crossing detected - all nodes inside mesh")

            return has_crossing, num_outside, max_dist

        except Exception as e:
            logger.warning("Failed to check surface crossing: %s", e)
            return False, 0, 0.0

    def _check_surface_crossing_parallel(
        self, all_pts: np.ndarray
    ) -> Tuple[bool, int, float]:
        """Parallel surface crossing check using batching."""
        try:
            # Process points in batches
            batch_size = self.parallel_options.batch_size
            batches = [
                all_pts[i : i + batch_size] for i in range(0, len(all_pts), batch_size)
            ]

            inside_results = []
            closest_point_results = []

            executor_class = (
                ProcessPoolExecutor
                if self.parallel_options.use_processes
                else ThreadPoolExecutor
            )

            with executor_class(
                max_workers=self.parallel_options.max_workers
            ) as executor:
                # Check containment in parallel
                inside_futures = [
                    executor.submit(self.mesh.contains, batch) for batch in batches
                ]

                # Collect results
                for future in inside_futures:
                    inside_results.append(future.result())

            # Combine results
            inside_mask = np.concatenate(inside_results)
            outside_mask = ~inside_mask
            num_outside = int(np.sum(outside_mask))

            max_dist = 0.0
            if num_outside > 0:
                outside_pts = all_pts[outside_mask]

                # Process outside points in parallel for closest point computation
                outside_batches = [
                    outside_pts[i : i + batch_size]
                    for i in range(0, len(outside_pts), batch_size)
                ]

                with executor_class(
                    max_workers=self.parallel_options.max_workers
                ) as executor:
                    closest_futures = [
                        executor.submit(self._compute_closest_points_batch, batch)
                        for batch in outside_batches
                    ]

                    for future in closest_futures:
                        distances = future.result()
                        max_dist = max(
                            max_dist,
                            float(np.max(distances)) if len(distances) > 0 else 0.0,
                        )

            has_crossing = num_outside > 0
            self._surface_crossing_detected = has_crossing

            if self.options.verbose:
                if has_crossing:
                    logger.info(
                        "Parallel surface crossing detected: %d/%d nodes outside mesh (max distance: %.4f)",
                        num_outside,
                        len(all_pts),
                        max_dist,
                    )
                else:
                    logger.info("No surface crossing detected - all nodes inside mesh")

            return has_crossing, num_outside, max_dist

        except Exception as e:
            logger.warning(
                "Parallel surface crossing check failed, falling back to sequential: %s",
                e,
            )
            return self._check_surface_crossing_sequential(all_pts)

    def _compute_closest_points_batch(self, points: np.ndarray) -> np.ndarray:
        """Compute closest point distances for a batch of points."""
        try:
            from trimesh.proximity import closest_point

            _, distances, _ = closest_point(self.mesh, points)
            return distances
        except Exception as e:
            logger.warning("Failed to compute closest points for batch: %s", e)
            return np.array([])

    def optimize(self) -> SkeletonGraph:
        """
        Parallel optimize the skeleton by pushing nodes toward the mesh medial axis.

        Returns:
            Optimized skeleton graph
        """
        if self.options.check_surface_crossing:
            self.check_surface_crossing()

        if self.options.verbose:
            logger.info("Starting parallel skeleton optimization...")
            logger.info("  Nodes: %d", self.skeleton.number_of_nodes())
            logger.info("  Max iterations: %d", self.options.max_iterations)
            logger.info("  Step size: %.4f", self.options.step_size)
            logger.info("  Smoothing weight: %.4f", self.options.smoothing_weight)
            logger.info("  Max workers: %d", self.parallel_options.max_workers)
            logger.info("  Use processes: %s", self.parallel_options.use_processes)

        # Get node sets for preservation
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

        # Optimization loop
        for iteration in range(self.options.max_iterations):
            # Store old positions
            old_positions = self.skeleton.get_all_positions()

            # Optimize nodes in parallel batches (using same logic as sequential)
            if (
                self.parallel_options.enable_node_parallel
                and self.skeleton.number_of_nodes() > 1
            ):
                self._optimize_nodes_parallel_batch_all(terminal_nodes, branch_nodes)
            else:
                self._optimize_nodes_sequential_all(terminal_nodes, branch_nodes)

            # Check convergence
            new_positions = self.skeleton.get_all_positions()
            movement = np.linalg.norm(new_positions - old_positions, axis=1).mean()

            if self.options.verbose and iteration % 10 == 0:
                logger.info("  Iteration %d: avg movement = %.6f", iteration, movement)

            if movement < self.options.convergence_threshold:
                if self.options.verbose:
                    logger.info("  Converged at iteration %d", iteration)
                break

        # Update edge lengths after optimization
        self._update_edge_lengths()

        if self.options.verbose:
            logger.info("Parallel optimization complete")

        return self.skeleton

    def _optimize_nodes_sequential_all(
        self, terminal_nodes: set, branch_nodes: set
    ) -> None:
        """Sequential node optimization using same logic as original."""
        for node in self.skeleton.nodes():
            # Skip terminal nodes if preserve_terminal_nodes is True
            if node in terminal_nodes:
                continue

            # Skip branch nodes if preserve_branch_nodes is True
            if node in branch_nodes:
                continue

            self._optimize_single_node(node)

    def _optimize_nodes_parallel_batch_all(
        self, terminal_nodes: set, branch_nodes: set
    ) -> None:
        """Optimize nodes in parallel batches using same logic as original."""
        # Get all nodes and filter them like the sequential version
        all_nodes = list(self.skeleton.nodes())
        optimizable_nodes = [
            node
            for node in all_nodes
            if node not in terminal_nodes and node not in branch_nodes
        ]

        if len(optimizable_nodes) == 0:
            return

        batch_size = self.parallel_options.batch_size
        node_batches = [
            optimizable_nodes[i : i + batch_size]
            for i in range(0, len(optimizable_nodes), batch_size)
        ]

        executor_class = (
            ProcessPoolExecutor
            if self.parallel_options.use_processes
            else ThreadPoolExecutor
        )

        with executor_class(max_workers=self.parallel_options.max_workers) as executor:
            # Submit batch optimization tasks
            futures = [
                executor.submit(self._optimize_node_batch, batch)
                for batch in node_batches
            ]

            # Apply results as they complete
            for future in futures:
                node_updates = future.result()
                for node_id, new_position in node_updates:
                    self.skeleton.set_node_position(node_id, new_position)

    def _optimize_node_batch(self, node_ids: List[int]) -> List[Tuple[int, np.ndarray]]:
        """Optimize a batch of nodes and return position updates."""
        updates = []

        for node_id in node_ids:
            # Get current position
            pos = self.skeleton.get_node_position(node_id)

            # Compute centering direction (with parallel ray tracing if enabled)
            if self.parallel_options.enable_ray_parallel:
                direction = self._compute_centering_direction_parallel(pos)
            else:
                direction = self._compute_centering_direction_sequential(pos)

            # Compute smoothing direction if needed
            smoothing_direction = np.zeros(3)
            if self.options.smoothing_weight > 0:
                smoothing_direction = self._compute_smoothing_direction_for_node(
                    node_id
                )

            # Combine directions
            total_direction = (
                1.0 - self.options.smoothing_weight
            ) * direction + self.options.smoothing_weight * smoothing_direction

            # Update position
            new_pos = pos + self.options.step_size * total_direction
            updates.append((node_id, new_pos))

        return updates

    def _optimize_single_node(self, node_id: int) -> None:
        """Optimize a single node (used in sequential mode)."""
        # Get current position
        pos = self.skeleton.get_node_position(node_id)

        # Compute centering direction
        if self.parallel_options.enable_ray_parallel:
            direction = self._compute_centering_direction_parallel(pos)
        else:
            direction = self._compute_centering_direction_sequential(pos)

        # Compute smoothing direction if needed
        smoothing_direction = np.zeros(3)
        if self.options.smoothing_weight > 0:
            smoothing_direction = self._compute_smoothing_direction_for_node(node_id)

        # Combine directions
        total_direction = (
            1.0 - self.options.smoothing_weight
        ) * direction + self.options.smoothing_weight * smoothing_direction

        # Update position
        new_pos = pos + self.options.step_size * total_direction
        self.skeleton.set_node_position(node_id, new_pos)

    def _compute_centering_direction_sequential(self, point: np.ndarray) -> np.ndarray:
        """Sequential centering direction computation (fallback)."""
        # Check if point is inside the mesh
        is_inside = self.mesh.contains(point.reshape(1, 3))[0]
        if not is_inside:
            # Point is outside - move toward closest surface point
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
                if d > 1e-6:
                    force -= direction / d

            # Normalize to unit vector
            force_mag = np.linalg.norm(force)
            if force_mag > 1e-10:
                return force / force_mag
            else:
                return np.zeros(3)

        except Exception as e:
            logger.warning("Failed to compute centering direction: %s", e)
            return self._compute_closest_point_direction(point)

    def _compute_centering_direction_parallel(self, point: np.ndarray) -> np.ndarray:
        """Parallel centering direction computation using ray tracing."""
        # Check if point is inside the mesh
        is_inside = self.mesh.contains(point.reshape(1, 3))[0]
        if not is_inside:
            # Point is outside - move toward closest surface point
            return self._compute_closest_point_direction(point)

        try:
            # Get uniformly distributed ray directions
            directions = self._get_uniform_sphere_directions(self.options.n_rays)

            # Compute ray distances in parallel
            ray_distances = self._compute_ray_distances_parallel(point, directions)

            # Compute force based on distance imbalance
            force = np.zeros(3)
            for direction, distance in zip(directions, ray_distances):
                # Force is inversely proportional to distance
                if distance > 1e-6:
                    force -= direction / distance

            # Normalize to unit vector
            force_mag = np.linalg.norm(force)
            if force_mag > 1e-10:
                return force / force_mag
            else:
                return np.zeros(3)

        except Exception as e:
            logger.warning(
                "Parallel centering direction computation failed, falling back to sequential: %s",
                e,
            )
            return self._compute_centering_direction_sequential(point)

    def _compute_ray_distances_parallel(
        self, point: np.ndarray, directions: np.ndarray
    ) -> List[float]:
        """Compute ray distances in parallel."""
        ray_batch_size = self.parallel_options.ray_batch_size

        # If we have few rays, process them all at once
        if len(directions) <= ray_batch_size:
            return self._compute_ray_distances_batch(point, directions)

        # Otherwise, process in batches
        direction_batches = [
            directions[i : i + ray_batch_size]
            for i in range(0, len(directions), ray_batch_size)
        ]

        all_distances = []
        executor_class = (
            ProcessPoolExecutor
            if self.parallel_options.use_processes
            else ThreadPoolExecutor
        )

        with executor_class(max_workers=self.parallel_options.max_workers) as executor:
            futures = [
                executor.submit(self._compute_ray_distances_batch, point, batch)
                for batch in direction_batches
            ]

            for future in futures:
                batch_distances = future.result()
                all_distances.extend(batch_distances)

        return all_distances

    def _compute_ray_distances_batch(
        self, point: np.ndarray, directions: np.ndarray
    ) -> List[float]:
        """Compute ray distances for a batch of directions."""
        try:
            # Cast multiple rays at once
            ray_origins = np.tile(point, (len(directions), 1))
            ray_directions = directions

            # Find intersections with the mesh
            locations, index_ray, index_tri = self.mesh.ray.intersects_location(
                ray_origins=ray_origins, ray_directions=ray_directions
            )

            # Organize results by ray
            distances = [self.options.fallback_distance] * len(directions)

            if len(locations) > 0:
                # Group intersections by ray index
                for i, ray_idx in enumerate(index_ray):
                    if ray_idx < len(distances):
                        dist = np.linalg.norm(locations[i] - point)
                        distances[ray_idx] = min(distances[ray_idx], dist)

            return distances

        except Exception as e:
            logger.warning("Ray tracing batch failed: %s", e)
            return [self.options.fallback_distance] * len(directions)

    def _compute_smoothing_direction_for_node(self, node: int) -> np.ndarray:
        """
        Compute smoothing direction for a node based on its neighbors.

        Args:
            node: Node ID

        Returns:
            (3,) array representing the smoothing direction (unit vector)
        """
        neighbors = list(self.skeleton.neighbors(node))

        if len(neighbors) == 0:
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
            return direction / norm
        else:
            return np.zeros(3)

    def _compute_closest_point_direction(self, point: np.ndarray) -> np.ndarray:
        """
        Fallback method for points outside the mesh: move toward closest surface point.

        Args:
            point: (3,) array representing a single point

        Returns:
            (3,) array representing the direction to move (unit vector)
        """
        try:
            from trimesh.proximity import closest_point

            cp, _, _ = closest_point(self.mesh, point.reshape(1, 3))
            surface_point = cp[0]

            to_surface = surface_point - point
            dist_to_surface = np.linalg.norm(to_surface)

            if dist_to_surface < 1e-10:
                return np.zeros(3)

            return to_surface / dist_to_surface

        except Exception as e:
            logger.warning("Failed to compute closest point direction: %s", e)
            return np.zeros(3)

    def _get_uniform_sphere_directions(self, n_points: int) -> np.ndarray:
        """
        Generate uniformly distributed points on a unit sphere using Fibonacci spiral.

        Args:
            n_points: Number of points to generate

        Returns:
            (n_points, 3) array of unit direction vectors
        """
        # Special case: axis-aligned rays for debugging
        if n_points == 6:
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

        # Normalize to ensure unit vectors
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        directions = directions / (norms + 1e-10)

        return directions

    def _ray_distance_to_surface(
        self, point: np.ndarray, direction: np.ndarray
    ) -> float:
        """
        Compute distance from a point to the mesh surface along a ray direction.

        This is the fallback sequential ray tracing method.
        """
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
                return self.options.fallback_distance

            # Find the closest intersection point
            distances = np.linalg.norm(locations - point, axis=1)
            min_dist = float(np.min(distances))

            return min_dist

        except Exception as e:
            logger.warning("Ray tracing failed, using fallback distance: %s", e)
            return self.options.fallback_distance

    def _update_edge_lengths(self) -> None:
        """Update edge lengths after node positions have changed."""
        for u, v in self.skeleton.edges():
            pos_u = self.skeleton.get_node_position(u)
            pos_v = self.skeleton.get_node_position(v)
            length = float(np.linalg.norm(pos_v - pos_u))
            self.skeleton.edges[u, v]["length"] = length

    def get_optimization_stats(self) -> dict:
        """
        Get statistics about the optimization process.

        Returns:
            Dictionary containing optimization statistics
        """
        stats = {
            "surface_crossing_detected": self._surface_crossing_detected,
            "num_nodes": self.skeleton.number_of_nodes(),
            "num_edges": self.skeleton.number_of_edges(),
            "num_terminal_nodes": len(self.skeleton.get_terminal_nodes()),
            "num_branch_nodes": len(self.skeleton.get_branch_nodes()),
            "total_length": self.skeleton.get_total_length(),
            "max_workers": self.parallel_options.max_workers,
            "use_processes": self.parallel_options.use_processes,
            "enable_ray_parallel": self.parallel_options.enable_ray_parallel,
            "enable_node_parallel": self.parallel_options.enable_node_parallel,
        }

        if self.options.check_surface_crossing:
            has_crossing, num_outside, max_dist = self.check_surface_crossing()
            stats["nodes_outside_mesh"] = num_outside
            stats["max_distance_outside"] = max_dist

        return stats
