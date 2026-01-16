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
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import trimesh

from .polylines import PolylinesSkeleton

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def _optimize_polyline_worker(
    polyline: np.ndarray,
    poly_idx: int,
    mesh: trimesh.Trimesh,
    options: SkeletonOptimizerOptions,
) -> np.ndarray:
    """
    Worker function for parallel polyline optimization.

    This function is defined at module level so it can be pickled for multiprocessing.

    Args:
        polyline: (N, 3) array of points
        poly_idx: Index of this polyline for logging
        mesh: Target mesh
        options: Optimization options

    Returns:
        Optimized (N, 3) array of points
    """
    # Handle empty polylines
    if len(polyline) == 0:
        return polyline.copy()

    # Create a temporary optimizer instance to use its methods
    # We can't pickle the full optimizer, but we can create one in the worker
    from .polylines import PolylinesSkeleton

    temp_skeleton = PolylinesSkeleton([polyline])
    temp_optimizer = SkeletonOptimizer(temp_skeleton, mesh, options)

    # Optimize this single polyline
    return temp_optimizer._optimize_polyline(polyline, poly_idx)


@dataclass
class SkeletonOptimizerOptions:
    """
    Configuration for skeleton optimization.

    Attributes:
        check_surface_crossing: If True, check whether skeleton points cross
            the mesh surface before optimization. Default: True
        max_iterations: Maximum number of optimization iterations. Default: 100
        step_size: Step size for moving points toward the center. Smaller values
            are more conservative. Default: 0.1
        convergence_threshold: Stop optimization when average point movement
            is below this threshold. Default: 1e-4
        preserve_endpoints: If True, do not move the first and last points of
            each polyline. Default: True
        preserve_branch_points: If True, do not move branch points (where 3+
            polylines meet). Requires topology detection. Default: False
        branch_point_tolerance: Distance threshold for detecting branch points.
            Default: 1e-6
        n_rays: Number of evenly spaced rays to cast in 3D for distance sampling.
            Uses Fibonacci sphere algorithm for uniform distribution. If set to 6,
            uses axis-aligned rays (+/- x, y, z) for simpler debugging. Default: 6
        fallback_distance: Distance to use when ray tracing fails to find an
            intersection with the mesh. Default: 10.0
        smoothing_weight: Weight for smoothing regularization to maintain
            polyline smoothness (0 = no smoothing, 1 = strong smoothing).
            Default: 0.5
        n_jobs: Number of parallel processes to use for optimizing polylines.
            If 1, run sequentially. If -1, use all available CPU cores.
            Default: 1 (sequential)
        verbose: If True, print optimization progress. Default: False
    """

    check_surface_crossing: bool = True
    max_iterations: int = 100
    step_size: float = 0.1
    convergence_threshold: float = 1e-4
    preserve_endpoints: bool = True
    preserve_branch_points: bool = False
    branch_point_tolerance: float = 1e-6
    n_rays: int = 6
    fallback_distance: float = 10.0
    smoothing_weight: float = 0.5
    n_jobs: int = 1
    verbose: bool = False


class SkeletonOptimizer:
    """
    Optimizer for skeleton polylines to push points toward the mesh medial axis.

    This class takes a skeleton (polylines format) produced by MCF skeletonization
    and a mesh, then optimizes the skeleton points to better approximate the
    medial axis of the mesh volume.

    Example:
        >>> from mcf2swc import PolylinesSkeleton, MeshManager, SkeletonOptimizer
        >>> skeleton = PolylinesSkeleton.from_txt("skeleton.polylines.txt")
        >>> mesh_mgr = MeshManager(mesh_path="mesh.obj")
        >>> optimizer = SkeletonOptimizer(skeleton, mesh_mgr.mesh)
        >>> optimized_skeleton = optimizer.optimize()
    """

    def __init__(
        self,
        skeleton: PolylinesSkeleton,
        mesh: trimesh.Trimesh,
        options: Optional[SkeletonOptimizerOptions] = None,
    ):
        """
        Initialize the skeleton optimizer.

        Args:
            skeleton: Input skeleton polylines from MCF calculation
            mesh: Target mesh that the skeleton should approximate
            options: Optimization configuration options
        """
        self.skeleton = skeleton.copy()
        self.mesh = mesh
        self.options = options or SkeletonOptimizerOptions()

        self._surface_crossing_detected = False
        self._optimization_history = []
        self._branch_points = None  # Cache for branch point detection

    def check_surface_crossing(self) -> Tuple[bool, int, float]:
        """
        Check if any skeleton points are outside the mesh surface.

        Returns:
            Tuple of (has_crossing, num_outside_points, max_distance)
            - has_crossing: True if any points are outside the mesh
            - num_outside_points: Number of points outside the mesh
            - max_distance: Maximum distance to surface for outside points
        """
        if not self.skeleton.polylines:
            return False, 0, 0.0

        all_pts = np.vstack(self.skeleton.polylines)

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
                        "Surface crossing detected: %d/%d points outside mesh (max distance: %.4f)",
                        num_outside,
                        len(all_pts),
                        max_dist,
                    )
                else:
                    logger.info("No surface crossing detected - all points inside mesh")

            return has_crossing, num_outside, max_dist

        except Exception as e:
            logger.warning("Failed to check surface crossing: %s", e)
            return False, 0, 0.0

    def optimize(self) -> PolylinesSkeleton:
        """
        Optimize the skeleton by pushing points toward the mesh medial axis.

        Returns:
            Optimized skeleton polylines
        """
        if self.options.check_surface_crossing:
            self.check_surface_crossing()

        if self.options.verbose:
            logger.info("Starting skeleton optimization...")
            logger.info("  Max iterations: %d", self.options.max_iterations)
            logger.info("  Step size: %.4f", self.options.step_size)
            logger.info("  Smoothing weight: %.4f", self.options.smoothing_weight)
            logger.info("  Parallel jobs: %d", self.options.n_jobs)

        # Determine number of workers
        n_jobs = self.options.n_jobs
        if n_jobs == -1:
            import os

            n_jobs = os.cpu_count() or 1

        # Use parallel processing if n_jobs > 1 and we have multiple polylines
        if n_jobs > 1 and len(self.skeleton.polylines) > 1:
            optimized_polylines = self._optimize_parallel(n_jobs)
        else:
            optimized_polylines = self._optimize_sequential()

        result = PolylinesSkeleton(optimized_polylines)

        if self.options.verbose:
            logger.info("Optimization complete")

        return result

    def _optimize_sequential(self) -> list:
        """
        Optimize polylines sequentially (original implementation).

        Returns:
            List of optimized polylines
        """
        optimized_polylines = []
        for poly_idx, polyline in enumerate(self.skeleton.polylines):
            if len(polyline) == 0:
                optimized_polylines.append(polyline.copy())
                continue

            optimized = self._optimize_polyline(polyline, poly_idx)
            optimized_polylines.append(optimized)

        return optimized_polylines

    def _optimize_parallel(self, n_jobs: int) -> list:
        """
        Optimize polylines in parallel using multiprocessing.

        Args:
            n_jobs: Number of parallel workers

        Returns:
            List of optimized polylines
        """
        if self.options.verbose:
            logger.info("  Using %d parallel workers", n_jobs)

        # Prepare tasks: (polyline_index, polyline_data)
        tasks = []
        for poly_idx, polyline in enumerate(self.skeleton.polylines):
            tasks.append((poly_idx, polyline))

        # Run optimization in parallel
        optimized_polylines = [None] * len(tasks)

        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            # Submit all tasks
            future_to_idx = {
                executor.submit(
                    _optimize_polyline_worker,
                    polyline,
                    poly_idx,
                    self.mesh,
                    self.options,
                ): poly_idx
                for poly_idx, polyline in tasks
            }

            # Collect results as they complete
            for future in as_completed(future_to_idx):
                poly_idx = future_to_idx[future]
                try:
                    optimized = future.result()
                    optimized_polylines[poly_idx] = optimized
                except Exception as exc:
                    logger.error("Polyline %d optimization failed: %s", poly_idx, exc)
                    # Fallback to original polyline
                    optimized_polylines[poly_idx] = self.skeleton.polylines[
                        poly_idx
                    ].copy()

        return optimized_polylines

    def _optimize_polyline(self, polyline: np.ndarray, poly_idx: int) -> np.ndarray:
        """
        Optimize a single polyline.

        Args:
            polyline: (N, 3) array of points
            poly_idx: Index of this polyline for logging

        Returns:
            Optimized (N, 3) array of points
        """
        points = polyline.copy()
        n_points = len(points)

        if n_points <= 1:
            return points

        # Detect branch points if needed
        if self.options.preserve_branch_points:
            if self._branch_points is None:
                self._branch_points = self._detect_branch_points()

        for iteration in range(self.options.max_iterations):
            points_old = points.copy()

            for i in range(n_points):
                if self.options.preserve_endpoints and (i == 0 or i == n_points - 1):
                    continue

                # Skip branch points if preserve_branch_points is True
                if self.options.preserve_branch_points:
                    if self._is_branch_point(poly_idx, i):
                        continue

                # Compute centering direction using uniform 3D ray sampling
                direction = self._compute_centering_direction(points[i])

                smoothing_direction = np.zeros(3)
                if self.options.smoothing_weight > 0 and n_points > 2:
                    smoothing_direction = self._compute_smoothing_direction(points, i)

                total_direction = (
                    1.0 - self.options.smoothing_weight
                ) * direction + self.options.smoothing_weight * smoothing_direction

                points[i] = points[i] + self.options.step_size * total_direction

            movement = np.linalg.norm(points - points_old, axis=1).mean()

            if self.options.verbose and iteration % 10 == 0:
                logger.info(
                    "  Polyline %d, iteration %d: avg movement = %.6f",
                    poly_idx,
                    iteration,
                    movement,
                )

            if movement < self.options.convergence_threshold:
                if self.options.verbose:
                    logger.info(
                        "  Polyline %d converged at iteration %d", poly_idx, iteration
                    )
                break

        return points

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
                # Points should move away from closer surfaces
                # We use 1/d as the "pressure" from that direction
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

        This gives approximately evenly spaced directions in 3D space.
        Special case: if n_points=6, uses exact axis-aligned rays (+/- x, y, z).

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

        # Normalize to ensure unit vectors (should already be, but for numerical stability)
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        directions = directions / (norms + 1e-10)

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
                if self.options.verbose:
                    logger.debug("Ray found no intersection, using fallback distance")
                return self.options.fallback_distance

            # Find the closest intersection point
            distances = np.linalg.norm(locations - point, axis=1)
            min_dist = float(np.min(distances))

            if self.options.verbose and len(locations) > 1:
                logger.debug(
                    "Ray found %d intersections, using closest (%.4f)",
                    len(locations),
                    min_dist,
                )

            return min_dist

        except Exception as e:
            logger.warning("Ray tracing failed, using fallback distance: %s", e)
            return self.options.fallback_distance

    def _compute_smoothing_direction(
        self, points: np.ndarray, index: int
    ) -> np.ndarray:
        """
        Compute smoothing direction using Laplacian smoothing.

        Args:
            points: (N, 3) array of all points in the polyline
            index: Index of the current point

        Returns:
            (3,) array representing the smoothing direction
        """
        n = len(points)
        if n <= 2:
            return np.zeros(3)

        if index == 0:
            neighbor_avg = points[1]
        elif index == n - 1:
            neighbor_avg = points[n - 2]
        else:
            neighbor_avg = (points[index - 1] + points[index + 1]) / 2.0

        direction = neighbor_avg - points[index]
        norm = np.linalg.norm(direction)
        if norm > 1e-10:
            return direction / norm
        return np.zeros(3)

    def _detect_branch_points(self) -> set:
        """
        Detect branch points where 3+ polylines meet.

        Returns:
            Set of (polyline_idx, point_idx) tuples for branch points
        """
        branch_points = set()
        tolerance = self.options.branch_point_tolerance

        # Collect all endpoints (potential branch points)
        all_points = []
        for poly_idx, polyline in enumerate(self.skeleton.polylines):
            if len(polyline) > 0:
                all_points.append((poly_idx, 0, polyline[0]))  # First point
                all_points.append(
                    (poly_idx, len(polyline) - 1, polyline[-1])
                )  # Last point

        # Check for points that are close to each other
        for i, (poly_i, idx_i, pt_i) in enumerate(all_points):
            close_points = [(poly_i, idx_i)]
            for j, (poly_j, idx_j, pt_j) in enumerate(all_points):
                if i != j:
                    dist = np.linalg.norm(pt_i - pt_j)
                    if dist < tolerance:
                        close_points.append((poly_j, idx_j))

            # If 3+ points are close together, mark them as branch points
            if len(close_points) >= 3:
                for poly_idx, pt_idx in close_points:
                    branch_points.add((poly_idx, pt_idx))

        return branch_points

    def _is_branch_point(self, poly_idx: int, point_idx: int) -> bool:
        """
        Check if a point is a branch point.

        Args:
            poly_idx: Index of the polyline
            point_idx: Index of the point within the polyline

        Returns:
            True if the point is a branch point
        """
        if self._branch_points is None:
            return False
        return (poly_idx, point_idx) in self._branch_points

    def get_optimization_stats(self) -> dict:
        """
        Get statistics about the optimization process.

        Returns:
            Dictionary containing optimization statistics
        """
        stats = {
            "surface_crossing_detected": self._surface_crossing_detected,
            "num_polylines": len(self.skeleton.polylines),
            "total_points": self.skeleton.total_points(),
        }

        if self.options.check_surface_crossing:
            has_crossing, num_outside, max_dist = self.check_surface_crossing()
            stats["points_outside_mesh"] = num_outside
            stats["max_distance_outside"] = max_dist

        return stats
