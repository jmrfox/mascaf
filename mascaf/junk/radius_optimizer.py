"""
Radius optimization.

This module implements a segment-by-segment optimization approach where each
frustum's radii are optimized to minimize the distance from surface points
to the mesh surface.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import networkx as nx
import numpy as np
import trimesh
from scipy.optimize import minimize

from mascaf.morphology_graph import MorphologyGraph

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def _get_node_xyz(graph: MorphologyGraph, node_id: int) -> np.ndarray:
    """Extract xyz coordinates from MorphologyGraph."""
    node = graph.nodes[node_id]
    if "xyz" not in node:
        raise ValueError(f"Node {node_id} missing xyz coordinates")
    return np.asarray(node["xyz"], dtype=float)


def _get_node_radius(graph: MorphologyGraph, node_id: int) -> float:
    """Extract radius from MorphologyGraph."""
    node = graph.nodes[node_id]
    if "radius" not in node:
        raise ValueError(f"Node {node_id} missing radius")
    return float(node["radius"])


@dataclass
class RadiusOptimizerOptions:
    """
    Configuration for radius optimization.

    Attributes:
        n_longitudinal: Number of sample points along the frustum axis
        n_radial: Number of sample points around the circumference
        max_iterations: Maximum number of passes over all segments
        convergence_threshold: Stop when max radius change is below this value
        min_radius: Minimum allowed radius
        max_radius: Maximum allowed radius
        step_size_factor: Factor for computing optimization step size from distances
        verbose: If True, print optimization progress
        normalize: If True, normalize radii after optimization to match mesh
        normalize_metric: Metric to normalize ('surface_area' or 'volume')
        remove_overlaps: If True, subtract overlap corrections for branch
            points (degree > 2)
    """

    n_longitudinal: int = 3
    n_radial: int = 8
    max_iterations: int = 10
    convergence_threshold: float = 1e-4
    min_radius: float = 0.01
    max_radius: Optional[float] = None
    step_size_factor: float = 0.1
    verbose: bool = False
    normalize: bool = False
    normalize_metric: str = "surface_area"
    remove_overlaps: bool = False


def _sample_frustum_surface_points(
    xyz_a: np.ndarray,
    xyz_b: np.ndarray,
    r_a: float,
    r_b: float,
    n_long: int,
    n_rad: int,
) -> np.ndarray:
    """
    Sample points uniformly on the lateral surface of a frustum.

    Args:
        xyz_a: Position of endpoint A (3,)
        xyz_b: Position of endpoint B (3,)
        r_a: Radius at endpoint A
        r_b: Radius at endpoint B
        n_long: Number of samples along the axis
        n_rad: Number of samples around the circumference

    Returns:
        Array of surface points, shape (n_long * n_rad, 3)
    """
    # Compute axis direction
    axis = xyz_b - xyz_a
    axis_length = np.linalg.norm(axis)

    if axis_length < 1e-12:
        # Degenerate segment - return points on a sphere
        if r_a > 0:
            points = []
            for i in range(n_rad):
                theta = 2.0 * np.pi * (i / n_rad)
                x = xyz_a[0] + r_a * np.cos(theta)
                y = xyz_a[1] + r_a * np.sin(theta)
                z = xyz_a[2]
                points.append([x, y, z])
            return np.array(points)
        else:
            return np.array([xyz_a])

    axis_unit = axis / axis_length

    # Create orthonormal frame (U, V, W) with W along axis
    # Choose U perpendicular to axis
    if abs(axis_unit[0]) < 0.9:
        temp = np.array([1.0, 0.0, 0.0])
    else:
        temp = np.array([0.0, 1.0, 0.0])

    U = np.cross(temp, axis_unit)
    U = U / (np.linalg.norm(U) + 1e-12)
    V = np.cross(axis_unit, U)
    V = V / (np.linalg.norm(V) + 1e-12)

    # Sample points
    points = []
    for i in range(n_long):
        # Parameter along axis [0, 1]
        t = i / max(1, n_long - 1) if n_long > 1 else 0.5

        # Position and radius at this t
        center = xyz_a + t * axis
        radius = r_a + t * (r_b - r_a)

        # Sample points around the circumference
        for j in range(n_rad):
            theta = 2.0 * np.pi * (j / n_rad)
            offset = radius * (np.cos(theta) * U + np.sin(theta) * V)
            point = center + offset
            points.append(point)

    return np.array(points)


def _compute_distances_to_mesh(
    points: np.ndarray,
    mesh: trimesh.Trimesh,
) -> np.ndarray:
    """
    Compute unsigned distances from points to nearest mesh surface.

    Args:
        points: Array of points, shape (N, 3)
        mesh: Target mesh

    Returns:
        Array of distances, shape (N,)
    """
    try:
        from trimesh.proximity import closest_point

        _, distances, _ = closest_point(mesh, points)
        return distances
    except Exception:
        # Fallback: compute distances to vertices
        vertices = np.asarray(mesh.vertices, dtype=float)
        distances = []
        for p in points:
            dists = np.linalg.norm(vertices - p, axis=1)
            distances.append(np.min(dists))
        return np.array(distances)


def _optimize_segment_radii(
    xyz_a: np.ndarray,
    xyz_b: np.ndarray,
    r_a_init: float,
    r_b_init: float,
    mesh: trimesh.Trimesh,
    options: RadiusOptimizerOptions,
) -> Tuple[float, float, float]:
    """
    Optimize radii for a single segment to minimize MSE distance to mesh.

    Args:
        xyz_a: Position of endpoint A
        xyz_b: Position of endpoint B
        r_a_init: Initial radius at A
        r_b_init: Initial radius at B
        mesh: Target mesh
        options: Optimization options

    Returns:
        Tuple of (optimized r_a, optimized r_b, final MSE)
    """

    def objective(radii):
        r_a, r_b = radii

        # Sample surface points with current radii
        points = _sample_frustum_surface_points(
            xyz_a, xyz_b, r_a, r_b, options.n_longitudinal, options.n_radial
        )

        # Compute distances to mesh
        distances = _compute_distances_to_mesh(points, mesh)

        # Return MSE
        return np.mean(distances**2)

    # Initial guess
    x0 = np.array([r_a_init, r_b_init])

    # Bounds
    bounds = [
        (options.min_radius, options.max_radius if options.max_radius else np.inf),
        (options.min_radius, options.max_radius if options.max_radius else np.inf),
    ]

    # Optimize
    result = minimize(
        objective,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 50, "ftol": 1e-6},
    )

    r_a_opt, r_b_opt = result.x
    mse = result.fun

    return float(r_a_opt), float(r_b_opt), float(mse)


class RadiusOptimizer:
    """
    Segment-by-segment radius optimizer.

    This optimizer iterates over all segments (edges) in the skeleton graph,
    optimizing each segment's endpoint radii to minimize the distance from
    frustum surface points to the mesh surface.
    """

    def __init__(
        self,
        morphology: MorphologyGraph,
        mesh: trimesh.Trimesh,
        *,
        options: Optional[RadiusOptimizerOptions] = None,
    ):
        """
        Initialize the radius optimizer.

        Args:
            morphology: MorphologyGraph with initial radius estimates
            mesh: Target mesh to fit
            options: Optimization options
        """
        self.skeleton = morphology
        self.mesh = mesh
        self.options = options if options is not None else RadiusOptimizerOptions()

        # Build node index mapping
        self.node_to_idx = {nid: i for i, nid in enumerate(sorted(morphology.nodes()))}
        self.idx_to_node = {i: nid for nid, i in self.node_to_idx.items()}

        # Extract initial radii using helper function
        self.initial_radii = np.array(
            [
                _get_node_radius(morphology, self.idx_to_node[i])
                for i in range(len(self.node_to_idx))
            ]
        )

        # Current radii (will be updated during optimization)
        self.current_radii = self.initial_radii.copy()

        logger.info(
            "RadiusOptimizer initialized: nodes=%d, edges=%d, n_long=%d, n_rad=%d",
            self.skeleton.number_of_nodes(),
            self.skeleton.number_of_edges(),
            self.options.n_longitudinal,
            self.options.n_radial,
        )

    def optimize(self) -> MorphologyGraph:
        """
        Run the iterative segment-by-segment optimization.

        Returns:
            New MorphologyGraph with optimized radii
        """
        edges = list(self.skeleton.edges())
        n_edges = len(edges)

        if self.options.verbose:
            print(
                f"Starting local optimization: {n_edges} segments, max {self.options.max_iterations} iterations"
            )

        for iteration in range(self.options.max_iterations):
            max_change = 0.0
            total_mse = 0.0

            # Process each segment
            for edge_idx, (u, v) in enumerate(edges):
                # Get node indices
                i_u = self.node_to_idx[u]
                i_v = self.node_to_idx[v]

                # Get positions using helper function
                xyz_u = _get_node_xyz(self.skeleton, u)
                xyz_v = _get_node_xyz(self.skeleton, v)

                # Get current radii
                r_u = self.current_radii[i_u]
                r_v = self.current_radii[i_v]

                # Optimize this segment
                r_u_new, r_v_new, mse = _optimize_segment_radii(
                    xyz_u, xyz_v, r_u, r_v, self.mesh, self.options
                )

                # Update radii
                self.current_radii[i_u] = r_u_new
                self.current_radii[i_v] = r_v_new

                # Track changes
                change_u = abs(r_u_new - r_u)
                change_v = abs(r_v_new - r_v)
                max_change = max(max_change, change_u, change_v)
                total_mse += mse

            avg_mse = total_mse / n_edges

            if self.options.verbose:
                print(
                    f"  Iteration {iteration + 1}: max_change={max_change:.6f}, avg_mse={avg_mse:.6f}"
                )

            logger.info(
                "Iteration %d: max_change=%.6f, avg_mse=%.6f",
                iteration + 1,
                max_change,
                avg_mse,
            )

            # Check convergence
            if max_change < self.options.convergence_threshold:
                if self.options.verbose:
                    print(f"Converged after {iteration + 1} iterations")
                logger.info("Converged after %d iterations", iteration + 1)
                break

        # Apply normalization if requested
        self._normalize_radii()

        # Compute final global metrics for monitoring
        final_sa = self._compute_skeleton_surface_area(
            self.current_radii, self.options.remove_overlaps
        )
        final_vol = self._compute_skeleton_volume(
            self.current_radii, self.options.remove_overlaps
        )
        mesh_sa = float(self.mesh.area)
        mesh_vol = float(self.mesh.volume)

        sa_error = abs(final_sa - mesh_sa) / mesh_sa if mesh_sa > 0 else 0.0
        vol_error = abs(final_vol - mesh_vol) / mesh_vol if mesh_vol > 0 else 0.0

        if self.options.verbose:
            print(f"\nFinal metrics:")
            print(
                f"  Surface area: skeleton={final_sa:.2f}, mesh={mesh_sa:.2f}, error={sa_error:.2%}"
            )
            print(
                f"  Volume: skeleton={final_vol:.2f}, mesh={mesh_vol:.2f}, error={vol_error:.2%}"
            )

        logger.info(
            "Optimization complete: SA_error=%.2f%%, Vol_error=%.2f%%",
            sa_error * 100,
            vol_error * 100,
        )

        # Create new skeleton with optimized radii
        return self._create_optimized_skeleton()

    def _compute_skeleton_surface_area(
        self, radii: np.ndarray, remove_overlaps: bool = False
    ) -> float:
        """Compute total surface area of frustum segments.

        Args:
            radii: Array of node radii
            remove_overlaps: If True, subtract overlap corrections for branch
                points (degree > 2)
        """
        total_area = 0.0

        # Add lateral surface area of all segments
        for u, v in self.skeleton.edges():
            i_u = self.node_to_idx[u]
            i_v = self.node_to_idx[v]

            xyz_u = _get_node_xyz(self.skeleton, u)
            xyz_v = _get_node_xyz(self.skeleton, v)
            r_u = radii[i_u]
            r_v = radii[i_v]

            # Frustum lateral surface area: π * (r1 + r2) * s
            # where s = sqrt(h^2 + (r2 - r1)^2) is slant height
            h = float(np.linalg.norm(xyz_v - xyz_u))
            if h <= 0:
                continue

            s = math.sqrt(h * h + (r_v - r_u) * (r_v - r_u))
            area = math.pi * (r_u + r_v) * s
            total_area += area

        # Process nodes for end caps and overlap correction
        for node_id in self.skeleton.nodes():
            degree = self.skeleton.degree[node_id]
            idx = self.node_to_idx[node_id]
            r = radii[idx]

            if degree == 1:
                # Add end cap area for terminal nodes
                cap_area = math.pi * r * r
                total_area += cap_area
            elif degree > 2 and remove_overlaps:
                # Subtract overlap correction for branch points
                num_overlaps = degree - 2
                overlap_area = 2.0 * math.pi * r * r
                total_area -= num_overlaps * overlap_area

        return total_area

    def _compute_skeleton_volume(
        self, radii: np.ndarray, remove_overlaps: bool = False
    ) -> float:
        """Compute total volume of frustum segments.

        Args:
            radii: Array of node radii
            remove_overlaps: If True, subtract overlap corrections for branch
                points (degree > 2)
        """
        total_volume = 0.0

        # Add frustum volumes for all edges
        for u, v in self.skeleton.edges():
            i_u = self.node_to_idx[u]
            i_v = self.node_to_idx[v]

            xyz_u = _get_node_xyz(self.skeleton, u)
            xyz_v = _get_node_xyz(self.skeleton, v)
            r_u = radii[i_u]
            r_v = radii[i_v]

            # Frustum volume: V = (π * h / 3) * (r1^2 + r1*r2 + r2^2)
            h = float(np.linalg.norm(xyz_v - xyz_u))
            if h <= 0:
                continue

            volume = (math.pi * h / 3.0) * (r_u * r_u + r_u * r_v + r_v * r_v)
            total_volume += volume

        # Subtract overlap correction for branch points (degree > 2)
        if remove_overlaps:
            for node_id in self.skeleton.nodes():
                degree = self.skeleton.degree[node_id]
                if degree > 2:
                    idx = self.node_to_idx[node_id]
                    r = radii[idx]
                    num_overlaps = degree - 2
                    overlap_volume = math.pi * r * r * r
                    total_volume -= num_overlaps * overlap_volume

        return total_volume

    def _normalize_radii(self) -> None:
        """
        Normalize radii to match mesh surface area or volume.

        Scales all radii by a constant factor so that the total surface area
        or volume of the skeleton matches the corresponding mesh metric.
        """
        if not self.options.normalize:
            return

        metric = self.options.normalize_metric.lower()

        if metric == "surface_area":
            skeleton_metric = self._compute_skeleton_surface_area(
                self.current_radii, self.options.remove_overlaps
            )
            mesh_metric = float(self.mesh.area)
            metric_name = "surface area"
        elif metric == "volume":
            skeleton_metric = self._compute_skeleton_volume(
                self.current_radii, self.options.remove_overlaps
            )
            mesh_metric = float(self.mesh.volume)
            metric_name = "volume"
        else:
            raise ValueError(
                f"Invalid normalize_metric: {self.options.normalize_metric}. "
                "Must be 'surface_area' or 'volume'."
            )

        if skeleton_metric <= 0:
            logger.warning(
                "Cannot normalize: skeleton %s is zero or negative",
                metric_name,
            )
            return

        if mesh_metric <= 0:
            logger.warning(
                "Cannot normalize: mesh %s is zero or negative",
                metric_name,
            )
            return

        # For frusta:
        # SA = π(r1+r2)s ∝ r (linear in radius)
        # V = (πh/3)(r1²+r1*r2+r2²) ∝ r² (quadratic in radius)
        if metric == "surface_area":
            scale_factor = mesh_metric / skeleton_metric
        else:
            scale_factor = math.sqrt(mesh_metric / skeleton_metric)

        # Apply scaling
        self.current_radii *= scale_factor

        # Enforce bounds
        self.current_radii = np.clip(
            self.current_radii,
            self.options.min_radius,
            self.options.max_radius if self.options.max_radius else np.inf,
        )

        if self.options.verbose:
            new_metric = (
                self._compute_skeleton_surface_area(
                    self.current_radii, self.options.remove_overlaps
                )
                if metric == "surface_area"
                else self._compute_skeleton_volume(
                    self.current_radii, self.options.remove_overlaps
                )
            )
            print(f"\nNormalization applied (metric={metric}):")
            print(f"  Scale factor: {scale_factor:.6f}")
            print(f"  Before: skeleton {metric_name}={skeleton_metric:.2f}")
            print(
                f"  After: skeleton {metric_name}={new_metric:.2f}, "
                f"mesh {metric_name}={mesh_metric:.2f}"
            )

        logger.info(
            "Normalization applied: metric=%s, scale_factor=%.6f",
            metric,
            scale_factor,
        )

    def _create_optimized_skeleton(self) -> MorphologyGraph:
        """Create a new MorphologyGraph with optimized radii."""
        new_skeleton = MorphologyGraph()

        # Copy nodes with updated radii
        for nid in self.skeleton.nodes():
            idx = self.node_to_idx[nid]
            node_data = dict(self.skeleton.nodes[nid])

            # Update radius
            node_data["radius"] = float(self.current_radii[idx])

            new_skeleton.add_node(nid, **node_data)

        # Copy edges
        for u, v in self.skeleton.edges():
            edge_data = dict(self.skeleton.edges[u, v])
            new_skeleton.add_edge(u, v, **edge_data)

        return new_skeleton
