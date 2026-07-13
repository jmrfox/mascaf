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
    """Configuration for morphology-basis optimization.

    Forcing uses a weighted localized centering force with optional
    magnitude-matched smoothing and vertex repulsion::

        delta_v = f
                 + lambda_smooth * s * ||f|| / ||s||
                 + lambda_vertex * z * ||f|| / ||z||

    Step size ``step_scale`` is ``h`` in ``v <- v + h * delta_v``.

    All fields are keyword arguments to the dataclass constructor; see
    each field's inline annotation for defaults and semantics.
    """

    do_pruning: bool = False
    pruning_min_length: Optional[float] = None
    pruning_min_length_fraction: Optional[float] = None
    pruning_iterative: bool = True
    do_snapping: bool = True
    do_forcing: bool = True
    max_iterations: int = 100
    step_scale: float = 0.1
    convergence_threshold: float = 1e-4
    preserve_terminal_nodes: bool = True
    preserve_branch_nodes: bool = False
    n_rays: int = 6
    lambda_smooth: float = 0.2
    lambda_vertex: float = 0.0
    vertex_repulsion_distance: Optional[float] = None
    repulsion_power: float = 1.0
    localization_beta: float = 2.0
    weight_epsilon: float = 1e-6
    step_cap_factor: float = 0.5
    snap_ray_perturb_scales: tuple[float, ...] = (1e-4, 1e-3, 1e-2, 5e-2)
    snap_ray_perturb_angles: int = 8
    snap_min_chord_length: Optional[float] = None
    snap_min_chord_fraction: float = 5e-4
    verbose: bool = False


class BasisOptimizer:
    """Optimize a downsampled :class:`~mascaf.morphology_graph.MorphologyGraph` basis against a target mesh.

    Runs up to three sequential phases controlled by
    :class:`BasisOptimizerOptions`:

    1. **Pruning** — remove short terminal branches.
    2. **Snapping** — move outside nodes back inside the mesh.
    3. **Forcing** — iteratively pull nodes toward the medial axis using a
       weighted localized centering force with optional smoothing and
       vertex–vertex repulsion, with steps capped by surface distance.

    Parameters
    ----------
    graph : MorphologyGraph
        The morphology basis to optimize. A deep copy is made internally so
        the original is not modified.
    mesh : trimesh.Trimesh
        The target mesh that defines the interior/exterior and surface.
    options : BasisOptimizerOptions or None
        Optimization configuration. Defaults to :class:`BasisOptimizerOptions`
        with all defaults when ``None``.

    Examples
    --------
    >>> from mascaf import BasisOptimizer, BasisOptimizerOptions
    >>> opts = BasisOptimizerOptions(do_snapping=True, do_forcing=True)
    >>> optimized = BasisOptimizer(morphology, mesh, opts).optimize()
    """

    def __init__(
        self,
        graph: MorphologyGraph,
        mesh: trimesh.Trimesh,
        options: Optional[BasisOptimizerOptions] = None,
    ):
        self.graph = graph.copy()
        self.mesh = mesh
        self.options = options or BasisOptimizerOptions()

        logger.debug(
            "Initialized BasisOptimizer with %d nodes, %d edges, do_pruning=%s, "
            "do_snapping=%s, do_forcing=%s",
            self.graph.number_of_nodes(),
            self.graph.number_of_edges(),
            self.options.do_pruning,
            self.options.do_snapping,
            self.options.do_forcing,
        )

    def optimize(self) -> MorphologyGraph:
        """Run the configured optimization phases and return the result.

        Runs pruning, snapping, and forcing in sequence (each phase is
        skipped when its corresponding ``do_*`` flag is ``False``).

        Returns
        -------
        MorphologyGraph
            The optimized morphology basis (a modified copy of the input).
        """
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
        """Resolve the branch-pruning threshold from absolute or fraction input."""
        if self.options.pruning_min_length is not None:
            return float(self.options.pruning_min_length)

        fraction = self.options.pruning_min_length_fraction
        if fraction is None:
            return None

        branch_lengths = list(self._compute_branch_lengths(self.graph).values())
        if not branch_lengths:
            return None
        if fraction <= 0 or fraction >= 1:
            raise ValueError(f"Pruning fraction must be in (0,1), got {fraction}")
        return float(np.percentile(branch_lengths, float(fraction * 100.0)))

    def _run_snapping_phase(self) -> None:
        """Snap outside basis nodes into the mesh via chord midpoints.

        For each outside node, move along the nearest-surface direction to the
        midpoint between the first and second ray–mesh intersections. Failures
        for individual nodes are logged as warnings; a summary warning is
        emitted if any nodes remain outside after the phase.
        """
        outside_node_ids = self.graph.get_outside_nodes(self.mesh)
        logger.info(
            "Phase 1 - Snapping: %d nodes outside mesh",
            len(outside_node_ids),
        )
        logger.debug("Snapping candidate nodes: %s", outside_node_ids)

        if not outside_node_ids:
            logger.debug("No snapping required because all nodes are inside the mesh")
            return

        for node in outside_node_ids:
            pos = self.graph.get_node_position(node)
            new_pos = self._snap_point_to_chord_midpoint(pos)
            if new_pos is None:
                logger.warning(
                    "Failed to snap node %s at %s; leaving position unchanged",
                    node,
                    pos,
                )
                continue
            logger.debug(
                "Snapping node %s from %s to chord midpoint %s",
                node,
                pos,
                new_pos,
            )
            self.graph.set_node_position(node, new_pos)

        still_outside = self.graph.get_outside_nodes(self.mesh)
        if still_outside:
            logger.warning(
                "Snapping complete but %d node(s) remain outside the mesh: %s",
                len(still_outside),
                still_outside,
            )

    def _run_forcing_phase(self) -> None:
        """Iteratively move basis nodes toward the medial axis.

        Each step forms::

            delta_v = f
                     + lambda_smooth * s * ||f|| / ||s||
                     + lambda_vertex * z * ||f|| / ||z||

        then applies ``step = h * delta_v`` and caps the step length by
        ``step_cap_factor`` times the surface distance along ``delta_v``.
        """
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
        lambda_smooth = self.options.lambda_smooth
        lambda_vertex = self.options.lambda_vertex
        if self.options.vertex_repulsion_distance is not None:
            repulsion_radius = float(self.options.vertex_repulsion_distance)
        else:
            repulsion_radius = self._max_edge_length()
        logger.info(
            "  Forcing lambdas: smooth=%.4f vertex=%.4f repulsion_radius=%.6f",
            lambda_smooth,
            lambda_vertex,
            repulsion_radius,
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
                f = self._compute_centering_force(pos)
                f_norm = float(np.linalg.norm(f))
                delta_v = f.copy()

                if lambda_smooth > 0 and f_norm > 1e-10:
                    s = self._compute_smoothing_direction_for_node(node)
                    s_norm = float(np.linalg.norm(s))
                    if s_norm > 1e-10:
                        delta_v = delta_v + lambda_smooth * s * (f_norm / s_norm)

                if lambda_vertex > 0 and f_norm > 1e-10 and repulsion_radius > 0:
                    z = self._compute_vertex_repulsion(node, repulsion_radius)
                    z_norm = float(np.linalg.norm(z))
                    if z_norm > 1e-10:
                        delta_v = delta_v + lambda_vertex * z * (f_norm / z_norm)

                step = self.options.step_scale * delta_v
                step = self._cap_step_by_surface_distance(pos, step)
                self.graph.set_node_position(node, pos + step)

            movement = self._average_movement(
                old_positions,
                self.graph.get_all_positions(),
            )
            logger.info("  Iteration %d: avg movement = %.6f", iteration, movement)
            if movement < self.options.convergence_threshold:
                logger.info("  Converged at iteration %d", iteration)
                break

    def _max_edge_length(self) -> float:
        """Return the maximum current edge length in the basis graph."""
        max_len = 0.0
        for u, v in self.graph.edges():
            pos_u = self.graph.get_node_position(u)
            pos_v = self.graph.get_node_position(v)
            max_len = max(max_len, float(np.linalg.norm(pos_v - pos_u)))
        return max_len

    def _compute_vertex_repulsion(self, node: int, radius: float) -> np.ndarray:
        """Inverse-square repulsion from other vertices within ``radius``.

        Returns
        -------
        np.ndarray
            ``z = sum_j (x - x_j) / r_ij^3`` over other nodes with
            ``0 < r_ij < radius``.
        """
        pos = self.graph.get_node_position(node)
        force = np.zeros(3)
        if radius <= 0:
            return force

        for other in self.graph.nodes():
            if other == node:
                continue
            delta = pos - self.graph.get_node_position(other)
            dist = float(np.linalg.norm(delta))
            if dist <= 0.0 or dist >= radius:
                continue
            force += delta / (dist**3)
        return force

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

    def _compute_centering_force(self, point: np.ndarray) -> np.ndarray:
        """Compute a weighted localized centering force toward the medial axis.

        For an interior point, casts ``n_rays`` unit directions ``x_i`` with
        surface distances ``d_i`` and forms weights

        ``w_i = (d_i + eps)^(-p) * exp(-beta * (d_i - d_min) / d_min)``,

        then returns ``f = -sum_i x_i w_i / sum_i w_i`` without unit
        normalization. Outside points fall back to the closest-point direction.
        """
        is_inside = self.mesh.contains(point.reshape(1, 3))[0]
        if not is_inside:
            return self._compute_closest_point_direction(point)

        try:
            directions = self._get_uniform_sphere_directions(self.options.n_rays)
            distances: list[float] = []
            valid_directions: list[np.ndarray] = []
            for direction in directions:
                distance = self._ray_distance_to_surface(point, direction)
                if distance > 1e-6:
                    distances.append(distance)
                    valid_directions.append(direction)

            if not distances:
                return np.zeros(3)

            d_min = min(distances)
            if d_min <= 0:
                return np.zeros(3)

            eps = self.options.weight_epsilon
            p = self.options.repulsion_power
            beta = self.options.localization_beta

            force = np.zeros(3)
            weight_sum = 0.0
            for direction, distance in zip(valid_directions, distances):
                weight = (distance + eps) ** (-p) * np.exp(
                    -beta * (distance - d_min) / d_min
                )
                force -= direction * weight
                weight_sum += weight

            if weight_sum <= 1e-10:
                return np.zeros(3)
            return force / weight_sum
        except Exception as exc:
            logger.error("Failed to compute centering force: %s", exc)
            return self._compute_closest_point_direction(point)

    def _cap_step_by_surface_distance(
        self,
        point: np.ndarray,
        step: np.ndarray,
    ) -> np.ndarray:
        """Cap a displacement by surface distance along the step direction.

        Limits ``||step||`` to ``step_cap_factor`` times the ray distance from
        ``point`` along ``step``, so a forcing update cannot leave the mesh.
        """
        step_norm = float(np.linalg.norm(step))
        if step_norm <= 1e-10:
            return step

        direction = step / step_norm
        d_force = self._ray_distance_to_surface(point, direction)
        max_step = self.options.step_cap_factor * d_force
        if step_norm > max_step and max_step > 0:
            return step * (max_step / step_norm)
        return step

    def _snap_min_chord(self) -> float:
        """Minimum accepted enter–exit chord length for snapping."""
        if self.options.snap_min_chord_length is not None:
            return float(self.options.snap_min_chord_length)
        extents = np.asarray(self.mesh.extents, dtype=float)
        diagonal = float(np.linalg.norm(extents))
        return float(self.options.snap_min_chord_fraction) * diagonal

    def _snap_point_to_chord_midpoint(
        self, point: np.ndarray
    ) -> Optional[np.ndarray]:
        """Return the midpoint of the first two ray hits toward the nearest surface.

        If the closest-point ray yields an odd hit count or a vanishingly short
        chord (grazing double-hit near an edge/crease), the direction is
        perturbed until a usable even-hit chord is found. Returns ``None`` if
        no suitable ray is found.
        """
        direction, dist = self._compute_snap_direction(point)
        if dist < 1e-10 or float(np.linalg.norm(direction)) < 1e-10:
            logger.warning(
                "Snap failed: point %s is too close to the surface (dist=%s)",
                point,
                dist,
            )
            return None

        hits = self._ray_intersections_even_hits(point, direction)
        if hits is None:
            logger.warning(
                "Snap failed: could not find an even-hit ray from point %s "
                "near direction %s",
                point,
                direction,
            )
            return None
        return 0.5 * (hits[0] + hits[1])

    def _ray_intersections_even_hits(
        self,
        point: np.ndarray,
        direction: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Return sorted ray hits with an even count and usable chord length.

        Tries ``direction`` first, then a deterministic ring of small angular
        perturbations in the plane orthogonal to ``direction``. Candidates with
        fewer than two even hits, or with a first-pair chord shorter than
        :meth:`_snap_min_chord`, are skipped (near-surface grazing hits).
        """
        direction = np.asarray(direction, dtype=float)
        direction = direction / (np.linalg.norm(direction) + 1e-15)
        min_chord = self._snap_min_chord()

        for candidate in self._snap_ray_direction_candidates(direction):
            try:
                hits = self._ray_intersections_sorted(point, candidate)
            except RuntimeError:
                continue
            n_hits = int(hits.shape[0])
            if n_hits < 2 or n_hits % 2 != 0:
                logger.debug(
                    "Snap ray from %s along %s produced %d hits (need even >= 2)",
                    point,
                    candidate,
                    n_hits,
                )
                continue

            chord = float(np.linalg.norm(hits[1] - hits[0]))
            if chord < min_chord:
                logger.debug(
                    "Snap ray from %s along %s: chord %.3e < min %.3e (reject)",
                    point,
                    candidate,
                    chord,
                    min_chord,
                )
                continue

            if not np.allclose(candidate, direction):
                logger.debug(
                    "Snap ray perturbed for point %s: %d hits, chord %.3e, "
                    "direction %s",
                    point,
                    n_hits,
                    chord,
                    candidate,
                )
            return hits
        return None

    def _snap_ray_direction_candidates(
        self, direction: np.ndarray
    ) -> list[np.ndarray]:
        """Yield the base snap direction then small orthogonal perturbations."""
        candidates = [direction]

        # Orthonormal basis of the plane perpendicular to ``direction``.
        axis = np.array([1.0, 0.0, 0.0])
        if abs(float(direction @ axis)) > 0.9:
            axis = np.array([0.0, 1.0, 0.0])
        u = np.cross(direction, axis)
        u = u / (np.linalg.norm(u) + 1e-15)
        v = np.cross(direction, u)
        v = v / (np.linalg.norm(v) + 1e-15)

        n_angles = max(1, int(self.options.snap_ray_perturb_angles))
        for scale in self.options.snap_ray_perturb_scales:
            for k in range(n_angles):
                theta = 2.0 * np.pi * k / n_angles
                pert = np.cos(theta) * u + np.sin(theta) * v
                candidate = direction + float(scale) * pert
                norm = float(np.linalg.norm(candidate))
                if norm <= 1e-15:
                    continue
                candidates.append(candidate / norm)
        return candidates

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
            logger.error("Failed to compute snap direction: %s", exc)
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

    def _ray_intersections_sorted(
        self, point: np.ndarray, direction: np.ndarray
    ) -> np.ndarray:
        """Return mesh intersection points along a ray, sorted by distance.

        Returns
        -------
        np.ndarray
            Array of shape ``(n_hits, 3)`` sorted by increasing distance from
            ``point``. Empty if there are no hits.

        Raises
        ------
        RuntimeError
            If ray tracing itself fails.
        """
        try:
            locations, _, _ = self.mesh.ray.intersects_location(
                ray_origins=point.reshape(1, 3),
                ray_directions=direction.reshape(1, 3),
            )
        except Exception as exc:
            raise RuntimeError(
                f"Ray tracing failed from point {point} along direction {direction}"
            ) from exc

        if len(locations) == 0:
            return np.zeros((0, 3), dtype=float)

        distances = np.linalg.norm(locations - point, axis=1)
        order = np.argsort(distances)
        return np.asarray(locations[order], dtype=float)

    def _ray_distance_to_surface(
        self, point: np.ndarray, direction: np.ndarray
    ) -> float:
        """Compute distance from a point to the mesh surface along a ray.

        Raises
        ------
        RuntimeError
            If the ray does not hit the mesh or ray tracing fails.
        """
        hits = self._ray_intersections_sorted(point, direction)
        if hits.shape[0] == 0:
            raise RuntimeError(
                f"Ray from point {point} along direction {direction} "
                "did not intersect the mesh surface"
            )
        return float(np.linalg.norm(hits[0] - point))

    def get_optimization_stats(self) -> dict:
        """Return summary statistics for the optimized basis graph."""
        outside_node_ids = self.graph.get_outside_nodes(self.mesh)
        return {
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "num_terminal_nodes": len(self.graph.get_terminal_nodes()),
            "num_branch_nodes": len(self.graph.get_branch_nodes()),
            "total_length": self.graph.get_total_length(),
            "nodes_outside_mesh": len(outside_node_ids),
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
