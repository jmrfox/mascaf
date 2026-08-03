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

    Pruning removes terminal stubs (terminal → branch through degree-2 nodes)
    shorter than a threshold. Absolute ``pruning_length`` wins if set; else
    ``pruning_length_fraction * longest_terminal_stub`` (from the graph at
    the start of pruning).

    Forcing uses a weighted localized centering force blended with
    magnitude-matched neighbor smoothing::

        delta_v = (1 - alpha_s) * F_centering
                + alpha_s * F_smoothing * ||F_centering|| / ||F_smoothing||

    where ``F_centering`` is scaled by ``d_min`` and ``F_smoothing`` is the
    raw neighbor-centroid pull. The update multiplies by ``step_scale``, then
    caps by ``step_cap_factor`` times surface distance along the step. Optional
    ``ray_jitter`` (default ``0``) randomly perturbs each centering-ray
    direction once per forcing iteration. Optional ``active_resample`` keeps
    edge lengths between ``active_resample_min_fraction`` and
    ``active_resample_max_fraction`` times the mesh bounding-box diagonal
    (consolidate short edges; bisect long edges) after each forcing iteration.
    Midpoints that fall outside the mesh are snapped inside. By default,
    consolidation that would change the graph cyclomatic number (cycle count)
    is skipped with a warning; set ``active_resample_allow_cycle_collapse``
    to allow such merges.

    Early stopping (any criterion may halt; all are active by default):

    - average node movement below ``convergence_threshold``
    - centering error ``E = sum_i ||F_centering_i||`` below
      ``centering_error_stop_fraction`` times the first-iteration value
    - relative change in ``E`` below ``centering_error_plateau_tol`` for
      ``centering_error_plateau_patience`` consecutive iterations
    - centering error increases for
      ``centering_error_increase_patience`` consecutive iterations

    All fields are keyword arguments to the dataclass constructor; see
    each field's inline annotation for defaults and semantics.
    """

    do_pruning: bool = False
    pruning_length: Optional[float] = None
    pruning_length_fraction: Optional[float] = None
    pruning_iterative: bool = True
    do_snapping: bool = True
    do_forcing: bool = False
    max_iterations: int = 10
    convergence_threshold: float = 1e-4
    preserve_terminal_nodes: bool = True
    preserve_branch_nodes: bool = False
    n_rays: int = 6
    ray_jitter: float = 0.0
    alpha_s: float = 0.1
    step_scale: float = 0.5
    repulsion_power: float = 1.0
    localization_beta: float = 2.0
    weight_epsilon: float = 1e-6
    step_cap_factor: float = 0.5
    outside_distance_tol: Optional[float] = None
    outside_distance_tol_fraction: float = 1e-6
    centering_error_stop_fraction: float = 0.1
    centering_error_plateau_tol: float = 1e-3
    centering_error_plateau_patience: int = 2
    centering_error_increase_patience: int = 2
    snap_ray_perturb_scales: tuple[float, ...] = (1e-4, 1e-3, 1e-2, 5e-2)
    snap_ray_perturb_angles: int = 8
    snap_chord_fraction: float = 0.25
    snap_min_chord_length: Optional[float] = None
    snap_min_chord_fraction: float = 5e-4
    active_resample: bool = False
    active_resample_min_fraction: Optional[float] = None
    active_resample_max_fraction: Optional[float] = None
    active_resample_allow_cycle_collapse: bool = False


class BasisOptimizer:
    """Optimize a downsampled :class:`~mascaf.morphology_graph.MorphologyGraph` basis against a target mesh.

    Runs up to three sequential phases controlled by
    :class:`BasisOptimizerOptions`:

    1. **Pruning** — remove short terminal→branch stubs.
    2. **Snapping** — move outside nodes back inside the mesh.
    3. **Forcing** — iteratively pull nodes toward the medial axis using a
       weighted localized centering force with optional neighbor smoothing,
       with steps scaled by ``step_scale`` then capped by surface distance.
       Optional ``active_resample`` may merge short / split long edges after
       each iteration.

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

        logger.debug("BasisOptimizer options: %s", self.options)
        logger.debug(
            "Initialized BasisOptimizer with %d nodes, %d edges, do_pruning=%s, "
            "do_snapping=%s, do_forcing=%s, active_resample=%s",
            self.graph.number_of_nodes(),
            self.graph.number_of_edges(),
            self.options.do_pruning,
            self.options.do_snapping,
            self.options.do_forcing,
            self.options.active_resample,
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
        logger.info(
            "  Input: %d nodes, %d edges, total length %.4f",
            self.graph.number_of_nodes(),
            self.graph.number_of_edges(),
            self.graph.get_total_length(),
        )

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
        stats = self.get_optimization_stats()
        logger.info(
            "Basis optimization complete: %d nodes, %d edges, "
            "%d outside, total length %.4f",
            stats["num_nodes"],
            stats["num_edges"],
            stats["nodes_outside_mesh"],
            stats["total_length"],
        )
        logger.debug("Optimization stats: %s", stats)
        return self.graph

    def _run_pruning_phase(self) -> None:
        """Prune short terminal→branch stubs before geometric optimization.

        Only paths from a terminal (degree 1) through degree-2 nodes to a
        branch (degree ≥ 3) are candidates. Tip↔tip components are left alone.
        The length threshold is resolved once from the graph at phase start.
        """
        logger.info("Phase 0 - Pruning")
        if self.graph.number_of_nodes() == 0:
            logger.debug("Skipping pruning because basis graph is empty")
            return

        threshold = self._resolve_pruning_threshold()
        if threshold is None:
            logger.debug("Skipping pruning because no pruning threshold was configured")
            return

        stub_lengths = self._terminal_stub_lengths(self.graph)
        logger.debug(
            "Pruning threshold=%.6f (%d terminal→branch stubs, longest=%.6f)",
            threshold,
            len(stub_lengths),
            max(stub_lengths) if stub_lengths else 0.0,
        )
        logger.info("  Removing terminal stubs with length < %.4f", threshold)
        current = self.graph.copy()
        prune_round = 0
        while True:
            terminal_nodes = sorted(current.get_terminal_nodes())
            nodes_to_remove: set[int] = set()
            visited_terminals: set[int] = set()

            for terminal in terminal_nodes:
                if terminal in visited_terminals or terminal not in current:
                    continue

                end, path, length = self._trace_from_terminal(current, terminal)
                visited_terminals.add(terminal)
                if len(path) <= 1:
                    continue

                ends_at_branch = end != terminal and current.degree(end) >= 3
                if not (ends_at_branch and length < threshold):
                    continue

                logger.debug(
                    "  prune round %d: terminal=%d branch=%d length=%.6f path=%s",
                    prune_round,
                    terminal,
                    end,
                    length,
                    path,
                )
                nodes_to_remove.update(path[:-1])

            if not nodes_to_remove:
                break

            logger.debug(
                "Pruning round %d: removing %d nodes",
                prune_round,
                len(nodes_to_remove),
            )
            prune_round += 1
            current.remove_nodes_from([n for n in nodes_to_remove if n in current])
            if not self.options.pruning_iterative:
                break

        logger.info(
            "  Pruning done: %d → %d nodes (%d rounds)",
            self.graph.number_of_nodes(),
            current.number_of_nodes(),
            prune_round,
        )
        self.graph = current

    def _resolve_pruning_threshold(self) -> Optional[float]:
        """Resolve absolute or fraction-of-longest-stub pruning threshold.

        Absolute ``pruning_length`` takes precedence. Otherwise
        ``pruning_length_fraction * max(terminal→branch stub lengths)`` from
        the current graph (called once at phase start).
        """
        if self.options.pruning_length is not None:
            return float(self.options.pruning_length)

        fraction = self.options.pruning_length_fraction
        if fraction is None:
            return None
        if fraction <= 0 or fraction > 1:
            raise ValueError(
                f"pruning_length_fraction must be in (0, 1], got {fraction}"
            )

        stub_lengths = self._terminal_stub_lengths(self.graph)
        if not stub_lengths:
            return None
        return float(fraction) * float(max(stub_lengths))

    def _run_snapping_phase(self) -> None:
        """Snap outside basis nodes into the mesh along ray chords.

        For each outside node, move along the nearest-surface direction to a
        point a fraction ``snap_chord_fraction`` of the way from the first to
        the second usable ray–mesh intersection (default 0.25). Failures for
        individual nodes are logged as warnings; a summary warning is emitted
        if any nodes remain outside after the phase.
        """
        outside_node_ids = self.graph.get_outside_nodes(
            self.mesh,
            tol=self.options.outside_distance_tol,
            tol_fraction=self.options.outside_distance_tol_fraction,
        )
        logger.info(
            "Phase 1 - Snapping: %d nodes outside mesh",
            len(outside_node_ids),
        )
        logger.debug("Snapping candidate nodes: %s", outside_node_ids)

        if not outside_node_ids:
            logger.debug("No snapping required because all nodes are inside the mesh")
            return

        for node in outside_node_ids:
            pos_before = self.graph.get_node_position(node)
            if not self._snap_node_inside(node):
                logger.warning(
                    "Failed to snap node %s at %s; leaving position unchanged",
                    node,
                    self.graph.get_node_position(node),
                )
            else:
                pos_after = self.graph.get_node_position(node)
                logger.debug(
                    "Snapped node %s: %s → %s (displacement %.6f)",
                    node,
                    pos_before,
                    pos_after,
                    float(np.linalg.norm(pos_after - pos_before)),
                )

        still_outside = self.graph.get_outside_nodes(
            self.mesh,
            tol=self.options.outside_distance_tol,
            tol_fraction=self.options.outside_distance_tol_fraction,
        )
        if still_outside:
            logger.warning(
                "Snapping complete but %d node(s) remain outside the mesh: %s",
                len(still_outside),
                still_outside,
            )

    def _run_forcing_phase(self) -> None:
        """Iteratively move basis nodes toward the medial axis.

        Each step forms::

            delta_v = (1 - alpha_s) * F_centering
                    + alpha_s * F_smoothing * ||F_centering|| / ||F_smoothing||

        then applies ``step_scale * delta_v`` and caps by ``step_cap_factor``
        times the surface distance along that direction.

        When ``ray_jitter > 0``, each forcing iteration independently perturbs
        every centering-ray direction by a random angular displacement of that
        magnitude (shared across nodes within the iteration).

        Requires every node to be inside the mesh; raises ``RuntimeError`` if
        any node is clearly outside before forcing or after a step (see
        ``outside_distance_tol`` / :func:`mascaf.mesh_contains.point_inside_mesh`;
        signed-distance check — not raw ``contains``, which is flaky on the
        surface). Early stopping (any may halt) uses average movement,
        centering-error fraction of the first-iteration value, and
        centering-error plateau detection.
        """
        logger.info("Phase 2 - Forcing: max %d iterations", self.options.max_iterations)

        outside_before = [
            n
            for n in self.graph.nodes()
            if self._is_clearly_outside(self.graph.get_node_position(n))
        ]
        if outside_before:
            raise RuntimeError(
                "Forcing requires all nodes inside the mesh; "
                f"outside nodes: {outside_before}"
            )

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
        protected = terminal_nodes | branch_nodes
        logger.debug(
            "Forcing protected nodes: %d terminals, %d branches (total %d)",
            len(terminal_nodes),
            len(branch_nodes),
            len(protected),
        )
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("  terminal ids: %s", sorted(terminal_nodes))
            logger.debug("  branch ids: %s", sorted(branch_nodes))
        alpha_s = float(self.options.alpha_s)
        step_scale = float(self.options.step_scale)
        ray_jitter = float(self.options.ray_jitter)
        stop_fraction = float(self.options.centering_error_stop_fraction)
        plateau_tol = float(self.options.centering_error_plateau_tol)
        plateau_patience = max(1, int(self.options.centering_error_plateau_patience))
        increase_patience = max(1, int(self.options.centering_error_increase_patience))
        logger.info(
            "  Forcing alpha_s=%.4f step_scale=%.4f ray_jitter=%.4f "
            "active_resample=%s",
            alpha_s,
            step_scale,
            ray_jitter,
            self.options.active_resample,
        )
        if self.options.active_resample:
            min_len, max_len = self._active_resample_length_bounds()
            logger.info(
                "  Active resample edge lengths in [%.6e, %.6e] "
                "(fractions of bbox diagonal)",
                min_len,
                max_len,
            )

        e0: Optional[float] = None
        e_prev: Optional[float] = None
        plateau_streak = 0
        increase_streak = 0
        rng = np.random.default_rng()

        for iteration in range(self.options.max_iterations):
            old_positions = self.graph.get_all_positions()
            if old_positions.size == 0:
                logger.info("Phase 2 - Forcing skipped because basis graph is empty")
                break

            ray_directions = self._forcing_ray_directions(rng)

            centering_error = 0.0
            for node in list(self.graph.nodes()):
                if node not in self.graph or node in protected:
                    continue

                pos = self.graph.get_node_position(node)
                f_centering = self._compute_centering_force(
                    pos, directions=ray_directions
                )
                f_c_norm = float(np.linalg.norm(f_centering))
                centering_error += f_c_norm
                f_smoothing = self._compute_smoothing_force_for_node(node)
                f_s_norm = float(np.linalg.norm(f_smoothing))
                if f_s_norm > 1e-15 and f_c_norm > 1e-15 and alpha_s > 0.0:
                    delta_v = (1.0 - alpha_s) * f_centering + (
                        alpha_s * f_smoothing * (f_c_norm / f_s_norm)
                    )
                else:
                    delta_v = (1.0 - alpha_s) * f_centering
                delta_v = step_scale * delta_v
                step = self._cap_step_by_surface_distance(pos, delta_v)
                new_pos = pos + step
                if self._is_clearly_outside(new_pos):
                    detail = self._format_forcing_outside_debug(
                        node=node,
                        iteration=iteration,
                        pos=pos,
                        new_pos=new_pos,
                        step=step,
                        delta_v=delta_v,
                        f_centering=f_centering,
                        f_smoothing=f_smoothing,
                    )
                    logger.error(detail)
                    raise RuntimeError(detail)
                self.graph.set_node_position(node, new_pos)
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "  iter=%d node=%d pos=%s → %s step=%s "
                        "||step||=%.6e ||F_c||=%.6e ||F_s||=%.6e",
                        iteration,
                        node,
                        np.array2string(pos, precision=4, suppress_small=True),
                        np.array2string(new_pos, precision=4, suppress_small=True),
                        np.array2string(step, precision=4, suppress_small=True),
                        float(np.linalg.norm(step)),
                        f_c_norm,
                        f_s_norm,
                    )

            movement = self._average_movement(
                old_positions,
                self.graph.get_all_positions(),
            )

            if self.options.active_resample:
                n_merged, n_split = self._active_resample_edges()
                if n_merged or n_split:
                    logger.info(
                        "  Iteration %d: active resample "
                        "merged=%d split=%d",
                        iteration,
                        n_merged,
                        n_split,
                    )
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
                    protected = terminal_nodes | branch_nodes

            logger.info(
                "  Iteration %d: avg movement = %.6f, centering_error = %.6e",
                iteration,
                movement,
                centering_error,
            )

            if e0 is None:
                e0 = centering_error
                if e0 <= 1e-15:
                    logger.info(
                        "  Converged at iteration %d: initial centering error ~0",
                        iteration,
                    )
                    break
            elif centering_error < stop_fraction * e0:
                logger.info(
                    "  Converged at iteration %d: centering error %.6e "
                    "< %.3f * initial %.6e",
                    iteration,
                    centering_error,
                    stop_fraction,
                    e0,
                )
                break

            if e_prev is not None:
                denom = max(e_prev, 1e-15)
                rel_change = abs(centering_error - e_prev) / denom
                if rel_change < plateau_tol:
                    plateau_streak += 1
                else:
                    plateau_streak = 0
                if plateau_streak >= plateau_patience:
                    logger.info(
                        "  Converged at iteration %d: centering error plateau "
                        "(rel change < %.3e for %d iterations)",
                        iteration,
                        plateau_tol,
                        plateau_patience,
                    )
                    break

                if centering_error > e_prev:
                    increase_streak += 1
                    if increase_streak >= increase_patience:
                        logger.info(
                            "  Stopping at iteration %d: centering error "
                            "increased for %d consecutive iterations "
                            "(now %.6e, was %.6e at iter %d)",
                            iteration,
                            increase_streak,
                            centering_error,
                            e_prev,
                            iteration - 1,
                        )
                        break
                else:
                    increase_streak = 0

            e_prev = centering_error

            if movement < self.options.convergence_threshold:
                logger.info(
                    "  Converged at iteration %d: avg movement %.6e "
                    "< threshold %.6e",
                    iteration,
                    movement,
                    self.options.convergence_threshold,
                )
                break

    def _update_edge_lengths(self) -> None:
        """Update edge lengths after node positions have changed."""
        for u, v in self.graph.edges():
            pos_u = self.graph.get_node_position(u)
            pos_v = self.graph.get_node_position(v)
            self.graph.edges[u, v]["length"] = float(np.linalg.norm(pos_v - pos_u))

    def _mesh_bbox_diagonal(self) -> float:
        """Return the mesh axis-aligned bounding-box space diagonal."""
        extents = np.asarray(self.mesh.extents, dtype=float)
        return float(np.linalg.norm(extents))

    def _active_resample_length_bounds(self) -> Tuple[float, float]:
        """Absolute ``(min_length, max_length)`` for active edge resampling.

        Requires ``active_resample_min_fraction`` and
        ``active_resample_max_fraction`` in ``(0, 1]`` with
        ``max_fraction >= 2 * min_fraction`` so bisecting an edge just above
        the max cannot produce halves below the min (avoids merge/split
        oscillation).
        """
        min_frac = self.options.active_resample_min_fraction
        max_frac = self.options.active_resample_max_fraction
        if min_frac is None or max_frac is None:
            raise ValueError(
                "active_resample requires both active_resample_min_fraction "
                "and active_resample_max_fraction (fractions of the mesh "
                "bounding-box diagonal)"
            )
        min_frac = float(min_frac)
        max_frac = float(max_frac)
        if not (0.0 < min_frac <= 1.0):
            raise ValueError(
                f"active_resample_min_fraction must be in (0, 1], got {min_frac}"
            )
        if not (0.0 < max_frac <= 1.0):
            raise ValueError(
                f"active_resample_max_fraction must be in (0, 1], got {max_frac}"
            )
        if max_frac < 2.0 * min_frac:
            raise ValueError(
                "active_resample_max_fraction must be >= 2 * "
                f"active_resample_min_fraction (got max={max_frac}, "
                f"min={min_frac}); otherwise bisecting a long edge can "
                "create edges shorter than the merge threshold"
            )
        diagonal = self._mesh_bbox_diagonal()
        return min_frac * diagonal, max_frac * diagonal

    def _active_resample_edges(self) -> Tuple[int, int]:
        """Merge short edges and bisect long edges toward the length band.

        Alternates consolidate-shortest / bisect-longest until all edges lie
        in ``[min_length, max_length]`` (or a safety iteration cap is hit).

        Returns
        -------
        tuple of int
            ``(n_merged, n_split)``.
        """
        min_length, max_length = self._active_resample_length_bounds()
        n_merged = 0
        n_split = 0
        allow_cycle_collapse = self.options.active_resample_allow_cycle_collapse
        # Bound work by current complexity; each op changes edge count by ±1.
        max_ops = max(1, 4 * self.graph.number_of_edges() + 4)
        for _ in range(max_ops):
            short_edges = sorted(
                (
                    (
                        self._edge_length(self.graph, int(u), int(v)),
                        int(u),
                        int(v),
                    )
                    for u, v in self.graph.edges()
                    if self._edge_length(self.graph, int(u), int(v)) < min_length
                ),
                key=lambda item: item[0],
            )

            merged_this_round = False
            for length, iu, iv in short_edges:
                if (
                    not allow_cycle_collapse
                    and self.graph.consolidation_changes_cycle_count(iu, iv)
                ):
                    cycles_before = self.graph.cyclomatic_number()
                    trial = self.graph.copy()
                    trial.consolidate_nodes(iu, iv)
                    cycles_after = trial.cyclomatic_number()
                    logger.warning(
                        "Skipping active resample consolidation of edge "
                        "(%d, %d) (length=%.6f): would change cycle count "
                        "%d → %d (set active_resample_allow_cycle_collapse=True "
                        "to allow)",
                        iu,
                        iv,
                        length,
                        cycles_before,
                        cycles_after,
                    )
                    continue

                keep = self.graph.consolidate_nodes(iu, iv)
                n_merged += 1
                merged_this_round = True
                logger.debug(
                    "  active resample merge: edge (%d,%d) length=%.6f → node %d",
                    iu,
                    iv,
                    length,
                    keep,
                )
                self._snap_active_resample_node(keep)
                break

            if merged_this_round:
                continue

            longest: Optional[Tuple[float, int, int]] = None
            for u, v in self.graph.edges():
                iu, iv = int(u), int(v)
                length = self._edge_length(self.graph, iu, iv)
                if length > max_length and (
                    longest is None or length > longest[0]
                ):
                    longest = (length, iu, iv)

            if longest is not None:
                length, iu, iv = longest
                mid = self.graph.bisect_edge(iu, iv)
                n_split += 1
                logger.debug(
                    "  active resample split: edge (%d,%d) length=%.6f → node %d",
                    iu,
                    iv,
                    length,
                    mid,
                )
                self._snap_active_resample_node(mid)
                continue
            break
        else:
            logger.warning(
                "Active resample hit operation cap (%d); "
                "some edges may remain outside [%.6e, %.6e]",
                max_ops,
                min_length,
                max_length,
            )
        return n_merged, n_split

    def _snap_active_resample_node(self, node: int) -> None:
        """Snap a consolidate/bisect result inside the mesh if it landed outside.

        Raises
        ------
        RuntimeError
            If the node is outside and chord snapping cannot bring it inside.
        """
        if not self._is_clearly_outside(self.graph.get_node_position(node)):
            return
        if self._snap_node_inside(node):
            logger.debug(
                "Active resample snapped node %s inside the mesh",
                node,
            )
            return
        pos = self.graph.get_node_position(node)
        raise RuntimeError(
            "Active resample placed node "
            f"{node} outside the mesh at {pos} and snapping failed"
        )

    def _snap_node_inside(self, node: int) -> bool:
        """If ``node`` is clearly outside, snap it along a surface chord.

        Returns
        -------
        bool
            ``True`` if the node is not clearly outside afterward (either it
            was already inside, or snapping succeeded). ``False`` if snapping
            failed and the node remains outside.
        """
        pos = self.graph.get_node_position(node)
        if not self._is_clearly_outside(pos):
            return True
        new_pos = self._snap_point_to_chord_midpoint(pos)
        if new_pos is None:
            return False
        logger.debug(
            "Snapping node %s from %s to chord point %s",
            node,
            pos,
            new_pos,
        )
        self.graph.set_node_position(node, new_pos)
        for nbr in list(self.graph.neighbors(node)):
            self.graph.edges[node, nbr]["length"] = float(
                np.linalg.norm(self.graph.get_node_position(nbr) - new_pos)
            )
        return not self._is_clearly_outside(new_pos)

    def _compute_smoothing_force_for_node(self, node: int) -> np.ndarray:
        """Return ``mean(neighbor positions) - v`` (zeros if no neighbors)."""
        neighbors = list(self.graph.neighbors(node))
        if not neighbors:
            return np.zeros(3)

        pos = self.graph.get_node_position(node)
        neighbor_positions = np.array(
            [self.graph.get_node_position(n) for n in neighbors],
            dtype=float,
        )
        return neighbor_positions.mean(axis=0) - pos

    def _compute_centering_force(
        self,
        point: np.ndarray,
        directions: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute a weighted localized centering force toward the medial axis.

        For an interior point, casts ``n_rays`` unit directions ``x_i`` with
        surface distances ``d_i`` and forms weights

        ``w_i = (d_i + eps)^(-p) * exp(-beta * (d_i - d_min) / d_min)``,

        then returns ``F = -d_min * sum_i x_i w_i / sum_i w_i``. Outside
        points fall back to the closest-point direction. Failed individual
        rays are skipped; if no valid rays remain, returns zeros (never
        wall-attracts an interior point via closest-point).

        Parameters
        ----------
        point :
            Query position.
        directions :
            Optional ``(n, 3)`` unit ray directions. When ``None``, uses
            :meth:`_get_uniform_sphere_directions` for ``n_rays``.
        """
        is_inside = self._point_inside_mesh(point)
        if not is_inside:
            return self._compute_closest_point_direction(point)

        if directions is None:
            directions = self._get_uniform_sphere_directions(self.options.n_rays)
        distances: list[float] = []
        valid_directions: list[np.ndarray] = []
        for direction in directions:
            try:
                distance = self._ray_distance_to_surface(point, direction)
            except Exception as exc:
                # Skip failed rays; do not fall back to closest-point for
                # interior queries (that would attract toward the wall).
                logger.debug(
                    "Skipping centering ray from %s along %s: %s",
                    point,
                    direction,
                    exc,
                )
                continue
            if distance > 1e-6:
                distances.append(distance)
                valid_directions.append(np.asarray(direction, dtype=float))

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
        return d_min * (force / weight_sum)

    def _outside_distance_tol(self) -> float:
        """Tolerance for treating a point as clearly outside the mesh."""
        from .mesh_contains import default_distance_tol

        return default_distance_tol(
            self.mesh,
            tol=self.options.outside_distance_tol,
            tol_fraction=self.options.outside_distance_tol_fraction,
        )

    def _signed_distance(self, point: np.ndarray) -> float:
        """Return signed distance to the mesh (positive inside)."""
        from .mesh_contains import signed_distances

        return float(signed_distances(self.mesh, point)[0])

    def _point_inside_mesh(self, point: np.ndarray) -> bool:
        """True if point is inside or within the exterior distance tolerance."""
        from .mesh_contains import point_inside_mesh

        return point_inside_mesh(
            self.mesh,
            point,
            tol=self.options.outside_distance_tol,
            tol_fraction=self.options.outside_distance_tol_fraction,
        )

    def _is_clearly_outside(self, point: np.ndarray) -> bool:
        """True only if the point is exterior beyond the distance tolerance."""
        return not self._point_inside_mesh(point)
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

    def _format_forcing_outside_debug(
        self,
        *,
        node: int,
        iteration: int,
        pos: np.ndarray,
        new_pos: np.ndarray,
        step: np.ndarray,
        delta_v: np.ndarray,
        f_centering: np.ndarray,
        f_smoothing: np.ndarray,
    ) -> str:
        """Build a diagnostic message when a forcing step leaves the mesh."""
        step_norm = float(np.linalg.norm(step))
        delta_norm = float(np.linalg.norm(delta_v))
        f_c_norm = float(np.linalg.norm(f_centering))
        f_s_norm = float(np.linalg.norm(f_smoothing))
        tol = self._outside_distance_tol()

        try:
            inside_before = self._point_inside_mesh(pos)
        except Exception as exc:
            inside_before = f"<error: {exc}>"
        try:
            inside_after = self._point_inside_mesh(new_pos)
        except Exception as exc:
            inside_after = f"<error: {exc}>"
        try:
            signed_before = self._signed_distance(pos)
        except Exception as exc:
            signed_before = f"<error: {exc}>"
        try:
            signed_after = self._signed_distance(new_pos)
        except Exception as exc:
            signed_after = f"<error: {exc}>"

        d_force: float | str
        max_step: float | str
        was_capped: str
        if step_norm <= 1e-10:
            d_force = "n/a (zero step)"
            max_step = "n/a"
            was_capped = "n/a"
            direction = np.zeros(3)
        else:
            direction = step / step_norm
            try:
                d_force = self._ray_distance_to_surface(pos, direction)
                max_step = self.options.step_cap_factor * float(d_force)
                was_capped = (
                    "yes"
                    if delta_norm > float(max_step) + 1e-12
                    else "no"
                )
            except Exception as exc:
                d_force = f"<error: {exc}>"
                max_step = "n/a"
                was_capped = "n/a"

        ratio = (
            step_norm / float(d_force)
            if isinstance(d_force, float) and d_force > 1e-15
            else "n/a"
        )

        return (
            f"Forcing moved node {node} outside the mesh at iteration {iteration}:\n"
            f"  pos_before = {pos}\n"
            f"  pos_after  = {new_pos}\n"
            f"  step       = {step}\n"
            f"  ||step||   = {step_norm:.6e}\n"
            f"  ||delta_v||= {delta_norm:.6e}\n"
            f"  ||F_c||    = {f_c_norm:.6e}\n"
            f"  ||F_s||    = {f_s_norm:.6e}\n"
            f"  direction  = {direction}\n"
            f"  d_force (ray to first hit) = {d_force}\n"
            f"  step_scale = {self.options.step_scale}\n"
            f"  step_cap_factor = {self.options.step_cap_factor}\n"
            f"  max_step   = {max_step}\n"
            f"  ||step||/d_force = {ratio}\n"
            f"  step was capped vs uncapped delta_v: {was_capped}\n"
            f"  inside(before) = {inside_before}\n"
            f"  inside(after)  = {inside_after}\n"
            f"  signed_distance(before) = {signed_before} "
            f"(positive inside, negative outside)\n"
            f"  signed_distance(after)  = {signed_after}\n"
            f"  outside_distance_tol    = {tol:.6e}\n"
            f"  clearly_outside(after)  = "
            f"{isinstance(signed_after, float) and signed_after < -tol}"
        )

    def _snap_min_chord(self) -> float:
        """Minimum accepted enter–exit chord length for snapping."""
        if self.options.snap_min_chord_length is not None:
            return float(self.options.snap_min_chord_length)
        return float(self.options.snap_min_chord_fraction) * self._mesh_bbox_diagonal()

    def _snap_point_to_chord_midpoint(
        self, point: np.ndarray
    ) -> Optional[np.ndarray]:
        """Return a point along a chord toward the nearest surface.

        Tries the closest-point direction, then small angular perturbations.
        Along each ray, consecutive hit pairs are considered (not only the
        first pair, and not only even hit counts): the first pair whose chord
        length is at least :meth:`_snap_min_chord` and whose interpolated
        point (``snap_chord_fraction`` from hit 1 toward hit 2) lies inside
        the mesh is accepted. Returns ``None`` if no suitable chord is found.
        """
        direction, dist = self._compute_snap_direction(point)
        if dist < 1e-10 or float(np.linalg.norm(direction)) < 1e-10:
            logger.warning(
                "Snap failed: point %s is too close to the surface (dist=%s)",
                point,
                dist,
            )
            return None

        mid = self._snap_midpoint_from_ray_candidates(point, direction)
        if mid is None:
            logger.warning(
                "Snap failed: could not find an interior chord from point %s "
                "near direction %s",
                point,
                direction,
            )
            return None
        return mid

    def _snap_point_on_chord(
        self, hit_enter: np.ndarray, hit_exit: np.ndarray
    ) -> np.ndarray:
        """Interpolate ``snap_chord_fraction`` of the way from enter to exit."""
        frac = float(self.options.snap_chord_fraction)
        return (1.0 - frac) * hit_enter + frac * hit_exit

    def _snap_midpoint_from_ray_candidates(
        self,
        point: np.ndarray,
        direction: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Try snap directions until a chord sample point lies inside the mesh.

        Prefers hit pairs with chord length at least :meth:`_snap_min_chord`.
        If none yield an interior sample at ``snap_chord_fraction``, falls back
        to any pair whose sample is inside (short grazing chords included).
        """
        direction = np.asarray(direction, dtype=float)
        direction = direction / (np.linalg.norm(direction) + 1e-15)
        min_chord = self._snap_min_chord()
        fallback: Optional[np.ndarray] = None

        for candidate in self._snap_ray_direction_candidates(direction):
            try:
                hits = self._ray_intersections_sorted(point, candidate)
            except RuntimeError:
                continue
            n_hits = int(hits.shape[0])
            if n_hits < 2:
                logger.debug(
                    "Snap ray from %s along %s produced %d hits (need >= 2)",
                    point,
                    candidate,
                    n_hits,
                )
                continue

            for i in range(n_hits - 1):
                chord = float(np.linalg.norm(hits[i + 1] - hits[i]))
                sample = self._snap_point_on_chord(hits[i], hits[i + 1])
                if not self._point_inside_mesh(sample):
                    continue
                if chord >= min_chord:
                    if i > 0 or not np.allclose(candidate, direction):
                        logger.debug(
                            "Snap chord for point %s: hits pair (%d,%d) of %d, "
                            "chord %.3e, fraction %.3f, direction %s",
                            point,
                            i,
                            i + 1,
                            n_hits,
                            chord,
                            float(self.options.snap_chord_fraction),
                            candidate,
                        )
                    return sample
                if fallback is None:
                    fallback = sample
                    logger.debug(
                        "Snap ray from %s along %s: pair (%d,%d) chord %.3e "
                        "< min %.3e but sample inside (fallback candidate)",
                        point,
                        candidate,
                        i,
                        i + 1,
                        chord,
                        min_chord,
                    )
        return fallback

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

    def _forcing_ray_directions(
        self, rng: Optional[np.random.Generator] = None
    ) -> np.ndarray:
        """Base sphere directions, optionally with per-ray angular jitter."""
        directions = self._get_uniform_sphere_directions(self.options.n_rays)
        jitter = float(self.options.ray_jitter)
        if jitter <= 0.0:
            return directions
        if rng is None:
            rng = np.random.default_rng()
        return self._apply_angular_ray_jitter(directions, jitter, rng)

    def _apply_angular_ray_jitter(
        self,
        directions: np.ndarray,
        jitter: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Perturb each unit direction independently by angular scale ``jitter``.

        For each direction ``d``, samples a random unit vector in the plane
        orthogonal to ``d`` and sets ``d' = normalize(d + jitter * ortho)``.
        For small ``jitter`` this is approximately an angular displacement of
        magnitude ``jitter`` radians.
        """
        dirs = np.asarray(directions, dtype=float)
        out = np.empty_like(dirs)
        for i, d in enumerate(dirs):
            n = float(np.linalg.norm(d))
            if n <= 1e-15:
                out[i] = d
                continue
            d = d / n
            # Random direction in the tangent plane.
            r = rng.normal(size=3)
            ortho = r - d * float(r @ d)
            on = float(np.linalg.norm(ortho))
            if on <= 1e-15:
                # Degenerate sample; leave unperturbed.
                out[i] = d
                continue
            ortho = ortho / on
            jittered = d + float(jitter) * ortho
            jn = float(np.linalg.norm(jittered))
            out[i] = jittered / jn if jn > 1e-15 else d
        return out

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
        outside_node_ids = self.graph.get_outside_nodes(
            self.mesh,
            tol=self.options.outside_distance_tol,
            tol_fraction=self.options.outside_distance_tol_fraction,
        )
        return {
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "num_terminal_nodes": len(self.graph.get_terminal_nodes()),
            "num_branch_nodes": len(self.graph.get_branch_nodes()),
            "total_length": self.graph.get_total_length(),
            "nodes_outside_mesh": len(outside_node_ids),
        }

    def _terminal_stub_lengths(self, graph: Graph3D) -> list[float]:
        """Lengths of terminal→branch stubs (degree-2 chain, no tip↔tip)."""
        lengths: list[float] = []
        for terminal in graph.get_terminal_nodes():
            end, path, length = self._trace_from_terminal(graph, terminal)
            if len(path) <= 1:
                continue
            if end != terminal and graph.degree(end) >= 3:
                lengths.append(float(length))
        return lengths

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
