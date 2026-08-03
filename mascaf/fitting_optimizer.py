"""Optimize cable-fit ``max_edge_length`` against mesh volume after SA normalization."""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import List, Optional, Sequence, Tuple, Union

import trimesh
from scipy.optimize import minimize_scalar

from .cable_fitting import CableFitter, FitOptions
from .mesh import MeshManager
from .morphology_graph import MorphologyGraph
from .skeleton import SkeletonGraph

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@dataclass
class FittingOptimizerOptions:
    """Configuration for max-edge-length fitting optimization.

    The search variable is ``max_edge_length / mesh_bbox_diagonal``. Each trial
    fits a cable model, scales radii to match mesh surface area, then records
    absolute relative volume error. Among trials within an indifference band of
    the best volume error, the largest fraction is selected.
    """

    fraction_bounds: Tuple[float, float] = (0.02, 0.2)
    account_for_overlaps: bool = False
    xatol: float = 1e-3
    maxiter: int = 25
    volume_error_rel_tol: float = 0.05
    volume_error_abs_tol: float = 0.0


@dataclass
class FittingEvalRecord:
    """One evaluation of the fitting objective at a given edge-length fraction."""

    fraction: float
    max_edge_length: float
    volume_relative_error: float
    scale_factor: float
    morphology: MorphologyGraph


@dataclass
class FittingOptimizeResult:
    """Result of :meth:`FittingOptimizer.optimize`."""

    max_edge_length_fraction: float
    max_edge_length: float
    morphology: MorphologyGraph
    volume_relative_error: float
    best_volume_relative_error: float
    scale_factor: float
    n_evals: int
    history: List[FittingEvalRecord]


def select_prefer_larger(
    history: Sequence[FittingEvalRecord],
    *,
    volume_error_rel_tol: float = 0.05,
    volume_error_abs_tol: float = 0.0,
) -> FittingEvalRecord:
    """Pick the largest fraction among near-best volume-error trials.

    A trial is eligible if its absolute relative volume error satisfies
    ``err <= best_err + abs_tol`` or ``err <= best_err * (1 + rel_tol)``
    (equivalently ``err <= max(best_err + abs_tol, best_err * (1 + rel_tol))``).
    Among eligible trials, the one with the largest ``fraction`` is returned.
    Ties on fraction keep the first such entry in ``history`` order.
    """
    if not history:
        raise ValueError("history must be non-empty")
    if volume_error_rel_tol < 0.0 or volume_error_abs_tol < 0.0:
        raise ValueError("volume error tolerances must be non-negative")

    best_err = min(rec.volume_relative_error for rec in history)
    threshold = max(
        best_err + float(volume_error_abs_tol),
        best_err * (1.0 + float(volume_error_rel_tol)),
    )
    eligible = [
        rec for rec in history if rec.volume_relative_error <= threshold
    ]
    # max with key: for equal fractions, Python keeps the first maximum
    return max(eligible, key=lambda rec: rec.fraction)


class FittingOptimizer:
    """Search ``max_edge_length`` (as a bbox-diagonal fraction) for volume fidelity.

    For each candidate fraction ``f``, runs :class:`~mascaf.CableFitter` with
    ``max_edge_length = f * diagonal`` (and optional
    :attr:`FitOptions.basis_optimizer_options`), scales radii to match mesh
    surface area, then measures absolute relative volume error. After a bounded
    1D search, applies an ε-indifference rule that prefers larger fractions
    among near-best volume errors.

    Parameters
    ----------
    fit_options : FitOptions or None
        Base cable-fit options. ``max_edge_length`` is overwritten each trial.
        Pass ``basis_optimizer_options`` to enable basis optimization inside
        each evaluation.
    options : FittingOptimizerOptions or None
        Search bounds, tolerances, and prefer-larger settings.
    """

    def __init__(
        self,
        fit_options: Optional[FitOptions] = None,
        options: Optional[FittingOptimizerOptions] = None,
    ) -> None:
        self.fit_options = fit_options or FitOptions()
        self.options = options or FittingOptimizerOptions()

    def optimize(
        self,
        mesh: Union[trimesh.Trimesh, MeshManager],
        skeleton: SkeletonGraph,
    ) -> FittingOptimizeResult:
        """Run the search and return the preferred SA-normalized morphology."""
        mesh_obj, diagonal = _resolve_mesh_and_diagonal(mesh)
        if not isinstance(skeleton, SkeletonGraph):
            raise TypeError("skeleton must be a SkeletonGraph instance")
        if skeleton.number_of_nodes() == 0:
            raise ValueError("skeleton is empty")

        lo, hi = self.options.fraction_bounds
        if not (0.0 < lo < hi):
            raise ValueError(
                f"fraction_bounds must satisfy 0 < lo < hi, got {self.options.fraction_bounds}"
            )
        if diagonal <= 0.0:
            raise ValueError("mesh bounding-box diagonal must be positive")

        mesh_volume = float(mesh_obj.volume)
        if not mesh_volume > 0.0:
            raise ValueError("Mesh has zero volume.")

        history: List[FittingEvalRecord] = []

        def objective(fraction: float) -> float:
            rec = self._evaluate(
                float(fraction),
                mesh_obj=mesh_obj,
                skeleton=skeleton,
                diagonal=diagonal,
                mesh_volume=mesh_volume,
            )
            history.append(rec)
            return rec.volume_relative_error

        logger.info(
            "Starting fitting optimization over fraction bounds %s "
            "(diagonal=%.6g, maxiter=%d)",
            self.options.fraction_bounds,
            diagonal,
            self.options.maxiter,
        )
        minimize_scalar(
            objective,
            bounds=(float(lo), float(hi)),
            method="bounded",
            options={
                "xatol": float(self.options.xatol),
                "maxiter": int(self.options.maxiter),
            },
        )

        if not history:
            raise RuntimeError("fitting optimization produced no evaluations")

        selected = select_prefer_larger(
            history,
            volume_error_rel_tol=self.options.volume_error_rel_tol,
            volume_error_abs_tol=self.options.volume_error_abs_tol,
        )
        best_err = min(rec.volume_relative_error for rec in history)

        logger.info(
            "Fitting optimization selected fraction=%.6g (mel=%.6g) with "
            "volume_relative_error=%.6g (best over history=%.6g, n_evals=%d)",
            selected.fraction,
            selected.max_edge_length,
            selected.volume_relative_error,
            best_err,
            len(history),
        )

        return FittingOptimizeResult(
            max_edge_length_fraction=selected.fraction,
            max_edge_length=selected.max_edge_length,
            morphology=selected.morphology,
            volume_relative_error=selected.volume_relative_error,
            best_volume_relative_error=best_err,
            scale_factor=selected.scale_factor,
            n_evals=len(history),
            history=list(history),
        )

    def _evaluate(
        self,
        fraction: float,
        *,
        mesh_obj: trimesh.Trimesh,
        skeleton: SkeletonGraph,
        diagonal: float,
        mesh_volume: float,
    ) -> FittingEvalRecord:
        mel = float(fraction) * float(diagonal)
        trial_options = replace(self.fit_options, max_edge_length=mel)
        morph = CableFitter(trial_options).fit(mesh_obj, skeleton)
        scale_factor = morph.scale_radii_to_match_mesh(
            mesh_obj,
            metric="surface_area",
            account_for_overlaps=self.options.account_for_overlaps,
        )
        morph_volume = morph.compute_volume(
            account_for_overlaps=self.options.account_for_overlaps
        )
        volume_relative_error = abs(morph_volume - mesh_volume) / mesh_volume
        return FittingEvalRecord(
            fraction=float(fraction),
            max_edge_length=mel,
            volume_relative_error=float(volume_relative_error),
            scale_factor=float(scale_factor),
            morphology=morph,
        )


def _resolve_mesh_and_diagonal(
    mesh: Union[trimesh.Trimesh, MeshManager],
) -> Tuple[trimesh.Trimesh, float]:
    """Return ``(trimesh, bbox_diagonal)`` from either supported mesh input."""
    if isinstance(mesh, MeshManager):
        if mesh.mesh is None:
            raise ValueError("MeshManager has no mesh loaded")
        return mesh.mesh, mesh.bounding_box_diagonal()
    if isinstance(mesh, trimesh.Trimesh):
        if len(mesh.vertices) == 0:
            raise ValueError("Mesh is empty or not provided")
        mgr = MeshManager(mesh=mesh, verbose=False)
        return mesh, mgr.bounding_box_diagonal()
    raise TypeError("mesh must be a trimesh.Trimesh or MeshManager")
