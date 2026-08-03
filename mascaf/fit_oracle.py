"""Heuristic fit-parameter suggestions from mesh thickness + skeleton topology.

Skeletons have no radius. Local scale comes from mesh shape-diameter (SDF)
thickness; connectivity / length come from the skeleton. Suggestions are
starting values for cable fitting and basis optimization — all fields remain
user-overridable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Any, Mapping, Optional, Union

import numpy as np
import trimesh

from .basis_optimizer import BasisOptimizerOptions
from .mesh import MeshManager
from .shape_diameter import ThicknessSummary, mesh_thickness_summary
from .skeleton import SkeletonGraph

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

MeshLike = Union[trimesh.Trimesh, MeshManager]


@dataclass(frozen=True)
class FitFeatures:
    """Cheap geometric features for fit-parameter heuristics."""

    bbox_diagonal: float
    thickness: ThicknessSummary
    skeleton_length: float
    skeleton_nodes: int
    skeleton_edges: int
    n_terminals: int
    n_branches: int
    cyclomatic_number: int
    watertight: bool


@dataclass(frozen=True)
class SuggestedFitParameters:
    """Oracle suggestions plus the features / rationale used to build them."""

    max_edge_length: float
    max_edge_length_fraction: float
    """``max_edge_length / bbox_diagonal`` (for logging / FittingOptimizer)."""
    basis_optimizer_options: BasisOptimizerOptions
    features: FitFeatures
    mel_over_thickness: float
    """``max_edge_length / thickness.median`` used by the heuristic."""
    rationale: tuple[str, ...]


@dataclass
class FitOracleOptions:
    """Knobs for :func:`suggest_fit_parameters`.

    Default ``mel_over_thickness`` (~2) matches TS1's historical
    ``~0.06 × bbox_diagonal`` sampling when the skeleton is long relative to
    local thickness. For compact spines (short skeleton / thickness), mel is
    also capped by ``skeleton_length / n_target`` so the basis is not
    under-sampled.
    """

    mel_over_thickness: float = 2.0
    mel_over_thickness_min: float = 0.5
    mel_over_thickness_max: float = 3.5
    min_target_edges: int = 12
    """Lower bound on target edge count when capping mel by skeleton length."""
    sdf_n_samples: int = 200
    sdf_n_rays: int = 8
    sdf_seed: Optional[int] = 0
    # Basis defaults informed by TS2 forcing sweeps
    n_rays: int = 12
    ray_jitter: float = 0.1
    localization_beta: float = 2.0
    step_scale: float = 0.5
    alpha_s: float = 0.1
    max_iterations: int = 30
    do_pruning: bool = True
    pruning_length_fraction: float = 0.2
    do_snapping: bool = True
    do_forcing: bool = True
    preserve_terminal_nodes: bool = True
    preserve_branch_nodes: bool = False
    active_resample: bool = True
    # Active-resample band as multiples of suggested mel (converted to D-frac)
    active_resample_min_over_mel: float = 0.5
    active_resample_max_over_mel: float = 1.0
    active_resample_allow_cycle_collapse: bool = False


def _as_mesh(mesh: MeshLike) -> trimesh.Trimesh:
    if isinstance(mesh, MeshManager):
        return mesh.mesh
    return mesh


def _bbox_diagonal(mesh: trimesh.Trimesh) -> float:
    return float(np.linalg.norm(np.asarray(mesh.extents, dtype=float)))


def compute_fit_features(
    mesh: MeshLike,
    skeleton: SkeletonGraph,
    *,
    oracle_options: Optional[FitOracleOptions] = None,
) -> FitFeatures:
    """Extract thickness + skeleton features for the fit oracle."""
    opts = oracle_options or FitOracleOptions()
    tri = _as_mesh(mesh)
    thickness = mesh_thickness_summary(
        tri,
        n_samples=opts.sdf_n_samples,
        n_rays=opts.sdf_n_rays,
        seed=opts.sdf_seed,
    )
    return FitFeatures(
        bbox_diagonal=_bbox_diagonal(tri),
        thickness=thickness,
        skeleton_length=float(skeleton.get_total_length()),
        skeleton_nodes=int(skeleton.number_of_nodes()),
        skeleton_edges=int(skeleton.number_of_edges()),
        n_terminals=int(len(skeleton.get_terminal_nodes())),
        n_branches=int(len(skeleton.get_branch_nodes())),
        cyclomatic_number=int(skeleton.cyclomatic_number()),
        watertight=bool(tri.is_watertight),
    )


def suggest_fit_parameters(
    mesh: MeshLike,
    skeleton: SkeletonGraph,
    *,
    oracle_options: Optional[FitOracleOptions] = None,
    features: Optional[FitFeatures] = None,
    overrides: Optional[Mapping[str, Any]] = None,
) -> SuggestedFitParameters:
    """Suggest ``max_edge_length`` and :class:`BasisOptimizerOptions`.

    Parameters
    ----------
    mesh :
        Target mesh or :class:`~mascaf.mesh.MeshManager`.
    skeleton :
        Guidance skeleton (no radii required).
    oracle_options :
        Heuristic knobs (mel / thickness ratio, SDF sampling, basis defaults).
    features :
        Optional precomputed :class:`FitFeatures` (skips SDF recomputation).
    overrides :
        Optional mapping applied after suggestions. Supported keys:

        - ``max_edge_length`` (absolute)
        - ``max_edge_length_fraction`` (fraction of bbox diagonal)
        - any :class:`BasisOptimizerOptions` field name

    Returns
    -------
    SuggestedFitParameters
    """
    opts = oracle_options or FitOracleOptions()
    feats = features or compute_fit_features(
        mesh, skeleton, oracle_options=opts
    )
    D = feats.bbox_diagonal
    t = feats.thickness.median
    rationale: list[str] = []

    if not np.isfinite(t) or t <= 0:
        # Fallback: historical bbox fraction when SDF fails
        mel = 0.1 * D
        k_used = float("nan")
        rationale.append(
            "SDF thickness unavailable; fell back to max_edge_length = 0.1 × "
            f"bbox_diagonal ({mel:.6g})"
        )
    else:
        k = float(
            np.clip(
                opts.mel_over_thickness,
                opts.mel_over_thickness_min,
                opts.mel_over_thickness_max,
            )
        )
        mel_thickness = k * t
        n_target = max(
            int(opts.min_target_edges),
            int(
                3 * feats.cyclomatic_number
                + feats.n_branches
                + feats.n_terminals
            ),
        )
        mel_length = (
            feats.skeleton_length / n_target
            if feats.skeleton_length > 0 and n_target > 0
            else mel_thickness
        )
        mel = mel_thickness
        k_used = k
        rationale.append(
            f"thickness-based mel = {k:.3g} × SDF median ({t:.6g}) "
            f"→ {mel_thickness:.6g}"
        )
        if mel_length + 1e-12 < mel_thickness:
            mel = mel_length
            k_used = mel / t
            rationale.append(
                f"capped by skeleton sampling density: L/n_target = "
                f"{feats.skeleton_length:.6g}/{n_target} → {mel_length:.6g} "
                f"(mel/t={k_used:.4g})"
            )
        rationale.append(
            f"equivalent bbox fraction = {mel / D:.4g} "
            f"(D={D:.6g}, D/t={D / t:.4g}, L/t="
            f"{feats.skeleton_length / t:.4g})"
        )

    if not feats.watertight:
        rationale.append(
            "Mesh is not watertight; SDF and signed-distance queries may be "
            "less reliable"
        )

    # Active resample as multiples of mel, expressed as D-fractions
    min_len = opts.active_resample_min_over_mel * mel
    max_len = opts.active_resample_max_over_mel * mel
    # Enforce max >= 2 * min for BasisOptimizer oscillation guard
    if max_len < 2.0 * min_len:
        max_len = 2.0 * min_len
        rationale.append(
            "Raised active_resample max to 2× min to satisfy basis-optimizer "
            "merge/split guard"
        )
    min_frac = min_len / D if D > 0 else 0.05
    max_frac = max_len / D if D > 0 else 0.1
    min_frac = float(np.clip(min_frac, 1e-6, 1.0))
    max_frac = float(np.clip(max_frac, min_frac * 2.0, 1.0))

    # Slightly denser rays / jitter when the mesh is “compact” (small D/t)
    n_rays = opts.n_rays
    ray_jitter = opts.ray_jitter
    if np.isfinite(t) and t > 0 and D / t < 12.0:
        n_rays = max(n_rays, 12)
        ray_jitter = max(ray_jitter, 0.1)
        rationale.append(
            f"Compact extent (D/t={D / t:.3g}): using n_rays>={n_rays}, "
            f"ray_jitter>={ray_jitter}"
        )

    basis = BasisOptimizerOptions(
        do_pruning=opts.do_pruning,
        pruning_length_fraction=opts.pruning_length_fraction,
        do_snapping=opts.do_snapping,
        do_forcing=opts.do_forcing,
        n_rays=int(n_rays),
        ray_jitter=float(ray_jitter),
        localization_beta=opts.localization_beta,
        step_scale=opts.step_scale,
        alpha_s=opts.alpha_s,
        max_iterations=opts.max_iterations,
        preserve_terminal_nodes=opts.preserve_terminal_nodes,
        preserve_branch_nodes=opts.preserve_branch_nodes,
        active_resample=opts.active_resample,
        active_resample_min_fraction=min_frac,
        active_resample_max_fraction=max_frac,
        active_resample_allow_cycle_collapse=(
            opts.active_resample_allow_cycle_collapse
        ),
    )

    mel_frac = mel / D if D > 0 else float("nan")

    if overrides:
        ov = dict(overrides)
        if "max_edge_length" in ov:
            mel = float(ov.pop("max_edge_length"))
            mel_frac = mel / D if D > 0 else float("nan")
            rationale.append(f"Override max_edge_length → {mel:.6g}")
        if "max_edge_length_fraction" in ov:
            mel_frac = float(ov.pop("max_edge_length_fraction"))
            mel = mel_frac * D
            rationale.append(
                f"Override max_edge_length_fraction → {mel_frac:.4g} "
                f"(mel={mel:.6g})"
            )
        basis_fields = {
            f.name for f in BasisOptimizerOptions.__dataclass_fields__.values()
        }
        basis_updates = {k: v for k, v in ov.items() if k in basis_fields}
        unknown = sorted(set(ov) - set(basis_updates))
        if unknown:
            raise ValueError(f"Unknown suggest_fit_parameters overrides: {unknown}")
        if basis_updates:
            basis = replace(basis, **basis_updates)
            rationale.append(
                "Override basis options: " + ", ".join(sorted(basis_updates))
            )
        if np.isfinite(t) and t > 0:
            k_used = mel / t
        mel_frac = mel / D if D > 0 else float("nan")

    logger.info(
        "Fit oracle: mel=%.6g (frac=%.4g, mel/t=%.4g), n_rays=%d, "
        "active_resample=[%.4g, %.4g]×D",
        mel,
        mel_frac,
        k_used if np.isfinite(k_used) else float("nan"),
        basis.n_rays,
        float(basis.active_resample_min_fraction or 0.0),
        float(basis.active_resample_max_fraction or 0.0),
    )

    return SuggestedFitParameters(
        max_edge_length=float(mel),
        max_edge_length_fraction=float(mel_frac),
        basis_optimizer_options=basis,
        features=feats,
        mel_over_thickness=float(k_used) if np.isfinite(k_used) else float("nan"),
        rationale=tuple(rationale),
    )


def fraction_bounds_around_suggestion(
    suggested: SuggestedFitParameters,
    *,
    rel_span: float = 0.5,
    absolute_floor: float = 0.01,
    absolute_ceil: float = 0.5,
) -> tuple[float, float]:
    """Build ``FittingOptimizer`` fraction bounds around an oracle suggestion.

    Returns ``(lo, hi)`` clipped to ``[absolute_floor, absolute_ceil]`` with
    ``hi > lo``. Falls back to ``(0.02, 0.2)`` if the suggestion fraction is
    non-finite.
    """
    f = float(suggested.max_edge_length_fraction)
    if not np.isfinite(f) or f <= 0:
        return (0.02, 0.2)
    lo = max(absolute_floor, f * (1.0 - rel_span))
    hi = min(absolute_ceil, f * (1.0 + rel_span))
    if hi <= lo:
        hi = min(absolute_ceil, lo * 1.5 + 1e-6)
    return (float(lo), float(hi))
