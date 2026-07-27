"""Single or batch run of MASCAF pipeline for toric spines.
Outputs figures, SWCs, and validation tables.

Example::

    uv run python scripts/ts_pipeline.py
    uv run python scripts/ts_pipeline.py --spines 1,2,4
    uv run python scripts/ts_pipeline.py --all-spines
    uv run python scripts/ts_pipeline.py --spines 1,2 --log-level DEBUG
    uv run python scripts/ts_pipeline.py --spines 2 --log-file outputs/ts2_debug.log --log-level DEBUG
    uv run python scripts/ts_pipeline.py --spines 1 --log-level DEBUG --full-debug
"""

from __future__ import annotations

import argparse
import logging
import traceback
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation
from swctools import SWCModel, plot_model

from mascaf import (
    BasisOptimizer,
    BasisOptimizerOptions,
    FitOptions,
    MeshManager,
    MorphologyGraph,
    SkeletonGraph,
    Validation,
)
from mascaf.cable_fitting import _compute_morphology_node_radii
from mascaf.logging_config import configure_logging

logger = logging.getLogger(__name__)
_PIPELINE_STEPS = 8

_REPO_ROOT = Path(__file__).resolve().parent.parent
_MESH_DIR = _REPO_ROOT / "data" / "mesh" / "processed"
_SKEL_DIR = _REPO_ROOT / "data" / "mcf_skeletons"
_SWC_ROOT = _REPO_ROOT / "data" / "swc" / "current"
_OUT_DIR = _REPO_ROOT / "outputs"

FIG_WIDTH = 600
FIG_HEIGHT = 450
PDF_SCALE = 2
PLOT_WIDTH = 800
PLOT_HEIGHT = 600

# ---------------------------------------------------------------------------
# Per-spine parameters (edit these)
# ---------------------------------------------------------------------------

_DEFAULT_OPTIMIZER = dict(
    do_pruning=True,
    do_snapping=True,
    do_forcing=True,
    n_rays=6,
    max_iterations=10,
    alpha_s=0.1,
    step_scale=0.5,
    preserve_terminal_nodes=True,
    preserve_branch_nodes=False,
    active_resample=True,
    active_resample_min_fraction=0.05,
    active_resample_max_fraction=0.1,
    active_resample_allow_cycle_collapse=False,
)

SPINES: dict[int, dict[str, Any]] = {
    1: {
        "rotation_deg": [0, -30, 20],
        "zoom": 1.0,
        "qst": 0.5,
        "mcst": 5,
        "max_edge_length_fraction": 0.1,
        "pruning_length_fraction": 0.2,
        **_DEFAULT_OPTIMIZER,
    },
    2: {
        "rotation_deg": [-40, 0, 0],
        "zoom": 1.0,
        "qst": 0.5,
        "mcst": 5,
        "max_edge_length_fraction": 0.1,
        "pruning_length_fraction": 0.2,
        **_DEFAULT_OPTIMIZER,
    },
    3: {
        "rotation_deg": [60, 0, 20],
        "zoom": 1.0,
        "qst": 0.5,
        "mcst": 5,
        "max_edge_length_fraction": 0.1,
        "pruning_length_fraction": 0.2,
        **_DEFAULT_OPTIMIZER,
    },
    4: {
        "rotation_deg": [-30, 40, 50],
        "zoom": 1.0,
        "qst": 0.5,
        "mcst": 5,
        "max_edge_length_fraction": 0.1,
        "pruning_length_fraction": 0.2,
        **_DEFAULT_OPTIMIZER,
    },
    21: {
        "rotation_deg": [0, 30, 95],
        "zoom": 1.0,
        "qst": 0.5,
        "mcst": 5,
        "max_edge_length_fraction": 0.1,
        "pruning_length_fraction": 0.2,
        **_DEFAULT_OPTIMIZER,
    },
    24: {
        "rotation_deg": [0, 0, 0],
        "zoom": 1.0,
        "qst": 0.5,
        "mcst": 5,
        "max_edge_length_fraction": 0.1,
        "pruning_length_fraction": 0.2,
        **_DEFAULT_OPTIMIZER,
    },
    48: {
        "rotation_deg": [0, 0, 0],
        "zoom": 1.0,
        "qst": 0.5,
        "mcst": 5,
        "max_edge_length_fraction": 0.1,
        "pruning_length_fraction": 0.2,
        **_DEFAULT_OPTIMIZER,
    },
    67: {
        "rotation_deg": [0, 0, 0],
        "zoom": 1.0,
        "qst": 0.5,
        "mcst": 5,
        "max_edge_length_fraction": 0.1,
        "pruning_length_fraction": 0.2,
        **_DEFAULT_OPTIMIZER,
    },
    76: {
        "rotation_deg": [0, 0, 0],
        "zoom": 1.0,
        "qst": 0.5,
        "mcst": 5,
        "max_edge_length_fraction": 0.1,
        "pruning_length_fraction": 0.2,
        **_DEFAULT_OPTIMIZER,
    },
}


def _eye_coord(cfg: dict[str, Any]) -> np.ndarray:
    rot = Rotation.from_euler("xyz", cfg["rotation_deg"], degrees=True)
    return rot.apply(np.array([1.0, 1.0, 1.0])) * float(cfg["zoom"])


def _optimizer_options(cfg: dict[str, Any]) -> BasisOptimizerOptions:
    return BasisOptimizerOptions(
        do_pruning=bool(cfg["do_pruning"]),
        pruning_length_fraction=float(cfg["pruning_length_fraction"]),
        do_snapping=bool(cfg["do_snapping"]),
        do_forcing=bool(cfg["do_forcing"]),
        n_rays=int(cfg["n_rays"]),
        max_iterations=int(cfg["max_iterations"]),
        alpha_s=float(cfg["alpha_s"]),
        step_scale=float(cfg["step_scale"]),
        preserve_terminal_nodes=bool(cfg["preserve_terminal_nodes"]),
        preserve_branch_nodes=bool(cfg["preserve_branch_nodes"]),
        active_resample=bool(cfg.get("active_resample", False)),
        active_resample_min_fraction=cfg.get("active_resample_min_fraction"),
        active_resample_max_fraction=cfg.get("active_resample_max_fraction"),
        active_resample_allow_cycle_collapse=bool(
            cfg.get("active_resample_allow_cycle_collapse", False)
        ),
    )


def _pipeline_step(step: int, name: str, detail: str = "") -> None:
    """Log a numbered pipeline phase (INFO banner; DEBUG may add detail)."""
    prefix = f"[{step}/{_PIPELINE_STEPS}] {name}"
    if detail:
        logger.info("%s — %s", prefix, detail)
    else:
        logger.info("%s", prefix)
    logger.debug("Entering pipeline step %d/%d: %s", step, _PIPELINE_STEPS, name)


def _log_geom_metrics(label: str, metrics: dict[str, float]) -> None:
    logger.debug(
        "%s geometry: vol_ratio=%.6f vol_rel_err=%.4f area_ratio=%.6f area_rel_err=%.4f",
        label,
        metrics.get("vol_ratio_raw", float("nan")),
        metrics.get("vol_rel_err_raw", float("nan")),
        metrics.get("area_ratio_raw", float("nan")),
        metrics.get("area_rel_err_raw", float("nan")),
    )


def _apply_camera(fig: Any, eye: np.ndarray) -> None:
    fig.update_layout(
        scene=dict(
            camera=dict(
                eye={"x": float(eye[0]), "y": float(eye[1]), "z": float(eye[2])},
                projection=dict(type="perspective"),
            ),
            aspectmode="data",
        )
    )


def _write_figure(fig: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(
        str(path),
        format="pdf",
        width=FIG_WIDTH,
        height=FIG_HEIGHT,
        scale=PDF_SCALE,
    )
    logger.info("Wrote %s", path.relative_to(_REPO_ROOT))


def _geom_metrics(validator: Validation) -> dict[str, float]:
    """Volume/area ratios with and without overlap correction when needed."""
    has_branches = any(validator.morphology.degree[n] > 2 for n in validator.morphology)
    out: dict[str, float] = {}
    for overlaps in ([False, True] if has_branches else [False]):
        tag = "ov" if overlaps else "raw"
        vol = validator.compare_volumes(account_for_overlaps=overlaps)
        area = validator.compare_surface_areas(account_for_overlaps=overlaps)
        out[f"vol_ratio_{tag}"] = float(vol["ratio"])
        out[f"vol_rel_err_{tag}"] = float(vol["relative_error"])
        out[f"area_ratio_{tag}"] = float(area["ratio"])
        out[f"area_rel_err_{tag}"] = float(area["relative_error"])
        out[f"mesh_volume_{tag}"] = float(vol["mesh_volume"])
        out[f"morph_volume_{tag}"] = float(vol["morphology_volume"])
        out[f"mesh_area_{tag}"] = float(area["mesh_area"])
        out[f"morph_area_{tag}"] = float(area["morphology_area"])
    out["has_branches"] = float(has_branches)
    return out


def run_spine(idx: int, cfg: dict[str, Any]) -> dict[str, Any]:
    """Run the full pipeline for one spine; return metrics row."""
    object_name = f"TS{idx}"
    prefix = f"ts{idx}"
    eye = _eye_coord(cfg)
    qst = float(cfg["qst"])
    mcst = int(cfg["mcst"])

    logger.info("======== %s pipeline start ========", object_name)
    logger.debug("Spine config: %s", cfg)

    mesh_path = _MESH_DIR / f"{object_name}.obj"
    if not mesh_path.exists():
        raise FileNotFoundError(mesh_path)

    polylines_name = f"{object_name}_qst{qst}_mcst{mcst}"
    skel_path = _SKEL_DIR / f"{polylines_name}.polylines.txt"
    if not skel_path.exists():
        raise FileNotFoundError(skel_path)

    _pipeline_step(1, "Load mesh and skeleton", str(mesh_path.name))
    mm = MeshManager(mesh_path=str(mesh_path))
    model_length = mm.bounding_box_diagonal()
    logger.debug(
        "Mesh: %d vertices, %d faces, bbox diagonal=%.4f, watertight=%s",
        len(mm.mesh.vertices),
        len(mm.mesh.faces),
        model_length,
        mm.mesh.is_watertight,
    )

    skeleton = SkeletonGraph.from_txt(str(skel_path))
    logger.debug(
        "Skeleton: %d nodes, %d edges, total length=%.4f",
        skeleton.number_of_nodes(),
        skeleton.number_of_edges(),
        skeleton.get_total_length(),
    )

    max_edge_length = int(model_length * float(cfg["max_edge_length_fraction"]))
    logger.debug(
        "max_edge_length=%d (fraction=%.4f × diagonal)",
        max_edge_length,
        float(cfg["max_edge_length_fraction"]),
    )

    _pipeline_step(2, "Export mesh figures")
    mesh_fig = mm.visualize_mesh_3d(skel=None, show_axes=False, title="")
    _apply_camera(mesh_fig, eye)
    _write_figure(mesh_fig, _OUT_DIR / f"{prefix}_mesh.pdf")

    mesh_skel_fig = mm.visualize_mesh_3d(skel=skeleton, show_axes=False, title="")
    _apply_camera(mesh_skel_fig, eye)
    _write_figure(mesh_skel_fig, _OUT_DIR / f"{prefix}_mesh_skel.pdf")

    _pipeline_step(3, "Resample skeleton to morphology basis")
    basis = MorphologyGraph.from_skeleton_graph_resample(
        skeleton, float(max_edge_length)
    )
    logger.info(
        "%s initial basis: %d nodes, %d edges (mel=%d)",
        object_name,
        basis.number_of_nodes(),
        basis.number_of_edges(),
        max_edge_length,
    )

    opt_opts = _optimizer_options(cfg)
    logger.debug("BasisOptimizerOptions: %s", opt_opts)

    _pipeline_step(4, "Optimize basis (prune / snap / force)")
    optimizer = BasisOptimizer(basis, mm.mesh, opt_opts)
    optimized_basis = optimizer.optimize()
    stats = optimizer.get_optimization_stats()
    logger.debug("Optimizer stats: %s", stats)

    basis_opt_fig = mm.visualize_mesh_3d(
        skel=[basis, optimized_basis],
        show_axes=False,
        title="",
        skel_color=["red", "blue"],
        skel_line_width=2.0,
        skel_marker_size=1.0,
    )
    _apply_camera(basis_opt_fig, eye)
    _write_figure(basis_opt_fig, _OUT_DIR / f"{prefix}_skel_opt.pdf")

    _pipeline_step(5, "Fit radii and write SWC")
    swc_dir = _SWC_ROOT / polylines_name
    swc_dir.mkdir(parents=True, exist_ok=True)
    swc_path = swc_dir / f"{object_name}_mel{max_edge_length}.swc"
    swc_norm_path = swc_dir / f"{object_name}_mel{max_edge_length}_norm.swc"

    fit_options = FitOptions(
        max_edge_length=max_edge_length,
        radius_strategy="equivalent_area",
        section_probe_eps=1e-4,
        section_probe_tries=3,
        multi_tangent_reduction="mean",
        basis_optimizer_options=None,
    )
    logger.debug("FitOptions: %s", fit_options)
    morph = optimized_basis.copy()
    _compute_morphology_node_radii(morph, mm.mesh, fit_options)
    morph.to_swc_file(str(swc_path))
    logger.info("Wrote %s", swc_path.relative_to(_REPO_ROOT))

    _pipeline_step(6, "Pre-normalization validation")
    pre_validator = Validation(mm, skeleton, morph)
    pre_metrics = _geom_metrics(pre_validator)
    _log_geom_metrics("pre-norm", pre_metrics)

    model = SWCModel.from_swc_file(str(swc_path))
    morph_fig = plot_model(
        swc_model=model,
        slider=False,
        title="",
        width=PLOT_WIDTH,
        height=PLOT_HEIGHT,
        show_axes=False,
    )
    _apply_camera(morph_fig, eye)
    _write_figure(morph_fig, _OUT_DIR / f"{prefix}_swc.pdf")

    _pipeline_step(7, "Normalize radii to match mesh surface area")
    morph.scale_radii_to_match_mesh(
        mm.mesh, metric="surface_area", account_for_overlaps=False
    )
    morph.to_swc_file(str(swc_norm_path))
    logger.info("Wrote %s", swc_norm_path.relative_to(_REPO_ROOT))

    post_validator = Validation(mm, skeleton, morph)
    post_metrics = _geom_metrics(post_validator)
    _log_geom_metrics("post-norm", post_metrics)

    swc_model = SWCModel.from_swc_file(str(swc_norm_path))
    norm_fig = plot_model(
        swc_model=swc_model,
        slider=False,
        title="",
        width=PLOT_WIDTH,
        height=PLOT_HEIGHT,
        show_axes=False,
    )
    _apply_camera(norm_fig, eye)
    _write_figure(norm_fig, _OUT_DIR / f"{prefix}_swc_norm.pdf")

    _pipeline_step(8, "Export comparison figures")
    skel_pointset = skeleton.to_point_set()
    vs_fig = plot_model(
        swc_model=swc_model,
        opacity=0.2,
        title="",
        width=PLOT_WIDTH,
        height=PLOT_HEIGHT,
        show_axes=False,
        point_set=skel_pointset,
        point_color="red",
        point_size=model_length * 0.0015,
    )
    _apply_camera(vs_fig, eye)
    _write_figure(vs_fig, _OUT_DIR / f"{prefix}_swc_vs_skel.pdf")

    row: dict[str, Any] = {
        "spine": object_name,
        "idx": idx,
        "max_edge_length": max_edge_length,
        "basis_nodes": basis.number_of_nodes(),
        "basis_edges": basis.number_of_edges(),
        "opt_nodes": stats.get("num_nodes"),
        "opt_edges": stats.get("num_edges"),
        "opt_nodes_outside": stats.get("nodes_outside_mesh"),
        "opt_total_length": stats.get("total_length"),
        "swc_path": str(swc_path.relative_to(_REPO_ROOT)),
        "swc_norm_path": str(swc_norm_path.relative_to(_REPO_ROOT)),
    }
    for k, v in pre_metrics.items():
        row[f"pre_{k}"] = v
    for k, v in post_metrics.items():
        row[f"post_{k}"] = v

    logger.info(
        "======== %s pipeline complete: opt %d→%d nodes, "
        "post-norm vol ratio %.4f ========",
        object_name,
        basis.number_of_nodes(),
        stats.get("num_nodes"),
        post_metrics.get("vol_ratio_raw", float("nan")),
    )
    return row


def _fmt(v: Any, *, pct: bool = False) -> str:
    if v is None:
        return "—"
    try:
        x = float(v)
    except (TypeError, ValueError):
        return str(v)
    if pct:
        return f"{100.0 * x:.2f}%"
    if abs(x) >= 1000:
        return f"{x:.4g}"
    return f"{x:.4f}"


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def write_validation_report(rows: list[dict[str, Any]], path: Path) -> str:
    """Build markdown validation tables; write to ``path`` and return text."""
    if not rows:
        text = "# TS pipeline validation\n\nNo successful spines.\n"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return text

    any_branches = any(
        float(r.get("pre_has_branches", 0)) > 0.5
        or float(r.get("post_has_branches", 0)) > 0.5
        for r in rows
    )

    meta_headers = [
        "spine",
        "mel",
        "basis_n",
        "opt_n",
        "opt_e",
        "outside",
        "total_len",
    ]
    meta_rows = [
        [
            str(r["spine"]),
            _fmt(r["max_edge_length"]),
            _fmt(r["basis_nodes"]),
            _fmt(r["opt_nodes"]),
            _fmt(r["opt_edges"]),
            _fmt(r["opt_nodes_outside"]),
            _fmt(r["opt_total_length"]),
        ]
        for r in rows
    ]

    geom_headers = [
        "spine",
        "pre_vol_ratio",
        "pre_vol_rel_err",
        "pre_area_ratio",
        "pre_area_rel_err",
        "post_vol_ratio",
        "post_vol_rel_err",
        "post_area_ratio",
        "post_area_rel_err",
    ]
    geom_rows = [
        [
            str(r["spine"]),
            _fmt(r.get("pre_vol_ratio_raw")),
            _fmt(r.get("pre_vol_rel_err_raw"), pct=True),
            _fmt(r.get("pre_area_ratio_raw")),
            _fmt(r.get("pre_area_rel_err_raw"), pct=True),
            _fmt(r.get("post_vol_ratio_raw")),
            _fmt(r.get("post_vol_rel_err_raw"), pct=True),
            _fmt(r.get("post_area_ratio_raw")),
            _fmt(r.get("post_area_rel_err_raw"), pct=True),
        ]
        for r in rows
    ]

    sections = [
        "# TS pipeline validation",
        "",
        f"Spines: {', '.join(str(r['spine']) for r in rows)}",
        "",
        "## Run / optimizer summary",
        "",
        _md_table(meta_headers, meta_rows),
        "",
        "## Geometry (account_for_overlaps=False)",
        "",
        _md_table(geom_headers, geom_rows),
    ]

    if any_branches:
        ov_headers = [
            "spine",
            "pre_vol_ratio_ov",
            "pre_area_ratio_ov",
            "post_vol_ratio_ov",
            "post_area_ratio_ov",
        ]
        ov_rows = [
            [
                str(r["spine"]),
                _fmt(r.get("pre_vol_ratio_ov", r.get("pre_vol_ratio_raw"))),
                _fmt(r.get("pre_area_ratio_ov", r.get("pre_area_ratio_raw"))),
                _fmt(r.get("post_vol_ratio_ov", r.get("post_vol_ratio_raw"))),
                _fmt(r.get("post_area_ratio_ov", r.get("post_area_ratio_raw"))),
            ]
            for r in rows
        ]
        sections.extend(
            [
                "",
                "## Geometry (account_for_overlaps=True when branches present)",
                "",
                _md_table(ov_headers, ov_rows),
            ]
        )

    sections.append("")
    text = "\n".join(sections) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    logger.info("Wrote %s", path.relative_to(_REPO_ROOT))
    return text


def _parse_spines(arg: str | None, all_spines: bool) -> list[int]:
    if all_spines:
        return sorted(SPINES.keys())
    if not arg:
        return sorted(SPINES.keys())
    ids: list[int] = []
    for part in arg.split(","):
        part = part.strip()
        if not part:
            continue
        ids.append(int(part))
    unknown = [i for i in ids if i not in SPINES]
    if unknown:
        raise SystemExit(f"Unknown spine indices (not in SPINES): {unknown}")
    return ids


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--all-spines",
        action="store_true",
        help="Run all spines in SPINES (default: only spines specified by --spines).",
    )
    parser.add_argument(
        "--spines",
        default=None,
        help="Comma-separated spine indices (default: all keys in SPINES).",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Write log messages to this file (default: stderr only).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Log level (default: INFO). Use DEBUG for per-step pipeline detail.",
    )
    parser.add_argument(
        "--full-debug",
        action="store_true",
        help=(
            "At DEBUG log level, also emit third-party plotting/export logs "
            "(kaleido, choreographer, etc.). By default those stay at WARNING."
        ),
    )
    args = parser.parse_args(argv)

    configure_logging(
        args.log_level,
        log_file=args.log_file,
        console=True,
        # () disables quieting; None / omitted uses the default quiet list.
        quiet_loggers=() if args.full_debug else None,
    )

    if args.log_file:
        log_path = Path(args.log_file).resolve()
        logger.info("Logging to %s at level %s", log_path, args.log_level)
    else:
        logger.info("Logging to stderr at level %s", args.log_level)

    spine_ids = _parse_spines(args.spines, args.all_spines)
    logger.info("Spines to run: %s", spine_ids)

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    failures: list[tuple[int, str]] = []

    for idx in spine_ids:
        logger.info("==== TS%d ====", idx)
        try:
            row = run_spine(idx, SPINES[idx])
            rows.append(row)
        except Exception:
            msg = traceback.format_exc()
            logger.error("TS%d failed:\n%s", idx, msg)
            failures.append((idx, msg))

    report = write_validation_report(rows, _OUT_DIR / "ts_pipeline_validation.md")
    print(report)

    if failures:
        logger.warning(
            "Failed spines: %s",
            ", ".join(f"TS{i}" for i, _ in failures),
        )
        return 1
    logger.info("All %d spines completed successfully.", len(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
