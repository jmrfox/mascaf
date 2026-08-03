"""Diagnose basis-optimizer centering and local thickness scale for toric spines.

Compares bbox-diagonal mel fractions vs mesh-derived thickness (SDF prototype),
runs batch vs notebook basis configs, and sweeps key forcing params on TS2.

Example::

    uv run python scripts/diagnose_basis_centering.py
    uv run python scripts/diagnose_basis_centering.py --spines 1,2 --skip-sweep
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import traceback
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
from trimesh.proximity import closest_point

from mascaf import (
    BasisOptimizer,
    BasisOptimizerOptions,
    MeshManager,
    MorphologyGraph,
    SkeletonGraph,
)
from mascaf.logging_config import configure_logging
from mascaf.mesh_contains import signed_distances

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_MESH_DIR = _REPO_ROOT / "data" / "mesh" / "processed"
_SKEL_DIR = _REPO_ROOT / "data" / "mcf_skeletons"
_OUT_DIR = _REPO_ROOT / "outputs" / "basis_diag"


# ---------------------------------------------------------------------------
# Config presets (mirrors ts_pipeline / ts_pipeline_dev)
# ---------------------------------------------------------------------------

BATCH_OPTIMIZER = dict(
    do_pruning=True,
    pruning_length_fraction=0.2,
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
    ray_jitter=0.0,
    localization_beta=2.0,
)

NOTEBOOK_OPTIMIZER = dict(
    do_pruning=True,
    pruning_length_fraction=0.1,
    do_snapping=True,
    do_forcing=True,
    n_rays=6,
    max_iterations=10,
    alpha_s=0.1,
    step_scale=0.1,
    step_cap_factor=0.5,
    preserve_terminal_nodes=True,
    preserve_branch_nodes=False,
    active_resample=True,
    active_resample_min_fraction=0.05,
    active_resample_max_fraction=0.2,
    active_resample_allow_cycle_collapse=False,
    ray_jitter=0.1,
    localization_beta=1.0,
)


def _options_from_dict(d: dict[str, Any]) -> BasisOptimizerOptions:
    fields = {f.name for f in BasisOptimizerOptions.__dataclass_fields__.values()}
    return BasisOptimizerOptions(**{k: v for k, v in d.items() if k in fields})


# ---------------------------------------------------------------------------
# Thickness / SDF prototype (mesh-derived; skeleton has no radius)
# ---------------------------------------------------------------------------


def _uniform_cone_directions(
    axis: np.ndarray,
    n_rays: int,
    cone_angle: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Unit directions in a cone around ``axis`` (radians half-angle)."""
    axis = np.asarray(axis, dtype=float)
    axis = axis / (np.linalg.norm(axis) + 1e-15)
    # Orthonormal basis
    tmp = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(axis, tmp)
    u /= np.linalg.norm(u) + 1e-15
    v = np.cross(axis, u)
    dirs = []
    for _ in range(n_rays):
        # Uniform in cone: cos(theta) in [cos(cone), 1]
        cos_a = np.cos(cone_angle)
        z = rng.uniform(cos_a, 1.0)
        phi = rng.uniform(0.0, 2.0 * np.pi)
        r = np.sqrt(max(0.0, 1.0 - z * z))
        d = z * axis + r * np.cos(phi) * u + r * np.sin(phi) * v
        dirs.append(d / (np.linalg.norm(d) + 1e-15))
    return np.asarray(dirs, dtype=float)


def shape_diameter_samples(
    mesh,
    *,
    n_samples: int = 200,
    n_rays: int = 8,
    cone_angle: float = np.deg2rad(45.0),
    normal_offset_fraction: float = 1e-4,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Classic surface SDF: cone of inward rays → opposite-surface distances.

    Returns one diameter estimate per successful sample (empty if none).
    """
    rng = rng or np.random.default_rng(0)
    mesh = mesh.copy()
    if not hasattr(mesh, "face_normals") or mesh.face_normals is None:
        mesh.rezero()
    n_faces = len(mesh.faces)
    if n_faces == 0:
        return np.zeros(0, dtype=float)

    face_idx = rng.choice(n_faces, size=min(n_samples, n_faces), replace=False)
    triangles = mesh.triangles[face_idx]
    # Face centroids
    origins = triangles.mean(axis=1)
    normals = mesh.face_normals[face_idx]
    # Ensure normals point outward-ish via signed distance at a slight outward probe
    diag = float(np.linalg.norm(mesh.extents))
    eps = max(diag * normal_offset_fraction, 1e-9)

    diameters: list[float] = []
    for origin, normal in zip(origins, normals):
        n = np.asarray(normal, dtype=float)
        n = n / (np.linalg.norm(n) + 1e-15)
        # Probe just outside and inside to orient normal outward
        sd_out = float(signed_distances(mesh, origin + eps * n)[0])
        # If outward probe is more inside (higher sd), flip
        if sd_out > 0:
            n = -n
        # Start slightly inside along -outward (= inward)
        start = origin - eps * n
        inward = -n
        dirs = _uniform_cone_directions(inward, n_rays, cone_angle, rng)
        hits: list[float] = []
        for d in dirs:
            try:
                locations, _, _ = mesh.ray.intersects_location(
                    ray_origins=start.reshape(1, 3),
                    ray_directions=d.reshape(1, 3),
                )
            except Exception:
                continue
            if len(locations) == 0:
                continue
            dist = np.linalg.norm(locations - start, axis=1)
            # Skip the near-zero self-hit; take first meaningful opposite hit
            order = np.argsort(dist)
            for di in dist[order]:
                if di > 5.0 * eps:
                    hits.append(float(di))
                    break
        if hits:
            diameters.append(float(np.median(hits)))
    return np.asarray(diameters, dtype=float)


def skeleton_closest_thickness(mesh, skeleton: SkeletonGraph) -> np.ndarray:
    """2 * |signed distance| at skeleton nodes (underestimates if off-medial)."""
    positions = skeleton.get_all_positions()
    if positions.size == 0:
        return np.zeros(0, dtype=float)
    sd = signed_distances(mesh, positions)
    return 2.0 * np.abs(sd)


def skeleton_anchored_diameter(
    mesh,
    skeleton: SkeletonGraph,
    *,
    n_rays: int = 12,
    max_nodes: int = 80,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Multi-ray diameter estimates at skeleton sample points."""
    rng = rng or np.random.default_rng(1)
    positions = skeleton.get_all_positions()
    if positions.size == 0:
        return np.zeros(0, dtype=float)
    if len(positions) > max_nodes:
        idx = rng.choice(len(positions), size=max_nodes, replace=False)
        positions = positions[idx]

    # Fibonacci-ish sphere directions
    indices = np.arange(n_rays, dtype=float) + 0.5
    phi = (1 + np.sqrt(5)) / 2
    theta = 2 * np.pi * indices / phi
    z = 1 - (2 * indices / n_rays)
    radius = np.sqrt(np.maximum(0.0, 1 - z * z))
    dirs = np.column_stack([radius * np.cos(theta), radius * np.sin(theta), z])

    diameters: list[float] = []
    for pos in positions:
        dists: list[float] = []
        for d in dirs:
            try:
                locations, _, _ = mesh.ray.intersects_location(
                    ray_origins=pos.reshape(1, 3),
                    ray_directions=d.reshape(1, 3),
                )
            except Exception:
                continue
            if len(locations) == 0:
                continue
            dist = float(np.min(np.linalg.norm(locations - pos, axis=1)))
            if dist > 1e-8:
                dists.append(dist)
        if len(dists) >= 2:
            # Robust diameter ≈ 2 * median ray distance if roughly centered
            diameters.append(2.0 * float(np.median(dists)))
    return np.asarray(diameters, dtype=float)


def _summary(arr: np.ndarray, prefix: str) -> dict[str, float]:
    if arr.size == 0:
        return {
            f"{prefix}_n": 0.0,
            f"{prefix}_med": float("nan"),
            f"{prefix}_mean": float("nan"),
            f"{prefix}_p10": float("nan"),
            f"{prefix}_p90": float("nan"),
            f"{prefix}_cv": float("nan"),
        }
    med = float(np.median(arr))
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    return {
        f"{prefix}_n": float(arr.size),
        f"{prefix}_med": med,
        f"{prefix}_mean": mean,
        f"{prefix}_p10": float(np.percentile(arr, 10)),
        f"{prefix}_p90": float(np.percentile(arr, 90)),
        f"{prefix}_cv": float(std / med) if med > 1e-12 else float("nan"),
    }


# ---------------------------------------------------------------------------
# Centering metrics
# ---------------------------------------------------------------------------


@dataclass
class CenteringNodeStats:
    node: int
    d_min: float
    d_max: float
    d_ratio: float
    n_valid_rays: int
    used_fallback: bool


def measure_node_centering(
    optimizer: BasisOptimizer,
    node: int,
    directions: np.ndarray,
) -> CenteringNodeStats:
    pos = optimizer.graph.get_node_position(node)
    distances: list[float] = []
    used_fallback = False
    try:
        for direction in directions:
            try:
                d = optimizer._ray_distance_to_surface(pos, direction)
                if d > 1e-6:
                    distances.append(float(d))
            except Exception:
                continue
        if not distances:
            used_fallback = True
            d_min = d_max = float("nan")
            ratio = float("nan")
        else:
            d_min = float(min(distances))
            d_max = float(max(distances))
            ratio = d_max / d_min if d_min > 1e-12 else float("inf")
    except Exception:
        used_fallback = True
        d_min = d_max = ratio = float("nan")
        distances = []
    return CenteringNodeStats(
        node=int(node),
        d_min=d_min,
        d_max=d_max,
        d_ratio=ratio,
        n_valid_rays=len(distances),
        used_fallback=used_fallback,
    )


def summarize_centering(
    optimizer: BasisOptimizer,
    *,
    n_rays: int = 12,
) -> dict[str, float]:
    directions = optimizer._get_uniform_sphere_directions(n_rays)
    ratios: list[float] = []
    dmins: list[float] = []
    fallbacks = 0
    n = 0
    for node in optimizer.graph.nodes():
        st = measure_node_centering(optimizer, node, directions)
        n += 1
        if st.used_fallback or not np.isfinite(st.d_ratio):
            fallbacks += 1
            continue
        ratios.append(st.d_ratio)
        dmins.append(st.d_min)
    if not ratios:
        return {
            "n_nodes": float(n),
            "fallback_frac": 1.0 if n else float("nan"),
            "ratio_med": float("nan"),
            "ratio_mean": float("nan"),
            "ratio_p90": float("nan"),
            "dmin_med": float("nan"),
        }
    return {
        "n_nodes": float(n),
        "fallback_frac": float(fallbacks / max(n, 1)),
        "ratio_med": float(np.median(ratios)),
        "ratio_mean": float(np.mean(ratios)),
        "ratio_p90": float(np.percentile(ratios, 90)),
        "dmin_med": float(np.median(dmins)),
    }


def instrument_fallback_counter(
    optimizer: BasisOptimizer,
) -> Callable[[], int]:
    """Wrap centering force to count closest-point fallbacks; return getter."""
    count = {"n": 0}
    original = optimizer._compute_centering_force

    def wrapped(point, directions=None):
        # Replicate interior path with tracking
        if not optimizer._point_inside_mesh(point):
            count["n"] += 1
            return optimizer._compute_closest_point_direction(point)
        try:
            return original(point, directions=directions)
        except Exception:
            count["n"] += 1
            return optimizer._compute_closest_point_direction(point)

    # Also detect when original itself returns closest-point after internal except
    # by monkeypatching the except path via wrapping _ray_distance to count misses
    ray_misses = {"n": 0}
    orig_ray = optimizer._ray_distance_to_surface

    def ray_wrapped(point, direction):
        try:
            return orig_ray(point, direction)
        except Exception:
            ray_misses["n"] += 1
            raise

    optimizer._ray_distance_to_surface = ray_wrapped  # type: ignore[method-assign]
    optimizer._compute_centering_force = wrapped  # type: ignore[method-assign]

    def getter() -> int:
        return int(count["n"] + ray_misses["n"])

    return getter


# ---------------------------------------------------------------------------
# Per-spine diagnosis
# ---------------------------------------------------------------------------


def load_spine(idx: int, qst: float = 0.5, mcst: int = 5):
    mesh_path = _MESH_DIR / f"TS{idx}.obj"
    skel_path = _SKEL_DIR / f"TS{idx}_qst{qst}_mcst{mcst}.polylines.txt"
    mm = MeshManager(mesh_path=str(mesh_path))
    skeleton = SkeletonGraph.from_txt(str(skel_path))
    return mm, skeleton


def geometric_features(mm: MeshManager, skeleton: SkeletonGraph) -> dict[str, float]:
    mesh = mm.mesh
    D = float(mm.bounding_box_diagonal())
    sdf = shape_diameter_samples(mesh, n_samples=150, n_rays=6)
    sk_close = skeleton_closest_thickness(mesh, skeleton)
    sk_diam = skeleton_anchored_diameter(mesh, skeleton, n_rays=12, max_nodes=60)

    feats: dict[str, float] = {
        "bbox_diagonal": D,
        "n_mesh_verts": float(len(mesh.vertices)),
        "n_mesh_faces": float(len(mesh.faces)),
        "watertight": float(bool(mesh.is_watertight)),
        "skel_nodes": float(skeleton.number_of_nodes()),
        "skel_edges": float(skeleton.number_of_edges()),
        "skel_length": float(skeleton.get_total_length()),
        "skel_terminals": float(len(skeleton.get_terminal_nodes())),
        "skel_branches": float(len(skeleton.get_branch_nodes())),
        "cyclomatic": float(skeleton.cyclomatic_number()),
    }
    feats.update(_summary(sdf, "sdf"))
    feats.update(_summary(sk_close, "sk_closest_diam"))
    feats.update(_summary(sk_diam, "sk_anchored_diam"))

    mel_01 = 0.1 * D
    t = feats["sdf_med"]
    if np.isfinite(t) and t > 0:
        feats["mel_0.1D"] = mel_01
        feats["mel_0.1D_over_sdf_med"] = mel_01 / t
        feats["D_over_sdf_med"] = D / t
        feats["L_over_sdf_med"] = feats["skel_length"] / t
    else:
        feats["mel_0.1D"] = mel_01
        feats["mel_0.1D_over_sdf_med"] = float("nan")
        feats["D_over_sdf_med"] = float("nan")
        feats["L_over_sdf_med"] = float("nan")
    return feats


def run_basis_case(
    mm: MeshManager,
    skeleton: SkeletonGraph,
    *,
    mel_fraction: float,
    opt_dict: dict[str, Any],
    label: str,
) -> dict[str, Any]:
    D = float(mm.bounding_box_diagonal())
    mel = max(1.0, D * mel_fraction)
    basis = MorphologyGraph.from_skeleton_graph_resample(skeleton, float(mel))
    opts = _options_from_dict(opt_dict)
    optimizer = BasisOptimizer(basis, mm.mesh, opts)

    pre = summarize_centering(optimizer, n_rays=12)
    fallback_getter = instrument_fallback_counter(optimizer)

    try:
        optimized = optimizer.optimize()
        ok = True
        err = ""
    except Exception as exc:
        optimized = optimizer.graph
        ok = False
        err = f"{type(exc).__name__}: {exc}"
        logger.error("Optimize failed (%s): %s", label, err)
        logger.debug(traceback.format_exc())

    # Rebuild a fresh optimizer view on result for post metrics
    post_opt = BasisOptimizer(optimized, mm.mesh, replace(opts, do_forcing=False))
    post = summarize_centering(post_opt, n_rays=12)
    stats = optimizer.get_optimization_stats()

    return {
        "label": label,
        "ok": ok,
        "error": err,
        "mel": float(mel),
        "mel_fraction": float(mel_fraction),
        "pre_ratio_med": pre["ratio_med"],
        "pre_ratio_p90": pre["ratio_p90"],
        "pre_dmin_med": pre["dmin_med"],
        "post_ratio_med": post["ratio_med"],
        "post_ratio_p90": post["ratio_p90"],
        "post_dmin_med": post["dmin_med"],
        "post_fallback_frac": post["fallback_frac"],
        "force_fallback_events": float(fallback_getter()),
        "n_nodes_out": float(stats["num_nodes"]),
        "n_edges_out": float(stats["num_edges"]),
        "n_terminals": float(stats["num_terminal_nodes"]),
        "n_branches": float(stats["num_branch_nodes"]),
        "nodes_outside": float(stats["nodes_outside_mesh"]),
        "total_length": float(stats["total_length"]),
    }


def ts2_param_sweep(mm: MeshManager, skeleton: SkeletonGraph) -> list[dict[str, Any]]:
    base = dict(BATCH_OPTIMIZER)
    grid: list[dict[str, Any]] = []
    for n_rays in (6, 12, 24):
        for ray_jitter in (0.0, 0.1):
            for step_scale in (0.1, 0.5):
                for localization_beta in (1.0, 2.0):
                    for max_iterations in (10, 50):
                        # Keep sweep tractable: skip some combos
                        if max_iterations == 50 and n_rays == 24 and ray_jitter == 0.1:
                            continue
                        cfg = dict(base)
                        cfg.update(
                            n_rays=n_rays,
                            ray_jitter=ray_jitter,
                            step_scale=step_scale,
                            localization_beta=localization_beta,
                            max_iterations=max_iterations,
                        )
                        label = (
                            f"nr{n_rays}_jit{ray_jitter}_ss{step_scale}"
                            f"_beta{localization_beta}_it{max_iterations}"
                        )
                        grid.append((label, cfg))

    # Subsample grid to ~24 runs for runtime
    rng = np.random.default_rng(0)
    if len(grid) > 24:
        pick = rng.choice(len(grid), size=24, replace=False)
        grid = [grid[i] for i in sorted(pick)]

    rows: list[dict[str, Any]] = []
    for label, cfg in grid:
        logger.info("TS2 sweep: %s", label)
        row = run_basis_case(
            mm,
            skeleton,
            mel_fraction=0.1,
            opt_dict=cfg,
            label=label,
        )
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        if not np.isfinite(v):
            return "nan"
        return f"{v:.4g}"
    return str(v)


def write_markdown_report(
    path: Path,
    feature_rows: list[dict[str, Any]],
    case_rows: list[dict[str, Any]],
    sweep_rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# Basis centering / thickness diagnosis",
        "",
        "## Geometric features (mesh thickness + skeleton topology)",
        "",
        "SDF = classic surface shape-diameter samples (mesh-derived). "
        "Skeletons have no radius; `sk_closest_diam` is `2*|signed_distance|` "
        "at skeleton nodes (underestimates if off-medial).",
        "",
    ]
    keys = [
        "spine",
        "bbox_diagonal",
        "watertight",
        "sdf_med",
        "sdf_cv",
        "sk_closest_diam_med",
        "sk_anchored_diam_med",
        "mel_0.1D",
        "mel_0.1D_over_sdf_med",
        "D_over_sdf_med",
        "skel_length",
        "cyclomatic",
        "skel_terminals",
        "skel_branches",
    ]
    lines.append("| " + " | ".join(keys) + " |")
    lines.append("| " + " | ".join("---" for _ in keys) + " |")
    for r in feature_rows:
        lines.append("| " + " | ".join(_fmt(r.get(k, "")) for k in keys) + " |")

    lines += [
        "",
        "## Basis opt: batch vs notebook (mel_fraction=0.1 unless noted)",
        "",
    ]
    ckeys = [
        "spine",
        "label",
        "mel",
        "pre_ratio_med",
        "post_ratio_med",
        "post_ratio_p90",
        "force_fallback_events",
        "n_nodes_out",
        "n_terminals",
        "nodes_outside",
        "ok",
    ]
    lines.append("| " + " | ".join(ckeys) + " |")
    lines.append("| " + " | ".join("---" for _ in ckeys) + " |")
    for r in case_rows:
        lines.append("| " + " | ".join(_fmt(r.get(k, "")) for k in ckeys) + " |")

    if sweep_rows:
        lines += [
            "",
            "## TS2 param sweep (lower post_ratio_med is better centering)",
            "",
        ]
        # Sort by post_ratio_med
        ranked = sorted(
            sweep_rows,
            key=lambda r: (
                not r.get("ok", False),
                r.get("post_ratio_med", float("inf"))
                if np.isfinite(r.get("post_ratio_med", float("nan")))
                else float("inf"),
            ),
        )
        skeys = [
            "label",
            "post_ratio_med",
            "post_ratio_p90",
            "force_fallback_events",
            "n_nodes_out",
            "ok",
        ]
        lines.append("| " + " | ".join(skeys) + " |")
        lines.append("| " + " | ".join("---" for _ in skeys) + " |")
        for r in ranked[:20]:
            lines.append("| " + " | ".join(_fmt(r.get(k, "")) for k in skeys) + " |")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spines", default="1,2,3", help="Comma-separated spine indices")
    parser.add_argument("--skip-sweep", action="store_true", help="Skip TS2 param sweep")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    configure_logging(level=args.log_level)
    spines = [int(x) for x in args.spines.split(",") if x.strip()]
    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    feature_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    sweep_rows: list[dict[str, Any]] = []

    loaded: dict[int, tuple] = {}

    for idx in spines:
        logger.info("======== TS%d geometry ========", idx)
        mm, skeleton = load_spine(idx)
        loaded[idx] = (mm, skeleton)
        feats = geometric_features(mm, skeleton)
        feats["spine"] = idx
        feature_rows.append(feats)
        logger.info(
            "TS%d: D=%.4g sdf_med=%.4g mel/sdf=%.4g cyclomatic=%s watertight=%s",
            idx,
            feats["bbox_diagonal"],
            feats["sdf_med"],
            feats["mel_0.1D_over_sdf_med"],
            int(feats["cyclomatic"]),
            bool(feats["watertight"]),
        )

        for name, mel_frac, opt in (
            ("batch_mel0.1", 0.1, BATCH_OPTIMIZER),
            ("notebook_mel0.06", 0.06, NOTEBOOK_OPTIMIZER),
            ("batch_mel0.06", 0.06, BATCH_OPTIMIZER),
        ):
            logger.info("TS%d case %s", idx, name)
            row = run_basis_case(
                mm,
                skeleton,
                mel_fraction=mel_frac,
                opt_dict=opt,
                label=name,
            )
            row["spine"] = idx
            case_rows.append(row)
            logger.info(
                "  post_ratio_med=%.4g nodes=%s fallback_events=%s",
                row["post_ratio_med"],
                int(row["n_nodes_out"]),
                int(row["force_fallback_events"]),
            )

    if 2 in loaded and not args.skip_sweep:
        logger.info("======== TS2 param sweep ========")
        mm, skeleton = loaded[2]
        sweep_rows = ts2_param_sweep(mm, skeleton)

    _write_csv(_OUT_DIR / "features.csv", feature_rows)
    _write_csv(_OUT_DIR / "basis_cases.csv", case_rows)
    _write_csv(_OUT_DIR / "ts2_sweep.csv", sweep_rows)
    write_markdown_report(
        _OUT_DIR / "report.md",
        feature_rows,
        case_rows,
        sweep_rows,
    )
    # Also dump features JSON for easy reading
    (_OUT_DIR / "features.json").write_text(
        json.dumps(feature_rows, indent=2), encoding="utf-8"
    )
    logger.info("Wrote diagnosis under %s", _OUT_DIR)


if __name__ == "__main__":
    main()
