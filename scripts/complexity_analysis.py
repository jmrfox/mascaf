"""Sweep MASCAF resolution and write a complexity table as CSV.

Fits cylinder, torus, branching, and the toric spines at three
thickness-relative ``max_edge_length`` values. Each (model, resolution)
case is fit ``N`` times (default 3) because basis optimization is
stochastic. Demo geometries (``group=demo``) use notebook ``max_edge_length`` values,
skip basis optimization, and use a single replicate. Only ``branching`` is
scored with and without ``extend_terminals()``; other demos use the
unextended cable only. Summarization keeps the extend setting with lower
overlap-subtracted volume error for demos.
Each fit is then scored twice for spines and branching: as fitted, and after
``extend_terminals()`` with its default length and radius scales.
Volume and surface-area errors are recorded with and without branch
overlap corrections. Each cable is then scaled so its total surface area
matches the mesh, and the post-normalization volume error (no overlap
subtraction) is recorded. Scaling exponents and downstream summaries use
the replicate whose overlap-subtracted volume ratio is nearest 1,
separately for each terminal-extension setting.

Each row records mesh and skeleton size, fit fidelity, wall time, and
process working-set RAM. After the fits for a model, all rows for that
model receive a log-log scaling exponent of runtime and peak RAM versus
morphology node count (from the best replicate at each resolution).

Example::

    uv run python scripts/complexity_analysis.py
    uv run python scripts/complexity_analysis.py --models cylinder
    uv run python scripts/complexity_analysis.py --models cylinder,TS1 --mel-over-thickness 1,2,3
"""

from __future__ import annotations

import argparse
import copy
import csv
import ctypes
import logging
import sys
import threading
import time
from ctypes import wintypes
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mascaf import (
    CableFitter,
    FitOptions,
    FitOracleOptions,
    MeshManager,
    SkeletonGraph,
    Validation,
    compute_fit_features,
    suggest_fit_parameters,
)
from mascaf.logging_config import configure_logging

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_OUTPUT = _REPO_ROOT / "outputs" / "complexity_analysis.csv"
_DEMO_NAMES = ("cylinder", "torus", "branching")
# Notebook ``max_edge_length`` values (see ``notebooks/demo_geometries/*_mascaf.py``).
_DEMO_MAX_EDGE_LENGTH: dict[str, float] = {
    "cylinder": 3.0,
    "torus": 1.0,
    "branching": 1.0,
}
_SPINE_IDS = (1, 2, 3, 4, 21, 24, 48, 67, 76)
_DEFAULT_K = (1.0, 2.0, 3.0)
_DEFAULT_N_RUNS = 3
_SAMPLE_INTERVAL_S = 0.05

_COLUMNS = (
    "model",
    "group",
    "mel_over_thickness",
    "extend_terminals",
    "run_index",
    "n_runs",
    "max_edge_length",
    "max_edge_length_fraction",
    "mesh_vertices",
    "mesh_faces",
    "mesh_area",
    "mesh_volume",
    "bbox_diagonal",
    "watertight",
    "genus",
    "component_count",
    "thickness_median",
    "thickness_p10",
    "thickness_p90",
    "skeleton_nodes",
    "skeleton_edges",
    "skeleton_length",
    "n_terminals",
    "n_branches",
    "cyclomatic_number",
    "morphology_nodes",
    "morphology_edges",
    "volume_ratio",
    "volume_relative_error",
    "area_ratio",
    "area_relative_error",
    "volume_ratio_overlaps",
    "volume_relative_error_overlaps",
    "area_ratio_overlaps",
    "area_relative_error_overlaps",
    "volume_ratio_sa_norm",
    "volume_relative_error_sa_norm",
    "runtime_s",
    "rss_before_mb",
    "peak_rss_mb",
    "rss_delta_mb",
    "runtime_scaling_exponent",
    "ram_scaling_exponent",
)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    group: str
    mesh_path: Path
    skeleton_path: Path


class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def working_set_bytes() -> int:
    """Current process working set in bytes.

    On Windows this is ``WorkingSetSize`` from ``GetProcessMemoryInfo``.
    On Linux it is ``VmRSS``. Other platforms return 0.
    """
    if sys.platform == "win32":
        return _windows_working_set()
    status = Path("/proc/self/status")
    if status.is_file():
        return _linux_vm_rss(status)
    return 0


def _windows_working_set() -> int:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
        wintypes.DWORD,
    ]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    counters = PROCESS_MEMORY_COUNTERS()
    counters.cb = ctypes.sizeof(counters)
    ok = psapi.GetProcessMemoryInfo(
        kernel32.GetCurrentProcess(),
        ctypes.byref(counters),
        counters.cb,
    )
    if not ok:
        raise OSError(ctypes.get_last_error(), "GetProcessMemoryInfo failed")
    return int(counters.WorkingSetSize)


def _linux_vm_rss(status_path: Path) -> int:
    for line in status_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    raise OSError(f"VmRSS missing from {status_path}")


class WorkingSetMonitor:
    """Sample process working set on a background thread."""

    def __init__(self, interval_s: float = _SAMPLE_INTERVAL_S) -> None:
        self.interval_s = interval_s
        self.baseline = 0
        self.peak = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> WorkingSetMonitor:
        self.baseline = working_set_bytes()
        self.peak = self.baseline
        self._stop.clear()
        self._thread = threading.Thread(target=self._sample, name="rss-sampler", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._note(working_set_bytes())

    def _sample(self) -> None:
        while not self._stop.wait(self.interval_s):
            self._note(working_set_bytes())

    def _note(self, rss: int) -> None:
        if rss > self.peak:
            self.peak = rss


def loglog_slope(x: list[float], y: list[float]) -> float:
    """OLS slope of ``log(y)`` versus ``log(x)``.

    Returns NaN when fewer than two distinct positive x values are available
    or any y value is non-positive.
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if x_arr.size < 2 or y_arr.size != x_arr.size:
        return float("nan")
    if np.any(x_arr <= 0) or np.any(y_arr <= 0):
        return float("nan")
    log_x = np.log(x_arr)
    log_y = np.log(y_arr)
    if not np.all(np.isfinite(log_x)) or not np.all(np.isfinite(log_y)):
        return float("nan")
    if np.unique(np.round(log_x, decimals=12)).size < 2:
        return float("nan")
    slope, _intercept = np.polyfit(log_x, log_y, 1)
    return float(slope)


def catalog() -> list[ModelSpec]:
    demo_root = _REPO_ROOT / "data" / "demo"
    models = [
        ModelSpec(
            name=name,
            group="demo",
            mesh_path=demo_root / f"{name}.obj",
            skeleton_path=demo_root / f"{name}.polylines.txt",
        )
        for name in _DEMO_NAMES
    ]
    for spine_idx in _SPINE_IDS:
        models.append(
            ModelSpec(
                name=f"TS{spine_idx}",
                group="spine",
                mesh_path=_REPO_ROOT / "data" / "mesh" / "processed" / f"TS{spine_idx}.obj",
                skeleton_path=(
                    _REPO_ROOT
                    / "data"
                    / "mcf_skeletons"
                    / f"TS{spine_idx}_qst0.5_mcst5.polylines.txt"
                ),
            )
        )
    return models


def select_models(spec: str | None) -> list[ModelSpec]:
    models = catalog()
    if spec is None or not spec.strip():
        return models
    by_name = {model.name.lower(): model for model in models}
    selected: list[ModelSpec] = []
    for raw in spec.split(","):
        name = raw.strip()
        if not name:
            continue
        match = by_name.get(name.lower())
        if match is None:
            known = ", ".join(model.name for model in models)
            raise SystemExit(f"Unknown model {name!r}. Known models: {known}")
        selected.append(match)
    if not selected:
        raise SystemExit("No models selected.")
    return selected


def parse_k_values(text: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in text.split(",") if part.strip())
    if len(values) < 1:
        raise SystemExit("--mel-over-thickness requires at least one value.")
    if any(k <= 0 for k in values):
        raise SystemExit("--mel-over-thickness values must be positive.")
    return values


def resample_fractions(mel: float, diagonal: float) -> tuple[float, float]:
    """Active-resample band as fractions of the bbox diagonal, tied to ``mel``."""
    opts = FitOracleOptions()
    min_len = opts.active_resample_min_over_mel * mel
    max_len = opts.active_resample_max_over_mel * mel
    if max_len < 2.0 * min_len:
        max_len = 2.0 * min_len
    min_frac = min_len / diagonal if diagonal > 0 else 0.0
    max_frac = max_len / diagonal if diagonal > 0 else 0.1
    min_frac = float(np.clip(min_frac, 1e-6, 1.0))
    max_frac = float(np.clip(max_frac, min_frac * 2.0, 1.0))
    return min_frac, max_frac


def _cell(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, float) and not np.isfinite(value):
        return ""
    return value


def volume_ratio_distance(row: dict[str, object]) -> float:
    """Absolute distance of the overlap-subtracted volume ratio from 1.

    Falls back to ``volume_ratio`` when the overlap column is absent.
    """
    raw = row.get("volume_ratio_overlaps")
    if raw is None or (isinstance(raw, str) and not str(raw).strip()):
        raw = row.get("volume_ratio")
    if raw is None or (isinstance(raw, str) and not str(raw).strip()):
        return float("inf")
    return abs(float(raw) - 1.0)


def _extend_terminals_key(row: dict[str, object]) -> str:
    raw = row.get("extend_terminals", "")
    if raw is None or str(raw).strip() == "":
        return ""
    text = str(raw).strip().lower()
    if text in {"1", "true", "yes"}:
        return "1"
    if text in {"0", "false", "no"}:
        return "0"
    return text


def _resolution_group_key(row: dict[str, object]) -> tuple[str, float, str]:
    return (
        str(row["model"]),
        float(row["mel_over_thickness"]),
        _extend_terminals_key(row),
    )


def _replicate_rank_key(row: dict[str, object]) -> tuple[float, float]:
    runtime_raw = row.get("runtime_s", "")
    if runtime_raw is None or (isinstance(runtime_raw, str) and not str(runtime_raw).strip()):
        runtime = float("inf")
    else:
        runtime = float(runtime_raw)
    return (volume_ratio_distance(row), runtime)


def select_best_per_resolution(
    rows: list[dict[str, object]],
    *,
    model: str | None = None,
) -> list[dict[str, object]]:
    """One row per (model, mel_over_thickness, terminal extension) nearest volume ratio 1."""
    subset = rows if model is None else [row for row in rows if row["model"] == model]
    if not subset:
        return []
    has_replicates = any("run_index" in row and str(row.get("run_index", "")).strip() != "" for row in subset)
    if not has_replicates:
        return list(subset)

    grouped: dict[tuple[str, float], list[dict[str, object]]] = {}
    for row in subset:
        grouped.setdefault(_resolution_group_key(row), []).append(row)
    return [
        min(group, key=_replicate_rank_key)
        for _key, group in sorted(grouped.items())
    ]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _cell(row.get(key)) for key in _COLUMNS})
        handle.flush()


def apply_scaling(rows: list[dict[str, object]], model_name: str) -> None:
    group = [row for row in rows if row["model"] == model_name]
    buckets: dict[str, list[dict[str, object]]] = {}
    for row in group:
        buckets.setdefault(_extend_terminals_key(row), []).append(row)
    for bucket in buckets.values():
        picked = select_best_per_resolution(bucket, model=model_name)
        nodes = [float(row["morphology_nodes"]) for row in picked]
        runtime_exponent = loglog_slope(nodes, [float(row["runtime_s"]) for row in picked])
        ram_exponent = loglog_slope(nodes, [float(row["peak_rss_mb"]) for row in picked])
        for row in bucket:
            row["runtime_scaling_exponent"] = runtime_exponent
            row["ram_scaling_exponent"] = ram_exponent


def resolve_max_edge_length(
    model: ModelSpec, k: float, thickness_median: float
) -> tuple[float, float]:
    """Return ``(max_edge_length, mel_over_thickness)`` for CSV reporting."""
    if model.name in _DEMO_MAX_EDGE_LENGTH:
        mel = float(_DEMO_MAX_EDGE_LENGTH[model.name])
        k_report = mel / thickness_median if thickness_median > 0 else float(k)
        return mel, k_report
    mel = float(k) * float(thickness_median)
    return mel, float(k)


def effective_n_runs(model: ModelSpec, n_runs: int) -> int:
    """Demo geometries skip stochastic basis optimization and use one replicate."""
    if model.group == "demo":
        return 1
    return n_runs


def result_variants_for_model(
    model: ModelSpec,
    morphology: object,
    extended: object,
    n_extended: int,
) -> tuple[tuple[int, object, int], ...]:
    """Return (extend_flag, graph, n_tips) rows to score for this model."""
    if model.group == "demo" and model.name != "branching":
        return ((0, morphology, 0),)
    return (
        (0, morphology, 0),
        (1, extended, n_extended),
    )


def fit_model(
    model: ModelSpec,
    k_values: tuple[float, ...],
    rows: list[dict[str, object]],
    output_path: Path,
    n_runs: int,
) -> None:
    if not model.mesh_path.is_file():
        raise FileNotFoundError(f"Mesh not found for {model.name}: {model.mesh_path}")
    if not model.skeleton_path.is_file():
        raise FileNotFoundError(
            f"Skeleton not found for {model.name}: {model.skeleton_path}"
        )

    logger.info("Loading %s (%s)", model.name, model.group)
    mesh = MeshManager(mesh_path=str(model.mesh_path))
    skeleton = SkeletonGraph.from_txt(str(model.skeleton_path))
    features = compute_fit_features(mesh, skeleton)
    analysis = mesh.analyze_mesh()
    thickness = features.thickness
    diagonal = float(features.bbox_diagonal)
    if not np.isfinite(thickness.median) or thickness.median <= 0:
        raise RuntimeError(
            f"{model.name} has no usable thickness median ({thickness.median})"
        )

    shared = {
        "model": model.name,
        "group": model.group,
        "mesh_vertices": analysis["vertex_count"],
        "mesh_faces": analysis["face_count"],
        "mesh_area": float(mesh.mesh.area),
        "mesh_volume": analysis["volume"],
        "bbox_diagonal": diagonal,
        "watertight": analysis["is_watertight"],
        "genus": analysis["genus"],
        "component_count": analysis["component_count"],
        "thickness_median": float(thickness.median),
        "thickness_p10": float(thickness.p10),
        "thickness_p90": float(thickness.p90),
        "skeleton_nodes": features.skeleton_nodes,
        "skeleton_edges": features.skeleton_edges,
        "skeleton_length": features.skeleton_length,
        "n_terminals": features.n_terminals,
        "n_branches": features.n_branches,
        "cyclomatic_number": features.cyclomatic_number,
        "runtime_scaling_exponent": "",
        "ram_scaling_exponent": "",
    }

    for k in k_values:
        mel, mel_k_report = resolve_max_edge_length(
            model, k, float(thickness.median)
        )
        if model.name in _DEMO_MAX_EDGE_LENGTH:
            logger.info(
                "%s: demo max_edge_length=%.6g (mel/t=%.4g)",
                model.name,
                mel,
                mel_k_report,
            )
        min_frac, max_frac = resample_fractions(mel, diagonal)
        suggested = suggest_fit_parameters(
            mesh,
            skeleton,
            features=features,
            overrides={
                "max_edge_length": mel,
                "active_resample_min_fraction": min_frac,
                "active_resample_max_fraction": max_frac,
            },
        )
        model_runs = effective_n_runs(model, n_runs)
        basis_opts = (
            None
            if model.group == "demo"
            else suggested.basis_optimizer_options
        )
        if model.group == "demo":
            logger.info(
                "%s: demo geometry — skipping basis optimization, %d replicate",
                model.name,
                model_runs,
            )
        options = FitOptions(
            max_edge_length=suggested.max_edge_length,
            basis_optimizer_options=basis_opts,
        )
        for run_index in range(model_runs):
            logger.info(
                "Fitting %s at mel/t=%.3g run %d/%d (max_edge_length=%.6g)",
                model.name,
                mel_k_report,
                run_index + 1,
                model_runs,
                suggested.max_edge_length,
            )
            with WorkingSetMonitor() as monitor:
                started = time.perf_counter()
                morphology = CableFitter(options).fit(mesh, skeleton)
                runtime_s = time.perf_counter() - started
            rss_before = monitor.baseline / (1024 * 1024)
            peak_rss = monitor.peak / (1024 * 1024)
            extended = copy.deepcopy(morphology)
            n_extended = extended.extend_terminals()
            variants = result_variants_for_model(
                model, morphology, extended, n_extended
            )
            for extend_flag, graph, n_tips in variants:
                validator = Validation(mesh, skeleton, graph)
                volume = validator.compare_volumes(account_for_overlaps=False)
                area = validator.compare_surface_areas(account_for_overlaps=False)
                volume_overlaps = validator.compare_volumes(account_for_overlaps=True)
                area_overlaps = validator.compare_surface_areas(account_for_overlaps=True)
                try:
                    graph.scale_radii_to_match_mesh(
                        mesh,
                        metric="surface_area",
                        account_for_overlaps=False,
                    )
                    volume_sa_norm = Validation(mesh, skeleton, graph).compare_volumes(
                        account_for_overlaps=False
                    )
                    volume_ratio_sa_norm = volume_sa_norm["ratio"]
                    volume_relative_error_sa_norm = volume_sa_norm["relative_error"]
                except ValueError as exc:
                    logger.warning(
                        "%s mel/t=%.3g run %d/%d extend=%d: surface-area normalization skipped: %s",
                        model.name,
                        k,
                        run_index + 1,
                        model_runs,
                        extend_flag,
                        exc,
                    )
                    volume_ratio_sa_norm = float("nan")
                    volume_relative_error_sa_norm = float("nan")
                row = {
                    **shared,
                    "mel_over_thickness": float(mel_k_report),
                    "extend_terminals": extend_flag,
                    "run_index": run_index,
                    "n_runs": model_runs,
                    "max_edge_length": float(suggested.max_edge_length),
                    "max_edge_length_fraction": float(suggested.max_edge_length_fraction),
                    "morphology_nodes": graph.number_of_nodes(),
                    "morphology_edges": graph.number_of_edges(),
                    "volume_ratio": volume["ratio"],
                    "volume_relative_error": volume["relative_error"],
                    "area_ratio": area["ratio"],
                    "area_relative_error": area["relative_error"],
                    "volume_ratio_overlaps": volume_overlaps["ratio"],
                    "volume_relative_error_overlaps": volume_overlaps["relative_error"],
                    "area_ratio_overlaps": area_overlaps["ratio"],
                    "area_relative_error_overlaps": area_overlaps["relative_error"],
                    "volume_ratio_sa_norm": volume_ratio_sa_norm,
                    "volume_relative_error_sa_norm": volume_relative_error_sa_norm,
                    "runtime_s": runtime_s,
                    "rss_before_mb": rss_before,
                    "peak_rss_mb": peak_rss,
                    "rss_delta_mb": peak_rss - rss_before,
                }
                rows.append(row)
                write_csv(output_path, rows)
                logger.info(
                    "%s mel/t=%.3g run %d/%d extend=%d (%d tips): %d nodes, "
                    "vol=%.4f vol_ov=%.4f area=%.4f area_ov=%.4f vol_sa_norm=%.4f, runtime=%.3fs, peak RSS=%.1f MB",
                    model.name,
                    k,
                    run_index + 1,
                    model_runs,
                    extend_flag,
                    n_tips,
                    row["morphology_nodes"],
                    row["volume_ratio"],
                    row["volume_ratio_overlaps"],
                    row["area_ratio"],
                    row["area_ratio_overlaps"],
                    row["volume_ratio_sa_norm"],
                    runtime_s,
                    peak_rss,
                )

    apply_scaling(rows, model.name)
    write_csv(output_path, rows)
    exponent = next(row["runtime_scaling_exponent"] for row in rows if row["model"] == model.name)
    logger.info("%s runtime scaling exponent: %s", model.name, exponent)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep MASCAF resolution and write a complexity CSV.",
    )
    parser.add_argument(
        "--output",
        default=str(_DEFAULT_OUTPUT),
        help="CSV path. Relative paths are resolved from the repo root.",
    )
    parser.add_argument(
        "--models",
        default=None,
        help=(
            "Comma-separated model names (cylinder, torus, branching, TS1, ...). "
            "Default: all demos and toric spines."
        ),
    )
    parser.add_argument(
        "--mel-over-thickness",
        default=",".join(str(k) for k in _DEFAULT_K),
        help="Comma-separated max_edge_length / thickness.median values.",
    )
    parser.add_argument(
        "--replicates",
        type=int,
        default=_DEFAULT_N_RUNS,
        help="Independent fits per model and mel/t (basis optimization is stochastic).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR).",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help=(
            "Also write logs to this path (parent dirs are created). "
            "Relative paths are resolved from the repo root."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_file = None
    if args.log_file:
        log_file = Path(args.log_file)
        if not log_file.is_absolute():
            log_file = _REPO_ROOT / log_file
    configure_logging(
        level=getattr(logging, str(args.log_level).upper()),
        log_file=log_file,
    )
    models = select_models(args.models)
    k_values = parse_k_values(args.mel_over_thickness)
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = _REPO_ROOT / output_path
    n_runs = int(args.replicates)
    if n_runs < 1:
        raise SystemExit("--replicates must be at least 1.")

    rows: list[dict[str, object]] = []
    logger.info(
        "Complexity sweep: %d models, %d replicate(s) each, mel/t=%s, output=%s",
        len(models),
        n_runs,
        ", ".join(f"{k:g}" for k in k_values),
        output_path,
    )
    for model in models:
        fit_model(model, k_values, rows, output_path, n_runs)
    logger.info("Wrote %d rows to %s", len(rows), output_path)


if __name__ == "__main__":
    main()
