"""Sweep MASCAF resolution and write a complexity table as CSV.

Fits cylinder, torus, branching, and the toric spines at three
thickness-relative ``max_edge_length`` values. Each row records mesh and
skeleton size, fit fidelity, wall time, and process working-set RAM.
After the three fits for a model, both rows of that model get a log-log
scaling exponent of runtime and peak RAM versus morphology node count.

Example::

    uv run python scripts/complexity_analysis.py
    uv run python scripts/complexity_analysis.py --models cylinder
    uv run python scripts/complexity_analysis.py --models cylinder,TS1 --mel-over-thickness 1,2,3.5
"""

from __future__ import annotations

import argparse
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
_SPINE_IDS = (1, 2, 3, 4, 21, 24, 48, 67, 76)
_DEFAULT_K = (1.0, 2.0, 3.5)
_SAMPLE_INTERVAL_S = 0.05

_COLUMNS = (
    "model",
    "group",
    "mel_over_thickness",
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
    if len(values) < 2:
        raise SystemExit("--mel-over-thickness needs at least two values to estimate scaling.")
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
    min_frac = min_len / diagonal if diagonal > 0 else 0.05
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
    nodes = [float(row["morphology_nodes"]) for row in group]
    runtime_exponent = loglog_slope(nodes, [float(row["runtime_s"]) for row in group])
    ram_exponent = loglog_slope(nodes, [float(row["peak_rss_mb"]) for row in group])
    for row in group:
        row["runtime_scaling_exponent"] = runtime_exponent
        row["ram_scaling_exponent"] = ram_exponent


def fit_model(
    model: ModelSpec,
    k_values: tuple[float, ...],
    rows: list[dict[str, object]],
    output_path: Path,
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
        mel = float(k) * float(thickness.median)
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
        options = FitOptions(
            max_edge_length=suggested.max_edge_length,
            basis_optimizer_options=suggested.basis_optimizer_options,
        )
        logger.info(
            "Fitting %s at mel/t=%.3g (max_edge_length=%.6g)",
            model.name,
            k,
            suggested.max_edge_length,
        )
        with WorkingSetMonitor() as monitor:
            started = time.perf_counter()
            morphology = CableFitter(options).fit(mesh, skeleton)
            runtime_s = time.perf_counter() - started
        rss_before = monitor.baseline / (1024 * 1024)
        peak_rss = monitor.peak / (1024 * 1024)
        has_branches = any(morphology.degree[node] > 2 for node in morphology.nodes())
        validator = Validation(mesh, skeleton, morphology)
        volume = validator.compare_volumes(account_for_overlaps=has_branches)
        area = validator.compare_surface_areas(account_for_overlaps=has_branches)
        row = {
            **shared,
            "mel_over_thickness": float(k),
            "max_edge_length": float(suggested.max_edge_length),
            "max_edge_length_fraction": float(suggested.max_edge_length_fraction),
            "morphology_nodes": morphology.number_of_nodes(),
            "morphology_edges": morphology.number_of_edges(),
            "volume_ratio": volume["ratio"],
            "volume_relative_error": volume["relative_error"],
            "area_ratio": area["ratio"],
            "area_relative_error": area["relative_error"],
            "runtime_s": runtime_s,
            "rss_before_mb": rss_before,
            "peak_rss_mb": peak_rss,
            "rss_delta_mb": peak_rss - rss_before,
        }
        rows.append(row)
        write_csv(output_path, rows)
        logger.info(
            "%s mel/t=%.3g: %d nodes, runtime=%.3fs, peak RSS=%.1f MB",
            model.name,
            k,
            row["morphology_nodes"],
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
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(level=getattr(logging, str(args.log_level).upper()))
    models = select_models(args.models)
    k_values = parse_k_values(args.mel_over_thickness)
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = _REPO_ROOT / output_path

    rows: list[dict[str, object]] = []
    logger.info(
        "Complexity sweep: %d models, mel/t=%s, output=%s",
        len(models),
        ", ".join(f"{k:g}" for k in k_values),
        output_path,
    )
    for model in models:
        fit_model(model, k_values, rows, output_path)
    logger.info("Wrote %d rows to %s", len(rows), output_path)


if __name__ == "__main__":
    main()
