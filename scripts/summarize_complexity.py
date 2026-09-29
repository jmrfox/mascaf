"""Collapse a complexity-analysis CSV to one paper-table row per model.

The sweep in ``scripts/complexity_analysis.py`` fits each model at three
``max_edge_length / thickness`` values so scaling can be estimated. When
the input includes multiple stochastic replicates per resolution, this
script keeps the replicate whose overlap-subtracted volume ratio is nearest
1 at each resolution and terminal-extension setting before summarizing.

This script keeps the scaling estimate and reports size, cost, and
overlap-subtracted volume and surface-area errors, plus the volume error
after scaling radii to the mesh surface area, at the oracle's default
resolution (``mel_over_thickness`` = 1).

The script also writes a runtime-versus-skeleton-nodes scatter beside the
summary CSV.

Example::

    uv run python scripts/summarize_complexity.py
    uv run python scripts/summarize_complexity.py --input outputs/complexity_analysis.csv
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import OrderedDict
from pathlib import Path

from complexity_analysis import select_best_per_resolution

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_INPUT = _REPO_ROOT / "outputs" / "complexity_analysis.csv"
_DEFAULT_OUTPUT = _REPO_ROOT / "outputs" / "complexity_summary.csv"
_REFERENCE_K = 1.0

# Size columns do not depend on resolution. Cost, morphology size, and
# volume/area errors are taken at mel/t = 1 for each terminal-extension
# setting. The exponent is the log-log slope of runtime versus morphology
# node count across resolutions within that setting.
_COLUMNS = (
    "model",
    "extend_terminals",
    "mesh_vertices",
    "skeleton_nodes",
    "n_branches",
    "cyclomatic_number",
    "morphology_nodes",
    "runtime_s",
    "peak_rss_mb",
    "volume_relative_error",
    "area_relative_error",
    "volume_relative_error_sa_norm",
    "runtime_scaling_exponent",
)

_SIZE_FIELDS = (
    "mesh_vertices",
    "skeleton_nodes",
    "n_branches",
    "cyclomatic_number",
)
_REFERENCE_FIELDS = (
    "morphology_nodes",
    "runtime_s",
    "peak_rss_mb",
    "volume_relative_error_sa_norm",
)
# Reported errors are the overlap-subtracted values. Older CSVs that lack
# those columns fall back to the uncorrected errors.
_ERROR_FIELDS = {
    "volume_relative_error": (
        "volume_relative_error_overlaps",
        "volume_relative_error",
    ),
    "area_relative_error": (
        "area_relative_error_overlaps",
        "area_relative_error",
    ),
}
_SCALING_FIELDS = ("runtime_scaling_exponent",)


def _float(text: str) -> float:
    if text is None or str(text).strip() == "":
        return float("nan")
    return float(text)


def _format(value: float) -> str:
    if not math.isfinite(value):
        return ""
    rounded = round(value)
    if math.isclose(value, rounded, rel_tol=0.0, abs_tol=1e-9):
        return str(int(rounded))
    return f"{value:.6g}"


def _reference_row(rows: list[dict[str, str]]) -> dict[str, str]:
    def distance(row: dict[str, str]) -> float:
        k = _float(row.get("mel_over_thickness", ""))
        if not math.isfinite(k):
            return float("inf")
        return abs(k - _REFERENCE_K)

    return min(rows, key=distance)


def _consistent(rows: list[dict[str, str]], field: str) -> float:
    values = [_float(row.get(field, "")) for row in rows]
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return float("nan")
    first = finite[0]
    drifted = any(
        not math.isclose(value, first, rel_tol=1e-6, abs_tol=1e-9)
        for value in finite[1:]
    )
    if drifted:
        model = rows[0].get("model", "")
        raise ValueError(
            f"{model}: {field} differs across resolutions "
            f"({', '.join(str(value) for value in finite)})"
        )
    return first


def _summary_key(row: dict[str, str]) -> tuple[str, str]:
    name = row["model"].strip()
    extend = str(row.get("extend_terminals", "")).strip()
    return (name, extend)


def summarize(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: OrderedDict[tuple[str, str], list[dict[str, str]]] = OrderedDict()
    for row in rows:
        name = row["model"].strip()
        if not name:
            continue
        grouped.setdefault(_summary_key(row), []).append(row)

    summary: list[dict[str, str]] = []
    for (name, extend), model_rows in grouped.items():
        if len(model_rows) < 2:
            raise ValueError(
                f"{name} extend_terminals={extend or 'unset'} has "
                f"{len(model_rows)} resolution row(s); scaling needs at least two."
            )
        reference = _reference_row(model_rows)
        record = {"model": name, "extend_terminals": extend}
        for field in _SIZE_FIELDS:
            record[field] = _format(_consistent(model_rows, field))
        for field in _REFERENCE_FIELDS:
            record[field] = _format(_float(reference.get(field, "")))
        for field, sources in _ERROR_FIELDS.items():
            raw = ""
            for source in sources:
                candidate = reference.get(source, "")
                if candidate is not None and str(candidate).strip() != "":
                    raw = candidate
                    break
            record[field] = _format(_float(raw))
        for field in _SCALING_FIELDS:
            record[field] = _format(_consistent(model_rows, field))
        summary.append(record)
    return summary


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def plot_runtime(rows: list[dict[str, str]], path: Path) -> None:
    """Scatter runtime against skeleton node count, one point per model."""
    import matplotlib.pyplot as plt

    points: list[tuple[float, float, str]] = []
    for row in rows:
        skeleton_nodes = _float(row.get("skeleton_nodes", ""))
        runtime_s = _float(row.get("runtime_s", ""))
        if math.isfinite(skeleton_nodes) and math.isfinite(runtime_s) and skeleton_nodes > 0 and runtime_s > 0:
            points.append((skeleton_nodes, runtime_s, row["model"]))
    if not points:
        raise ValueError("No finite runtime and skeleton-node pairs to plot.")

    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    colors = plt.get_cmap("tab20").colors
    for index, (skeleton_nodes, runtime_s, name) in enumerate(points):
        axis.scatter(
            [skeleton_nodes],
            [runtime_s],
            s=120,
            color=colors[index % len(colors)],
            label=name,
            zorder=3,
        )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Skeleton nodes")
    axis.set_ylabel("Runtime (s)")
    axis.grid(True, which="both", linestyle=":", linewidth=0.6, alpha=0.7)
    axis.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=8,
    )
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize complexity_analysis.csv to one row per model.",
    )
    parser.add_argument(
        "--input",
        default=str(_DEFAULT_INPUT),
        help="Complexity sweep CSV. Relative paths are resolved from the repo root.",
    )
    parser.add_argument(
        "--output",
        default=str(_DEFAULT_OUTPUT),
        help="Summary CSV. Relative paths are resolved from the repo root.",
    )
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    if path.is_absolute():
        return path
    return _REPO_ROOT / path


def main() -> None:
    args = parse_args()
    input_path = _resolve(Path(args.input))
    output_path = _resolve(Path(args.output))
    if not input_path.is_file():
        raise SystemExit(f"Input CSV not found: {input_path}")
    raw_rows = read_rows(input_path)
    rows = summarize(select_best_per_resolution(raw_rows))
    write_rows(output_path, rows)
    figure_path = output_path.with_name(f"{output_path.stem}_runtime.png")
    plot_runtime(rows, figure_path)
    print(f"Wrote {len(rows)} model rows to {output_path}")
    print(f"Wrote runtime plot to {figure_path}")


if __name__ == "__main__":
    main()
