"""Collapse a complexity-analysis CSV to one paper-table row per model.

The sweep in ``scripts/complexity_analysis.py`` fits each model at three
``max_edge_length / thickness`` values so scaling can be estimated. When
the input includes multiple stochastic replicates per resolution, this
script keeps the replicate whose overlap-subtracted volume ratio is nearest
1 at each resolution and terminal-extension setting before summarizing.

This script reports size, cost, and
overlap-subtracted volume and surface-area errors, plus the volume error
after scaling radii to the mesh surface area, at the oracle's default
resolution (``mel_over_thickness`` = 2). The summary also records
``max_edge_length`` (MEL) and ``mel_over_thickness`` (mel/t) from that
same reference row.

``radius_relative_error`` and ``radius_relative_error_spread`` infer an
equivalent uniform radius-scale bias from the same overlap-subtracted
volume and area ratios (signed fractions, like ``volume_relative_error``):
``delta_* = V^(1/6) A^(1/4) - 1`` with spread ``|delta_V - delta_A| / 2``
where ``delta_V = V^(1/3) - 1`` and ``delta_A = A^(1/2) - 1``.

Reference models emit a single ``extend_terminals`` setting each; spines emit
both 0 and 1 (summarized as separate rows).

The script also writes runtime scatter plots (one row per model at
``extend_terminals=0`` when present, since extension does not change fit time)
versus skeleton nodes and versus morphology (cable) nodes.

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
_REFERENCE_K = 2.0

# Size columns do not depend on resolution. Cost, morphology size, the cable
# cyclomatic number, and volume/area errors are taken at mel/t = 2.
_COLUMNS = (
    "model",
    "extend_terminals",
    "max_edge_length",
    "mel_over_thickness",
    "mesh_vertices",
    "skeleton_nodes",
    "n_branches",
    "genus",
    "cyclomatic_number",
    "morphology_nodes",
    "runtime_s",
    "peak_rss_mb",
    "volume_relative_error",
    "area_relative_error",
    "radius_relative_error",
    "radius_relative_error_spread",
    "volume_relative_error_sa_norm",
)

_SIZE_FIELDS = (
    "mesh_vertices",
    "skeleton_nodes",
    "n_branches",
    "genus",
)
_REFERENCE_FIELDS = (
    "max_edge_length",
    "mel_over_thickness",
    "morphology_nodes",
    "cyclomatic_number",
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
_VOLUME_RATIO_KEYS = ("volume_ratio_overlaps", "volume_ratio")
_VOLUME_REL_KEYS = (
    "volume_relative_error_overlaps",
    "volume_relative_error",
)
_AREA_RATIO_KEYS = ("area_ratio_overlaps", "area_ratio")
_AREA_REL_KEYS = (
    "area_relative_error_overlaps",
    "area_relative_error",
)


def estimate_radius_relative_error(
    v_ratio: float,
    a_ratio: float,
) -> tuple[float, float]:
    """Infer signed uniform radius-scale error and vol/area disagreement.

    Returns ``(delta_star, spread)`` as fractions (0.05 = 5%). When ratios
    are invalid, returns ``(nan, nan)``.
    """
    if not (
        math.isfinite(v_ratio)
        and math.isfinite(a_ratio)
        and v_ratio > 0.0
        and a_ratio > 0.0
    ):
        return (float("nan"), float("nan"))
    delta_v = v_ratio ** (1.0 / 3.0) - 1.0
    delta_a = a_ratio**0.5 - 1.0
    lambda_star = v_ratio ** (1.0 / 6.0) * a_ratio ** (1.0 / 4.0)
    delta_star = lambda_star - 1.0
    spread = abs(delta_v - delta_a) / 2.0
    return (delta_star, spread)


def _ratio_from_row(
    row: dict[str, str],
    ratio_keys: tuple[str, ...],
    rel_keys: tuple[str, ...],
) -> float:
    for key in ratio_keys:
        raw = row.get(key, "")
        if raw is not None and str(raw).strip() != "":
            value = _float(raw)
            if math.isfinite(value) and value > 0.0:
                return value
    for key in rel_keys:
        raw = row.get(key, "")
        if raw is not None and str(raw).strip() != "":
            rel = _float(raw)
            if math.isfinite(rel):
                ratio = 1.0 + rel
                if ratio > 0.0:
                    return ratio
    return float("nan")


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
        if len(model_rows) < 1:
            raise ValueError(
                f"{name} extend_terminals={extend or 'unset'} has no resolution rows."
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
        v_ratio = _ratio_from_row(reference, _VOLUME_RATIO_KEYS, _VOLUME_REL_KEYS)
        a_ratio = _ratio_from_row(reference, _AREA_RATIO_KEYS, _AREA_REL_KEYS)
        radius_err, radius_spread = estimate_radius_relative_error(v_ratio, a_ratio)
        record["radius_relative_error"] = _format(radius_err)
        record["radius_relative_error_spread"] = _format(radius_spread)
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


def rows_for_runtime_plots(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """One summary row per model; terminal extension does not change fit runtime."""
    by_model: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_model.setdefault(row["model"].strip(), []).append(row)
    picked: list[dict[str, str]] = []
    for model in sorted(by_model):
        group = by_model[model]
        unextended = [
            row
            for row in group
            if str(row.get("extend_terminals", "")).strip() in {"0", "false", "no"}
        ]
        picked.append(unextended[0] if unextended else group[0])
    return picked


def plot_runtime_scatter(
    rows: list[dict[str, str]],
    path: Path,
    *,
    x_field: str,
    xlabel: str,
) -> None:
    """Log-log scatter of fit runtime versus ``x_field``, one point per model."""
    import matplotlib.pyplot as plt

    plot_rows = rows_for_runtime_plots(rows)
    points: list[tuple[float, float, str]] = []
    for row in plot_rows:
        x_value = _float(row.get(x_field, ""))
        runtime_s = _float(row.get("runtime_s", ""))
        if math.isfinite(x_value) and math.isfinite(runtime_s) and x_value > 0 and runtime_s > 0:
            points.append((x_value, runtime_s, row["model"]))
    if not points:
        raise ValueError(f"No finite runtime and {x_field} pairs to plot.")

    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    colors = plt.get_cmap("tab20").colors
    for index, (x_value, runtime_s, name) in enumerate(points):
        axis.scatter(
            [x_value],
            [runtime_s],
            s=120,
            color=colors[index % len(colors)],
            label=name,
            zorder=3,
        )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel(xlabel)
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
    filtered = select_best_per_resolution(raw_rows)
    rows = summarize(filtered)
    write_rows(output_path, rows)
    skeleton_figure = output_path.with_name(f"{output_path.stem}_runtime.png")
    morphology_figure = output_path.with_name(
        f"{output_path.stem}_runtime_morphology.png"
    )
    plot_runtime_scatter(
        rows,
        skeleton_figure,
        x_field="skeleton_nodes",
        xlabel="Skeleton nodes",
    )
    plot_runtime_scatter(
        rows,
        morphology_figure,
        x_field="morphology_nodes",
        xlabel="Morphology nodes",
    )
    print(f"Wrote {len(rows)} model rows to {output_path}")
    print(f"Wrote runtime vs skeleton plot to {skeleton_figure}")
    print(f"Wrote runtime vs morphology plot to {morphology_figure}")


if __name__ == "__main__":
    main()
