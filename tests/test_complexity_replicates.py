"""Tests for multi-replicate complexity sweep selection."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def _load_script_module(name: str):
    path = _SCRIPTS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    if str(_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(_SCRIPTS))
    spec.loader.exec_module(module)
    return module


complexity_analysis = _load_script_module("complexity_analysis")
summarize_complexity = _load_script_module("summarize_complexity")


def _row(
    model: str,
    k: float,
    ratio: float,
    run_index: int,
    runtime_s: float = 1.0,
    **extra: object,
) -> dict[str, object]:
    return {
        "model": model,
        "mel_over_thickness": k,
        "run_index": run_index,
        "n_runs": 3,
        "volume_ratio": ratio,
        "volume_relative_error": ratio - 1.0,
        "runtime_s": runtime_s,
        "morphology_nodes": 10,
        "peak_rss_mb": 100.0,
        "mesh_vertices": 100,
        "skeleton_nodes": 5,
        "n_branches": 0,
        "cyclomatic_number": 0,
        "runtime_scaling_exponent": 1.0,
        **extra,
    }


def test_resolve_max_edge_length_uses_notebook_values_for_demos() -> None:
    demo = complexity_analysis.ModelSpec(
        name="torus",
        group="demo",
        mesh_path=Path("x.obj"),
        skeleton_path=Path("x.txt"),
    )
    mel, k_report = complexity_analysis.resolve_max_edge_length(demo, 2.0, 2.0)
    assert mel == pytest.approx(1.0)
    assert k_report == pytest.approx(0.5)


def test_select_best_extend_per_demo() -> None:
    rows = [
        {
            "model": "branching",
            "group": "demo",
            "mel_over_thickness": "1",
            "run_index": "0",
            "extend_terminals": "0",
            "volume_relative_error_overlaps": "-0.13",
        },
        {
            "model": "branching",
            "group": "demo",
            "mel_over_thickness": "1",
            "run_index": "0",
            "extend_terminals": "1",
            "volume_relative_error_overlaps": "-0.065",
        },
        {
            "model": "TS1",
            "group": "spine",
            "mel_over_thickness": "2",
            "run_index": "0",
            "extend_terminals": "0",
            "volume_relative_error_overlaps": "0.1",
        },
        {
            "model": "TS1",
            "group": "spine",
            "mel_over_thickness": "2",
            "run_index": "0",
            "extend_terminals": "1",
            "volume_relative_error_overlaps": "0.2",
        },
    ]
    picked = summarize_complexity.select_best_extend_per_demo(rows)
    branching = [r for r in picked if r["model"] == "branching"]
    ts1 = [r for r in picked if r["model"] == "TS1"]
    assert len(branching) == 1
    assert branching[0]["extend_terminals"] == "1"
    assert len(ts1) == 2


def test_result_variants_demo_branching_only_extended_row() -> None:
    def spec(name: str, group: str) -> complexity_analysis.ModelSpec:
        return complexity_analysis.ModelSpec(
            name=name,
            group=group,
            mesh_path=Path("x.obj"),
            skeleton_path=Path("x.txt"),
        )

    morph, extended = object(), object()
    cylinder = spec("cylinder", "demo")
    branching = spec("branching", "demo")
    spine = spec("TS1", "spine")
    assert complexity_analysis.result_variants_for_model(
        cylinder, morph, extended, 2
    ) == ((0, morph, 0),)
    assert len(
        complexity_analysis.result_variants_for_model(branching, morph, extended, 2)
    ) == 2
    assert len(
        complexity_analysis.result_variants_for_model(spine, morph, extended, 2)
    ) == 2


def test_effective_n_runs_demo_single_replicate() -> None:
    demo = complexity_analysis.ModelSpec(
        name="cylinder",
        group="demo",
        mesh_path=Path("x.obj"),
        skeleton_path=Path("x.txt"),
    )
    spine = complexity_analysis.ModelSpec(
        name="TS1",
        group="spine",
        mesh_path=Path("x.obj"),
        skeleton_path=Path("x.txt"),
    )
    assert complexity_analysis.effective_n_runs(demo, 10) == 1
    assert complexity_analysis.effective_n_runs(spine, 10) == 10


def test_volume_ratio_distance() -> None:
    assert complexity_analysis.volume_ratio_distance({"volume_ratio": 1.02}) == pytest.approx(0.02)
    assert complexity_analysis.volume_ratio_distance({"volume_ratio": 0.9}) == pytest.approx(0.1)
    assert complexity_analysis.volume_ratio_distance({}) == float("inf")
    assert complexity_analysis.volume_ratio_distance(
        {"volume_ratio": 1.2, "volume_ratio_overlaps": 0.99}
    ) == pytest.approx(0.01)


def test_select_best_per_resolution_picks_nearest_ratio() -> None:
    rows = [
        _row("demo", 2.0, 0.9, run_index=0),
        _row("demo", 2.0, 1.02, run_index=1),
        _row("demo", 2.0, 0.98, run_index=2),
    ]
    picked = complexity_analysis.select_best_per_resolution(rows)
    assert len(picked) == 1
    assert picked[0]["run_index"] == 1
    assert picked[0]["volume_ratio"] == pytest.approx(1.02)


def test_select_best_per_resolution_runtime_tiebreak() -> None:
    rows = [
        _row("demo", 2.0, 1.01, run_index=0, runtime_s=2.0),
        _row("demo", 2.0, 0.99, run_index=1, runtime_s=0.5),
    ]
    picked = complexity_analysis.select_best_per_resolution(rows)
    assert len(picked) == 1
    assert picked[0]["run_index"] == 1


def test_select_best_per_resolution_keeps_terminal_extension_axis() -> None:
    rows = [
        _row("demo", 2.0, 1.2, run_index=0, extend_terminals=0),
        _row("demo", 2.0, 0.95, run_index=1, extend_terminals=0),
        _row("demo", 2.0, 1.4, run_index=0, extend_terminals=1),
        _row("demo", 2.0, 1.05, run_index=1, extend_terminals=1),
    ]
    picked = complexity_analysis.select_best_per_resolution(rows)
    assert len(picked) == 2
    by_extend = {row["extend_terminals"]: row for row in picked}
    assert by_extend[0]["volume_ratio"] == pytest.approx(0.95)
    assert by_extend[1]["volume_ratio"] == pytest.approx(1.05)


def test_select_best_per_resolution_legacy_rows_without_run_index() -> None:
    rows = [
        {"model": "demo", "mel_over_thickness": 1.0, "volume_ratio": 1.0},
        {"model": "demo", "mel_over_thickness": 2.0, "volume_ratio": 1.0},
    ]
    picked = complexity_analysis.select_best_per_resolution(rows)
    assert len(picked) == 2


def test_estimate_radius_relative_error_consistent_scale() -> None:
    lam = 1.1
    v_ratio = lam**3
    a_ratio = lam**2
    delta_star, spread = summarize_complexity.estimate_radius_relative_error(
        v_ratio, a_ratio
    )
    assert delta_star == pytest.approx(lam - 1.0)
    assert spread == pytest.approx(0.0)


def test_estimate_radius_relative_error_inconsistent_ratios() -> None:
    v_ratio = 1.2
    a_ratio = 1.1
    delta_star, spread = summarize_complexity.estimate_radius_relative_error(
        v_ratio, a_ratio
    )
    delta_v = v_ratio ** (1.0 / 3.0) - 1.0
    delta_a = a_ratio**0.5 - 1.0
    assert delta_star == pytest.approx(v_ratio ** (1.0 / 6.0) * a_ratio ** (1.0 / 4.0) - 1.0)
    assert spread == pytest.approx(abs(delta_v - delta_a) / 2.0)


def test_estimate_radius_relative_error_invalid_ratios() -> None:
    delta_star, spread = summarize_complexity.estimate_radius_relative_error(-1.0, 1.0)
    assert math.isnan(delta_star)
    assert math.isnan(spread)


def test_summarize_radius_columns_from_overlap_ratios() -> None:
    lam = 1.05
    v_ratio = lam**3
    a_ratio = lam**2
    rows = [
        _row(
            "demo",
            1.0,
            v_ratio,
            run_index=0,
            extend_terminals=0,
            area_ratio=a_ratio,
            area_relative_error=a_ratio - 1.0,
            volume_ratio_overlaps=v_ratio,
            volume_relative_error_overlaps=v_ratio - 1.0,
            area_ratio_overlaps=a_ratio,
            area_relative_error_overlaps=a_ratio - 1.0,
        ),
        _row(
            "demo",
            2.0,
            v_ratio,
            run_index=0,
            extend_terminals=0,
            area_ratio=a_ratio,
            area_relative_error=a_ratio - 1.0,
            volume_ratio_overlaps=v_ratio,
            volume_relative_error_overlaps=v_ratio - 1.0,
            area_ratio_overlaps=a_ratio,
            area_relative_error_overlaps=a_ratio - 1.0,
        ),
    ]
    summary = summarize_complexity.summarize(rows)
    assert float(summary[0]["radius_relative_error"]) == pytest.approx(lam - 1.0)
    assert float(summary[0]["radius_relative_error_spread"]) == pytest.approx(0.0)


def test_summarize_radius_from_legacy_relative_errors_only() -> None:
    rows = [
        _row(
            "demo",
            1.0,
            1.0,
            run_index=0,
            volume_relative_error=0.0,
            area_relative_error=0.0,
        ),
        _row(
            "demo",
            2.0,
            1.331,
            run_index=0,
            volume_relative_error=0.331,
            area_relative_error=0.21,
        ),
    ]
    summary = summarize_complexity.summarize(rows)
    assert float(summary[0]["radius_relative_error"]) == pytest.approx(0.1, rel=1e-3)
    assert float(summary[0]["radius_relative_error_spread"]) == pytest.approx(0.0, abs=1e-9)


def test_summarize_uses_best_replicate_at_reference_k() -> None:
    rows = [
        _row("demo", 0.5, 0.85, run_index=0),
        _row("demo", 0.5, 0.95, run_index=1),
        _row("demo", 2.0, 0.80, run_index=0, morphology_nodes=20),
        _row("demo", 2.0, 1.01, run_index=1, morphology_nodes=22),
        _row("demo", 2.0, 0.90, run_index=2, morphology_nodes=18),
        _row("demo", 3.0, 0.88, run_index=0),
        _row("demo", 3.0, 0.92, run_index=1),
        _row("demo", 3.0, 0.91, run_index=2),
    ]
    filtered = complexity_analysis.select_best_per_resolution(rows)
    summary = summarize_complexity.summarize(filtered)
    assert len(summary) == 1
    assert summary[0]["model"] == "demo"
    assert summary[0]["morphology_nodes"] == "22"
    assert float(summary[0]["volume_relative_error"]) == pytest.approx(0.01)
