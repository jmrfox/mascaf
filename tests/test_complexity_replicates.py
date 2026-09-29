"""Tests for multi-replicate complexity sweep selection."""

from __future__ import annotations

import importlib.util
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


def test_summarize_uses_best_replicate_at_reference_k() -> None:
    rows = [
        _row("demo", 0.5, 0.85, run_index=0),
        _row("demo", 0.5, 0.95, run_index=1),
        _row("demo", 1.0, 0.80, run_index=0, morphology_nodes=20),
        _row("demo", 1.0, 1.01, run_index=1, morphology_nodes=22),
        _row("demo", 1.0, 0.90, run_index=2, morphology_nodes=18),
        _row("demo", 1.5, 0.88, run_index=0),
        _row("demo", 1.5, 0.92, run_index=1),
        _row("demo", 1.5, 0.91, run_index=2),
    ]
    filtered = complexity_analysis.select_best_per_resolution(rows)
    summary = summarize_complexity.summarize(filtered)
    assert len(summary) == 1
    assert summary[0]["model"] == "demo"
    assert summary[0]["morphology_nodes"] == "22"
    assert float(summary[0]["volume_relative_error"]) == pytest.approx(0.01)
