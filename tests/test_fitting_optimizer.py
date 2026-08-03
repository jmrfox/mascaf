"""Tests for FittingOptimizer and prefer-larger selection."""

from pathlib import Path

import pytest

from mascaf import (
    FittingEvalRecord,
    FittingOptimizer,
    FittingOptimizerOptions,
    FitOptions,
    MorphologyGraph,
    SkeletonGraph,
    example_mesh,
    select_prefer_larger,
)


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "mascaf" / "demo"


def _dummy_morph() -> MorphologyGraph:
    return MorphologyGraph()


def _record(fraction: float, err: float) -> FittingEvalRecord:
    return FittingEvalRecord(
        fraction=fraction,
        max_edge_length=fraction,
        volume_relative_error=err,
        scale_factor=1.0,
        morphology=_dummy_morph(),
    )


def test_select_prefer_larger_wide_tol_picks_largest_near_best():
    history = [
        _record(0.05, 0.010),
        _record(0.10, 0.0104),  # within 5% of best (threshold 0.0105)
        _record(0.15, 0.020),  # outside band
    ]
    selected = select_prefer_larger(history, volume_error_rel_tol=0.05)
    assert selected.fraction == pytest.approx(0.10)


def test_select_prefer_larger_zero_tol_picks_true_minimum():
    history = [
        _record(0.05, 0.010),
        _record(0.10, 0.011),
        _record(0.15, 0.010),  # same best error, larger fraction
    ]
    selected = select_prefer_larger(
        history, volume_error_rel_tol=0.0, volume_error_abs_tol=0.0
    )
    assert selected.fraction == pytest.approx(0.15)
    assert selected.volume_relative_error == pytest.approx(0.010)


def test_select_prefer_larger_abs_tol():
    history = [
        _record(0.04, 0.010),
        _record(0.12, 0.012),  # within abs_tol=0.003
        _record(0.18, 0.020),
    ]
    selected = select_prefer_larger(
        history, volume_error_rel_tol=0.0, volume_error_abs_tol=0.003
    )
    assert selected.fraction == pytest.approx(0.12)


def test_select_prefer_larger_empty_raises():
    with pytest.raises(ValueError, match="non-empty"):
        select_prefer_larger([])


def test_fitting_optimizer_cylinder():
    mesh = example_mesh("cylinder")
    skeleton = SkeletonGraph.from_txt(str(DATA / "cylinder.polylines.txt"))
    assert skeleton.number_of_nodes() > 0

    opts = FittingOptimizerOptions(
        fraction_bounds=(0.05, 0.2),
        maxiter=8,
        xatol=1e-2,
        volume_error_rel_tol=0.05,
    )
    result = FittingOptimizer(
        fit_options=FitOptions(radius_strategy="equivalent_area"),
        options=opts,
    ).optimize(mesh, skeleton)

    lo, hi = opts.fraction_bounds
    assert lo <= result.max_edge_length_fraction <= hi
    assert result.n_evals == len(result.history)
    assert result.n_evals >= 1
    assert result.volume_relative_error >= 0.0
    assert result.best_volume_relative_error <= result.volume_relative_error + 1e-15
    assert result.morphology.number_of_nodes() > 0

    mesh_area = float(mesh.area)
    morph_area = result.morphology.compute_surface_area(account_for_overlaps=False)
    assert morph_area == pytest.approx(mesh_area, rel=1e-4)

    # Selected entry matches reported metrics
    selected = select_prefer_larger(
        result.history,
        volume_error_rel_tol=opts.volume_error_rel_tol,
        volume_error_abs_tol=opts.volume_error_abs_tol,
    )
    assert result.max_edge_length_fraction == pytest.approx(selected.fraction)
    assert result.volume_relative_error == pytest.approx(
        selected.volume_relative_error
    )
