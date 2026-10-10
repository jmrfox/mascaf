"""Tests for complexity sweep protocol helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "scripts"))

import complexity_analysis as ca
from mascaf import BasisOptimizerOptions
from mascaf.fit_oracle import FitFeatures, SuggestedFitParameters
from mascaf.shape_diameter import ThicknessSummary


def _suggested_with_basis(**kwargs: object) -> SuggestedFitParameters:
    basis = BasisOptimizerOptions(**kwargs)
    thickness = ThicknessSummary(
        n_samples=10,
        median=1.0,
        mean=1.0,
        p10=0.5,
        p90=1.5,
        cv=0.1,
    )
    features = FitFeatures(
        bbox_diagonal=10.0,
        thickness=thickness,
        skeleton_length=5.0,
        skeleton_nodes=4,
        skeleton_edges=3,
        n_terminals=2,
        n_branches=0,
        cyclomatic_number=0,
        watertight=True,
    )
    return SuggestedFitParameters(
        max_edge_length=1.0,
        max_edge_length_fraction=0.01,
        basis_optimizer_options=basis,
        features=features,
        mel_over_thickness=1.0,
        rationale=(),
    )


def test_configure_basis_options_spine_forcing_20() -> None:
    suggested = _suggested_with_basis(do_forcing=False, max_iterations=10)
    opts = ca.configure_basis_options(suggested, "spine")
    assert opts.do_snapping is True
    assert opts.do_forcing is True
    assert opts.max_iterations == ca._SPINE_FORCING_ITERATIONS
    assert opts.forcing_run_all_iterations is True


def test_configure_basis_options_reference_snap_only() -> None:
    suggested = _suggested_with_basis(do_forcing=True, max_iterations=99)
    opts = ca.configure_basis_options(suggested, "reference")
    assert opts.do_snapping is True
    assert opts.do_forcing is False
    assert opts.forcing_run_all_iterations is False


def test_k_values_for_model() -> None:
    spine = next(m for m in ca.catalog() if m.name == "TS1")
    assert ca.k_values_for_model(spine, (1.0, 2.0, 3.0)) == (1.0, 2.0, 3.0)
    human = next(m for m in ca.catalog() if m.name == "human")
    assert ca.k_values_for_model(human, (1.0, 2.0, 3.0)) == (2.0,)
    cylinder = next(m for m in ca.catalog() if m.name == "cylinder")
    assert ca.k_values_for_model(cylinder, (1.0, 2.0, 3.0)) == (1.0,)


def test_result_variants_respects_extend_flags() -> None:
    model = next(m for m in ca.catalog() if m.name == "cylinder")
    morph = object()
    extended = object()
    variants = ca.result_variants_for_model(model, morph, extended, 3)
    assert variants == ((0, morph, 0),)
    spine = next(m for m in ca.catalog() if m.name == "TS1")
    variants = ca.result_variants_for_model(spine, morph, extended, 5)
    assert variants == ((0, morph, 0), (1, extended, 5))


def test_effective_n_runs() -> None:
    spine = next(m for m in ca.catalog() if m.name == "TS1")
    human = next(m for m in ca.catalog() if m.name == "human")
    assert ca.effective_n_runs(spine, 3) == 3
    assert ca.effective_n_runs(human, 3) == 1
