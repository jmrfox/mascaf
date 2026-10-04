"""Tests for FitParameterOracle heuristics."""

from __future__ import annotations

import numpy as np
import pytest

from mascaf import (
    FitOracleOptions,
    example_mesh,
    fraction_bounds_around_suggestion,
    suggest_fit_parameters,
)
from mascaf.skeleton import SkeletonGraph


def _torus_skeleton(major: float = 4.0, n: int = 24) -> SkeletonGraph:
    """Polyline around the torus centerline (no radii)."""
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    positions = np.column_stack(
        [
            major * np.cos(angles),
            major * np.sin(angles),
            np.zeros(n),
        ]
    )
    sk = SkeletonGraph()
    for i, p in enumerate(positions):
        sk.add_node(i, xyz=p)
    for i in range(n):
        j = (i + 1) % n
        length = float(np.linalg.norm(positions[j] - positions[i]))
        sk.add_edge(i, j, length=length)
    return sk


def test_oracle_uses_thickness_and_differs_by_bbox_fraction():
    """Same tube radius → similar thickness; bbox fractions still differ."""
    minor = 1.0
    mesh_small = example_mesh(
        "torus",
        major_radius=3.0,
        minor_radius=minor,
        major_sections=40,
        minor_sections=20,
    )
    mesh_large = example_mesh(
        "torus",
        major_radius=8.0,
        minor_radius=minor,
        major_sections=48,
        minor_sections=20,
    )
    sk_small = _torus_skeleton(major=3.0)
    sk_large = _torus_skeleton(major=8.0)
    opts = FitOracleOptions(
        mel_over_thickness=2.0, sdf_n_samples=60, sdf_n_rays=6, sdf_seed=0
    )
    a = suggest_fit_parameters(mesh_small, sk_small, oracle_options=opts)
    b = suggest_fit_parameters(mesh_large, sk_large, oracle_options=opts)
    assert a.features.thickness.median == pytest.approx(2.0 * minor, rel=0.4)
    assert b.features.thickness.median == pytest.approx(2.0 * minor, rel=0.4)
    assert a.max_edge_length == pytest.approx(b.max_edge_length, rel=0.05)
    assert a.mel_over_thickness == pytest.approx(2.0, rel=0.05)
    assert b.mel_over_thickness == pytest.approx(2.0, rel=0.05)
    assert a.max_edge_length_fraction > b.max_edge_length_fraction


def test_oracle_mel_independent_of_skeleton_sampling():
    """Sparse skeleton polylines must not reduce mel below mel/t × thickness."""
    minor = 1.0
    mesh = example_mesh(
        "torus",
        major_radius=3.0,
        minor_radius=minor,
        major_sections=40,
        minor_sections=20,
    )
    sk_sparse = _torus_skeleton(major=3.0, n=8)
    sk_dense = _torus_skeleton(major=3.0, n=48)
    opts = FitOracleOptions(mel_over_thickness=2.0, sdf_n_samples=40, sdf_seed=0)
    sparse = suggest_fit_parameters(mesh, sk_sparse, oracle_options=opts)
    dense = suggest_fit_parameters(mesh, sk_dense, oracle_options=opts)
    t = sparse.features.thickness.median
    assert sparse.max_edge_length == pytest.approx(2.0 * t, rel=0.05)
    assert dense.max_edge_length == pytest.approx(2.0 * t, rel=0.05)
    assert sparse.mel_over_thickness == pytest.approx(2.0, rel=0.05)


def test_oracle_default_basis_n_rays_is_six():
    mesh = example_mesh("torus", major_radius=4.0, minor_radius=1.0)
    sk = _torus_skeleton(major=4.0)
    sug = suggest_fit_parameters(
        mesh, sk, oracle_options=FitOracleOptions(sdf_n_samples=30, sdf_seed=0)
    )
    assert sug.basis_optimizer_options.n_rays == 6


def test_oracle_overrides():
    mesh = example_mesh("torus", major_radius=4.0, minor_radius=1.0)
    sk = _torus_skeleton(major=4.0)
    sug = suggest_fit_parameters(
        mesh,
        sk,
        oracle_options=FitOracleOptions(sdf_n_samples=40, sdf_seed=0),
        overrides={"max_edge_length": 1.5, "n_rays": 24},
    )
    assert sug.max_edge_length == pytest.approx(1.5)
    assert sug.basis_optimizer_options.n_rays == 24


def test_active_resample_min_fraction_does_not_track_mel():
    mesh = example_mesh("torus", major_radius=4.0, minor_radius=1.0)
    sk = _torus_skeleton(major=4.0)
    sug = suggest_fit_parameters(
        mesh, sk, oracle_options=FitOracleOptions(sdf_n_samples=30, sdf_seed=0)
    )
    assert sug.basis_optimizer_options.active_resample is False
    min_frac = sug.basis_optimizer_options.active_resample_min_fraction
    max_frac = sug.basis_optimizer_options.active_resample_max_fraction
    assert min_frac == pytest.approx(1e-6)
    assert max_frac > 100.0 * min_frac


def test_fraction_bounds_around_suggestion():
    mesh = example_mesh("torus", major_radius=4.0, minor_radius=1.0)
    sk = _torus_skeleton(major=4.0)
    sug = suggest_fit_parameters(
        mesh, sk, oracle_options=FitOracleOptions(sdf_n_samples=30, sdf_seed=0)
    )
    lo, hi = fraction_bounds_around_suggestion(sug, rel_span=0.5)
    assert 0 < lo < sug.max_edge_length_fraction < hi
