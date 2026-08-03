"""Tests for shape-diameter / local thickness."""

from __future__ import annotations

import numpy as np
import pytest

from mascaf import example_mesh
from mascaf.shape_diameter import (
    compute_shape_diameter,
    mesh_thickness_summary,
    summarize_thickness,
)


def test_sdf_on_torus_near_tube_diameter():
    major, minor = 4.0, 1.0
    mesh = example_mesh(
        "torus",
        major_radius=major,
        minor_radius=minor,
        major_sections=48,
        minor_sections=24,
    )
    samples = compute_shape_diameter(
        mesh, n_samples=80, n_rays=10, seed=0
    )
    assert samples.size >= 20
    summary = summarize_thickness(samples)
    # Local diameter should be near 2 * minor_radius
    assert summary.median == pytest.approx(2.0 * minor, rel=0.35)
    assert summary.radius_proxy == pytest.approx(minor, rel=0.35)


def test_sdf_on_cylinder_near_diameter():
    radius = 0.5
    mesh = example_mesh("cylinder", radius=radius, height=4.0, sections=32)
    summary = mesh_thickness_summary(mesh, n_samples=60, n_rays=8, seed=1)
    assert summary.n_samples > 10
    assert summary.median == pytest.approx(2.0 * radius, rel=0.4)


def test_summarize_thickness_empty():
    s = summarize_thickness(np.zeros(0))
    assert s.n_samples == 0
    assert np.isnan(s.median)
