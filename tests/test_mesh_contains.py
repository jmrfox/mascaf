"""Tests for unified mesh containment (signed-distance based)."""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

from mascaf.mesh_contains import (
    default_distance_tol,
    point_inside_mesh,
    points_inside_mesh,
    signed_distances,
)


@pytest.fixture
def unit_sphere() -> trimesh.Trimesh:
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    mesh.process(validate=True)
    return mesh


def test_signed_distance_positive_inside_negative_outside(unit_sphere):
    dists = signed_distances(
        unit_sphere,
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ],
    )
    assert dists.shape == (2,)
    assert dists[0] > 0.5
    assert dists[1] < -0.5


def test_center_inside_far_point_outside(unit_sphere):
    assert point_inside_mesh(unit_sphere, [0.0, 0.0, 0.0])
    assert not point_inside_mesh(unit_sphere, [2.0, 0.0, 0.0])


def test_near_surface_within_tol_counts_as_inside(unit_sphere):
    # Just outside the unit sphere by a tiny amount — within default mesh-scale tol.
    on_shell = np.array([1.0 + 1e-9, 0.0, 0.0])
    assert point_inside_mesh(unit_sphere, on_shell)
    # Clearly exterior.
    assert not point_inside_mesh(unit_sphere, [1.5, 0.0, 0.0])


def test_points_inside_mesh_vectorized(unit_sphere):
    pts = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ]
    )
    mask = points_inside_mesh(unit_sphere, pts)
    assert mask.dtype == bool
    assert mask.shape == (4,)
    np.testing.assert_array_equal(mask, [True, True, False, False])


def test_absolute_tol_overrides_fraction(unit_sphere):
    # Point ~0.01 outside the surface; absolute tol 0.05 keeps it inside.
    near = np.array([1.01, 0.0, 0.0])
    assert point_inside_mesh(unit_sphere, near, tol=0.05)
    assert not point_inside_mesh(unit_sphere, near, tol=1e-6)


def test_default_distance_tol_scales_with_extents(unit_sphere):
    tol = default_distance_tol(unit_sphere, tol_fraction=1e-6)
    assert tol > 0.0
    assert tol == pytest.approx(1e-6 * float(np.linalg.norm(unit_sphere.extents)))
    assert default_distance_tol(unit_sphere, tol=0.01) == 0.01
