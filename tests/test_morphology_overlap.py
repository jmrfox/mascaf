"""Tests for branch-point overlap corrections on MorphologyGraph metrics."""

import numpy as np
import pytest

from mascaf import MorphologyGraph
from mascaf.morphology_graph import (
    DEFAULT_OVERLAP_SCALING_AREA,
    DEFAULT_OVERLAP_SCALING_VOLUME,
)


def _y_junction(radius: float = 1.0) -> MorphologyGraph:
    graph = MorphologyGraph()
    graph.add_node(0, xyz=np.array([0.0, 0.0, 0.0]), radius=radius)
    graph.add_node(1, xyz=np.array([0.0, 0.0, 5.0]), radius=radius)
    graph.add_node(2, xyz=np.array([0.0, 0.0, 10.0]), radius=radius)
    graph.add_node(3, xyz=np.array([5.0, 0.0, 5.0]), radius=radius)
    graph.add_edge(0, 1)
    graph.add_edge(1, 2)
    graph.add_edge(1, 3)
    return graph


def test_default_overlap_volume_is_c_v_r_cubed():
    r = 1.5
    graph = _y_junction(radius=r)
    naive = graph.compute_volume(account_for_overlaps=False)
    corrected = graph.compute_volume(account_for_overlaps=True)
    assert naive - corrected == pytest.approx(DEFAULT_OVERLAP_SCALING_VOLUME * r**3)


def test_default_overlap_area_is_c_a_r_squared():
    r = 1.5
    graph = _y_junction(radius=r)
    naive = graph.compute_surface_area(account_for_overlaps=False)
    corrected = graph.compute_surface_area(account_for_overlaps=True)
    assert naive - corrected == pytest.approx(DEFAULT_OVERLAP_SCALING_AREA * r**2)


def test_custom_overlap_scaling_constants():
    r = 2.0
    graph = _y_junction(radius=r)
    c_a = 1.25
    c_v = 0.5
    naive_vol = graph.compute_volume(account_for_overlaps=False)
    vol = graph.compute_volume(
        account_for_overlaps=True, overlap_scaling_volume=c_v
    )
    naive_area = graph.compute_surface_area(account_for_overlaps=False)
    area = graph.compute_surface_area(
        account_for_overlaps=True, overlap_scaling_area=c_a
    )
    assert naive_vol - vol == pytest.approx(c_v * r**3)
    assert naive_area - area == pytest.approx(c_a * r**2)
