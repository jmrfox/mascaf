"""Tests for MorphologyGraph.get_outside_nodes."""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

from mascaf import MeshManager, MorphologyGraph


@pytest.fixture
def unit_sphere() -> trimesh.Trimesh:
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    mesh.process(validate=True)
    return mesh


def test_get_outside_nodes_returns_only_outside_ids(unit_sphere):
    graph = MorphologyGraph()
    graph.add_node(0, xyz=np.array([0.0, 0.0, 0.0]), radius=0.1)
    graph.add_node(1, xyz=np.array([2.0, 0.0, 0.0]), radius=0.1)
    graph.add_edge(0, 1)

    outside = graph.get_outside_nodes(unit_sphere)
    assert outside == [1]


def test_get_outside_nodes_accepts_mesh_manager(unit_sphere):
    graph = MorphologyGraph()
    graph.add_node(0, xyz=np.array([0.0, 0.0, 0.0]), radius=0.1)
    graph.add_node(1, xyz=np.array([3.0, 0.0, 0.0]), radius=0.1)

    mm = MeshManager(unit_sphere)
    outside = graph.get_outside_nodes(mm)
    assert outside == [1]


def test_get_outside_nodes_all_inside(unit_sphere):
    graph = MorphologyGraph()
    graph.add_node(0, xyz=np.array([0.0, 0.0, 0.0]), radius=0.1)
    graph.add_node(1, xyz=np.array([0.2, 0.0, 0.0]), radius=0.1)

    assert graph.get_outside_nodes(unit_sphere) == []


def test_get_outside_nodes_empty_graph(unit_sphere):
    graph = MorphologyGraph()
    assert graph.get_outside_nodes(unit_sphere) == []
