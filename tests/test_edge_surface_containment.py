"""Tests for centerline edges that leave a mesh volume."""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

from mascaf import MorphologyGraph, SkeletonGraph, Validation


@pytest.fixture
def unit_sphere() -> trimesh.Trimesh:
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    mesh.process(validate=True)
    return mesh


def _edge_sets(result: dict) -> tuple[set[frozenset], set[frozenset]]:
    crossing = {frozenset(edge) for edge in result["crossing"]}
    fully_outside = {frozenset(edge) for edge in result["fully_outside"]}
    return crossing, fully_outside


def _sphere_graph() -> MorphologyGraph:
    graph = MorphologyGraph()
    points = {
        0: [0.0, 0.0, 0.0],
        1: [0.2, 0.0, 0.0],
        2: [2.0, 0.0, 0.0],
        3: [2.5, 0.0, 0.0],
        4: [3.0, 0.0, 0.0],
        5: [-2.0, 0.0, 0.0],
        6: [2.0, 0.0, 0.0],
    }
    for node, xyz in points.items():
        graph.add_node(node, xyz=np.array(xyz, dtype=float), radius=0.1)
    graph.add_edge(0, 1)
    graph.add_edge(0, 2)
    graph.add_edge(3, 4)
    graph.add_edge(5, 6)
    return graph


def test_classify_edges_against_mesh(unit_sphere):
    result = _sphere_graph().classify_edges_against_mesh(unit_sphere)
    crossing, fully_outside = _edge_sets(result)

    assert crossing == {frozenset((0, 2)), frozenset((5, 6))}
    assert fully_outside == {frozenset((3, 4))}


def test_compare_containment_counts(unit_sphere):
    graph = _sphere_graph()
    validator = Validation(unit_sphere, SkeletonGraph(), graph)
    counts = validator.compare_containment()

    assert counts == {
        "n_outside_nodes": 5,
        "n_crossing_edges": 2,
        "n_fully_outside_edges": 1,
    }
    assert validator.full_validation() == counts
