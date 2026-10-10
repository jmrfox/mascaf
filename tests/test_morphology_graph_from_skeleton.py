"""Tests for MorphologyGraph skeleton import and resampling."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from mascaf import MorphologyGraph, SkeletonGraph


def test_from_skeleton_graph_exact_copy():
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    skeleton = SkeletonGraph.from_polylines([points])
    morph = MorphologyGraph.from_skeleton_graph(skeleton)

    assert morph.number_of_nodes() == skeleton.number_of_nodes()
    assert morph.number_of_edges() == skeleton.number_of_edges()
    assert set(morph.nodes()) == set(int(n) for n in skeleton.nodes())
    for node in skeleton.nodes():
        np.testing.assert_allclose(
            morph.get_node_position(int(node)),
            skeleton.get_node_position(node),
        )


def test_from_skeleton_graph_resample_preserves_endpoints_and_edge_bound():
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    skeleton = SkeletonGraph.from_polylines([points])
    max_edge_length = 2.0

    morph = MorphologyGraph.from_skeleton_graph_resample(skeleton, max_edge_length)

    assert morph.number_of_nodes() >= 2
    assert morph.number_of_edges() >= 1

    terminals = morph.get_terminal_nodes()
    assert len(terminals) == 2
    terminal_positions = sorted(
        [tuple(morph.get_node_position(n)) for n in terminals]
    )
    assert terminal_positions[0] == pytest.approx((0.0, 0.0, 0.0))
    assert terminal_positions[1] == pytest.approx((10.0, 0.0, 0.0))

    for u, v in morph.edges():
        length = float(
            np.linalg.norm(morph.get_node_position(v) - morph.get_node_position(u))
        )
        assert length <= max_edge_length + 1e-9

    assert morph.number_of_nodes() == 6
    assert morph.number_of_edges() == 5


def test_resample_instance_method():
    points = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=float)
    skeleton = SkeletonGraph.from_polylines([points])
    exact = MorphologyGraph.from_skeleton_graph(skeleton)
    resampled = exact.resample(2.0)
    assert exact.number_of_nodes() == 2
    assert resampled.number_of_nodes() == 6


def _two_routes_between_junctions() -> MorphologyGraph:
    """Two short paths between the same junctions, plus a stub at each end."""
    graph = MorphologyGraph()
    positions = {
        0: [0.0, 0.0, 0.0],
        1: [0.5, 0.2, 0.0],
        2: [1.0, 0.0, 0.0],
        3: [0.5, -0.2, 0.0],
        4: [-1.0, 0.0, 0.0],
        5: [2.0, 0.0, 0.0],
    }
    for node, xyz in positions.items():
        graph.add_node(node, xyz=np.array(xyz, dtype=float))
    for u, v in [(0, 1), (1, 2), (0, 3), (3, 2), (0, 4), (2, 5)]:
        graph.add_edge(u, v)
    return graph


def test_resample_warns_when_a_short_cycle_collapses(caplog: pytest.LogCaptureFixture):
    graph = _two_routes_between_junctions()
    assert graph.cyclomatic_number() == 1

    with caplog.at_level(logging.WARNING):
        kept = graph.resample(0.4)
    assert kept.cyclomatic_number() == 1
    assert not any("dropped" in record.message for record in caplog.records)

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        collapsed = graph.resample(2.0)
    assert collapsed.cyclomatic_number() == 0
    assert any(
        "dropped 1 cycle(s)" in record.message and "1 → 0" in record.message
        for record in caplog.records
    )


def test_from_skeleton_graph_empty_skeleton():
    skeleton = SkeletonGraph()
    morph = MorphologyGraph.from_skeleton_graph(skeleton)
    assert morph.number_of_nodes() == 0
    assert morph.number_of_edges() == 0
    assert MorphologyGraph.from_skeleton_graph_resample(
        skeleton, max_edge_length=1.0
    ).number_of_nodes() == 0
