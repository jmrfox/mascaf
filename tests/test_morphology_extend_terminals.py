"""Tests for MorphologyGraph.extend_terminals."""

import numpy as np
import pytest

from mascaf import Junction, MorphologyGraph


def _two_node_cable() -> MorphologyGraph:
    graph = MorphologyGraph()
    graph.add_junction(Junction(id=1, xyz=np.array([0.0, 0.0, 0.0]), radius=2.0))
    graph.add_junction(Junction(id=2, xyz=np.array([4.0, 0.0, 0.0]), radius=1.0))
    graph.add_edge(1, 2)
    return graph


def test_extend_terminals_both_ends():
    graph = _two_node_cable()

    extended = graph.extend_terminals()

    assert extended == 2
    assert graph.degree(1) == 2
    assert graph.degree(2) == 2
    assert graph.number_of_nodes() == 4

    new_ids = [node for node in graph.nodes() if node not in (1, 2)]
    positions = {
        node: np.asarray(graph.nodes[node]["xyz"], dtype=float) for node in new_ids
    }
    radii = {node: graph.nodes[node]["radius"] for node in new_ids}
    by_x = sorted(positions, key=lambda node: positions[node][0])
    left, right = by_x

    # Node 1 (r=2) is the parent of the left tip; node 2 (r=1) parents the right.
    np.testing.assert_allclose(positions[left], [-2.0, 0.0, 0.0])
    assert radii[left] == pytest.approx(1.0)
    np.testing.assert_allclose(positions[right], [5.0, 0.0, 0.0])
    assert radii[right] == pytest.approx(0.5)
    assert graph.has_edge(1, left)
    assert graph.has_edge(2, right)


def test_extend_terminals_noop_without_degree_one():
    graph = MorphologyGraph()
    for node_id, x in ((1, 0.0), (2, 1.0), (3, 0.5)):
        graph.add_junction(
            Junction(id=node_id, xyz=np.array([x, 0.0, 0.0]), radius=1.0)
        )
    graph.add_edge(1, 2)
    graph.add_edge(2, 3)
    graph.add_edge(3, 1)

    assert graph.extend_terminals() == 0
    assert graph.number_of_nodes() == 3
    assert graph.number_of_edges() == 3
