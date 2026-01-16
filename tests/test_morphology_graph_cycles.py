"""Test MorphologyGraph cycle-breaking functionality."""

from pathlib import Path
import tempfile

import numpy as np

from mcf2swc import MorphologyGraph, Junction, SWCModel


def test_morphology_graph_cycle_breaking():
    """Test that MorphologyGraph correctly breaks cycles when exporting to SWC."""
    # Create a simple graph with a cycle: 0 -> 1 -> 2 -> 3 -> 0
    graph = MorphologyGraph()

    # Add nodes
    for i in range(4):
        graph.add_node(
            i,
            xyz=np.array([float(i), 0.0, 0.0]),
            radius=1.0,
        )

    # Add edges to form a cycle
    graph.add_edge(0, 1)
    graph.add_edge(1, 2)
    graph.add_edge(2, 3)
    graph.add_edge(3, 0)  # This creates the cycle

    # Verify we have a cycle
    assert graph.number_of_nodes() == 4
    assert graph.number_of_edges() == 4

    # Verify cycle is detected
    import networkx as nx

    cycle_basis = nx.cycle_basis(graph)
    assert len(cycle_basis) == 1, f"Expected 1 cycle, got {len(cycle_basis)}"

    # Export to SWC (should break the cycle)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".swc", delete=False) as f:
        temp_path = f.name

    try:
        swc_text = graph.to_swc_file(temp_path)

        # Check that the file was written
        assert Path(temp_path).exists()

        # Check for cycle break annotation in the output
        assert "CYCLE_BREAK" in swc_text

        # Reload as SWCModel to verify it's a valid tree
        model = SWCModel.from_swc_file(temp_path)

        # Should have more nodes than original (due to duplication)
        assert model.number_of_nodes() >= graph.number_of_nodes()

        # Should be a tree (no cycles)
        import networkx as nx

        assert nx.is_tree(model)

    finally:
        Path(temp_path).unlink(missing_ok=True)


def test_morphology_graph_no_cycle():
    """Test that MorphologyGraph handles acyclic graphs correctly."""
    # Create a simple tree: 0 -> 1 -> 2
    graph = MorphologyGraph()

    # Add nodes
    for i in range(3):
        graph.add_node(
            i,
            xyz=np.array([float(i), 0.0, 0.0]),
            radius=1.0,
        )

    # Add edges (no cycle)
    graph.add_edge(0, 1)
    graph.add_edge(1, 2)

    # Verify structure
    assert graph.number_of_nodes() == 3
    assert graph.number_of_edges() == 2

    # Export to SWC
    with tempfile.NamedTemporaryFile(mode="w", suffix=".swc", delete=False) as f:
        temp_path = f.name

    try:
        swc_text = graph.to_swc_file(temp_path)

        # Should not have cycle break annotations
        assert "CYCLE_BREAK" not in swc_text

        # Reload and verify same structure
        model = SWCModel.from_swc_file(temp_path)

        assert model.number_of_nodes() == graph.number_of_nodes()
        assert model.number_of_edges() == graph.number_of_edges()

    finally:
        Path(temp_path).unlink(missing_ok=True)


def test_morphology_graph_multiple_cycles():
    """Test that MorphologyGraph handles multiple cycles correctly."""
    # Create a graph with two cycles
    graph = MorphologyGraph()

    # First cycle: 0 -> 1 -> 2 -> 0
    for i in range(3):
        graph.add_node(i, xyz=np.array([float(i), 0.0, 0.0]), radius=1.0)
    graph.add_edge(0, 1)
    graph.add_edge(1, 2)
    graph.add_edge(2, 0)

    # Second cycle: 3 -> 4 -> 5 -> 3
    for i in range(3, 6):
        graph.add_node(i, xyz=np.array([float(i), 1.0, 0.0]), radius=1.0)
    graph.add_edge(3, 4)
    graph.add_edge(4, 5)
    graph.add_edge(5, 3)

    assert graph.number_of_nodes() == 6
    assert graph.number_of_edges() == 6

    # Export to SWC
    with tempfile.NamedTemporaryFile(mode="w", suffix=".swc", delete=False) as f:
        temp_path = f.name

    try:
        swc_text = graph.to_swc_file(temp_path)

        # Should have cycle break annotations (at least 2)
        cycle_breaks = swc_text.count("CYCLE_BREAK")
        assert (
            cycle_breaks >= 2
        ), f"Expected at least 2 cycle breaks, got {cycle_breaks}"

        # Reload and verify it's a valid tree
        model = SWCModel.from_swc_file(temp_path)

        import networkx as nx

        assert nx.is_tree(model) or nx.is_forest(model)

    finally:
        Path(temp_path).unlink(missing_ok=True)
