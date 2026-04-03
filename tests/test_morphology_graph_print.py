"""Test MorphologyGraph print_attributes method."""

import numpy as np
from io import StringIO
import sys

from mascaf import MorphologyGraph


def test_print_attributes_basic():
    """Test basic print_attributes functionality."""
    graph = MorphologyGraph()

    # Add nodes with xyz and radius
    for i in range(3):
        graph.add_node(
            i,
            xyz=np.array([float(i), 0.0, 0.0]),
            radius=1.0 + i * 0.5,
        )

    # Add edges
    graph.add_edge(0, 1)
    graph.add_edge(1, 2)

    # Capture output
    old_stdout = sys.stdout
    sys.stdout = captured = StringIO()

    try:
        graph.print_attributes()
        output = captured.getvalue()

        # Check that output contains expected information
        assert "MorphologyGraph:" in output
        assert "nodes=3" in output
        assert "edges=2" in output
        assert "cycles=0" in output

    finally:
        sys.stdout = old_stdout


def test_print_attributes_with_cycle():
    """Test print_attributes with a cycle."""
    graph = MorphologyGraph()

    # Create a cycle: 0 -> 1 -> 2 -> 0
    for i in range(3):
        graph.add_node(i, xyz=np.array([float(i), 0.0, 0.0]), radius=1.0)

    graph.add_edge(0, 1)
    graph.add_edge(1, 2)
    graph.add_edge(2, 0)

    # Capture output
    old_stdout = sys.stdout
    sys.stdout = captured = StringIO()

    try:
        graph.print_attributes()
        output = captured.getvalue()

        # Should report 1 cycle
        assert "cycles=1" in output

    finally:
        sys.stdout = old_stdout


def test_print_attributes_node_info():
    """Test print_attributes with node_info=True."""
    graph = MorphologyGraph()

    # Add nodes with additional attributes
    graph.add_node(
        0, xyz=np.array([1.0, 2.0, 3.0]), radius=0.5, radius_strategy="equivalent_area"
    )
    graph.add_node(1, xyz=np.array([4.0, 5.0, 6.0]), radius=1.0, inside_mesh=True)

    graph.add_edge(0, 1)

    # Capture output
    old_stdout = sys.stdout
    sys.stdout = captured = StringIO()

    try:
        graph.print_attributes(node_info=True)
        output = captured.getvalue()

        # Check header
        assert "MorphologyGraph:" in output

        # Check node info is printed
        assert "Nodes:" in output
        assert "xyz=" in output
        assert "r=" in output
        assert "radius_strategy" in output
        assert "inside_mesh" in output

    finally:
        sys.stdout = old_stdout


def test_print_attributes_edge_info():
    """Test print_attributes with edge_info=True."""
    graph = MorphologyGraph()

    graph.add_node(0, xyz=np.array([0.0, 0.0, 0.0]), radius=1.0)
    graph.add_node(1, xyz=np.array([1.0, 0.0, 0.0]), radius=1.0)

    # Add edge with attribute
    graph.add_edge(0, 1, weight=2.5)

    # Capture output
    old_stdout = sys.stdout
    sys.stdout = captured = StringIO()

    try:
        graph.print_attributes(edge_info=True)
        output = captured.getvalue()

        # Check edge info is printed
        assert "Edges:" in output
        assert "0 -- 1" in output
        assert "weight" in output

    finally:
        sys.stdout = old_stdout
