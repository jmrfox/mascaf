"""Tests for MorphologyGraph SWC file I/O with cycle restoration."""

import tempfile
from pathlib import Path

import numpy as np

from mcf2swc import MorphologyGraph, Junction


def test_from_swc_file_basic():
    """Test loading a simple SWC file without cycles."""
    graph = MorphologyGraph()
    
    # Create a simple linear structure
    graph.add_junction(Junction(id=1, xyz=np.array([0.0, 0.0, 0.0]), radius=1.0))
    graph.add_junction(Junction(id=2, xyz=np.array([1.0, 0.0, 0.0]), radius=0.8))
    graph.add_junction(Junction(id=3, xyz=np.array([2.0, 0.0, 0.0]), radius=0.6))
    
    graph.add_edge(1, 2)
    graph.add_edge(2, 3)
    
    # Export to SWC
    with tempfile.NamedTemporaryFile(mode='w', suffix='.swc', delete=False) as f:
        swc_path = f.name
    
    try:
        graph.to_swc_file(swc_path)
        
        # Load it back
        loaded_graph = MorphologyGraph.from_swc_file(swc_path)
        
        # Verify structure
        assert loaded_graph.number_of_nodes() == 3
        assert loaded_graph.number_of_edges() == 2
        
        # Verify node attributes
        for node_id in [1, 2, 3]:
            assert node_id in loaded_graph
            assert 'xyz' in loaded_graph.nodes[node_id]
            assert 'radius' in loaded_graph.nodes[node_id]
        
        # Verify edges
        assert loaded_graph.has_edge(1, 2)
        assert loaded_graph.has_edge(2, 3)
        
    finally:
        Path(swc_path).unlink(missing_ok=True)


def test_from_swc_file_with_cycle():
    """Test loading an SWC file with cycle annotations."""
    graph = MorphologyGraph()
    
    # Create a simple cycle: 1-2-3-1
    graph.add_junction(Junction(id=1, xyz=np.array([0.0, 0.0, 0.0]), radius=1.0))
    graph.add_junction(Junction(id=2, xyz=np.array([1.0, 0.0, 0.0]), radius=0.8))
    graph.add_junction(Junction(id=3, xyz=np.array([0.5, 1.0, 0.0]), radius=0.6))
    
    graph.add_edge(1, 2)
    graph.add_edge(2, 3)
    graph.add_edge(3, 1)  # This creates a cycle
    
    # Export to SWC (should break the cycle)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.swc', delete=False) as f:
        swc_path = f.name
    
    try:
        graph.to_swc_file(swc_path, annotate_breaks=True)
        
        # Load it back (should restore the cycle)
        loaded_graph = MorphologyGraph.from_swc_file(swc_path)
        
        # Verify structure - should have 3 nodes (duplicate removed)
        assert loaded_graph.number_of_nodes() == 3
        
        # Verify the cycle is restored
        assert loaded_graph.number_of_edges() == 3
        assert loaded_graph.has_edge(1, 2)
        assert loaded_graph.has_edge(2, 3)
        assert loaded_graph.has_edge(3, 1)
        
        # Verify it's actually a cycle
        import networkx as nx
        assert not nx.is_forest(loaded_graph)
        cycles = nx.cycle_basis(loaded_graph)
        assert len(cycles) > 0
        
    finally:
        Path(swc_path).unlink(missing_ok=True)


def test_from_swc_file_branching():
    """Test loading an SWC file with branching structure."""
    graph = MorphologyGraph()
    
    # Create a branching structure: 1-2-3
    #                                  \-4
    graph.add_junction(Junction(id=1, xyz=np.array([0.0, 0.0, 0.0]), radius=1.0))
    graph.add_junction(Junction(id=2, xyz=np.array([1.0, 0.0, 0.0]), radius=0.8))
    graph.add_junction(Junction(id=3, xyz=np.array([2.0, 0.0, 0.0]), radius=0.6))
    graph.add_junction(Junction(id=4, xyz=np.array([1.0, 1.0, 0.0]), radius=0.6))
    
    graph.add_edge(1, 2)
    graph.add_edge(2, 3)
    graph.add_edge(2, 4)
    
    # Export to SWC
    with tempfile.NamedTemporaryFile(mode='w', suffix='.swc', delete=False) as f:
        swc_path = f.name
    
    try:
        graph.to_swc_file(swc_path)
        
        # Load it back
        loaded_graph = MorphologyGraph.from_swc_file(swc_path)
        
        # Verify structure
        assert loaded_graph.number_of_nodes() == 4
        assert loaded_graph.number_of_edges() == 3
        
        # Verify branching
        assert loaded_graph.degree[2] == 3  # Node 2 is branch point
        assert loaded_graph.has_edge(1, 2)
        assert loaded_graph.has_edge(2, 3)
        assert loaded_graph.has_edge(2, 4)
        
    finally:
        Path(swc_path).unlink(missing_ok=True)
