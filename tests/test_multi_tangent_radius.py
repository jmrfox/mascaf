"""
Tests for multi-tangent radius computation in graph fitting.
"""

import numpy as np
import pytest
from mcf2swc import (
    SkeletonGraph,
    FitOptions,
    fit_morphology,
    example_mesh,
)


class TestMultiTangentRadius:
    """Test multi-tangent radius computation functionality."""

    @pytest.fixture
    def cylinder_mesh(self):
        """Create a simple cylinder mesh for testing."""
        return example_mesh("cylinder", radius=1.5, height=6.0, sections=32)

    @pytest.fixture
    def y_shaped_skeleton(self):
        """Create a Y-shaped skeleton with terminal and branch nodes."""
        points = [
            [0.0, 0.0, -2.0],  # Terminal node 1
            [0.0, 0.0, -1.0],  # Branch node
            [0.0, 0.0, 0.0],  # Center branch node
            [0.0, 0.0, 1.0],  # Terminal node 2
            [1.0, 0.0, 0.0],  # Terminal node 3
            [-1.0, 0.0, 0.0],  # Terminal node 4
        ]

        edges = [
            (0, 1),  # Terminal 1 to branch
            (1, 2),  # Branch to center
            (2, 3),  # Center to terminal 2
            (2, 4),  # Center to terminal 3
            (2, 5),  # Center to terminal 4
        ]

        skeleton = SkeletonGraph()
        for i, point in enumerate(points):
            skeleton.add_node(i, pos=np.array(point, dtype=float))

        for u, v in edges:
            skeleton.add_edge(u, v)

        return skeleton

    @pytest.fixture
    def linear_skeleton(self):
        """Create a simple linear skeleton."""
        points = [
            [0.0, 0.0, -2.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 2.0],
        ]

        edges = [(0, 1), (1, 2), (2, 3), (3, 4)]

        skeleton = SkeletonGraph()
        for i, point in enumerate(points):
            skeleton.add_node(i, pos=np.array(point, dtype=float))

        for u, v in edges:
            skeleton.add_edge(u, v)

        return skeleton

    def test_tangent_computation_terminal_nodes(self, linear_skeleton):
        """Test tangent computation for terminal nodes."""
        from mcf2swc.graph_fitting import _compute_node_tangents

        # Terminal nodes should have 1 tangent
        terminal_tangents = _compute_node_tangents(linear_skeleton, 0)
        assert len(terminal_tangents) == 1
        assert np.allclose(np.linalg.norm(terminal_tangents[0]), 1.0)

    def test_tangent_computation_branch_nodes(self, y_shaped_skeleton):
        """Test tangent computation for branch nodes."""
        from mcf2swc.graph_fitting import _compute_node_tangents

        # Center branch node (degree 4) should have 4 tangents
        center_tangents = _compute_node_tangents(y_shaped_skeleton, 2)
        assert len(center_tangents) == 4

        # All tangents should be unit vectors
        for tangent in center_tangents:
            assert np.allclose(np.linalg.norm(tangent), 1.0)

    def test_multi_tangent_reduction_methods(self, cylinder_mesh, y_shaped_skeleton):
        """Test different multi-tangent reduction methods."""
        reduction_methods = ["mean", "min", "max", "median"]

        for method in reduction_methods:
            opts = FitOptions(
                spacing=0.5,
                radius_strategy="equivalent_area",
                multi_tangent_reduction=method,
            )

            morphology = fit_morphology(cylinder_mesh, y_shaped_skeleton, opts)

            # Should successfully create morphology
            assert len(morphology.nodes) > 0

            # All nodes should have radii
            radii = [
                node_data.get("radius", 0.0) for node_data in morphology.nodes.values()
            ]
            assert all(r > 0 for r in radii)

    def test_radius_strategies_with_multi_tangent(
        self, cylinder_mesh, y_shaped_skeleton
    ):
        """Test different radius strategies with multi-tangent approach."""
        strategies = [
            "equivalent_area",
            "equivalent_perimeter",
            "section_median",
            "section_circle_fit",
            "nearest_surface",
        ]

        for strategy in strategies:
            opts = FitOptions(
                spacing=0.5,
                radius_strategy=strategy,
                multi_tangent_reduction="median",
            )

            morphology = fit_morphology(cylinder_mesh, y_shaped_skeleton, opts)

            # Should successfully create morphology
            assert len(morphology.nodes) > 0

            # All nodes should have radii
            radii = [
                node_data.get("radius", 0.0) for node_data in morphology.nodes.values()
            ]
            assert all(r >= 0 for r in radii)

    def test_linear_skeleton_multi_tangent(self, cylinder_mesh, linear_skeleton):
        """Test multi-tangent approach on linear skeleton."""
        opts = FitOptions(
            spacing=0.5,
            radius_strategy="equivalent_area",
            multi_tangent_reduction="median",
        )

        morphology = fit_morphology(cylinder_mesh, linear_skeleton, opts)

        # Should successfully create morphology
        assert len(morphology.nodes) > 0

        # All nodes should have radii
        radii = [
            node_data.get("radius", 0.0) for node_data in morphology.nodes.values()
        ]
        assert all(r > 0 for r in radii)

    def test_invalid_reduction_method(self, cylinder_mesh, y_shaped_skeleton):
        """Test that invalid reduction method raises error."""
        opts = FitOptions(
            spacing=0.5,
            radius_strategy="equivalent_area",
            multi_tangent_reduction="invalid_method",
        )

        with pytest.raises(ValueError, match="Unknown reduction method"):
            fit_morphology(cylinder_mesh, y_shaped_skeleton, opts)

    def test_radius_reduction_function(self):
        """Test the radius reduction function directly."""
        from mcf2swc.graph_fitting import _reduce_multi_radii

        # Test with multiple radii
        radii = [1.0, 2.0, 3.0, 4.0]

        assert np.isclose(_reduce_multi_radii(radii, "mean"), 2.5)
        assert np.isclose(_reduce_multi_radii(radii, "min"), 1.0)
        assert np.isclose(_reduce_multi_radii(radii, "max"), 4.0)
        assert np.isclose(_reduce_multi_radii(radii, "median"), 2.5)

        # Test with empty list
        assert _reduce_multi_radii([], "mean") == 0.0

        # Test with single radius
        assert _reduce_multi_radii([2.5], "mean") == 2.5

    def test_fit_options_default_multi_tangent(self):
        """Test that FitOptions has correct default multi_tangent_reduction."""
        opts = FitOptions()
        assert opts.multi_tangent_reduction == "median"

    def test_skeleton_node_radii_computation(self, cylinder_mesh, y_shaped_skeleton):
        """Test the skeleton node radii computation function."""
        from mcf2swc.graph_fitting import _compute_skeleton_node_radii
        from scipy.spatial import cKDTree

        opts = FitOptions(spacing=0.5, radius_strategy="equivalent_area")
        eps = 1e-4 * max(cylinder_mesh.bounding_box.extents)
        V = np.asarray(cylinder_mesh.vertices, dtype=float)
        v_kdtree = cKDTree(V)

        node_radii = _compute_skeleton_node_radii(
            y_shaped_skeleton, cylinder_mesh, opts, eps, V, v_kdtree
        )

        # Should have radii for all nodes
        assert len(node_radii) == y_shaped_skeleton.number_of_nodes()

        # All radii should be positive
        for node_id, radius in node_radii.items():
            assert radius >= 0
            assert isinstance(radius, (int, float, np.number))
