"""
Tests for MorphologyGraph.scale_radii_to_match_mesh method.
"""

import numpy as np
import pytest
import trimesh

from mascaf import MorphologyGraph
from mascaf.mesh import MeshManager, example_mesh


class TestScaleRadiiToMatchMesh:
    """Test suite for scale_radii_to_match_mesh method."""

    def test_scale_factor_calculation_surface_area(self):
        """Test that scale factor is correctly calculated for surface area."""
        mesh = example_mesh("cylinder", radius=2.0, height=10.0)
        target_area = mesh.area

        graph = MorphologyGraph()
        graph.add_node(0, xyz=np.array([0.0, 0.0, 0.0]), radius=1.0)
        graph.add_node(1, xyz=np.array([0.0, 0.0, 10.0]), radius=1.0)
        graph.add_edge(0, 1)

        initial_area = graph.compute_surface_area(remove_overlaps=False)
        expected_scale = np.sqrt(target_area / initial_area)

        scale_factor = graph.scale_radii_to_match_mesh(
            mesh, metric="surface_area"
        )

        assert scale_factor == pytest.approx(expected_scale, rel=1e-6)
        
        # Verify area after scaling matches target
        final_area = graph.compute_surface_area(remove_overlaps=False)
        assert final_area == pytest.approx(target_area, rel=1e-6)

    def test_scale_factor_calculation_volume(self):
        """Test that scale factor is correctly calculated for volume."""
        mesh = example_mesh("cylinder", radius=2.0, height=10.0)
        target_volume = mesh.volume

        graph = MorphologyGraph()
        graph.add_node(0, xyz=np.array([0.0, 0.0, 0.0]), radius=1.0)
        graph.add_node(1, xyz=np.array([0.0, 0.0, 10.0]), radius=1.0)
        graph.add_edge(0, 1)

        initial_volume = graph.compute_volume(remove_overlaps=False)
        expected_scale = np.cbrt(target_volume / initial_volume)

        scale_factor = graph.scale_radii_to_match_mesh(mesh, metric="volume")

        assert scale_factor == pytest.approx(expected_scale, rel=1e-6)
        
        # Verify volume after scaling matches target
        final_volume = graph.compute_volume(remove_overlaps=False)
        assert final_volume == pytest.approx(target_volume, rel=1e-6)

    def test_radii_scaled_uniformly(self):
        """Test that all radii are scaled by the same factor."""
        mesh = example_mesh("cylinder", radius=2.0, height=10.0)

        graph = MorphologyGraph()
        graph.add_node(0, xyz=np.array([0.0, 0.0, 0.0]), radius=1.0)
        graph.add_node(1, xyz=np.array([0.0, 0.0, 10.0]), radius=1.5)
        graph.add_edge(0, 1)

        initial_r0 = graph.nodes[0]["radius"]
        initial_r1 = graph.nodes[1]["radius"]

        scale_factor = graph.scale_radii_to_match_mesh(
            mesh, metric="surface_area"
        )

        assert graph.nodes[0]["radius"] == pytest.approx(
            initial_r0 * scale_factor
        )
        assert graph.nodes[1]["radius"] == pytest.approx(
            initial_r1 * scale_factor
        )

    def test_scale_preserves_relative_proportions(self):
        """Test that scaling preserves relative proportions of radii."""
        mesh = example_mesh("cylinder", radius=2.0, height=10.0)

        graph = MorphologyGraph()
        graph.add_node(0, xyz=np.array([0.0, 0.0, 0.0]), radius=1.0)
        graph.add_node(1, xyz=np.array([0.0, 0.0, 5.0]), radius=2.0)
        graph.add_node(2, xyz=np.array([0.0, 0.0, 10.0]), radius=3.0)
        graph.add_edge(0, 1)
        graph.add_edge(1, 2)

        initial_r0 = graph.nodes[0]["radius"]
        initial_r1 = graph.nodes[1]["radius"]
        initial_r2 = graph.nodes[2]["radius"]
        ratio_01 = initial_r0 / initial_r1
        ratio_12 = initial_r1 / initial_r2

        graph.scale_radii_to_match_mesh(mesh, metric="volume")

        new_r0 = graph.nodes[0]["radius"]
        new_r1 = graph.nodes[1]["radius"]
        new_r2 = graph.nodes[2]["radius"]
        
        assert new_r0 / new_r1 == pytest.approx(ratio_01)
        assert new_r1 / new_r2 == pytest.approx(ratio_12)

    def test_scale_with_mesh_manager(self):
        """Test that method works with MeshManager input."""
        mesh = example_mesh("cylinder", radius=2.0, height=10.0)
        mesh_mgr = MeshManager(mesh=mesh)
        target_area = mesh_mgr.mesh.area

        graph = MorphologyGraph()
        graph.add_node(0, xyz=np.array([0.0, 0.0, 0.0]), radius=1.0)
        graph.add_node(1, xyz=np.array([0.0, 0.0, 10.0]), radius=1.0)
        graph.add_edge(0, 1)

        scale_factor = graph.scale_radii_to_match_mesh(
            mesh_mgr, metric="surface_area"
        )

        assert scale_factor > 0.0
        graph_area = graph.compute_surface_area(remove_overlaps=False)
        assert graph_area == pytest.approx(target_area, rel=1e-6)

    def test_scale_with_remove_overlaps(self):
        """Test scaling with remove_overlaps option."""
        mesh = example_mesh("cylinder", radius=2.0, height=10.0)
        target_area = mesh.area

        graph = MorphologyGraph()
        graph.add_node(0, xyz=np.array([0.0, 0.0, 0.0]), radius=1.0)
        graph.add_node(1, xyz=np.array([0.0, 0.0, 5.0]), radius=1.0)
        graph.add_node(2, xyz=np.array([0.0, 0.0, 10.0]), radius=1.0)
        graph.add_node(3, xyz=np.array([5.0, 0.0, 5.0]), radius=1.0)
        graph.add_edge(0, 1)
        graph.add_edge(1, 2)
        graph.add_edge(1, 3)

        initial_area = graph.compute_surface_area(remove_overlaps=True)

        scale_factor = graph.scale_radii_to_match_mesh(
            mesh, metric="surface_area", remove_overlaps=True
        )

        assert scale_factor > 0.0
        
        # Verify scaling relationship
        graph_area = graph.compute_surface_area(remove_overlaps=True)
        expected_area = initial_area * (scale_factor ** 2)
        assert graph_area == pytest.approx(expected_area, rel=1e-6)
        assert graph_area == pytest.approx(target_area, rel=1e-6)

    def test_invalid_metric_raises_error(self):
        """Test that invalid metric raises ValueError."""
        mesh = example_mesh("cylinder", radius=2.0, height=10.0)
        graph = MorphologyGraph()
        graph.add_node(0, xyz=np.array([0.0, 0.0, 0.0]), radius=1.0)
        graph.add_node(1, xyz=np.array([0.0, 0.0, 10.0]), radius=1.0)
        graph.add_edge(0, 1)

        with pytest.raises(ValueError, match="metric must be"):
            graph.scale_radii_to_match_mesh(mesh, metric="invalid")

    def test_zero_mesh_area_raises_error(self):
        """Test that mesh with zero area raises ValueError."""
        vertices = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        faces = np.array([[0, 1, 2]])
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

        graph = MorphologyGraph()
        graph.add_node(0, xyz=np.array([0.0, 0.0, 0.0]), radius=1.0)
        graph.add_node(1, xyz=np.array([0.0, 0.0, 10.0]), radius=1.0)
        graph.add_edge(0, 1)

        with pytest.raises(ValueError, match="zero or negative surface area"):
            graph.scale_radii_to_match_mesh(mesh, metric="surface_area")

    def test_zero_graph_area_raises_error(self):
        """Test that graph with zero area raises ValueError."""
        mesh = example_mesh("cylinder", radius=2.0, height=10.0)

        graph = MorphologyGraph()
        graph.add_node(0, xyz=np.array([0.0, 0.0, 0.0]), radius=0.0)
        graph.add_node(1, xyz=np.array([0.0, 0.0, 10.0]), radius=0.0)
        graph.add_edge(0, 1)

        with pytest.raises(ValueError, match="zero or negative surface area"):
            graph.scale_radii_to_match_mesh(mesh, metric="surface_area")

    def test_area_scales_quadratically_with_radius(self):
        """Test that surface area scales as r^2."""
        mesh = example_mesh("cylinder", radius=2.0, height=10.0)

        graph = MorphologyGraph()
        graph.add_node(0, xyz=np.array([0.0, 0.0, 0.0]), radius=1.0)
        graph.add_node(1, xyz=np.array([0.0, 0.0, 10.0]), radius=1.0)
        graph.add_edge(0, 1)

        initial_area = graph.compute_surface_area(remove_overlaps=False)
        
        scale_factor = graph.scale_radii_to_match_mesh(
            mesh, metric="surface_area"
        )
        
        final_area = graph.compute_surface_area(remove_overlaps=False)
        
        # Area should scale as r^2
        assert final_area == pytest.approx(
            initial_area * (scale_factor ** 2), rel=1e-6
        )

    def test_volume_scales_cubically_with_radius(self):
        """Test that volume scales as r^3."""
        mesh = example_mesh("cylinder", radius=2.0, height=10.0)

        graph = MorphologyGraph()
        graph.add_node(0, xyz=np.array([0.0, 0.0, 0.0]), radius=1.0)
        graph.add_node(1, xyz=np.array([0.0, 0.0, 10.0]), radius=1.0)
        graph.add_edge(0, 1)

        initial_volume = graph.compute_volume(remove_overlaps=False)
        
        scale_factor = graph.scale_radii_to_match_mesh(mesh, metric="volume")
        
        final_volume = graph.compute_volume(remove_overlaps=False)
        
        # Volume should scale as r^3
        assert final_volume == pytest.approx(
            initial_volume * (scale_factor ** 3), rel=1e-6
        )

    def test_scale_multiple_times(self):
        """Test that scaling can be applied multiple times."""
        mesh1 = example_mesh("cylinder", radius=1.0, height=10.0)
        mesh2 = example_mesh("cylinder", radius=2.0, height=10.0)

        graph = MorphologyGraph()
        graph.add_node(0, xyz=np.array([0.0, 0.0, 0.0]), radius=1.0)
        graph.add_node(1, xyz=np.array([0.0, 0.0, 10.0]), radius=1.0)
        graph.add_edge(0, 1)

        scale1 = graph.scale_radii_to_match_mesh(mesh1, metric="volume")
        volume1 = graph.compute_volume(remove_overlaps=False)
        assert volume1 == pytest.approx(mesh1.volume, rel=1e-6)

        scale2 = graph.scale_radii_to_match_mesh(mesh2, metric="volume")
        volume2 = graph.compute_volume(remove_overlaps=False)
        assert volume2 == pytest.approx(mesh2.volume, rel=1e-6)

        assert scale1 > 0.0
        assert scale2 > 0.0

    def test_scale_complex_graph(self):
        """Test scaling on a more complex graph structure."""
        mesh = example_mesh("cylinder", radius=3.0, height=20.0)
        target_area = mesh.area

        graph = MorphologyGraph()
        graph.add_node(0, xyz=np.array([0.0, 0.0, 0.0]), radius=2.0)
        graph.add_node(1, xyz=np.array([0.0, 0.0, 5.0]), radius=1.8)
        graph.add_node(2, xyz=np.array([0.0, 0.0, 10.0]), radius=1.5)
        graph.add_node(3, xyz=np.array([5.0, 0.0, 10.0]), radius=1.2)
        graph.add_node(4, xyz=np.array([0.0, 5.0, 10.0]), radius=1.0)
        graph.add_edge(0, 1)
        graph.add_edge(1, 2)
        graph.add_edge(1, 3)
        graph.add_edge(1, 4)

        initial_radii = {
            node: graph.nodes[node]["radius"] for node in graph.nodes()
        }

        scale_factor = graph.scale_radii_to_match_mesh(
            mesh, metric="surface_area"
        )

        # Verify all radii scaled correctly
        for node in graph.nodes():
            expected = initial_radii[node] * scale_factor
            assert graph.nodes[node]["radius"] == pytest.approx(expected)

        # Verify surface area matches
        graph_area = graph.compute_surface_area(remove_overlaps=False)
        assert graph_area == pytest.approx(target_area, rel=1e-6)
