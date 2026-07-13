"""Tests for BasisOptimizer weighted centering force and step capping."""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

from mascaf import BasisOptimizer, BasisOptimizerOptions, MorphologyGraph


def _sphere_mesh(radius: float = 1.0, subdivisions: int = 3) -> trimesh.Trimesh:
    mesh = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
    mesh.process(validate=True)
    return mesh


def _chain_graph(positions: list[np.ndarray], radius: float = 0.05) -> MorphologyGraph:
    graph = MorphologyGraph()
    for i, pos in enumerate(positions):
        graph.add_node(i, xyz=np.asarray(pos, dtype=float), radius=radius)
    for i in range(len(positions) - 1):
        graph.add_edge(i, i + 1)
    return graph


@pytest.fixture
def unit_sphere() -> trimesh.Trimesh:
    return _sphere_mesh(radius=1.0)


def test_options_new_force_defaults():
    opts = BasisOptimizerOptions()
    assert opts.repulsion_power == 1.0
    assert opts.localization_beta == 2.0
    assert opts.weight_epsilon == 1e-6
    assert opts.step_cap_factor == 0.5
    assert opts.lambda_centering == 0.5
    assert opts.lambda_smoothing == 0.2
    assert not hasattr(opts, "step_scale")
    assert not hasattr(opts, "lambda_smooth")
    assert not hasattr(opts, "lambda_vertex")
    assert not hasattr(opts, "vertex_repulsion_distance")


def test_centering_force_near_zero_at_sphere_center(unit_sphere):
    graph = _chain_graph(
        [
            np.array([-0.5, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0]),
            np.array([0.5, 0.0, 0.0]),
        ]
    )
    opt = BasisOptimizer(graph, unit_sphere, BasisOptimizerOptions(n_rays=6))
    force = opt._compute_centering_force(np.array([0.0, 0.0, 0.0]))
    assert np.linalg.norm(force) < 0.05


def test_centering_force_pushes_off_center_point_inward(unit_sphere):
    graph = _chain_graph(
        [
            np.array([-0.5, 0.0, 0.0]),
            np.array([0.4, 0.0, 0.0]),
            np.array([0.5, 0.0, 0.0]),
        ]
    )
    opt = BasisOptimizer(graph, unit_sphere, BasisOptimizerOptions(n_rays=6))
    point = np.array([0.4, 0.0, 0.0])
    force = opt._compute_centering_force(point)

    assert np.linalg.norm(force) > 1e-3
    # F_centering = -d_min * weighted unit sum, so ||F|| <= d_min.
    d_min = opt._ray_distance_to_surface(point, np.array([1.0, 0.0, 0.0]))
    assert np.linalg.norm(force) <= d_min + 1e-6
    # Closest surface is +x, so repulsion should push toward -x (center).
    assert force[0] < 0.0


def test_centering_force_outside_uses_closest_point(unit_sphere):
    graph = _chain_graph([np.array([0.0, 0.0, 0.0])])
    opt = BasisOptimizer(graph, unit_sphere)
    outside = np.array([2.0, 0.0, 0.0])
    force = opt._compute_centering_force(outside)
    assert force[0] == pytest.approx(-1.0, abs=1e-5)
    assert np.linalg.norm(force) == pytest.approx(1.0, abs=1e-5)


def test_localization_reduces_far_ray_weights(unit_sphere):
    """Larger beta should more strongly favor the nearest-ray direction."""
    graph = _chain_graph([np.array([0.0, 0.0, 0.0])])
    point = np.array([0.5, 0.0, 0.0])

    weak = BasisOptimizer(
        graph,
        unit_sphere,
        BasisOptimizerOptions(n_rays=6, localization_beta=0.0),
    )
    strong = BasisOptimizer(
        graph,
        unit_sphere,
        BasisOptimizerOptions(n_rays=6, localization_beta=20.0),
    )

    f_weak = weak._compute_centering_force(point)
    f_strong = strong._compute_centering_force(point)

    # Strong localization should align more with -x (nearest wall at +x).
    assert f_strong[0] < f_weak[0]
    assert abs(f_strong[0]) > abs(f_weak[0]) * 0.9


def test_smoothing_force_is_neighbor_centroid_pull(unit_sphere):
    graph = _chain_graph(
        [
            np.array([-0.5, 0.0, 0.0]),
            np.array([0.3, 0.1, 0.0]),
            np.array([0.5, 0.0, 0.0]),
        ]
    )
    opt = BasisOptimizer(graph, unit_sphere)
    s = opt._compute_smoothing_force_for_node(1)
    expected = 0.5 * (
        graph.get_node_position(0) + graph.get_node_position(2)
    ) - graph.get_node_position(1)
    np.testing.assert_allclose(s, expected)


def test_cap_step_by_surface_distance(unit_sphere):
    graph = _chain_graph([np.array([0.0, 0.0, 0.0])])
    opts = BasisOptimizerOptions(step_cap_factor=0.5)
    opt = BasisOptimizer(graph, unit_sphere, opts)

    point = np.array([0.0, 0.0, 0.0])
    # From center along +x, surface is at distance ~1.
    huge_step = np.array([10.0, 0.0, 0.0])
    capped = opt._cap_step_by_surface_distance(point, huge_step)

    assert np.linalg.norm(capped) == pytest.approx(0.5, rel=0.05)
    assert capped[0] > 0.0
    assert abs(capped[1]) < 1e-8
    assert abs(capped[2]) < 1e-8


def test_cap_leaves_small_step_unchanged(unit_sphere):
    graph = _chain_graph([np.array([0.0, 0.0, 0.0])])
    opt = BasisOptimizer(
        graph, unit_sphere, BasisOptimizerOptions(step_cap_factor=0.5)
    )
    point = np.array([0.0, 0.0, 0.0])
    small = np.array([0.01, 0.0, 0.0])
    capped = opt._cap_step_by_surface_distance(point, small)
    np.testing.assert_allclose(capped, small)


def test_forcing_moves_interior_node_toward_center(unit_sphere):
    graph = _chain_graph(
        [
            np.array([-0.6, 0.0, 0.0]),
            np.array([0.45, 0.0, 0.0]),
            np.array([0.6, 0.0, 0.0]),
        ]
    )
    opts = BasisOptimizerOptions(
        do_pruning=False,
        do_snapping=False,
        do_forcing=True,
        preserve_terminal_nodes=True,
        preserve_branch_nodes=False,
        lambda_centering=0.5,
        lambda_smoothing=0.0,
        max_iterations=30,
        convergence_threshold=1e-6,
        step_cap_factor=0.5,
        n_rays=6,
    )
    before = graph.get_node_position(1).copy()
    optimized = BasisOptimizer(graph, unit_sphere, opts).optimize()
    after = optimized.get_node_position(1)

    assert np.linalg.norm(after) < np.linalg.norm(before)
    assert unit_sphere.contains(after.reshape(1, 3))[0]


def test_forcing_preserves_terminal_nodes(unit_sphere):
    terminals = [
        np.array([-0.5, 0.0, 0.0]),
        np.array([0.5, 0.0, 0.0]),
    ]
    graph = _chain_graph(
        [terminals[0], np.array([0.3, 0.0, 0.0]), terminals[1]]
    )
    opts = BasisOptimizerOptions(
        do_pruning=False,
        do_snapping=False,
        do_forcing=True,
        preserve_terminal_nodes=True,
        lambda_smoothing=0.0,
        max_iterations=5,
    )
    optimized = BasisOptimizer(graph, unit_sphere, opts).optimize()
    np.testing.assert_allclose(optimized.get_node_position(0), terminals[0])
    np.testing.assert_allclose(optimized.get_node_position(2), terminals[1])


def test_forcing_blend_matches_new_force_model(unit_sphere):
    """delta_v = lambda_centering * F_c + lambda_smoothing * F_s, then capped."""
    graph = _chain_graph(
        [
            np.array([-0.5, 0.0, 0.0]),
            np.array([0.3, 0.1, 0.0]),
            np.array([0.5, 0.0, 0.0]),
        ]
    )
    lambda_centering = 0.5
    lambda_smoothing = 0.4
    opts = BasisOptimizerOptions(
        do_pruning=False,
        do_snapping=False,
        do_forcing=True,
        preserve_terminal_nodes=True,
        lambda_centering=lambda_centering,
        lambda_smoothing=lambda_smoothing,
        max_iterations=1,
        step_cap_factor=1.0,
        n_rays=6,
        convergence_threshold=0.0,
    )
    opt = BasisOptimizer(graph, unit_sphere, opts)
    pos = opt.graph.get_node_position(1).copy()
    f_c = opt._compute_centering_force(pos)
    f_s = opt._compute_smoothing_force_for_node(1)
    assert np.linalg.norm(f_c) > 1e-8
    assert np.linalg.norm(f_s) > 1e-8

    expected_delta = lambda_centering * f_c + lambda_smoothing * f_s
    expected_step = opt._cap_step_by_surface_distance(pos, expected_delta)

    opt._run_forcing_phase()
    actual = opt.graph.get_node_position(1) - pos
    np.testing.assert_allclose(actual, expected_step, rtol=1e-6, atol=1e-8)


def test_snap_moves_outside_point_to_chord_midpoint(unit_sphere):
    graph = _chain_graph(
        [
            np.array([-0.5, 0.0, 0.0]),
            np.array([1.5, 0.0, 0.0]),
            np.array([0.5, 0.0, 0.0]),
        ]
    )
    opts = BasisOptimizerOptions(
        do_pruning=False,
        do_snapping=True,
        do_forcing=False,
        preserve_terminal_nodes=False,
    )
    opt = BasisOptimizer(graph, unit_sphere, opts)
    assert 1 in opt.graph.get_outside_nodes(unit_sphere)

    opt._run_snapping_phase()
    snapped = opt.graph.get_node_position(1)

    assert unit_sphere.contains(snapped.reshape(1, 3))[0]
    # Diameter along +x through a unit sphere: hits near (±1, 0, 0) → midpoint ~0.
    np.testing.assert_allclose(snapped, np.zeros(3), atol=0.05)
    assert opt.graph.get_outside_nodes(unit_sphere) == []


def test_snap_point_to_chord_midpoint_helper(unit_sphere):
    graph = _chain_graph([np.array([0.0, 0.0, 0.0])])
    opt = BasisOptimizer(graph, unit_sphere)
    outside = np.array([1.5, 0.0, 0.0])
    mid = opt._snap_point_to_chord_midpoint(outside)
    assert mid is not None
    assert unit_sphere.contains(mid.reshape(1, 3))[0]
    np.testing.assert_allclose(mid, np.zeros(3), atol=0.05)


def test_snap_ray_perturbation_recovers_even_hits_on_ts1():
    """Closest-point ray can hit a mesh edge (odd hits); perturbation should recover."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    mesh_path = root / "data" / "mesh" / "processed" / "TS1.obj"
    skel_path = root / "data" / "mcf_skeletons" / "TS1_qst0.5_mcst5.polylines.txt"
    if not mesh_path.exists() or not skel_path.exists():
        pytest.skip("TS1 mesh/skeleton not available")

    from mascaf import MeshManager, SkeletonGraph

    mm = MeshManager(mesh_path=str(mesh_path))
    skeleton = SkeletonGraph.from_txt(str(skel_path))
    basis = MorphologyGraph.from_skeleton_graph_resample(
        skeleton, 0.06 * mm.bounding_box_diagonal()
    )
    opt = BasisOptimizer(basis, mm.mesh, BasisOptimizerOptions(do_forcing=False))

    node = 14
    assert node in basis.get_outside_nodes(mm.mesh)
    pos = basis.get_node_position(node)
    direction, _ = opt._compute_snap_direction(pos)
    raw_hits = opt._ray_intersections_sorted(pos, direction)
    assert raw_hits.shape[0] % 2 == 1

    mid = opt._snap_point_to_chord_midpoint(pos)
    assert mid is not None
    assert mm.mesh.contains(mid.reshape(1, 3))[0]
    # Grazing even-hit chords (~1e-3) leave the midpoint on the surface;
    # a usable lumen chord should move it well off the boundary.
    from trimesh.proximity import signed_distance

    assert abs(float(signed_distance(mm.mesh, mid.reshape(1, 3))[0])) > 0.05


def test_options_no_longer_expose_snap_distance_multiplier():
    opts = BasisOptimizerOptions()
    assert not hasattr(opts, "snap_distance_multiplier")
    assert opts.snap_min_chord_fraction == 5e-4
