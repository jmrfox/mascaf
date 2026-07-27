"""Tests for BasisOptimizer weighted centering force and step capping."""

from __future__ import annotations

import logging

import numpy as np
import pytest
import trimesh

from mascaf import BasisOptimizer, BasisOptimizerOptions, MorphologyGraph
from mascaf.mesh_contains import point_inside_mesh, signed_distances


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
    assert opts.alpha_s == 0.1
    assert opts.step_scale == 0.5
    assert opts.ray_jitter == 0.0
    assert opts.active_resample is False
    assert opts.active_resample_min_fraction is None
    assert opts.active_resample_max_fraction is None
    assert opts.active_resample_allow_cycle_collapse is False
    assert opts.pruning_length is None
    assert not hasattr(opts, "do_consolidation")
    assert not hasattr(opts, "consolidation_length_fraction")
    assert opts.pruning_length_fraction is None
    assert not hasattr(opts, "lambda_centering")
    assert not hasattr(opts, "lambda_smoothing")
    assert not hasattr(opts, "pruning_min_length")
    assert not hasattr(opts, "pruning_min_length_fraction")
    assert opts.centering_error_stop_fraction == 0.1
    assert opts.centering_error_plateau_tol == 1e-3
    assert opts.centering_error_plateau_patience == 2
    assert opts.centering_error_increase_patience == 2
    assert opts.outside_distance_tol is None
    assert opts.outside_distance_tol_fraction == 1e-6
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


def test_ray_jitter_perturbs_directions_independently():
    graph = _chain_graph([np.array([0.0, 0.0, 0.0])])
    opt = BasisOptimizer(
        graph,
        trimesh.creation.icosphere(subdivisions=2),
        BasisOptimizerOptions(n_rays=6, ray_jitter=0.2),
    )
    base = opt._get_uniform_sphere_directions(6)
    rng = np.random.default_rng(0)
    jittered = opt._apply_angular_ray_jitter(base, 0.2, rng)
    assert jittered.shape == base.shape
    # Each direction remains unit length and differs from the base set.
    for b, j in zip(base, jittered):
        assert np.linalg.norm(j) == pytest.approx(1.0, abs=1e-9)
        assert not np.allclose(b, j, atol=1e-6)
    # Zero jitter is a no-op.
    same = opt._forcing_ray_directions(np.random.default_rng(1))
    # With ray_jitter=0.2, forcing_ray_directions should differ from base.
    opt_off = BasisOptimizer(
        graph,
        trimesh.creation.icosphere(subdivisions=2),
        BasisOptimizerOptions(n_rays=6, ray_jitter=0.0),
    )
    np.testing.assert_allclose(
        opt_off._forcing_ray_directions(np.random.default_rng(0)),
        base,
    )
    assert not np.allclose(same, base)


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
        alpha_s=0.0,
        max_iterations=30,
        convergence_threshold=1e-6,
        step_cap_factor=0.5,
        n_rays=6,
    )
    before = graph.get_node_position(1).copy()
    optimized = BasisOptimizer(graph, unit_sphere, opts).optimize()
    after = optimized.get_node_position(1)

    assert np.linalg.norm(after) < np.linalg.norm(before)
    assert point_inside_mesh(unit_sphere, after)


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
        alpha_s=0.0,
        max_iterations=5,
    )
    optimized = BasisOptimizer(graph, unit_sphere, opts).optimize()
    np.testing.assert_allclose(optimized.get_node_position(0), terminals[0])
    np.testing.assert_allclose(optimized.get_node_position(2), terminals[1])


def test_forcing_blend_matches_new_force_model(unit_sphere):
    """delta_v uses magnitude-matched alpha_s blend, step_scale, then capped."""
    graph = _chain_graph(
        [
            np.array([-0.5, 0.0, 0.0]),
            np.array([0.3, 0.1, 0.0]),
            np.array([0.5, 0.0, 0.0]),
        ]
    )
    alpha_s = 0.4
    step_scale = 0.5
    opts = BasisOptimizerOptions(
        do_pruning=False,
        do_snapping=False,
        do_forcing=True,
        preserve_terminal_nodes=True,
        alpha_s=alpha_s,
        step_scale=step_scale,
        max_iterations=1,
        step_cap_factor=1.0,
        n_rays=6,
        convergence_threshold=0.0,
        centering_error_stop_fraction=0.0,
        centering_error_plateau_tol=0.0,
        centering_error_plateau_patience=100,
    )
    opt = BasisOptimizer(graph, unit_sphere, opts)
    pos = opt.graph.get_node_position(1).copy()
    f_c = opt._compute_centering_force(pos)
    f_s = opt._compute_smoothing_force_for_node(1)
    f_c_norm = float(np.linalg.norm(f_c))
    f_s_norm = float(np.linalg.norm(f_s))
    assert f_c_norm > 1e-8
    assert f_s_norm > 1e-8

    expected_delta = step_scale * (
        (1.0 - alpha_s) * f_c + alpha_s * f_s * (f_c_norm / f_s_norm)
    )
    expected_step = opt._cap_step_by_surface_distance(pos, expected_delta)

    opt._run_forcing_phase()
    actual = opt.graph.get_node_position(1) - pos
    np.testing.assert_allclose(actual, expected_step, rtol=1e-6, atol=1e-8)


def test_forcing_requires_all_nodes_inside(unit_sphere):
    graph = _chain_graph(
        [
            np.array([-0.5, 0.0, 0.0]),
            np.array([1.5, 0.0, 0.0]),
            np.array([0.5, 0.0, 0.0]),
        ]
    )
    opts = BasisOptimizerOptions(
        do_pruning=False,
        do_snapping=False,
        do_forcing=True,
        preserve_terminal_nodes=False,
        max_iterations=1,
    )
    opt = BasisOptimizer(graph, unit_sphere, opts)
    with pytest.raises(RuntimeError, match="Forcing requires all nodes inside"):
        opt._run_forcing_phase()


def test_forcing_raises_when_uncapped_step_exits_mesh(unit_sphere, monkeypatch):
    graph = _chain_graph(
        [
            np.array([-0.5, 0.0, 0.0]),
            np.array([0.2, 0.0, 0.0]),
            np.array([0.5, 0.0, 0.0]),
        ]
    )
    opts = BasisOptimizerOptions(
        do_pruning=False,
        do_snapping=False,
        do_forcing=True,
        preserve_terminal_nodes=True,
        alpha_s=0.0,
        max_iterations=1,
        n_rays=6,
        centering_error_stop_fraction=0.0,
    )
    opt = BasisOptimizer(graph, unit_sphere, opts)

    def _uncapped(point, step):
        return np.asarray(step, dtype=float) * 100.0

    monkeypatch.setattr(opt, "_cap_step_by_surface_distance", _uncapped)
    monkeypatch.setattr(
        opt,
        "_compute_centering_force",
        lambda point, directions=None: np.array([1.0, 0.0, 0.0]),
    )
    with pytest.raises(RuntimeError, match="outside the mesh") as exc_info:
        opt._run_forcing_phase()
    msg = str(exc_info.value)
    assert "d_force" in msg
    assert "signed_distance" in msg
    assert "outside_distance_tol" in msg
    assert "||step||" in msg


def test_near_surface_signed_distance_not_clearly_outside(unit_sphere):
    """Boundary numeric shell should not count as a hard exterior failure."""
    graph = _chain_graph([np.array([0.0, 0.0, 0.0])])
    opt = BasisOptimizer(graph, unit_sphere)
    # Point essentially on the unit sphere surface (noise-scale exterior).
    on_shell = np.array([1.0 + 1e-9, 0.0, 0.0])
    assert not opt._is_clearly_outside(on_shell)
    assert opt._is_clearly_outside(np.array([1.5, 0.0, 0.0]))


def test_forcing_stops_when_centering_error_below_fraction(unit_sphere, monkeypatch):
    graph = _chain_graph(
        [
            np.array([-0.5, 0.0, 0.0]),
            np.array([0.3, 0.0, 0.0]),
            np.array([0.5, 0.0, 0.0]),
        ]
    )
    opts = BasisOptimizerOptions(
        do_pruning=False,
        do_snapping=False,
        do_forcing=True,
        preserve_terminal_nodes=True,
        alpha_s=0.0,
        max_iterations=20,
        convergence_threshold=0.0,
        centering_error_stop_fraction=0.1,
        centering_error_plateau_tol=0.0,
        centering_error_plateau_patience=100,
        n_rays=6,
    )
    opt = BasisOptimizer(graph, unit_sphere, opts)
    call_count = {"n": 0}

    def _scripted_force(point, directions=None):
        call_count["n"] += 1
        # One free node per iteration → E equals this magnitude.
        # Iter 0: E0=1.0; later: E=0.05 < 0.1*E0 → stop on iter 1.
        if call_count["n"] == 1:
            return np.array([1.0, 0.0, 0.0])
        return np.array([0.05, 0.0, 0.0])

    monkeypatch.setattr(opt, "_compute_centering_force", _scripted_force)
    opt._run_forcing_phase()
    # First iteration sets E0; second iteration sees E < 0.1*E0 and stops.
    # Two iterations × one free node = 2 force evaluations.
    assert call_count["n"] == 2


def test_forcing_stops_on_centering_error_plateau(unit_sphere, monkeypatch):
    graph = _chain_graph(
        [
            np.array([-0.5, 0.0, 0.0]),
            np.array([0.3, 0.0, 0.0]),
            np.array([0.5, 0.0, 0.0]),
        ]
    )
    opts = BasisOptimizerOptions(
        do_pruning=False,
        do_snapping=False,
        do_forcing=True,
        preserve_terminal_nodes=True,
        alpha_s=0.0,
        max_iterations=20,
        convergence_threshold=0.0,
        centering_error_stop_fraction=0.0,
        centering_error_plateau_tol=1e-3,
        centering_error_plateau_patience=2,
        n_rays=6,
    )
    opt = BasisOptimizer(graph, unit_sphere, opts)
    call_count = {"n": 0}

    def _flat_force(point, directions=None):
        call_count["n"] += 1
        return np.array([1.0, 0.0, 0.0])

    monkeypatch.setattr(opt, "_compute_centering_force", _flat_force)
    opt._run_forcing_phase()
    # Iter 0: E0=1; iter 1: plateau streak=1; iter 2: streak=2 → stop.
    assert call_count["n"] == 3


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

    assert point_inside_mesh(unit_sphere, snapped)
    # Diameter along +x: hits ~(+1) then ~(-1); fraction 0.25 → x≈0.5.
    np.testing.assert_allclose(
        snapped,
        np.array([0.5, 0.0, 0.0]),
        atol=0.05,
    )
    assert opt.graph.get_outside_nodes(unit_sphere) == []


def test_snap_point_to_chord_midpoint_helper(unit_sphere):
    graph = _chain_graph([np.array([0.0, 0.0, 0.0])])
    opt = BasisOptimizer(
        graph, unit_sphere, BasisOptimizerOptions(snap_chord_fraction=0.5)
    )
    outside = np.array([1.5, 0.0, 0.0])
    mid = opt._snap_point_to_chord_midpoint(outside)
    assert mid is not None
    assert point_inside_mesh(unit_sphere, mid)
    np.testing.assert_allclose(mid, np.zeros(3), atol=0.05)


def test_snap_chord_fraction_default_quarter(unit_sphere):
    graph = _chain_graph([np.array([0.0, 0.0, 0.0])])
    opt = BasisOptimizer(graph, unit_sphere)
    assert opt.options.snap_chord_fraction == 0.25
    outside = np.array([1.5, 0.0, 0.0])
    pos = opt._snap_point_to_chord_midpoint(outside)
    assert pos is not None
    assert point_inside_mesh(unit_sphere, pos)
    np.testing.assert_allclose(pos, np.array([0.5, 0.0, 0.0]), atol=0.05)


def test_snap_ray_perturbation_recovers_even_hits_on_ts1():
    """Closest-point ray can hit a mesh edge (odd hits); snap should still recover."""
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
    # Mid-chord recovers a deep interior point after perturbation.
    opt = BasisOptimizer(
        basis,
        mm.mesh,
        BasisOptimizerOptions(do_forcing=False, snap_chord_fraction=0.5),
    )

    node = 14
    assert node in basis.get_outside_nodes(mm.mesh)
    pos = basis.get_node_position(node)
    direction, _ = opt._compute_snap_direction(pos)
    raw_hits = opt._ray_intersections_sorted(pos, direction)
    assert raw_hits.shape[0] % 2 == 1

    mid = opt._snap_point_to_chord_midpoint(pos)
    assert mid is not None
    assert point_inside_mesh(mm.mesh, mid)
    # Grazing even-hit chords (~1e-3) leave the midpoint on the surface;
    # a usable lumen chord should move it well off the boundary.
    assert abs(float(signed_distances(mm.mesh, mid)[0])) > 0.05


def test_snap_odd_hit_skips_short_pair_on_ts1():
    """Odd-hit rays with a short first chord should use a later interior pair."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    mesh_path = root / "data" / "mesh" / "processed" / "TS1.obj"
    skel_path = root / "data" / "mcf_skeletons" / "TS1_qst0.5_mcst5.polylines.txt"
    if not mesh_path.exists() or not skel_path.exists():
        pytest.skip("TS1 mesh/skeleton not available")

    from mascaf import MeshManager, SkeletonGraph

    mm = MeshManager(mesh_path=str(mesh_path))
    skeleton = SkeletonGraph.from_txt(str(skel_path))
    diag = mm.bounding_box_diagonal()
    basis = MorphologyGraph.from_skeleton_graph_resample(skeleton, int(diag * 0.07))
    opts = BasisOptimizerOptions(
        do_pruning=True,
        pruning_length_fraction=0.2,
        do_snapping=True,
        do_forcing=False,
    )
    opt = BasisOptimizer(basis, mm.mesh, opts)
    opt._run_pruning_phase()

    assert 2 in opt.graph.get_outside_nodes(mm.mesh)
    pos = opt.graph.get_node_position(2)
    direction, _ = opt._compute_snap_direction(pos)
    hits = opt._ray_intersections_sorted(pos, direction)
    assert hits.shape[0] == 3
    assert float(np.linalg.norm(hits[1] - hits[0])) < opt._snap_min_chord()

    mid = opt._snap_point_to_chord_midpoint(pos)
    assert mid is not None
    assert point_inside_mesh(mm.mesh, mid)

    opt._run_snapping_phase()
    assert opt.graph.get_outside_nodes(mm.mesh) == []


def test_options_no_longer_expose_snap_distance_multiplier():
    opts = BasisOptimizerOptions()
    assert not hasattr(opts, "snap_distance_multiplier")
    assert opts.snap_min_chord_fraction == 5e-4
    assert opts.snap_chord_fraction == 0.25


def _y_graph_short_and_long_stubs() -> MorphologyGraph:
    """Branch at origin: short stub along +y (len 0.2), long stub along +x (len 1.0).

    Nodes: 0 = left terminal, 1 = branch, 2 = short mid, 3 = short tip,
    4 = long mid, 5 = long tip.
    """
    graph = MorphologyGraph()
    positions = {
        0: np.array([-0.5, 0.0, 0.0]),
        1: np.array([0.0, 0.0, 0.0]),
        2: np.array([0.0, 0.1, 0.0]),
        3: np.array([0.0, 0.2, 0.0]),
        4: np.array([0.5, 0.0, 0.0]),
        5: np.array([1.0, 0.0, 0.0]),
    }
    for nid, pos in positions.items():
        graph.add_node(nid, xyz=pos, radius=0.05)
    for u, v in ((0, 1), (1, 2), (2, 3), (1, 4), (4, 5)):
        graph.add_edge(u, v)
    return graph


def test_pruning_absolute_removes_short_terminal_stub(unit_sphere):
    graph = _y_graph_short_and_long_stubs()
    opts = BasisOptimizerOptions(
        do_pruning=True,
        pruning_length=0.5,
        do_snapping=False,
        do_forcing=False,
        pruning_iterative=False,
    )
    optimized = BasisOptimizer(graph, unit_sphere, opts).optimize()
    assert 3 not in optimized
    assert 2 not in optimized
    assert 1 in optimized
    assert 5 in optimized
    assert 0 in optimized


def test_pruning_absolute_keeps_stub_above_threshold(unit_sphere):
    graph = _y_graph_short_and_long_stubs()
    opts = BasisOptimizerOptions(
        do_pruning=True,
        pruning_length=0.15,
        do_snapping=False,
        do_forcing=False,
        pruning_iterative=False,
    )
    optimized = BasisOptimizer(graph, unit_sphere, opts).optimize()
    # Short stub length is 0.2 > 0.15, so nothing removed.
    assert 3 in optimized
    assert 5 in optimized


def test_pruning_fraction_of_longest_stub(unit_sphere):
    graph = _y_graph_short_and_long_stubs()
    # Long stub = 1.0, short = 0.2; fraction 0.3 → threshold 0.3 → remove short only.
    opts = BasisOptimizerOptions(
        do_pruning=True,
        pruning_length_fraction=0.3,
        do_snapping=False,
        do_forcing=False,
        pruning_iterative=False,
    )
    opt = BasisOptimizer(graph, unit_sphere, opts)
    assert opt._resolve_pruning_threshold() == pytest.approx(0.3)
    optimized = opt.optimize()
    assert 3 not in optimized
    assert 5 in optimized
    assert 0 in optimized


def test_pruning_leaves_tip_to_tip_chain(unit_sphere):
    """Isolated tip↔tip paths are not pruned (no branch endpoint)."""
    graph = _chain_graph(
        [
            np.array([-0.5, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0]),
            np.array([0.5, 0.0, 0.0]),
        ]
    )
    opts = BasisOptimizerOptions(
        do_pruning=True,
        pruning_length=10.0,
        do_snapping=False,
        do_forcing=False,
    )
    optimized = BasisOptimizer(graph, unit_sphere, opts).optimize()
    assert optimized.number_of_nodes() == 3


def test_consolidate_nodes_degree_formula_and_midpoint():
    """deg-m and deg-n merge to deg (m+n-2) when there are no shared neighbors."""
    g = MorphologyGraph()
    # Branch of degree 3 at 0, branch of degree 3 at 1, connected by 0-1.
    # Neighbors of 0: 1,2,3  -> deg 3
    # Neighbors of 1: 0,4,5  -> deg 3
    # After merge keep=0: neighbors 2,3,4,5 -> deg 4 = 3+3-2
    positions = {
        0: np.array([0.0, 0.0, 0.0]),
        1: np.array([2.0, 0.0, 0.0]),
        2: np.array([-1.0, 1.0, 0.0]),
        3: np.array([-1.0, -1.0, 0.0]),
        4: np.array([3.0, 1.0, 0.0]),
        5: np.array([3.0, -1.0, 0.0]),
    }
    for nid, xyz in positions.items():
        g.add_node(nid, xyz=xyz, radius=0.1 if nid % 2 == 0 else 0.3)
    for u, v in [(0, 1), (0, 2), (0, 3), (1, 4), (1, 5)]:
        g.add_edge(u, v)

    assert g.degree(0) == 3
    assert g.degree(1) == 3
    keep = g.consolidate_nodes(0, 1)
    assert keep == 0
    assert 1 not in g
    assert g.degree(0) == 3 + 3 - 2
    np.testing.assert_allclose(g.get_node_position(0), [1.0, 0.0, 0.0])
    assert g.nodes[0]["radius"] == pytest.approx(0.2)
    assert set(g.neighbors(0)) == {2, 3, 4, 5}


def test_consolidate_nodes_shared_neighbor_no_multiedge():
    g = MorphologyGraph()
    for nid, xyz in {
        0: np.array([0.0, 0.0, 0.0]),
        1: np.array([1.0, 0.0, 0.0]),
        2: np.array([0.5, 1.0, 0.0]),
    }.items():
        g.add_node(nid, xyz=xyz, radius=0.05)
    g.add_edge(0, 1)
    g.add_edge(0, 2)
    g.add_edge(1, 2)
    keep = g.consolidate_nodes(0, 1)
    assert keep == 0
    assert g.degree(0) == 1
    assert set(g.neighbors(0)) == {2}


def test_bisect_edge_inserts_midpoint_degree_2():
    g = MorphologyGraph()
    g.add_node(0, xyz=np.array([0.0, 0.0, 0.0]), radius=0.1)
    g.add_node(1, xyz=np.array([2.0, 0.0, 0.0]), radius=0.3)
    g.add_edge(0, 1)
    mid = g.bisect_edge(0, 1)
    assert mid == 2
    assert g.degree(mid) == 2
    assert not g.has_edge(0, 1)
    assert g.has_edge(0, mid) and g.has_edge(mid, 1)
    np.testing.assert_allclose(g.get_node_position(mid), [1.0, 0.0, 0.0])
    assert g.nodes[mid]["radius"] == pytest.approx(0.2)


def test_forcing_active_resample_merges_short_edge(unit_sphere):
    """With active_resample on, a tiny edge is merged after a forcing iteration."""
    graph = _chain_graph(
        [
            np.array([-0.2, 0.0, 0.0]),
            np.array([-0.199, 0.0, 0.0]),  # ~0.001 apart
            np.array([0.2, 0.0, 0.0]),
        ]
    )
    diagonal = float(np.linalg.norm(np.asarray(unit_sphere.extents, dtype=float)))
    min_frac = 0.01
    max_frac = 1.0  # high max so only the short edge merges, no split after
    assert min_frac * diagonal > 0.001

    opts = BasisOptimizerOptions(
        do_pruning=False,
        do_snapping=False,
        do_forcing=True,
        active_resample=True,
        active_resample_min_fraction=min_frac,
        active_resample_max_fraction=max_frac,
        max_iterations=1,
        preserve_terminal_nodes=False,
        preserve_branch_nodes=False,
        alpha_s=0.0,
        step_scale=0.0,
        n_rays=6,
    )
    result = BasisOptimizer(graph, unit_sphere, opts).optimize()
    assert result.number_of_nodes() == 2
    assert result.number_of_edges() == 1
    assert 0 in result and 1 not in result and 2 in result
    assert result.has_edge(0, 2)


def test_forcing_active_resample_splits_long_edge(unit_sphere):
    """Edges longer than max fraction are bisected during active resample."""
    graph = _chain_graph(
        [
            np.array([-0.4, 0.0, 0.0]),
            np.array([0.4, 0.0, 0.0]),  # length 0.8
        ]
    )
    diagonal = float(np.linalg.norm(np.asarray(unit_sphere.extents, dtype=float)))
    # max ~ 0.2 on unit sphere diagonal ~ 2√3 ≈ 3.46 → fraction 0.05 → ~0.17
    min_frac = 0.02
    max_frac = 0.05
    assert 0.8 > max_frac * diagonal

    opts = BasisOptimizerOptions(
        do_pruning=False,
        do_snapping=False,
        do_forcing=True,
        active_resample=True,
        active_resample_min_fraction=min_frac,
        active_resample_max_fraction=max_frac,
        max_iterations=1,
        preserve_terminal_nodes=True,  # tips stay put; still allow bisect
        preserve_branch_nodes=False,
        alpha_s=0.0,
        step_scale=0.0,  # no movement; resample alone
        n_rays=6,
    )
    # step_scale=0 still runs force loop; nodes don't move
    result = BasisOptimizer(graph, unit_sphere, opts).optimize()
    assert result.number_of_nodes() > 2
    assert result.number_of_edges() == result.number_of_nodes() - 1
    for u, v in result.edges():
        length = float(
            np.linalg.norm(result.get_node_position(v) - result.get_node_position(u))
        )
        assert length <= max_frac * diagonal + 1e-9
        assert length >= min_frac * diagonal - 1e-9


def test_active_resample_requires_fractions(unit_sphere):
    graph = _chain_graph([np.array([0.0, 0.0, 0.0])])
    opts = BasisOptimizerOptions(
        do_pruning=False,
        do_snapping=False,
        do_forcing=True,
        active_resample=True,
        active_resample_min_fraction=None,
        active_resample_max_fraction=0.1,
        max_iterations=1,
    )
    with pytest.raises(ValueError, match="active_resample_min_fraction"):
        BasisOptimizer(graph, unit_sphere, opts).optimize()


def test_active_resample_rejects_max_below_twice_min(unit_sphere):
    graph = _chain_graph([np.array([0.0, 0.0, 0.0])])
    opts = BasisOptimizerOptions(
        do_pruning=False,
        do_snapping=False,
        do_forcing=True,
        active_resample=True,
        active_resample_min_fraction=0.05,
        active_resample_max_fraction=0.09,
        max_iterations=1,
    )
    with pytest.raises(ValueError, match="2 \\*"):
        BasisOptimizer(graph, unit_sphere, opts).optimize()


def test_active_resample_snaps_outside_bisect_midpoint(unit_sphere, monkeypatch):
    """Bisect midpoints that land outside are chord-snapped before continuing."""
    graph = _chain_graph(
        [
            np.array([-0.4, 0.0, 0.0]),
            np.array([0.4, 0.0, 0.0]),
        ]
    )
    opts = BasisOptimizerOptions(
        do_pruning=False,
        do_snapping=False,
        do_forcing=False,
        active_resample=True,
        active_resample_min_fraction=0.02,
        active_resample_max_fraction=0.05,
    )
    opt = BasisOptimizer(graph, unit_sphere, opts)
    snap_calls: list[np.ndarray] = []
    snap_target = np.array([0.1, 0.0, 0.0], dtype=float)

    def is_outside(point: np.ndarray) -> bool:
        # Geometric midpoints near the origin count as outside; snap target inside.
        if np.allclose(point, snap_target):
            return False
        return float(np.linalg.norm(point)) < 0.05

    def fake_snap(point: np.ndarray) -> np.ndarray:
        snap_calls.append(np.asarray(point, dtype=float).copy())
        return snap_target.copy()

    monkeypatch.setattr(opt, "_is_clearly_outside", is_outside)
    monkeypatch.setattr(opt, "_snap_point_to_chord_midpoint", fake_snap)

    n_merged, n_split = opt._active_resample_edges()
    assert n_split >= 1
    assert snap_calls, "expected snap of an outside bisect midpoint"
    positions = [opt.graph.get_node_position(n) for n in opt.graph.nodes()]
    assert any(np.allclose(p, snap_target) for p in positions)


def test_consolidation_changes_cycle_count_triangle():
    g = MorphologyGraph()
    for i, xyz in enumerate(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 0.866, 0.0],
        ]
    ):
        g.add_node(i, xyz=np.array(xyz), radius=0.05)
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    g.add_edge(2, 0)
    assert g.cyclomatic_number() == 1
    assert g.consolidation_changes_cycle_count(0, 1)
    assert g.consolidation_changes_cycle_count(1, 2)
    assert g.consolidation_changes_cycle_count(2, 0)


def test_consolidation_preserves_cycle_count_on_four_cycle():
    g = MorphologyGraph()
    for i, xyz in enumerate(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    ):
        g.add_node(i, xyz=np.array(xyz), radius=0.05)
    for u, v in [(0, 1), (1, 2), (2, 3), (3, 0)]:
        g.add_edge(u, v)
    assert g.cyclomatic_number() == 1
    assert not g.consolidation_changes_cycle_count(0, 1)


def test_active_resample_skips_cycle_collapsing_merge(unit_sphere, caplog):
    """Short edge on a triangle must not merge when cycle collapse is disallowed."""
    g = MorphologyGraph()
    positions = [
        np.array([0.0, 0.0, 0.0]),
        np.array([0.001, 0.0, 0.0]),
        np.array([0.0005, 0.001, 0.0]),
    ]
    for i, pos in enumerate(positions):
        g.add_node(i, xyz=pos, radius=0.05)
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    g.add_edge(2, 0)

    opts = BasisOptimizerOptions(
        do_pruning=False,
        do_snapping=False,
        do_forcing=False,
        active_resample=True,
        active_resample_min_fraction=0.05,
        active_resample_max_fraction=1.0,
        active_resample_allow_cycle_collapse=False,
    )
    opt = BasisOptimizer(g, unit_sphere, opts)
    with caplog.at_level(logging.WARNING):
        n_merged, n_split = opt._active_resample_edges()
    assert n_merged == 0
    assert opt.graph.number_of_nodes() == 3
    assert any("Skipping active resample consolidation" in r.message for r in caplog.records)


def test_active_resample_allows_cycle_collapsing_merge(unit_sphere):
    g = MorphologyGraph()
    positions = [
        np.array([0.0, 0.0, 0.0]),
        np.array([0.001, 0.0, 0.0]),
        np.array([0.0005, 0.001, 0.0]),
    ]
    for i, pos in enumerate(positions):
        g.add_node(i, xyz=pos, radius=0.05)
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    g.add_edge(2, 0)

    opts = BasisOptimizerOptions(
        do_pruning=False,
        do_snapping=False,
        do_forcing=False,
        active_resample=True,
        active_resample_min_fraction=0.05,
        active_resample_max_fraction=1.0,
        active_resample_allow_cycle_collapse=True,
    )
    opt = BasisOptimizer(g, unit_sphere, opts)
    n_merged, _ = opt._active_resample_edges()
    assert n_merged >= 1
    assert opt.graph.cyclomatic_number() == 0


def test_active_resample_raises_if_snap_fails(unit_sphere, monkeypatch):
    graph = _chain_graph(
        [
            np.array([-0.4, 0.0, 0.0]),
            np.array([0.4, 0.0, 0.0]),
        ]
    )
    opts = BasisOptimizerOptions(
        do_pruning=False,
        do_snapping=False,
        do_forcing=False,
        active_resample=True,
        active_resample_min_fraction=0.02,
        active_resample_max_fraction=0.05,
    )
    opt = BasisOptimizer(graph, unit_sphere, opts)
    monkeypatch.setattr(opt, "_is_clearly_outside", lambda _p: True)
    monkeypatch.setattr(opt, "_snap_point_to_chord_midpoint", lambda _p: None)
    with pytest.raises(RuntimeError, match="snapping failed"):
        opt._active_resample_edges()
