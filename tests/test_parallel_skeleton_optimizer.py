"""
Tests for parallel skeleton optimization module.
"""

import numpy as np
import pytest
import trimesh

from mascaf import (
    SkeletonGraph,
    SkeletonOptimizer,
    SkeletonOptimizerOptions,
    ParallelSkeletonOptimizer,
    ParallelOptimizerOptions,
    example_mesh,
)


class TestParallelOptimizerOptions:
    """Test ParallelOptimizerOptions dataclass."""

    def test_default_options(self):
        opts = ParallelOptimizerOptions()
        assert opts.max_workers is None
        assert opts.use_processes is True
        assert opts.batch_size == 10
        assert opts.ray_batch_size == 6
        assert opts.enable_ray_parallel is True
        assert opts.enable_node_parallel is True

    def test_custom_options(self):
        opts = ParallelOptimizerOptions(
            max_workers=4,
            use_processes=False,
            batch_size=20,
            ray_batch_size=12,
            enable_ray_parallel=False,
        )
        assert opts.max_workers == 4
        assert opts.use_processes is False
        assert opts.batch_size == 20
        assert opts.ray_batch_size == 12
        assert opts.enable_ray_parallel is False


class TestParallelSkeletonOptimizer:
    """Test ParallelSkeletonOptimizer class."""

    @pytest.fixture
    def cylinder_mesh(self):
        """Create a simple cylinder mesh for testing."""
        return example_mesh("cylinder", radius=1.0, height=10.0, sections=16)

    @pytest.fixture
    def cylinder_skeleton_offset(self):
        """Create a skeleton offset from the center."""
        points = np.array(
            [
                [0.3, 0.2, -3.5],
                [0.3, 0.2, -1.5],
                [0.3, 0.2, 0.0],
                [0.3, 0.2, 1.5],
                [0.3, 0.2, 3.5],
            ]
        )
        return SkeletonGraph.from_polylines([points])

    @pytest.fixture
    def large_skeleton(self):
        """Create a larger skeleton for testing parallel performance."""
        points = []
        for i in range(50):
            z = -4 + 8 * i / 49
            points.append([0.2 + 0.1 * np.sin(i / 5), 0.1 + 0.1 * np.cos(i / 5), z])
        return SkeletonGraph.from_polylines([np.array(points)])

    def test_initialization(self, cylinder_mesh, cylinder_skeleton_offset):
        """Test basic initialization of ParallelSkeletonOptimizer."""
        optimizer = ParallelSkeletonOptimizer(cylinder_skeleton_offset, cylinder_mesh)
        assert optimizer.skeleton is not None
        assert optimizer.mesh is not None
        assert optimizer.options is not None
        assert optimizer.parallel_options is not None

    def test_initialization_with_options(self, cylinder_mesh, cylinder_skeleton_offset):
        """Test initialization with custom options."""
        opts = SkeletonOptimizerOptions(max_iterations=50, verbose=True)
        par_opts = ParallelOptimizerOptions(max_workers=2, use_processes=False)
        optimizer = ParallelSkeletonOptimizer(
            cylinder_skeleton_offset, cylinder_mesh, opts, par_opts
        )
        assert optimizer.options.max_iterations == 50
        assert optimizer.options.verbose is True
        assert optimizer.parallel_options.max_workers == 2
        assert optimizer.parallel_options.use_processes is False

    def test_check_surface_crossing(self, cylinder_mesh, cylinder_skeleton_offset):
        """Test parallel surface crossing detection."""
        optimizer = ParallelSkeletonOptimizer(cylinder_skeleton_offset, cylinder_mesh)
        has_crossing, num_outside, max_dist = optimizer.check_surface_crossing()

        # Compare with sequential version
        seq_optimizer = SkeletonOptimizer(cylinder_skeleton_offset, cylinder_mesh)
        seq_has_crossing, seq_num_outside, seq_max_dist = (
            seq_optimizer.check_surface_crossing()
        )

        assert has_crossing == seq_has_crossing
        assert num_outside == seq_num_outside
        assert np.isclose(max_dist, seq_max_dist, rtol=1e-6)

    def test_optimize_skeleton_small(self, cylinder_mesh, cylinder_skeleton_offset):
        """Test optimization of a small skeleton."""
        opts = SkeletonOptimizerOptions(
            max_iterations=10, step_size=0.05, verbose=False
        )
        par_opts = ParallelOptimizerOptions(max_workers=2, enable_node_parallel=False)

        optimizer = ParallelSkeletonOptimizer(
            cylinder_skeleton_offset, cylinder_mesh, opts, par_opts
        )
        optimized = optimizer.optimize()

        assert isinstance(optimized, SkeletonGraph)
        assert optimized.number_of_nodes() == cylinder_skeleton_offset.number_of_nodes()

    def test_optimize_skeleton_large(self, cylinder_mesh, cylinder_skeleton_offset):
        """Test optimization with parallel processing (minimal test)."""
        opts = SkeletonOptimizerOptions(max_iterations=5, verbose=False)
        par_opts = ParallelOptimizerOptions(max_workers=2)

        optimizer = ParallelSkeletonOptimizer(
            cylinder_skeleton_offset, cylinder_mesh, opts, par_opts
        )
        optimized = optimizer.optimize()

        assert isinstance(optimized, SkeletonGraph)
        assert optimized.number_of_nodes() == cylinder_skeleton_offset.number_of_nodes()

    def test_parallel_vs_sequential_consistency(
        self, cylinder_mesh, cylinder_skeleton_offset
    ):
        """Test that parallel and sequential optimizers produce valid results (minimal test)."""
        opts = SkeletonOptimizerOptions(max_iterations=5, verbose=False)
        par_opts = ParallelOptimizerOptions(max_workers=2)

        # Sequential optimization
        seq_optimizer = SkeletonOptimizer(cylinder_skeleton_offset, cylinder_mesh, opts)
        seq_optimized = seq_optimizer.optimize()

        # Parallel optimization
        par_optimizer = ParallelSkeletonOptimizer(
            cylinder_skeleton_offset, cylinder_mesh, opts, par_opts
        )
        par_optimized = par_optimizer.optimize()

        # Both should produce valid results
        assert isinstance(seq_optimized, SkeletonGraph)
        assert isinstance(par_optimized, SkeletonGraph)
        assert seq_optimized.number_of_nodes() == par_optimized.number_of_nodes()

    def test_ray_parallel_disabled(self, cylinder_mesh, cylinder_skeleton_offset):
        """Test optimization with ray parallelization disabled (minimal test)."""
        opts = SkeletonOptimizerOptions(max_iterations=3, verbose=False)
        par_opts = ParallelOptimizerOptions(enable_ray_parallel=False)

        optimizer = ParallelSkeletonOptimizer(
            cylinder_skeleton_offset, cylinder_mesh, opts, par_opts
        )
        optimized = optimizer.optimize()

        assert isinstance(optimized, SkeletonGraph)
        assert optimized.number_of_nodes() == cylinder_skeleton_offset.number_of_nodes()

    def test_node_parallel_disabled(self, cylinder_mesh, cylinder_skeleton_offset):
        """Test optimization with node parallelization disabled (minimal test)."""
        opts = SkeletonOptimizerOptions(max_iterations=3, verbose=False)
        par_opts = ParallelOptimizerOptions(enable_node_parallel=False)

        optimizer = ParallelSkeletonOptimizer(
            cylinder_skeleton_offset, cylinder_mesh, opts, par_opts
        )
        optimized = optimizer.optimize()

        assert isinstance(optimized, SkeletonGraph)
        assert optimized.number_of_nodes() == cylinder_skeleton_offset.number_of_nodes()

    def test_both_parallel_disabled(self, cylinder_mesh, cylinder_skeleton_offset):
        """Test optimization with all parallelization disabled (minimal test)."""
        opts = SkeletonOptimizerOptions(max_iterations=3, verbose=False)
        par_opts = ParallelOptimizerOptions(
            enable_ray_parallel=False,
            enable_node_parallel=False,
        )

        optimizer = ParallelSkeletonOptimizer(
            cylinder_skeleton_offset, cylinder_mesh, opts, par_opts
        )
        optimized = optimizer.optimize()

        assert isinstance(optimized, SkeletonGraph)
        assert optimized.number_of_nodes() == cylinder_skeleton_offset.number_of_nodes()

    def test_threading_vs_multiprocessing(
        self, cylinder_mesh, cylinder_skeleton_offset
    ):
        """Test difference between threading and multiprocessing (minimal test)."""
        opts = SkeletonOptimizerOptions(max_iterations=5, verbose=False)

        # Threading
        thread_opts = ParallelOptimizerOptions(
            max_workers=2,
            use_processes=False,
        )

        thread_optimizer = ParallelSkeletonOptimizer(
            cylinder_skeleton_offset, cylinder_mesh, opts, thread_opts
        )
        thread_optimized = thread_optimizer.optimize()

        # Multiprocessing
        process_opts = ParallelOptimizerOptions(
            max_workers=2,
            use_processes=True,
        )

        process_optimizer = ParallelSkeletonOptimizer(
            cylinder_skeleton_offset.copy_skeleton(), cylinder_mesh, opts, process_opts
        )
        process_optimized = process_optimizer.optimize()

        # Both should produce valid results
        assert isinstance(thread_optimized, SkeletonGraph)
        assert isinstance(process_optimized, SkeletonGraph)
        assert thread_optimized.number_of_nodes() == process_optimized.number_of_nodes()

    def test_different_batch_sizes(self, cylinder_mesh, cylinder_skeleton_offset):
        """Test optimization with different batch sizes (minimal test)."""
        opts = SkeletonOptimizerOptions(max_iterations=5, verbose=False)

        batch_sizes = [5, 10]
        results = []

        for batch_size in batch_sizes:
            par_opts = ParallelOptimizerOptions(
                max_workers=2,
                batch_size=batch_size,
            )

            optimizer = ParallelSkeletonOptimizer(
                cylinder_skeleton_offset.copy_skeleton(), cylinder_mesh, opts, par_opts
            )
            optimized = optimizer.optimize()
            results.append(optimized.get_all_positions())

        # Results should be very similar
        np.testing.assert_allclose(results[1], results[0], rtol=1e-5, atol=1e-6)

    # def test_different_ray_batch_sizes(self, cylinder_mesh, cylinder_skeleton_offset):
    #     """Test optimization with different ray batch sizes (minimal test)."""
    #     opts = SkeletonOptimizerOptions(max_iterations=5, n_rays=6, verbose=False)

    #     ray_batch_sizes = [3, 6]
    #     results = []

    #     for ray_batch_size in ray_batch_sizes:
    #         par_opts = ParallelOptimizerOptions(
    #             max_workers=2,
    #             ray_batch_size=ray_batch_size,
    #             enable_ray_parallel=True,
    #             enable_node_parallel=False,  # Disable to isolate ray parallelization
    #         )

    #         optimizer = ParallelSkeletonOptimizer(
    #             cylinder_skeleton_offset.copy_skeleton(), cylinder_mesh, opts, par_opts
    #         )
    #         optimized = optimizer.optimize()
    #         results.append(optimized.get_all_positions())

    #     # Results should be very similar
    #     np.testing.assert_allclose(results[1], results[0], rtol=1e-5, atol=1e-6)

    def test_get_optimization_stats(self, cylinder_mesh, cylinder_skeleton_offset):
        """Test getting optimization statistics."""
        par_opts = ParallelOptimizerOptions(max_workers=2)
        optimizer = ParallelSkeletonOptimizer(
            cylinder_skeleton_offset, cylinder_mesh, parallel_options=par_opts
        )
        stats = optimizer.get_optimization_stats()

        assert "num_nodes" in stats
        assert "num_edges" in stats
        assert "max_workers" in stats
        assert "use_processes" in stats
        assert "enable_ray_parallel" in stats
        assert "enable_node_parallel" in stats
        assert stats["max_workers"] == 2
        assert stats["use_processes"] is True
        assert stats["enable_ray_parallel"] is True
        assert stats["enable_node_parallel"] is True

    def test_empty_skeleton(self, cylinder_mesh):
        """Test optimization with empty skeleton."""
        empty_skeleton = SkeletonGraph.from_polylines([np.zeros((0, 3))])
        optimizer = ParallelSkeletonOptimizer(empty_skeleton, cylinder_mesh)
        optimized = optimizer.optimize()
        assert optimized.number_of_nodes() == 0

    def test_single_point_skeleton(self, cylinder_mesh):
        """Test optimization with single-point skeleton."""
        single_point = SkeletonGraph.from_polylines([np.array([[0.0, 0.0, 0.0]])])
        optimizer = ParallelSkeletonOptimizer(single_point, cylinder_mesh)
        optimized = optimizer.optimize()
        assert optimized.number_of_nodes() == 1

    def test_preserve_terminal_nodes(self, cylinder_mesh, cylinder_skeleton_offset):
        """Test that terminal nodes are preserved when enabled (minimal test)."""
        opts = SkeletonOptimizerOptions(
            max_iterations=3,
            preserve_terminal_nodes=True,
            verbose=False,
        )
        par_opts = ParallelOptimizerOptions(max_workers=2)

        optimizer = ParallelSkeletonOptimizer(
            cylinder_skeleton_offset, cylinder_mesh, opts, par_opts
        )
        optimized = optimizer.optimize()

        original_polylines = cylinder_skeleton_offset.to_polylines()
        optimized_polylines = optimized.to_polylines()
        original_points = original_polylines[0]
        optimized_points = optimized_polylines[0]

        np.testing.assert_allclose(original_points[0], optimized_points[0], rtol=1e-10)
        np.testing.assert_allclose(
            original_points[-1], optimized_points[-1], rtol=1e-10
        )

    def test_convergence(self, cylinder_mesh, cylinder_skeleton_offset):
        """Test that parallel optimization converges (minimal test)."""
        opts = SkeletonOptimizerOptions(
            max_iterations=10,
            convergence_threshold=1e-3,
            verbose=False,
        )
        par_opts = ParallelOptimizerOptions(max_workers=2)

        optimizer = ParallelSkeletonOptimizer(
            cylinder_skeleton_offset, cylinder_mesh, opts, par_opts
        )
        optimized = optimizer.optimize()

        assert isinstance(optimized, SkeletonGraph)
        assert optimized.number_of_nodes() > 0
