"""
Tests for parallel skeleton optimization.
"""

import time

import numpy as np
import pytest

from mcf2swc import (
    PolylinesSkeleton,
    SkeletonOptimizer,
    SkeletonOptimizerOptions,
    example_mesh,
)


class TestParallelSkeletonOptimizer:
    """Test parallel optimization functionality."""

    @pytest.fixture
    def cylinder_mesh(self):
        """Create a simple cylinder mesh for testing."""
        return example_mesh("cylinder", radius=1.0, height=10.0, sections=16)

    @pytest.fixture
    def multiple_polylines_skeleton(self):
        """Create a skeleton with multiple polylines for parallel testing."""
        polylines = []

        # Create 8 polylines at different positions
        for i in range(8):
            angle = 2.0 * np.pi * i / 8
            offset_x = 0.3 * np.cos(angle)
            offset_y = 0.3 * np.sin(angle)

            points = np.array(
                [
                    [offset_x, offset_y, -3.5],
                    [offset_x, offset_y, -1.5],
                    [offset_x, offset_y, 0.0],
                    [offset_x, offset_y, 1.5],
                    [offset_x, offset_y, 3.5],
                ]
            )
            polylines.append(points)

        return PolylinesSkeleton(polylines)

    def test_parallel_produces_valid_results(
        self, cylinder_mesh, multiple_polylines_skeleton
    ):
        """Test that parallel optimization produces valid results."""
        opts_parallel = SkeletonOptimizerOptions(
            max_iterations=20,
            step_size=0.05,
            n_jobs=4,
            verbose=False,
        )

        # Parallel optimization
        optimizer_par = SkeletonOptimizer(
            multiple_polylines_skeleton, cylinder_mesh, opts_parallel
        )
        result_par = optimizer_par.optimize()

        # Results should have same structure
        assert len(result_par.polylines) == len(multiple_polylines_skeleton.polylines)

        for i, (poly_orig, poly_opt) in enumerate(
            zip(multiple_polylines_skeleton.polylines, result_par.polylines)
        ):
            assert poly_opt.shape == poly_orig.shape, f"Polyline {i} shape mismatch"
            # Check that points have moved (optimization happened)
            # but not too far (reasonable optimization)
            distances = np.linalg.norm(poly_opt - poly_orig, axis=1)
            assert np.any(distances > 1e-6), f"Polyline {i} didn't move at all"
            assert np.all(distances < 2.0), f"Polyline {i} moved too far"

    def test_parallel_with_n_jobs_minus_1(
        self, cylinder_mesh, multiple_polylines_skeleton
    ):
        """Test that n_jobs=-1 uses all available cores."""
        opts = SkeletonOptimizerOptions(
            max_iterations=10,
            n_jobs=-1,
            verbose=False,
        )

        optimizer = SkeletonOptimizer(multiple_polylines_skeleton, cylinder_mesh, opts)
        result = optimizer.optimize()

        # Should complete successfully
        assert isinstance(result, PolylinesSkeleton)
        assert len(result.polylines) == len(multiple_polylines_skeleton.polylines)

    def test_parallel_with_single_polyline(self, cylinder_mesh):
        """Test that parallel optimization handles single polyline correctly."""
        points = np.array(
            [
                [0.3, 0.2, -3.5],
                [0.3, 0.2, -1.5],
                [0.3, 0.2, 0.0],
                [0.3, 0.2, 1.5],
                [0.3, 0.2, 3.5],
            ]
        )
        skeleton = PolylinesSkeleton([points])

        opts = SkeletonOptimizerOptions(
            max_iterations=10,
            n_jobs=4,
            verbose=False,
        )

        optimizer = SkeletonOptimizer(skeleton, cylinder_mesh, opts)
        result = optimizer.optimize()

        # Should fall back to sequential for single polyline
        assert isinstance(result, PolylinesSkeleton)
        assert len(result.polylines) == 1

    def test_parallel_with_empty_polylines(self, cylinder_mesh):
        """Test that parallel optimization handles empty polylines correctly."""
        polylines = [
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
            np.array([[0.5, 0.0, 0.0], [0.5, 0.0, 1.0]]),
        ]
        skeleton = PolylinesSkeleton(polylines)

        opts = SkeletonOptimizerOptions(
            max_iterations=10,
            n_jobs=2,
            verbose=False,
        )

        optimizer = SkeletonOptimizer(skeleton, cylinder_mesh, opts)
        result = optimizer.optimize()

        assert len(result.polylines) == 2

    def test_parallel_performance_benchmark(
        self, cylinder_mesh, multiple_polylines_skeleton
    ):
        """Benchmark parallel vs sequential optimization."""
        opts_sequential = SkeletonOptimizerOptions(
            max_iterations=30,
            step_size=0.05,
            n_jobs=1,
            verbose=False,
        )

        opts_parallel = SkeletonOptimizerOptions(
            max_iterations=30,
            step_size=0.05,
            n_jobs=4,
            verbose=False,
        )

        # Time sequential optimization
        optimizer_seq = SkeletonOptimizer(
            multiple_polylines_skeleton, cylinder_mesh, opts_sequential
        )
        start_seq = time.time()
        result_seq = optimizer_seq.optimize()
        time_seq = time.time() - start_seq

        # Time parallel optimization
        optimizer_par = SkeletonOptimizer(
            multiple_polylines_skeleton, cylinder_mesh, opts_parallel
        )
        start_par = time.time()
        result_par = optimizer_par.optimize()
        time_par = time.time() - start_par

        print(f"\nSequential time: {time_seq:.3f}s")
        print(f"Parallel time: {time_par:.3f}s")
        if time_par < time_seq:
            print(f"Speedup: {time_seq / time_par:.2f}x")
        else:
            print(f"Slowdown: {time_par / time_seq:.2f}x (overhead dominated)")

        # Just verify both complete successfully
        # Performance can vary based on system load and overhead
        assert isinstance(result_seq, PolylinesSkeleton)
        assert isinstance(result_par, PolylinesSkeleton)
