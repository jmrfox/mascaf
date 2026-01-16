"""
Benchmark script for parallel skeleton optimization.

This script demonstrates the performance improvement from parallel processing
on a realistic workload with many polylines.
"""

import time

import numpy as np

from mcf2swc import (
    PolylinesSkeleton,
    SkeletonOptimizer,
    SkeletonOptimizerOptions,
    example_mesh,
)


def create_complex_skeleton(n_polylines: int = 20, n_points_per_polyline: int = 10):
    """Create a complex skeleton with many polylines."""
    polylines = []

    for i in range(n_polylines):
        # Create polylines at different radial positions
        angle = 2.0 * np.pi * i / n_polylines
        radius = 0.3 + 0.2 * (i % 3) / 3.0

        offset_x = radius * np.cos(angle)
        offset_y = radius * np.sin(angle)

        # Create points along z-axis with some variation
        points = []
        z_range = np.linspace(-4.0, 4.0, n_points_per_polyline)
        for z in z_range:
            # Add small random perturbations
            x = offset_x + 0.05 * np.sin(z)
            y = offset_y + 0.05 * np.cos(z)
            points.append([x, y, z])

        polylines.append(np.array(points))

    return PolylinesSkeleton(polylines)


def benchmark_optimization(n_polylines: int = 20, max_iterations: int = 50):
    """Benchmark sequential vs parallel optimization."""
    print(f"\n{'='*70}")
    print(f"Benchmark: {n_polylines} polylines, {max_iterations} iterations")
    print(f"{'='*70}\n")

    # Create test data
    mesh = example_mesh("cylinder", radius=1.0, height=10.0, sections=32)
    skeleton = create_complex_skeleton(n_polylines=n_polylines)

    print(
        f"Skeleton: {len(skeleton.polylines)} polylines, "
        f"{skeleton.total_points()} total points"
    )
    print(f"Mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces\n")

    # Sequential optimization
    print("Running sequential optimization...")
    opts_seq = SkeletonOptimizerOptions(
        max_iterations=max_iterations,
        step_size=0.05,
        n_jobs=1,
        verbose=False,
    )

    optimizer_seq = SkeletonOptimizer(skeleton, mesh, opts_seq)
    start = time.time()
    result_seq = optimizer_seq.optimize()
    time_seq = time.time() - start

    print(f"  Time: {time_seq:.3f}s\n")

    # Parallel optimization with 2 workers
    print("Running parallel optimization (2 workers)...")
    opts_par2 = SkeletonOptimizerOptions(
        max_iterations=max_iterations,
        step_size=0.05,
        n_jobs=2,
        verbose=False,
    )

    optimizer_par2 = SkeletonOptimizer(skeleton, mesh, opts_par2)
    start = time.time()
    result_par2 = optimizer_par2.optimize()
    time_par2 = time.time() - start

    print(f"  Time: {time_par2:.3f}s")
    print(f"  Speedup: {time_seq / time_par2:.2f}x\n")

    # Parallel optimization with 4 workers
    print("Running parallel optimization (4 workers)...")
    opts_par4 = SkeletonOptimizerOptions(
        max_iterations=max_iterations,
        step_size=0.05,
        n_jobs=4,
        verbose=False,
    )

    optimizer_par4 = SkeletonOptimizer(skeleton, mesh, opts_par4)
    start = time.time()
    result_par4 = optimizer_par4.optimize()
    time_par4 = time.time() - start

    print(f"  Time: {time_par4:.3f}s")
    print(f"  Speedup: {time_seq / time_par4:.2f}x\n")

    # Parallel optimization with all cores
    print("Running parallel optimization (all cores)...")
    opts_par_all = SkeletonOptimizerOptions(
        max_iterations=max_iterations,
        step_size=0.05,
        n_jobs=-1,
        verbose=False,
    )

    optimizer_par_all = SkeletonOptimizer(skeleton, mesh, opts_par_all)
    start = time.time()
    result_par_all = optimizer_par_all.optimize()
    time_par_all = time.time() - start

    print(f"  Time: {time_par_all:.3f}s")
    print(f"  Speedup: {time_seq / time_par_all:.2f}x\n")

    # Summary
    print(f"{'='*70}")
    print("Summary:")
    print(f"  Sequential:        {time_seq:.3f}s  (baseline)")
    print(f"  Parallel (2 jobs): {time_par2:.3f}s  ({time_seq/time_par2:.2f}x speedup)")
    print(f"  Parallel (4 jobs): {time_par4:.3f}s  ({time_seq/time_par4:.2f}x speedup)")
    print(
        f"  Parallel (all):    {time_par_all:.3f}s  ({time_seq/time_par_all:.2f}x speedup)"
    )
    print(f"{'='*70}\n")


if __name__ == "__main__":
    # Run benchmarks with different workload sizes
    benchmark_optimization(n_polylines=10, max_iterations=30)
    benchmark_optimization(n_polylines=20, max_iterations=50)
    benchmark_optimization(n_polylines=40, max_iterations=50)
