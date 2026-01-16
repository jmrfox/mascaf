"""
Benchmark script to compare sequential vs parallel skeleton optimizer performance.

This script creates test scenarios and measures the performance improvement
when using parallel processing for skeleton optimization.
"""

import time
import numpy as np
import multiprocessing as mp
from typing import List, Dict, Tuple
import argparse
import json

from mcf2swc import (
    SkeletonGraph,
    SkeletonOptimizer,
    SkeletonOptimizerOptions,
    ParallelSkeletonOptimizer,
    ParallelOptimizerOptions,
    example_mesh,
)


def create_test_skeletons(mesh, sizes: List[int]) -> List[SkeletonGraph]:
    """Create test skeletons of different sizes."""
    skeletons = []

    for size in sizes:
        # Create a more complex skeleton with multiple branches
        points = []

        # Main backbone
        main_backbone = np.linspace(-5, 5, size // 3)
        for z in main_backbone:
            points.append([0.2, 0.1, z])  # Slightly offset from center

        # Add branches
        n_branches = min(3, size // 10)
        for i in range(n_branches):
            branch_start = main_backbone[
                len(main_backbone) // (n_branches + 1) * (i + 1)
            ]
            branch_points = np.linspace(0, 2, size // (3 * n_branches))
            for r in branch_points:
                angle = 2 * np.pi * i / n_branches
                points.append(
                    [r * np.cos(angle) + 0.2, r * np.sin(angle) + 0.1, branch_start]
                )

        skeleton_points = np.array(points)
        skeleton = SkeletonGraph.from_polylines([skeleton_points])
        skeletons.append(skeleton)

    return skeletons


def benchmark_sequential_optimizer(
    skeleton: SkeletonGraph, mesh, options: SkeletonOptimizerOptions
) -> Tuple[float, Dict]:
    """Benchmark the sequential skeleton optimizer."""
    start_time = time.time()

    optimizer = SkeletonOptimizer(skeleton, mesh, options)
    optimized = optimizer.optimize()

    end_time = time.time()
    elapsed_time = end_time - start_time

    stats = optimizer.get_optimization_stats()
    return elapsed_time, stats


def benchmark_parallel_optimizer(
    skeleton: SkeletonGraph,
    mesh,
    options: SkeletonOptimizerOptions,
    parallel_options: ParallelOptimizerOptions,
) -> Tuple[float, Dict]:
    """Benchmark the parallel skeleton optimizer."""
    start_time = time.time()

    optimizer = ParallelSkeletonOptimizer(skeleton, mesh, options, parallel_options)
    optimized = optimizer.optimize()

    end_time = time.time()
    elapsed_time = end_time - start_time

    stats = optimizer.get_optimization_stats()
    return elapsed_time, stats


def run_comprehensive_benchmark():
    """Run a comprehensive benchmark comparing different configurations."""
    print("=" * 80)
    print("Parallel Skeleton Optimizer Benchmark")
    print("=" * 80)

    # Test parameters
    mesh_sizes = ["cylinder", "torus"]
    skeleton_sizes = [50, 100, 200, 500]
    worker_counts = [1, 2, 4, mp.cpu_count()]

    # Create test mesh
    mesh = example_mesh("cylinder", radius=1.5, height=12.0, sections=32)
    print(f"Using test mesh: cylinder (vertices: {len(mesh.vertices)})")

    # Optimization options
    base_options = SkeletonOptimizerOptions(
        max_iterations=50,
        step_size=0.1,
        convergence_threshold=1e-4,
        n_rays=6,
        verbose=False,
    )

    # Create test skeletons
    skeletons = create_test_skeletons(mesh, skeleton_sizes)

    results = []

    for i, (skeleton, size) in enumerate(zip(skeletons, skeleton_sizes)):
        print(f"\n{'-' * 60}")
        print(f"Testing skeleton with {size} nodes")
        print(f"{'-' * 60}")

        # Baseline sequential
        seq_time, seq_stats = benchmark_sequential_optimizer(
            skeleton.copy_skeleton(), mesh, base_options
        )
        print(f"Sequential optimizer: {seq_time:.3f}s")

        # Test different parallel configurations
        for workers in worker_counts:
            # Test with threading
            parallel_options_thread = ParallelOptimizerOptions(
                max_workers=workers,
                use_processes=False,
                enable_ray_parallel=True,
                enable_node_parallel=True,
            )

            thread_time, thread_stats = benchmark_parallel_optimizer(
                skeleton.copy_skeleton(), mesh, base_options, parallel_options_thread
            )
            thread_speedup = seq_time / thread_time if thread_time > 0 else float("inf")

            # Test with multiprocessing
            parallel_options_process = ParallelOptimizerOptions(
                max_workers=workers,
                use_processes=True,
                enable_ray_parallel=True,
                enable_node_parallel=True,
            )

            process_time, process_stats = benchmark_parallel_optimizer(
                skeleton.copy_skeleton(), mesh, base_options, parallel_options_process
            )
            process_speedup = (
                seq_time / process_time if process_time > 0 else float("inf")
            )

            result = {
                "skeleton_size": size,
                "workers": workers,
                "sequential_time": seq_time,
                "thread_time": thread_time,
                "thread_speedup": thread_speedup,
                "process_time": process_time,
                "process_speedup": process_speedup,
            }
            results.append(result)

            print(
                f"  {workers:2d} workers (threads): {thread_time:.3f}s ({thread_speedup:.2f}x speedup)"
            )
            print(
                f"  {workers:2d} workers (processes): {process_time:.3f}s ({process_speedup:.2f}x speedup)"
            )

    # Summary
    print(f"\n{'=' * 80}")
    print("Benchmark Summary")
    print(f"{'=' * 80}")

    # Find best configuration for each skeleton size
    for size in skeleton_sizes:
        size_results = [r for r in results if r["skeleton_size"] == size]

        best_thread = max(size_results, key=lambda x: x["thread_speedup"])
        best_process = max(size_results, key=lambda x: x["process_speedup"])

        print(f"\nSkeleton size {size}:")
        print(
            f"  Best threading: {best_thread['workers']} workers, {best_thread['thread_speedup']:.2f}x speedup"
        )
        print(
            f"  Best multiprocessing: {best_process['workers']} workers, {best_process['process_speedup']:.2f}x speedup"
        )

    # Overall best
    best_overall = max(
        results, key=lambda x: max(x["thread_speedup"], x["process_speedup"])
    )
    print(f"\nOverall best configuration:")
    if best_overall["thread_speedup"] > best_overall["process_speedup"]:
        print(
            f"  Threading with {best_overall['workers']} workers: {best_overall['thread_speedup']:.2f}x speedup"
        )
    else:
        print(
            f"  Multiprocessing with {best_overall['workers']} workers: {best_overall['process_speedup']:.2f}x speedup"
        )

    return results


def run_ray_tracing_benchmark():
    """Benchmark specifically the ray tracing performance."""
    print(f"\n{'=' * 80}")
    print("Ray Tracing Performance Benchmark")
    print(f"{'=' * 80}")

    mesh = example_mesh("torus", major_radius=4.0, minor_radius=1.0, major_sections=48)

    # Test different ray counts
    ray_counts = [6, 12, 24, 48]
    test_point = np.array([3.0, 0.0, 0.0])

    options = SkeletonOptimizerOptions(max_iterations=10, verbose=False)

    for n_rays in ray_counts:
        options.n_rays = n_rays

        # Sequential
        seq_options = ParallelOptimizerOptions(
            enable_ray_parallel=False,
            enable_node_parallel=False,
        )

        skeleton = SkeletonGraph.from_polylines([np.array([test_point])])
        seq_optimizer = ParallelSkeletonOptimizer(skeleton, mesh, options, seq_options)

        start_time = time.time()
        seq_optimizer._compute_centering_direction_sequential(test_point)
        seq_time = time.time() - start_time

        # Parallel
        par_options = ParallelOptimizerOptions(
            enable_ray_parallel=True,
            enable_node_parallel=False,
            ray_batch_size=n_rays,  # Process all rays at once
        )

        par_optimizer = ParallelSkeletonOptimizer(skeleton, mesh, options, par_options)

        start_time = time.time()
        par_optimizer._compute_centering_direction_parallel(test_point)
        par_time = time.time() - start_time

        speedup = seq_time / par_time if par_time > 0 else float("inf")

        print(
            f"Rays: {n_rays:2d} | Sequential: {seq_time:.4f}s | Parallel: {par_time:.4f}s | Speedup: {speedup:.2f}x"
        )


def run_memory_usage_test():
    """Test memory usage with different configurations."""
    print(f"\n{'=' * 80}")
    print("Memory Usage Test")
    print(f"{'=' * 80}")

    import psutil
    import os

    process = psutil.Process(os.getpid())

    mesh = example_mesh("cylinder", radius=1.5, height=12.0, sections=32)

    # Create a larger skeleton
    size = 1000
    points = []
    for i in range(size):
        z = -6 + 12 * i / size
        points.append([0.2 + 0.1 * np.sin(i / 10), 0.1 + 0.1 * np.cos(i / 10), z])

    skeleton = SkeletonGraph.from_polylines([np.array(points)])

    options = SkeletonOptimizerOptions(max_iterations=20, verbose=False)

    # Test sequential
    baseline_memory = process.memory_info().rss / 1024 / 1024  # MB

    seq_optimizer = SkeletonOptimizer(skeleton.copy_skeleton(), mesh, options)
    seq_optimizer.optimize()

    seq_memory = process.memory_info().rss / 1024 / 1024  # MB
    seq_memory_usage = seq_memory - baseline_memory

    # Test parallel with different worker counts
    for workers in [2, 4, mp.cpu_count()]:
        # Reset memory baseline
        del seq_optimizer
        baseline_memory = process.memory_info().rss / 1024 / 1024

        parallel_options = ParallelOptimizerOptions(
            max_workers=workers,
            use_processes=True,
            batch_size=50,
        )

        par_optimizer = ParallelSkeletonOptimizer(
            skeleton.copy_skeleton(), mesh, options, parallel_options
        )
        par_optimizer.optimize()

        par_memory = process.memory_info().rss / 1024 / 1024
        par_memory_usage = par_memory - baseline_memory

        print(
            f"Workers: {workers:2d} | Sequential memory: {seq_memory_usage:.1f}MB | Parallel memory: {par_memory_usage:.1f}MB | Ratio: {par_memory_usage/seq_memory_usage:.2f}x"
        )


def save_benchmark_results(
    results: List[Dict], filename: str = "benchmark_results.json"
):
    """Save benchmark results to JSON file."""
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nBenchmark results saved to {filename}")


def main():
    """Main benchmark function."""
    parser = argparse.ArgumentParser(
        description="Benchmark parallel skeleton optimizer"
    )
    parser.add_argument("--quick", action="store_true", help="Run quick benchmark only")
    parser.add_argument("--save", action="store_true", help="Save results to file")
    parser.add_argument(
        "--rays-only", action="store_true", help="Only run ray tracing benchmark"
    )
    parser.add_argument(
        "--memory-only", action="store_true", help="Only run memory usage test"
    )

    args = parser.parse_args()

    if args.rays_only:
        run_ray_tracing_benchmark()
    elif args.memory_only:
        run_memory_usage_test()
    else:
        results = run_comprehensive_benchmark()

        if not args.quick:
            run_ray_tracing_benchmark()
            run_memory_usage_test()

        if args.save:
            save_benchmark_results(results)


if __name__ == "__main__":
    main()
