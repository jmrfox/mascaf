"""
Demo script for parallel skeleton optimization.

This example demonstrates how to use the ParallelSkeletonOptimizer to speed up
skeleton refinement on larger skeletons using multithreading and multiprocessing.
"""

import numpy as np
import time
import multiprocessing as mp

from mascaf import (
    MeshManager,
    SkeletonGraph,
    SkeletonOptimizer,
    SkeletonOptimizerOptions,
    ParallelSkeletonOptimizer,
    ParallelOptimizerOptions,
    example_mesh,
)


def create_complex_skeleton(size: int = 200) -> SkeletonGraph:
    """Create a complex skeleton with multiple branches for testing."""
    points = []

    # Main backbone
    main_backbone = np.linspace(-8, 8, size // 3)
    for z in main_backbone:
        points.append([0.3, 0.2, z])

    # Add multiple branches
    n_branches = 4
    for i in range(n_branches):
        branch_start = main_backbone[len(main_backbone) // (n_branches + 1) * (i + 1)]
        branch_points = np.linspace(0, 3, size // (3 * n_branches))
        for r in branch_points:
            angle = 2 * np.pi * i / n_branches + np.pi / (2 * n_branches)
            points.append(
                [r * np.cos(angle) + 0.3, r * np.sin(angle) + 0.2, branch_start]
            )

    # Add some sub-branches
    for i in range(2):
        sub_branch_start = main_backbone[len(main_backbone) // 2]
        sub_branch_points = np.linspace(0, 2, size // 12)
        for r in sub_branch_points:
            angle = np.pi * i + np.pi / 4
            points.append(
                [r * np.cos(angle) + 0.3, r * np.sin(angle) + 0.2, sub_branch_start]
            )

    skeleton_points = np.array(points)
    return SkeletonGraph.from_polylines([skeleton_points])


def demo_sequential_vs_parallel():
    """Demonstrate performance difference between sequential and parallel optimizers."""
    print("=" * 80)
    print("Sequential vs Parallel Skeleton Optimization Demo")
    print("=" * 80)

    # Create test mesh and skeleton
    mesh = example_mesh("cylinder", radius=2.0, height=16.0, sections=32)
    skeleton = create_complex_skeleton(300)

    print(f"\nTest setup:")
    print(f"  Mesh: cylinder (vertices: {len(mesh.vertices)})")
    print(
        f"  Skeleton: {skeleton.number_of_nodes()} nodes, {skeleton.number_of_edges()} edges"
    )

    # Optimization options
    opts = SkeletonOptimizerOptions(
        max_iterations=50,
        step_size=0.1,
        convergence_threshold=1e-4,
        n_rays=6,
        smoothing_weight=0.5,
        verbose=False,
    )

    # Sequential optimization
    print(f"\n{'-' * 60}")
    print("Sequential Optimization")
    print(f"{'-' * 60}")

    start_time = time.time()
    seq_optimizer = SkeletonOptimizer(skeleton.copy_skeleton(), mesh, opts)
    seq_optimized = seq_optimizer.optimize()
    seq_time = time.time() - start_time

    print(f"Time: {seq_time:.3f} seconds")
    seq_stats = seq_optimizer.get_optimization_stats()
    print(f"Final nodes outside mesh: {seq_stats.get('nodes_outside_mesh', 0)}")

    # Parallel optimization with different configurations
    worker_configs = [
        {"workers": 2, "use_processes": False, "name": "Threading"},
        {"workers": 2, "use_processes": True, "name": "Multiprocessing"},
        {"workers": 4, "use_processes": False, "name": "Threading (4 workers)"},
        {"workers": 4, "use_processes": True, "name": "Multiprocessing (4 workers)"},
    ]

    print(f"\n{'-' * 60}")
    print("Parallel Optimization")
    print(f"{'-' * 60}")

    for config in worker_configs:
        par_opts = ParallelOptimizerOptions(
            max_workers=config["workers"],
            use_processes=config["use_processes"],
            enable_ray_parallel=True,
            enable_node_parallel=True,
            batch_size=20,
        )

        start_time = time.time()
        par_optimizer = ParallelSkeletonOptimizer(
            skeleton.copy_skeleton(), mesh, opts, par_opts
        )
        par_optimized = par_optimizer.optimize()
        par_time = time.time() - start_time

        speedup = seq_time / par_time if par_time > 0 else float("inf")
        par_stats = par_optimizer.get_optimization_stats()

        print(
            f"{config['name']:25} | Time: {par_time:6.3f}s | Speedup: {speedup:5.2f}x | "
            f"Nodes outside: {par_stats.get('nodes_outside_mesh', 0)}"
        )


def demo_ray_tracing_parallelization():
    """Demonstrate ray tracing parallelization specifically."""
    print(f"\n\n{'=' * 80}")
    print("Ray Tracing Parallelization Demo")
    print(f"{'=' * 80}")

    mesh = example_mesh("torus", major_radius=5.0, minor_radius=1.5, major_sections=48)

    # Test points at different locations
    test_points = [
        np.array([4.0, 0.0, 0.0]),  # Near surface
        np.array([2.5, 0.0, 0.0]),  # Inside
        np.array([6.0, 0.0, 0.0]),  # Outside
    ]

    ray_counts = [6, 12, 24, 48]

    opts = SkeletonOptimizerOptions(max_iterations=1, verbose=False)

    for point in test_points:
        print(f"\nTest point: {point}")
        print(f"{'-' * 60}")

        for n_rays in ray_counts:
            opts.n_rays = n_rays

            # Sequential
            seq_opts = ParallelOptimizerOptions(enable_ray_parallel=False)
            skeleton = SkeletonGraph.from_polylines([np.array([point])])
            seq_optimizer = ParallelSkeletonOptimizer(skeleton, mesh, opts, seq_opts)

            start_time = time.time()
            seq_optimizer._compute_centering_direction_sequential(point)
            seq_time = time.time() - start_time

            # Parallel
            par_opts = ParallelOptimizerOptions(
                enable_ray_parallel=True,
                ray_batch_size=n_rays,
            )
            par_optimizer = ParallelSkeletonOptimizer(skeleton, mesh, opts, par_opts)

            start_time = time.time()
            par_optimizer._compute_centering_direction_parallel(point)
            par_time = time.time() - start_time

            speedup = seq_time / par_time if par_time > 0 else float("inf")

            print(
                f"  Rays: {n_rays:2d} | Sequential: {seq_time:6.4f}s | "
                f"Parallel: {par_time:6.4f}s | Speedup: {speedup:5.2f}x"
            )


def demo_batch_processing():
    """Demonstrate batch processing for different skeleton sizes."""
    print(f"\n\n{'=' * 80}")
    print("Batch Processing Demo")
    print(f"{'=' * 80}")

    mesh = example_mesh("cylinder", radius=1.5, height=12.0, sections=32)

    skeleton_sizes = [50, 100, 200, 500]
    batch_sizes = [5, 10, 20, 50]

    opts = SkeletonOptimizerOptions(max_iterations=30, verbose=False)

    for size in skeleton_sizes:
        print(f"\nSkeleton size: {size} nodes")
        print(f"{'-' * 60}")

        skeleton = create_complex_skeleton(size)

        # Sequential baseline
        start_time = time.time()
        seq_optimizer = SkeletonOptimizer(skeleton.copy_skeleton(), mesh, opts)
        seq_optimizer.optimize()
        seq_time = time.time() - start_time

        print(f"Sequential: {seq_time:.3f}s")

        # Test different batch sizes
        for batch_size in batch_sizes:
            if batch_size >= size:
                continue

            par_opts = ParallelOptimizerOptions(
                max_workers=2,
                batch_size=batch_size,
                enable_ray_parallel=True,
                enable_node_parallel=True,
            )

            start_time = time.time()
            par_optimizer = ParallelSkeletonOptimizer(
                skeleton.copy_skeleton(), mesh, opts, par_opts
            )
            par_optimizer.optimize()
            par_time = time.time() - start_time

            speedup = seq_time / par_time if par_time > 0 else float("inf")
            print(
                f"  Batch size {batch_size:2d}: {par_time:6.3f}s ({speedup:5.2f}x speedup)"
            )


def demo_memory_efficiency():
    """Demonstrate memory usage with different configurations."""
    print(f"\n\n{'=' * 80}")
    print("Memory Efficiency Demo")
    print(f"{'=' * 80}")

    try:
        import psutil
        import os

        process = psutil.Process(os.getpid())
    except ImportError:
        print("psutil not available - skipping memory demo")
        return

    mesh = example_mesh("cylinder", radius=2.0, height=16.0, sections=32)
    skeleton = create_complex_skeleton(1000)

    opts = SkeletonOptimizerOptions(max_iterations=20, verbose=False)

    configurations = [
        {"name": "Sequential", "parallel": False},
        {"name": "Threading", "parallel": True, "use_processes": False},
        {"name": "Multiprocessing", "parallel": True, "use_processes": True},
    ]

    for config in configurations:
        # Measure baseline memory
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB

        if config["parallel"]:
            par_opts = ParallelOptimizerOptions(
                max_workers=2,
                use_processes=config.get("use_processes", True),
                batch_size=50,
            )
            optimizer = ParallelSkeletonOptimizer(
                skeleton.copy_skeleton(), mesh, opts, par_opts
            )
        else:
            optimizer = SkeletonOptimizer(skeleton.copy_skeleton(), mesh, opts)

        optimizer.optimize()

        peak_memory = process.memory_info().rss / 1024 / 1024
        memory_used = peak_memory - baseline_memory

        print(f"{config['name']:15} | Memory used: {memory_used:6.1f} MB")

        # Clean up
        del optimizer


def demo_real_world_scenario():
    """Demonstrate a realistic scenario with a complex skeleton."""
    print(f"\n\n{'=' * 80}")
    print("Real-World Scenario Demo")
    print(f"{'=' * 80}")

    # Create a more complex mesh
    mesh = example_mesh("torus", major_radius=6.0, minor_radius=2.0, major_sections=64)

    # Create a complex skeleton that might come from MCF
    skeleton = create_complex_skeleton(800)

    print(f"Scenario: Complex torus with branched skeleton")
    print(f"  Mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    print(f"  Skeleton: {skeleton.number_of_nodes()} nodes")

    # Check initial state
    par_opts = ParallelOptimizerOptions(max_workers=mp.cpu_count())
    optimizer = ParallelSkeletonOptimizer(skeleton, mesh, parallel_options=par_opts)

    has_crossing, num_outside, max_dist = optimizer.check_surface_crossing()
    print(
        f"  Initial state: {num_outside} nodes outside mesh (max dist: {max_dist:.4f})"
    )

    # Optimization with aggressive settings
    opts = SkeletonOptimizerOptions(
        max_iterations=100,
        step_size=0.15,
        convergence_threshold=1e-5,
        n_rays=12,
        smoothing_weight=0.3,
        verbose=True,
    )

    par_opts = ParallelOptimizerOptions(
        max_workers=mp.cpu_count(),
        use_processes=True,
        batch_size=25,
        ray_batch_size=12,
        enable_ray_parallel=True,
        enable_node_parallel=True,
    )

    print(f"\nRunning parallel optimization...")
    start_time = time.time()

    optimizer = ParallelSkeletonOptimizer(skeleton, mesh, opts, par_opts)
    optimized = optimizer.optimize()

    total_time = time.time() - start_time

    # Check final state
    has_crossing_after, num_outside_after, max_dist_after = (
        optimizer.check_surface_crossing()
    )

    print(f"\nResults:")
    print(f"  Total time: {total_time:.3f} seconds")
    print(
        f"  Nodes outside after: {num_outside_after} (improvement: {num_outside - num_outside_after})"
    )
    print(f"  Max distance after: {max_dist_after:.4f}")

    stats = optimizer.get_optimization_stats()
    print(f"  Workers used: {stats['max_workers']}")
    print(f"  Parallel ray tracing: {stats['enable_ray_parallel']}")
    print(f"  Parallel node processing: {stats['enable_node_parallel']}")


if __name__ == "__main__":
    demo_sequential_vs_parallel()
    demo_ray_tracing_parallelization()
    demo_batch_processing()
    demo_memory_efficiency()
    demo_real_world_scenario()

    print(f"\n{'=' * 80}")
    print("All parallel optimization demos complete!")
    print(f"{'=' * 80}")
