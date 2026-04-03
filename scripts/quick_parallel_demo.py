"""
Quick demonstration of parallel skeleton optimization.
"""

import numpy as np
import time
from mascaf import (
    SkeletonGraph,
    SkeletonOptimizer,
    SkeletonOptimizerOptions,
    ParallelSkeletonOptimizer,
    ParallelOptimizerOptions,
    example_mesh,
)


def create_test_skeleton():
    """Create a simple test skeleton."""
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


def main():
    print("Quick Parallel Skeleton Optimizer Demo")
    print("=" * 50)

    # Create test data
    mesh = example_mesh("cylinder", radius=1.0, height=10.0, sections=16)
    skeleton = create_test_skeleton()

    print(f"Mesh: cylinder ({len(mesh.vertices)} vertices)")
    print(f"Skeleton: {skeleton.number_of_nodes()} nodes")

    # Options
    opts = SkeletonOptimizerOptions(
        max_iterations=20,
        step_size=0.1,
        verbose=False,
    )

    # Sequential
    print("\nSequential optimization...")
    start = time.time()
    seq_opt = SkeletonOptimizer(skeleton.copy_skeleton(), mesh, opts)
    seq_result = seq_opt.optimize()
    seq_time = time.time() - start
    print(f"Time: {seq_time:.3f}s")

    # Parallel (threading)
    print("\nParallel optimization (threading)...")
    par_opts = ParallelOptimizerOptions(
        max_workers=2,
        use_processes=False,
        enable_ray_parallel=True,
        enable_node_parallel=True,
    )
    start = time.time()
    par_opt = ParallelSkeletonOptimizer(skeleton.copy_skeleton(), mesh, opts, par_opts)
    par_result = par_opt.optimize()
    par_time = time.time() - start
    print(f"Time: {par_time:.3f}s")

    # Parallel (multiprocessing)
    print("\nParallel optimization (multiprocessing)...")
    par_opts_mp = ParallelOptimizerOptions(
        max_workers=2,
        use_processes=True,
        enable_ray_parallel=True,
        enable_node_parallel=True,
    )
    start = time.time()
    par_opt_mp = ParallelSkeletonOptimizer(
        skeleton.copy_skeleton(), mesh, opts, par_opts_mp
    )
    par_result_mp = par_opt_mp.optimize()
    par_time_mp = time.time() - start
    print(f"Time: {par_time_mp:.3f}s")

    print(f"\nSpeedup:")
    if par_time > 0:
        print(f"Threading: {seq_time/par_time:.2f}x")
    if par_time_mp > 0:
        print(f"Multiprocessing: {seq_time/par_time_mp:.2f}x")

    print("\nDemo complete!")


if __name__ == "__main__":
    main()
