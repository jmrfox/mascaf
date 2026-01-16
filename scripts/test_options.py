"""
Test script to verify options handling in parallel optimizer.
"""

import numpy as np
from mcf2swc import (
    SkeletonGraph,
    SkeletonOptimizerOptions,
    ParallelSkeletonOptimizer,
    ParallelOptimizerOptions,
    example_mesh,
)


def test_options():
    print("Testing options handling...")

    # Create test data
    mesh = example_mesh("cylinder", radius=1.0, height=10.0, sections=16)
    points = np.array([[0.3, 0.2, -3.5], [0.3, 0.2, 0.0], [0.3, 0.2, 3.5]])
    skeleton = SkeletonGraph.from_polylines([points])

    # Test options
    opts = SkeletonOptimizerOptions(
        max_iterations=5,
        step_size=0.1,
        check_surface_crossing=True,
        verbose=False,
    )

    par_opts = ParallelOptimizerOptions(
        max_workers=2,
        use_processes=False,
    )

    print(
        f"SkeletonOptimizerOptions.check_surface_crossing: {opts.check_surface_crossing}"
    )
    print(f"ParallelOptimizerOptions attributes: {dir(par_opts)}")

    # Create optimizer
    optimizer = ParallelSkeletonOptimizer(skeleton, mesh, opts, par_opts)

    print(
        f"Optimizer.options.check_surface_crossing: {optimizer.options.check_surface_crossing}"
    )
    print(
        f"Optimizer.parallel_options.max_workers: {optimizer.parallel_options.max_workers}"
    )

    # Test optimization
    try:
        optimized = optimizer.optimize()
        print("Optimization successful!")
    except AttributeError as e:
        print(f"AttributeError: {e}")
    except Exception as e:
        print(f"Other error: {e}")


if __name__ == "__main__":
    test_options()
