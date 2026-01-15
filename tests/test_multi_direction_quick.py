"""
Test multi-directional sampling for medial axis centering.
"""

import pytest
import numpy as np
from mcf2swc import (
    PolylinesSkeleton,
    SkeletonOptimizer,
    SkeletonOptimizerOptions,
    example_mesh,
)


@pytest.mark.parametrize("num_directions", [4, 8])
def test_multi_direction_sampling(num_directions):
    """Test medial axis centering with different numbers of probe directions."""
    mesh = example_mesh("cylinder", radius=1.0, height=10.0)

    points = np.array(
        [
            [0.5, 0.5, -3.5],
            [0.5, 0.5, 0.0],
            [0.5, 0.5, 3.5],
        ]
    )
    skeleton = PolylinesSkeleton([points])

    options = SkeletonOptimizerOptions(
        centering_method="medial_axis",
        probe_distance=5.0,
        num_probe_directions=num_directions,
        max_iterations=10,
        step_size=0.1,
        preserve_endpoints=True,
        smoothing_weight=0.3,
        verbose=False,
    )

    optimizer = SkeletonOptimizer(skeleton, mesh, options)
    optimized = optimizer.optimize()

    assert len(optimized.polylines) == 1
    assert optimized.total_points() == skeleton.total_points()

    optimized_points = optimized.polylines[0]
    optimized_distances = np.linalg.norm(optimized_points[:, :2], axis=1)

    assert np.all(optimized_distances < 1.0)


def test_more_directions_improves_centering():
    """Test that more probe directions can improve centering quality."""
    mesh = example_mesh("cylinder", radius=1.0, height=10.0)

    points = np.array(
        [
            [0.6, 0.6, -3.5],
            [0.6, 0.6, 0.0],
            [0.6, 0.6, 3.5],
        ]
    )
    skeleton = PolylinesSkeleton([points])

    results = []
    for num_dirs in [4, 8]:
        options = SkeletonOptimizerOptions(
            centering_method="medial_axis",
            probe_distance=5.0,
            num_probe_directions=num_dirs,
            max_iterations=15,
            step_size=0.1,
            preserve_endpoints=False,
            smoothing_weight=0.3,
            verbose=False,
        )

        optimizer = SkeletonOptimizer(skeleton, mesh, options)
        optimized = optimizer.optimize()

        optimized_points = optimized.polylines[0]
        mean_distance = np.mean(np.linalg.norm(optimized_points[:, :2], axis=1))
        results.append(mean_distance)

    assert results[1] <= results[0] * 1.1
