"""Test Fibonacci sphere distribution for different n_rays values."""

import numpy as np
from mcf2swc import SkeletonOptimizer, PolylinesSkeleton, example_mesh

# Create a simple optimizer to access the method
mesh = example_mesh("cylinder")
skeleton = PolylinesSkeleton([np.array([[0, 0, 0], [0, 0, 1]])])
optimizer = SkeletonOptimizer(skeleton, mesh)

print("Testing Fibonacci sphere distribution:\n")

for n_rays in [4, 6, 8, 12, 20]:
    directions = optimizer._get_uniform_sphere_directions(n_rays)

    print(f"n_rays = {n_rays}:")
    print(f"  Shape: {directions.shape}")

    # Check that all are unit vectors
    norms = np.linalg.norm(directions, axis=1)
    print(f"  All unit vectors: {np.allclose(norms, 1.0)}")

    # For n_rays=6, check if they approximate +/- x, y, z axes
    if n_rays == 6:
        print(f"  Directions (should approximate ±x, ±y, ±z):")
        for i, d in enumerate(directions):
            print(f"    {i}: [{d[0]:7.4f}, {d[1]:7.4f}, {d[2]:7.4f}]")

    # Compute minimum angle between any two directions
    min_angle = np.inf
    for i in range(len(directions)):
        for j in range(i + 1, len(directions)):
            dot = np.dot(directions[i], directions[j])
            angle = np.arccos(np.clip(dot, -1, 1))
            min_angle = min(min_angle, angle)

    print(f"  Min angle between rays: {np.degrees(min_angle):.1f}°")
    print()
