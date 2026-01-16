"""Detailed debug to understand force directions."""

import numpy as np
from mcf2swc import (
    PolylinesSkeleton,
    SkeletonOptimizer,
    SkeletonOptimizerOptions,
    example_mesh,
)

# Create a torus mesh
mesh = example_mesh("torus", major_radius=4.0, minor_radius=1.0)

# Single point on the torus centerline
point = np.array([4.0, 0.0, 0.0])
skeleton = PolylinesSkeleton(
    [np.array([point, point + [0, 0.1, 0]])]
)  # Need at least 2 points

# Create optimizer
opts = SkeletonOptimizerOptions(
    max_iterations=1, step_size=0.1, centering_method="medial_axis", verbose=False
)
optimizer = SkeletonOptimizer(skeleton, mesh, opts)

# Manually compute the centering direction
tangent = np.array([0.0, 1.0, 0.0])  # Along the torus
direction = optimizer._compute_centering_direction(point, tangent)

print(f"Point: {point}")
print(f"Tangent: {tangent}")
print(f"Computed direction: {direction}")
print(f"Direction magnitude: {np.linalg.norm(direction)}")

# Check what this direction does
from trimesh.proximity import closest_point

is_inside = mesh.contains(point.reshape(1, 3))[0]
cp_before, dist_before, _ = closest_point(mesh, point.reshape(1, 3))
print(f"\nBefore move:")
print(f"  Inside: {is_inside}")
print(f"  Distance to surface: {dist_before[0]:.4f}")
print(f"  Closest point: {cp_before[0]}")

# Move in the computed direction
new_point = point + 0.1 * direction
cp_after, dist_after, _ = closest_point(mesh, new_point.reshape(1, 3))
is_inside_after = mesh.contains(new_point.reshape(1, 3))[0]
print(f"\nAfter move:")
print(f"  New point: {new_point}")
print(f"  Inside: {is_inside_after}")
print(f"  Distance to surface: {dist_after[0]:.4f}")
print(f"  Closest point: {cp_after[0]}")

# Manually check ray distances
perp1, perp2 = optimizer._compute_perpendicular_axes(tangent)
print(f"\nPerpendicular axes:")
print(f"  perp1: {perp1}")
print(f"  perp2: {perp2}")

# Check distances in perpendicular directions
for i, angle in enumerate([0, np.pi / 2, np.pi, 3 * np.pi / 2]):
    dir_vec = np.cos(angle) * perp1 + np.sin(angle) * perp2
    d_pos = optimizer._ray_distance_to_surface(point, dir_vec)
    d_neg = optimizer._ray_distance_to_surface(point, -dir_vec)
    print(f"\nAngle {np.degrees(angle):.0f}°:")
    print(f"  Direction: {dir_vec}")
    print(f"  d_pos: {d_pos:.4f}, d_neg: {d_neg:.4f}, diff: {d_pos - d_neg:.4f}")
