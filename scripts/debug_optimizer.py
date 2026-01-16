"""Debug script to understand skeleton optimizer behavior."""

import numpy as np
from mcf2swc import (
    PolylinesSkeleton,
    SkeletonOptimizer,
    SkeletonOptimizerOptions,
    example_mesh,
)

# Create a torus mesh
mesh = example_mesh("torus", major_radius=4.0, minor_radius=1.0)

# Create a simple skeleton along the torus centerline
# This should be well-centered
theta = np.linspace(0, 2 * np.pi, 20)
major_radius = 4.0
skeleton_points = np.column_stack(
    [major_radius * np.cos(theta), major_radius * np.sin(theta), np.zeros_like(theta)]
)

skeleton = PolylinesSkeleton([skeleton_points])

# Check distances before optimization
from trimesh.proximity import closest_point

print("=== BEFORE OPTIMIZATION ===")
for i, pt in enumerate(skeleton_points[::5]):  # Check every 5th point
    is_inside = mesh.contains(pt.reshape(1, 3))[0]
    cp, dist, _ = closest_point(mesh, pt.reshape(1, 3))
    print(f"Point {i*5}: inside={is_inside}, dist_to_surface={dist[0]:.4f}")

# Optimize
opts = SkeletonOptimizerOptions(
    max_iterations=50, step_size=0.1, centering_method="medial_axis", verbose=True
)
optimizer = SkeletonOptimizer(skeleton, mesh, opts)
optimized = optimizer.optimize()

# Check distances after optimization
print("\n=== AFTER OPTIMIZATION ===")
for i, pt in enumerate(optimized.polylines[0][::5]):  # Check every 5th point
    is_inside = mesh.contains(pt.reshape(1, 3))[0]
    cp, dist, _ = closest_point(mesh, pt.reshape(1, 3))
    print(f"Point {i*5}: inside={is_inside}, dist_to_surface={dist[0]:.4f}")

# Check if any points went outside
all_pts_after = optimized.polylines[0]
inside_mask = mesh.contains(all_pts_after)
num_outside = np.sum(~inside_mask)
print(f"\nPoints outside mesh after optimization: {num_outside}/{len(all_pts_after)}")
