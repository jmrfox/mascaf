"""Test with a more realistic skeleton that might have issues."""

import numpy as np
from mcf2swc import (
    PolylinesSkeleton,
    SkeletonOptimizer,
    SkeletonOptimizerOptions,
    example_mesh,
)
from trimesh.proximity import closest_point

# Create a torus mesh
mesh = example_mesh("torus", major_radius=4.0, minor_radius=1.0)

# Create a skeleton with some points offset from centerline
# Simulate what MCF might produce - not perfectly centered
theta = np.linspace(0, 2 * np.pi, 20, endpoint=False)
major_radius = 4.0

# Add some noise/offset to simulate imperfect MCF skeleton
np.random.seed(42)
radial_offset = np.random.uniform(-0.3, 0.3, len(theta))
vertical_offset = np.random.uniform(-0.3, 0.3, len(theta))

skeleton_points = np.column_stack(
    [
        (major_radius + radial_offset) * np.cos(theta),
        (major_radius + radial_offset) * np.sin(theta),
        vertical_offset,
    ]
)

skeleton = PolylinesSkeleton([skeleton_points])

print("=== BEFORE OPTIMIZATION ===")
inside_before = mesh.contains(skeleton_points)
dists_before = []
for i, pt in enumerate(skeleton_points):
    cp, dist, _ = closest_point(mesh, pt.reshape(1, 3))
    dists_before.append(dist[0])
    if i % 5 == 0:
        print(f"Point {i:2d}: inside={inside_before[i]}, dist={dist[0]:.4f}")

print(f"\nAverage distance to surface: {np.mean(dists_before):.4f}")
print(f"Min distance to surface: {np.min(dists_before):.4f}")
print(f"Points outside: {np.sum(~inside_before)}/{len(skeleton_points)}")

# Optimize
opts = SkeletonOptimizerOptions(
    max_iterations=50,
    step_size=0.1,
    preserve_endpoints=False,  # Allow all points to move
    n_rays=12,
    smoothing_weight=0.5,
    verbose=False,
)
optimizer = SkeletonOptimizer(skeleton, mesh, opts)
optimized = optimizer.optimize()

print("\n=== AFTER OPTIMIZATION ===")
opt_points = optimized.polylines[0]
inside_after = mesh.contains(opt_points)
dists_after = []
for i, pt in enumerate(opt_points):
    cp, dist, _ = closest_point(mesh, pt.reshape(1, 3))
    dists_after.append(dist[0])
    if i % 5 == 0:
        print(
            f"Point {i:2d}: inside={inside_after[i]}, dist={dist[0]:.4f}, change={dist[0]-dists_before[i]:+.4f}"
        )

print(
    f"\nAverage distance to surface: {np.mean(dists_after):.4f} (change: {np.mean(dists_after)-np.mean(dists_before):+.4f})"
)
print(
    f"Min distance to surface: {np.min(dists_after):.4f} (change: {np.min(dists_after)-np.min(dists_before):+.4f})"
)
print(f"Points outside: {np.sum(~inside_after)}/{len(opt_points)}")

# Check how many points moved closer vs farther
closer = np.sum(np.array(dists_after) < np.array(dists_before))
farther = np.sum(np.array(dists_after) > np.array(dists_before))
same = np.sum(np.abs(np.array(dists_after) - np.array(dists_before)) < 1e-6)

print(f"\nPoints that moved closer to surface: {closer}")
print(f"Points that moved farther from surface: {farther}")
print(f"Points that stayed same: {same}")

if closer > farther:
    print("\n⚠️  MORE POINTS MOVED CLOSER TO SURFACE THAN FARTHER!")
    print("This confirms the bug you're seeing.")
