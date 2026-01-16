"""Debug with vertically offset point."""

import numpy as np
from mcf2swc import (
    PolylinesSkeleton,
    SkeletonOptimizer,
    SkeletonOptimizerOptions,
    example_mesh,
)

# Create a torus mesh
mesh = example_mesh("torus", major_radius=4.0, minor_radius=1.0)

# Point offset UPWARD from centerline (toward top of tube)
# This should be closer to the top surface
point = np.array([4.0, 0.0, 0.5])  # Moved up by 0.5 (half the minor radius)
skeleton = PolylinesSkeleton([np.array([point, point + [0, 0.1, 0]])])

# Create optimizer
opts = SkeletonOptimizerOptions(
    max_iterations=1, step_size=0.1, centering_method="medial_axis", verbose=False
)
optimizer = SkeletonOptimizer(skeleton, mesh, opts)

# Manually compute the centering direction
tangent = np.array([0.0, 1.0, 0.0])  # Along the torus
direction = optimizer._compute_centering_direction(point, tangent)

print(f"Point (offset upward): {point}")
print(f"Tangent: {tangent}")
print(f"Computed direction: {direction}")

# Check distances
from trimesh.proximity import closest_point

is_inside = mesh.contains(point.reshape(1, 3))[0]
cp_before, dist_before, _ = closest_point(mesh, point.reshape(1, 3))
print(f"\nBefore move:")
print(f"  Inside: {is_inside}")
print(f"  Distance to surface: {dist_before[0]:.4f}")
print(f"  Closest point on surface: {cp_before[0]}")

# Move in the computed direction
new_point = point + 0.1 * direction
cp_after, dist_after, _ = closest_point(mesh, new_point.reshape(1, 3))
is_inside_after = mesh.contains(new_point.reshape(1, 3))[0]
print(f"\nAfter move:")
print(f"  New point: {new_point}")
print(f"  Inside: {is_inside_after}")
print(f"  Distance to surface: {dist_after[0]:.4f}")
print(f"  Change in distance: {dist_after[0] - dist_before[0]:.4f}")
if dist_after[0] < dist_before[0]:
    print("  ⚠️  MOVED CLOSER TO SURFACE!")

# Check ray distances
perp1, perp2 = optimizer._compute_perpendicular_axes(tangent)
print(f"\nPerpendicular axes:")
print(f"  perp1 (vertical): {perp1}")
print(f"  perp2 (radial): {perp2}")

# Vertical direction
d_up = optimizer._ray_distance_to_surface(point, np.array([0, 0, 1]))
d_down = optimizer._ray_distance_to_surface(point, np.array([0, 0, -1]))
print(f"\nVertical direction:")
print(f"  d_up: {d_up:.4f}")
print(f"  d_down: {d_down:.4f}")
print(f"  diff (d_up - d_down): {d_up - d_down:.4f}")
print(f"  Expected: point is closer to top, so d_up < d_down")
print(f"  Force should move DOWN (negative z)")

# Check all angles
print("\nAll perpendicular directions:")
for i in range(8):
    angle = 2.0 * np.pi * i / 8
    dir_vec = np.cos(angle) * perp1 + np.sin(angle) * perp2
    d_pos = optimizer._ray_distance_to_surface(point, dir_vec)
    d_neg = optimizer._ray_distance_to_surface(point, -dir_vec)
    diff = d_pos - d_neg
    force_contrib = diff * dir_vec
    print(
        f"  Angle {np.degrees(angle):5.0f}°: diff={diff:7.4f}, force_z={force_contrib[2]:7.4f}"
    )

print(f"\nNet force direction: {direction}")
print(f"Net force z-component: {direction[2]:.4f}")
if direction[2] > 0:
    print("⚠️  Force moves UP (toward closer surface) - BUG!")
elif direction[2] < 0:
    print("✓ Force moves DOWN (away from closer surface) - CORRECT!")
