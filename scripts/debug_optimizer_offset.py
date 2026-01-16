"""Debug with offset point to see force direction."""

import numpy as np
from mcf2swc import (
    PolylinesSkeleton,
    SkeletonOptimizer,
    SkeletonOptimizerOptions,
    example_mesh,
)

# Create a torus mesh
mesh = example_mesh("torus", major_radius=4.0, minor_radius=1.0)

# Point offset INWARD (toward the hole) from centerline
# Centerline is at radius 4.0, so move to radius 3.5 (closer to inner surface)
point = np.array([3.5, 0.0, 0.0])
skeleton = PolylinesSkeleton([np.array([point, point + [0, 0.1, 0]])])

# Create optimizer
opts = SkeletonOptimizerOptions(
    max_iterations=1, step_size=0.1, centering_method="medial_axis", verbose=False
)
optimizer = SkeletonOptimizer(skeleton, mesh, opts)

# Manually compute the centering direction
tangent = np.array([0.0, 1.0, 0.0])  # Along the torus
direction = optimizer._compute_centering_direction(point, tangent)

print(f"Point (offset inward): {point}")
print(f"Tangent: {tangent}")
print(f"Computed direction: {direction}")

# Check distances
from trimesh.proximity import closest_point

is_inside = mesh.contains(point.reshape(1, 3))[0]
cp_before, dist_before, _ = closest_point(mesh, point.reshape(1, 3))
print(f"\nBefore move:")
print(f"  Inside: {is_inside}")
print(f"  Distance to surface: {dist_before[0]:.4f}")

# Move in the computed direction
new_point = point + 0.1 * direction
cp_after, dist_after, _ = closest_point(mesh, new_point.reshape(1, 3))
is_inside_after = mesh.contains(new_point.reshape(1, 3))[0]
print(f"\nAfter move:")
print(f"  New point: {new_point}")
print(f"  Inside: {is_inside_after}")
print(f"  Distance to surface: {dist_after[0]:.4f}")
print(f"  Change in distance: {dist_after[0] - dist_before[0]:.4f}")

# Check ray distances in key directions
perp1, perp2 = optimizer._compute_perpendicular_axes(tangent)
print(f"\nPerpendicular axes:")
print(f"  perp1: {perp1}")
print(f"  perp2: {perp2}")

# Radial direction (toward/away from torus center)
radial_dir = np.array([1.0, 0.0, 0.0])  # Outward from torus center
d_out = optimizer._ray_distance_to_surface(point, radial_dir)
d_in = optimizer._ray_distance_to_surface(point, -radial_dir)
print(f"\nRadial direction (outward from torus center):")
print(f"  d_outward: {d_out:.4f} (to outer surface)")
print(f"  d_inward: {d_in:.4f} (to inner surface)")
print(f"  diff (d_out - d_in): {d_out - d_in:.4f}")
print(f"  Expected: point is closer to inner surface, so d_in < d_out")
print(
    f"  Force contribution: {(d_out - d_in):.4f} * radial_dir = moves OUTWARD (correct!)"
)

# Check all angles
print("\nAll perpendicular directions:")
for i in range(4):
    angle = 2.0 * np.pi * i / 4
    dir_vec = np.cos(angle) * perp1 + np.sin(angle) * perp2
    d_pos = optimizer._ray_distance_to_surface(point, dir_vec)
    d_neg = optimizer._ray_distance_to_surface(point, -dir_vec)
    diff = d_pos - d_neg
    print(
        f"  Angle {np.degrees(angle):.0f}°: d_pos={d_pos:.4f}, d_neg={d_neg:.4f}, diff={diff:.4f}"
    )
