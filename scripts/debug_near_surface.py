"""Debug with point very close to inner surface."""

import numpy as np
from mcf2swc import (
    PolylinesSkeleton,
    SkeletonOptimizer,
    SkeletonOptimizerOptions,
    example_mesh,
)

# Create a torus mesh
mesh = example_mesh("torus", major_radius=4.0, minor_radius=1.0)

# Point VERY close to inner surface (almost touching)
# Inner surface is at radius 3.0, so use 3.05
point = np.array([3.05, 0.0, 0.0])
skeleton = PolylinesSkeleton([np.array([point, point + [0, 0.1, 0]])])

# Create optimizer
opts = SkeletonOptimizerOptions(
    max_iterations=10, step_size=0.1, centering_method="medial_axis", verbose=True
)
optimizer = SkeletonOptimizer(skeleton, mesh, opts)

from trimesh.proximity import closest_point

print("=== INITIAL STATE ===")
is_inside = mesh.contains(point.reshape(1, 3))[0]
cp, dist, _ = closest_point(mesh, point.reshape(1, 3))
print(f"Point: {point}")
print(f"Inside: {is_inside}")
print(f"Distance to surface: {dist[0]:.4f}")
print(f"Closest point: {cp[0]}")

# Optimize
print("\n=== OPTIMIZING ===")
optimized = optimizer.optimize()

print("\n=== AFTER OPTIMIZATION ===")
final_point = optimized.polylines[0][0]
is_inside_final = mesh.contains(final_point.reshape(1, 3))[0]
cp_final, dist_final, _ = closest_point(mesh, final_point.reshape(1, 3))
print(f"Final point: {final_point}")
print(f"Inside: {is_inside_final}")
print(f"Distance to surface: {dist_final[0]:.4f}")
print(f"Movement: {np.linalg.norm(final_point - point):.4f}")

if not is_inside_final:
    print("\n⚠️  POINT PUSHED OUTSIDE MESH!")
elif dist_final[0] < dist[0]:
    print(
        f"\n⚠️  POINT MOVED CLOSER TO SURFACE! (from {dist[0]:.4f} to {dist_final[0]:.4f})"
    )
else:
    print(
        f"\n✓ Point moved away from surface (from {dist[0]:.4f} to {dist_final[0]:.4f})"
    )
