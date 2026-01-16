"""Debug script to check what centering forces are being computed."""

import numpy as np
from mcf2swc import (
    PolylinesSkeleton,
    MeshManager,
    SkeletonOptimizer,
    SkeletonOptimizerOptions,
)

# Load your data
skeleton = PolylinesSkeleton.from_txt(
    r"data\mcf_skeletons\TS2_qst0.5_mcst10.polylines.txt"
)
mesh_mgr = MeshManager(mesh_path=r"data\mesh\processed\TS2_simplified.obj")

# Create optimizer with 6 axis-aligned rays
options = SkeletonOptimizerOptions(
    step_size=1.0,
    smoothing_weight=0.0,
    max_iterations=1,
    verbose=False,
    n_rays=6,  # Use axis-aligned rays
)

optimizer = SkeletonOptimizer(skeleton, mesh_mgr.mesh, options)

# Ray labels for clarity
ray_labels = ["+X", "-X", "+Y", "-Y", "+Z", "-Z"]

# Test centering direction for a few points
print("Testing centering forces with axis-aligned rays:\n")
print("Force computation: force -= direction / distance")
print("(closer surfaces push harder in opposite direction)\n")

for poly_idx, polyline in enumerate(skeleton.polylines[:2]):  # First 2 polylines
    print(f"\n{'='*70}")
    print(f"Polyline {poly_idx}")
    print(f"{'='*70}")

    for i in [len(polyline) // 2]:  # Just middle point for clarity
        point = polyline[i]

        # Check if inside
        is_inside = mesh_mgr.mesh.contains(point.reshape(1, 3))[0]
        print(f"\nPoint {i}: [{point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f}]")
        print(f"Inside mesh: {is_inside}\n")

        if not is_inside:
            print("  Point is OUTSIDE mesh!")
            continue

        # Get ray directions
        directions = optimizer._get_uniform_sphere_directions(6)

        # Compute force manually to show each contribution
        print(
            f"{'Ray':<6} {'Direction':<20} {'Distance':>10} {'1/d':>10} {'Force Contrib':<30}"
        )
        print("-" * 90)

        force = np.zeros(3)
        for j, (ray_dir, label) in enumerate(zip(directions, ray_labels)):
            d = optimizer._ray_distance_to_surface(point, ray_dir)

            # Compute contribution: force -= direction / d
            if d > 1e-6:
                contrib = -ray_dir / d
                force += contrib
                contrib_str = (
                    f"[{contrib[0]:>7.4f}, {contrib[1]:>7.4f}, {contrib[2]:>7.4f}]"
                )
            else:
                contrib_str = "[skipped - d too small]"

            dir_str = f"[{ray_dir[0]:>4.1f}, {ray_dir[1]:>4.1f}, {ray_dir[2]:>4.1f}]"
            print(
                f"{label:<6} {dir_str:<20} {d:>10.4f} {1/d if d > 1e-6 else 0:>10.4f} {contrib_str:<30}"
            )

        # Normalize force
        force_mag = np.linalg.norm(force)
        if force_mag > 1e-10:
            force_normalized = force / force_mag
        else:
            force_normalized = np.zeros(3)

        print("-" * 90)
        print(
            f"\nRaw force sum:        [{force[0]:>8.4f}, {force[1]:>8.4f}, {force[2]:>8.4f}]"
        )
        print(f"Force magnitude:      {force_mag:.4f}")
        print(
            f"Normalized direction: [{force_normalized[0]:>8.4f}, {force_normalized[1]:>8.4f}, {force_normalized[2]:>8.4f}]"
        )

        # Verify with optimizer's method
        computed_direction = optimizer._compute_centering_direction(point)
        print(
            f"Optimizer result:     [{computed_direction[0]:>8.4f}, {computed_direction[1]:>8.4f}, {computed_direction[2]:>8.4f}]"
        )

        # Show what the actual movement would be
        movement = options.step_size * force_normalized
        print(
            f"\nMovement (step_size={options.step_size}): [{movement[0]:>8.4f}, {movement[1]:>8.4f}, {movement[2]:>8.4f}]"
        )
        print(f"Movement magnitude: {np.linalg.norm(movement):.4f}")
