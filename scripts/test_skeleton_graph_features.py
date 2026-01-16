"""Test the new SkeletonGraph features: resampling, snapping, branch lengths."""

import numpy as np
from mcf2swc.skeleton import SkeletonGraph
from mcf2swc import MeshManager

print("Testing SkeletonGraph features\n")
print("=" * 70)

# Load skeleton and mesh
print("\nLoading skeleton and mesh...")
skeleton = SkeletonGraph.from_txt(
    r"data\mcf_skeletons\TS2_qst0.5_mcst10.polylines.txt", tolerance=1e-6
)
mesh_mgr = MeshManager(mesh_path=r"data\mesh\processed\TS2_simplified.obj")

print(f"Original: {skeleton}")
print(f"Total length: {skeleton.get_total_length():.2f}")

# Test 1: Branch length computation
print("\n" + "=" * 70)
print("\nTest 1: Compute branch lengths")
print("-" * 70)

branch_lengths = skeleton.compute_branch_lengths()
print(f"Found {len(branch_lengths)} branches:")
for (start, end), length in sorted(
    branch_lengths.items(), key=lambda x: x[1], reverse=True
):
    start_deg = skeleton.degree(start)
    end_deg = skeleton.degree(end)
    start_type = "terminal" if start_deg == 1 else f"branch(deg={start_deg})"
    end_type = "terminal" if end_deg == 1 else f"branch(deg={end_deg})"
    print(
        f"  Node {start} ({start_type}) -> Node {end} ({end_type}): {length:.2f} units"
    )

# Test 2: Resampling
print("\n" + "=" * 70)
print("\nTest 2: Resample skeleton")
print("-" * 70)

# Original spacing statistics
original_edge_lengths = [
    data.get("length", 0.0) for _, _, data in skeleton.edges(data=True)
]
print(f"Original edge lengths:")
print(f"  Mean: {np.mean(original_edge_lengths):.2f}")
print(f"  Min: {np.min(original_edge_lengths):.2f}")
print(f"  Max: {np.max(original_edge_lengths):.2f}")
print(f"  Std: {np.std(original_edge_lengths):.2f}")

# Resample with spacing of 10 units
target_spacing = 10.0
print(f"\nResampling with target spacing: {target_spacing}")
resampled = skeleton.resample(spacing=target_spacing)

print(f"Resampled: {resampled}")
print(f"Total length: {resampled.get_total_length():.2f}")

# Check new spacing
resampled_edge_lengths = [
    data.get("length", 0.0) for _, _, data in resampled.edges(data=True)
]
print(f"\nResampled edge lengths:")
print(f"  Mean: {np.mean(resampled_edge_lengths):.2f}")
print(f"  Min: {np.min(resampled_edge_lengths):.2f}")
print(f"  Max: {np.max(resampled_edge_lengths):.2f}")
print(f"  Std: {np.std(resampled_edge_lengths):.2f}")

# Test 3: Snap to mesh surface
print("\n" + "=" * 70)
print("\nTest 3: Snap to mesh surface")
print("-" * 70)

# Make a copy to test snapping
skeleton_copy = skeleton.copy_skeleton()

# Check how many points are outside
positions = skeleton_copy.get_all_positions()
inside_mask = mesh_mgr.mesh.contains(positions)
num_outside = np.sum(~inside_mask)
print(f"Points outside mesh before snapping: {num_outside}/{len(positions)}")

if num_outside > 0:
    # Snap to surface
    n_moved, mean_dist = skeleton_copy.snap_to_mesh_surface(
        mesh_mgr.mesh, project_outside_only=True
    )

    print(f"Snapped {n_moved} points to surface")
    print(f"Mean movement distance: {mean_dist:.4f}")

    # Check again
    positions_after = skeleton_copy.get_all_positions()
    inside_mask_after = mesh_mgr.mesh.contains(positions_after)
    num_outside_after = np.sum(~inside_mask_after)
    print(
        f"Points outside mesh after snapping: {num_outside_after}/{len(positions_after)}"
    )
else:
    print("All points already inside mesh - no snapping needed")

# Test 4: Combined workflow
print("\n" + "=" * 70)
print("\nTest 4: Combined workflow (resample + snap)")
print("-" * 70)

# Resample then snap
workflow_skeleton = skeleton.resample(spacing=15.0)
print(f"After resampling: {workflow_skeleton}")

n_moved, mean_dist = workflow_skeleton.snap_to_mesh_surface(
    mesh_mgr.mesh, project_outside_only=True
)
print(f"Snapped {n_moved} points, mean distance: {mean_dist:.4f}")
print(f"Final: {workflow_skeleton}")

# Save the processed skeleton
output_path = r"data\test_skeleton_processed.polylines.txt"
workflow_skeleton.to_txt(output_path)
print(f"\nSaved processed skeleton to: {output_path}")

print("\n" + "=" * 70)
print("\nAll feature tests completed successfully!")
