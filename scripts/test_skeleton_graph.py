"""Test the new SkeletonGraph class."""

import numpy as np
from mcf2swc.skeleton import SkeletonGraph

print("Testing SkeletonGraph class\n")
print("=" * 70)

# Test 1: Load from polylines text file
print("\nTest 1: Load from polylines text file")
print("-" * 70)

skeleton = SkeletonGraph.from_txt(
    r"data\mcf_skeletons\TS2_qst0.5_mcst10.polylines.txt", tolerance=1e-6
)

print(f"Loaded: {skeleton}")
print(f"\nStatistics:")
stats = skeleton.get_statistics()
for key, value in stats.items():
    print(f"  {key}: {value}")

# Test 2: Node classification
print("\n" + "=" * 70)
print("\nTest 2: Node classification")
print("-" * 70)

terminal_nodes = skeleton.get_terminal_nodes()
branch_nodes = skeleton.get_branch_nodes()
continuation_nodes = skeleton.get_continuation_nodes()

print(f"Terminal nodes (degree 1): {len(terminal_nodes)}")
print(f"  Node IDs: {sorted(terminal_nodes)}")

print(f"\nBranch nodes (degree 3+): {len(branch_nodes)}")
print(f"  Node IDs: {sorted(branch_nodes)}")

print(f"\nContinuation nodes (degree 2): {len(continuation_nodes)}")
print(f"  Count: {len(continuation_nodes)}")

# Show degree distribution
print("\nDegree distribution:")
degree_counts = {}
for node in skeleton.nodes():
    deg = skeleton.degree(node)
    degree_counts[deg] = degree_counts.get(deg, 0) + 1

for degree in sorted(degree_counts.keys()):
    print(f"  Degree {degree}: {degree_counts[degree]} nodes")

# Test 3: Node positions
print("\n" + "=" * 70)
print("\nTest 3: Node positions")
print("-" * 70)

# Show positions of a few nodes
for node in list(skeleton.nodes())[:3]:
    pos = skeleton.get_node_position(node)
    degree = skeleton.degree(node)
    node_type = (
        "terminal" if degree == 1 else ("branch" if degree >= 3 else "continuation")
    )
    print(
        f"Node {node} ({node_type}, degree {degree}): [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]"
    )

# Test 4: Conversion back to polylines
print("\n" + "=" * 70)
print("\nTest 4: Convert back to polylines")
print("-" * 70)

polylines = skeleton.to_polylines()
print(f"Converted to {len(polylines)} polylines")

# Show info about first few polylines
for i, pl in enumerate(polylines[:3]):
    print(f"  Polyline {i}: {len(pl)} points")
    if len(pl) > 0:
        print(f"    Start: [{pl[0][0]:.2f}, {pl[0][1]:.2f}, {pl[0][2]:.2f}]")
        print(f"    End:   [{pl[-1][0]:.2f}, {pl[-1][1]:.2f}, {pl[-1][2]:.2f}]")

# Test 5: Bounds and centroid
print("\n" + "=" * 70)
print("\nTest 5: Bounds and centroid")
print("-" * 70)

bounds = skeleton.bounds()
if bounds:
    print("Bounding box:")
    for axis, (lo, hi) in bounds.items():
        print(f"  {axis}: [{lo:.2f}, {hi:.2f}] (range: {hi-lo:.2f})")

centroid = skeleton.centroid()
if centroid is not None:
    print(f"\nCentroid: [{centroid[0]:.2f}, {centroid[1]:.2f}, {centroid[2]:.2f}]")

# Test 6: Copy
print("\n" + "=" * 70)
print("\nTest 6: Copy skeleton")
print("-" * 70)

skeleton_copy = skeleton.copy_skeleton()
print(f"Original: {skeleton}")
print(f"Copy:     {skeleton_copy}")

# Verify positions are the same
orig_pos = skeleton.get_all_positions()
copy_pos = skeleton_copy.get_all_positions()
print(f"\nPositions match: {np.allclose(orig_pos, copy_pos)}")

# Modify copy and verify original is unchanged
if len(list(skeleton_copy.nodes())) > 0:
    first_node = list(skeleton_copy.nodes())[0]
    orig_first_pos = skeleton.get_node_position(first_node).copy()
    skeleton_copy.set_node_position(
        first_node, orig_first_pos + np.array([100, 100, 100])
    )
    new_first_pos = skeleton_copy.get_node_position(first_node)

    print(f"\nModified first node in copy:")
    print(
        f"  Original graph node {first_node}: [{orig_first_pos[0]:.2f}, {orig_first_pos[1]:.2f}, {orig_first_pos[2]:.2f}]"
    )
    print(
        f"  Copy graph node {first_node}:     [{new_first_pos[0]:.2f}, {new_first_pos[1]:.2f}, {new_first_pos[2]:.2f}]"
    )
    print(
        f"  Original unchanged: {np.allclose(skeleton.get_node_position(first_node), orig_first_pos)}"
    )

# Test 7: Save to file
print("\n" + "=" * 70)
print("\nTest 7: Save to file")
print("-" * 70)

output_path = r"data\test_skeleton_graph_output.polylines.txt"
skeleton.to_txt(output_path)
print(f"Saved to: {output_path}")

# Reload and verify
skeleton_reloaded = SkeletonGraph.from_txt(output_path, tolerance=1e-6)
print(f"Reloaded: {skeleton_reloaded}")

print("\n" + "=" * 70)
print("\nAll tests completed successfully!")
