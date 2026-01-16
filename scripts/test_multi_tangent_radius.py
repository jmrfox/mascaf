"""
Test script to verify multi-tangent radius computation.
"""

import numpy as np
from mcf2swc import (
    SkeletonGraph,
    FitOptions,
    fit_morphology,
    example_mesh,
)


def create_test_skeleton():
    """Create a test skeleton with terminal and branch nodes."""
    # Create a simple Y-shaped skeleton
    points = [
        [0.0, 0.0, -2.0],  # Terminal node 1
        [0.0, 0.0, -1.0],  # Branch node
        [0.0, 0.0, 0.0],  # Branch node
        [0.0, 0.0, 1.0],  # Terminal node 2
        [1.0, 0.0, 0.0],  # Terminal node 3
        [-1.0, 0.0, 0.0],  # Terminal node 4
    ]

    # Create edges to form a Y shape
    edges = [
        (0, 1),  # Terminal 1 to branch
        (1, 2),  # Branch to center
        (2, 3),  # Center to terminal 2
        (2, 4),  # Center to terminal 3
        (2, 5),  # Center to terminal 4
    ]

    skeleton = SkeletonGraph()
    for i, point in enumerate(points):
        skeleton.add_node(i, pos=np.array(point, dtype=float))

    for u, v in edges:
        skeleton.add_edge(u, v)

    return skeleton


def test_multi_tangent_radius():
    print("Testing multi-tangent radius computation...")

    # Create test mesh and skeleton
    mesh = example_mesh("cylinder", radius=1.5, height=6.0, sections=32)
    skeleton = create_test_skeleton()

    print(f"Created skeleton with {skeleton.number_of_nodes()} nodes")

    # Test different reduction methods
    reduction_methods = ["mean", "min", "max", "median"]

    for method in reduction_methods:
        print(f"\n--- Testing {method} reduction ---")

        opts = FitOptions(
            spacing=0.5,
            radius_strategy="equivalent_area",
            multi_tangent_reduction=method,
        )

        try:
            morphology = fit_morphology(mesh, skeleton, opts)

            print(f"Successfully created morphology with {len(morphology.nodes)} nodes")

            # Check some radii
            radii = [
                node_data.get("radius", 0.0) for node_data in morphology.nodes.values()
            ]
            print(f"Radius range: {min(radii):.4f} to {max(radii):.4f}")
            print(f"Mean radius: {np.mean(radii):.4f}")

        except Exception as e:
            print(f"Error with {method} reduction: {e}")


def test_tangent_computation():
    print("\n\nTesting tangent computation...")

    from mcf2swc.graph_fitting import _compute_node_tangents

    skeleton = create_test_skeleton()

    for node in skeleton.nodes():
        tangents = _compute_node_tangents(skeleton, node)
        neighbors = list(skeleton.neighbors(node))

        print(f"Node {node}: degree={len(neighbors)}, tangents={len(tangents)}")
        for i, tangent in enumerate(tangents):
            print(f"  Tangent {i}: {tangent}")


def test_radius_algorithms():
    print("\n\nTesting different radius algorithms...")

    mesh = example_mesh("cylinder", radius=1.5, height=6.0, sections=32)
    skeleton = create_test_skeleton()

    strategies = [
        "equivalent_area",
        "equivalent_perimeter",
        "section_median",
        "section_circle_fit",
        "nearest_surface",
    ]

    for strategy in strategies:
        print(f"\n--- Testing {strategy} strategy ---")

        opts = FitOptions(
            spacing=0.5,
            radius_strategy=strategy,
            multi_tangent_reduction="median",
        )

        try:
            morphology = fit_morphology(mesh, skeleton, opts)

            radii = [
                node_data.get("radius", 0.0) for node_data in morphology.nodes.values()
            ]
            print(f"Radius range: {min(radii):.4f} to {max(radii):.4f}")
            print(f"Mean radius: {np.mean(radii):.4f}")

        except Exception as e:
            print(f"Error with {strategy} strategy: {e}")


if __name__ == "__main__":
    test_tangent_computation()
    test_multi_tangent_radius()
    test_radius_algorithms()
    print("\nTest complete!")
