"""
Example demonstrating radius normalization in the radius optimizer.

This example shows how to use the normalize option to scale all radii
so that the total surface area or volume of the skeleton matches the mesh.
"""

import trimesh

from mascaf.morphology_graph import MorphologyGraph
from mascaf.radius_optimizer import RadiusOptimizer, RadiusOptimizerOptions


def example_normalize_surface_area():
    """Normalize radii to match mesh surface area."""
    # Load your mesh and skeleton
    mesh = trimesh.load("path/to/mesh.obj")
    skeleton = MorphologyGraph.from_swc_file("path/to/skeleton.swc")

    # Configure optimizer with surface area normalization
    options = RadiusOptimizerOptions(
        n_longitudinal=5,
        n_radial=12,
        max_iterations=20,
        normalize=True,
        normalize_metric="surface_area",
        verbose=True,
    )

    # Run optimization
    optimizer = RadiusOptimizer(skeleton, mesh, options=options)
    optimized_skeleton = optimizer.optimize()

    # Save result (MorphologyGraph has to_swc_file method)
    optimized_skeleton.to_swc_file("output_normalized_sa.swc")


def example_normalize_volume():
    """Normalize radii to match mesh volume."""
    # Load your mesh and skeleton
    mesh = trimesh.load("path/to/mesh.obj")
    skeleton = MorphologyGraph.from_swc_file("path/to/skeleton.swc")

    # Configure optimizer with volume normalization
    options = RadiusOptimizerOptions(
        n_longitudinal=5,
        n_radial=12,
        max_iterations=20,
        normalize=True,
        normalize_metric="volume",
        verbose=True,
    )

    # Run optimization
    optimizer = RadiusOptimizer(skeleton, mesh, options=options)
    optimized_skeleton = optimizer.optimize()

    # Save result
    optimized_skeleton.to_swc_file("output_normalized_vol.swc")


def example_no_normalization():
    """Run optimization without normalization (default behavior)."""
    # Load your mesh and skeleton
    mesh = trimesh.load("path/to/mesh.obj")
    skeleton = MorphologyGraph.from_swc_file("path/to/skeleton.swc")

    # Configure optimizer without normalization
    options = RadiusOptimizerOptions(
        n_longitudinal=5,
        n_radial=12,
        max_iterations=20,
        normalize=False,  # This is the default
        verbose=True,
    )

    # Run optimization
    optimizer = RadiusOptimizer(skeleton, mesh, options=options)
    optimized_skeleton = optimizer.optimize()

    # Save result
    optimized_skeleton.to_swc_file("output_no_normalization.swc")


def example_with_overlap_removal():
    """Normalize while subtracting branch-point overlap corrections."""
    # Load your mesh and skeleton
    mesh = trimesh.load("path/to/mesh.obj")
    skeleton = MorphologyGraph.from_swc_file("path/to/skeleton.swc")

    # Configure optimizer with normalization and branch overlap accounting
    # This subtracts overlap corrections at branch points (degree > 2)
    options = RadiusOptimizerOptions(
        n_longitudinal=5,
        n_radial=12,
        max_iterations=20,
        normalize=True,
        normalize_metric="surface_area",
        account_for_overlaps=True,
        verbose=True,
    )

    # Run optimization
    optimizer = RadiusOptimizer(skeleton, mesh, options=options)
    optimized_skeleton = optimizer.optimize()

    # Save result
    optimized_skeleton.to_swc_file("output_with_overlap_removal.swc")


if __name__ == "__main__":
    print("Example 1: Normalize to match surface area")
    example_normalize_surface_area()

    print("\nExample 2: Normalize to match volume")
    example_normalize_volume()

    print("\nExample 3: No normalization")
    example_no_normalization()

    print("\nExample 4: Normalize with overlap removal")
    example_with_overlap_removal()
