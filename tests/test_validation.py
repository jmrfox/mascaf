from pathlib import Path

import numpy as np

from mascaf import (
    example_mesh,
    fit_morphology,
    FitOptions,
    SkeletonGraph,
    Validation,
)


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "demo"


def test_cylinder_trace_non_empty():
    """Test cylinder trace and perform validation."""
    mesh = example_mesh("cylinder")

    skel = SkeletonGraph.from_txt(str(DATA / "cylinder.polylines.txt"))
    assert skel.number_of_nodes() > 0

    # Use a reasonable spacing to ensure several samples along the path
    opts = FitOptions(max_edge_length=0.5, radius_strategy="equivalent_area")
    G = fit_morphology(mesh, skel, options=opts)

    assert G.number_of_nodes() > 0, "Cylinder trace produced no nodes"
    assert G.number_of_edges() > 0, "Cylinder trace produced no edges"

    # Perform validation
    validator = Validation(mesh, skel, G)
    vol_result = validator.compare_volumes()
    area_result = validator.compare_surface_areas()
    assert vol_result["relative_error"] < 0.05
    assert area_result["relative_error"] < 0.05


def test_torus_trace_non_empty():
    """Test torus trace and perform validation."""
    mesh = example_mesh("torus")

    skel = SkeletonGraph.from_txt(str(DATA / "torus.polylines.txt"))
    assert skel.number_of_nodes() > 0

    # Spacing small enough to capture curvature around the torus major ring
    opts = FitOptions(max_edge_length=0.5, radius_strategy="equivalent_area")
    G = fit_morphology(mesh, skel, options=opts)

    assert G.number_of_nodes() > 0, "Torus trace produced no nodes"
    assert G.number_of_edges() > 0, "Torus trace produced no edges"

    # Perform validation
    validator = Validation(mesh, skel, G)
    vol_result = validator.compare_volumes()
    area_result = validator.compare_surface_areas()
    assert vol_result["relative_error"] < 0.05
    assert area_result["relative_error"] < 0.05
