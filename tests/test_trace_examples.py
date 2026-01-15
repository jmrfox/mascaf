from pathlib import Path

import numpy as np

from mcf2swc import (
    example_mesh,
    fit_swc,
    SWCFitOptions,
    PolylinesSkeleton,
)


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "demo"


def test_cylinder_trace_non_empty():
    mesh = example_mesh("cylinder")

    pls = PolylinesSkeleton.from_txt(str(DATA / "cylinder.polylines.txt"))
    assert pls.total_points() > 0

    # Use a reasonable spacing to ensure several samples along the path
    opts = SWCFitOptions(spacing=0.5, radius_strategy="equivalent_area")
    G = fit_swc(mesh, pls, options=opts)

    assert G.number_of_nodes() > 0, "Cylinder trace produced no nodes"
    assert G.number_of_edges() > 0, "Cylinder trace produced no edges"


def test_torus_trace_non_empty():
    mesh = example_mesh("torus")

    pls = PolylinesSkeleton.from_txt(str(DATA / "torus.polylines.txt"))
    assert pls.total_points() > 0

    # Spacing small enough to capture curvature around the torus major ring
    opts = SWCFitOptions(spacing=0.5, radius_strategy="equivalent_area")
    G = fit_swc(mesh, pls, options=opts)

    assert G.number_of_nodes() > 0, "Torus trace produced no nodes"
    assert G.number_of_edges() > 0, "Torus trace produced no edges"
