from pathlib import Path
import tempfile

import numpy as np

from mcf2swc import (
    example_mesh,
    fit_swc,
    SWCFitOptions,
    PolylinesSkeleton,
    SWCModel,
)


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "demo"


def test_swc_save_reload_preserves_edges():
    """Test that saving and reloading an SWC model preserves edges."""
    mesh = example_mesh("cylinder")
    pls = PolylinesSkeleton.from_txt(str(DATA / "cylinder.polylines.txt"))

    opts = SWCFitOptions(spacing=1.0, radius_strategy="equivalent_area")
    model_original = fit_swc(mesh, pls, options=opts)

    assert model_original.number_of_nodes() > 0, "Original model has no nodes"
    assert model_original.number_of_edges() > 0, "Original model has no edges"

    original_nodes = model_original.number_of_nodes()
    original_edges = model_original.number_of_edges()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".swc", delete=False) as f:
        temp_path = f.name

    try:
        model_original.to_swc_file(temp_path)

        model_reloaded = SWCModel.from_swc_file(temp_path)

        reloaded_nodes = model_reloaded.number_of_nodes()
        reloaded_edges = model_reloaded.number_of_edges()

        assert (
            reloaded_nodes == original_nodes
        ), f"Node count mismatch: original={original_nodes}, reloaded={reloaded_nodes}"
        assert reloaded_edges == original_edges, (
            f"Edge count mismatch: original={original_edges}, reloaded={reloaded_edges}. "
            f"Edges were lost during save/reload!"
        )

    finally:
        Path(temp_path).unlink(missing_ok=True)
