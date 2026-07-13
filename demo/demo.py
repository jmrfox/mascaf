# %% [markdown]
# # MASCAF Demo
#
# This notebook walks through the full MASCAF pipeline:
#
# 1. Load a demo mesh and bundled skeleton.
# 2. Visualize the mesh and skeleton.
# 3. Fit a cable-graph morphology with `CableFitter`.
# 4. Visualize the resulting SWC model.
# 5. Run validation and print a summary.
#
# **Requirements:** `mascaf`, `swctools`
# (`pip install git+https://github.com/jmrfox/swctools.git`)

# %%
import logging
from importlib.resources import files
from pathlib import Path

from mascaf import (
    BasisOptimizerOptions,
    CableFitter,
    FitOptions,
    MeshManager,
    SkeletonGraph,
    Validation,
)
from swctools import SWCModel, plot_model

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# %% [markdown]
# ## Choose a demo model
#
# Set `demo_model` to one of `"torus"`, `"cylinder"`, or `"branching"`.

# %%
demo_model = "branching"  # "torus" | "cylinder" | "branching"

_DEMO = files("mascaf.demo")
mesh_path = Path(str(_DEMO / f"{demo_model}.obj"))
skeleton_path = Path(str(_DEMO / f"{demo_model}.polylines.txt"))
output_dir = Path("outputs")
output_dir.mkdir(parents=True, exist_ok=True)

print(f"Mesh:     {mesh_path}")
print(f"Skeleton: {skeleton_path}")

# %% [markdown]
# ## Load mesh and skeleton

# %%
mm = MeshManager(mesh_path=str(mesh_path))
skeleton = SkeletonGraph.from_txt(str(skeleton_path))

diagonal = mm.bounding_box_diagonal()
print(f"Bounding box diagonal: {diagonal:.4f}")
print(f"Skeleton: {skeleton.number_of_nodes()} nodes")

# %% [markdown]
# ## Visualize mesh

# %%
fig = mm.visualize_mesh_3d(skel=None, show_axes=False, title="")
fig.show()

# %% [markdown]
# ## Visualize mesh with skeleton overlay

# %%
fig = mm.visualize_mesh_3d(skel=skeleton, show_axes=False, title="")
fig.show()

# %% [markdown]
# ## Configure fitting options
#
# `max_edge_length` is set to 10 % of the bounding box diagonal by default.
# Adjust `max_edge_length_fraction` or set `max_edge_length` directly.

# %%
max_edge_length_fraction = 0.1
max_edge_length = max_edge_length_fraction * diagonal

basis_options = BasisOptimizerOptions(
    do_snapping=True,
    do_forcing=False,
    max_iterations=10,
    lambda_smooth=0.1,
)

fit_options = FitOptions(
    max_edge_length=max_edge_length,
    radius_strategy="equivalent_area",
    basis_optimizer_options=basis_options,
)

print(f"max_edge_length: {max_edge_length:.4f}")

# %% [markdown]
# ## Run CableFitter

# %%
morphology = CableFitter(fit_options).fit(mm, skeleton)

swc_path = output_dir / f"{demo_model}.swc"
morphology.to_swc_file(str(swc_path))
print(f"SWC written to: {swc_path}")
print(
    f"MorphologyGraph: {morphology.number_of_nodes()} nodes, "
    f"{morphology.number_of_edges()} edges"
)

# %% [markdown]
# ## Visualize morphology

# %%
model = SWCModel.from_swc_file(str(swc_path))
model.print_attributes(node_info=False, edge_info=False)

fig = plot_model(
    swc_model=model,
    slider=False,
    title="",
    width=800,
    height=600,
    show_axes=False,
    plot_endcaps=True,
)
fig.show()

# %% [markdown]
# ## Validation

# %%
validator = Validation(mm, skeleton, morphology)
validator.full_validation()

volume_result = validator.compare_volumes()
area_result = validator.compare_surface_areas()

print(
    f"Volume ratio:      {volume_result['ratio']:.4f} "
    f"(relative error {volume_result['relative_error']:.2%})"
)
print(
    f"Surface area ratio: {area_result['ratio']:.4f} "
    f"(relative error {area_result['relative_error']:.2%})"
)

# %%
