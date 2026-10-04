# %% [markdown]
# # TS pipeline
#
# Fit one spine from mesh and skeleton through basis optimization, radius estimation, and validation.

# %% [markdown]
# ## Imports
#
# Load MaSCaF fitting tools and logging. Plots use the principal-axis camera:
# the longest axis lies across the figure unless ``orientation="vertical"``.

# %%
import logging
import os
from typing import TYPE_CHECKING

from mascaf import *
from mascaf.cable_fitting import _compute_morphology_node_radii
from swctools import SWCModel

if TYPE_CHECKING:
    import plotly.graph_objects as go  # type: ignore

logging.basicConfig(level=logging.INFO)

print("✅ Libraries imported successfully!")


# %% [markdown]
# ## Load mesh and skeleton
#
# Load the processed mesh and MCF skeleton for the chosen spine, then take `max_edge_length` and basis-optimizer options from the fit oracle. The figures are the mesh alone and the mesh with the skeleton, written under `viz/`.

# %%
spine_idx = 1
qst = 0.5
mcst = 5

pdf_scale = 2

mesh_path = f"../../data/mesh/processed/TS{spine_idx}.obj"
mm = MeshManager(mesh_path=mesh_path)
model_length = mm.bounding_box_diagonal()

polylines_name = f"TS{spine_idx}_qst{qst}_mcst{mcst}"
skeleton = SkeletonGraph.from_txt(
    f"../../data/mcf_skeletons/{polylines_name}.polylines.txt"
)

suggested = suggest_fit_parameters(
    mm,
    skeleton,
)
for line in suggested.rationale:
    print(f"Oracle: {line}")

params = suggested
params.basis_optimizer_options.active_resample = False
params.basis_optimizer_options.active_resample_min_fraction = 0.02
# params.basis_optimizer_options.active_resample_max_fraction = 0.4

mel_tag = f"{params.max_edge_length:0.2f}"

print(params)

# %%

fig_out_dir = f"../../viz/ts{spine_idx}"
os.makedirs(fig_out_dir, exist_ok=True)

mesh_fig: "go.Figure"
mesh_fig, camera = visualize_mesh_3d(
    mm,
    skel=None,
    show_axes=False,
    title="",
    orientation="horizontal",
    return_camera=True,
    eye_scale=1.0,
)
mesh_fig.write_image(
    f"{fig_out_dir}/TS{spine_idx}_mesh.pdf",
    format="pdf",
    engine="kaleido",
    width=600,
    height=450,
    scale=pdf_scale,
)
mesh_fig.show()

mesh_skel_fig: "go.Figure" = visualize_mesh_3d(
    mm,
    skel=skeleton,
    show_axes=False,
    title="",
    orientation="horizontal",
    camera=camera,
    # eye_scale=1.0,
)
mesh_skel_fig.write_image(
    f"{fig_out_dir}/TS{spine_idx}_mesh_skel.pdf",
    format="pdf",
    engine="kaleido",
    width=600,
    height=450,
    scale=pdf_scale,
)
mesh_skel_fig.show()

# %% [markdown]
# ## Basis optimization
#
# Resample the skeleton into a morphology basis and move nodes with `BasisOptimizer` before any radius fitting. The figure overlays the original basis (red) and the optimized basis (blue).

# %%
print(params.basis_optimizer_options)

# %%
basis = MorphologyGraph.from_skeleton_graph_resample(
    skeleton,
    float(params.max_edge_length),
)
print(
    f"Initial basis: {basis.number_of_nodes()} nodes, "
    f"{basis.number_of_edges()} edges"
)

optimizer = BasisOptimizer(basis, mm.mesh, params.basis_optimizer_options)
optimized_basis = optimizer.optimize()
stats = optimizer.get_optimization_stats()
print("Basis optimization statistics:")
for key, value in stats.items():
    print(f"  {key}: {value}")

basis_opt_fig: "go.Figure" = visualize_mesh_3d(
    mm,
    skel=[basis, optimized_basis],
    show_axes=False,
    title="",
    orientation="horizontal",
    camera=camera,
    skel_color=["red", "blue"],
    skel_line_width=3.0,
    skel_marker_size=2.0,
)
basis_opt_fig.write_image(
    f"{fig_out_dir}/TS{spine_idx}_mel{mel_tag}_basis_opt.pdf",
    format="pdf",
    engine="kaleido",
    width=600,
    height=450,
    scale=pdf_scale,
)
basis_opt_fig.show()

# %% [markdown]
# ## Cable fitting
#
# Estimate node radii on the optimized basis with the equivalent-area strategy, write the SWC. The figure is the fitted morphology.

# %%
swc_out_dir = f"../../data/swc/current/{polylines_name}"
swc_filepath = f"{swc_out_dir}/TS{spine_idx}_mel{mel_tag}.swc"

if not os.path.exists(swc_out_dir):
    os.makedirs(swc_out_dir)

fit_options = FitOptions(
    max_edge_length=params.max_edge_length,
    radius_strategy="equivalent_area",
    section_probe_eps=1e-4,
    section_probe_tries=3,
    multi_tangent_reduction="mean",
    basis_optimizer_options=None,
)
morph = optimized_basis.copy()
_compute_morphology_node_radii(morph, mm.mesh, fit_options)

morph.to_swc_file(swc_filepath)

model = SWCModel.from_swc_file(swc_filepath)
model.print_attributes(node_info=False, edge_info=False)
morph_fig = visualize_cable_3d(
    model,
    slider=False,
    title="",
    width=800,
    height=600,
    show_axes=False,
    orientation="horizontal",
    camera=camera,
)
morph_filename = (
    f"{fig_out_dir}/TS{spine_idx}_mel{mel_tag}_" f"morph.pdf"
)
morph_fig.write_image(
    morph_filename,
    format="pdf",
    engine="kaleido",
    width=600,
    height=450,
    scale=pdf_scale,
)
morph_fig.show()

# %% [markdown]
# ### Validation after cable fitting

# %%
validator = Validation(mm, skeleton, morph)
validator.full_validation()

# %% [markdown]
# ## Terminal extension

# %%
morph.extend_terminals(length_scale=0.5, radius_fraction=0.5)
validator = Validation(mm, skeleton, morph)
validator.full_validation()

# %% [markdown]
# ## Surface-area normalization
#
# Scale radii so the morphology surface area matches the mesh, without overlap correction.

# %%
morph.scale_radii_to_match_mesh(
    mm.mesh, metric="surface_area", account_for_overlaps=False
)

swc_filepath = f"{swc_out_dir}/TS{spine_idx}_mel{mel_tag}_norm.swc"
morph.to_swc_file(swc_filepath)

swc_model = SWCModel.from_swc_file(swc_filepath)
swc_model.print_attributes(node_info=False, edge_info=False)
norm_fig: "go.Figure" = visualize_cable_3d(
    swc_model,
    slider=False,
    title="",
    width=800,
    height=600,
    show_axes=False,
    orientation="horizontal",
    camera=camera,
)
norm_filename = (
    f"{fig_out_dir}/TS{spine_idx}_mel{mel_tag}_" f"morph_norm.pdf"
)
norm_fig.write_image(
    norm_filename,
    format="pdf",
    engine="kaleido",
    width=600,
    height=450,
    scale=pdf_scale,
)
norm_fig.show()

print(swc_filepath)

# %% [markdown]
# ### Validation with area fit

# %%
validator = Validation(mm, skeleton, morph)
validator.full_validation()

# %% [markdown]
# ## Morphology versus skeleton
#
# Overlay the normalized morphology (translucent) with the original skeleton points.

# %%
skel_pointset = skeleton.to_point_set()

vs_fig: "go.Figure" = visualize_cable_3d(
    swc_model,
    opacity=0.2,
    title="",
    width=800,
    height=600,
    show_axes=False,
    orientation="horizontal",
    camera=camera,
    point_set=skel_pointset,
    point_color="red",
    point_size=model_length * 0.0015,
)
vs_filename = (
    f"{fig_out_dir}/TS{spine_idx}_mel{mel_tag}_"
    f"morph_vs_skel.pdf"
)
vs_fig.write_image(
    vs_filename,
    format="pdf",
    engine="kaleido",
    width=600,
    height=450,
    scale=pdf_scale,
)
vs_fig.show()

# %%
visualize_mesh_cable_3d(mm, skeleton, morph, eye_scale=0.6)

# %%
