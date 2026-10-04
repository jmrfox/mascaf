# %% [markdown]
# # Human demo: cable fit figures
#
# Load the human demo mesh and skeleton, suggest fit parameters from the oracle,
# run basis optimization and radius fitting, then validate and normalize to mesh
# surface area.

# %%
import logging

from mascaf import *
from swctools import SWCModel

logging.basicConfig(level=logging.INFO)

print("✅ Libraries imported successfully!")

# %%
name = "human"
demo_dir = f"../../data/demo"

mm = MeshManager(mesh_path=f"{demo_dir}/{name}.obj")
raw_skeleton = SkeletonGraph.from_txt(f"{demo_dir}/{name}.polylines.txt")
# Remove short terminal twigs (fraction = percentile / 100, e.g. 0.2 → 20th percentile).
raw_skeleton.prune_short_branches_inplace(min_length_fraction=0.2)
skeleton = raw_skeleton

suggested = suggest_fit_parameters(mm, skeleton)
for line in suggested.rationale:
    print(f"Oracle: {line}")
print(suggested)

basis_optimizer_options = suggested.basis_optimizer_options
basis_optimizer_options.do_forcing = False

fit_options = FitOptions(
    max_edge_length=suggested.max_edge_length,
    radius_strategy="equivalent_area",
    section_probe_eps=1e-4,
    section_probe_tries=3,
    multi_tangent_reduction="median",
    basis_optimizer_options=suggested.basis_optimizer_options,
)

# %%
mesh_fig, camera = visualize_mesh_3d(
    mm,
    skel=None,
    show_axes=False,
    title="",
    orientation="horizontal",
    return_camera=True,
    eye_scale=1.0,
)
mesh_fig.show()

mesh_skel_fig = visualize_mesh_3d(
    mm,
    skel=skeleton,
    show_axes=False,
    title="",
    orientation="horizontal",
    camera=camera,
)
mesh_skel_fig.show()

# %% [markdown]
# ## Cable fit

# %%
swc_filepath = f"{demo_dir}/{name}.swc"

morph = CableFitter(options=fit_options).fit(mm, skeleton)
morph.to_swc_file(swc_filepath)

model = SWCModel.from_swc_file(swc_filepath)
model.print_attributes(node_info=False, edge_info=False)

cable_fig = visualize_cable_3d(
    model,
    slider=False,
    title="",
    width=800,
    height=600,
    show_axes=False,
    orientation="horizontal",
    camera=camera,
)
cable_fig.show()

validator = Validation(mm, skeleton, morph)
validator.full_validation()

# %% [markdown]
# ## Terminal extension and surface-area normalization

# %%
morph.extend_terminals(length_scale=0.5, radius_fraction=0.5)
validator = Validation(mm, skeleton, morph)
validator.full_validation()

morph.scale_radii_to_match_mesh(
    mm.mesh, metric="surface_area", account_for_overlaps=False
)

swc_norm_path = f"{demo_dir}/{name}_norm.swc"
morph.to_swc_file(swc_norm_path)

swc_model = SWCModel.from_swc_file(swc_norm_path)
swc_model.print_attributes(node_info=False, edge_info=False)

norm_fig = visualize_cable_3d(
    swc_model,
    slider=False,
    title="",
    width=800,
    height=600,
    show_axes=False,
    orientation="horizontal",
    camera=camera,
)
norm_fig.show()

validator = Validation(mm, skeleton, morph)
validator.full_validation()

# %%
visualize_mesh_cable_3d(
    mm,
    skeleton,
    morph,
    show_axes=False,
    orientation="horizontal",
    camera=camera,
    eye_scale=0.9,
)

# %%
