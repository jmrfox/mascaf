# %% [markdown]
# # Demo: Branching Skeleton

# %%
from mascaf import *

import logging

logging.basicConfig(level=logging.INFO)

# %%
mm = MeshManager(mesh_path="../../data/demo/test_branching_3.obj")
mm.print_mesh_analysis()
raw_skeleton = SkeletonGraph.from_txt(f"../../data/demo/test_branching_3.polylines.txt")
# raw_skeleton.prune_short_branches_inplace(min_length_percentile=1)
mm.visualize_mesh_3d(title="Branching test model", skel=raw_skeleton)

# %%
optimizer_options = BasisOptimizerOptions(
    do_pruning=False,
    do_snapping=True,
    do_forcing=True,
    max_iterations=5,
    step_size=0.01,
    smoothing_weight=0.1,
    preserve_terminal_nodes=False,
    preserve_branch_nodes=False,
)
skeleton = raw_skeleton

# %%
max_edge_length = 1.0

swc_filepath = "../../data/demo/test_branching_3.swc"

radius_strategy = "equivalent_area"
print(f"Computing skeleton for radius_strategy={radius_strategy} ...", end="")
morph = CableFitter(
    FitOptions(
        max_edge_length=max_edge_length,
        radius_strategy=radius_strategy,
        basis_optimizer_options=optimizer_options,
    )
).fit(
    mm.mesh,
    skeleton,
)
# write swc to file
morph.to_swc_file(swc_filepath)
# validation
validator = Validation(mm, skeleton, morph)
validator.full_validation()

# %%
# plot using swctools
from swctools import SWCModel, plot_model

model = SWCModel.from_swc_file(swc_filepath)
model.print_attributes(node_info=False, edge_info=False)
title = f"Branching Skeleton"
fig = plot_model(
    swc_model=model,
    title=title,
    plot_endcaps=True,
    hide_axes=True,
    centroid_color="black",
    centroid_line_width=5,
    opacity=0.5,
)
fig.show()

# %%
# normalize radii to match mesh surface area
morph.scale_radii_to_match_mesh(
    mm.mesh, metric="surface_area", account_for_overlaps=False
)

# save normalized to file
swc_filepath_normalized = "../../data/demo/test_branching_3_normalized.swc"
morph.to_swc_file(swc_filepath_normalized)

# load and plot
swc_model = SWCModel.from_swc_file(swc_filepath_normalized)
fig = plot_model(
    swc_model=swc_model,
    slider=False,
    title="",
    hide_axes=True,
    centroid_color="black",
    centroid_line_width=5,
    opacity=0.5,
    plot_endcaps=True,
)
fig.show()

validator = Validation(mm, skeleton, morph)
validator.full_validation()
