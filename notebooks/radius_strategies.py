# %% [markdown]
# # Comparing radius strategies

# %% [markdown]
# ## Setup

# %%
from mascaf import *
from swctools import SWCModel, FrustaSet, PointSet, plot_model
import logging

logging.basicConfig(level=logging.INFO)
import os

print("✅ Libraries imported successfully!")

# %% [markdown]
# ## Load Mesh

# %%
spine_idx = 1
mcf_qst = 0.5
mcf_mcst = 5
obj_name = f"TS{spine_idx}"
polylines_name = f"TS{spine_idx}_qst{mcf_qst}_mcst{mcf_mcst}"
mm = MeshManager(mesh_path=f"../data/mesh/processed/{obj_name}.obj")
raw_skeleton = SkeletonGraph.from_txt(
    f"../data/mcf_skeletons/{polylines_name}.polylines.txt"
)
raw_skeleton.prune_short_branches_inplace(min_length=50)
mm.visualize_mesh_3d(skel=raw_skeleton, show_axes=False, height=800, width=1000)

# %%
optimizer_options = BasisOptimizerOptions(
    do_pruning=False,
    max_iterations=20,
    step_scale=2.0,
    lambda_smooth=0.1,
    preserve_terminal_nodes=True,
    preserve_branch_nodes=False,
    do_snapping=True,
    do_forcing=True,
)
skeleton = raw_skeleton

# %%
max_edge_length = 200

swc_out_dir = f"../data/swc/current/{polylines_name}"

# check if directory exists, if not create it
if not os.path.exists(swc_out_dir):
    os.makedirs(swc_out_dir)

radius_strategy_list = [
    "equivalent_area",
    "equivalent_perimeter",
    "section_median",
    "section_circle_fit",
    "nearest_surface",
]
for radius_strategy in radius_strategy_list:
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
    morph.to_swc_file(
        f"{swc_out_dir}/TS{spine_idx}_s{max_edge_length}_{radius_strategy}.swc"
    )
    # validation
    validator = Validation(mm, skeleton, morph)
    validator.full_validation()


# %%
# plot using swctools
make_html = False

for radius_strategy in radius_strategy_list:
    swc_filepath = (
        f"{swc_out_dir}/TS{spine_idx}_s{max_edge_length}_{radius_strategy}.swc"
    )
    model = SWCModel.from_swc_file(swc_filepath)
    model.print_attributes(node_info=False, edge_info=False)
    frusta = FrustaSet.from_swc_model(model)
    title = f"TS{spine_idx}_s{max_edge_length}_{radius_strategy}"
    fig = plot_model(
        swc_model=model, frusta=frusta, slider=True, title=title, hide_axes=True
    )
    fig.show()
    if make_html:
        fig.write_html(
            f"../viz/TS{spine_idx}_s{max_edge_length}_{radius_strategy}.html"
        )

# %% [markdown]
# ## Normalize radii glabally to total surface area

# %%
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

morph.scale_radii_to_match_mesh(
    mm.mesh, metric="surface_area", account_for_overlaps=False
)

# save normalized to file
swc_filepath = (
    f"{swc_out_dir}/TS{spine_idx}_s{max_edge_length}_{radius_strategy}_SAnormalized.swc"
)
morph.to_swc_file(swc_filepath)

# load and plot
swc_model = SWCModel.from_swc_file(swc_filepath)
swc_model.print_attributes(node_info=False, edge_info=False)
frusta = FrustaSet.from_swc_model(swc_model)
title = f"TS{spine_idx}_s{max_edge_length}_{radius_strategy}"
fig = plot_model(swc_model=swc_model, frusta=frusta, slider=True, title=title)
fig.show()

validator = Validation(mm, skeleton, morph)
validator.full_validation()

# %%


# %%
