# %%
import logging

logging.basicConfig(level=logging.WARNING)

from mascaf import *
from swctools import SWCModel, FrustaSet, PointSet, plot_model
import numpy as np
from scipy.spatial.transform import Rotation

import os

print("✅ Libraries imported successfully!")

# %%
name = "human"
angles = [30, -30, 110]
zoom = 1.4
max_edge_length = 10

optimizer_options = BasisOptimizerOptions(
    do_pruning=False,
    do_snapping=True,
    do_forcing=False,
    n_rays=6,
    max_iterations=10,
    alpha_s=0.1,
    preserve_terminal_nodes=True,
    preserve_branch_nodes=False,
)

fit_options = FitOptions(
    max_edge_length=max_edge_length,
    radius_strategy="equivalent_area",
    section_probe_eps=1e-4,
    section_probe_tries=3,
    multi_tangent_reduction="median",
    basis_optimizer_options=optimizer_options,
)

mm = MeshManager(mesh_path=f"../data/demo/{name}.obj")
raw_skeleton = SkeletonGraph.from_txt(f"../data/demo/{name}.polylines.txt")
raw_skeleton.prune_short_branches_inplace(min_length_fraction=10)

# mesh only
fig = mm.visualize_mesh_3d(skel=None, show_axes=False, title="")


rot = Rotation.from_euler("xyz", angles, degrees=True)
eye_coord = rot.apply(np.array([1.0, 1.0, 1.0])) * zoom

fig.update_layout(
    scene=dict(
        camera=dict(
            eye={"x": eye_coord[0], "y": eye_coord[1], "z": eye_coord[2]},
            projection=dict(type="perspective"),
        ),
        aspectmode="data",  # or 'cube', 'auto', 'manual'
    )
)
fig.show()

# mesh with skeleton
fig = mm.visualize_mesh_3d(skel=raw_skeleton, show_axes=False, title="")
fig.update_layout(
    scene=dict(
        camera=dict(
            eye={"x": eye_coord[0], "y": eye_coord[1], "z": eye_coord[2]},
            projection=dict(type="perspective"),
        ),
        aspectmode="data",  # or 'cube', 'auto', 'manual'
    )
)
fig.show()

skeleton = raw_skeleton

# %%
swc_filepath = f"../data/demo/{name}.swc"

fitter = CableFitter(options=fit_options)

morph = fitter.fit(
    mm.mesh,
    skeleton,
)
# write swc to file
morph.to_swc_file(swc_filepath)
# validation
# validator = Validation(mm, skeleton, morph)
# validator.full_validation()

model = SWCModel.from_swc_file(swc_filepath)
model.print_attributes(node_info=False, edge_info=False)
frusta = FrustaSet.from_swc_model(model)
title = f"Human neuron"
fig = plot_model(
    swc_model=model,
    frusta=frusta,
    slider=False,
    title="",
    width=800,
    height=600,
    hide_axes=True,
)
fig.update_layout(
    scene=dict(
        camera=dict(
            eye={"x": eye_coord[0], "y": eye_coord[1], "z": eye_coord[2]},
            projection=dict(type="perspective"),
        ),
        aspectmode="data",  # or 'cube', 'auto', 'manual'
    )
)
fig.show()

# %%
morph.scale_radii_to_match_mesh(
    mm.mesh, metric="surface_area", account_for_overlaps=False
)

# save normalized to file
swc_filepath = f"../data/demo/{name}_norm.swc"
morph.to_swc_file(swc_filepath)

# load and plot
swc_model = SWCModel.from_swc_file(swc_filepath)
swc_model.print_attributes(node_info=False, edge_info=False)
frusta = FrustaSet.from_swc_model(swc_model)
title = f"Human cell SWC"
fig = plot_model(
    swc_model=swc_model,
    frusta=frusta,
    slider=False,
    title="",
    width=800,
    height=600,
    hide_axes=True,
)
fig.update_layout(
    scene=dict(
        camera=dict(
            eye={"x": eye_coord[0], "y": eye_coord[1], "z": eye_coord[2]},
            projection=dict(type="perspective"),
        ),
        aspectmode="data",  # or 'cube', 'auto', 'manual'
    )
)
fig.show()

# validator = Validation(mm, skeleton, morph)
# validator.full_validation()

# %%
