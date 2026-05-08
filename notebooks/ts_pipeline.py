# %%
import logging

logging.basicConfig(level=logging.DEBUG)

from mascaf import *
from swctools import SWCModel, FrustaSet, PointSet, plot_model
import numpy as np
from scipy.spatial.transform import Rotation

import os

print("✅ Libraries imported successfully!")


# %%
# per-spine parameters

def get_ts_pipeline_params(idx: int) -> dict:
    qst = 0.5  # for all
    mcst = 5  # for all

    fig_width = 800
    fig_height = 600

    rotations = {1: [0, -30, 20],
    2: [-40, 0, 0], 
    3: [60, 0, 0]}

    zooms = {1: 0.8, 2: 1.0, 3: 1.0}

    pruning_fractions = {1: 20, 2: 20, 3: 20}

    max_edge_lengths = {1: 200, 2: 50, 3: 200}

    optimizer_options = BasisOptimizerOptions(
            do_pruning=True,
            pruning_min_length_percentile=pruning_fractions[idx],
            do_snapping=True,
            do_forcing=True,
            n_rays=6,
            max_iterations=20,
            step_size=1.0,
            smoothing_weight=0.1,
            preserve_terminal_nodes=True,
            preserve_branch_nodes=False,
        )

    fit_options = FitOptions(
            max_edge_length=max_edge_lengths[idx],
            radius_strategy="equivalent_area",
            section_probe_eps=1e-4,
            section_probe_tries=3,
            multi_tangent_reduction="median",
            basis_optimizer_options=optimizer_options,
        )

    rot = Rotation.from_euler('xyz', rotations[idx], degrees=True)
    zoom = zooms[idx]
    eye_coord = rot.apply(np.array([1.0, 1.0, 1.0])) * zoom

    return {
        "object_name": f"TS{idx}",
        "qst": qst,
        "mcst": mcst,
        "eye_coord": eye_coord,
        "fit_options": fit_options,
        "fig_width": fig_width,
        "fig_height": fig_height,
        "max_edge_length": max_edge_lengths[idx],
    }


# %%
spine_idx = 2
params = get_ts_pipeline_params(spine_idx)

polylines_name = f"TS{spine_idx}_qst{params["qst"]}_mcst{params["mcst"]}"
mm = MeshManager(mesh_path=f"../data/mesh/processed/{params["object_name"]}.obj")
skeleton = SkeletonGraph.from_txt(
    f"../data/mcf_skeletons/{polylines_name}.polylines.txt"
)

# FIGURE: mesh
fig = mm.visualize_mesh_3d(skel=None, show_axes=False, title="")

eye_coord = params["eye_coord"]

fig.update_layout(
    scene=dict(
        camera=dict(
            eye={'x':eye_coord[0], 'y':eye_coord[1], 'z':eye_coord[2]},
            projection=dict(type="perspective"),
        ),
        aspectmode="data",  # or 'cube', 'auto', 'manual'
    )
)
fig.show()

# FIGURE: mesh with skeleton
fig = mm.visualize_mesh_3d(skel=skeleton, show_axes=False, title="")
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
swc_out_dir = f"../data/swc/current/{polylines_name}"
swc_filepath = f"{swc_out_dir}/TS{spine_idx}_mel{params["max_edge_length"]}.swc"

# check if directory exists, if not create it
if not os.path.exists(swc_out_dir):
    os.makedirs(swc_out_dir)

fitter = CableFitter(options=params["fit_options"])

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
title = f"TS{spine_idx}_s{params["max_edge_length"]}"
fig = plot_model(
    swc_model=model,
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
swc_filepath = (
    f"{swc_out_dir}/TS{spine_idx}_mel{params["max_edge_length"]}_norm.swc"
)
morph.to_swc_file(swc_filepath)

# load and plot
swc_model = SWCModel.from_swc_file(swc_filepath)
swc_model.print_attributes(node_info=False, edge_info=False)
title = f"TS{spine_idx}_s{params["max_edge_length"]}"
fig = plot_model(
    swc_model=swc_model,
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
# compare morphology basis to original skeleton

skel_pointset = skeleton.to_point_set()

fig = plot_model(
    swc_model=swc_model,
    opacity=0.2,
    title="",
    width=800,
    height=600,
    hide_axes=True,
    point_set=skel_pointset,
    point_color="red",
    point_size=1.0,
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
