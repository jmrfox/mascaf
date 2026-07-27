# %%
import logging
import os
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.transform import Rotation

from mascaf import (
    BasisOptimizer,
    BasisOptimizerOptions,
    FitOptions,
    MeshManager,
    MorphologyGraph,
    SkeletonGraph,
    Validation,
)
from mascaf.cable_fitting import _compute_morphology_node_radii
from swctools import SWCModel, plot_model

if TYPE_CHECKING:
    import plotly.graph_objects as go  # type: ignore

logging.basicConfig(level=logging.WARNING)

print("✅ Libraries imported successfully!")


# %%
# per-spine parameters
def get_ts_pipeline_params(idx: int) -> dict:
    qst = 0.5  # for all
    mcst = 5  # for all
    max_edge_length_fraction = 0.06 # fraction of the spine's bounding box diagonal

    fig_width = 800
    fig_height = 600

    rotations = {
        1: [0, -30, 20], 
        2: [-40, 0, 0], 
        3: [60, 0, 20],
        4: [-30, 40, 50],
        21: [0, 30, 95],
        24: [0, 0, 0],
        48: [0, 0, 0],
        67: [0, 0, 0],
        76: [0, 0, 0],
    }

    zooms = {
        1: 1.0, 
        2: 1.0, 
        3: 1.0, 
        4: 1.0, 
        21: 1.0, 
        24: 1.0, 
        48: 1.0, 
        67: 1.0, 
        76: 1.0,
    }

    pruning_length_fractions = {
        1: 0.1,
        2: 0.1,
        3: 0.1,
        4: 0.1,
        21: 0.1,
        24: 0.1,
        48: 0.1,
        67: 0.1,
        76: 0.1,
    }

    # max_edge_lengths = {
    #     1: 200, 
    #     2: 50, 
    #     3: 150, 
    #     4: 200, 
    #     21: 200, 
    #     24: 200, 
    #     48: 200, 
    #     67: 200, 
    #     76: 200,
    # }

    optimizer_options = BasisOptimizerOptions(
        do_pruning=True,
        pruning_length_fraction=pruning_length_fractions[idx],
        do_snapping=True,
        do_forcing=True,
        active_resample=True,
        n_rays=6,
        max_iterations=10,
        alpha_s=0.1,
        step_scale=0.1,
        step_cap_factor=0.5,
        preserve_terminal_nodes=True,
        preserve_branch_nodes=False,
        ray_jitter=0.1,
        localization_beta=1.0,
        active_resample_min_fraction=0.05,
        active_resample_max_fraction=0.2,
    )

    rot = Rotation.from_euler("xyz", rotations[idx], degrees=True)
    zoom = zooms[idx]
    eye_coord = rot.apply(np.array([1.0, 1.0, 1.0])) * zoom

    return {
        "object_name": f"TS{idx}",
        "qst": qst,
        "mcst": mcst,
        "eye_coord": eye_coord,
        # "fit_options": fit_options,
        "fig_width": fig_width,
        "fig_height": fig_height,
        "max_edge_length_fraction": max_edge_length_fraction,
        "optimizer_options": optimizer_options,
    }

pdf_scale = 2

# %%
spine_idx = 3
params = get_ts_pipeline_params(spine_idx)

mesh_path = f"../data/mesh/processed/{params['object_name']}.obj"
mm = MeshManager(mesh_path=mesh_path)
model_length = mm.bounding_box_diagonal()
params["max_edge_length"] = int(model_length * params["max_edge_length_fraction"])

polylines_name = f"TS{spine_idx}_qst{params['qst']}_mcst{params['mcst']}"
skeleton = SkeletonGraph.from_txt(
    f"../data/mcf_skeletons/{polylines_name}.polylines.txt"
)

fig_out_dir = f"../viz/ts{spine_idx}"
os.makedirs(fig_out_dir, exist_ok=True)

# FIGURE: mesh
mesh_fig: "go.Figure" = mm.visualize_mesh_3d(skel=None, show_axes=False, title="")

eye_coord = params["eye_coord"]

mesh_fig.update_layout(
    scene=dict(
        camera=dict(
            eye={"x": eye_coord[0], "y": eye_coord[1], "z": eye_coord[2]},
            projection=dict(type="perspective"),
        ),
        aspectmode="data",  # 'cube', 'auto', 'manual'
    )
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

# FIGURE: mesh with skeleton
mesh_skel_fig: "go.Figure" = mm.visualize_mesh_3d(
    skel=skeleton, show_axes=False, title=""
)
mesh_skel_fig.update_layout(
    scene=dict(
        camera=dict(
            eye={"x": eye_coord[0], "y": eye_coord[1], "z": eye_coord[2]},
            projection=dict(type="perspective"),
        ),
        aspectmode="data",  # 'cube', 'auto', 'manual'
    )
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

# %%
# Basis construction + optimization (before radius fitting)
basis = MorphologyGraph.from_skeleton_graph_resample(
    skeleton,
    float(params["max_edge_length"]),
)
print(
    f"Initial basis: {basis.number_of_nodes()} nodes, "
    f"{basis.number_of_edges()} edges"
)

optimizer = BasisOptimizer(basis, mm.mesh, params["optimizer_options"])
optimized_basis = optimizer.optimize()
stats = optimizer.get_optimization_stats()
print("Basis optimization statistics:")
for key, value in stats.items():
    print(f"  {key}: {value}")

# FIGURE: original (red) vs optimized (blue) basis
basis_opt_fig: "go.Figure" = mm.visualize_mesh_3d(
    skel=[basis, optimized_basis],
    show_axes=False,
    title="",
    skel_color=["red", "blue"],
    skel_line_width=3.0,
    skel_marker_size=2.0,
)
basis_opt_fig.update_layout(
    scene=dict(
        camera=dict(
            eye={"x": eye_coord[0], "y": eye_coord[1], "z": eye_coord[2]},
            projection=dict(type="perspective"),
        ),
        aspectmode="data",
    )
)
basis_opt_fig.write_image(
    f"{fig_out_dir}/TS{spine_idx}_mel{params['max_edge_length']}_basis_opt.pdf",
    format="pdf",
    engine="kaleido",
    width=600,
    height=450,
    scale=pdf_scale,
)
basis_opt_fig.show()

# %%
# Cable fitting: estimate radii on the optimized basis
swc_out_dir = f"../data/swc/current/{polylines_name}"
swc_filepath = f"{swc_out_dir}/TS{spine_idx}_mel{params['max_edge_length']}.swc"

if not os.path.exists(swc_out_dir):
    os.makedirs(swc_out_dir)

fit_options = FitOptions(
    max_edge_length=params["max_edge_length"],
    radius_strategy="equivalent_area",
    section_probe_eps=1e-4,
    section_probe_tries=3,
    multi_tangent_reduction="mean",
    basis_optimizer_options=None,
)
morph = optimized_basis.copy()
_compute_morphology_node_radii(morph, mm.mesh, fit_options)

# write swc to file
morph.to_swc_file(swc_filepath)
# validation
validator = Validation(mm, skeleton, morph)
validator.full_validation()

model = SWCModel.from_swc_file(swc_filepath)
model.print_attributes(node_info=False, edge_info=False)
title = f"TS{spine_idx}_mel{params['max_edge_length']}"
morph_fig = plot_model(
    swc_model=model,
    slider=False,
    title="",
    width=800,
    height=600,
    show_axes=False,
)
morph_fig.update_layout(
    scene=dict(
        camera=dict(
            eye={"x": eye_coord[0], "y": eye_coord[1], "z": eye_coord[2]},
            projection=dict(type="perspective"),
        ),
        aspectmode="data",  # 'cube', 'auto', 'manual'
    )
)
morph_filename = (
    f"{fig_out_dir}/TS{spine_idx}_mel{params['max_edge_length']}_" f"morph.pdf"
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

# %%
morph.scale_radii_to_match_mesh(
    mm.mesh, metric="surface_area", account_for_overlaps=False
)

# save normalized to file
swc_filepath = f"{swc_out_dir}/TS{spine_idx}_mel{params["max_edge_length"]}_norm.swc"
morph.to_swc_file(swc_filepath)

# load and plot
swc_model = SWCModel.from_swc_file(swc_filepath)
swc_model.print_attributes(node_info=False, edge_info=False)
title = f"TS{spine_idx}_s{params["max_edge_length"]}"
norm_fig: "go.Figure" = plot_model(
    swc_model=swc_model,
    slider=False,
    title="",
    width=800,
    height=600,
    show_axes=False,
)
norm_fig.update_layout(
    scene=dict(
        camera=dict(
            eye={"x": eye_coord[0], "y": eye_coord[1], "z": eye_coord[2]},
            projection=dict(type="perspective"),
        ),
        aspectmode="data",  # 'cube', 'auto', 'manual'
    )
)
norm_filename = (
    f"{fig_out_dir}/TS{spine_idx}_mel{params['max_edge_length']}_" f"morph_norm.pdf"
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

validator = Validation(mm, skeleton, morph)
validator.full_validation()

print(swc_filepath)

# %%
# compare morphology basis to original skeleton

skel_pointset = skeleton.to_point_set()

vs_fig: "go.Figure" = plot_model(
    swc_model=swc_model,
    opacity=0.2,
    title="",
    width=800,
    height=600,
    show_axes=False,
    point_set=skel_pointset,
    point_color="red",
    point_size=model_length * 0.0015,
)
vs_fig.update_layout(
    scene=dict(
        camera=dict(
            eye={"x": eye_coord[0], "y": eye_coord[1], "z": eye_coord[2]},
            projection=dict(type="perspective"),
        ),
        aspectmode="data",  # 'cube', 'auto', 'manual'
    )
)
vs_filename = (
    f"{fig_out_dir}/TS{spine_idx}_mel{params['max_edge_length']}_" f"morph_vs_skel.pdf"
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
