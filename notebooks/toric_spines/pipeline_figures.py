# %%
from mascaf import *
from swctools import SWCModel, FrustaSet, PointSet, plot_model
import logging

logging.basicConfig(level=logging.INFO)
import os

print("✅ Libraries imported successfully!")

# %%
spine_idx = 1
mcf_qst = 0.5
mcf_mcst = 5
obj_name = f"TS{spine_idx}"
polylines_name = f"TS{spine_idx}_qst{mcf_qst}_mcst{mcf_mcst}"
mm = MeshManager(mesh_path=f"../../data/mesh/processed/{obj_name}.obj")
raw_skeleton = SkeletonGraph.from_txt(
    f"../../data/mcf_skeletons/{polylines_name}.polylines.txt"
)
raw_skeleton.prune_short_branches_inplace(min_length_percentile=20)
mm.visualize_mesh_3d(skel=raw_skeleton)
