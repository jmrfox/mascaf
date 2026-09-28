# %%
from mascaf import MeshManager, SkeletonGraph, visualize_mesh_3d
import logging

logging.basicConfig(level=logging.INFO)

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
fig, _camera = visualize_mesh_3d(
    mm,
    skel=raw_skeleton,
    show_axes=False,
    title="",
    orientation="horizontal",
    return_camera=True,
)
fig.show()
