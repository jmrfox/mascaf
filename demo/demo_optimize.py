# %% [markdown]
# # Basis Optimization Demo (TS1)
#
# Load the TS1 mesh and MCF skeleton, run `BasisOptimizer`, and visualize
# before/after.
#
# **Requirements:** `mascaf`

# %%
import logging
from pathlib import Path

from mascaf import (
    BasisOptimizer,
    BasisOptimizerOptions,
    MeshManager,
    MorphologyGraph,
    SkeletonGraph,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# %% [markdown]
# ## Paths

# %%
ROOT = Path.cwd().parent
mesh_path = ROOT / "data" / "mesh" / "processed" / "TS1.obj"
skeleton_path = ROOT / "data" / "mcf_skeletons" / "TS1_qst0.5_mcst5.polylines.txt"

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
# ## Build morphology basis
#
# Resample the skeleton to a morphology basis for optimization
# (`max_edge_length` = 2 % of the bounding-box diagonal).

# %%
max_edge_length = 0.05 * diagonal
basis = MorphologyGraph.from_skeleton_graph_resample(skeleton, max_edge_length)
print(
    f"Basis: {basis.number_of_nodes()} nodes, "
    f"{basis.number_of_edges()} edges "
    f"(max_edge_length={max_edge_length:.4f})"
)

# check for outside nodes
outside_ids = basis.get_outside_nodes(mm)
print(f"Found {len(outside_ids)} vertices outside!")

# %% [markdown]
# ## Visualize mesh with initial basis

# %%
fig = mm.visualize_mesh_3d(
    skel=basis, 
    show_axes=False, 
    title="Initial basis",
    skel_marker_size=2.0,
    skel_line_width=2.0)
fig.show()

# %% [markdown]
# ## Run BasisOptimizer

# %%
basis_options = BasisOptimizerOptions(
    do_pruning=False,
    do_snapping=True,
    do_forcing=True,
    max_iterations=100,
    alpha_s=0.2,
    preserve_terminal_nodes=True,
    step_cap_factor=0.5,
)

optimizer = BasisOptimizer(basis, mm.mesh, basis_options)
optimized = optimizer.optimize()
stats = optimizer.get_optimization_stats()

outside_ids = optimized.get_outside_nodes(mm.mesh)
print(f"Found {len(outside_ids)} vertices outside!")
if len(outside_ids)>0:
    for id in outside_ids:
        print(id, optimized.get_node_position(id))

print("Basis optimization statistics:")
for key, value in stats.items():
    print(f"  {key}: {value}")

    

# %% [markdown]
# ## Visualize mesh with optimized basis

# %%
fig = mm.visualize_mesh_3d(
    skel=[basis, optimized], 
    show_axes=False, 
    title="Optimized basis",
    skel_marker_size=2.0,
    skel_line_width=2.0,
    skel_color=['red', 'blue']
)
fig.show()

# %%
