# %% [markdown]
# # Demo: Branching MASCAF
#
# Fit a cable morphology to the branching demo mesh. The skeleton stops short of the branch ends, so the fit is extended at each tip and then scaled to the mesh surface area.

# %%
from mascaf import *

import logging

logging.basicConfig(level=logging.INFO)

# %% [markdown]
# ## Mesh and skeleton
#
# Load the demo mesh and its curve skeleton. The skeleton is the centerline the cable fit follows.

# %%
mm = MeshManager(mesh_path="../../data/demo/branching.obj")
mm.print_mesh_analysis()
raw_skeleton = SkeletonGraph.from_txt(f"../../data/demo/branching.polylines.txt")
# raw_skeleton.prune_short_branches_inplace(min_length_fraction=1)
mesh_fig, camera = visualize_mesh_3d(
    mesh=mm,
    title="Branching test model",
    skel=raw_skeleton,
    show_axes=False,
    eye_scale=0.9,
    return_camera=True,
)
mesh_fig

# %% [markdown]
# ## Cable fit
#
# Basis optimization snaps nodes onto the mesh and pulls them toward the centerline. `CableFitter` then resamples that skeleton so edges are at most `max_edge_length` and estimates a radius at each sample from the mesh cross-section.

# %%
max_edge_length = 1.0
optimizer_options = BasisOptimizerOptions(
    # do_pruning=True,
    # pruning_length_fraction=0.2,
    # do_snapping=True,
    # do_forcing=True,
    # max_iterations=20,
    # alpha_s=0.1,
    # step_scale=0.5,
    # n_rays=12,
    # ray_jitter=0.1,
    # localization_beta=2.0,
    # preserve_terminal_nodes=True,
    # preserve_branch_nodes=False,
)

skeleton = raw_skeleton

# %%
swc_filepath = "../../data/demo/branching.swc"

morph = CableFitter(
    FitOptions(
        max_edge_length=max_edge_length,
        basis_optimizer_options=optimizer_options,
    )
).fit(
    mm.mesh,
    skeleton,
)


# %% [markdown]
# ## Fitted cable
#
# The mesh and skeleton are on the left; the cable model is on the right. Both panels use the same view. Validation compares cable volume and surface area with the mesh.

# %%
fig = visualize_mesh_cable_3d(mm, skeleton, morph, show_axes=False, camera=camera)
fig.show()

validator = Validation(mm, skeleton, morph)
validator.full_validation()

# %% [markdown]
# ## Terminal extension
#
# Each branch tip is continued along its outward tangent by half a parent radius. The new tip radius is also half the parent radius.

# %%
morph.extend_terminals()
fig = visualize_mesh_cable_3d(mm, skeleton, morph, show_axes=False, camera=camera)
fig.show()

validator = Validation(mm, skeleton, morph)
validator.full_validation()

# %% [markdown]
# ## Surface-area normalization
#
# Scale every radius by one factor so the cable surface area matches the mesh. Overlap between branches is left uncorrected.

# %%
morph.scale_radii_to_match_mesh(
    mm.mesh, metric="surface_area", account_for_overlaps=False
)
fig = visualize_mesh_cable_3d(mm, skeleton, morph, show_axes=False, camera=camera, show_nodes=True, node_size=8, centroid_line_width=5)
fig.show()


validator = Validation(mm, skeleton, morph)
validator.full_validation()

# %%
