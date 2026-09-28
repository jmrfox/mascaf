# %% [markdown]
# # Demo: Torus MASCAF
#
# Fit a cable morphology to the torus demo mesh, using the same tip extension and surface-area scaling as the other demos. The torus skeleton is already a closed loop, so tip extension does not add nodes.

# %%
from mascaf import *

import logging

logging.basicConfig(level=logging.INFO)

# %% [markdown]
# ## Mesh and skeleton
#
# Load the demo mesh and its curve skeleton. The skeleton is the centerline the cable fit follows.

# %%
mm = MeshManager(mesh_path="../../data/demo/torus.obj")
mm.print_mesh_analysis()
skeleton = SkeletonGraph.from_txt("../../data/demo/torus.polylines.txt")
mesh_fig, camera = visualize_mesh_3d(
    mesh=mm,
    title="Torus",
    skel=skeleton,
    show_axes=False,
    return_camera=True,
)
mesh_fig

# %% [markdown]
# ## Cable fit
#
# `CableFitter` resamples the skeleton so edges are at most `max_edge_length` and estimates a radius at each sample from the mesh cross-section.

# %%
max_edge_length = 1.0

morph = CableFitter(FitOptions(max_edge_length=max_edge_length)).fit(mm.mesh, skeleton)

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
# ## Surface-area normalization
#
# Scale every radius by one factor so the cable surface area matches the mesh.

# %%
morph.scale_radii_to_match_mesh(
    mm.mesh, metric="surface_area", account_for_overlaps=False
)
fig = visualize_mesh_cable_3d(mm, skeleton, morph, show_axes=False, camera=camera)
fig.show()

validator = Validation(mm, skeleton, morph)
validator.full_validation()

# %%
