# %% [markdown]
# # Demo: Cylinder MASCAF
#
# Fit a cable morphology to the cylinder demo mesh. The skeleton stops short of the ends, so the fit is extended at each tip and then scaled to the mesh surface area.

# %%
from mascaf import *
import logging

logging.basicConfig(level=logging.INFO)

# %% [markdown]
# ## Mesh and skeleton
#
# Load the demo mesh and its curve skeleton. The skeleton is the centerline the cable fit follows.

# %%
mm = MeshManager(mesh_path="../../data/demo/cylinder.obj")
mm.print_mesh_analysis()
skeleton = SkeletonGraph.from_txt("../../data/demo/cylinder.polylines.txt")
mesh_fig, camera = visualize_mesh_3d(
    mesh=mm,
    title="Cylinder",
    skel=skeleton,
    show_axes=False,
    width=800,
    height=600,
    eye_scale=2.2,
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
fig = visualize_mesh_cable_3d(
    mm, skeleton, morph, show_axes=False, eye_scale=2.2, camera=camera
)
fig.show()

validator = Validation(mm, skeleton, morph)
validator.full_validation()

# %% [markdown]
# ## Surface-area normalization
#
# Scale every radius by one factor so the cable surface area matches the mesh.

# %%
morph_scaled = morph.copy()
morph_scaled.scale_radii_to_match_mesh(
    mm.mesh, metric="surface_area", account_for_overlaps=False
)
fig = visualize_mesh_cable_3d(
    mm, skeleton, morph_scaled, show_axes=False, eye_scale=2, camera=camera
)
fig.show()

validator = Validation(mm, skeleton, morph_scaled)
validator.full_validation()
