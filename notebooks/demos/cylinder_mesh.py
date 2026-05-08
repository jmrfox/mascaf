# %% [markdown]
# # Demo: Cylinder Mesh
#
# Create and visualize cylindrical mesh geometries using mascaf.
#
# Resulting mesh saved to `../data/demo/cylinder.obj` for skeletonization in CGAL.
#

# %% [markdown]
# ## Setup

# %%
# Import mascaf functions
from mascaf import example_mesh, MeshManager

import logging
logging.basicConfig(level=logging.INFO)

print("✅ Libraries imported successfully!")

# %% [markdown]
# ## Create Cylinder

# %%
# Create a cylinder mesh
cylinder = example_mesh(kind='cylinder')

print(f"Created cylinder with {len(cylinder.vertices)} vertices and {len(cylinder.faces)} faces")

mm = MeshManager(cylinder)

# %% [markdown]
# ## 3D Visualization

# %%
# Visualize the cylinder
mm.visualize_mesh_3d(
    title="Simple Cylinder Mesh",
    color="lightblue"
)

# %% [markdown]
# ## Interactive Cross-Sections

# %%
mm.visualize_mesh_slice_interactive(
    title="Interactive Cylinder Cross-Sections",
    slice_color="red",
    mesh_color="lightblue",
    mesh_opacity=0.3
)


# %% [markdown]
# ## Mesh Properties

# %%
mm.print_mesh_analysis()

# %% [markdown]
# ## Save mesh to obj file

# %%
savefile = "../data/demo/cylinder.obj"
mm.save(savefile)
