# %% [markdown]
# # Demo: Torus Mesh
#
# Create and visualize torus (donut-shaped) mesh geometries using mascaf.
#
# Resulting mesh saved to `../data/demo/torus.obj` for skeletonization in CGAL.

# %% [markdown]
# ## Setup

# %%
import numpy as np
import trimesh
import plotly.graph_objects as go

# Import mascaf functions
from mascaf import MeshManager, example_mesh

import logging
logging.basicConfig(level=logging.INFO)

print("✅ Libraries imported successfully!")

# %% [markdown]
# ## Create Torus
#
# With a mesh loaded in a `MeshManager`, we can apply transformations to it. Here, we create a torus mesh and apply a rotation to it.

# %%
# Create a torus mesh
torus = example_mesh("torus")
print(f"Created torus with {len(torus.vertices)} vertices and {len(torus.faces)} faces")
mm = MeshManager(torus)

# %% [markdown]
# ## 3D Visualization

# %%
mm.visualize_mesh_3d()

# %% [markdown]
# ## Interactive Cross-Sections

# %%
mm.visualize_mesh_slice_interactive()

# %% [markdown]
# ## Mesh Properties

# %%
mm.print_mesh_analysis()

# %% [markdown]
# ## Save mesh to .obj file

# %%
savefile = "../data/demo/torus.obj"
mm.save(savefile)
