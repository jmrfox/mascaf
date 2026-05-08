# %% [markdown]
# # Demo: Torus Skeletonization
#

# %%
from mascaf import CableFitter, FitOptions, MeshManager, SkeletonGraph

import logging

logging.basicConfig(level=logging.DEBUG)

# %% [markdown]
# # Create and Visualize Torus

# %%
# Load torus mesh

mm = MeshManager(mesh_path="../../data/demo/torus.obj")
mm.print_mesh_analysis()
skel = SkeletonGraph.from_txt(f"../../data/demo/torus.polylines.txt")
mm.visualize_mesh_3d(title="Original Torus", skel=skel)


# %% [markdown]
# ## Fit Morphology model to Mesh

# %%
morph = CableFitter(FitOptions(max_edge_length=1.0)).fit(mm.mesh, skel)
morph.to_swc_file("../../data/demo/torus.swc")
morph.print_attributes()

# %% [markdown]
# # Plot fitted SWC model

# %%
# plot using swctools
from swctools import SWCModel, plot_model

model = SWCModel.from_swc_file("../../data/demo/torus.swc")
plot_model(swc_model=model)


# %%
from mascaf.validation import Validation

validator = Validation(mm, skel, morph)
vol_result = validator.compare_volumes()
area_result = validator.compare_surface_areas()

print("Validation Results:")
print("\nVolume Comparison:")
print(f"  Mesh volume:       {vol_result['mesh_volume']:.4f}")
print(f"  Morphology volume: {vol_result['morphology_volume']:.4f}")
print(f"  Ratio:             {vol_result['ratio']:.4f}")
print(f"  Absolute diff:     {vol_result['absolute_difference']:.4f}")
print(f"  Relative error:    {vol_result['relative_error']:.2%}")
print("\nSurface Area Comparison:")
print(f"  Mesh area:         {area_result['mesh_area']:.4f}")
print(f"  Morphology area:   {area_result['morphology_area']:.4f}")
print(f"  Ratio:             {area_result['ratio']:.4f}")
print(f"  Absolute diff:     {area_result['absolute_difference']:.4f}")
print(f"  Relative error:    {area_result['relative_error']:.2%}")
