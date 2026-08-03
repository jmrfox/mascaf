# %% [markdown]
# # FittingOptimizer demo — toric spine
#
# Search `max_edge_length` (as a fraction of the mesh bbox diagonal) to minimize
# volume error after surface-area normalization. Among near-best trials, prefer
# larger edge length (`volume_error_rel_tol`).
#
# Assumes the kernel cwd is `notebooks/` (repo-relative paths use `../data/...`).

# %%
import logging
import os

import matplotlib.pyplot as plt
import numpy as np

from mascaf import (
    FitOptions,
    FittingOptimizer,
    FittingOptimizerOptions,
    MeshManager,
    SkeletonGraph,
    Validation,
    fraction_bounds_around_suggestion,
    suggest_fit_parameters,
)
from swctools import SWCModel, plot_model

logging.basicConfig(level=logging.WARNING)

# %% [markdown]
# ## Load mesh and skeleton
#
# Set `spine_idx` to any available toric spine (`TS1`, `TS2`, …).

# %%
spine_idx = 2
mcf_qst = 0.5
mcf_mcst = 5

obj_name = f"TS{spine_idx}"
polylines_name = f"TS{spine_idx}_qst{mcf_qst}_mcst{mcf_mcst}"

mm = MeshManager(mesh_path=f"../data/mesh/processed/{obj_name}.obj")
skeleton = SkeletonGraph.from_txt(
    f"../data/mcf_skeletons/{polylines_name}.polylines.txt"
)

diagonal = mm.bounding_box_diagonal()
print(f"{obj_name}: {mm.mesh.vertices.shape[0]} verts, diagonal={diagonal:.3g}")
print(
    f"Skeleton: {skeleton.number_of_nodes()} nodes, "
    f"{skeleton.number_of_edges()} edges"
)

mm.visualize_mesh_3d(skel=skeleton, show_axes=False, height=700, width=900)

# %% [markdown]
# ## Configure and run `FittingOptimizer`
#
# Seed search bounds from the mesh-thickness FitParameterOracle (SDF), then
# optionally refine mel against volume error. Each trial: `CableFitter` → scale
# radii to match mesh SA → `|ΔV| / V_mesh`.
#
# Optional `BasisOptimizerOptions` are passed through `FitOptions` and run inside
# every trial (expensive). Leave as `None` for a faster mel search, or pass the
# oracle's basis options.

# %%
suggested = suggest_fit_parameters(mm, skeleton)
for line in suggested.rationale:
    print(f"Oracle: {line}")
bounds = fraction_bounds_around_suggestion(suggested, rel_span=0.5)
print(
    f"Oracle mel={suggested.max_edge_length:.4g} "
    f"(frac={suggested.max_edge_length_fraction:.4g}); "
    f"search bounds={bounds}"
)

# Set to suggested.basis_optimizer_options to enable per-trial basis opt (slow).
basis_optimizer_options = None

fit_options = FitOptions(
    radius_strategy="equivalent_area",
    basis_optimizer_options=basis_optimizer_options,
)

opt_options = FittingOptimizerOptions(
    fraction_bounds=bounds,
    maxiter=12,
    xatol=5e-3,
    volume_error_rel_tol=0.05,
    account_for_overlaps=False,
)

result = FittingOptimizer(fit_options=fit_options, options=opt_options).optimize(
    mm, skeleton
)

print(
    f"Selected fraction={result.max_edge_length_fraction:.4g} "
    f"(mel={result.max_edge_length:.4g})"
)
print(f"Volume |rel err| (selected)={result.volume_relative_error:.4g}")
print(f"Volume |rel err| (best)    ={result.best_volume_relative_error:.4g}")
print(f"SA scale factor={result.scale_factor:.4g}")
print(f"n_evals={result.n_evals}")
print(
    f"Morphology: {result.morphology.number_of_nodes()} nodes, "
    f"{result.morphology.number_of_edges()} edges"
)

# %% [markdown]
# ## Objective history
#
# Circles are evaluations; the star marks the ε-preferred selection (largest
# fraction among near-best volume errors).

# %%
fractions = np.array([rec.fraction for rec in result.history])
errors = np.array([rec.volume_relative_error for rec in result.history])
order = np.argsort(fractions)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(fractions[order], errors[order], "-o", color="0.4", ms=5, label="evals")
ax.scatter(
    [result.max_edge_length_fraction],
    [result.volume_relative_error],
    s=140,
    marker="*",
    color="C1",
    zorder=5,
    label="selected (prefer larger)",
)
ax.axhline(
    result.best_volume_relative_error,
    color="C0",
    ls="--",
    lw=1,
    label="best volume error",
)
ax.set_xlabel("max_edge_length / bbox diagonal")
ax.set_ylabel("|volume relative error| after SA norm")
ax.set_title(f"{obj_name} FittingOptimizer history")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
plt.show()

# %% [markdown]
# ## Validate and save SWC

# %%
validator = Validation(mm, skeleton, result.morphology)
vol = validator.compare_volumes()
area = validator.compare_surface_areas()
print(
    f"Volume ratio={vol['ratio']:.4f}, "
    f"rel_error={vol['relative_error']:.4g}"
)
print(
    f"Area ratio={area['ratio']:.4f}, "
    f"rel_error={area['relative_error']:.4g}"
)

swc_out_dir = f"../data/swc/current/{polylines_name}"
os.makedirs(swc_out_dir, exist_ok=True)
mel_tag = int(round(result.max_edge_length))
swc_filepath = (
    f"{swc_out_dir}/TS{spine_idx}_mel{mel_tag}_fitopt_SAnormalized.swc"
)
result.morphology.to_swc_file(swc_filepath)
print(f"Wrote {swc_filepath}")

# %% [markdown]
# ## Visualize selected morphology

# %%
model = SWCModel.from_swc_file(swc_filepath)
model.print_attributes(node_info=False, edge_info=False)
title = f"TS{spine_idx} mel={result.max_edge_length:.0f} (fitopt, SA-norm)"
fig = plot_model(
    swc_model=model,
    slider=True,
    title=title,
    hide_axes=True,
    width=900,
    height=700,
)
fig.show()

# %%
