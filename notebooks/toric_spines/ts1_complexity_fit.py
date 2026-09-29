# %% [markdown]
# # TS1 fit from the complexity summary
#
# Reproduces the calculation behind the first row of
# `outputs/complexity_summary.csv`: TS1, no terminal extension, at
# `max_edge_length = 1 ×` the mesh thickness median.
#
# The summary reports the best of three stochastic fits (volume ratio nearest
# 1, then shorter runtime). This notebook does the same three fits and plots
# the selected cable. A new run will not match the saved numbers exactly,
# because basis optimization uses unseeded ray jitter.
#
# Paths are resolved from the repo root. The kernel cwd can be the repo root
# or this notebook's directory.

# %%
import logging
from pathlib import Path

import numpy as np

from mascaf import (
    CableFitter,
    FitOptions,
    FitOracleOptions,
    MeshManager,
    SkeletonGraph,
    Validation,
    compute_fit_features,
    suggest_fit_parameters,
    visualize_mesh_cable_3d,
)

logging.basicConfig(level=logging.INFO)

# %%
def repo_root() -> Path:
    here = Path.cwd().resolve()
    for candidate in (here, *here.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "mascaf").is_dir():
            return candidate
    raise FileNotFoundError("Could not find the mascaf repo from the current directory.")


ROOT = repo_root()
MESH_PATH = ROOT / "data" / "mesh" / "processed" / "TS1.obj"
SKELETON_PATH = ROOT / "data" / "mcf_skeletons" / "TS1_qst0.5_mcst5.polylines.txt"
MEL_OVER_THICKNESS = 2.0
N_RUNS = 1

# %% [markdown]
# ## Load mesh and skeleton

# %%
mesh = MeshManager(mesh_path=str(MESH_PATH))
skeleton = SkeletonGraph.from_txt(str(SKELETON_PATH))
features = compute_fit_features(mesh, skeleton)
thickness = float(features.thickness.median)
mel = MEL_OVER_THICKNESS * thickness
print(
    f"TS1: {mesh.mesh.vertices.shape[0]} vertices, "
    f"thickness median={thickness:.6g}, mel={mel:.6g}"
)
print(
    f"Skeleton: {features.skeleton_nodes} nodes, "
    f"{features.n_branches} branches, cyclomatic={features.cyclomatic_number}"
)

# %% [markdown]
# ## Fit three times and keep the best volume ratio
#
# Same options as `scripts/complexity_analysis.py`: the fit oracle, with
# `max_edge_length` overridden to one thickness median. Active resampling is
# off. Terminals are not extended (`extend_terminals = 0` in the summary).

# %%
opts = FitOracleOptions()
min_len = opts.active_resample_min_over_mel * mel
max_len = opts.active_resample_max_over_mel * mel
diagonal = float(features.bbox_diagonal)
min_frac = min_len / diagonal if diagonal > 0 else 0.0
max_frac = max_len / diagonal if diagonal > 0 else 0.1
min_frac = float(np.clip(min_frac, 1e-6, 1.0))
max_frac = float(np.clip(max_frac, min_frac * 2.0, 1.0))

suggested = suggest_fit_parameters(
    mesh,
    skeleton,
    features=features,
    overrides={
        "max_edge_length": mel,
        "active_resample_min_fraction": min_frac,
        "active_resample_max_fraction": max_frac,
    },
)
for line in suggested.rationale:
    print(f"Oracle: {line}")
print(f"active_resample={suggested.basis_optimizer_options.active_resample}")

options = FitOptions(
    max_edge_length=suggested.max_edge_length,
    basis_optimizer_options=suggested.basis_optimizer_options,
)

do_extend = True

runs = []
for run_index in range(N_RUNS):
    morphology = CableFitter(options).fit(mesh, skeleton)
    if do_extend:
        morphology.extend_terminals()
    validator = Validation(mesh, skeleton, morphology)
    volume = validator.compare_volumes(account_for_overlaps=False)
    area = validator.compare_surface_areas(account_for_overlaps=False)
    volume_overlaps = validator.compare_volumes(account_for_overlaps=True)
    area_overlaps = validator.compare_surface_areas(account_for_overlaps=True)
    record = {
        "run_index": run_index,
        "morphology": morphology,
        "nodes": morphology.number_of_nodes(),
        "volume_ratio": volume["ratio"],
        "volume_relative_error": volume["relative_error"],
        "area_relative_error": area["relative_error"],
        "volume_relative_error_overlaps": volume_overlaps["relative_error"],
        "area_relative_error_overlaps": area_overlaps["relative_error"],
    }
    runs.append(record)
    print(
        f"run {run_index}: {record['nodes']} nodes, "
        f"volume error={record['volume_relative_error']:.4g}, "
        f"area error={record['area_relative_error']:.4g}, "
        f"volume error (overlaps)={record['volume_relative_error_overlaps']:.4g}, "
        f"area error (overlaps)={record['area_relative_error_overlaps']:.4g}"
    )

selected = min(runs, key=lambda rec: (abs(rec["volume_ratio"] - 1.0), rec["run_index"]))
print(
    f"Selected run {selected['run_index']} "
    f"(volume ratio {selected['volume_ratio']:.4g})"
)

# %% [markdown]
# ## Mesh, skeleton, and selected cable

# %%
fig = visualize_mesh_cable_3d(
    mesh,
    skeleton,
    selected["morphology"],
    show_axes=False,
    top_title="TS1 mesh and skeleton",
    bottom_title=f"Cable, run {selected['run_index']}, mel/t={MEL_OVER_THICKNESS:g}",
)
fig

# %%
validator.full_validation()

# %%
