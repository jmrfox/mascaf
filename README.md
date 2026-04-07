# MASCAF

**MASCAF** (Mesh and Skeleton Cable Fitting) is a Python package for fitting **cable-graph morphology** (like an [SWC](http://www.neuronland.org/NLMorphologyConverter/MorphologyFormats/SWC/Spec.html) model) to a **closed triangle mesh** and a **3D curve skeleton**.
This workflow was developed to create neuronal morphology models for multi-compartmental simulation from 3D surface meshes.

While other software tools exist with similar purposes (e.g., fitting cable models to neuronal imaging data), MASCAF was developed to handle complex neuronal morphologies, specifically non-tree structures.
By combining MASCAF with a topologically robust skeletonization procedure like that of [CGAL’s Triangulated Surface Mesh Skeletonization](https://doc.cgal.org/latest/Surface_mesh_skeletonization/index.html) (see instructions below), one forms a complete pipeline from 3D surface meshes to cable-graph models ready for multicompartmental simulators like [Arbor](https://arbor-sim.org/) and [NEURON](https://www.neuronsimulator.org).

## What you need

1. **Surface mesh** — Typically a watertight OBJ (or anything [trimesh](https://trimesh.org/) can load). `MeshManager` wraps loading and basic mesh utilities.
2. **Skeleton** — The mesh's midline skeleton represented in 3D polylines format: one text line per branch, with coordinates  
   `N x1 y1 z1 x2 y2 z2 … xN yN zN`  
   (`N` is the number of vertices on that line). `SkeletonGraph.from_txt()` also supports **GraphML** (`.graphml` / `.xml`) as the native round-trip format after optimization.

---

## Installation

Install in your own project environment by providing `pip` with the repo link. For instance, I like using [uv](https://docs.astral.sh/uv/).

```bash
uv pip install https://github.com/jmrfox/mascaf.git
```

Or, if you want to work in a local clone of this repo, just do `uv sync`.

Run code or tests inside that environment.

```bash
uv run python your_script.py
uv run pytest
```

---

## End-to-end workflow (conceptual)

0. **Skeletonize**: Run [CGAL-TSMS](https://doc.cgal.org/latest/Surface_mesh_skeletonization/index.html) on your 3D surface mesh to obtain skeleton, and save in polylines text format.
1. **Initialize**: Load the mesh (`.obj`) and skeleton (`.polylines.txt`) using `MeshManager` and `SkeletonGraph`.
2. **Optimize Skeleton** (optional): In the case of a low-quality skeleton, run `SkeletonOptimizer` to nudge skeleton points toward the midline.
3. **Fit**: Call `fit_morphology` to downsample the skeleton to cable-graph nodes and fit radii at each sample.
4. **Validate and Optimize** (optional): Compare mesh vs. cable metrics with `Validation`, or rescale all radii so cable surface area or volume matches the mesh via `MorphologyGraph.scale_radii_to_match_mesh()`.
5. **Export**: Write out result with `MorphologyGraph.to_swc_file()`.
6 **Simulate**: Load the `SWC` file as morphology in `Arbor` or `NEURON`. In case of cyclic topology, `SWC` file header will list cycle-closure directives.

---

## Main pieces of the API

| Piece | Role |
|--------|------|
| `MeshManager` | Load and hold a `trimesh.Trimesh` (e.g. from OBJ). |
| `SkeletonGraph` | Graph of skeleton vertices and edges; load/save polylines or GraphML. |
| `SkeletonOptimizer` | Optional geometric refinement of the skeleton against the mesh. |
| `FitOptions` | Resampling step (`max_edge_length`), radius strategy, section probing, optional snap-to-surface. |
| `fit_morphology` | Core routine: mesh + skeleton → `MorphologyGraph`. |
| `MorphologyGraph` | NetworkX-based cable graph (`xyz`, `radius` per node); SWC export; surface area / volume helpers; radius scaling to match mesh. |
| `Validation` | Compare mesh vs. morphology volume and surface area (and related checks). |

---

## How `fit_morphology` works

1. **Resampling** — Each skeleton edge is subdivided so that segment lengths do not exceed `FitOptions.max_edge_length` (endpoints and topology are preserved; the input graph may contain cycles).

2. **Local tangent** — At each sample point, a tangent direction along the skeleton defines a **cutting plane** through that point.

3. **Cross-section** — The mesh is intersected with that plane. The code selects a sensible polygon (for example the region containing the sample, or the boundary closest to it). If the exact plane misses geometry, it can try small offsets along ±tangent (`section_probe_eps`, `section_probe_tries`), then fall back to **nearest distance to the surface**.

4. **Radius** — `FitOptions.radius_strategy` chooses how to turn the section (or fallback) into a scalar radius, for example:
   - **equivalent_area** — \(r = \sqrt{A/\pi}\) from section area  
   - **equivalent_perimeter** — \(r = L/(2\pi)\) from exterior boundary length  
   - **section_median** — median ray distance in the section plane (robust for messy sections)  
   - **section_circle_fit** — algebraic circle fit to the boundary  
   - **nearest_surface** — distance to mesh (no section required)

   At nodes where several edges meet, radii from different directions can be combined (`multi_tangent_reduction`: mean, min, max, median).

5. **Output** — A `MorphologyGraph` with the same connectivity pattern as the resampled skeleton (including cycles). SWC export breaks cycles by duplicating nodes and can add header comments describing how to reconnect them.

---

## SWC and cycles

[SWC](http://www.neuronland.org/NLMorphologyConverter/MorphologyFormats/SWC/Spec.html) is inherently a **tree** (each node has one parent). MASCAF keeps **arbitrary graph topology** in memory and only enforces a tree when writing SWC, by duplicating an endpoint on closing edges and recording the mapping in comments when requested.

---

## Optional: skeleton optimization (`SkeletonOptimizer`)

MCF (and other methods) can leave the skeleton slightly off the medial axis or even outside the volume in awkward regions. `SkeletonOptimizer` iteratively:

- Pulls **outside** points back toward the surface,
- For **inside** points, uses multi-direction ray queries to estimate how centered the point is and moves it toward a more medial location,
- Applies **Laplacian-style smoothing** weighted by `smoothing_weight`,
- Stops when movement falls below a threshold.

Native save/load for optimized skeletons uses **GraphML** via `SkeletonGraph.to_txt()` / `from_txt()` (paths are normalized to `.graphml` when needed).

---

## Mean curvature flow skeletonization with CGAL

A convenient way to get a skeleton file compatible with MASCAF is [CGAL’s Triangulated Surface Mesh Skeletonization](https://doc.cgal.org/latest/Surface_mesh_skeletonization/index.html). 
**Mean curvature flow** evolves a surface so that each point moves with velocity given by mean curvature; in practice it is a standard way to collapse a shape toward a skeleton-like structure while preserving the topology.

The CGAL distribution includes a demo application (**CGAL Lab**) that can run TSMS, as well as many other useful operations. Typical steps:

0. Download [CGALLab](https://www.cgal.org/demo/6.1/CGALlab.zip).
1. Open CGALLab and load your mesh.
2. Run **Operations → Triangulated Surface Mesh Skeletonization → Mean Curvature Skeleton (Advanced)** (wording may vary slightly by version).
3. Export the skeleton / fixed-points result to a **polylines** text file (one branch per line, `N x1 y1 z1 …` as above).

That file is only **one** possible skeleton input; any tool that abides the same polylines text convention will work.

---

## Example: full pipeline in Python

```python
from mascaf import (
    MeshManager,
    SkeletonGraph,
    SkeletonOptimizer,
    SkeletonOptimizerOptions,
    fit_morphology,
    FitOptions,
)

# 1) Load mesh and skeleton (same coordinate system)
mesh_mgr = MeshManager(mesh_path="neuron.obj")
skeleton = SkeletonGraph.from_txt("neuron_mcf.polylines.txt")

# 2) Optional: refine skeleton inside the volume
opt_opts = SkeletonOptimizerOptions(
    max_iterations=50,
    step_size=0.1,
    preserve_terminal_nodes=True,
    smoothing_weight=0.5,
)
optimizer = SkeletonOptimizer(skeleton, mesh_mgr.mesh, opt_opts)
optimized = optimizer.optimize()
optimized.to_txt("neuron_skeleton.graphml")

# 3) Fit radii and build cable graph
fit_opts = FitOptions(
    max_edge_length=1.0,
    radius_strategy="equivalent_area",
)
morphology = fit_morphology(mesh_mgr, optimized, fit_opts)

# 4) Export SWC (cycles broken with annotations by default)
morphology.to_swc_file("neuron.swc")

# 5) Optional: match total cable surface area to mesh area
morphology.scale_radii_to_match_mesh(mesh_mgr, metric="surface_area")
morphology.to_swc_file("neuron_scaled.swc")
```

Minimal example without optimization:

```python
import trimesh
from mascaf import SkeletonGraph, fit_morphology, FitOptions

mesh = trimesh.load("shape.obj", force="mesh")
skel = SkeletonGraph.from_txt("shape.polylines.txt")
morph = fit_morphology(
    mesh,
    skel,
    FitOptions(max_edge_length=0.5, radius_strategy="section_median"),
)
morph.to_swc_file("shape.swc")
```

---

## Further reading in the repo

- `mascaf/graph_fitting.py` — `FitOptions` field documentation and tracing details.  
- `mascaf/morphology_graph.py` — SWC export, geometry summaries, `scale_radii_to_match_mesh`.  
- `mascaf/validation.py` — mesh vs. morphology comparisons.  
- `tests/` — runnable examples (e.g. `test_trace_examples.py` with `data/demo` meshes).

---

## References

- [SWC format overview](http://www.neuronland.org/NLMorphologyConverter/MorphologyFormats/SWC/Spec.html)  
- [CGAL: Triangulated Surface Mesh Skeletonization](https://doc.cgal.org/latest/Surface_mesh_skeletonization/index.html)  
