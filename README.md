# MaSCaF

**MaSCaF** (Mesh and Skeleton Cable Fitting) is a Python package (`import mascaf`) for turning a **closed triangle mesh** and a **3D curve skeleton** into a **cable-graph morphology** with estimated radii, primarily exported as [SWC](http://www.neuronland.org/NLMorphologyConverter/MorphologyFormats/SWC/Spec.html).

The library intersects the mesh with planes normal to the skeleton, interprets cross-sections, and builds a graph whose edges act like truncated cones (frusta) between sampled points. It is built on [trimesh](https://trimesh.org/), [NetworkX](https://networkx.org/), and [swctools](https://github.com/jmrfox/swctools) for SWC data structures.

---

## Name and scope (formerly `mcf2swc`)

The project was originally called **`mcf2swc`** because the intended workflow was: run **mean curvature flow (MCF)** skeletonization (MCFS) on a mesh, then convert that result into SWC. While we still **highly** recommend using CGAL's MCFS implementation, that name turned out to be too narrow.

Nothing in MaSCaF assumes the skeleton came from MCF. The skeleton only needs to be expressible as the polylines (or saved GraphML) format that `SkeletonGraph` understands, in the **same coordinate frame** as the mesh. You can use any method that produces compatible centerlines (manual tracing, other skeletonizers, merged branches, graphs with cycles, and so on).

**Output today is SWC-oriented** (via `MorphologyGraph` and `swctools.SWCModel`). The internal representation is a general graph with 3D points and radii, so **other export formats are a reasonable future extension** without changing the core fitting idea.

---

## What you provide

1. **Surface mesh** — Typically a watertight OBJ (or anything [trimesh](https://trimesh.org/) can load). `MeshManager` wraps loading and basic mesh utilities.
2. **Skeleton** — A graph of 3D polylines: one text line per branch, with coordinates  
   `N x1 y1 z1 x2 y2 z2 … xN yN zN`  
   (`N` is the point count on that line). `SkeletonGraph.from_txt()` also supports **GraphML** (`.graphml` / `.xml`) as the native round-trip format after optimization.

---

## Installation

From the repository root (uses [uv](https://github.com/astral-sh/uv) and `pyproject.toml`):

```bash
uv sync
```

Run code or tests inside that environment, for example:

```bash
uv run python your_script.py
uv run pytest
```

---

## End-to-end workflow (conceptual)

1. **Load** the mesh and skeleton.
2. **(Optional)** Run `SkeletonOptimizer` to nudge skeleton points toward a more medial, inside-the-volume placement (often helpful after MCF).
3. **Call `fit_morphology`** to resample the skeleton along edges, slice the mesh with planes normal to the local tangent, and assign a radius at each sample.
4. **Export** with `MorphologyGraph.to_swc_file()` (or `to_swc_model()` if you need an `SWCModel` object from `swctools`).
5. **(Optional)** Compare mesh vs. cable metrics with `Validation`, or **uniformly rescale all radii** so cable surface area or volume matches the mesh via `MorphologyGraph.scale_radii_to_match_mesh()` (solved numerically because lateral frustum area does not scale as a single power of radius when edge lengths are fixed).

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

[SWC](http://www.neuronland.org/NLMorphologyConverter/MorphologyFormats/SWC/Spec.html) is inherently a **tree** (each node has one parent). MaSCaF keeps **arbitrary graph topology** in memory and only enforces a tree when writing SWC, by duplicating an endpoint on closing edges and recording the mapping in comments when requested.

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

**Mean curvature flow** evolves a surface so that each point moves with velocity given by mean curvature; in practice it is a standard way to **collapse a shape toward a skeleton-like structure** while preserving topology in many settings.

A convenient way to get polylines compatible with MaSCaF is [CGAL’s Triangulated Surface Mesh Skeletonization](https://doc.cgal.org/latest/Surface_mesh_skeletonization/index.html). The CGAL distribution includes a demo application (**CGAL Lab**) that can run TSMS, as well as many other useful operations. Typical steps:

1. Open CGAL Lab, load your mesh.
2. Run something like **Operations → Triangulated Surface Mesh Skeletonization → Mean Curvature Skeleton (Advanced)** (wording may vary slightly by version).
3. Export the skeleton / fixed-points result to a **polylines** text file (one branch per line, `N x1 y1 z1 …` as above).

That file is only **one** possible input; any tool that abides the same polyline convention will work.

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
- [Triangle mesh (Wikipedia)](https://en.wikipedia.org/wiki/Triangle_mesh)
