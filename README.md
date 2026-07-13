# MASCAF ☕

[![GitHub release](https://img.shields.io/github/v/release/jmrfox/mascaf?label=version)](https://github.com/jmrfox/mascaf/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/jmrfox/mascaf/actions/workflows/publish-release.yml/badge.svg)](https://github.com/jmrfox/mascaf/actions/workflows/publish-release.yml)
[![Documentation](https://readthedocs.org/projects/mascaf/badge/?version=latest)](https://mascaf.readthedocs.io/en/latest/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20102245.svg)](https://doi.org/10.5281/zenodo.20102245)
[![bioRxiv](https://img.shields.io/badge/bioRxiv-2025.10.16.683802-b31b1b.svg)](https://www.biorxiv.org/content/10.64898/2026.05.10.721501v1)

**MASCAF** (Mesh and Skeleton Cable Fitting) is a Python package for fitting **cable-graph morphology** (such as an [SWC](http://www.neuronland.org/NLMorphologyConverter/MorphologyFormats/SWC/Spec.html) model) to a **closed triangle mesh** using a **3D curve skeleton** as geometric guidance.

This workflow was developed to create neuronal morphology models for multi-compartmental simulation from 3D surface meshes. MASCAF is especially useful for complex morphologies, including non-tree structures, where the output must preserve graph topology before eventual SWC export.

## [_Click here to read the preprint!_](https://www.biorxiv.org/content/10.64898/2026.05.10.721501v1)

## What you need

1. **Surface mesh** — Typically a watertight OBJ, or anything [trimesh](https://trimesh.org/) can load.
2. **Mesh skeletonization** — MASCAF includes an internal CGAL integration for skeletonization, but you can also use CGALLab or any other external method as long as the output matches the expected polylines format:
   - `.polylines.txt`
   - GraphML (`.graphml` / `.xml`)

---

## Installation

Install in your own project environment by providing `pip` with the repo link. This project works well with [uv](https://docs.astral.sh/uv/).

```bash
uv pip install https://github.com/jmrfox/mascaf.git
```

If you are working in a local clone of this repo:

```bash
uv sync
```

Run code or tests inside that environment:

```bash
uv run python your_script.py
uv run pytest
```

If you already have a valid skeleton file, you do **not** need the internal CGAL setup at all.

---

## End-to-end workflow

1. **Prepare a mesh** — Start from a closed triangle mesh.
2. **Generate or obtain a skeleton** — Use one of the skeletonization options below.
3. **Load inputs** — Load the mesh with `MeshManager` and the skeleton with `SkeletonGraph.from_txt()`.
4. **Fit a morphology graph** — Use `CableFitter` with `FitOptions`.
5. **Optionally optimize the morphology basis** — Use `BasisOptimizer` through `FitOptions.basis_optimizer_options` if you want additional geometric refinement before final radius fitting.
6. **Validate and export** — Inspect results, optionally scale radii to match the mesh, and export to SWC.

---

## Skeletonization options

### Option 1: internal CGAL integration

MASCAF includes a few CGAL operations:

- `repair(mesh -> mesh)`
- `simplify(mesh -> mesh)`
- `skeletonize(mesh -> .polylines.txt)`

The native C++ project lives in `cpp/`, and the Python interface for it lives in `mascaf.cgal`.

#### What the user must install manually

The Python module can help orchestrate the build, but it does **not** remove the need for a native toolchain. Before using internal CGAL, the user must install:

- a C++ compiler toolchain (e.g., Visual Studio Build Tools on Windows)
- [CMake](https://cmake.org/)
- [vcpkg](https://github.com/microsoft/vcpkg)
- the CGAL/Eigen dependencies resolved through the `cpp/vcpkg.json` manifest

#### Windows setup

1. Install Visual Studio or Build Tools for Visual Studio with C++ support.
2. Install CMake and make sure `cmake` is available on `PATH`.
3. Install vcpkg and set the `VCPKG_ROOT` environment variable.
4. Open the terminal where your compiler environment is available (e.g., Visual Studio Developer Command Prompt).
5. From the repo root, configure the native project:

```bash
cmake -S cpp -B cpp/build -DCMAKE_TOOLCHAIN_FILE="%VCPKG_ROOT%/scripts/buildsystems/vcpkg.cmake"
```

1. Build the executables:

```bash
cmake --build cpp/build --config Release
```

1. Confirm that executables such as `mesh_repair.exe`, `mesh_simplify.exe`, and `mesh_skeletonize.exe` exist in the build output.

#### Linux and macOS setup

1. Install a C++17-capable compiler toolchain.
2. Install CMake.
3. Install vcpkg and export `VCPKG_ROOT`.
4. Open a terminal where your compiler environment is available.
5. From the repo root, configure the native project:

```bash
cmake -S cpp -B cpp/build -DCMAKE_TOOLCHAIN_FILE="$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake"
```

1. Build the executables:

```bash
cmake --build cpp/build --config Release
```

1. Confirm that `mesh_repair`, `mesh_simplify`, and `mesh_skeletonize` are present in the build output.

#### Shell-based usage after build

The current skeletonization executable accepts:

```text
mesh_skeletonize <input.obj> <output.polylines.txt> [w_H] [w_M]
```

where:

- `w_H` is the quality/speed tradeoff (`quality_speed_tradeoff`)
- `w_M` is the medially-centered speed tradeoff (`medially_centered_speed_tradeoff`)

Example:

```bash
mesh_skeletonize neuron.obj neuron.polylines.txt 0.5 5.0
```

#### Python build orchestration with `CGALBuilder`

MASCAF includes helpers for configure/build orchestration and executable discovery:

- `CGALConfig`
- `CGALBuilder`
- `CGALOperator`

Example:

```python
from mascaf import CGALBuilder, CGALConfig

config = CGALConfig(build_dir="cpp/build")
builder = CGALBuilder(config)

builder.run_configure()
builder.run_build(config="Release")
```

This still assumes the native toolchain prerequisites are already installed correctly.

#### Python CGAL operation usage after build

```python
from mascaf import CGALConfig, CGALOperator

config = CGALConfig(build_dir="cpp/build")
ops = CGALOperator(config=config)

ops.skeletonize(
    "neuron.obj",
    "neuron.polylines.txt",
    quality_speed_tradeoff=0.5,
    medially_centered_speed_tradeoff=5.0,
)
```

MASCAF also includes a placeholder Python method for suggesting these parameters automatically in the future, but it does not yet perform real tuning.

### Option 2: CGAL Lab

The CGAL distribution includes a demo application (**CGAL Lab**) that can run triangulated surface mesh skeletonization interactively.

Typical steps:

1. Download CGAL Lab from the CGAL website: [CGAL Lab 6.1 Download](https://www.cgal.org/demo/6.1/CGALlab.zip).
2. Open CGAL Lab and load your mesh.
3. Run **Operations → Triangulated Surface Mesh Skeletonization → Mean Curvature Skeleton (Advanced)**.
4. Export the result as a **polylines** text file.

### Option 3: any other compatible tool

You can use any external method if it produces a MASCAF-compatible polylines file:

```text
N x1 y1 z1 x2 y2 z2 ... xN yN zN
```

That is, one line per branch, starting with the number of points on that line.

---

## Main pieces of the API

| Piece | Role |
|--------|------|
| `MeshManager` | Load and hold a `trimesh.Trimesh` mesh. |
| `SkeletonGraph` | Load and store skeleton geometry from polylines text or GraphML. |
| `FitOptions` | Controls cable fitting, radius estimation, and optional basis optimization. |
| `CableFitter` | Main workflow object for converting mesh + skeleton into a `MorphologyGraph`. |
| `BasisOptimizerOptions` | Configuration for morphology-basis optimization before final radius fitting. |
| `BasisOptimizer` | Geometric optimization of the morphology basis against the mesh. |
| `MorphologyGraph` | Graph-based morphology representation with radii and SWC export support. |
| `Validation` | Compare mesh and morphology geometry metrics. |
| `CGALConfig`, `CGALBuilder`, `CGALOperator` | Optional CGAL-backed native preprocessing and build orchestration. |

---

## How the fitting workflow works

1. **Resampling and basis construction** — `CableFitter` converts the skeleton into a morphology basis whose segment lengths satisfy `FitOptions.max_edge_length`.
2. **Optional basis optimization** — If `FitOptions.basis_optimizer_options` is set, MASCAF runs `BasisOptimizer` on the intermediate morphology basis before radius fitting.
3. **Cross-section and radius estimation** — MASCAF estimates local radii from mesh geometry using the configured `radius_strategy`.
4. **Output morphology graph** — The result is a `MorphologyGraph` that preserves graph connectivity, including cycles.

Supported radius strategies include:

- `equivalent_area`
- `equivalent_perimeter`
- `section_median`
- `section_circle_fit`
- `nearest_surface`

However, `equivalent_area` is recommended for most use cases.

At branch points, `multi_tangent_reduction` is applied to the set of per-edge radii, e.g. `median(r_1, r_2, ..., r_n) = r_branch`.

---

## SWC and cycles

[SWC](http://www.neuronland.org/NLMorphologyConverter/MorphologyFormats/SWC/Spec.html) is inherently a tree format, but MASCAF can handle morphologies with topological cycles. MASCAF keeps arbitrary graph topology in memory and only enforces a tree when writing SWC, by duplicating nodes where necessary and optionally recording cycle-closure information in the output.

By default, MASCAF will duplicate nodes to break cycles when writing SWC files. This ensures that the output SWC file is a valid tree structure while preserving the original graph topology in memory. For each broken cycle, a directive is added to the SWC file header specifying reconnection: `# CYCLE_BREAK <node_id> <parent_id>`.

---

## Example: current Python workflow

```python
from mascaf import (
    BasisOptimizerOptions,
    CableFitter,
    FitOptions,
    MeshManager,
    SkeletonGraph,
)

mesh_mgr = MeshManager(mesh_path="neuron.obj")
skeleton = SkeletonGraph.from_txt("neuron.polylines.txt")

fit_options = FitOptions(
    max_edge_length=1.0,
    radius_strategy="equivalent_area",
    basis_optimizer_options=BasisOptimizerOptions(
        do_snapping=True,
        do_forcing=True,
        max_iterations=50,
        lambda_smooth=0.5,
    ),
)

fitter = CableFitter(fit_options)
morphology = fitter.fit(mesh_mgr, skeleton)

morphology.scale_radii_to_match_mesh(mesh_mgr, metric="surface_area")
morphology.to_swc_file("neuron.swc")
```

Minimal example if you already have a skeleton and do not want basis optimization:

```python
from mascaf import CableFitter, FitOptions, MeshManager, SkeletonGraph

mesh_mgr = MeshManager(mesh_path="shape.obj")
skel = SkeletonGraph.from_txt("shape.polylines.txt")
fitter = CableFitter(FitOptions(max_edge_length=0.5))
morph = fitter.fit(mesh_mgr, skel)
morph.to_swc_file("shape.swc")
```

---

## Troubleshooting internal CGAL

- **Executable not found**
  - Build the native `cpp/` project first.
  - Make sure you are looking in the same build configuration that you compiled, such as `Release` or `Debug`.

- **`VCPKG_ROOT` is not set**
  - Set the environment variable so CMake or `CGALBuilder` can locate the vcpkg toolchain file.

- **CMake cannot find CGAL or Eigen**
  - Confirm that you configured with the vcpkg toolchain file and that the `cpp/vcpkg.json` dependencies were resolved.

- **The project built in `Debug`, but MASCAF cannot find the binary**
  - Point `CGALConfig(build_dir=...)` or `CGALConfig(executable_dir=...)` at the actual output location.

- **Skeletonization fails on the mesh**
  - The current CGAL skeletonization path expects a closed triangle mesh.

- **You already have a valid skeleton file**
  - Skip internal CGAL entirely and load the skeleton directly with `SkeletonGraph.from_txt()`.

---

## Further reading in the repo

- `mascaf/cable_fitting.py` — current fitting workflow, `CableFitter`, and `FitOptions`
- `mascaf/basis_optimizer.py` — morphology-basis optimization
- `mascaf/morphology_graph.py` — SWC export and geometry utilities
- `mascaf/cgal.py` — optional internal CGAL integration
- `tests/` — runnable examples and regression coverage

---

## Community

- [Contributing guide](CONTRIBUTING.md)
- [Code of Conduct](CODE_OF_CONDUCT.md)
- [License (MIT)](LICENSE)
- [Changelog](CHANGELOG.md)

---

## References

- [SWC format overview](http://www.neuronland.org/NLMorphologyConverter/MorphologyFormats/SWC/Spec.html)
- [CGAL: Triangulated Surface Mesh Skeletonization](https://doc.cgal.org/latest/Surface_mesh_skeletonization/index.html)
