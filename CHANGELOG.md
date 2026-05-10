# Changelog

All notable changes to MASCAF will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project uses [Semantic Versioning](https://semver.org/).

---

## [1.0.0] — 2026-05-09

### Added

- `CableFitter` and `FitOptions` for fitting cable-graph morphology to a closed triangle mesh using a 3D curve skeleton.
- `BasisOptimizer` and `BasisOptimizerOptions` for optional geometric refinement of the morphology basis prior to radius fitting.
- `MorphologyGraph` with full SWC export, cycle handling (node duplication + `CYCLE_BREAK` header directives), and radius scaling against mesh geometry.
- `SkeletonGraph` with loading from `.polylines.txt` and GraphML formats.
- `MeshManager` wrapping `trimesh` for mesh loading and querying.
- `Validation` utilities for comparing mesh and morphology geometry.
- Radius estimation strategies: `equivalent_area`, `equivalent_perimeter`, `section_median`, `section_circle_fit`, `nearest_surface`.
- Optional internal CGAL integration (`CGALConfig`, `CGALBuilder`, `CGALOperator`) for mesh repair, simplification, and skeletonization via compiled C++ executables.
- `mascaf.demo` subpackage with example meshes, skeletons, and SWC files (cylinder, branching, torus, human cell).
- MIT license.
- `CONTRIBUTING.md` and `CODE_OF_CONDUCT.md`.
