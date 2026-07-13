# Changelog

All notable changes to MASCAF will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project uses [Semantic Versioning](https://semver.org/).

---

## [1.1.0] — 2026-07-13

### Added

- Weighted localized centering force in `BasisOptimizer`, with magnitude-matched
  `lambda_smooth` / `lambda_vertex` terms, `step_cap_factor`, and related options
  (`repulsion_power`, `localization_beta`, `weight_epsilon`,
  `vertex_repulsion_distance`).
- Chord-midpoint snapping with orthogonal ray perturbation and a minimum chord
  length filter (`snap_ray_perturb_*`, `snap_min_chord_fraction` /
  `snap_min_chord_length`) so grazing surface hits are not treated as volume
  chords.
- `MorphologyGraph.from_skeleton_graph`, `resample`, and
  `from_skeleton_graph_resample` for exact copy vs resampled basis construction.
- `MorphologyGraph.get_outside_nodes` for mesh-containment queries.
- Optional skeleton node markers in `MeshManager.visualize_mesh_3d`
  (`skel_marker_size`).
- `demo/demo_optimize` notebook for TS1 basis optimization.

### Changed

- `BasisOptimizerOptions.step_size` renamed to `step_scale`.
- `BasisOptimizerOptions.smoothing_weight` renamed to `lambda_smooth`.
- `CableFitter` builds the morphology basis via
  `MorphologyGraph.from_skeleton_graph_resample`.

### Removed

- `BasisOptimizerOptions.snap_distance_multiplier` (replaced by chord-midpoint snap).
- `BasisOptimizerOptions.fallback_distance` (missed rays raise instead of falling back).

---

## [1.0.2] — 2026-05-10

### Added

- Sphinx documentation site with API reference and user guide (`docs/`).
- Read the Docs configuration (`.readthedocs.yaml`).
- NumPy-style docstrings across all public modules: `mesh`, `cable_fitting`, `basis_optimizer`, `cgal`, `morphology_graph`, `skeleton`, `validation`, `visualization`.
- `docs` dependency group in `pyproject.toml` (`sphinx`, `furo`, `myst-parser`, `sphinx-autodoc-typehints`).
- Read the Docs badge in `README.md`.

---

## [1.0.1] — 2026-05-10

### Added

- `demo/` directory with Jupyter notebooks and scripts demonstrating the full pipeline and CGAL integration.

### Fixed

- `CGALOperator._run_operation` no longer resolves relative input/output paths to absolute paths, so paths passed to `simplify` and `skeletonize` are forwarded to the subprocess unchanged.
- `FitOptions.multi_tangent_reduction` default corrected to `"mean"`.
- Test data paths updated to point at `mascaf/demo/` where demo polyline files are located.

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
