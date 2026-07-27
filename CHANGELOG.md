# Changelog

All notable changes to MASCAF will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project uses [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

---

## [1.2.0] — 2026-07-27

### Added

- `mascaf.logging_config.configure_logging` for CLI/notebook runs: timestamped
  console and/or file handlers, ``mascaf`` logger level, and optional
  suppression of noisy third-party DEBUG loggers (plotting/export: kaleido,
  choreographer, swctools, etc.). CLI scripts may expose ``--full-debug`` to
  include those loggers.
- Detailed DEBUG logging across the TS pipeline script and core stages
  (resample, basis optimizer phases, radius fitting, validation metrics).
- `BasisOptimizerOptions.ray_jitter` (default ``0.0``): per forcing iteration,
  independently perturb each centering-ray direction by this angular scale
  (approx. radians); shared across nodes within an iteration. Off when ``0``.
- Optional forcing ``active_resample`` (default off) with
  ``active_resample_min_fraction`` / ``active_resample_max_fraction``: after
  each forcing iteration, merge endpoints of edges shorter than the min
  (``consolidate_nodes``) and bisect edges longer than the max
  (``bisect_edge``), using fractions of the mesh bounding-box diagonal.
  Requires ``max_fraction >= 2 * min_fraction`` to avoid merge/split
  oscillation. Branch nodes are eligible; degree is preserved as neighbor
  union (``m+n-2`` when endpoints share no other neighbors). Midpoints that
  land outside the mesh are chord-snapped inside (same helper as the
  snapping phase); failure raises ``RuntimeError``. Consolidations that would
  change the cyclomatic number are skipped by default with a warning; set
  ``active_resample_allow_cycle_collapse`` to allow them.
- `mascaf.mesh_contains` with ``point_inside_mesh`` / ``points_inside_mesh``
  (signed-distance containment; positive inside, negative outside; surface
  shell within tolerance counts as inside). Used by MorphologyGraph outside
  queries, BasisOptimizer snapping/forcing/centering, and SkeletonGraph
  outside-only projection so those checks agree.
- `BasisOptimizerOptions.step_scale` (default ``0.5``): multiplies the blended
  forcing update before surface-distance capping.
- `BasisOptimizerOptions.pruning_length` and `pruning_length_fraction`:
  prune terminal→branch stubs shorter than an absolute length, or shorter than
  ``pruning_length_fraction * longest_terminal_stub`` (resolved once at phase
  start). Absolute wins if both are set.

### Changed

- Basis forcing blend is again magnitude-matched:
  ``delta_v = (1 - alpha_s) F_centering + alpha_s F_smoothing ||F_c|| / ||F_s||``
  (replaces independent ``lambda_centering`` / ``lambda_smoothing`` scales).
- Basis pruning only removes terminal→branch stubs under the threshold;
  tip↔tip isolated chains are no longer auto-removed.
- ``MorphologyGraph.get_outside_nodes`` no longer uses ``mesh.contains``.
- Chord-midpoint snapping considers consecutive ray-hit pairs (including on
  odd-hit rays), skipping short grazing chords, so thin/crease cases that
  previously left nodes outside after snapping can be recovered.
- Snapping places nodes at ``snap_chord_fraction`` along the enter→exit chord
  (default ``0.25``, was fixed midpoint ``0.5``).

### Removed

- `BasisOptimizerOptions.lambda_centering` and `lambda_smoothing`
  (use ``alpha_s``).
- `BasisOptimizerOptions.pruning_min_length` and `pruning_min_length_fraction`
  (percentile-based; replaced by ``pruning_length`` /
  ``pruning_length_fraction`` as fraction of the longest stub).

---

## [1.1.1] — 2026-07-13

### Changed

- Basis forcing now uses independent scales
  ``delta_v = lambda_centering * F_centering + lambda_smoothing * F_smoothing``,
  with ``F_centering`` scaled by ``d_min`` and ``F_smoothing`` equal to the
  raw neighbor-centroid pull (surface-distance step capping unchanged).

### Removed

- `BasisOptimizerOptions.step_scale`, `lambda_smooth`, `lambda_vertex`, and
  `vertex_repulsion_distance`.

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
