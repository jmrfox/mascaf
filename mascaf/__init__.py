"""
MaSCaF (Mesh and Skeleton Cable Fitting) — Python package ``mascaf``.

A lightweight toolkit for converting mesh cross-sections and skeleton graph guidance
into SWC models.

Terminology:
- "skeleton": The mesh centroid (SkeletonGraph, result of MCF calculation) without radii
- "SWC model" or "swc": Skeleton with radii information attached to each node

Public API:
- MeshManager
- SkeletonGraph
- SWCModel (from swctools)
- MorphologyGraph
- FitOptions, CableFitter
"""

from __future__ import annotations

import logging
from importlib.metadata import PackageNotFoundError, version

# Package version reported from installed metadata (fallback for editable/dev installs)
try:
    __version__ = version("mascaf")
except PackageNotFoundError:  # pragma: no cover - best-effort in dev
    __version__ = "0.1.0"

# Avoid "No handler found" warnings for library users; applications can configure logging.
logging.getLogger(__name__).addHandler(logging.NullHandler())

# Re-export primary classes and functions for convenient access at package level
from swctools import SWCModel  # noqa: E402

from .basis_optimizer import BasisOptimizer, BasisOptimizerOptions  # noqa: E402
from .cable_fitting import CableFitter, FitOptions  # noqa: E402
from .cgal import (  # noqa: E402
    CGALBuilder,
    CGALBuildError,
    CGALCommandResult,
    CGALConfig,
    CGALError,
    CGALExecutableNotFoundError,
    CGALMeshProcessor,
    CGALOperator,
)
from .graph3d import Graph3D  # noqa: E402
from .mesh import MeshManager, example_mesh  # noqa: E402
from .skeleton import SkeletonGraph  # noqa: E402
from .morphology_graph import (  # noqa: E402
    MorphologyGraph,
    Junction,
)

from .validation import (  # noqa: E402
    Validation,
)
from .visualization import (  # noqa: E402
    plot_surface_mesh_grid,
    save_surface_meshes_svg,
    save_surface_mesh_grid_svg,
)

__all__ = [
    "__version__",
    # Mesh and skeleton
    "MeshManager",
    "example_mesh",
    "Graph3D",
    "SkeletonGraph",
    # SWC model (from swctools)
    "SWCModel",
    # Tracing API
    "CableFitter",
    "FitOptions",
    "CGALError",
    "CGALConfig",
    "CGALBuilder",
    "CGALBuildError",
    "CGALExecutableNotFoundError",
    "CGALCommandResult",
    "CGALMeshProcessor",
    "CGALOperator",
    "BasisOptimizer",
    "BasisOptimizerOptions",
    "MorphologyGraph",
    "Junction",
    # Validation
    "Validation",
    # Visualization
    "plot_surface_mesh_grid",
    "save_surface_meshes_svg",
    "save_surface_mesh_grid_svg",
]
