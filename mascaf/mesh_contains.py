"""Unified mesh volume containment via signed distance.

Uses ``trimesh.proximity.signed_distance``. On the watertight volume meshes
used in this project (verified on unit icospheres), the sign convention is:

- **positive** — inside the volume (distance to the surface)
- **negative** — outside the volume

A point counts as **inside** when it is in the interior or within ``tol`` of
the surface (``sd >= -tol``). Only points with ``sd < -tol`` are exterior.
This module does **not** call ``mesh.contains``, which is flaky near the
surface.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import trimesh
from trimesh.proximity import signed_distance

ArrayLike = Union[np.ndarray, list, tuple]


def default_distance_tol(
    mesh: trimesh.Trimesh,
    *,
    tol: Optional[float] = None,
    tol_fraction: float = 1e-6,
) -> float:
    """Absolute exterior tolerance, or ``tol_fraction * ||mesh.extents||``."""
    if tol is not None:
        return float(tol)
    extents = np.asarray(mesh.extents, dtype=float)
    diagonal = float(np.linalg.norm(extents))
    return float(tol_fraction) * diagonal


def signed_distances(mesh: trimesh.Trimesh, points: ArrayLike) -> np.ndarray:
    """Return signed distances for one or more points, shape ``(N,)``.

    Positive values are inside; negative values are outside (see module docs).
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim == 1:
        pts = pts.reshape(1, 3)
    elif pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must have shape (3,) or (N, 3), got {pts.shape}")
    return np.asarray(signed_distance(mesh, pts), dtype=float).reshape(-1)


def points_inside_mesh(
    mesh: trimesh.Trimesh,
    points: ArrayLike,
    *,
    tol: Optional[float] = None,
    tol_fraction: float = 1e-6,
) -> np.ndarray:
    """Return a boolean mask: True where each point is inside (or on-shell).

    Parameters
    ----------
    mesh :
        Watertight volume mesh.
    points :
        Coordinates with shape ``(3,)`` or ``(N, 3)``.
    tol :
        Absolute exterior tolerance. When ``None``, uses
        ``tol_fraction * ||mesh.extents||``.
    tol_fraction :
        Relative scale used when ``tol`` is ``None``.

    Returns
    -------
    ndarray of bool, shape ``(N,)``
        ``True`` iff ``signed_distance >= -tol``.
    """
    dists = signed_distances(mesh, points)
    threshold = default_distance_tol(mesh, tol=tol, tol_fraction=tol_fraction)
    return dists >= -threshold


def point_inside_mesh(
    mesh: trimesh.Trimesh,
    point: ArrayLike,
    *,
    tol: Optional[float] = None,
    tol_fraction: float = 1e-6,
) -> bool:
    """Return whether a single point is inside the mesh (or within ``tol``)."""
    return bool(
        points_inside_mesh(
            mesh,
            point,
            tol=tol,
            tol_fraction=tol_fraction,
        )[0]
    )
