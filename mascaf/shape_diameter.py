"""Shape diameter function (SDF) and local mesh thickness summaries.

Skeletons and morphology graphs do not store radii. Local tube / feature scale
for parameter oracles must be derived from the **mesh**. This module provides a
classic surface shape-diameter estimate (Shapira et al.–style): from surface
samples, cast a cone of rays opposite the outward normal and robustly average
opposite-surface hit distances to get a local **diameter**.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import trimesh

from .mesh_contains import signed_distances

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@dataclass(frozen=True)
class ThicknessSummary:
    """Summary statistics of local mesh diameter samples."""

    n_samples: int
    median: float
    mean: float
    p10: float
    p90: float
    cv: float
    """Coefficient of variation ``std / median`` (NaN if median ~ 0)."""

    @property
    def radius_proxy(self) -> float:
        """``median / 2`` as a local radius proxy."""
        return 0.5 * self.median


def _uniform_cone_directions(
    axis: np.ndarray,
    n_rays: int,
    cone_angle: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Unit directions in a cone around ``axis`` (half-angle in radians)."""
    axis = np.asarray(axis, dtype=float)
    axis = axis / (np.linalg.norm(axis) + 1e-15)
    tmp = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(axis, tmp)
    u /= np.linalg.norm(u) + 1e-15
    v = np.cross(axis, u)
    dirs = np.empty((n_rays, 3), dtype=float)
    cos_a = float(np.cos(cone_angle))
    for i in range(n_rays):
        z = rng.uniform(cos_a, 1.0)
        phi = rng.uniform(0.0, 2.0 * np.pi)
        r = np.sqrt(max(0.0, 1.0 - z * z))
        d = z * axis + r * np.cos(phi) * u + r * np.sin(phi) * v
        dirs[i] = d / (np.linalg.norm(d) + 1e-15)
    return dirs


def compute_shape_diameter(
    mesh: trimesh.Trimesh,
    *,
    n_samples: int = 200,
    n_rays: int = 8,
    cone_angle: float = np.deg2rad(45.0),
    normal_offset_fraction: float = 1e-4,
    seed: Optional[int] = 0,
) -> np.ndarray:
    """Return per-sample local diameter estimates via surface SDF.

    Parameters
    ----------
    mesh :
        Target triangle mesh. Watertight meshes give the most reliable results;
        open meshes may yield fewer valid samples.
    n_samples :
        Number of face centroids to probe (capped by face count).
    n_rays :
        Rays per sample inside the inward cone.
    cone_angle :
        Cone half-angle in radians around the inward normal.
    normal_offset_fraction :
        Inward offset as a fraction of the bbox diagonal to avoid self-hits.
    seed :
        RNG seed for face and cone sampling (``None`` for nondeterministic).

    Returns
    -------
    ndarray
        1-D array of successful diameter samples (may be empty).
    """
    rng = np.random.default_rng(seed)
    n_faces = len(mesh.faces)
    if n_faces == 0:
        return np.zeros(0, dtype=float)

    # Ensure face normals exist
    _ = mesh.face_normals
    face_idx = rng.choice(n_faces, size=min(int(n_samples), n_faces), replace=False)
    origins = mesh.triangles[face_idx].mean(axis=1)
    normals = np.asarray(mesh.face_normals[face_idx], dtype=float)

    diag = float(np.linalg.norm(np.asarray(mesh.extents, dtype=float)))
    eps = max(diag * float(normal_offset_fraction), 1e-9)

    diameters: list[float] = []
    for origin, normal in zip(origins, normals):
        n = normal / (np.linalg.norm(normal) + 1e-15)
        # Orient normal outward: outward probe should not be deeply inside
        sd_out = float(signed_distances(mesh, origin + eps * n)[0])
        if sd_out > 0:
            n = -n
        start = origin - eps * n
        inward = -n
        dirs = _uniform_cone_directions(inward, int(n_rays), float(cone_angle), rng)
        hits: list[float] = []
        for d in dirs:
            try:
                locations, _, _ = mesh.ray.intersects_location(
                    ray_origins=start.reshape(1, 3),
                    ray_directions=d.reshape(1, 3),
                )
            except Exception:
                continue
            if len(locations) == 0:
                continue
            dist = np.linalg.norm(locations - start, axis=1)
            order = np.argsort(dist)
            for di in dist[order]:
                if float(di) > 5.0 * eps:
                    hits.append(float(di))
                    break
        if hits:
            diameters.append(float(np.median(hits)))

    arr = np.asarray(diameters, dtype=float)
    logger.debug(
        "SDF: %d/%d face samples yielded diameters (median=%.6g)",
        arr.size,
        len(face_idx),
        float(np.median(arr)) if arr.size else float("nan"),
    )
    return arr


def summarize_thickness(samples: np.ndarray) -> ThicknessSummary:
    """Summarize a 1-D array of diameter samples."""
    arr = np.asarray(samples, dtype=float).reshape(-1)
    if arr.size == 0:
        nan = float("nan")
        return ThicknessSummary(
            n_samples=0,
            median=nan,
            mean=nan,
            p10=nan,
            p90=nan,
            cv=nan,
        )
    med = float(np.median(arr))
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    return ThicknessSummary(
        n_samples=int(arr.size),
        median=med,
        mean=mean,
        p10=float(np.percentile(arr, 10)),
        p90=float(np.percentile(arr, 90)),
        cv=float(std / med) if med > 1e-12 else float("nan"),
    )


def mesh_thickness_summary(
    mesh: trimesh.Trimesh,
    **sdf_kwargs,
) -> ThicknessSummary:
    """Compute SDF samples on ``mesh`` and return a :class:`ThicknessSummary`."""
    return summarize_thickness(compute_shape_diameter(mesh, **sdf_kwargs))
