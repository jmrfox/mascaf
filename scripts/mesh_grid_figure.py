"""
Paper-style figures: grid of 3D surface meshes with parallel isometric views.

Meshes are centered, optionally auto-rotated with **vertex PCA** (SVD of centered
coordinates). Principal directions are ordered by **descending** singular value
(1st = most spread). They are aligned to a fixed **orthonormal isometric screen
basis**:

- **1st** principal → screen **up** (``cross(right, view)`` with world +Z),
- **2nd** → **horizontal** (``normalize(cross(view, world_up))``),
- **3rd** → **toward the camera** (``view_dir``, default ``(1,1,1)/‖·‖``).

Rotation is ``R = T @ P.T`` with ``P`` the 3×3 column matrix of principal axes
and ``T`` the target basis. Near-spherical clouds skip rotation.

Example::

    uv run python scripts/mesh_grid_figure.py --out figures/demo.png

    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path("scripts").resolve()))
    from mesh_grid_figure import plot_surface_mesh_grid

    import trimesh
    meshes = [trimesh.creation.cylinder(), trimesh.creation.box()]
    plot_surface_mesh_grid(meshes, grid_shape=(1, 2), out_path="grid.png")
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Sequence, Union

import matplotlib as mpl
import numpy as np
import pyvista as pv
import trimesh
from PIL import Image, ImageDraw, ImageFont

MeshLike = Union[str, trimesh.Trimesh, pv.PolyData]

# Matches PyVista's default isometric camera: from (+,+,+) toward the origin.
_DEFAULT_ISO_VIEW_DIR = np.array([1.0, 1.0, 1.0], dtype=float)
_DEFAULT_ISO_VIEW_DIR /= np.linalg.norm(_DEFAULT_ISO_VIEW_DIR)

_WORLD_UP = np.array([0.0, 0.0, 1.0])

# Repo root (parent of ``scripts/``).
_REPO_ROOT = Path(__file__).resolve().parent.parent

_DEFAULT_SIMPLIFIED_MESH_COLOR = "#5d7a99"


def _to_polydata(mesh: MeshLike) -> pv.PolyData:
    """Convert ``mesh`` to ``PolyData`` for plotting."""
    if isinstance(mesh, pv.PolyData):
        return mesh.copy(deep=True)
    if isinstance(mesh, trimesh.Trimesh):
        return pv.wrap(mesh).copy(deep=True)
    if isinstance(mesh, str):
        loaded = trimesh.load(mesh)
        if isinstance(loaded, trimesh.Scene):
            geoms = list(loaded.geometry.values())
            if not geoms:
                raise ValueError(f"No geometry in scene: {mesh}")
            loaded = geoms[0]
        if not isinstance(loaded, trimesh.Trimesh):
            raise TypeError(f"Expected a single mesh file, got {type(loaded)} for {mesh}")
        return pv.wrap(loaded).copy(deep=True)
    raise TypeError(f"Unsupported mesh type: {type(mesh)}")


def _subsample_points(pts: np.ndarray, max_points: int) -> np.ndarray:
    n = len(pts)
    if n <= max_points:
        return pts
    rng = np.random.default_rng(0)
    idx = rng.choice(n, size=max_points, replace=False)
    return pts[idx]


def _isometric_target_frame(view_dir: np.ndarray) -> np.ndarray:
    """
    Orthonormal 3×3 ``T`` with columns ``[screen_up, horizontal, toward_camera]``
    matching a default isometric camera with world +Z as view-up hint.

    ``toward_camera`` is a unit vector; ``horizontal`` and ``screen_up`` are
    built so ``{screen_up, horizontal, toward_camera}`` is right-handed.
    """
    v = np.asarray(view_dir, dtype=float)
    v = v / np.linalg.norm(v)
    up_w = _WORLD_UP
    r = np.cross(v, up_w)
    rn = float(np.linalg.norm(r))
    if rn < 1e-12:
        r = np.array([1.0, 0.0, 0.0])
    else:
        r = r / rn
    screen_up = np.cross(r, v)
    sun = float(np.linalg.norm(screen_up))
    if sun < 1e-12:
        return np.eye(3)
    screen_up = screen_up / sun
    t = np.column_stack([screen_up, r, v])
    if float(np.linalg.det(t)) < 0.0:
        t[:, 1] *= -1.0
    return t


def _pca_principal_frame(vertices_centered: np.ndarray) -> np.ndarray | None:
    """
    Columns are orthonormal principal directions for centered vertices, ordered by
    **descending** singular value (row ``vh[0]`` = strongest spread). Returns
    ``None`` if the cloud is nearly spherical (no stable frame).
    """
    pts = _subsample_points(np.asarray(vertices_centered, dtype=float), max_points=20_000)
    if len(pts) < 3:
        return None
    x = pts - pts.mean(axis=0)
    if np.allclose(x.std(axis=0), 0.0):
        return None
    _, s, vh = np.linalg.svd(x, full_matrices=False)
    s = np.asarray(s, dtype=float)
    if s[0] <= 1e-12 or len(s) < 3:
        return None
    if float(s[2] / s[0]) > 0.999:
        return None
    p = np.column_stack(
        [
            np.asarray(vh[0], dtype=float),
            np.asarray(vh[1], dtype=float),
            np.asarray(vh[2], dtype=float),
        ]
    )
    if float(np.linalg.det(p)) < 0.0:
        p[:, 2] *= -1.0
    return p


def _rotation_principal_axes_to_isometric_screen(
    vertices_centered: np.ndarray,
    view_dir: np.ndarray,
) -> np.ndarray:
    """
    Rotation ``R`` with ``R @ p_i = t_i`` for PCA columns ``p_i`` and target
    isometric basis columns ``t_i`` (up, horizontal, toward camera): ``R = T @ P.T``.
    """
    p = _pca_principal_frame(vertices_centered)
    if p is None:
        return np.eye(3)
    vdir = np.asarray(view_dir, dtype=float)
    vdir = vdir / np.linalg.norm(vdir)
    t = _isometric_target_frame(vdir)
    return t @ p.T


def _prepare_mesh_polydata(
    poly: pv.PolyData,
    *,
    center: bool,
    auto_rotate: bool,
    view_dir: np.ndarray,
) -> tuple[pv.PolyData, float]:
    """
    Center (optional), rotate (optional), return mesh and bounding-sphere radius
    for orthographic framing.
    """
    out = poly.copy(deep=True)
    pts = np.asarray(out.points, dtype=float)
    if pts.size == 0:
        return out, 1.0

    if center:
        c = pts.mean(axis=0)
        pts = pts - c
    else:
        pts = pts.copy()

    if auto_rotate:
        r = _rotation_principal_axes_to_isometric_screen(pts, view_dir)
        pts = (r @ pts.T).T

    out.points = pts
    norms = np.linalg.norm(pts, axis=1)
    radius = float(np.max(norms)) if len(norms) else 1.0
    if radius < 1e-12:
        radius = 1.0
    return out, radius


def _set_isometric_parallel_camera(plotter: pv.Plotter, parallel_scale: float) -> None:
    plotter.enable_parallel_projection()
    plotter.view_isometric()
    plotter.camera.parallel_scale = float(parallel_scale)


def _add_scale_bar_png_bottom_right(
    path: str | Path,
    *,
    parallel_scale: float,
    bar_length_um: float,
    pixels_to_um: float,
    grid_shape: tuple[int, int],
    margin_x: int = 20,
    margin_y: int = 22,
    line_width: int = 3,
    ui_scale: int = 1,
) -> None:
    """
    Draw a horizontal black scale bar and label on a saved PNG (image coordinates).

    Mesh coordinates are in pixels; ``micrometers = pixels * pixels_to_um``.
    Bar extent in mesh units is ``bar_length_um / pixels_to_um``. Orthographic
    vertical span per subplot is ``2 * parallel_scale`` mesh units over the
    subplot pixel height.
    """
    path = Path(path)
    if parallel_scale <= 0 or pixels_to_um <= 0:
        return

    world_len = float(bar_length_um) / float(pixels_to_um)
    n_rows, _n_cols = int(grid_shape[0]), int(grid_shape[1])
    us = max(1, int(ui_scale))

    img = Image.open(path).convert("RGBA")
    w_act, h_act = img.size
    cell_h = float(h_act) / float(n_rows)
    bar_px = world_len * cell_h / (2.0 * float(parallel_scale))
    bar_px = max(8.0 * us, bar_px)

    mx = int(margin_x * us)
    my = int(margin_y * us)
    lw = max(1, int(round(line_width * us)))

    draw = ImageDraw.Draw(img)
    y = h_act - my
    x2 = w_act - mx
    x1 = int(round(x2 - bar_px))
    draw.line([(x1, y), (x2, y)], fill=(0, 0, 0, 255), width=lw)

    label = f"{float(bar_length_um):g} μm"
    font_path = Path(mpl.get_data_path()) / "fonts/ttf/DejaVuSans.ttf"
    try:
        font = ImageFont.truetype(str(font_path), size=max(10, int(round(16 * us))))
    except OSError:
        font = ImageFont.load_default()

    if hasattr(draw, "textbbox"):
        bx0, by0, bx1, by1 = draw.textbbox((0, 0), label, font=font)
        tw, th = bx1 - bx0, by1 - by0
    else:
        tw, th = draw.textsize(label, font=font)
    pad = max(4, int(8 * us))
    text_x = x2 + pad if x2 + pad + tw <= w_act - 4 else max(4, x1 - tw - pad)
    text_y = int(y - th / 2 - 2)
    draw.text((text_x, text_y), label, fill=(0, 0, 0, 255), font=font)

    img.save(path)


def plot_surface_mesh_grid(
    meshes: Sequence[MeshLike],
    grid_shape: tuple[int, int],
    *,
    out_path: str | None = None,
    show: bool = False,
    center_each: bool = True,
    auto_rotate_each: bool = True,
    view_dir: np.ndarray | None = None,
    parallel_scale_margin: float = 1.08,
    zoom: float = 1.0,
    mesh_color: str | None = None,
    colors: Sequence[str] | None = None,
    window_size: tuple[int, int] | None = None,
    screenshot_scale: int | None = None,
    background: str = "white",
    off_screen: bool | None = None,
    scale_bar_um: float | None = None,
    pixels_to_um: float = 5.0 / 1000.0,
) -> pv.Plotter:
    """
    Plot multiple surface meshes on a regular grid with parallel isometric views.

    Each cell uses the same orthographic ``parallel_scale``, computed from the
    largest bounding-sphere radius among prepared meshes, so relative physical
    size (in the same units) is preserved across panels.

    Parameters
    ----------
    meshes
        Sequence of paths, ``trimesh.Trimesh``, or ``pyvista.PolyData``.
    grid_shape
        ``(n_rows, n_cols)``. Must satisfy ``n_rows * n_cols >= len(meshes)``.
    out_path
        If set, save a raster image (extension chooses format, e.g. ``.png``).
    show
        If True, open an interactive window (forces on-screen rendering).
    center_each
        Translate each mesh so its vertex centroid is at the origin.
    auto_rotate_each
        Apply vertex PCA then map 1st/2nd/3rd principal directions to isometric
        screen **up**, **horizontal**, and **toward the camera** (see module
        docstring). Skips rotation for nearly spherical point clouds.
    view_dir
        Unit vector from the scene origin toward the camera (default matches
        PyVista's isometric view: ``(1,1,1) / ||(1,1,1)||``).
    parallel_scale_margin
        Multiplier on the shared orthographic half-height after the largest mesh
        radius is chosen. Values **closer to 1** shrink the empty margin around
        every panel (meshes look larger and sit visually closer to neighbors).
        Typical range about ``1.0``–``1.1``; below ``1`` may clip the largest mesh.
    zoom
        Values ``> 1`` zoom in (smaller ``parallel_scale``, e.g. ``1.5`` is 1.5×).
        Combine with a lower ``parallel_scale_margin`` to reduce white space
        between panel contents.
    mesh_color
        If set, every mesh uses this color (overrides ``colors``).
    colors
        Optional color per mesh (named colors or hex). Cycles if shorter than
        ``meshes`` (ignored when ``mesh_color`` is set).
    window_size
        ``(width, height)`` in pixels for the whole render window before any
        ``screenshot_scale`` enlargement. **Increase** (e.g. ``(2400, 2400)``) for
        more pixels per panel at base scale.
    screenshot_scale
        Integer ≥ ``1``. If greater than ``1``, the render **window** is
        enlarged by this factor (``window_size × scale``) immediately before
        saving, then a full-size screenshot is taken. This avoids VTK's
        ``WindowToImageFilter`` scale path, which often leaves the PNG at the
        original window size on some off-screen / Windows setups. Omitted or
        ``1`` keeps ``window_size`` as-is.
    background
        Matplotlib-like color string for the render window background.
    off_screen
        If None, off-screen is used when neither ``show`` nor ``out_path`` is
        set defaults to False only when ``show`` is True; when saving, off-screen
        is used automatically on headless setups via PyVista.
    scale_bar_um
        If set (and ``out_path`` is set), draw a black horizontal scale bar and
        label in the bottom-right of the saved image. Mesh units are **pixels**;
        micrometers = pixels × ``pixels_to_um``.
    pixels_to_um
        Conversion from mesh pixel units to micrometers (default ``5/1000``).

    Returns
    -------
    pyvista.Plotter
        The plotter (already shown or screenshotted if requested).
    """
    n_rows, n_cols = int(grid_shape[0]), int(grid_shape[1])
    if n_rows < 1 or n_cols < 1:
        raise ValueError("grid_shape must be positive (n_rows, n_cols)")
    capacity = n_rows * n_cols
    if len(meshes) > capacity:
        raise ValueError(
            f"Need at least {len(meshes)} cells but grid_shape {grid_shape} has capacity {capacity}"
        )

    vdir = np.asarray(_DEFAULT_ISO_VIEW_DIR if view_dir is None else view_dir, dtype=float)
    vdir = vdir / np.linalg.norm(vdir)

    prepared: list[pv.PolyData] = []
    radii: list[float] = []
    for m in meshes:
        poly = _to_polydata(m)
        p, r = _prepare_mesh_polydata(
            poly,
            center=center_each,
            auto_rotate=auto_rotate_each,
            view_dir=vdir,
        )
        prepared.append(p)
        radii.append(r)

    shared_scale = max(radii) * float(parallel_scale_margin) if radii else 1.0
    zoom_f = float(zoom) if float(zoom) > 1e-12 else 1.0
    parallel_for_camera = shared_scale / zoom_f

    if window_size is None:
        base_w, base_h = 320, 320
        window_size = (n_cols * base_w, n_rows * base_h)

    if off_screen is None:
        off_screen = not show

    if show:
        off_screen = False

    plotter = pv.Plotter(
        shape=(n_rows, n_cols),
        off_screen=off_screen,
        window_size=window_size,
        border=False,
    )
    plotter.set_background(background)

    cmap = colors or ["#8ecae6", "#219ebc", "#023047", "#ffb703", "#fb8500", "#90a955"]
    idx = 0
    for r in range(n_rows):
        for c in range(n_cols):
            plotter.subplot(r, c)
            if idx < len(prepared):
                if mesh_color is not None:
                    color = mesh_color
                else:
                    color = cmap[idx % len(cmap)]
                plotter.add_mesh(
                    prepared[idx],
                    color=color,
                    show_edges=False,
                    smooth_shading=True,
                )
                _set_isometric_parallel_camera(plotter, parallel_for_camera)
            else:
                _set_isometric_parallel_camera(plotter, 1.0)
            idx += 1

    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        w0, h0 = int(window_size[0]), int(window_size[1])
        if screenshot_scale is not None:
            ss_used = max(1, int(round(float(screenshot_scale))))
        else:
            ss_used = 1

        # Enlarge the render window instead of vtkWindowToImageFilter.SetScale:
        # on several platforms (notably off-screen on Windows) the latter still
        # writes an image the size of the framebuffer, ignoring scale.
        if screenshot_scale is not None and ss_used > 1:
            plotter.window_size = (w0 * ss_used, h0 * ss_used)
            plotter.render()

        plotter.screenshot(out_path, return_img=False)

        if scale_bar_um is not None:
            ui_scale = ss_used if (screenshot_scale is not None and ss_used > 1) else 1
            _add_scale_bar_png_bottom_right(
                out_path,
                parallel_scale=parallel_for_camera,
                bar_length_um=float(scale_bar_um),
                pixels_to_um=float(pixels_to_um),
                grid_shape=(n_rows, n_cols),
                ui_scale=ui_scale,
            )

    if show:
        plotter.show()

    return plotter


def plot_simplified_processed_meshes_grid(
    *,
    processed_dir: str | Path | None = None,
    out_path: str | Path | None = None,
    grid_shape: tuple[int, int] = (3, 3),
    expected_count: int = 9,
    **kwargs: Any,
) -> pv.Plotter:
    """
    Plot every ``*_simplified.obj`` file in ``data/mesh/processed`` (sorted by name).

    Parameters
    ----------
    processed_dir
        Directory to glob. Default: ``<repo>/data/mesh/processed``.
    out_path
        Image path passed to :func:`plot_surface_mesh_grid`. Default:
        ``<repo>/figures/processed_simplified_grid.png``.
    grid_shape
        Subplot layout; default ``(3, 3)`` for nine meshes.
    expected_count
        If the number of matching files differs, raises ``ValueError``. Set to
        ``-1`` to disable the check.
    **kwargs
        Forwarded to :func:`plot_surface_mesh_grid` and override built-in defaults:
        ``mesh_color``, ``zoom``, ``parallel_scale_margin`` (lower ⇒ tighter
        framing), ``scale_bar_um``, ``pixels_to_um``, ``window_size``,
        ``screenshot_scale``, etc.

    Returns
    -------
    pyvista.Plotter
    """
    root = Path(processed_dir) if processed_dir is not None else _REPO_ROOT / "data" / "mesh" / "processed"
    paths = sorted(root.glob("*_simplified.obj"))
    n = len(paths)
    if expected_count >= 0 and n != expected_count:
        names = ", ".join(p.name for p in paths) or "(none)"
        raise ValueError(
            f"Expected {expected_count} *_simplified.obj files in {root}, found {n}: {names}"
        )
    capacity = int(grid_shape[0]) * int(grid_shape[1])
    if n > capacity:
        raise ValueError(
            f"{n} meshes do not fit in grid_shape {grid_shape} (capacity {capacity})"
        )

    if out_path is None:
        out_path = _REPO_ROOT / "figures" / "processed_simplified_grid.png"

    defaults: dict[str, Any] = {
        "mesh_color": _DEFAULT_SIMPLIFIED_MESH_COLOR,
        "zoom": 1.0,
        "parallel_scale_margin": 1.02,
        "scale_bar_um": 5.0,
        "pixels_to_um": 5.0 / 1000.0,
        "window_size": (800, 800),
    }
    merged: dict[str, Any] = {**defaults, **kwargs}

    return plot_surface_mesh_grid(
        [str(p) for p in paths],
        grid_shape=grid_shape,
        out_path=str(out_path),
        **merged,
    )


def _cli() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--simplified",
        action="store_true",
        help="Plot all *_simplified.obj in data/mesh/processed (3×3, nine files).",
    )
    p.add_argument(
        "--meshes",
        nargs="*",
        default=None,
        help="Mesh file paths (OBJ, PLY, etc.). Default: built-in trimesh primitives.",
    )
    p.add_argument("--rows", type=int, default=2)
    p.add_argument("--cols", type=int, default=3)
    p.add_argument("--out", default="mesh_grid.png", help="Output image path")
    p.add_argument(
        "--screenshot-scale",
        type=int,
        default=None,
        metavar="N",
        help="Multiply render window size before save (e.g. 2 → 2× pixel width/height).",
    )
    p.add_argument("--show", action="store_true", help="Open interactive window")
    p.add_argument("--no-center", action="store_true")
    p.add_argument("--no-rotate", action="store_true")
    args = p.parse_args()

    if args.simplified:
        out = args.out if args.out != "mesh_grid.png" else None
        plot_simplified_processed_meshes_grid(
            out_path=None if args.show else out,
            grid_shape=(3, 3),
            show=args.show,
            center_each=not args.no_center,
            auto_rotate_each=not args.no_rotate,
            screenshot_scale=args.screenshot_scale,
        )
        if not args.show:
            written = out or (_REPO_ROOT / "figures" / "processed_simplified_grid.png")
            print(f"Wrote {written}")
        return

    if args.meshes:
        mesh_list = args.meshes
    else:
        mesh_list = [
            trimesh.creation.cylinder(radius=0.25, height=1.2, sections=32),
            trimesh.creation.box([0.6, 0.45, 0.9]),
            trimesh.creation.icosphere(subdivisions=2, radius=0.55),
            trimesh.creation.cone(radius=0.4, height=1.0, sections=32),
            trimesh.creation.torus(major_radius=0.55, minor_radius=0.18),
            trimesh.creation.capsule(height=0.9, radius=0.22, count=[12, 12]),
        ]

    plot_surface_mesh_grid(
        mesh_list,
        grid_shape=(args.rows, args.cols),
        out_path=None if args.show else args.out,
        show=args.show,
        center_each=not args.no_center,
        auto_rotate_each=not args.no_rotate,
        screenshot_scale=args.screenshot_scale,
    )
    if not args.show:
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    _cli()
