from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Sequence, Union

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import trimesh
from matplotlib.collections import PolyCollection
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
            raise TypeError(
                f"Expected a single mesh file, got {type(loaded)} for {mesh}"
            )
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
    pts = _subsample_points(
        np.asarray(vertices_centered, dtype=float), max_points=20_000
    )
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


def _prepare_meshes_for_grid(
    meshes: Sequence[MeshLike],
    *,
    grid_shape: tuple[int, int],
    center_each: bool,
    auto_rotate_each: bool,
    view_dir: np.ndarray | None,
    parallel_scale_margin: float,
    zoom: float,
) -> tuple[list[pv.PolyData], np.ndarray, float]:
    n_rows, n_cols = int(grid_shape[0]), int(grid_shape[1])
    if n_rows < 1 or n_cols < 1:
        raise ValueError("grid_shape must be positive (n_rows, n_cols)")
    capacity = n_rows * n_cols
    if len(meshes) > capacity:
        raise ValueError(
            f"Need at least {len(meshes)} cells but grid_shape {grid_shape} has capacity {capacity}"
        )

    vdir = np.asarray(
        _DEFAULT_ISO_VIEW_DIR if view_dir is None else view_dir, dtype=float
    )
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
    return prepared, vdir, parallel_for_camera


def _tri_faces(poly: pv.PolyData) -> np.ndarray:
    tri = poly.triangulate()
    faces = np.asarray(tri.faces, dtype=np.int64)
    if faces.size == 0:
        return np.empty((0, 3), dtype=np.int64)
    if faces.size % 4 != 0:
        raise ValueError("Unexpected triangulated face layout")
    ff = faces.reshape(-1, 4)
    if not np.all(ff[:, 0] == 3):
        raise ValueError("Expected triangular faces after triangulate()")
    return ff[:, 1:4]


def _draw_vector_mesh_on_axis(
    ax: Any,
    poly: pv.PolyData,
    *,
    color: str,
    view_dir: np.ndarray,
    parallel_scale: float,
) -> None:
    pts = np.asarray(poly.points, dtype=float)
    if pts.size == 0:
        return
    faces = _tri_faces(poly)
    if len(faces) == 0:
        return

    # Same screen basis as camera setup: y=screen_up, x=horizontal, z=toward_camera.
    frame = _isometric_target_frame(view_dir)
    screen_up = frame[:, 0]
    horizontal = frame[:, 1]
    toward_camera = frame[:, 2]

    x = pts @ horizontal
    y = pts @ screen_up
    z = pts @ toward_camera

    # Per-face Lambert-style lighting for vector shading.
    p0 = pts[faces[:, 0]]
    p1 = pts[faces[:, 1]]
    p2 = pts[faces[:, 2]]
    fn = np.cross(p1 - p0, p2 - p0)
    fn_norm = np.linalg.norm(fn, axis=1, keepdims=True)
    fn_norm[fn_norm < 1e-12] = 1.0
    fn = fn / fn_norm

    light_dir = toward_camera + 0.45 * screen_up - 0.25 * horizontal
    light_dir = light_dir / np.linalg.norm(light_dir)
    lambert = np.clip(fn @ light_dir, 0.0, 1.0)
    ambient = 0.35
    diffuse = 0.65
    intensity = ambient + diffuse * lambert

    base_rgb = np.asarray(mcolors.to_rgb(color), dtype=float)
    shaded = np.clip(base_rgb[None, :] * intensity[:, None], 0.0, 1.0)

    depth = np.mean(z[faces], axis=1)
    order = np.argsort(depth)  # far-to-near painter's algorithm
    poly_paths = [np.column_stack([x[f], y[f]]) for f in faces[order]]
    face_colors = shaded[order]

    coll = PolyCollection(
        poly_paths,
        facecolors=face_colors,
        edgecolors="none",
        linewidths=0.0,
        antialiaseds=False,
    )
    ax.add_collection(coll)

    ax.set_aspect("equal", adjustable="box")
    ax.set_ylim(-parallel_scale, parallel_scale)
    xhalf = parallel_scale * float(ax.bbox.width / max(1.0, ax.bbox.height))
    ax.set_xlim(-xhalf, xhalf)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _save_mesh_grid_svg_vector(
    prepared: Sequence[pv.PolyData],
    *,
    grid_shape: tuple[int, int],
    out_path: str | Path,
    parallel_scale: float,
    view_dir: np.ndarray,
    mesh_color: str | None,
    colors: Sequence[str] | None,
    window_size: tuple[int, int] | None,
    background: str,
) -> Path:
    out = Path(out_path)
    if out.suffix.lower() != ".svg":
        raise ValueError(f"SVG output path must end with .svg, got: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)

    n_rows, n_cols = int(grid_shape[0]), int(grid_shape[1])
    if window_size is None:
        base_w, base_h = 320, 320
        window_size = (n_cols * base_w, n_rows * base_h)

    dpi = 100
    fig_w = float(window_size[0]) / dpi
    fig_h = float(window_size[1]) / dpi
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), dpi=dpi)
    fig.patch.set_facecolor(background)

    if n_rows == 1 and n_cols == 1:
        ax_list = [axes]
    elif n_rows == 1:
        ax_list = list(axes)
    elif n_cols == 1:
        ax_list = list(axes)
    else:
        ax_list = [ax for row in axes for ax in row]

    cmap = colors or ["#8ecae6", "#219ebc", "#023047", "#ffb703", "#fb8500", "#90a955"]
    for idx, ax in enumerate(ax_list):
        ax.set_facecolor(background)
        if idx < len(prepared):
            color = mesh_color if mesh_color is not None else cmap[idx % len(cmap)]
            _draw_vector_mesh_on_axis(
                ax,
                prepared[idx],
                color=color,
                view_dir=view_dir,
                parallel_scale=parallel_scale,
            )
        else:
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    fig.savefig(out, format="svg", facecolor=background)
    plt.close(fig)
    return out


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
    prepared, vdir, parallel_for_camera = _prepare_meshes_for_grid(
        meshes,
        grid_shape=grid_shape,
        center_each=center_each,
        auto_rotate_each=auto_rotate_each,
        view_dir=view_dir,
        parallel_scale_margin=parallel_scale_margin,
        zoom=zoom,
    )

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


def save_surface_meshes_svg(
    meshes: Sequence[MeshLike],
    *,
    out_dir: str | Path,
    file_stems: Sequence[str] | None = None,
    file_prefix: str = "mesh",
    start_index: int = 0,
    center_each: bool = True,
    auto_rotate_each: bool = True,
    view_dir: np.ndarray | None = None,
    parallel_scale_margin: float = 1.08,
    zoom: float = 1.0,
    mesh_color: str | None = None,
    colors: Sequence[str] | None = None,
    window_size: tuple[int, int] | None = None,
    background: str = "white",
    off_screen: bool = True,
) -> list[Path]:
    """Save one true-vector SVG file per mesh using the grid camera and framing style.

    Parameters
    ----------
    meshes : sequence of MeshLike
        Meshes to render, one SVG file each.
    out_dir : str or Path
        Output directory (created if it does not exist).
    file_stems : sequence of str or None
        Filenames without ``.svg`` extension, one per mesh. If ``None``,
        names are generated as ``{file_prefix}_{index:03d}.svg``.
    file_prefix : str, default "mesh"
        Prefix used for auto-generated filenames.
    start_index : int, default 0
        Starting index for auto-generated filenames.
    **kwargs
        Remaining keyword arguments (``center_each``, ``auto_rotate_each``,
        ``view_dir``, ``parallel_scale_margin``, ``zoom``, ``mesh_color``,
        ``colors``, ``window_size``, ``background``, ``off_screen``) are
        passed through to :func:`plot_surface_mesh_grid` unchanged.

    Returns
    -------
    list[pathlib.Path]
        Paths to each written SVG file, in input order.

    Raises
    ------
    ValueError
        If ``file_stems`` length does not match ``len(meshes)`` or contains
        an empty string.
    """
    if file_stems is not None and len(file_stems) != len(meshes):
        raise ValueError("file_stems must match len(meshes) when provided")

    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for idx, mesh in enumerate(meshes):
        if file_stems is None:
            stem = f"{file_prefix}_{start_index + idx:03d}"
        else:
            stem = str(file_stems[idx]).strip()
            if not stem:
                raise ValueError("file_stems entries must be non-empty")
        out_path = out_root / f"{stem}.svg"
        prepared, vdir, parallel_for_camera = _prepare_meshes_for_grid(
            [mesh],
            grid_shape=(1, 1),
            center_each=center_each,
            auto_rotate_each=auto_rotate_each,
            view_dir=view_dir,
            parallel_scale_margin=parallel_scale_margin,
            zoom=zoom,
        )
        written_path = _save_mesh_grid_svg_vector(
            prepared,
            grid_shape=(1, 1),
            out_path=out_path,
            parallel_scale=parallel_for_camera,
            view_dir=vdir,
            mesh_color=mesh_color,
            colors=colors,
            window_size=window_size,
            background=background,
        )
        written.append(written_path)

    return written


def save_surface_mesh_grid_svg(
    meshes: Sequence[MeshLike],
    grid_shape: tuple[int, int],
    *,
    out_path: str | Path,
    save_individual_meshes: bool = False,
    individual_out_dir: str | Path | None = None,
    individual_file_stems: Sequence[str] | None = None,
    individual_file_prefix: str = "mesh",
    individual_start_index: int = 0,
    center_each: bool = True,
    auto_rotate_each: bool = True,
    view_dir: np.ndarray | None = None,
    parallel_scale_margin: float = 1.08,
    zoom: float = 1.0,
    mesh_color: str | None = None,
    colors: Sequence[str] | None = None,
    window_size: tuple[int, int] | None = None,
    background: str = "white",
    off_screen: bool = True,
) -> tuple[Path, list[Path]]:
    """
    Save a mesh-grid figure as a true-vector SVG, with optional per-mesh SVGs.

    Parameters
    ----------
    meshes : sequence of MeshLike
        Meshes to render.
    grid_shape : tuple[int, int]
        ``(n_rows, n_cols)`` grid layout.
    out_path : str or Path
        Destination for the composite grid SVG.
    save_individual_meshes : bool, default False
        If ``True``, also save one SVG per mesh.
    individual_out_dir : str or Path or None
        Directory for individual SVGs. Defaults to the same directory as
        *out_path* when ``None``.
    individual_file_stems : sequence of str or None
        Filenames (without ``.svg``) for each individual mesh.
    individual_file_prefix : str, default "mesh"
        Prefix used when *individual_file_stems* is ``None``.
    individual_start_index : int, default 0
        Starting index for auto-generated filenames.
    **kwargs
        Remaining keyword arguments (``center_each``, ``auto_rotate_each``,
        ``view_dir``, ``parallel_scale_margin``, ``zoom``, ``mesh_color``,
        ``colors``, ``window_size``, ``background``, ``off_screen``) are
        passed through to :func:`plot_surface_mesh_grid` unchanged.

    Returns
    -------
    tuple[pathlib.Path, list[pathlib.Path]]
        ``(grid_svg_path, individual_svg_paths)`` — the second list is
        empty unless *save_individual_meshes* is ``True``.
    """
    prepared, vdir, parallel_for_camera = _prepare_meshes_for_grid(
        meshes,
        grid_shape=grid_shape,
        center_each=center_each,
        auto_rotate_each=auto_rotate_each,
        view_dir=view_dir,
        parallel_scale_margin=parallel_scale_margin,
        zoom=zoom,
    )
    grid_svg = _save_mesh_grid_svg_vector(
        prepared,
        grid_shape=grid_shape,
        out_path=out_path,
        parallel_scale=parallel_for_camera,
        view_dir=vdir,
        mesh_color=mesh_color,
        colors=colors,
        window_size=window_size,
        background=background,
    )

    individual: list[Path] = []
    if save_individual_meshes:
        save_dir = (
            Path(individual_out_dir)
            if individual_out_dir is not None
            else grid_svg.parent
        )
        individual = save_surface_meshes_svg(
            meshes,
            out_dir=save_dir,
            file_stems=individual_file_stems,
            file_prefix=individual_file_prefix,
            start_index=individual_start_index,
            center_each=center_each,
            auto_rotate_each=auto_rotate_each,
            view_dir=view_dir,
            parallel_scale_margin=parallel_scale_margin,
            zoom=zoom,
            mesh_color=mesh_color,
            colors=colors,
            window_size=window_size,
            background=background,
            off_screen=off_screen,
        )

    return grid_svg, individual
