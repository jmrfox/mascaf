# %%
from mascaf import plot_surface_mesh_grid
from pathlib import Path
from typing import Any, Union
import pyvista as pv
import trimesh
import numpy as np

MeshLike = Union[str, trimesh.Trimesh, pv.PolyData]

MESH_COLOR = "#5d7a99"


# %%
# plotting simplified surface meshes

mesh_filenames = [ "TS1_simplified.obj",
"TS2_simplified.obj",
"TS3_simplified.obj",
"TS4_simplified.obj",
"TS21_simplified.obj",
"TS24_simplified.obj",
"TS48_simplified.obj",
"TS67_simplified.obj",
"TS76_simplified.obj",
]

mesh_dir = Path("C:/Users/MainUser/Documents/Repos/mascaf/data/mesh/processed")
mesh_paths = [mesh_dir / f for f in mesh_filenames]
out_path = Path("C:/Users/MainUser/Documents/Repos/mascaf/viz/meshes_figure_1.png")

mesh_list = [trimesh.load(p) for p in mesh_paths]

plotter = plot_surface_mesh_grid(
    mesh_list,
    grid_shape=(3,3),
    out_path=out_path,
    colors=[MESH_COLOR] * len(mesh_list),
)
plotter.show()


# %%
from mascaf import save_surface_mesh_grid_svg, save_surface_meshes_svg

# Or just individual mesh SVGs
written = save_surface_meshes_svg(
    mesh_list,
    out_dir="../viz/mesh_svgs",
    file_stems= [ f.removesuffix(".obj").replace("_simplified", "") for f in mesh_filenames ]
)
