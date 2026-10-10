# %% [markdown]
# # TS1 cycle lost at resample
#
# The TS1 skeleton (`TS1_qst0.5_mcst5`) has cyclomatic number 8. After
# `MorphologyGraph.resample` at `max_edge_length = 2 ×` the thickness median,
# one cycle is gone.
#
# That cycle is not a separate loop. Two unbranching sections share the same
# junctions: a straight edge, and a two-edge bow whose middle sample sits
# about one length unit off that edge. Both sections are much shorter than
# the edge bound, so resample replaces each with the same chord. The morphology
# graph keeps only one of those edges.
#
# The plot draws the mesh and skeleton, highlights both sections, and marks
# the bow sample.

# %%
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from mascaf import MeshManager, MorphologyGraph, SkeletonGraph, compute_fit_features, visualize_mesh_3d
from mascaf.morphology_graph import _extract_unbranching_sections

# %%
def repo_root() -> Path:
    here = Path.cwd().resolve()
    for candidate in (here, *here.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "mascaf").is_dir():
            return candidate
    raise FileNotFoundError("Could not find the mascaf repo from the current directory.")


ROOT = repo_root()
MESH_PATH = ROOT / "data" / "mesh" / "processed" / "TS1.obj"
SKELETON_PATH = ROOT / "data" / "mcf_skeletons" / "TS1_qst0.5_mcst5.polylines.txt"
MEL_OVER_THICKNESS = 2.0

# %% [markdown]
# ## Load the skeleton and locate the doubled section

# %%
mesh = MeshManager(mesh_path=str(MESH_PATH))
skeleton = SkeletonGraph.from_txt(str(SKELETON_PATH))
features = compute_fit_features(mesh, skeleton)
thickness = float(features.thickness.median)
mel = MEL_OVER_THICKNESS * thickness
basis = MorphologyGraph.from_skeleton_graph(skeleton)

print(
    f"skeleton cyclomatic={skeleton.cyclomatic_number()}, "
    f"resampled cyclomatic={basis.resample(mel).cyclomatic_number()}, "
    f"mel={mel:.4g}"
)


def _polyline(graph: MorphologyGraph, nodes: list[int]) -> np.ndarray:
    return np.array([graph.get_node_position(node) for node in nodes], dtype=float)


def _length(poly: np.ndarray) -> float:
    if poly.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(poly[1:] - poly[:-1], axis=1).sum())


def _point_to_polyline(point: np.ndarray, poly: np.ndarray) -> float:
    best = np.inf
    for start, end in zip(poly[:-1], poly[1:]):
        segment = end - start
        length_sq = float(np.dot(segment, segment))
        if length_sq == 0.0:
            distance = float(np.linalg.norm(point - start))
        else:
            t = float(np.clip(np.dot(point - start, segment) / length_sq, 0.0, 1.0))
            distance = float(np.linalg.norm(point - (start + t * segment)))
        best = min(best, distance)
    return best


sections = [
    section
    for section in _extract_unbranching_sections(basis)
    if section["kind"] == "path"
]
by_ends: dict[tuple[int, int], list[dict]] = {}
for section in sections:
    nodes = section["nodes"]
    by_ends.setdefault((int(nodes[0]), int(nodes[-1])), []).append(section)

doubled = []
for ends, group in by_ends.items():
    if len(group) < 2:
        continue
    polys = [_polyline(basis, section["nodes"]) for section in group]
    if all(_length(poly) <= mel for poly in polys):
        doubled.append((ends, group, polys))

if len(doubled) != 1:
    raise RuntimeError(f"Expected one resample-collapsed pair, found {len(doubled)}")

(start_id, end_id), pair_sections, pair_polys = doubled[0]
order = np.argsort([_length(poly) for poly in pair_polys])
straight = pair_polys[int(order[0])]
bowed = pair_polys[int(order[-1])]
bow_nodes = pair_sections[int(order[-1])]["nodes"]
interior = bow_nodes[1:-1]
if len(interior) != 1:
    raise RuntimeError(f"Expected one interior sample on the bow, found {interior}")
marker = basis.get_node_position(interior[0])
offset = _point_to_polyline(marker, straight)

print(
    f"junctions {start_id} and {end_id}: "
    f"straight length={_length(straight):.4g}, "
    f"bow length={_length(bowed):.4g}, "
    f"bow offset={offset:.4g} (thickness median={thickness:.4g})"
)

# %% [markdown]
# ## Mesh, skeleton, and the collapsed cycle
#
# Crimson is the skeleton. Gold is the straight section and the bow. The
# marker is the extra sample that makes the two routes a cycle.

# %%
fig = visualize_mesh_3d(
    mesh,
    skel=skeleton,
    title="TS1: cycle collapsed by resample",
    show_axes=False,
    width=900,
    height=700,
)
for poly, name in ((straight, "Straight section"), (bowed, "Bowed section")):
    fig.add_trace(
        go.Scatter3d(
            x=poly[:, 0],
            y=poly[:, 1],
            z=poly[:, 2],
            mode="lines",
            line=dict(color="gold", width=8),
            name=name,
        )
    )
fig.add_trace(
    go.Scatter3d(
        x=[marker[0]],
        y=[marker[1]],
        z=[marker[2]],
        mode="markers",
        marker=dict(size=8, color="gold", symbol="diamond"),
        name="Collapsed cycle",
        hovertemplate=(
            "bow sample<br>"
            f"offset {offset:.3g}<br>"
            "x=%{x:.4g}<br>y=%{y:.4g}<br>z=%{z:.4g}<extra></extra>"
        ),
    )
)
fig

# %%
