from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import shapely.geometry as sgeom
import trimesh

from .mesh import MeshManager
from .morphology_graph import MorphologyGraph
from .skeleton import SkeletonGraph

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@dataclass
class FitOptions:
    """
    Configuration options for cable fitting and local radius estimation.

    Attributes:
        max_edge_length: Maximum edge length along unbranching sections in mesh
            units. Resampling keeps section endpoints fixed and introduces
            interior samples so no segment exceeds this value.
        radius_strategy: Strategy for estimating node radii. One of:
            - "equivalent_area" (default): r = sqrt(A/pi) using
              cross-section area.
            - "equivalent_perimeter": r = L/(2*pi) using exterior
              boundary length.
            - "section_median": median ray-to-boundary distance in the local
              section plane from the sample origin.
            - "section_circle_fit": algebraic circle fit (Kasa) to the section
              boundary.
            - "nearest_surface": distance from the sample point to nearest mesh
              surface.
        section_probe_eps: Step size (scaled by mesh bbox) for offsetting
            the section plane origin along the local normal when the exact
            plane yields no curves.
        section_probe_tries: Number of +/- k*eps offsets to try when seeking a
            section.
        multi_tangent_reduction: Reduction applied to the per-edge radii
            computed at each node. One of "mean", "min", "max", or
            "median".
    """

    max_edge_length: float = 1.0
    radius_strategy: str = "equivalent_area"
    section_probe_eps: float = 1e-4
    section_probe_tries: int = 3
    multi_tangent_reduction: str = "median"


class CableFitter:
    """Fit a cable-style MorphologyGraph from a mesh and SkeletonGraph."""

    def __init__(self, options: Optional[FitOptions] = None):
        """Store fitter configuration, using defaults when none are provided."""
        self.options = options or FitOptions()

    def fit(
        self,
        mesh: Union[trimesh.Trimesh, MeshManager],
        skeleton: SkeletonGraph,
    ) -> MorphologyGraph:
        """Build a MorphologyGraph from the input skeleton and mesh."""
        mesh_obj = _resolve_mesh(mesh)
        if len(mesh_obj.vertices) == 0:
            raise ValueError("Mesh is empty or not provided")
        if not isinstance(skeleton, SkeletonGraph):
            raise TypeError("skeleton must be a SkeletonGraph instance")
        if skeleton.number_of_nodes() == 0:
            logger.info("Cable fitting skipped because skeleton is empty")
            return MorphologyGraph()

        logger.info(
            "Starting cable fit with %d skeleton nodes, %d skeleton edges, "
            "%d mesh vertices, and max_edge_length=%s",
            skeleton.number_of_nodes(),
            skeleton.number_of_edges(),
            len(mesh_obj.vertices),
            self.options.max_edge_length,
        )

        morphology = _build_morphology_graph_from_skeleton(
            skeleton,
            float(self.options.max_edge_length),
        )
        _compute_morphology_node_radii(morphology, mesh_obj, self.options)
        logger.info(
            "Finished cable fit with %d morphology nodes and %d morphology " "edges",
            morphology.number_of_nodes(),
            morphology.number_of_edges(),
        )
        return morphology


def _resolve_mesh(
    mesh: Union[trimesh.Trimesh, MeshManager],
) -> trimesh.Trimesh:
    """Return a raw trimesh object from either supported mesh wrapper."""
    if not isinstance(mesh, (trimesh.Trimesh, MeshManager)):
        raise TypeError("mesh must be a trimesh.Trimesh or MeshManager")
    return mesh.mesh if isinstance(mesh, MeshManager) else mesh


def _build_morphology_graph_from_skeleton(
    skeleton: SkeletonGraph,
    max_edge_length: float,
) -> MorphologyGraph:
    """Construct morphology nodes and edges from unbranching skeleton sections."""
    graph = MorphologyGraph()
    node_map: Dict[Tuple[str, int], int] = {}
    next_id = 0

    def alloc_id() -> int:
        """Allocate the next morphology node identifier."""
        nonlocal next_id
        nid = next_id
        next_id += 1
        return nid

    def ensure_fixed_node(skel_node: int) -> int:
        """Create or return the shared morphology node for a fixed skeleton node."""
        key = ("fixed", int(skel_node))
        if key not in node_map:
            nid = alloc_id()
            # Junctions and terminals are carried over exactly so topology is
            # inherited directly from the skeleton graph.
            graph.add_node(nid, xyz=skeleton.get_node_position(skel_node))
            node_map[key] = nid
        return node_map[key]

    sections = _extract_unbranching_sections(skeleton)
    logger.debug(
        "Building morphology graph from %d unbranching sections",
        len(sections),
    )
    for section in sections:
        section_nodes = section["nodes"]
        # Each section is handled independently: keep endpoints fixed, then
        # resample only the interior to satisfy the edge length bound.
        polyline = np.array(
            [skeleton.get_node_position(node) for node in section_nodes],
            dtype=float,
        )
        samples = _resample_polyline(polyline, max_edge_length)
        if samples.shape[0] == 0:
            logger.debug("Skipping empty resampled section: %s", section)
            continue

        if section["kind"] == "cycle":
            anchor = ensure_fixed_node(section_nodes[0])
            sequence = [anchor]
            interior = samples[1:-1] if samples.shape[0] >= 2 else np.zeros((0, 3))
            for point in interior:
                nid = alloc_id()
                graph.add_node(nid, xyz=np.asarray(point, dtype=float))
                sequence.append(nid)
            sequence.append(anchor)
        else:
            start = ensure_fixed_node(section_nodes[0])
            end = ensure_fixed_node(section_nodes[-1])
            sequence = [start]
            interior = samples[1:-1] if samples.shape[0] >= 2 else np.zeros((0, 3))
            for point in interior:
                nid = alloc_id()
                graph.add_node(nid, xyz=np.asarray(point, dtype=float))
                sequence.append(nid)
            if start != end or len(sequence) == 1:
                sequence.append(end)

        logger.debug(
            "Section kind=%s skeleton_nodes=%d resampled_points=%d graph_nodes=%d",
            section["kind"],
            len(section_nodes),
            samples.shape[0],
            len(sequence),
        )

        for u, v in zip(sequence[:-1], sequence[1:]):
            if u != v:
                graph.add_edge(u, v)

    return graph


def _extract_unbranching_sections(skeleton: SkeletonGraph) -> List[dict]:
    """Split the skeleton into maximal non-branching paths and cycles."""
    if skeleton.number_of_edges() == 0:
        return []

    # Nodes with degree != 2 mark section boundaries. Everything between them is
    # an unbranching cable segment.
    critical = {node for node in skeleton.nodes() if skeleton.degree(node) != 2}
    visited_edges: set[frozenset[int]] = set()
    sections: List[dict] = []

    for start in sorted(critical):
        for neighbor in sorted(skeleton.neighbors(start)):
            edge_key = frozenset((start, neighbor))
            if edge_key in visited_edges:
                continue
            path = [start]
            prev = start
            current = neighbor
            visited_edges.add(edge_key)
            path.append(current)

            while current not in critical:
                nbrs = [n for n in skeleton.neighbors(current) if n != prev]
                if not nbrs:
                    break
                nxt = nbrs[0]
                edge_key = frozenset((current, nxt))
                if edge_key in visited_edges:
                    break
                visited_edges.add(edge_key)
                prev, current = current, nxt
                path.append(current)

            sections.append({"kind": "path", "nodes": path})
            logger.debug("Extracted path section with %d nodes", len(path))

    for u, v in skeleton.edges():
        edge_key = frozenset((u, v))
        if edge_key in visited_edges:
            continue
        cycle_nodes = _trace_cycle_section(skeleton, u, v, visited_edges)
        if len(cycle_nodes) >= 2:
            sections.append({"kind": "cycle", "nodes": cycle_nodes})
            logger.debug(
                "Extracted cycle section with %d nodes",
                len(cycle_nodes),
            )

    logger.debug(
        "Extracted %d sections from skeleton (%d critical nodes)",
        len(sections),
        len(critical),
    )

    return sections


def _trace_cycle_section(
    skeleton: SkeletonGraph,
    start: int,
    neighbor: int,
    visited_edges: set[frozenset[int]],
) -> List[int]:
    """Follow one unvisited degree-2 loop and return its cyclic node sequence."""
    path = [start, neighbor]
    visited_edges.add(frozenset((start, neighbor)))
    prev = start
    current = neighbor

    while True:
        nbrs = [n for n in skeleton.neighbors(current) if n != prev]
        if not nbrs:
            break
        nxt = nbrs[0]
        edge_key = frozenset((current, nxt))
        if nxt == start:
            visited_edges.add(edge_key)
            path.append(start)
            break
        if edge_key in visited_edges:
            break
        visited_edges.add(edge_key)
        path.append(nxt)
        prev, current = current, nxt

    return path


def _compute_node_tangents(
    graph: Union[SkeletonGraph, MorphologyGraph],
    node: int,
) -> List[np.ndarray]:
    """Return unit tangent directions from a node toward each neighbor."""
    tangents: List[np.ndarray] = []
    origin = _get_graph_node_position(graph, node)
    for neighbor in graph.neighbors(node):
        vec = _get_graph_node_position(graph, neighbor) - origin
        norm = float(np.linalg.norm(vec))
        if norm > 1e-12 and np.isfinite(norm):
            tangents.append(vec / norm)
    return tangents


def _reduce_multi_radii(radii: Sequence[float], reduction: str) -> float:
    """Reduce multiple per-edge radius estimates to one scalar value."""
    values = [float(r) for r in radii if np.isfinite(r)]
    if not values:
        return 0.0
    if reduction == "mean":
        return float(np.mean(values))
    if reduction == "min":
        return float(np.min(values))
    if reduction == "max":
        return float(np.max(values))
    if reduction == "median":
        return float(np.median(values))
    raise ValueError(f"Unknown reduction method: {reduction}")


def _compute_skeleton_node_radii(
    skeleton: SkeletonGraph,
    mesh: Union[trimesh.Trimesh, MeshManager],
    options: FitOptions,
) -> Dict[int, float]:
    """Estimate radii directly on skeleton nodes using local edge tangents."""
    mesh_obj = _resolve_mesh(mesh)
    bbox_size = _mesh_bbox_size(mesh_obj)
    eps = max(1e-12, float(options.section_probe_eps) * bbox_size)
    radii: Dict[int, float] = {}
    for node in skeleton.nodes():
        point = skeleton.get_node_position(node)
        tangents = _compute_node_tangents(skeleton, node)
        per_edge_radii = [
            _compute_radius_for_tangent(
                point=point,
                tangent=tangent,
                mesh=mesh_obj,
                radius_strategy=options.radius_strategy,
                eps=eps,
                max_tries=int(options.section_probe_tries),
            )[0]
            for tangent in tangents
        ]
        radii[int(node)] = _reduce_multi_radii(
            per_edge_radii,
            options.multi_tangent_reduction,
        )
    return radii


def _compute_morphology_node_radii(
    graph: MorphologyGraph,
    mesh: trimesh.Trimesh,
    options: FitOptions,
) -> None:
    """Populate radius-related node attributes on the morphology graph."""
    bbox_size = _mesh_bbox_size(mesh)
    eps = max(1e-12, float(options.section_probe_eps) * bbox_size)

    for node in graph.nodes():
        point = _get_graph_node_position(graph, node)
        tangents = _compute_node_tangents(graph, node)
        # At junctions we compute one radius estimate per incident edge
        # direction, then reduce them to a single node radius.
        per_edge_radii = []
        strategies = []
        for tangent in tangents:
            radius, strategy = _compute_radius_for_tangent(
                point=point,
                tangent=tangent,
                mesh=mesh,
                radius_strategy=options.radius_strategy,
                eps=eps,
                max_tries=int(options.section_probe_tries),
            )
            per_edge_radii.append(radius)
            strategies.append(strategy)

        graph.nodes[node]["radius"] = _reduce_multi_radii(
            per_edge_radii,
            options.multi_tangent_reduction,
        )
        graph.nodes[node]["radius_strategy"] = options.multi_tangent_reduction
        graph.nodes[node]["radius_sources"] = strategies
        logger.debug(
            "Node %s radius=%s from %d tangents via %s",
            node,
            graph.nodes[node]["radius"],
            len(per_edge_radii),
            options.multi_tangent_reduction,
        )


def _get_graph_node_position(
    graph: Union[SkeletonGraph, MorphologyGraph],
    node: int,
) -> np.ndarray:
    """Return a node position regardless of whether it uses `pos` or `xyz`."""
    data = graph.nodes[node]
    if "pos" in data:
        return np.asarray(data["pos"], dtype=float)
    return np.asarray(data["xyz"], dtype=float)


def _mesh_bbox_size(mesh: trimesh.Trimesh) -> float:
    """Return a characteristic mesh size based on its bounding-box diagonal."""
    vertices = np.asarray(mesh.vertices, dtype=float)
    if vertices.size == 0:
        return 1.0
    return float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0))) or 1.0


def _resample_polyline(pl: np.ndarray, max_edge_length: float) -> np.ndarray:
    """Resample a polyline so that adjacent output samples obey the length bound."""
    P = np.asarray(pl, dtype=float)
    if P.ndim != 2 or P.shape[1] != 3 or P.shape[0] == 0:
        return np.zeros((0, 3), dtype=float)
    if P.shape[0] == 1:
        return P.copy()

    seg = np.linalg.norm(P[1:] - P[:-1], axis=1)
    L = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(L[-1])
    if total <= 0.0:
        return P[[0], :].copy()

    n_segments = max(1, int(np.ceil(total / max_edge_length)))
    targets = np.linspace(0.0, total, n_segments + 1)

    out: List[np.ndarray] = []
    si = 0
    for t in targets:
        while si < len(seg) and L[si + 1] < t:
            si += 1
        if si >= len(seg):
            out.append(P[-1])
            continue
        t0 = L[si]
        t1 = L[si + 1]
        if t1 <= t0:
            out.append(P[si])
            continue
        alpha = (t - t0) / (t1 - t0)
        out.append((1.0 - alpha) * P[si] + alpha * P[si + 1])

    return np.vstack(out) if out else P[[0], :].copy()


def _compute_radius_for_tangent(
    point: np.ndarray,
    tangent: np.ndarray,
    mesh: trimesh.Trimesh,
    radius_strategy: str,
    eps: float,
    max_tries: int,
) -> Tuple[float, str]:
    """Estimate one radius at a point using a single tangent direction."""
    n = np.asarray(tangent, dtype=float)
    norm = float(np.linalg.norm(n))
    if norm <= 1e-12 or not np.isfinite(norm):
        logger.debug(
            "Skipping invalid tangent at point %s",
            point,
        )
        return 0.0, "invalid_tangent"
    n = n / norm

    if radius_strategy == "nearest_surface":
        return _nearest_surface_distance(point, mesh), "nearest_surface"

    # Slice the mesh with a plane normal to the local cable direction.
    # Then derive a radius from the resulting cross-section polygon.
    poly2d = _cross_section_polygon_near_point(
        mesh=mesh,
        origin=point,
        normal=n,
        eps=eps,
        max_tries=max_tries,
    )
    if poly2d is None:
        logger.debug(
            "Cross-section extraction failed at point %s; "
            "using nearest-surface fallback",
            point,
        )
        return (
            _nearest_surface_distance(point, mesh),
            "nearest_surface_fallback",
        )

    area = float(poly2d.area)
    if radius_strategy == "equivalent_perimeter":
        perim = float(poly2d.exterior.length)
        return (
            perim / (2.0 * math.pi) if perim > 0 else 0.0,
            "equivalent_perimeter",
        )
    if radius_strategy == "section_median":
        return _radius_from_section_median(poly2d), "section_median"
    if radius_strategy == "section_circle_fit":
        r_fit = _radius_from_section_circle_fit(poly2d)
        if not np.isfinite(r_fit) or r_fit <= 0:
            radius = math.sqrt(area / math.pi) if area > 0 else 0.0
            return radius, "equivalent_area_fallback"
        return float(r_fit), "section_circle_fit"

    radius = math.sqrt(area / math.pi) if area > 0 else 0.0
    return radius, "equivalent_area"


def _plane_basis(
    normal: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build an orthonormal basis whose third axis is aligned to `normal`."""
    n = np.asarray(normal, dtype=float)
    n = n / (np.linalg.norm(n) + 1e-12)
    # Pick a stable auxiliary axis that is not nearly parallel to the plane
    # normal, then build an orthonormal basis for the section plane.
    ax = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n, ax)
    u /= np.linalg.norm(u) + 1e-12
    v = np.cross(n, u)
    v /= np.linalg.norm(v) + 1e-12
    return u, v, n


def _world_to_local_plane(P: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Return a transform from world coordinates into the local section plane."""
    u, v, n = _plane_basis(normal)
    R = np.column_stack([u, v, n])
    M = np.eye(4, dtype=float)
    M[:3, :3] = R.T
    M[:3, 3] = -R.T @ np.asarray(P, dtype=float)
    return M


def _compose_polygons_with_holes(
    polys: List[sgeom.Polygon],
) -> List[sgeom.Polygon]:
    """Rebuild nested 2D polygons so immediate children become holes."""
    if not polys:
        return []
    n = len(polys)
    contains = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            try:
                contains[i][j] = polys[i].contains(polys[j])
            except Exception:
                contains[i][j] = False

    def depth_of(idx: int) -> int:
        """Count how many other polygons contain the indexed polygon."""
        return sum(1 for k in range(n) if contains[k][idx])

    depths = [depth_of(i) for i in range(n)]
    result: List[sgeom.Polygon] = []
    for i in range(n):
        if depths[i] % 2 != 0:
            continue
        holes_coords: List[List[Tuple[float, float]]] = []
        for j in range(n):
            if i == j:
                continue
            if contains[i][j] and depths[j] == depths[i] + 1:
                ring = list(polys[j].exterior.coords)
                holes_coords.append([(float(x), float(y)) for x, y in ring])
        try:
            composed = sgeom.Polygon(
                polys[i].exterior.coords,
                holes=holes_coords,
            )
            if composed.is_valid and composed.area > 0:
                result.append(composed)
        except Exception:
            if polys[i].is_valid and polys[i].area > 0:
                result.append(polys[i])
    return result


def _cross_section_polygon_near_point(
    mesh: trimesh.Trimesh,
    origin: np.ndarray,
    normal: np.ndarray,
    *,
    eps: float,
    max_tries: int,
) -> Optional[sgeom.Polygon]:
    """Find a nearby mesh cross-section polygon in the plane normal direction."""
    P = np.asarray(origin, dtype=float)
    n = np.asarray(normal, dtype=float)
    n = n / (np.linalg.norm(n) + 1e-12)

    # If the exact plane misses due to numerical issues, probe nearby offsets
    # along the normal before giving up.
    for scale in (1.0, 2.0, 4.0):
        eps_s = float(eps) * scale
        offsets = [0.0]
        for k in range(1, int(max_tries) + 1):
            offsets.extend([+k * eps_s, -k * eps_s])

        for off in offsets:
            o = P + off * n
            try:
                path = mesh.section(plane_origin=o, plane_normal=n)
            except Exception:
                path = None
            if path is None:
                continue
            entities_candidate = getattr(path, "entities", None)
            has_entities = (
                entities_candidate is not None and len(entities_candidate) > 0
            )
            loops_candidate = getattr(path, "discrete", None)
            has_loops = loops_candidate is not None and len(loops_candidate) > 0
            if not (has_entities or has_loops):
                continue

            M = _world_to_local_plane(P, n)
            polys_2d: List[sgeom.Polygon] = []
            try:
                loops = getattr(path, "discrete", None) or []
                if loops:
                    src_iter = loops
                else:
                    src_iter = [
                        np.asarray(getattr(ent, "points", None), dtype=float)
                        for ent in getattr(path, "entities", [])
                    ]
                for pts3 in src_iter:
                    pts3 = np.asarray(pts3, dtype=float)
                    if pts3.ndim != 2 or pts3.shape[1] != 3 or pts3.shape[0] < 2:
                        continue
                    ones = np.ones((pts3.shape[0], 1), dtype=float)
                    v2 = (M @ np.hstack([pts3, ones]).T).T[:, :3]
                    XY = v2[:, :2]
                    if XY.shape[0] < 3:
                        continue
                    if not np.allclose(XY[0], XY[-1]):
                        XY = np.vstack([XY, XY[0]])
                    poly = sgeom.Polygon(XY)
                    if poly.is_valid and poly.area > 0:
                        polys_2d.append(poly)
            except Exception:
                polys_2d = []

            if not polys_2d:
                continue

            try:
                composed = _compose_polygons_with_holes(polys_2d)
            except Exception:
                composed = polys_2d
            if not composed:
                continue

            origin_pt = sgeom.Point(0.0, 0.0)
            containing: List[sgeom.Polygon] = []
            for poly in composed:
                try:
                    if hasattr(poly, "covers") and poly.covers(origin_pt):
                        containing.append(poly)
                    elif poly.contains(origin_pt):
                        containing.append(poly)
                except Exception:
                    continue
            if containing:
                containing.sort(key=lambda p: float(p.area))
                return containing[0]
            composed.sort(key=lambda p: float(p.exterior.distance(origin_pt)))
            logger.debug(
                "No polygon covered sample origin at %s; using nearest " "section",
                origin,
            )
            return composed[0]

    return None


def _nearest_surface_distance(P: np.ndarray, mesh: trimesh.Trimesh) -> float:
    """Return the closest distance from a point to the mesh surface."""
    try:
        from trimesh.proximity import closest_point

        _, dist, _ = closest_point(
            mesh,
            np.asarray(P, dtype=float).reshape(1, 3),
        )
        return float(dist[0])
    except Exception:
        return 0.0


def _radius_from_section_median(
    poly: sgeom.Polygon,
    *,
    n_rays: int = 64,
) -> float:
    """Estimate section radius from the median ray intersection distance."""
    try:
        if poly is None or not poly.is_valid or poly.area <= 0:
            return 0.0
        origin = sgeom.Point(0.0, 0.0)
        if not poly.contains(origin):
            area = float(poly.area)
            return math.sqrt(area / math.pi) if area > 0 else 0.0

        minx, miny, maxx, maxy = poly.bounds
        R = float(max(maxx - minx, maxy - miny)) * 2.0
        if not np.isfinite(R) or R <= 0:
            R = float(max(1.0, math.sqrt(poly.area / math.pi) * 4.0))

        distances: List[float] = []
        for k in range(int(max(8, n_rays))):
            th = (2.0 * math.pi) * (k / float(n_rays))
            dx = math.cos(th)
            dy = math.sin(th)
            ray = sgeom.LineString([(0.0, 0.0), (dx * R, dy * R)])
            try:
                inter = ray.intersection(poly.exterior)
            except Exception:
                continue
            pts: List[Tuple[float, float]] = []
            if inter.is_empty:
                continue
            if hasattr(inter, "geoms"):
                for g in inter.geoms:
                    if hasattr(g, "x") and hasattr(g, "y"):
                        pts.append((float(g.x), float(g.y)))
            elif hasattr(inter, "x") and hasattr(inter, "y"):
                pts.append((float(inter.x), float(inter.y)))
            best = None
            for x, y in pts:
                d = math.hypot(x, y)
                if x * dx + y * dy >= -1e-9:
                    if best is None or d < best:
                        best = d
            if best is not None and np.isfinite(best):
                distances.append(float(best))
        if not distances:
            return 0.0
        distances.sort()
        return float(distances[len(distances) // 2])
    except Exception:
        return 0.0


def _radius_from_section_circle_fit(poly: sgeom.Polygon) -> float:
    """Estimate section radius by least-squares circle fitting in 2D."""
    try:
        if poly is None or not poly.is_valid or poly.area <= 0:
            return 0.0
        coords = np.asarray(poly.exterior.coords, dtype=float)
        if coords.ndim != 2 or coords.shape[0] < 3:
            return 0.0
        XY = coords[:, :2]
        if XY.shape[0] >= 2 and np.allclose(XY[0], XY[-1]):
            XY = XY[:-1]
        if XY.shape[0] < 3:
            return 0.0
        x = XY[:, 0]
        y = XY[:, 1]
        A = np.column_stack([x, y, np.ones_like(x)])
        b = -(x * x + y * y)
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        a, b2, c = sol
        cx = -0.5 * a
        cy = -0.5 * b2
        r2 = cx * cx + cy * cy - c
        if not np.isfinite(r2) or r2 <= 0:
            return 0.0
        return float(math.sqrt(float(r2)))
    except Exception:
        return 0.0
