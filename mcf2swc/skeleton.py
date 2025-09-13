from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import networkx as nx
import numpy as np


@dataclass
class Junction:
    """
    Container for a traced skeleton node.

    Fields mirror what the tracing pipeline in `trace.py` constructs for each
    sample along a polyline. The essential geometry is in `center` (XYZ) and
    `radius`; other fields are retained for diagnostics/bookkeeping.
    """

    id: int
    z: float
    center: np.ndarray
    radius: float
    area: float
    slice_index: int
    cross_section_index: int


class SkeletonGraph(nx.Graph):
    """
    Minimal skeleton graph built during tracing.

    This class subclasses `networkx.Graph` and exposes a compatibility property
    `.G` that returns `self`, since callers may expect an internal graph
    accessible via `.G`.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Populated by trace.build_traced_skeleton_graph for optional SWC export adjustments
        self.transforms_applied: List[Dict[str, Any]] = []

    @property
    def G(self) -> nx.Graph:
        """Compatibility shim: return the graph itself."""
        return self

    def add_junction(self, j: Junction) -> None:
        """Add a junction as a node with attributes.

        Node key = `j.id`.
        Stored attributes include `center`, `z`, `radius`, `area`, and indices
        for diagnostics.
        """
        # Ensure array shape/type for consistent downstream usage
        center = np.asarray(j.center, dtype=float).reshape(3)
        self.add_node(
            int(j.id),
            id=int(j.id),
            center=center,
            z=float(j.z),
            radius=float(j.radius),
            area=float(j.area),
            slice_index=int(j.slice_index),
            cross_section_index=int(j.cross_section_index),
        )


__all__ = ["SkeletonGraph", "Junction"]
