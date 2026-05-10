"""
Validation module for SWC models.

Provides the Validation class to compare SWC models against original mesh
and skeleton data.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import trimesh

from .mesh import MeshManager
from .morphology_graph import MorphologyGraph
from .skeleton import SkeletonGraph

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class Validation:
    """
    Validation of MorphologyGraph models against original mesh and skeleton.

    This class provides methods to validate a MorphologyGraph by comparing
    it to the original mesh (geometry and surface) and the original skeleton
    (topology and centerline).

    Parameters
    ----------
    mesh : trimesh.Trimesh or MeshManager
        The original mesh to validate against. Can be either a
        trimesh.Trimesh object or a MeshManager instance.
    skeleton : SkeletonGraph
        The original skeleton graph used to generate the morphology.
    morphology : MorphologyGraph or str or Path
        Either a MorphologyGraph instance to validate, or a path to an
        SWC file which will be loaded as a MorphologyGraph.

    Attributes
    ----------
    mesh : trimesh.Trimesh
        The mesh object (extracted from MeshManager if needed).
    mesh_manager : MeshManager or None
        The MeshManager instance if provided, otherwise None.
    skeleton : SkeletonGraph
        The original skeleton graph.
    morphology : MorphologyGraph
        The morphology graph to validate.
    swc_path : Path or None
        Path to the SWC file if loaded from file, otherwise None.

    Examples
    --------
    >>> from mascaf import MeshManager, SkeletonGraph, Validation
    >>> mesh_mgr = MeshManager(mesh_path="neuron.obj")
    >>> skeleton = SkeletonGraph.from_polylines(polylines)
    >>> # Option 1: Validate from SWC file
    >>> validator = Validation(mesh_mgr, skeleton, "output.swc")
    >>> # Option 2: Validate existing MorphologyGraph
    >>> graph = MorphologyGraph.from_swc_file("output.swc")
    >>> validator = Validation(mesh_mgr, skeleton, graph)
    >>> volume_ratio = validator.compare_volumes()
    >>> radius_errors = validator.validate_radii()
    """

    def __init__(
        self,
        mesh: Union[trimesh.Trimesh, MeshManager],
        skeleton: SkeletonGraph,
        morphology: Union[MorphologyGraph, str, Path],
    ):
        # Handle mesh input
        if isinstance(mesh, MeshManager):
            self.mesh_manager = mesh
            self.mesh = mesh.mesh
        elif isinstance(mesh, trimesh.Trimesh):
            self.mesh = mesh
            self.mesh_manager = None
        else:
            raise TypeError(
                f"mesh must be trimesh.Trimesh or MeshManager, " f"got {type(mesh)}"
            )

        # Handle skeleton input
        if not isinstance(skeleton, SkeletonGraph):
            raise TypeError(f"skeleton must be SkeletonGraph, got {type(skeleton)}")
        self.skeleton = skeleton

        # Handle morphology input
        if isinstance(morphology, MorphologyGraph):
            self.morphology = morphology
            self.swc_path = None
        elif isinstance(morphology, (str, Path)):
            self.swc_path = Path(morphology)
            if not self.swc_path.exists():
                raise FileNotFoundError(f"SWC file not found: {self.swc_path}")
            self.morphology = MorphologyGraph.from_swc_file(str(self.swc_path))
        else:
            raise TypeError(
                f"morphology must be MorphologyGraph or path to SWC file, "
                f"got {type(morphology)}"
            )

        # Log initialization
        if self.swc_path:
            logger.info(f"Initialized Validation from SWC: {self.swc_path.name}")
        else:
            logger.info("Initialized Validation from MorphologyGraph")

        logger.info(
            f"  Mesh: {len(self.mesh.vertices)} vertices, "
            f"{len(self.mesh.faces)} faces"
        )
        logger.info(
            f"  Skeleton: {self.skeleton.number_of_nodes()} nodes, "
            f"{self.skeleton.number_of_edges()} edges"
        )
        logger.info(
            f"  MorphologyGraph: {self.morphology.number_of_nodes()} nodes, "
            f"{self.morphology.number_of_edges()} edges"
        )

    def __repr__(self) -> str:
        return (
            f"Validation(\n"
            f"  swc_path={self.swc_path.name if self.swc_path else None},\n"
            f"  mesh_vertices={len(self.mesh.vertices)},\n"
            f"  skeleton_nodes={self.skeleton.number_of_nodes()},\n"
            f"  morphology_nodes={self.morphology.number_of_nodes()}\n"
            f")"
        )

    def compare_volumes(self, account_for_overlaps: bool = False) -> dict:
        """
        Compare total volume between mesh and morphology model.

        Computes the mesh volume using trimesh and the morphology volume
        by summing truncated cone volumes for each edge segment.

        Parameters
        ----------
        account_for_overlaps : bool, default False
            If True, subtract branch-point overlap corrections in the morphology
            volume (see :meth:`MorphologyGraph.compute_volume`).

        Returns
        -------
        dict
            Dictionary containing:
            - 'mesh_volume': float, volume of the mesh
            - 'morphology_volume': float, volume of the morphology model
            - 'ratio': float, morphology_volume / mesh_volume
            - 'absolute_difference': float, |morphology_volume - mesh_volume|
            - 'relative_error': float, abs difference / mesh_volume

        Examples
        --------
        >>> result = validator.compare_volumes()
        >>> print(f"Volume ratio: {result['ratio']:.3f}")
        """
        # Get mesh volume
        mesh_volume = float(self.mesh.volume)

        if not mesh_volume > 0.0:
            raise ValueError("Mesh has zero volume.")

        # Calculate morphology volume using MorphologyGraph method
        morphology_volume = self.morphology.compute_volume(
            account_for_overlaps=account_for_overlaps
        )

        # Calculate comparison metrics
        ratio = morphology_volume / mesh_volume
        error = morphology_volume - mesh_volume
        rel_error = error / mesh_volume

        return {
            "mesh_volume": mesh_volume,
            "morphology_volume": morphology_volume,
            "ratio": ratio,
            "error": error,
            "relative_error": rel_error,
        }

    def compare_surface_areas(self, account_for_overlaps: bool = False) -> dict:
        """
        Compare total surface area between mesh and morphology model.

        Computes the mesh surface area using trimesh and the morphology
        surface area by summing lateral surface areas of truncated cones
        for each edge segment.

        Parameters
        ----------
        account_for_overlaps : bool, default False
            If True, subtract branch-point overlap corrections in the morphology
            surface area (see :meth:`MorphologyGraph.compute_surface_area`).

        Returns
        -------
        dict
            Dictionary containing:
            - 'mesh_area': float, surface area of the mesh
            - 'morphology_area': float, surface area of the morphology model
            - 'ratio': float, morphology_area / mesh_area
            - 'error': float, morphology_area - mesh_area
            - 'relative_error': float, error / mesh_area

        Examples
        --------
        >>> result = validator.compare_surface_areas()
        >>> print(f"Surface area ratio: {result['ratio']:.3f}")
        """
        # Get mesh surface area
        mesh_area = float(self.mesh.area)

        if not mesh_area > 0.0:
            raise ValueError("Mesh has zero area.")

        # Calculate morphology surface area using MorphologyGraph method
        morphology_area = self.morphology.compute_surface_area(
            account_for_overlaps=account_for_overlaps
        )

        # Calculate comparison metrics
        ratio = morphology_area / mesh_area
        error = morphology_area - mesh_area
        rel_error = error / mesh_area

        return {
            "mesh_area": mesh_area,
            "morphology_area": morphology_area,
            "ratio": ratio,
            "error": error,
            "relative_error": rel_error,
        }

    def full_validation(self):
        """
        Run all validation checks and return comprehensive results.
        Returns
        -------
        dict
            Dictionary containing all validation results organized by category.
        """
        has_branch_vertices = any(
            self.morphology.degree[n] > 2 for n in self.morphology.nodes()
        )
        if not has_branch_vertices:
            logger.info(
                "No branch vertices (degree > 2) found; overlap correction "
                "has no effect. Showing account_for_overlaps=False only."
            )

        overlap_flags = [False, True] if has_branch_vertices else [False]
        for account_for_overlaps in overlap_flags:
            vol_result = self.compare_volumes(account_for_overlaps=account_for_overlaps)
            area_result = self.compare_surface_areas(
                account_for_overlaps=account_for_overlaps
            )
            logger.info(
                f"Validation Results, account_for_overlaps={account_for_overlaps}:"
            )
            logger.info("-- Volume Comparison:")
            logger.info(f"---- Mesh volume:       {vol_result['mesh_volume']:.4f}")
            logger.info(
                f"---- Morphology volume: {vol_result['morphology_volume']:.4f}"
            )
            logger.info(f"---- Ratio:             {vol_result['ratio']:.4f}")
            logger.info(f"---- Error:             {vol_result['error']:.4f}")
            logger.info(f"---- Relative error:    {vol_result['relative_error']:.2%}")
            logger.info("-- Surface Area Comparison:")
            logger.info(f"---- Mesh area:         {area_result['mesh_area']:.4f}")
            logger.info(f"---- Morphology area:   {area_result['morphology_area']:.4f}")
            logger.info(f"---- Ratio:             {area_result['ratio']:.4f}")
            logger.info(f"---- Error:             {area_result['error']:.4f}")
            logger.info(f"---- Relative error:    {area_result['relative_error']:.2%}")
        logger.info("")

        return None
