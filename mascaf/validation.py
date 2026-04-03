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

    def compare_volumes(self, remove_overlaps=False) -> dict:
        """
        Compare total volume between mesh and morphology model.

        Computes the mesh volume using trimesh and the morphology volume
        by summing truncated cone volumes for each edge segment.

        Parameters
        ----------
        remove_overlaps : bool, default False
            If True, remove overlap correction for branch points.

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
            remove_overlaps=remove_overlaps
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

    def compare_surface_areas(self, remove_overlaps=False) -> dict:
        """
        Compare total surface area between mesh and morphology model.

        Computes the mesh surface area using trimesh and the morphology
        surface area by summing lateral surface areas of truncated cones
        for each edge segment.

        Parameters
        ----------
        remove_overlaps : bool, default False
            If True, remove overlap correction for branch points.

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
            remove_overlaps=remove_overlaps
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

    def validate_radii(self) -> dict:
        """
        Validate SWC radii by comparing to actual mesh surface distances.

        For each node in the SWC model, measures the actual distance to the
        mesh surface and compares it to the stored radius value.

        Returns
        -------
        dict
            Dictionary containing:
            - 'mean_error': float, mean absolute error in radius
            - 'max_error': float, maximum absolute error
            - 'std_error': float, standard deviation of errors
            - 'errors': np.ndarray, per-node radius errors
            - 'node_ids': np.ndarray, corresponding node IDs
        """
        raise NotImplementedError("validate_radii not yet implemented")

    def check_skeleton_coverage(self, threshold: float = 1.0) -> dict:
        """
        Check what percentage of mesh is covered by the skeleton.

        Parameters
        ----------
        threshold : float
            Distance threshold in mesh units. Points within this distance
            of the skeleton are considered "covered".

        Returns
        -------
        dict
            Dictionary containing:
            - 'coverage_percentage': float, percentage of mesh vertices
              covered
            - 'max_distance': float, maximum distance from any mesh point
              to skeleton
            - 'mean_distance': float, mean distance from mesh points to
              skeleton
        """
        raise NotImplementedError("check_skeleton_coverage not yet implemented")

    def analyze_cross_sections(self, num_samples: int = 100) -> dict:
        """
        Analyze cross-sections along the skeleton.

        At regular intervals along skeleton edges, computes actual mesh
        cross-section area and compares to πr² from SWC radius.

        Parameters
        ----------
        num_samples : int
            Number of cross-sections to sample along the structure.

        Returns
        -------
        dict
            Dictionary containing cross-section analysis results.
        """
        raise NotImplementedError("analyze_cross_sections not yet implemented")

    def validate_point_cloud(self, num_samples: int = 10000) -> dict:
        """
        Sample points from mesh surface and compute distances to SWC model.

        Parameters
        ----------
        num_samples : int
            Number of points to sample from the mesh surface.

        Returns
        -------
        dict
            Dictionary containing:
            - 'mean_distance': float, mean distance from sampled points to SWC
            - 'max_distance': float, maximum distance
            - 'std_distance': float, standard deviation
            - 'distances': np.ndarray, per-point distances
        """
        raise NotImplementedError("validate_point_cloud not yet implemented")

    def compute_all_metrics(self) -> dict:
        """
        Compute all available validation metrics.

        Returns
        -------
        dict
            Dictionary containing all validation metrics organized by category.
        """
        results = {
            "swc_path": str(self.swc_path) if self.swc_path else None,
            "mesh_info": {
                "vertices": len(self.mesh.vertices),
                "faces": len(self.mesh.faces),
            },
            "skeleton_info": {
                "nodes": self.skeleton.number_of_nodes(),
                "edges": self.skeleton.number_of_edges(),
            },
            "morphology_info": {
                "nodes": self.morphology.number_of_nodes(),
                "edges": self.morphology.number_of_edges(),
            },
        }

        logger.info("Computing all validation metrics...")

        # Add implemented metrics here as they are developed
        # try:
        #     results['volume_comparison'] = self.compare_volumes()
        # except NotImplementedError:
        #     pass

        return results
