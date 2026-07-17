Quickstart
==========

End-to-end workflow
-------------------

1. **Prepare a mesh** — Start from a closed triangle mesh.
2. **Generate or obtain a skeleton** — See :doc:`skeletonization` for options.
3. **Load inputs** — Load the mesh with :class:`~mascaf.mesh.MeshManager` and
   the skeleton with :meth:`~mascaf.skeleton.SkeletonGraph.from_txt`.
4. **Fit a morphology graph** — Use :class:`~mascaf.cable_fitting.CableFitter`
   with :class:`~mascaf.cable_fitting.FitOptions`.
5. **Optionally optimize the morphology basis** — Configure
   :class:`~mascaf.basis_optimizer.BasisOptimizerOptions` via
   ``FitOptions.basis_optimizer_options``.
6. **Validate and export** — Inspect results, optionally scale radii, and
   export to SWC.

Full example
------------

.. code-block:: python

   from mascaf import (
       BasisOptimizerOptions,
       CableFitter,
       FitOptions,
       MeshManager,
       SkeletonGraph,
   )

   mesh_mgr = MeshManager(mesh_path="neuron.obj")
   skeleton = SkeletonGraph.from_txt("neuron.polylines.txt")

   fit_options = FitOptions(
       max_edge_length=1.0,
       radius_strategy="equivalent_area",
       basis_optimizer_options=BasisOptimizerOptions(
           do_snapping=True,
           do_forcing=True,
           max_iterations=50,
           alpha_s=0.5,
       ),
   )

   fitter = CableFitter(fit_options)
   morphology = fitter.fit(mesh_mgr, skeleton)

   morphology.scale_radii_to_match_mesh(mesh_mgr, metric="surface_area")
   morphology.to_swc_file("neuron.swc")

Minimal example
---------------

If you already have a skeleton and do not want basis optimization:

.. code-block:: python

   from mascaf import CableFitter, FitOptions, MeshManager, SkeletonGraph

   mesh_mgr = MeshManager(mesh_path="shape.obj")
   skel = SkeletonGraph.from_txt("shape.polylines.txt")
   fitter = CableFitter(FitOptions(max_edge_length=0.5))
   morph = fitter.fit(mesh_mgr, skel)
   morph.to_swc_file("shape.swc")

Radius strategies
-----------------

Supported values for ``FitOptions.radius_strategy``:

* ``equivalent_area`` *(recommended)* — ``r = sqrt(A/π)`` from cross-section area.
* ``equivalent_perimeter`` — ``r = L/(2π)`` from boundary length.
* ``section_median`` — median ray-to-boundary distance in the section plane.
* ``section_circle_fit`` — algebraic circle fit (Kasa) to the section boundary.
* ``nearest_surface`` — distance from sample point to nearest mesh surface.
