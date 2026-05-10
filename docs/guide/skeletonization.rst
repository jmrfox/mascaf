Skeletonization
===============

MASCAF supports three ways to obtain a skeleton.

Option 1: Internal CGAL integration
-------------------------------------

MASCAF includes native CGAL operations:

* ``repair(mesh → mesh)``
* ``simplify(mesh → mesh)``
* ``skeletonize(mesh → .polylines.txt)``

The C++ project lives in ``cpp/`` and the Python interface lives in
:mod:`mascaf.cgal`.

Windows setup
~~~~~~~~~~~~~

1. Install Visual Studio or Build Tools for Visual Studio with C++ support.
2. Install CMake and make sure ``cmake`` is on ``PATH``.
3. Install vcpkg and set the ``VCPKG_ROOT`` environment variable.
4. Open a terminal where your compiler environment is available (e.g., Visual
   Studio Developer Command Prompt).
5. Configure the native project from the repo root:

   .. code-block:: bash

      cmake -S cpp -B cpp/build -DCMAKE_TOOLCHAIN_FILE="%VCPKG_ROOT%/scripts/buildsystems/vcpkg.cmake"

6. Build the executables:

   .. code-block:: bash

      cmake --build cpp/build --config Release

7. Confirm that ``mesh_repair.exe``, ``mesh_simplify.exe``, and
   ``mesh_skeletonize.exe`` exist in the build output.

Linux and macOS setup
~~~~~~~~~~~~~~~~~~~~~

1. Install a C++17-capable compiler toolchain.
2. Install CMake.
3. Install vcpkg and export ``VCPKG_ROOT``.
4. Configure from the repo root:

   .. code-block:: bash

      cmake -S cpp -B cpp/build -DCMAKE_TOOLCHAIN_FILE="$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake"

5. Build:

   .. code-block:: bash

      cmake --build cpp/build --config Release

6. Confirm ``mesh_repair``, ``mesh_simplify``, and ``mesh_skeletonize`` exist in
   the build output.

Python build orchestration
~~~~~~~~~~~~~~~~~~~~~~~~~~

MASCAF provides helpers to drive the build from Python:

.. code-block:: python

   from mascaf import CGALBuilder, CGALConfig

   config = CGALConfig(build_dir="cpp/build")
   builder = CGALBuilder(config)

   builder.run_configure()
   builder.run_build(config="Release")

Running CGAL operations from Python
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from mascaf import CGALConfig, CGALOperator

   config = CGALConfig(build_dir="cpp/build")
   ops = CGALOperator(config=config)

   ops.skeletonize(
       "neuron.obj",
       "neuron.polylines.txt",
       quality_speed_tradeoff=0.5,
       medially_centered_speed_tradeoff=5.0,
   )

Shell usage
~~~~~~~~~~~

The skeletonization executable accepts:

.. code-block:: text

   mesh_skeletonize <input.obj> <output.polylines.txt> [w_H] [w_M]

where ``w_H`` is ``quality_speed_tradeoff`` and ``w_M`` is
``medially_centered_speed_tradeoff``.

Example:

.. code-block:: bash

   mesh_skeletonize neuron.obj neuron.polylines.txt 0.5 5.0

Option 2: CGAL Lab
------------------

The CGAL distribution includes a demo application (**CGAL Lab**) for
interactive skeletonization.

1. Download `CGAL Lab 6.1 <https://www.cgal.org/demo/6.1/CGALlab.zip>`_.
2. Open CGAL Lab and load your mesh.
3. Run **Operations → Triangulated Surface Mesh Skeletonization → Mean
   Curvature Skeleton (Advanced)**.
4. Export the result as a **polylines** text file.

Option 3: Any compatible tool
------------------------------

Any external tool that produces a MASCAF-compatible polylines file works:

.. code-block:: text

   N x1 y1 z1 x2 y2 z2 ... xN yN zN

One line per branch, starting with the number of points on that line.

Troubleshooting
---------------

* **Executable not found** — Build the native ``cpp/`` project first and
  confirm the configuration (``Release``/``Debug``) matches what you built.
* **VCPKG_ROOT is not set** — Export the environment variable before invoking
  CMake or :class:`~mascaf.cgal.CGALBuilder`.
* **CMake cannot find CGAL or Eigen** — Confirm you configured with the vcpkg
  toolchain file and that ``cpp/vcpkg.json`` dependencies were resolved.
* **Skeletonization fails on the mesh** — The CGAL path expects a closed
  triangle mesh.
* **You already have a skeleton** — Skip CGAL entirely and load with
  :meth:`~mascaf.skeleton.SkeletonGraph.from_txt`.
