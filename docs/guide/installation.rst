Installation
============

Install in your own project environment by providing ``pip`` with the repo link.
This project works well with `uv <https://docs.astral.sh/uv/>`_.

.. code-block:: bash

   uv pip install https://github.com/jmrfox/mascaf.git

If you are working in a local clone of this repo:

.. code-block:: bash

   uv sync

Run code or tests inside that environment:

.. code-block:: bash

   uv run python your_script.py
   uv run pytest

If you already have a valid skeleton file, you do **not** need the internal CGAL
setup at all.

Optional CGAL prerequisites
----------------------------

To use the built-in CGAL skeletonization you must install a native toolchain
separately (MASCAF cannot do this for you):

* A C++ compiler toolchain (e.g., Visual Studio Build Tools on Windows)
* `CMake <https://cmake.org/>`_
* `vcpkg <https://github.com/microsoft/vcpkg>`_ with ``VCPKG_ROOT`` set
* The CGAL/Eigen dependencies resolved through ``cpp/vcpkg.json``

See :doc:`skeletonization` for full setup instructions.
