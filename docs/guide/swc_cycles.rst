SWC Export and Cycles
=====================

`SWC <http://www.neuronland.org/NLMorphologyConverter/MorphologyFormats/SWC/Spec.html>`_
is inherently a tree format, but MASCAF can handle morphologies with topological
cycles.

MASCAF keeps arbitrary graph topology in memory via
:class:`~mascaf.morphology_graph.MorphologyGraph` and only enforces a tree when
writing SWC, by duplicating nodes where necessary.

For each broken cycle a directive is written to the SWC file header:

.. code-block:: text

   # CYCLE_BREAK <node_id> <parent_id>

When you later reload the file with
:meth:`~mascaf.morphology_graph.MorphologyGraph.from_swc_file`, MASCAF reads
these annotations and reconnects the duplicated nodes to restore the original
cycle topology.

Example
-------

.. code-block:: python

   from mascaf import MorphologyGraph

   # Save — cycles are broken, CYCLE_BREAK annotations written
   morphology.to_swc_file("neuron.swc")

   # Reload — cycles are automatically restored from annotations
   restored = MorphologyGraph.from_swc_file("neuron.swc")
