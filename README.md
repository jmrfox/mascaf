# mcf2swc

This is a Python package designed to take a 3D closed mesh surface (obj format, triangle-faced mesh) and the results of a mean curvature flow calculation (MCF) (polylines format) and produce an SWC model.

[Triangle mesh format](https://en.wikipedia.org/wiki/Triangle_mesh)

[SWC format](http://www.neuronland.org/NLMorphologyConverter/MorphologyFormats/SWC/Spec.html)

Polylines text format is
```N x1 y1 z1 x2 y2 z2 ... xN yN zN```
for each branch of the model.

## Algorithm

The main purpose of this package is to take a closed triangle mesh and the output of a mean curvature flow calculation (MCF) and convert it into an SWC model. The algorithm is as follows:

1. Load the triangle mesh and the MCF polylines.
2. Determine locations of junctions (sometimes called "samples" in SWC format) along the polylines.
3. At each junction, estimate the radius of the cross-section of the mesh at that location.
4. Build a graph of the skeleton using the junctions and the estimated radii.
5. Export the graph as an SWC file.

Since SWC format does not support cycles, a graph with cycles is broken by locating a duplicate node and rewiring one incident cycle edge to the duplicate.