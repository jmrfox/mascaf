swctools Reference (help() output)
Generated via pydoc.render_doc for modules, classes, and functions.

================================================================================
Module: swctools
--------------------------------------------------------------------------------
Python Library Documentation: package swctools

NAME
    swctools - swctools package scaffolding.

DESCRIPTION
    Public API is evolving; currently exposes SWC parsing, models, geometry, and visualization utilities.

PACKAGE CONTENTS
    config
    geometry
    io
    model
    viz

CLASSES
    builtins.object
        swctools.geometry.FrustaSet
        swctools.geometry.PointSet
        swctools.geometry.Segment
        swctools.io.SWCParseResult
        swctools.io.SWCRecord
    networkx.classes.graph.Graph(builtins.object)
        swctools.model.SWCModel

    class FrustaSet(builtins.object)
     |  FrustaSet(vertices: 'List[Point3]', faces: 'List[Face]', sides: 'int', end_caps: 'bool', segment_count: 'int', edge_count: 'int', segments: 'List[Segment]', edge_uvs: 'Optional[List[Tuple[int, int]]]' = None) -> None
     |
     |  A batched frusta mesh derived from a `SWCModel`.
     |
     |  Attributes
     |  ----------
     |  vertices: List[Point3]
     |      Concatenated vertices for all frusta.
     |  faces: List[Face]
     |      Triangular faces indexing into `vertices`.
     |  sides: int
     |      Circumferential resolution used per frustum.
     |  end_caps: bool
     |      Whether end caps were included during construction.
     |  segment_count: int
     |      Number of segments used (one per graph edge).
     |  edge_count: int
     |      Alias for `segment_count` for clarity.
     |  segments: List[Segment]
     |      The segments used to construct the frusta.
     |  edge_uvs: Optional[List[Tuple[int, int]]]
     |      Optional labels preserving which (u, v) edge generated each segment, in the same order.
     |
     |  Methods defined here:
     |
     |  __delattr__(self, name)
     |      Implement delattr(self, name).
     |
     |  __eq__(self, other)
     |      Return self==value.
     |
     |  __hash__(self)
     |      Return hash(self).
     |
     |  __init__(self, vertices: 'List[Point3]', faces: 'List[Face]', sides: 'int', end_caps: 'bool', segment_count: 'int', edge_count: 'int', segments: 'List[Segment]', edge_uvs: 'Optional[List[Tuple[int, int]]]' = None) -> None
     |      Initialize self.  See help(type(self)) for accurate signature.
     |
     |  __repr__(self)
     |      Return repr(self).
     |
     |  __setattr__(self, name, value)
     |      Implement setattr(self, name, value).
     |
     |  per_segment_face_slices(self) -> 'List[Tuple[int, int]]'
     |      Return (start, count) face spans for each segment in current mesh order.
     |
     |  print_segment_order(self) -> 'None'
     |      Print the current segment ordering; uses `(u, v)` labels if available.
     |
     |  reordered(self, new_order: 'Sequence[int] | None' = None, *, label_remap: 'Optional[Mapping[Tuple[int, int], int]]' = None) -> "'FrustaSet'"
     |      Return a new set with segments reordered by index or (u, v) label mapping.
     |
     |  scale(self, scalar: 'float') -> "'FrustaSet'"
     |      Return a new `FrustaSet` with coordinates and radii scaled by `scalar`.
     |
     |  scaled(self, radius_scale: 'float') -> "'FrustaSet'"
     |      Return a new FrustaSet with all segment radii scaled by `radius_scale`.
     |
     |      This rebuilds vertices/faces from the stored `segments` list.
     |
     |  to_mesh3d_arrays(self) -> 'Tuple[List[float], List[float], List[float], List[int], List[int], List[int]]'
     |      Return Plotly Mesh3d arrays: x, y, z, i, j, k.
     |
     |  ----------------------------------------------------------------------
     |  Class methods defined here:
     |
     |  from_swc_model(model: 'SWCModel', *, sides: 'int' = 16, end_caps: 'bool' = False, flip_tag_assignment: 'bool' = False) -> "'FrustaSet'"
     |      Build a `FrustaSet` by converting each undirected edge into a `Segment`.
     |
     |      Expects nodes to have attributes `x, y, z, r`.
     |
     |      Parameters
     |      ----------
     |      model: SWCModel
     |          The model to convert.
     |      sides: int
     |          Number of sides per frustum.
     |      end_caps: bool
     |          Whether to include end caps.
     |      flip_tag_assignment: bool
     |          If True, assign tags from the child node to the parent node. Otherwise, assign tags from the parent node to the child node.
     |
     |  ----------------------------------------------------------------------
     |  Data descriptors defined here:
     |
     |  __dict__
     |      dictionary for instance variables
     |
     |  __weakref__
     |      list of weak references to the object
     |
     |  ----------------------------------------------------------------------
     |  Data and other attributes defined here:
     |
     |  __annotations__ = {'edge_count': 'int', 'edge_uvs': 'Optional[List[Tup...
     |
     |  __dataclass_fields__ = {'edge_count': Field(name='edge_count',type='in...
     |
     |  __dataclass_params__ = _DataclassParams(init=True,repr=True,eq=True,or...
     |
     |  __match_args__ = ('vertices', 'faces', 'sides', 'end_caps', 'segment_c...
     |
     |  edge_uvs = None

    class PointSet(builtins.object)
     |  PointSet(vertices: 'List[Point3]', faces: 'List[Face]', points: 'List[Point3]', base_radius: 'float', stacks: 'int', slices: 'int') -> None
     |
     |  A batched mesh of small spheres placed at given 3D points.
     |
     |  Methods defined here:
     |
     |  __delattr__(self, name)
     |      Implement delattr(self, name).
     |
     |  __eq__(self, other)
     |      Return self==value.
     |
     |  __hash__(self)
     |      Return hash(self).
     |
     |  __init__(self, vertices: 'List[Point3]', faces: 'List[Face]', points: 'List[Point3]', base_radius: 'float', stacks: 'int', slices: 'int') -> None
     |      Initialize self.  See help(type(self)) for accurate signature.
     |
     |  __repr__(self)
     |      Return repr(self).
     |
     |  __setattr__(self, name, value)
     |      Implement setattr(self, name, value).
     |
     |  project_onto_frusta(self, frusta: "'FrustaSet'", include_end_caps: 'Optional[bool]' = None) -> "'PointSet'"
     |      Project each point to the nearest surface of the nearest frustum.
     |
     |      Parameters
     |      ----------
     |      frusta: FrustaSet
     |          Set of oriented frusta (as `Segment`s) to project onto.
     |      include_end_caps: Optional[bool]
     |          If None (default), follow `frusta.end_caps`. If True/False, explicitly
     |          include or ignore projections to the circular end caps.
     |
     |      Returns
     |      -------
     |      PointSet
     |          A new `PointSet` whose `points` have been moved onto the closest
     |          surface points of the closest frusta; sphere mesh is rebuilt.
     |
     |      Notes
     |      -----
     |      For each input point, the algorithm iterates all frusta segments and
     |      evaluates the squared distance to:
     |      - The lateral surface: project the point to the segment axis (clamped
     |        t in [0,1]), interpolate radius r(t), then move along the radial
     |        direction to the mantle.
     |      - The end caps (optional): orthogonal distance to each cap plane; if
     |        the projected point falls outside the disk, distance to the rim is used.
     |      Degenerate segments (zero length) are treated as a sphere of radius
     |      max(ra, rb) centered at the endpoint.
     |      Complexity is O(N_points × N_segments), implemented in pure Python.
     |
     |  scale(self, scalar: 'float') -> "'PointSet'"
     |      Return a new `PointSet` with coordinates and radii scaled by `scalar`.
     |
     |  scaled(self, radius_scale: 'float') -> "'PointSet'"
     |      Return a new `PointSet` with all sphere radii scaled by `radius_scale`.
     |
     |  to_mesh3d_arrays(self) -> 'Tuple[List[float], List[float], List[float], List[int], List[int], List[int]]'
     |      Return Plotly `Mesh3d` arrays `(x, y, z, i, j, k)` for this point set.
     |
     |  to_txt_file(self, path: 'Union[str, os.PathLike]') -> 'None'
     |
     |  ----------------------------------------------------------------------
     |  Class methods defined here:
     |
     |  from_points(points: 'Sequence[Point3]', *, base_radius: 'float' = 1.0, stacks: 'int' = 6, slices: 'int' = 12) -> "'PointSet'"
     |      Build a batched low-res spheres mesh from a list of 3D points.
     |
     |      Parameters
     |      ----------
     |      points: sequence of (x, y, z)
     |          Sphere centers.
     |      base_radius: float
     |          Sphere radius used when building the mesh (scaled later via `scaled()`).
     |      stacks, slices: int
     |          Sphere tessellation parameters (>=2 and >=3 respectively).
     |
     |  from_txt_file(path: 'Union[str, os.PathLike]', *, base_radius: 'float' = 1.0, stacks: 'int' = 6, slices: 'int' = 12, comments: 'str' = '#') -> "'PointSet'"
     |
     |  ----------------------------------------------------------------------
     |  Data descriptors defined here:
     |
     |  __dict__
     |      dictionary for instance variables
     |
     |  __weakref__
     |      list of weak references to the object
     |
     |  ----------------------------------------------------------------------
     |  Data and other attributes defined here:
     |
     |  __annotations__ = {'base_radius': 'float', 'faces': 'List[Face]', 'poi...
     |
     |  __dataclass_fields__ = {'base_radius': Field(name='base_radius',type='...
     |
     |  __dataclass_params__ = _DataclassParams(init=True,repr=True,eq=True,or...
     |
     |  __match_args__ = ('vertices', 'faces', 'points', 'base_radius', 'stack...

    class SWCModel(networkx.classes.graph.Graph)
     |  SWCModel() -> 'None'
     |
     |  SWC morphology graph (undirected storage with directed-tree metadata).
     |
     |  Nodes are keyed by SWC id `n` and store attributes:
     |  - t: int (tag)
     |  - x, y, z: float (coordinates)
     |  - r: float (radius)
     |  - line: int (line number in source; informational)
     |
     |  Directed parent -> child relationships from the SWC are preserved via an
     |  internal parent map (`_parents`). The underlying graph is undirected. Cycle
     |  connections can be applied via `make_cycle_connections()` which may merge nodes.
     |
     |  Method resolution order:
     |      SWCModel
     |      networkx.classes.graph.Graph
     |      builtins.object
     |
     |  Methods defined here:
     |
     |  __init__(self) -> 'None'
     |      Initialize a graph with edges, name, or graph attributes.
     |
     |      Parameters
     |      ----------
     |      incoming_graph_data : input graph (optional, default: None)
     |          Data to initialize graph. If None (default) an empty
     |          graph is created.  The data can be an edge list, or any
     |          NetworkX graph object.  If the corresponding optional Python
     |          packages are installed the data can also be a 2D NumPy array, a
     |          SciPy sparse array, or a PyGraphviz graph.
     |
     |      attr : keyword arguments, optional (default= no attributes)
     |          Attributes to add to graph as key=value pairs.
     |
     |      See Also
     |      --------
     |      convert
     |
     |      Examples
     |      --------
     |      >>> G = nx.Graph()  # or DiGraph, MultiGraph, MultiDiGraph, etc
     |      >>> G = nx.Graph(name="my graph")
     |      >>> e = [(1, 2), (2, 3), (3, 4)]  # list of edges
     |      >>> G = nx.Graph(e)
     |
     |      Arbitrary graph attribute pairs (key=value) may be assigned
     |
     |      >>> G = nx.Graph(e, day="Friday")
     |      >>> G.graph
     |      {'day': 'Friday'}
     |
     |  add_junction(self, node_id: 'int | None' = None, *, t: 'int' = 0, x: 'float' = 0.0, y: 'float' = 0.0, z: 'float' = 0.0, r: 'float' = 0.0, parent: 'int | None' = None, **kwargs: 'Any') -> 'int'
     |      Add a junction (node) to the model.
     |
     |      Parameters
     |      ----------
     |      node_id: int | None
     |          Node ID to use. If None, automatically assigns the next available ID.
     |      t: int
     |          Node tag. Default 0.
     |      x, y, z: float
     |          Node coordinates. Default 0.0.
     |      r: float
     |          Node radius. Default 0.0.
     |      parent: int | None
     |          Parent node ID. If specified, creates an edge to the parent.
     |          Default None (root node).
     |      **kwargs: Any
     |          Additional node attributes.
     |
     |      Returns
     |      -------
     |      int
     |          The ID of the added node.
     |
     |  copy(self) -> "'SWCModel'"
     |      Return a shallow copy of this model (nodes/edges/attributes).
     |
     |  make_cycle_connections(self, *, validate_reconnections: 'bool' = True, float_tol: 'float' = 1e-09) -> "'SWCModel'"
     |      Return a new SWCModel with reconnection pairs merged and undirected edges across merged reps.
     |
     |      Uses reconnection pairs stored under `self.graph['reconnections']` if present.
     |      Node attributes are merged; provenance kept under `merged_ids` and `lines`.
     |      After merges, the original tree parent mapping no longer applies and parents are set to None.
     |
     |  parent_of(self, n: 'int') -> 'int | None'
     |      Return the parent id of node n from the original SWC tree (or None).
     |
     |  path_to_root(self, n: 'int') -> 'list[int]'
     |      Return the path from node n up to its root, inclusive.
     |
     |      Example: For edges 1->2->3, `path_to_root(3)` returns `[3, 2, 1]`.
     |
     |  print_attributes(self, *, node_info: 'bool' = False, edge_info: 'bool' = False) -> 'None'
     |      Print graph attributes and optional node/edge details.
     |
     |      Parameters
     |      ----------
     |      node_info: bool
     |          If True, print per-node attributes (t, x, y, z, r, line where present).
     |      edge_info: bool
     |          If True, print all edges (u -- v) with edge attributes if any.
     |
     |  remove_junction(self, node_id: 'int', *, reconnect_children: 'bool' = False) -> 'None'
     |      Remove a junction (node) from the model.
     |
     |      Parameters
     |      ----------
     |      node_id: int
     |          ID of the node to remove.
     |      reconnect_children: bool
     |          If True, reconnect children of the removed node to its parent.
     |          If False (default), children become orphaned (roots).
     |
     |  roots(self) -> 'list[int]'
     |      Return nodes with no parent in the original SWC tree.
     |
     |  scale(self, scalar: 'float') -> "'SWCModel'"
     |      Return a new model with all node coordinates and radii scaled by `scalar`.
     |
     |      Multiplies each node's `x`, `y`, `z`, and `r` by `scalar` on a copy.
     |
     |  set_tag_by_sphere(self, center: 'tuple[float, float, float] | list[float]', radius: 'float', new_tag: 'int', old_tag: 'int | None' = None, include_boundary: 'bool' = True, copy: 'bool' = False) -> "'SWCModel'"
     |      Override node 't' values for points inside a sphere.
     |
     |      Sets the tag 't' for all nodes whose Euclidean distance from
     |      `center` is less than `radius` (or equal if `include_boundary` is True).
     |
     |      If `old_tag` is specified, only nodes with that tag are modified.
     |
     |      Parameters
     |      ----------
     |      center: tuple[float, float, float] | list[float]
     |          Sphere center as (x, y, z).
     |      radius: float
     |          Sphere radius (same units as coordinates).
     |      new_tag: int
     |          Tag to assign to matching nodes.
     |      old_tag: int | None
     |          If specified, only nodes with this tag are modified.
     |      include_boundary: bool
     |          If True, include nodes exactly at distance == radius. Default True.
     |      copy: bool
     |          If True, operate on and return a copy; otherwise mutate in place and return self.
     |
     |  to_swc_file(self, path: 'str | os.PathLike[str]', *, precision: 'int' = 6, header: 'Iterable[str] | None' = None) -> 'None'
     |      Write the model to an SWC file.
     |
     |      The output uses the standard 7-column SWC format per row:
     |      "n T x y z r parent" with floats formatted to the requested precision.
     |
     |      Parameters
     |      ----------
     |      path: str | os.PathLike[str]
     |          Destination file path.
     |      precision: int
     |          Decimal places for floating-point fields (x, y, z, r). Default 6.
     |      header: Iterable[str] | None
     |          Optional additional header comment lines (without leading '#').
     |
     |  ----------------------------------------------------------------------
     |  Class methods defined here:
     |
     |  from_parse_result(result: 'SWCParseResult') -> "'SWCModel'"
     |      Build a model from a parsed SWC result.
     |
     |  from_records(records: 'Mapping[int, SWCRecord] | Iterable[SWCRecord]') -> "'SWCModel'"
     |      Build a model from SWC records.
     |
     |      Accepts either a mapping of id->record or any iterable of SWCRecord.
     |
     |  from_swc_file(source: 'str | os.PathLike[str] | Iterable[str]', *, strict: 'bool' = True, validate_reconnections: 'bool' = True, float_tol: 'float' = 1e-09) -> "'SWCModel'"
     |      Parse an SWC source then build a model.
     |
     |      The `source` is passed through to `parse_swc`, which supports a path,
     |      a file-like object, a string with the full contents, or an iterable of lines.
     |
     |  ----------------------------------------------------------------------
     |  Methods inherited from networkx.classes.graph.Graph:
     |
     |  __contains__(self, n)
     |      Returns True if n is a node, False otherwise. Use: 'n in G'.
     |
     |      Examples
     |      --------
     |      >>> G = nx.path_graph(4)  # or DiGraph, MultiGraph, MultiDiGraph, etc
     |      >>> 1 in G
     |      True
     |
     |  __getitem__(self, n)
     |      Returns a dict of neighbors of node n.  Use: 'G[n]'.
     |
     |      Parameters
     |      ----------
     |      n : node
     |         A node in the graph.
     |
     |      Returns
     |      -------
     |      adj_dict : dictionary
     |         The adjacency dictionary for nodes connected to n.
     |
     |      Notes
     |      -----
     |      G[n] is the same as G.adj[n] and similar to G.neighbors(n)
     |      (which is an iterator over G.adj[n])
     |
     |      Examples
     |      --------
     |      >>> G = nx.path_graph(4)  # or DiGraph, MultiGraph, MultiDiGraph, etc
     |      >>> G[0]
     |      AtlasView({1: {}})
     |
     |  __iter__(self)
     |      Iterate over the nodes. Use: 'for n in G'.
     |
     |      Returns
     |      -------
     |      niter : iterator
     |          An iterator over all nodes in the graph.
     |
     |      Examples
     |      --------
     |      >>> G = nx.path_graph(4)  # or DiGraph, MultiGraph, MultiDiGraph, etc
     |      >>> [n for n in G]
     |      [0, 1, 2, 3]
     |      >>> list(G)
     |      [0, 1, 2, 3]
     |
     |  __len__(self)
     |      Returns the number of nodes in the graph. Use: 'len(G)'.
     |
     |      Returns
     |      -------
     |      nnodes : int
     |          The number of nodes in the graph.
     |
     |      See Also
     |      --------
     |      number_of_nodes: identical method
     |      order: identical method
     |
     |      Examples
     |      --------
     |      >>> G = nx.path_graph(4)  # or DiGraph, MultiGraph, MultiDiGraph, etc
     |      >>> len(G)
     |      4
     |
     |  __str__(self)
     |      Returns a short summary of the graph.
     |
     |      Returns
     |      -------
     |      info : string
     |          Graph information including the graph name (if any), graph type, and the
     |          number of nodes and edges.
     |
     |      Examples
     |      --------
     |      >>> G = nx.Graph(name="foo")
     |      >>> str(G)
     |      "Graph named 'foo' with 0 nodes and 0 edges"
     |
     |      >>> G = nx.path_graph(3)
     |      >>> str(G)
     |      'Graph with 3 nodes and 2 edges'
     |
     |  add_edge(self, u_of_edge, v_of_edge, **attr)
     |      Add an edge between u and v.
     |
     |      The nodes u and v will be automatically added if they are
     |      not already in the graph.
     |
     |      Edge attributes can be specified with keywords or by directly
     |      accessing the edge's attribute dictionary. See examples below.
     |
     |      Parameters
     |      ----------
     |      u_of_edge, v_of_edge : nodes
     |          Nodes can be, for example, strings or numbers.
     |          Nodes must be hashable (and not None) Python objects.
     |      attr : keyword arguments, optional
     |          Edge data (or labels or objects) can be assigned using
     |          keyword arguments.
     |
     |      See Also
     |      --------
     |      add_edges_from : add a collection of edges
     |
     |      Notes
     |      -----
     |      Adding an edge that already exists updates the edge data.
     |
     |      Many NetworkX algorithms designed for weighted graphs use
     |      an edge attribute (by default `weight`) to hold a numerical value.
     |
     |      Examples
     |      --------
     |      The following all add the edge e=(1, 2) to graph G:
     |
     |      >>> G = nx.Graph()  # or DiGraph, MultiGraph, MultiDiGraph, etc
     |      >>> e = (1, 2)
     |      >>> G.add_edge(1, 2)  # explicit two-node form
     |      >>> G.add_edge(*e)  # single edge as tuple of two nodes
     |      >>> G.add_edges_from([(1, 2)])  # add edges from iterable container
     |
     |      Associate data to edges using keywords:
     |
     |      >>> G.add_edge(1, 2, weight=3)
     |      >>> G.add_edge(1, 3, weight=7, capacity=15, length=342.7)
     |
     |      For non-string attribute keys, use subscript notation.
     |
     |      >>> G.add_edge(1, 2)
     |      >>> G[1][2].update({0: 5})
     |      >>> G.edges[1, 2].update({0: 5})
     |
     |  add_edges_from(self, ebunch_to_add, **attr)
     |      Add all the edges in ebunch_to_add.
     |
     |      Parameters
     |      ----------
     |      ebunch_to_add : container of edges
     |          Each edge given in the container will be added to the
     |          graph. The edges must be given as 2-tuples (u, v) or
     |          3-tuples (u, v, d) where d is a dictionary containing edge data.
     |      attr : keyword arguments, optional
     |          Edge data (or labels or objects) can be assigned using
     |          keyword arguments.
     |
     |      See Also
     |      --------
     |      add_edge : add a single edge
     |      add_weighted_edges_from : convenient way to add weighted edges
     |
     |      Notes
     |      -----
     |      Adding the same edge twice has no effect but any edge data
     |      will be updated when each duplicate edge is added.
     |
     |      Edge attributes specified in an ebunch take precedence over
     |      attributes specified via keyword arguments.
     |
     |      When adding edges from an iterator over the graph you are changing,
     |      a `RuntimeError` can be raised with message:
     |      `RuntimeError: dictionary changed size during iteration`. This
     |      happens when the graph's underlying dictionary is modified during
     |      iteration. To avoid this error, evaluate the iterator into a separate
     |      object, e.g. by using `list(iterator_of_edges)`, and pass this
     |      object to `G.add_edges_from`.
     |
     |      Examples
     |      --------
     |      >>> G = nx.Graph()  # or DiGraph, MultiGraph, MultiDiGraph, etc
     |      >>> G.add_edges_from([(0, 1), (1, 2)])  # using a list of edge tuples
     |      >>> e = zip(range(0, 3), range(1, 4))
     |      >>> G.add_edges_from(e)  # Add the path graph 0-1-2-3
     |
     |      Associate data to edges
     |
     |      >>> G.add_edges_from([(1, 2), (2, 3)], weight=3)
     |      >>> G.add_edges_from([(3, 4), (1, 4)], label="WN2898")
     |
     |      Evaluate an iterator over a graph if using it to modify the same graph
     |
     |      >>> G = nx.Graph([(1, 2), (2, 3), (3, 4)])
     |      >>> # Grow graph by one new node, adding edges to all existing nodes.
     |      >>> # wrong way - will raise RuntimeError
     |      >>> # G.add_edges_from(((5, n) for n in G.nodes))
     |      >>> # correct way - note that there will be no self-edge for node 5
     |      >>> G.add_edges_from(list((5, n) for n in G.nodes))
     |
     |  add_node(self, node_for_adding, **attr)
     |      Add a single node `node_for_adding` and update node attributes.
     |
     |      Parameters
     |      ----------
     |      node_for_adding : node
     |          A node can be any hashable Python object except None.
     |      attr : keyword arguments, optional
     |          Set or change node attributes using key=value.
     |
     |      See Also
     |      --------
     |      add_nodes_from
     |
     |      Examples
     |      --------
     |      >>> G = nx.Graph()  # or DiGraph, MultiGraph, MultiDiGraph, etc
     |      >>> G.add_node(1)
     |      >>> G.add_node("Hello")
     |      >>> K3 = nx.Graph([(0, 1), (1, 2), (2, 0)])
     |      >>> G.add_node(K3)
     |      >>> G.number_of_nodes()
     |      3
     |
     |      Use keywords set/change node attributes:
     |
     |      >>> G.add_node(1, size=10)
     |      >>> G.add_node(3, weight=0.4, UTM=("13S", 382871, 3972649))
     |
     |      Notes
     |      -----
     |      A hashable object is one that can be used as a key in a Python
     |      dictionary. This includes strings, numbers, tuples of strings
     |      and numbers, etc.
     |
     |      On many platforms hashable items also include mutables such as
     |      NetworkX Graphs, though one should be careful that the hash
     |      doesn't change on mutables.
     |
     |  add_nodes_from(self, nodes_for_adding, **attr)
     |      Add multiple nodes.
     |
     |      Parameters
     |      ----------
     |      nodes_for_adding : iterable container
     |          A container of nodes (list, dict, set, etc.).
     |          OR
     |          A container of (node, attribute dict) tuples.
     |          Node attributes are updated using the attribute dict.
     |      attr : keyword arguments, optional (default= no attributes)
     |          Update attributes for all nodes in nodes.
     |          Node attributes specified in nodes as a tuple take
     |          precedence over attributes specified via keyword arguments.
     |
     |      See Also
     |      --------
     |      add_node
     |
     |      Notes
     |      -----
     |      When adding nodes from an iterator over the graph you are changing,
     |      a `RuntimeError` can be raised with message:
     |      `RuntimeError: dictionary changed size during iteration`. This
     |      happens when the graph's underlying dictionary is modified during
     |      iteration. To avoid this error, evaluate the iterator into a separate
     |      object, e.g. by using `list(iterator_of_nodes)`, and pass this
     |      object to `G.add_nodes_from`.
     |
     |      Examples
     |      --------
     |      >>> G = nx.Graph()  # or DiGraph, MultiGraph, MultiDiGraph, etc
     |      >>> G.add_nodes_from("Hello")
     |      >>> K3 = nx.Graph([(0, 1), (1, 2), (2, 0)])
     |      >>> G.add_nodes_from(K3)
     |      >>> sorted(G.nodes(), key=str)
     |      [0, 1, 2, 'H', 'e', 'l', 'o']
     |
     |      Use keywords to update specific node attributes for every node.
     |
     |      >>> G.add_nodes_from([1, 2], size=10)
     |      >>> G.add_nodes_from([3, 4], weight=0.4)
     |
     |      Use (node, attrdict) tuples to update attributes for specific nodes.
     |
     |      >>> G.add_nodes_from([(1, dict(size=11)), (2, {"color": "blue"})])
     |      >>> G.nodes[1]["size"]
     |      11
     |      >>> H = nx.Graph()
     |      >>> H.add_nodes_from(G.nodes(data=True))
     |      >>> H.nodes[1]["size"]
     |      11
     |
     |      Evaluate an iterator over a graph if using it to modify the same graph
     |
     |      >>> G = nx.Graph([(0, 1), (1, 2), (3, 4)])
     |      >>> # wrong way - will raise RuntimeError
     |      >>> # G.add_nodes_from(n + 1 for n in G.nodes)
     |      >>> # correct way
     |      >>> G.add_nodes_from(list(n + 1 for n in G.nodes))
     |
     |  add_weighted_edges_from(self, ebunch_to_add, weight='weight', **attr)
     |      Add weighted edges in `ebunch_to_add` with specified weight attr
     |
     |      Parameters
     |      ----------
     |      ebunch_to_add : container of edges
     |          Each edge given in the list or container will be added
     |          to the graph. The edges must be given as 3-tuples (u, v, w)
     |          where w is a number.
     |      weight : string, optional (default= 'weight')
     |          The attribute name for the edge weights to be added.
     |      attr : keyword arguments, optional (default= no attributes)
     |          Edge attributes to add/update for all edges.
     |
     |      See Also
     |      --------
     |      add_edge : add a single edge
     |      add_edges_from : add multiple edges
     |
     |      Notes
     |      -----
     |      Adding the same edge twice for Graph/DiGraph simply updates
     |      the edge data. For MultiGraph/MultiDiGraph, duplicate edges
     |      are stored.
     |
     |      When adding edges from an iterator over the graph you are changing,
     |      a `RuntimeError` can be raised with message:
     |      `RuntimeError: dictionary changed size during iteration`. This
     |      happens when the graph's underlying dictionary is modified during
     |      iteration. To avoid this error, evaluate the iterator into a separate
     |      object, e.g. by using `list(iterator_of_edges)`, and pass this
     |      object to `G.add_weighted_edges_from`.
     |
     |      Examples
     |      --------
     |      >>> G = nx.Graph()  # or DiGraph, MultiGraph, MultiDiGraph, etc
     |      >>> G.add_weighted_edges_from([(0, 1, 3.0), (1, 2, 7.5)])
     |
     |      Evaluate an iterator over edges before passing it
     |
     |      >>> G = nx.Graph([(1, 2), (2, 3), (3, 4)])
     |      >>> weight = 0.1
     |      >>> # Grow graph by one new node, adding edges to all existing nodes.
     |      >>> # wrong way - will raise RuntimeError
     |      >>> # G.add_weighted_edges_from(((5, n, weight) for n in G.nodes))
     |      >>> # correct way - note that there will be no self-edge for node 5
     |      >>> G.add_weighted_edges_from(list((5, n, weight) for n in G.nodes))
     |
     |  adj = <functools.cached_property object>
     |      Graph adjacency object holding the neighbors of each node.
     |
     |      This object is a read-only dict-like structure with node keys
     |      and neighbor-dict values.  The neighbor-dict is keyed by neighbor
     |      to the edge-data-dict.  So `G.adj[3][2]['color'] = 'blue'` sets
     |      the color of the edge `(3, 2)` to `"blue"`.
     |
     |      Iterating over G.adj behaves like a dict. Useful idioms include
     |      `for nbr, datadict in G.adj[n].items():`.
     |
     |      The neighbor information is also provided by subscripting the graph.
     |      So `for nbr, foovalue in G[node].data('foo', default=1):` works.
     |
     |      For directed graphs, `G.adj` holds outgoing (successor) info.
     |
     |  adjacency(self)
     |      Returns an iterator over (node, adjacency dict) tuples for all nodes.
     |
     |      For directed graphs, only outgoing neighbors/adjacencies are included.
     |
     |      Returns
     |      -------
     |      adj_iter : iterator
     |         An iterator over (node, adjacency dictionary) for all nodes in
     |         the graph.
     |
     |      Examples
     |      --------
     |      >>> G = nx.path_graph(4)  # or DiGraph, MultiGraph, MultiDiGraph, etc
     |      >>> [(n, nbrdict) for n, nbrdict in G.adjacency()]
     |      [(0, {1: {}}), (1, {0: {}, 2: {}}), (2, {1: {}, 3: {}}), (3, {2: {}})]
     |
     |  clear(self)
     |      Remove all nodes and edges from the graph.
     |
     |      This also removes the name, and all graph, node, and edge attributes.
     |
     |      Examples
     |      --------
     |      >>> G = nx.path_graph(4)  # or DiGraph, MultiGraph, MultiDiGraph, etc
     |      >>> G.clear()
     |      >>> list(G.nodes)
     |      []
     |      >>> list(G.edges)
     |      []
     |
     |  clear_edges(self)
     |      Remove all edges from the graph without altering nodes.
     |
     |      Examples
     |      --------
     |      >>> G = nx.path_graph(4)  # or DiGraph, MultiGraph, MultiDiGraph, etc
     |      >>> G.clear_edges()
     |      >>> list(G.nodes)
     |      [0, 1, 2, 3]
     |      >>> list(G.edges)
     |      []
     |
     |  degree = <functools.cached_property object>
     |      A DegreeView for the Graph as G.degree or G.degree().
     |
     |      The node degree is the number of edges adjacent to the node.
     |      The weighted node degree is the sum of the edge weights for
     |      edges incident to that node.
     |
     |      This object provides an iterator for (node, degree) as well as
     |      lookup for the degree for a single node.
     |
     |      Parameters
     |      ----------
     |      nbunch : single node, container, or all nodes (default= all nodes)
     |          The view will only report edges incident to these nodes.
     |
     |      weight : string or None, optional (default=None)
     |         The name of an edge attribute that holds the numerical value used
     |         as a weight.  If None, then each edge has weight 1.
     |         The degree is the sum of the edge weights adjacent to the node.
     |
     |      Returns
     |      -------
     |      DegreeView or int
     |          If multiple nodes are requested (the default), returns a `DegreeView`
     |          mapping nodes to their degree.
     |          If a single node is requested, returns the degree of the node as an integer.
     |
     |      Examples
     |      --------
     |      >>> G = nx.path_graph(4)  # or DiGraph, MultiGraph, MultiDiGraph, etc
     |      >>> G.degree[0]  # node 0 has degree 1
     |      1
     |      >>> list(G.degree([0, 1, 2]))
     |      [(0, 1), (1, 2), (2, 2)]
     |
     |  edge_subgraph(self, edges)
     |      Returns the subgraph induced by the specified edges.
     |
     |      The induced subgraph contains each edge in `edges` and each
     |      node incident to any one of those edges.
     |
     |      Parameters
     |      ----------
     |      edges : iterable
     |          An iterable of edges in this graph.
     |
     |      Returns
     |      -------
     |      G : Graph
     |          An edge-induced subgraph of this graph with the same edge
     |          attributes.
     |
     |      Notes
     |      -----
     |      The graph, edge, and node attributes in the returned subgraph
     |      view are references to the corresponding attributes in the original
     |      graph. The view is read-only.
     |
     |      To create a full graph version of the subgraph with its own copy
     |      of the edge or node attributes, use::
     |
     |          G.edge_subgraph(edges).copy()
     |
     |      Examples
     |      --------
     |      >>> G = nx.path_graph(5)
     |      >>> H = G.edge_subgraph([(0, 1), (3, 4)])
     |      >>> list(H.nodes)
     |      [0, 1, 3, 4]
     |      >>> list(H.edges)
     |      [(0, 1), (3, 4)]
     |
     |  edges = <functools.cached_property object>
     |      An EdgeView of the Graph as G.edges or G.edges().
     |
     |      edges(self, nbunch=None, data=False, default=None)
     |
     |      The EdgeView provides set-like operations on the edge-tuples
     |      as well as edge attribute lookup. When called, it also provides
     |      an EdgeDataView object which allows control of access to edge
     |      attributes (but does not provide set-like operations).
     |      Hence, `G.edges[u, v]['color']` provides the value of the color
     |      attribute for edge `(u, v)` while
     |      `for (u, v, c) in G.edges.data('color', default='red'):`
     |      iterates through all the edges yielding the color attribute
     |      with default `'red'` if no color attribute exists.
     |
     |      Parameters
     |      ----------
     |      nbunch : single node, container, or all nodes (default= all nodes)
     |          The view will only report edges from these nodes.
     |      data : string or bool, optional (default=False)
     |          The edge attribute returned in 3-tuple (u, v, ddict[data]).
     |          If True, return edge attribute dict in 3-tuple (u, v, ddict).
     |          If False, return 2-tuple (u, v).
     |      default : value, optional (default=None)
     |          Value used for edges that don't have the requested attribute.
     |          Only relevant if data is not True or False.
     |
     |      Returns
     |      -------
     |      edges : EdgeView
     |          A view of edge attributes, usually it iterates over (u, v)
     |          or (u, v, d) tuples of edges, but can also be used for
     |          attribute lookup as `edges[u, v]['foo']`.
     |
     |      Notes
     |      -----
     |      Nodes in nbunch that are not in the graph will be (quietly) ignored.
     |      For directed graphs this returns the out-edges.
     |
     |      Examples
     |      --------
     |      >>> G = nx.path_graph(3)  # or MultiGraph, etc
     |      >>> G.add_edge(2, 3, weight=5)
     |      >>> [e for e in G.edges]
     |      [(0, 1), (1, 2), (2, 3)]
     |      >>> G.edges.data()  # default data is {} (empty dict)
     |      EdgeDataView([(0, 1, {}), (1, 2, {}), (2, 3, {'weight': 5})])
     |      >>> G.edges.data("weight", default=1)
     |      EdgeDataView([(0, 1, 1), (1, 2, 1), (2, 3, 5)])
     |      >>> G.edges([0, 3])  # only edges from these nodes
     |      EdgeDataView([(0, 1), (3, 2)])
     |      >>> G.edges(0)  # only edges from node 0
     |      EdgeDataView([(0, 1)])
     |
     |  get_edge_data(self, u, v, default=None)
     |      Returns the attribute dictionary associated with edge (u, v).
     |
     |      This is identical to `G[u][v]` except the default is returned
     |      instead of an exception if the edge doesn't exist.
     |
     |      Parameters
     |      ----------
     |      u, v : nodes
     |      default:  any Python object (default=None)
     |          Value to return if the edge (u, v) is not found.
     |
     |      Returns
     |      -------
     |      edge_dict : dictionary
     |          The edge attribute dictionary.
     |
     |      Examples
     |      --------
     |      >>> G = nx.path_graph(4)  # or DiGraph, MultiGraph, MultiDiGraph, etc
     |      >>> G[0][1]
     |      {}
     |
     |      Warning: Assigning to `G[u][v]` is not permitted.
     |      But it is safe to assign attributes `G[u][v]['foo']`
     |
     |      >>> G[0][1]["weight"] = 7
     |      >>> G[0][1]["weight"]
     |      7
     |      >>> G[1][0]["weight"]
     |      7
     |
     |      >>> G = nx.path_graph(4)  # or DiGraph, MultiGraph, MultiDiGraph, etc
     |      >>> G.get_edge_data(0, 1)  # default edge data is {}
     |      {}
     |      >>> e = (0, 1)
     |      >>> G.get_edge_data(*e)  # tuple form
     |      {}
     |      >>> G.get_edge_data("a", "b", default=0)  # edge not in graph, return 0
     |      0
     |
     |  has_edge(self, u, v)
     |      Returns True if the edge (u, v) is in the graph.
     |
     |      This is the same as `v in G[u]` without KeyError exceptions.
     |
     |      Parameters
     |      ----------
     |      u, v : nodes
     |          Nodes can be, for example, strings or numbers.
     |          Nodes must be hashable (and not None) Python objects.
     |
     |      Returns
     |      -------
     |      edge_ind : bool
     |          True if edge is in the graph, False otherwise.
     |
     |      Examples
     |      --------
     |      >>> G = nx.path_graph(4)  # or DiGraph, MultiGraph, MultiDiGraph, etc
     |      >>> G.has_edge(0, 1)  # using two nodes
     |      True
     |      >>> e = (0, 1)
     |      >>> G.has_edge(*e)  #  e is a 2-tuple (u, v)
     |      True
     |      >>> e = (0, 1, {"weight": 7})
     |      >>> G.has_edge(*e[:2])  # e is a 3-tuple (u, v, data_dictionary)
     |      True
     |
     |      The following syntax are equivalent:
     |
     |      >>> G.has_edge(0, 1)
     |      True
     |      >>> 1 in G[0]  # though this gives KeyError if 0 not in G
     |      True
     |
     |  has_node(self, n)
     |      Returns True if the graph contains the node n.
     |
     |      Identical to `n in G`
     |
     |      Parameters
     |      ----------
     |      n : node
     |
     |      Examples
     |      --------
     |      >>> G = nx.path_graph(3)  # or DiGraph, MultiGraph, MultiDiGraph, etc
     |      >>> G.has_node(0)
     |      True
     |
     |      It is more readable and simpler to use
     |
     |      >>> 0 in G
     |      True
     |
     |  is_directed(self)
     |      Returns True if graph is directed, False otherwise.
     |
     |  is_multigraph(self)
     |      Returns True if graph is a multigraph, False otherwise.
     |
     |  nbunch_iter(self, nbunch=None)
     |      Returns an iterator over nodes contained in nbunch that are
     |      also in the graph.
     |
     |      The nodes in an iterable nbunch are checked for membership in the graph
     |      and if not are silently ignored.
     |
     |      Parameters
     |      ----------
     |      nbunch : single node, container, or all nodes (default= all nodes)
     |          The view will only report edges incident to these nodes.
     |
     |      Returns
     |      -------
     |      niter : iterator
     |          An iterator over nodes in nbunch that are also in the graph.
     |          If nbunch is None, iterate over all nodes in the graph.
     |
     |      Raises
     |      ------
     |      NetworkXError
     |          If nbunch is not a node or sequence of nodes.
     |          If a node in nbunch is not hashable.
     |
     |      See Also
     |      --------
     |      Graph.__iter__
     |
     |      Notes
     |      -----
     |      When nbunch is an iterator, the returned iterator yields values
     |      directly from nbunch, becoming exhausted when nbunch is exhausted.
     |
     |      To test whether nbunch is a single node, one can use
     |      "if nbunch in self:", even after processing with this routine.
     |
     |      If nbunch is not a node or a (possibly empty) sequence/iterator
     |      or None, a :exc:`NetworkXError` is raised.  Also, if any object in
     |      nbunch is not hashable, a :exc:`NetworkXError` is raised.
     |
     |  neighbors(self, n)
     |      Returns an iterator over all neighbors of node n.
     |
     |      This is identical to `iter(G[n])`
     |
     |      Parameters
     |      ----------
     |      n : node
     |         A node in the graph
     |
     |      Returns
     |      -------
     |      neighbors : iterator
     |          An iterator over all neighbors of node n
     |
     |      Raises
     |      ------
     |      NetworkXError
     |          If the node n is not in the graph.
     |
     |      Examples
     |      --------
     |      >>> G = nx.path_graph(4)  # or DiGraph, MultiGraph, MultiDiGraph, etc
     |      >>> [n for n in G.neighbors(0)]
     |      [1]
     |
     |      Notes
     |      -----
     |      Alternate ways to access the neighbors are ``G.adj[n]`` or ``G[n]``:
     |
     |      >>> G = nx.Graph()  # or DiGraph, MultiGraph, MultiDiGraph, etc
     |      >>> G.add_edge("a", "b", weight=7)
     |      >>> G["a"]
     |      AtlasView({'b': {'weight': 7}})
     |      >>> G = nx.path_graph(4)
     |      >>> [n for n in G[0]]
     |      [1]
     |
     |  nodes = <functools.cached_property object>
     |      A NodeView of the Graph as G.nodes or G.nodes().
     |
     |      Can be used as `G.nodes` for data lookup and for set-like operations.
     |      Can also be used as `G.nodes(data='color', default=None)` to return a
     |      NodeDataView which reports specific node data but no set operations.
     |      It presents a dict-like interface as well with `G.nodes.items()`
     |      iterating over `(node, nodedata)` 2-tuples and `G.nodes[3]['foo']`
     |      providing the value of the `foo` attribute for node `3`. In addition,
     |      a view `G.nodes.data('foo')` provides a dict-like interface to the
     |      `foo` attribute of each node. `G.nodes.data('foo', default=1)`
     |      provides a default for nodes that do not have attribute `foo`.
     |
     |      Parameters
     |      ----------
     |      data : string or bool, optional (default=False)
     |          The node attribute returned in 2-tuple (n, ddict[data]).
     |          If True, return entire node attribute dict as (n, ddict).
     |          If False, return just the nodes n.
     |
     |      default : value, optional (default=None)
     |          Value used for nodes that don't have the requested attribute.
     |          Only relevant if data is not True or False.
     |
     |      Returns
     |      -------
     |      NodeView
     |          Allows set-like operations over the nodes as well as node
     |          attribute dict lookup and calling to get a NodeDataView.
     |          A NodeDataView iterates over `(n, data)` and has no set operations.
     |          A NodeView iterates over `n` and includes set operations.
     |
     |          When called, if data is False, an iterator over nodes.
     |          Otherwise an iterator of 2-tuples (node, attribute value)
     |          where the attribute is specified in `data`.
     |          If data is True then the attribute becomes the
     |          entire data dictionary.
     |
     |      Notes
     |      -----
     |      If your node data is not needed, it is simpler and equivalent
     |      to use the expression ``for n in G``, or ``list(G)``.
     |
     |      Examples
     |      --------
     |      There are two simple ways of getting a list of all nodes in the graph:
     |
     |      >>> G = nx.path_graph(3)
     |      >>> list(G.nodes)
     |      [0, 1, 2]
     |      >>> list(G)
     |      [0, 1, 2]
     |
     |      To get the node data along with the nodes:
     |
     |      >>> G.add_node(1, time="5pm")
     |      >>> G.nodes[0]["foo"] = "bar"
     |      >>> list(G.nodes(data=True))
     |      [(0, {'foo': 'bar'}), (1, {'time': '5pm'}), (2, {})]
     |      >>> list(G.nodes.data())
     |      [(0, {'foo': 'bar'}), (1, {'time': '5pm'}), (2, {})]
     |
     |      >>> list(G.nodes(data="foo"))
     |      [(0, 'bar'), (1, None), (2, None)]
     |      >>> list(G.nodes.data("foo"))
     |      [(0, 'bar'), (1, None), (2, None)]
     |
     |      >>> list(G.nodes(data="time"))
     |      [(0, None), (1, '5pm'), (2, None)]
     |      >>> list(G.nodes.data("time"))
     |      [(0, None), (1, '5pm'), (2, None)]
     |
     |      >>> list(G.nodes(data="time", default="Not Available"))
     |      [(0, 'Not Available'), (1, '5pm'), (2, 'Not Available')]
     |      >>> list(G.nodes.data("time", default="Not Available"))
     |      [(0, 'Not Available'), (1, '5pm'), (2, 'Not Available')]
     |
     |      If some of your nodes have an attribute and the rest are assumed
     |      to have a default attribute value you can create a dictionary
     |      from node/attribute pairs using the `default` keyword argument
     |      to guarantee the value is never None::
     |
     |          >>> G = nx.Graph()
     |          >>> G.add_node(0)
     |          >>> G.add_node(1, weight=2)
     |          >>> G.add_node(2, weight=3)
     |          >>> dict(G.nodes(data="weight", default=1))
     |          {0: 1, 1: 2, 2: 3}
     |
     |  number_of_edges(self, u=None, v=None)
     |      Returns the number of edges between two nodes.
     |
     |      Parameters
     |      ----------
     |      u, v : nodes, optional (default=all edges)
     |          If u and v are specified, return the number of edges between
     |          u and v. Otherwise return the total number of all edges.
     |
     |      Returns
     |      -------
     |      nedges : int
     |          The number of edges in the graph.  If nodes `u` and `v` are
     |          specified return the number of edges between those nodes. If
     |          the graph is directed, this only returns the number of edges
     |          from `u` to `v`.
     |
     |      See Also
     |      --------
     |      size
     |
     |      Examples
     |      --------
     |      For undirected graphs, this method counts the total number of
     |      edges in the graph:
     |
     |      >>> G = nx.path_graph(4)
     |      >>> G.number_of_edges()
     |      3
     |
     |      If you specify two nodes, this counts the total number of edges
     |      joining the two nodes:
     |
     |      >>> G.number_of_edges(0, 1)
     |      1
     |
     |      For directed graphs, this method can count the total number of
     |      directed edges from `u` to `v`:
     |
     |      >>> G = nx.DiGraph()
     |      >>> G.add_edge(0, 1)
     |      >>> G.add_edge(1, 0)
     |      >>> G.number_of_edges(0, 1)
     |      1
     |
     |  number_of_nodes(self)
     |      Returns the number of nodes in the graph.
     |
     |      Returns
     |      -------
     |      nnodes : int
     |          The number of nodes in the graph.
     |
     |      See Also
     |      --------
     |      order: identical method
     |      __len__: identical method
     |
     |      Examples
     |      --------
     |      >>> G = nx.path_graph(3)  # or DiGraph, MultiGraph, MultiDiGraph, etc
     |      >>> G.number_of_nodes()
     |      3
     |
     |  order(self)
     |      Returns the number of nodes in the graph.
     |
     |      Returns
     |      -------
     |      nnodes : int
     |          The number of nodes in the graph.
     |
     |      See Also
     |      --------
     |      number_of_nodes: identical method
     |      __len__: identical method
     |
     |      Examples
     |      --------
     |      >>> G = nx.path_graph(3)  # or DiGraph, MultiGraph, MultiDiGraph, etc
     |      >>> G.order()
     |      3
     |
     |  remove_edge(self, u, v)
     |      Remove the edge between u and v.
     |
     |      Parameters
     |      ----------
     |      u, v : nodes
     |          Remove the edge between nodes u and v.
     |
     |      Raises
     |      ------
     |      NetworkXError
     |          If there is not an edge between u and v.
     |
     |      See Also
     |      --------
     |      remove_edges_from : remove a collection of edges
     |
     |      Examples
     |      --------
     |      >>> G = nx.path_graph(4)  # or DiGraph, etc
     |      >>> G.remove_edge(0, 1)
     |      >>> e = (1, 2)
     |      >>> G.remove_edge(*e)  # unpacks e from an edge tuple
     |      >>> e = (2, 3, {"weight": 7})  # an edge with attribute data
     |      >>> G.remove_edge(*e[:2])  # select first part of edge tuple
     |
     |  remove_edges_from(self, ebunch)
     |      Remove all edges specified in ebunch.
     |
     |      Parameters
     |      ----------
     |      ebunch: list or container of edge tuples
     |          Each edge given in the list or container will be removed
     |          from the graph. The edges can be:
     |
     |              - 2-tuples (u, v) edge between u and v.
     |              - 3-tuples (u, v, k) where k is ignored.
     |
     |      See Also
     |      --------
     |      remove_edge : remove a single edge
     |
     |      Notes
     |      -----
     |      Will fail silently if an edge in ebunch is not in the graph.
     |
     |      Examples
     |      --------
     |      >>> G = nx.path_graph(4)  # or DiGraph, MultiGraph, MultiDiGraph, etc
     |      >>> ebunch = [(1, 2), (2, 3)]
     |      >>> G.remove_edges_from(ebunch)
     |
     |  remove_node(self, n)
     |      Remove node n.
     |
     |      Removes the node n and all adjacent edges.
     |      Attempting to remove a nonexistent node will raise an exception.
     |
     |      Parameters
     |      ----------
     |      n : node
     |         A node in the graph
     |
     |      Raises
     |      ------
     |      NetworkXError
     |         If n is not in the graph.
     |
     |      See Also
     |      --------
     |      remove_nodes_from
     |
     |      Examples
     |      --------
     |      >>> G = nx.path_graph(3)  # or DiGraph, MultiGraph, MultiDiGraph, etc
     |      >>> list(G.edges)
     |      [(0, 1), (1, 2)]
     |      >>> G.remove_node(1)
     |      >>> list(G.edges)
     |      []
     |
     |  remove_nodes_from(self, nodes)
     |      Remove multiple nodes.
     |
     |      Parameters
     |      ----------
     |      nodes : iterable container
     |          A container of nodes (list, dict, set, etc.).  If a node
     |          in the container is not in the graph it is silently
     |          ignored.
     |
     |      See Also
     |      --------
     |      remove_node
     |
     |      Notes
     |      -----
     |      When removing nodes from an iterator over the graph you are changing,
     |      a `RuntimeError` will be raised with message:
     |      `RuntimeError: dictionary changed size during iteration`. This
     |      happens when the graph's underlying dictionary is modified during
     |      iteration. To avoid this error, evaluate the iterator into a separate
     |      object, e.g. by using `list(iterator_of_nodes)`, and pass this
     |      object to `G.remove_nodes_from`.
     |
     |      Examples
     |      --------
     |      >>> G = nx.path_graph(3)  # or DiGraph, MultiGraph, MultiDiGraph, etc
     |      >>> e = list(G.nodes)
     |      >>> e
     |      [0, 1, 2]
     |      >>> G.remove_nodes_from(e)
     |      >>> list(G.nodes)
     |      []
     |
     |      Evaluate an iterator over a graph if using it to modify the same graph
     |
     |      >>> G = nx.Graph([(0, 1), (1, 2), (3, 4)])
     |      >>> # this command will fail, as the graph's dict is modified during iteration
     |      >>> # G.remove_nodes_from(n for n in G.nodes if n < 2)
     |      >>> # this command will work, since the dictionary underlying graph is not modified
     |      >>> G.remove_nodes_from(list(n for n in G.nodes if n < 2))
     |
     |  size(self, weight=None)
     |      Returns the number of edges or total of all edge weights.
     |
     |      Parameters
     |      ----------
     |      weight : string or None, optional (default=None)
     |          The edge attribute that holds the numerical value used
     |          as a weight. If None, then each edge has weight 1.
     |
     |      Returns
     |      -------
     |      size : numeric
     |          The number of edges or
     |          (if weight keyword is provided) the total weight sum.
     |
     |          If weight is None, returns an int. Otherwise a float
     |          (or more general numeric if the weights are more general).
     |
     |      See Also
     |      --------
     |      number_of_edges
     |
     |      Examples
     |      --------
     |      >>> G = nx.path_graph(4)  # or DiGraph, MultiGraph, MultiDiGraph, etc
     |      >>> G.size()
     |      3
     |
     |      >>> G = nx.Graph()  # or DiGraph, MultiGraph, MultiDiGraph, etc
     |      >>> G.add_edge("a", "b", weight=2)
     |      >>> G.add_edge("b", "c", weight=4)
     |      >>> G.size()
     |      2
     |      >>> G.size(weight="weight")
     |      6.0
     |
     |  subgraph(self, nodes)
     |      Returns a SubGraph view of the subgraph induced on `nodes`.
     |
     |      The induced subgraph of the graph contains the nodes in `nodes`
     |      and the edges between those nodes.
     |
     |      Parameters
     |      ----------
     |      nodes : list, iterable
     |          A container of nodes which will be iterated through once.
     |
     |      Returns
     |      -------
     |      G : SubGraph View
     |          A subgraph view of the graph. The graph structure cannot be
     |          changed but node/edge attributes can and are shared with the
     |          original graph.
     |
     |      Notes
     |      -----
     |      The graph, edge and node attributes are shared with the original graph.
     |      Changes to the graph structure is ruled out by the view, but changes
     |      to attributes are reflected in the original graph.
     |
     |      To create a subgraph with its own copy of the edge/node attributes use:
     |      G.subgraph(nodes).copy()
     |
     |      For an inplace reduction of a graph to a subgraph you can remove nodes:
     |      G.remove_nodes_from([n for n in G if n not in set(nodes)])
     |
     |      Subgraph views are sometimes NOT what you want. In most cases where
     |      you want to do more than simply look at the induced edges, it makes
     |      more sense to just create the subgraph as its own graph with code like:
     |
     |      ::
     |
     |          # Create a subgraph SG based on a (possibly multigraph) G
     |          SG = G.__class__()
     |          SG.add_nodes_from((n, G.nodes[n]) for n in largest_wcc)
     |          if SG.is_multigraph():
     |              SG.add_edges_from(
     |                  (n, nbr, key, d)
     |                  for n, nbrs in G.adj.items()
     |                  if n in largest_wcc
     |                  for nbr, keydict in nbrs.items()
     |                  if nbr in largest_wcc
     |                  for key, d in keydict.items()
     |              )
     |          else:
     |              SG.add_edges_from(
     |                  (n, nbr, d)
     |                  for n, nbrs in G.adj.items()
     |                  if n in largest_wcc
     |                  for nbr, d in nbrs.items()
     |                  if nbr in largest_wcc
     |              )
     |          SG.graph.update(G.graph)
     |
     |      Subgraphs are not guaranteed to preserve the order of nodes or edges
     |      as they appear in the original graph. For example:
     |
     |      >>> G = nx.Graph()
     |      >>> G.add_nodes_from(reversed(range(10)))
     |      >>> list(G)
     |      [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
     |      >>> list(G.subgraph([1, 3, 2]))
     |      [1, 2, 3]
     |
     |      Examples
     |      --------
     |      >>> G = nx.path_graph(4)  # or DiGraph, MultiGraph, MultiDiGraph, etc
     |      >>> H = G.subgraph([0, 1, 2])
     |      >>> list(H.edges)
     |      [(0, 1), (1, 2)]
     |
     |  to_directed(self, as_view=False)
     |      Returns a directed representation of the graph.
     |
     |      Returns
     |      -------
     |      G : DiGraph
     |          A directed graph with the same name, same nodes, and with
     |          each edge (u, v, data) replaced by two directed edges
     |          (u, v, data) and (v, u, data).
     |
     |      Notes
     |      -----
     |      This returns a "deepcopy" of the edge, node, and
     |      graph attributes which attempts to completely copy
     |      all of the data and references.
     |
     |      This is in contrast to the similar D=DiGraph(G) which returns a
     |      shallow copy of the data.
     |
     |      See the Python copy module for more information on shallow
     |      and deep copies, https://docs.python.org/3/library/copy.html.
     |
     |      Warning: If you have subclassed Graph to use dict-like objects
     |      in the data structure, those changes do not transfer to the
     |      DiGraph created by this method.
     |
     |      Examples
     |      --------
     |      >>> G = nx.Graph()  # or MultiGraph, etc
     |      >>> G.add_edge(0, 1)
     |      >>> H = G.to_directed()
     |      >>> list(H.edges)
     |      [(0, 1), (1, 0)]
     |
     |      If already directed, return a (deep) copy
     |
     |      >>> G = nx.DiGraph()  # or MultiDiGraph, etc
     |      >>> G.add_edge(0, 1)
     |      >>> H = G.to_directed()
     |      >>> list(H.edges)
     |      [(0, 1)]
     |
     |  to_directed_class(self)
     |      Returns the class to use for empty directed copies.
     |
     |      If you subclass the base classes, use this to designate
     |      what directed class to use for `to_directed()` copies.
     |
     |  to_undirected(self, as_view=False)
     |      Returns an undirected copy of the graph.
     |
     |      Parameters
     |      ----------
     |      as_view : bool (optional, default=False)
     |        If True return a view of the original undirected graph.
     |
     |      Returns
     |      -------
     |      G : Graph/MultiGraph
     |          A deepcopy of the graph.
     |
     |      See Also
     |      --------
     |      Graph, copy, add_edge, add_edges_from
     |
     |      Notes
     |      -----
     |      This returns a "deepcopy" of the edge, node, and
     |      graph attributes which attempts to completely copy
     |      all of the data and references.
     |
     |      This is in contrast to the similar `G = nx.DiGraph(D)` which returns a
     |      shallow copy of the data.
     |
     |      See the Python copy module for more information on shallow
     |      and deep copies, https://docs.python.org/3/library/copy.html.
     |
     |      Warning: If you have subclassed DiGraph to use dict-like objects
     |      in the data structure, those changes do not transfer to the
     |      Graph created by this method.
     |
     |      Examples
     |      --------
     |      >>> G = nx.path_graph(2)  # or MultiGraph, etc
     |      >>> H = G.to_directed()
     |      >>> list(H.edges)
     |      [(0, 1), (1, 0)]
     |      >>> G2 = H.to_undirected()
     |      >>> list(G2.edges)
     |      [(0, 1)]
     |
     |  to_undirected_class(self)
     |      Returns the class to use for empty undirected copies.
     |
     |      If you subclass the base classes, use this to designate
     |      what directed class to use for `to_directed()` copies.
     |
     |  update(self, edges=None, nodes=None)
     |      Update the graph using nodes/edges/graphs as input.
     |
     |      Like dict.update, this method takes a graph as input, adding the
     |      graph's nodes and edges to this graph. It can also take two inputs:
     |      edges and nodes. Finally it can take either edges or nodes.
     |      To specify only nodes the keyword `nodes` must be used.
     |
     |      The collections of edges and nodes are treated similarly to
     |      the add_edges_from/add_nodes_from methods. When iterated, they
     |      should yield 2-tuples (u, v) or 3-tuples (u, v, datadict).
     |
     |      Parameters
     |      ----------
     |      edges : Graph object, collection of edges, or None
     |          The first parameter can be a graph or some edges. If it has
     |          attributes `nodes` and `edges`, then it is taken to be a
     |          Graph-like object and those attributes are used as collections
     |          of nodes and edges to be added to the graph.
     |          If the first parameter does not have those attributes, it is
     |          treated as a collection of edges and added to the graph.
     |          If the first argument is None, no edges are added.
     |      nodes : collection of nodes, or None
     |          The second parameter is treated as a collection of nodes
     |          to be added to the graph unless it is None.
     |          If `edges is None` and `nodes is None` an exception is raised.
     |          If the first parameter is a Graph, then `nodes` is ignored.
     |
     |      Examples
     |      --------
     |      >>> G = nx.path_graph(5)
     |      >>> G.update(nx.complete_graph(range(4, 10)))
     |      >>> from itertools import combinations
     |      >>> edges = (
     |      ...     (u, v, {"power": u * v})
     |      ...     for u, v in combinations(range(10, 20), 2)
     |      ...     if u * v < 225
     |      ... )
     |      >>> nodes = [1000]  # for singleton, use a container
     |      >>> G.update(edges, nodes)
     |
     |      Notes
     |      -----
     |      It you want to update the graph using an adjacency structure
     |      it is straightforward to obtain the edges/nodes from adjacency.
     |      The following examples provide common cases, your adjacency may
     |      be slightly different and require tweaks of these examples::
     |
     |      >>> # dict-of-set/list/tuple
     |      >>> adj = {1: {2, 3}, 2: {1, 3}, 3: {1, 2}}
     |      >>> e = [(u, v) for u, nbrs in adj.items() for v in nbrs]
     |      >>> G.update(edges=e, nodes=adj)
     |
     |      >>> DG = nx.DiGraph()
     |      >>> # dict-of-dict-of-attribute
     |      >>> adj = {1: {2: 1.3, 3: 0.7}, 2: {1: 1.4}, 3: {1: 0.7}}
     |      >>> e = [
     |      ...     (u, v, {"weight": d})
     |      ...     for u, nbrs in adj.items()
     |      ...     for v, d in nbrs.items()
     |      ... ]
     |      >>> DG.update(edges=e, nodes=adj)
     |
     |      >>> # dict-of-dict-of-dict
     |      >>> adj = {1: {2: {"weight": 1.3}, 3: {"color": 0.7, "weight": 1.2}}}
     |      >>> e = [
     |      ...     (u, v, {"weight": d})
     |      ...     for u, nbrs in adj.items()
     |      ...     for v, d in nbrs.items()
     |      ... ]
     |      >>> DG.update(edges=e, nodes=adj)
     |
     |      >>> # predecessor adjacency (dict-of-set)
     |      >>> pred = {1: {2, 3}, 2: {3}, 3: {3}}
     |      >>> e = [(v, u) for u, nbrs in pred.items() for v in nbrs]
     |
     |      >>> # MultiGraph dict-of-dict-of-dict-of-attribute
     |      >>> MDG = nx.MultiDiGraph()
     |      >>> adj = {
     |      ...     1: {2: {0: {"weight": 1.3}, 1: {"weight": 1.2}}},
     |      ...     3: {2: {0: {"weight": 0.7}}},
     |      ... }
     |      >>> e = [
     |      ...     (u, v, ekey, d)
     |      ...     for u, nbrs in adj.items()
     |      ...     for v, keydict in nbrs.items()
     |      ...     for ekey, d in keydict.items()
     |      ... ]
     |      >>> MDG.update(edges=e)
     |
     |      See Also
     |      --------
     |      add_edges_from: add multiple edges to a graph
     |      add_nodes_from: add multiple nodes to a graph
     |
     |  ----------------------------------------------------------------------
     |  Data descriptors inherited from networkx.classes.graph.Graph:
     |
     |  __dict__
     |      dictionary for instance variables
     |
     |  __weakref__
     |      list of weak references to the object
     |
     |  name
     |      String identifier of the graph.
     |
     |      This graph attribute appears in the attribute dict G.graph
     |      keyed by the string `"name"`. as well as an attribute (technically
     |      a property) `G.name`. This is entirely user controlled.
     |
     |  ----------------------------------------------------------------------
     |  Data and other attributes inherited from networkx.classes.graph.Graph:
     |
     |  __networkx_backend__ = 'networkx'
     |
     |  adjlist_inner_dict_factory = <class 'dict'>
     |      dict() -> new empty dictionary
     |      dict(mapping) -> new dictionary initialized from a mapping object's
     |          (key, value) pairs
     |      dict(iterable) -> new dictionary initialized as if via:
     |          d = {}
     |          for k, v in iterable:
     |              d[k] = v
     |      dict(**kwargs) -> new dictionary initialized with the name=value pairs
     |          in the keyword argument list.  For example:  dict(one=1, two=2)
     |
     |
     |  adjlist_outer_dict_factory = <class 'dict'>
     |      dict() -> new empty dictionary
     |      dict(mapping) -> new dictionary initialized from a mapping object's
     |          (key, value) pairs
     |      dict(iterable) -> new dictionary initialized as if via:
     |          d = {}
     |          for k, v in iterable:
     |              d[k] = v
     |      dict(**kwargs) -> new dictionary initialized with the name=value pairs
     |          in the keyword argument list.  For example:  dict(one=1, two=2)
     |
     |
     |  edge_attr_dict_factory = <class 'dict'>
     |      dict() -> new empty dictionary
     |      dict(mapping) -> new dictionary initialized from a mapping object's
     |          (key, value) pairs
     |      dict(iterable) -> new dictionary initialized as if via:
     |          d = {}
     |          for k, v in iterable:
     |              d[k] = v
     |      dict(**kwargs) -> new dictionary initialized with the name=value pairs
     |          in the keyword argument list.  For example:  dict(one=1, two=2)
     |
     |
     |  graph_attr_dict_factory = <class 'dict'>
     |      dict() -> new empty dictionary
     |      dict(mapping) -> new dictionary initialized from a mapping object's
     |          (key, value) pairs
     |      dict(iterable) -> new dictionary initialized as if via:
     |          d = {}
     |          for k, v in iterable:
     |              d[k] = v
     |      dict(**kwargs) -> new dictionary initialized with the name=value pairs
     |          in the keyword argument list.  For example:  dict(one=1, two=2)
     |
     |
     |  node_attr_dict_factory = <class 'dict'>
     |      dict() -> new empty dictionary
     |      dict(mapping) -> new dictionary initialized from a mapping object's
     |          (key, value) pairs
     |      dict(iterable) -> new dictionary initialized as if via:
     |          d = {}
     |          for k, v in iterable:
     |              d[k] = v
     |      dict(**kwargs) -> new dictionary initialized with the name=value pairs
     |          in the keyword argument list.  For example:  dict(one=1, two=2)
     |
     |
     |  node_dict_factory = <class 'dict'>
     |      dict() -> new empty dictionary
     |      dict(mapping) -> new dictionary initialized from a mapping object's
     |          (key, value) pairs
     |      dict(iterable) -> new dictionary initialized as if via:
     |          d = {}
     |          for k, v in iterable:
     |              d[k] = v
     |      dict(**kwargs) -> new dictionary initialized with the name=value pairs
     |          in the keyword argument list.  For example:  dict(one=1, two=2)

    class SWCParseResult(builtins.object)
     |  SWCParseResult(records: 'Dict[int, SWCRecord]', reconnections: 'List[Tuple[int, int]]', header: 'List[str]') -> None
     |
     |  Parsed SWC content.
     |
     |  Methods defined here:
     |
     |  __delattr__(self, name)
     |      Implement delattr(self, name).
     |
     |  __eq__(self, other)
     |      Return self==value.
     |
     |  __hash__(self)
     |      Return hash(self).
     |
     |  __init__(self, records: 'Dict[int, SWCRecord]', reconnections: 'List[Tuple[int, int]]', header: 'List[str]') -> None
     |      Initialize self.  See help(type(self)) for accurate signature.
     |
     |  __repr__(self) -> 'str'
     |      Return repr(self).
     |
     |  __setattr__(self, name, value)
     |      Implement setattr(self, name, value).
     |
     |  __str__(self) -> 'str'
     |      Return str(self).
     |
     |  ----------------------------------------------------------------------
     |  Data descriptors defined here:
     |
     |  __dict__
     |      dictionary for instance variables
     |
     |  __weakref__
     |      list of weak references to the object
     |
     |  ----------------------------------------------------------------------
     |  Data and other attributes defined here:
     |
     |  __annotations__ = {'header': 'List[str]', 'reconnections': 'List[Tuple...
     |
     |  __dataclass_fields__ = {'header': Field(name='header',type='List[str]'...
     |
     |  __dataclass_params__ = _DataclassParams(init=True,repr=True,eq=True,or...
     |
     |  __match_args__ = ('records', 'reconnections', 'header')

    class SWCRecord(builtins.object)
     |  SWCRecord(n: 'int', t: 'int', x: 'float', y: 'float', z: 'float', r: 'float', parent: 'int', line: 'int') -> None
     |
     |  One SWC row.
     |
     |  Attributes
     |  ----------
     |  n: int
     |      Node id (unique within file)
     |  t: int
     |      Tag index
     |  x, y, z: float
     |      Coordinates (usually micrometers)
     |  r: float
     |      Radius
     |  parent: int
     |      Parent id; -1 indicates root
     |  line: int
     |      1-based line number in the source file/string
     |
     |  Methods defined here:
     |
     |  __delattr__(self, name)
     |      Implement delattr(self, name).
     |
     |  __eq__(self, other)
     |      Return self==value.
     |
     |  __hash__(self)
     |      Return hash(self).
     |
     |  __init__(self, n: 'int', t: 'int', x: 'float', y: 'float', z: 'float', r: 'float', parent: 'int', line: 'int') -> None
     |      Initialize self.  See help(type(self)) for accurate signature.
     |
     |  __repr__(self)
     |      Return repr(self).
     |
     |  __setattr__(self, name, value)
     |      Implement setattr(self, name, value).
     |
     |  ----------------------------------------------------------------------
     |  Data descriptors defined here:
     |
     |  __dict__
     |      dictionary for instance variables
     |
     |  __weakref__
     |      list of weak references to the object
     |
     |  ----------------------------------------------------------------------
     |  Data and other attributes defined here:
     |
     |  __annotations__ = {'line': 'int', 'n': 'int', 'parent': 'int', 'r': 'f...
     |
     |  __dataclass_fields__ = {'line': Field(name='line',type='int',default=<...
     |
     |  __dataclass_params__ = _DataclassParams(init=True,repr=True,eq=True,or...
     |
     |  __match_args__ = ('n', 't', 'x', 'y', 'z', 'r', 'parent', 'line')

    class Segment(builtins.object)
     |  Segment(a: 'Point3', b: 'Point3', ra: 'float', rb: 'float', tag: 'int') -> None
     |
     |  Oriented frustum segment between endpoints `a` and `b`.
     |
     |  Attributes
     |  ----------
     |  a, b: Point3
     |      Endpoints in model/world coordinates.
     |  ra, rb: float
     |      Radii at `a` and `b`.
     |
     |  Methods defined here:
     |
     |  __delattr__(self, name)
     |      Implement delattr(self, name).
     |
     |  __eq__(self, other)
     |      Return self==value.
     |
     |  __hash__(self)
     |      Return hash(self).
     |
     |  __init__(self, a: 'Point3', b: 'Point3', ra: 'float', rb: 'float', tag: 'int') -> None
     |      Initialize self.  See help(type(self)) for accurate signature.
     |
     |  __repr__(self)
     |      Return repr(self).
     |
     |  __setattr__(self, name, value)
     |      Implement setattr(self, name, value).
     |
     |  length(self) -> 'float'
     |
     |  midpoint(self) -> 'Point3'
     |
     |  scale(self, scalar: 'float') -> "'Segment'"
     |      Return a new `Segment` uniformly scaled by `scalar` (positions and radii).
     |
     |  vector(self) -> 'Vec3'
     |
     |  ----------------------------------------------------------------------
     |  Data descriptors defined here:
     |
     |  __dict__
     |      dictionary for instance variables
     |
     |  __weakref__
     |      list of weak references to the object
     |
     |  ----------------------------------------------------------------------
     |  Data and other attributes defined here:
     |
     |  __annotations__ = {'a': 'Point3', 'b': 'Point3', 'ra': 'float', 'rb': ...
     |
     |  __dataclass_fields__ = {'a': Field(name='a',type='Point3',default=<dat...
     |
     |  __dataclass_params__ = _DataclassParams(init=True,repr=True,eq=True,or...
     |
     |  __match_args__ = ('a', 'b', 'ra', 'rb', 'tag')

FUNCTIONS
    apply_layout(fig, *, title: 'str | None' = None) -> 'None'
        Apply global layout defaults to a Plotly figure in-place.

    batch_frusta(segments: 'Iterable[Segment]', *, sides: 'int' = 16, end_caps: 'bool' = False) -> 'Tuple[List[Point3], List[Face]]'
        Batch multiple frusta into a single mesh.

        Returns a concatenated list of `vertices` and `faces` with the proper index offsets.

    frustum_mesh(seg: 'Segment', *, sides: 'int' = 16, end_caps: 'bool' = False) -> 'Tuple[List[Point3], List[Face]]'
        Generate a frustum mesh for a single `Segment`.

        Returns
        -------
        (vertices, faces):
            - vertices: List[Point3]
            - faces: List[Face], each = (i, j, k) indexing into `vertices`

    get_config() -> 'VizConfig'
        Return the current visualization configuration (live object).

    parse_swc(source: 'Union[str, os.PathLike, Iterable[str], io.TextIOBase]', *, strict: 'bool' = True, validate_reconnections: 'bool' = True, float_tol: 'float' = 1e-09) -> 'SWCParseResult'
        Parse an SWC file or text stream.

        Parameters
        ----------
        source
            Path to an SWC file, a file-like object, an iterable of lines, or a string
            containing SWC content.
        strict
            If True, enforce 7-column rows and validate parent references exist.
        validate_reconnections
            If True, ensure reconnection node pairs share identical (x, y, z, r).
        float_tol
            Tolerance used when comparing floating-point coordinates/radii.

        Returns
        -------
        SWCParseResult
            Parsed records, reconnection pairs, and header lines.

        Raises
        ------
        ValueError
            If parsing or validation fails.
        FileNotFoundError
            If a string path is provided that does not exist.

    plot_centroid(swc_model: 'SWCModel', *, marker_size: 'float' = 2.0, line_width: 'float' = 2.0, show_nodes: 'bool' = True, title: 'str | None' = None) -> 'go.Figure'
        Plot centroid skeleton from an `SWCModel`.

        Edges are drawn as line segments in 3D using Scatter3d.

    plot_frusta(frusta: 'FrustaSet', *, color: 'str' = 'lightblue', opacity: 'float' = 0.8, flatshading: 'bool' = True, radius_scale: 'float' = 1.0, tag_colors: 'dict[int, str] | None' = None, title: 'str | None' = None) -> 'go.Figure'
        Plot a FrustaSet as a Mesh3d figure.

        Parameters
        ----------
        frusta: FrustaSet
            Batched frusta mesh to render.
        color: str
            Mesh color.
        opacity: float
            Mesh opacity.
        flatshading: bool
            Whether to enable flat shading.
        radius_scale: float
            Uniform scale applied to all segment radii before meshing (1.0 = no change).
        tag_colors: dict[int, str] | None
            Optional mapping {tag: color}. If provided, each frustum segment is colored
            uniformly according to its tag (fallback to `color` if a tag is missing).

    plot_frusta_slider(frusta: 'FrustaSet', *, color: 'str' = 'lightblue', opacity: 'float' = 0.8, flatshading: 'bool' = True, tag_colors: 'dict[int, str] | None' = None, min_scale: 'float' = 0.0, max_scale: 'float' = 1.0, steps: 'int' = 21, title: 'str | None' = None) -> 'go.Figure'
        Interactive slider (0..1 default) controlling uniform `radius_scale`.

        Precomputes frames at evenly spaced scales between `min_scale` and `max_scale`.

    plot_frusta_timeseries(frusta: 'FrustaSet', values: 'Sequence[Sequence[float]]', *, colorscale: 'str | list' = 'Viridis', cmin: 'Optional[float]' = None, cmax: 'Optional[float]' = None, opacity: 'float' = 0.8, flatshading: 'bool' = True, radius_scale: 'float' = 1.0, fps: 'int' = 10, title: 'str | None' = None) -> 'go.Figure'
        Animate per-segment values over time by coloring frusta faces.

        Parameters
        ----------
        frusta: FrustaSet
            Batched frusta mesh. Optionally scaled via `radius_scale` before rendering.
        values: Sequence[Sequence[float]]
            Time series V_i(t) shaped [T][N], where N = `frusta.segment_count`.
            Each time step provides one scalar per segment in the current order.
        colorscale: str | list
            Plotly colorscale for mapping intensities.
        cmin, cmax: float | None
            Fixed color range. If omitted, inferred from the data across all frames.
        opacity: float
            Mesh opacity.
        flatshading: bool
            Enable flat shading on the mesh.
        radius_scale: float
            Uniform radius scaling applied before meshing (1.0 = no change).
        fps: int
            Playback frame rate for the Play button.

        Returns
        -------
        plotly.graph_objects.Figure
            A figure with a Mesh3d trace and animation frames, plus a slider and play/pause controls.

    plot_frusta_with_centroid(swc_model: 'SWCModel', frusta: 'FrustaSet', *, color: 'str' = 'lightblue', opacity: 'float' = 0.8, flatshading: 'bool' = True, radius_scale: 'float' = 1.0, tag_colors: 'dict[int, str] | None' = None, centroid_color: 'str' = '#1f77b4', centroid_line_width: 'float' = 2.0, show_nodes: 'bool' = False, node_size: 'float' = 2.0, title: 'str | None' = None) -> 'go.Figure'
        Overlay frusta mesh with centroid skeleton from an `SWCModel`.

        Parameters mirror `plot_centroid` and `plot_frusta` with an extra `radius_scale`.

    plot_model(*, swc_model: 'SWCModel | None' = None, frusta: 'FrustaSet | None' = None, show_frusta: 'bool' = True, show_centroid: 'bool' = True, title: 'str | None' = None, sides: 'int' = 16, end_caps: 'bool' = False, color: 'str' = 'lightblue', opacity: 'float' = 0.8, flatshading: 'bool' = True, tag_colors: 'dict[int, str] | None' = None, radius_scale: 'float' = 1.0, slider: 'bool' = False, min_scale: 'float' = 0.0, max_scale: 'float' = 1.0, steps: 'int' = 21, centroid_color: 'str' = '#1f77b4', centroid_line_width: 'float' = 2.0, show_nodes: 'bool' = False, node_size: 'float' = 2.0, point_set: 'PointSet | None' = None, point_size: 'float' = 1.0, point_color: 'str' = '#d62728') -> 'go.Figure'
        Master visualization combining centroid, frusta, slider, and overlay points.

        - If `frusta` is not provided and `gm` is, a `FrustaSet` is built from `gm`.
        - If `slider=True` and `show_frusta=True`, a Plotly slider controls `radius_scale`.
        - `points` overlays arbitrary xyz positions as small markers.

    set_config(**kwargs: 'Any') -> 'None'
        Update global visualization configuration.

        Example:
            set_config(width=800, height=600, scene_aspectmode="cube")

DATA
    __all__ = ['SWCRecord', 'SWCParseResult', 'parse_swc', 'SWCModel', 'Se...

FILE
    c:\users\mainuser\documents\repos\mascaf\.venv\lib\site-packages\swctools\__init__.py


~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Class: swctools.FrustaSet
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Python Library Documentation: class FrustaSet in module swctools.geometry

class FrustaSet(builtins.object)
 |  FrustaSet(vertices: 'List[Point3]', faces: 'List[Face]', sides: 'int', end_caps: 'bool', segment_count: 'int', edge_count: 'int', segments: 'List[Segment]', edge_uvs: 'Optional[List[Tuple[int, int]]]' = None) -> None
 |
 |  A batched frusta mesh derived from a `SWCModel`.
 |
 |  Attributes
 |  ----------
 |  vertices: List[Point3]
 |      Concatenated vertices for all frusta.
 |  faces: List[Face]
 |      Triangular faces indexing into `vertices`.
 |  sides: int
 |      Circumferential resolution used per frustum.
 |  end_caps: bool
 |      Whether end caps were included during construction.
 |  segment_count: int
 |      Number of segments used (one per graph edge).
 |  edge_count: int
 |      Alias for `segment_count` for clarity.
 |  segments: List[Segment]
 |      The segments used to construct the frusta.
 |  edge_uvs: Optional[List[Tuple[int, int]]]
 |      Optional labels preserving which (u, v) edge generated each segment, in the same order.
 |
 |  Methods defined here:
 |
 |  __delattr__(self, name)
 |      Implement delattr(self, name).
 |
 |  __eq__(self, other)
 |      Return self==value.
 |
 |  __hash__(self)
 |      Return hash(self).
 |
 |  __init__(self, vertices: 'List[Point3]', faces: 'List[Face]', sides: 'int', end_caps: 'bool', segment_count: 'int', edge_count: 'int', segments: 'List[Segment]', edge_uvs: 'Optional[List[Tuple[int, int]]]' = None) -> None
 |      Initialize self.  See help(type(self)) for accurate signature.
 |
 |  __repr__(self)
 |      Return repr(self).
 |
 |  __setattr__(self, name, value)
 |      Implement setattr(self, name, value).
 |
 |  per_segment_face_slices(self) -> 'List[Tuple[int, int]]'
 |      Return (start, count) face spans for each segment in current mesh order.
 |
 |  print_segment_order(self) -> 'None'
 |      Print the current segment ordering; uses `(u, v)` labels if available.
 |
 |  reordered(self, new_order: 'Sequence[int] | None' = None, *, label_remap: 'Optional[Mapping[Tuple[int, int], int]]' = None) -> "'FrustaSet'"
 |      Return a new set with segments reordered by index or (u, v) label mapping.
 |
 |  scale(self, scalar: 'float') -> "'FrustaSet'"
 |      Return a new `FrustaSet` with coordinates and radii scaled by `scalar`.
 |
 |  scaled(self, radius_scale: 'float') -> "'FrustaSet'"
 |      Return a new FrustaSet with all segment radii scaled by `radius_scale`.
 |
 |      This rebuilds vertices/faces from the stored `segments` list.
 |
 |  to_mesh3d_arrays(self) -> 'Tuple[List[float], List[float], List[float], List[int], List[int], List[int]]'
 |      Return Plotly Mesh3d arrays: x, y, z, i, j, k.
 |
 |  ----------------------------------------------------------------------
 |  Class methods defined here:
 |
 |  from_swc_model(model: 'SWCModel', *, sides: 'int' = 16, end_caps: 'bool' = False, flip_tag_assignment: 'bool' = False) -> "'FrustaSet'"
 |      Build a `FrustaSet` by converting each undirected edge into a `Segment`.
 |
 |      Expects nodes to have attributes `x, y, z, r`.
 |
 |      Parameters
 |      ----------
 |      model: SWCModel
 |          The model to convert.
 |      sides: int
 |          Number of sides per frustum.
 |      end_caps: bool
 |          Whether to include end caps.
 |      flip_tag_assignment: bool
 |          If True, assign tags from the child node to the parent node. Otherwise, assign tags from the parent node to the child node.
 |
 |  ----------------------------------------------------------------------
 |  Data descriptors defined here:
 |
 |  __dict__
 |      dictionary for instance variables
 |
 |  __weakref__
 |      list of weak references to the object
 |
 |  ----------------------------------------------------------------------
 |  Data and other attributes defined here:
 |
 |  __annotations__ = {'edge_count': 'int', 'edge_uvs': 'Optional[List[Tup...
 |
 |  __dataclass_fields__ = {'edge_count': Field(name='edge_count',type='in...
 |
 |  __dataclass_params__ = _DataclassParams(init=True,repr=True,eq=True,or...
 |
 |  __match_args__ = ('vertices', 'faces', 'sides', 'end_caps', 'segment_c...
 |
 |  edge_uvs = None

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Class: swctools.PointSet
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Python Library Documentation: class PointSet in module swctools.geometry

class PointSet(builtins.object)
 |  PointSet(vertices: 'List[Point3]', faces: 'List[Face]', points: 'List[Point3]', base_radius: 'float', stacks: 'int', slices: 'int') -> None
 |
 |  A batched mesh of small spheres placed at given 3D points.
 |
 |  Methods defined here:
 |
 |  __delattr__(self, name)
 |      Implement delattr(self, name).
 |
 |  __eq__(self, other)
 |      Return self==value.
 |
 |  __hash__(self)
 |      Return hash(self).
 |
 |  __init__(self, vertices: 'List[Point3]', faces: 'List[Face]', points: 'List[Point3]', base_radius: 'float', stacks: 'int', slices: 'int') -> None
 |      Initialize self.  See help(type(self)) for accurate signature.
 |
 |  __repr__(self)
 |      Return repr(self).
 |
 |  __setattr__(self, name, value)
 |      Implement setattr(self, name, value).
 |
 |  project_onto_frusta(self, frusta: "'FrustaSet'", include_end_caps: 'Optional[bool]' = None) -> "'PointSet'"
 |      Project each point to the nearest surface of the nearest frustum.
 |
 |      Parameters
 |      ----------
 |      frusta: FrustaSet
 |          Set of oriented frusta (as `Segment`s) to project onto.
 |      include_end_caps: Optional[bool]
 |          If None (default), follow `frusta.end_caps`. If True/False, explicitly
 |          include or ignore projections to the circular end caps.
 |
 |      Returns
 |      -------
 |      PointSet
 |          A new `PointSet` whose `points` have been moved onto the closest
 |          surface points of the closest frusta; sphere mesh is rebuilt.
 |
 |      Notes
 |      -----
 |      For each input point, the algorithm iterates all frusta segments and
 |      evaluates the squared distance to:
 |      - The lateral surface: project the point to the segment axis (clamped
 |        t in [0,1]), interpolate radius r(t), then move along the radial
 |        direction to the mantle.
 |      - The end caps (optional): orthogonal distance to each cap plane; if
 |        the projected point falls outside the disk, distance to the rim is used.
 |      Degenerate segments (zero length) are treated as a sphere of radius
 |      max(ra, rb) centered at the endpoint.
 |      Complexity is O(N_points × N_segments), implemented in pure Python.
 |
 |  scale(self, scalar: 'float') -> "'PointSet'"
 |      Return a new `PointSet` with coordinates and radii scaled by `scalar`.
 |
 |  scaled(self, radius_scale: 'float') -> "'PointSet'"
 |      Return a new `PointSet` with all sphere radii scaled by `radius_scale`.
 |
 |  to_mesh3d_arrays(self) -> 'Tuple[List[float], List[float], List[float], List[int], List[int], List[int]]'
 |      Return Plotly `Mesh3d` arrays `(x, y, z, i, j, k)` for this point set.
 |
 |  to_txt_file(self, path: 'Union[str, os.PathLike]') -> 'None'
 |
 |  ----------------------------------------------------------------------
 |  Class methods defined here:
 |
 |  from_points(points: 'Sequence[Point3]', *, base_radius: 'float' = 1.0, stacks: 'int' = 6, slices: 'int' = 12) -> "'PointSet'"
 |      Build a batched low-res spheres mesh from a list of 3D points.
 |
 |      Parameters
 |      ----------
 |      points: sequence of (x, y, z)
 |          Sphere centers.
 |      base_radius: float
 |          Sphere radius used when building the mesh (scaled later via `scaled()`).
 |      stacks, slices: int
 |          Sphere tessellation parameters (>=2 and >=3 respectively).
 |
 |  from_txt_file(path: 'Union[str, os.PathLike]', *, base_radius: 'float' = 1.0, stacks: 'int' = 6, slices: 'int' = 12, comments: 'str' = '#') -> "'PointSet'"
 |
 |  ----------------------------------------------------------------------
 |  Data descriptors defined here:
 |
 |  __dict__
 |      dictionary for instance variables
 |
 |  __weakref__
 |      list of weak references to the object
 |
 |  ----------------------------------------------------------------------
 |  Data and other attributes defined here:
 |
 |  __annotations__ = {'base_radius': 'float', 'faces': 'List[Face]', 'poi...
 |
 |  __dataclass_fields__ = {'base_radius': Field(name='base_radius',type='...
 |
 |  __dataclass_params__ = _DataclassParams(init=True,repr=True,eq=True,or...
 |
 |  __match_args__ = ('vertices', 'faces', 'points', 'base_radius', 'stack...

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Class: swctools.SWCModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Python Library Documentation: class SWCModel in module swctools.model

class SWCModel(networkx.classes.graph.Graph)
 |  SWCModel() -> 'None'
 |
 |  SWC morphology graph (undirected storage with directed-tree metadata).
 |
 |  Nodes are keyed by SWC id `n` and store attributes:
 |  - t: int (tag)
 |  - x, y, z: float (coordinates)
 |  - r: float (radius)
 |  - line: int (line number in source; informational)
 |
 |  Directed parent -> child relationships from the SWC are preserved via an
 |  internal parent map (`_parents`). The underlying graph is undirected. Cycle
 |  connections can be applied via `make_cycle_connections()` which may merge nodes.
 |
 |  Method resolution order:
 |      SWCModel
 |      networkx.classes.graph.Graph
 |      builtins.object
 |
 |  Methods defined here:
 |
 |  __init__(self) -> 'None'
 |      Initialize a graph with edges, name, or graph attributes.
 |
 |      Parameters
 |      ----------
 |      incoming_graph_data : input graph (optional, default: None)
 |          Data to initialize graph. If None (default) an empty
 |          graph is created.  The data can be an edge list, or any
 |          NetworkX graph object.  If the corresponding optional Python
 |          packages are installed the data can also be a 2D NumPy array, a
 |          SciPy sparse array, or a PyGraphviz graph.
 |
 |      attr : keyword arguments, optional (default= no attributes)
 |          Attributes to add to graph as key=value pairs.
 |
 |      See Also
 |      --------
 |      convert
 |
 |      Examples
 |      --------
 |      >>> G = nx.Graph()  # or DiGraph, MultiGraph, MultiDiGraph, etc
 |      >>> G = nx.Graph(name="my graph")
 |      >>> e = [(1, 2), (2, 3), (3, 4)]  # list of edges
 |      >>> G = nx.Graph(e)
 |
 |      Arbitrary graph attribute pairs (key=value) may be assigned
 |
 |      >>> G = nx.Graph(e, day="Friday")
 |      >>> G.graph
 |      {'day': 'Friday'}
 |
 |  add_junction(self, node_id: 'int | None' = None, *, t: 'int' = 0, x: 'float' = 0.0, y: 'float' = 0.0, z: 'float' = 0.0, r: 'float' = 0.0, parent: 'int | None' = None, **kwargs: 'Any') -> 'int'
 |      Add a junction (node) to the model.
 |
 |      Parameters
 |      ----------
 |      node_id: int | None
 |          Node ID to use. If None, automatically assigns the next available ID.
 |      t: int
 |          Node tag. Default 0.
 |      x, y, z: float
 |          Node coordinates. Default 0.0.
 |      r: float
 |          Node radius. Default 0.0.
 |      parent: int | None
 |          Parent node ID. If specified, creates an edge to the parent.
 |          Default None (root node).
 |      **kwargs: Any
 |          Additional node attributes.
 |
 |      Returns
 |      -------
 |      int
 |          The ID of the added node.
 |
 |  copy(self) -> "'SWCModel'"
 |      Return a shallow copy of this model (nodes/edges/attributes).
 |
 |  make_cycle_connections(self, *, validate_reconnections: 'bool' = True, float_tol: 'float' = 1e-09) -> "'SWCModel'"
 |      Return a new SWCModel with reconnection pairs merged and undirected edges across merged reps.
 |
 |      Uses reconnection pairs stored under `self.graph['reconnections']` if present.
 |      Node attributes are merged; provenance kept under `merged_ids` and `lines`.
 |      After merges, the original tree parent mapping no longer applies and parents are set to None.
 |
 |  parent_of(self, n: 'int') -> 'int | None'
 |      Return the parent id of node n from the original SWC tree (or None).
 |
 |  path_to_root(self, n: 'int') -> 'list[int]'
 |      Return the path from node n up to its root, inclusive.
 |
 |      Example: For edges 1->2->3, `path_to_root(3)` returns `[3, 2, 1]`.
 |
 |  print_attributes(self, *, node_info: 'bool' = False, edge_info: 'bool' = False) -> 'None'
 |      Print graph attributes and optional node/edge details.
 |
 |      Parameters
 |      ----------
 |      node_info: bool
 |          If True, print per-node attributes (t, x, y, z, r, line where present).
 |      edge_info: bool
 |          If True, print all edges (u -- v) with edge attributes if any.
 |
 |  remove_junction(self, node_id: 'int', *, reconnect_children: 'bool' = False) -> 'None'
 |      Remove a junction (node) from the model.
 |
 |      Parameters
 |      ----------
 |      node_id: int
 |          ID of the node to remove.
 |      reconnect_children: bool
 |          If True, reconnect children of the removed node to its parent.
 |          If False (default), children become orphaned (roots).
 |
 |  roots(self) -> 'list[int]'
 |      Return nodes with no parent in the original SWC tree.
 |
 |  scale(self, scalar: 'float') -> "'SWCModel'"
 |      Return a new model with all node coordinates and radii scaled by `scalar`.
 |
 |      Multiplies each node's `x`, `y`, `z`, and `r` by `scalar` on a copy.
 |
 |  set_tag_by_sphere(self, center: 'tuple[float, float, float] | list[float]', radius: 'float', new_tag: 'int', old_tag: 'int | None' = None, include_boundary: 'bool' = True, copy: 'bool' = False) -> "'SWCModel'"
 |      Override node 't' values for points inside a sphere.
 |
 |      Sets the tag 't' for all nodes whose Euclidean distance from
 |      `center` is less than `radius` (or equal if `include_boundary` is True).
 |
 |      If `old_tag` is specified, only nodes with that tag are modified.
 |
 |      Parameters
 |      ----------
 |      center: tuple[float, float, float] | list[float]
 |          Sphere center as (x, y, z).
 |      radius: float
 |          Sphere radius (same units as coordinates).
 |      new_tag: int
 |          Tag to assign to matching nodes.
 |      old_tag: int | None
 |          If specified, only nodes with this tag are modified.
 |      include_boundary: bool
 |          If True, include nodes exactly at distance == radius. Default True.
 |      copy: bool
 |          If True, operate on and return a copy; otherwise mutate in place and return self.
 |
 |  to_swc_file(self, path: 'str | os.PathLike[str]', *, precision: 'int' = 6, header: 'Iterable[str] | None' = None) -> 'None'
 |      Write the model to an SWC file.
 |
 |      The output uses the standard 7-column SWC format per row:
 |      "n T x y z r parent" with floats formatted to the requested precision.
 |
 |      Parameters
 |      ----------
 |      path: str | os.PathLike[str]
 |          Destination file path.
 |      precision: int
 |          Decimal places for floating-point fields (x, y, z, r). Default 6.
 |      header: Iterable[str] | None
 |          Optional additional header comment lines (without leading '#').
 |
 |  ----------------------------------------------------------------------
 |  Class methods defined here:
 |
 |  from_parse_result(result: 'SWCParseResult') -> "'SWCModel'"
 |      Build a model from a parsed SWC result.
 |
 |  from_records(records: 'Mapping[int, SWCRecord] | Iterable[SWCRecord]') -> "'SWCModel'"
 |      Build a model from SWC records.
 |
 |      Accepts either a mapping of id->record or any iterable of SWCRecord.
 |
 |  from_swc_file(source: 'str | os.PathLike[str] | Iterable[str]', *, strict: 'bool' = True, validate_reconnections: 'bool' = True, float_tol: 'float' = 1e-09) -> "'SWCModel'"
 |      Parse an SWC source then build a model.
 |
 |      The `source` is passed through to `parse_swc`, which supports a path,
 |      a file-like object, a string with the full contents, or an iterable of lines.
 |
 |  ----------------------------------------------------------------------
 |  Methods inherited from networkx.classes.graph.Graph:
 |
 |  __contains__(self, n)
 |      Returns True if n is a node, False otherwise. Use: 'n in G'.
 |
 |      Examples
 |      --------
 |      >>> G = nx.path_graph(4)  # or DiGraph, MultiGraph, MultiDiGraph, etc
 |      >>> 1 in G
 |      True
 |
 |  __getitem__(self, n)
 |      Returns a dict of neighbors of node n.  Use: 'G[n]'.
 |
 |      Parameters
 |      ----------
 |      n : node
 |         A node in the graph.
 |
 |      Returns
 |      -------
 |      adj_dict : dictionary
 |         The adjacency dictionary for nodes connected to n.
 |
 |      Notes
 |      -----
 |      G[n] is the same as G.adj[n] and similar to G.neighbors(n)
 |      (which is an iterator over G.adj[n])
 |
 |      Examples
 |      --------
 |      >>> G = nx.path_graph(4)  # or DiGraph, MultiGraph, MultiDiGraph, etc
 |      >>> G[0]
 |      AtlasView({1: {}})
 |
 |  __iter__(self)
 |      Iterate over the nodes. Use: 'for n in G'.
 |
 |      Returns
 |      -------
 |      niter : iterator
 |          An iterator over all nodes in the graph.
 |
 |      Examples
 |      --------
 |      >>> G = nx.path_graph(4)  # or DiGraph, MultiGraph, MultiDiGraph, etc
 |      >>> [n for n in G]
 |      [0, 1, 2, 3]
 |      >>> list(G)
 |      [0, 1, 2, 3]
 |
 |  __len__(self)
 |      Returns the number of nodes in the graph. Use: 'len(G)'.
 |
 |      Returns
 |      -------
 |      nnodes : int
 |          The number of nodes in the graph.
 |
 |      See Also
 |      --------
 |      number_of_nodes: identical method
 |      order: identical method
 |
 |      Examples
 |      --------
 |      >>> G = nx.path_graph(4)  # or DiGraph, MultiGraph, MultiDiGraph, etc
 |      >>> len(G)
 |      4
 |
 |  __str__(self)
 |      Returns a short summary of the graph.
 |
 |      Returns
 |      -------
 |      info : string
 |          Graph information including the graph name (if any), graph type, and the
 |          number of nodes and edges.
 |
 |      Examples
 |      --------
 |      >>> G = nx.Graph(name="foo")
 |      >>> str(G)
 |      "Graph named 'foo' with 0 nodes and 0 edges"
 |
 |      >>> G = nx.path_graph(3)
 |      >>> str(G)
 |      'Graph with 3 nodes and 2 edges'
 |
 |  add_edge(self, u_of_edge, v_of_edge, **attr)
 |      Add an edge between u and v.
 |
 |      The nodes u and v will be automatically added if they are
 |      not already in the graph.
 |
 |      Edge attributes can be specified with keywords or by directly
 |      accessing the edge's attribute dictionary. See examples below.
 |
 |      Parameters
 |      ----------
 |      u_of_edge, v_of_edge : nodes
 |          Nodes can be, for example, strings or numbers.
 |          Nodes must be hashable (and not None) Python objects.
 |      attr : keyword arguments, optional
 |          Edge data (or labels or objects) can be assigned using
 |          keyword arguments.
 |
 |      See Also
 |      --------
 |      add_edges_from : add a collection of edges
 |
 |      Notes
 |      -----
 |      Adding an edge that already exists updates the edge data.
 |
 |      Many NetworkX algorithms designed for weighted graphs use
 |      an edge attribute (by default `weight`) to hold a numerical value.
 |
 |      Examples
 |      --------
 |      The following all add the edge e=(1, 2) to graph G:
 |
 |      >>> G = nx.Graph()  # or DiGraph, MultiGraph, MultiDiGraph, etc
 |      >>> e = (1, 2)
 |      >>> G.add_edge(1, 2)  # explicit two-node form
 |      >>> G.add_edge(*e)  # single edge as tuple of two nodes
 |      >>> G.add_edges_from([(1, 2)])  # add edges from iterable container
 |
 |      Associate data to edges using keywords:
 |
 |      >>> G.add_edge(1, 2, weight=3)
 |      >>> G.add_edge(1, 3, weight=7, capacity=15, length=342.7)
 |
 |      For non-string attribute keys, use subscript notation.
 |
 |      >>> G.add_edge(1, 2)
 |      >>> G[1][2].update({0: 5})
 |      >>> G.edges[1, 2].update({0: 5})
 |
 |  add_edges_from(self, ebunch_to_add, **attr)
 |      Add all the edges in ebunch_to_add.
 |
 |      Parameters
 |      ----------
 |      ebunch_to_add : container of edges
 |          Each edge given in the container will be added to the
 |          graph. The edges must be given as 2-tuples (u, v) or
 |          3-tuples (u, v, d) where d is a dictionary containing edge data.
 |      attr : keyword arguments, optional
 |          Edge data (or labels or objects) can be assigned using
 |          keyword arguments.
 |
 |      See Also
 |      --------
 |      add_edge : add a single edge
 |      add_weighted_edges_from : convenient way to add weighted edges
 |
 |      Notes
 |      -----
 |      Adding the same edge twice has no effect but any edge data
 |      will be updated when each duplicate edge is added.
 |
 |      Edge attributes specified in an ebunch take precedence over
 |      attributes specified via keyword arguments.
 |
 |      When adding edges from an iterator over the graph you are changing,
 |      a `RuntimeError` can be raised with message:
 |      `RuntimeError: dictionary changed size during iteration`. This
 |      happens when the graph's underlying dictionary is modified during
 |      iteration. To avoid this error, evaluate the iterator into a separate
 |      object, e.g. by using `list(iterator_of_edges)`, and pass this
 |      object to `G.add_edges_from`.
 |
 |      Examples
 |      --------
 |      >>> G = nx.Graph()  # or DiGraph, MultiGraph, MultiDiGraph, etc
 |      >>> G.add_edges_from([(0, 1), (1, 2)])  # using a list of edge tuples
 |      >>> e = zip(range(0, 3), range(1, 4))
 |      >>> G.add_edges_from(e)  # Add the path graph 0-1-2-3
 |
 |      Associate data to edges
 |
 |      >>> G.add_edges_from([(1, 2), (2, 3)], weight=3)
 |      >>> G.add_edges_from([(3, 4), (1, 4)], label="WN2898")
 |
 |      Evaluate an iterator over a graph if using it to modify the same graph
 |
 |      >>> G = nx.Graph([(1, 2), (2, 3), (3, 4)])
 |      >>> # Grow graph by one new node, adding edges to all existing nodes.
 |      >>> # wrong way - will raise RuntimeError
 |      >>> # G.add_edges_from(((5, n) for n in G.nodes))
 |      >>> # correct way - note that there will be no self-edge for node 5
 |      >>> G.add_edges_from(list((5, n) for n in G.nodes))
 |
 |  add_node(self, node_for_adding, **attr)
 |      Add a single node `node_for_adding` and update node attributes.
 |
 |      Parameters
 |      ----------
 |      node_for_adding : node
 |          A node can be any hashable Python object except None.
 |      attr : keyword arguments, optional
 |          Set or change node attributes using key=value.
 |
 |      See Also
 |      --------
 |      add_nodes_from
 |
 |      Examples
 |      --------
 |      >>> G = nx.Graph()  # or DiGraph, MultiGraph, MultiDiGraph, etc
 |      >>> G.add_node(1)
 |      >>> G.add_node("Hello")
 |      >>> K3 = nx.Graph([(0, 1), (1, 2), (2, 0)])
 |      >>> G.add_node(K3)
 |      >>> G.number_of_nodes()
 |      3
 |
 |      Use keywords set/change node attributes:
 |
 |      >>> G.add_node(1, size=10)
 |      >>> G.add_node(3, weight=0.4, UTM=("13S", 382871, 3972649))
 |
 |      Notes
 |      -----
 |      A hashable object is one that can be used as a key in a Python
 |      dictionary. This includes strings, numbers, tuples of strings
 |      and numbers, etc.
 |
 |      On many platforms hashable items also include mutables such as
 |      NetworkX Graphs, though one should be careful that the hash
 |      doesn't change on mutables.
 |
 |  add_nodes_from(self, nodes_for_adding, **attr)
 |      Add multiple nodes.
 |
 |      Parameters
 |      ----------
 |      nodes_for_adding : iterable container
 |          A container of nodes (list, dict, set, etc.).
 |          OR
 |          A container of (node, attribute dict) tuples.
 |          Node attributes are updated using the attribute dict.
 |      attr : keyword arguments, optional (default= no attributes)
 |          Update attributes for all nodes in nodes.
 |          Node attributes specified in nodes as a tuple take
 |          precedence over attributes specified via keyword arguments.
 |
 |      See Also
 |      --------
 |      add_node
 |
 |      Notes
 |      -----
 |      When adding nodes from an iterator over the graph you are changing,
 |      a `RuntimeError` can be raised with message:
 |      `RuntimeError: dictionary changed size during iteration`. This
 |      happens when the graph's underlying dictionary is modified during
 |      iteration. To avoid this error, evaluate the iterator into a separate
 |      object, e.g. by using `list(iterator_of_nodes)`, and pass this
 |      object to `G.add_nodes_from`.
 |
 |      Examples
 |      --------
 |      >>> G = nx.Graph()  # or DiGraph, MultiGraph, MultiDiGraph, etc
 |      >>> G.add_nodes_from("Hello")
 |      >>> K3 = nx.Graph([(0, 1), (1, 2), (2, 0)])
 |      >>> G.add_nodes_from(K3)
 |      >>> sorted(G.nodes(), key=str)
 |      [0, 1, 2, 'H', 'e', 'l', 'o']
 |
 |      Use keywords to update specific node attributes for every node.
 |
 |      >>> G.add_nodes_from([1, 2], size=10)
 |      >>> G.add_nodes_from([3, 4], weight=0.4)
 |
 |      Use (node, attrdict) tuples to update attributes for specific nodes.
 |
 |      >>> G.add_nodes_from([(1, dict(size=11)), (2, {"color": "blue"})])
 |      >>> G.nodes[1]["size"]
 |      11
 |      >>> H = nx.Graph()
 |      >>> H.add_nodes_from(G.nodes(data=True))
 |      >>> H.nodes[1]["size"]
 |      11
 |
 |      Evaluate an iterator over a graph if using it to modify the same graph
 |
 |      >>> G = nx.Graph([(0, 1), (1, 2), (3, 4)])
 |      >>> # wrong way - will raise RuntimeError
 |      >>> # G.add_nodes_from(n + 1 for n in G.nodes)
 |      >>> # correct way
 |      >>> G.add_nodes_from(list(n + 1 for n in G.nodes))
 |
 |  add_weighted_edges_from(self, ebunch_to_add, weight='weight', **attr)
 |      Add weighted edges in `ebunch_to_add` with specified weight attr
 |
 |      Parameters
 |      ----------
 |      ebunch_to_add : container of edges
 |          Each edge given in the list or container will be added
 |          to the graph. The edges must be given as 3-tuples (u, v, w)
 |          where w is a number.
 |      weight : string, optional (default= 'weight')
 |          The attribute name for the edge weights to be added.
 |      attr : keyword arguments, optional (default= no attributes)
 |          Edge attributes to add/update for all edges.
 |
 |      See Also
 |      --------
 |      add_edge : add a single edge
 |      add_edges_from : add multiple edges
 |
 |      Notes
 |      -----
 |      Adding the same edge twice for Graph/DiGraph simply updates
 |      the edge data. For MultiGraph/MultiDiGraph, duplicate edges
 |      are stored.
 |
 |      When adding edges from an iterator over the graph you are changing,
 |      a `RuntimeError` can be raised with message:
 |      `RuntimeError: dictionary changed size during iteration`. This
 |      happens when the graph's underlying dictionary is modified during
 |      iteration. To avoid this error, evaluate the iterator into a separate
 |      object, e.g. by using `list(iterator_of_edges)`, and pass this
 |      object to `G.add_weighted_edges_from`.
 |
 |      Examples
 |      --------
 |      >>> G = nx.Graph()  # or DiGraph, MultiGraph, MultiDiGraph, etc
 |      >>> G.add_weighted_edges_from([(0, 1, 3.0), (1, 2, 7.5)])
 |
 |      Evaluate an iterator over edges before passing it
 |
 |      >>> G = nx.Graph([(1, 2), (2, 3), (3, 4)])
 |      >>> weight = 0.1
 |      >>> # Grow graph by one new node, adding edges to all existing nodes.
 |      >>> # wrong way - will raise RuntimeError
 |      >>> # G.add_weighted_edges_from(((5, n, weight) for n in G.nodes))
 |      >>> # correct way - note that there will be no self-edge for node 5
 |      >>> G.add_weighted_edges_from(list((5, n, weight) for n in G.nodes))
 |
 |  adj = <functools.cached_property object>
 |      Graph adjacency object holding the neighbors of each node.
 |
 |      This object is a read-only dict-like structure with node keys
 |      and neighbor-dict values.  The neighbor-dict is keyed by neighbor
 |      to the edge-data-dict.  So `G.adj[3][2]['color'] = 'blue'` sets
 |      the color of the edge `(3, 2)` to `"blue"`.
 |
 |      Iterating over G.adj behaves like a dict. Useful idioms include
 |      `for nbr, datadict in G.adj[n].items():`.
 |
 |      The neighbor information is also provided by subscripting the graph.
 |      So `for nbr, foovalue in G[node].data('foo', default=1):` works.
 |
 |      For directed graphs, `G.adj` holds outgoing (successor) info.
 |
 |  adjacency(self)
 |      Returns an iterator over (node, adjacency dict) tuples for all nodes.
 |
 |      For directed graphs, only outgoing neighbors/adjacencies are included.
 |
 |      Returns
 |      -------
 |      adj_iter : iterator
 |         An iterator over (node, adjacency dictionary) for all nodes in
 |         the graph.
 |
 |      Examples
 |      --------
 |      >>> G = nx.path_graph(4)  # or DiGraph, MultiGraph, MultiDiGraph, etc
 |      >>> [(n, nbrdict) for n, nbrdict in G.adjacency()]
 |      [(0, {1: {}}), (1, {0: {}, 2: {}}), (2, {1: {}, 3: {}}), (3, {2: {}})]
 |
 |  clear(self)
 |      Remove all nodes and edges from the graph.
 |
 |      This also removes the name, and all graph, node, and edge attributes.
 |
 |      Examples
 |      --------
 |      >>> G = nx.path_graph(4)  # or DiGraph, MultiGraph, MultiDiGraph, etc
 |      >>> G.clear()
 |      >>> list(G.nodes)
 |      []
 |      >>> list(G.edges)
 |      []
 |
 |  clear_edges(self)
 |      Remove all edges from the graph without altering nodes.
 |
 |      Examples
 |      --------
 |      >>> G = nx.path_graph(4)  # or DiGraph, MultiGraph, MultiDiGraph, etc
 |      >>> G.clear_edges()
 |      >>> list(G.nodes)
 |      [0, 1, 2, 3]
 |      >>> list(G.edges)
 |      []
 |
 |  degree = <functools.cached_property object>
 |      A DegreeView for the Graph as G.degree or G.degree().
 |
 |      The node degree is the number of edges adjacent to the node.
 |      The weighted node degree is the sum of the edge weights for
 |      edges incident to that node.
 |
 |      This object provides an iterator for (node, degree) as well as
 |      lookup for the degree for a single node.
 |
 |      Parameters
 |      ----------
 |      nbunch : single node, container, or all nodes (default= all nodes)
 |          The view will only report edges incident to these nodes.
 |
 |      weight : string or None, optional (default=None)
 |         The name of an edge attribute that holds the numerical value used
 |         as a weight.  If None, then each edge has weight 1.
 |         The degree is the sum of the edge weights adjacent to the node.
 |
 |      Returns
 |      -------
 |      DegreeView or int
 |          If multiple nodes are requested (the default), returns a `DegreeView`
 |          mapping nodes to their degree.
 |          If a single node is requested, returns the degree of the node as an integer.
 |
 |      Examples
 |      --------
 |      >>> G = nx.path_graph(4)  # or DiGraph, MultiGraph, MultiDiGraph, etc
 |      >>> G.degree[0]  # node 0 has degree 1
 |      1
 |      >>> list(G.degree([0, 1, 2]))
 |      [(0, 1), (1, 2), (2, 2)]
 |
 |  edge_subgraph(self, edges)
 |      Returns the subgraph induced by the specified edges.
 |
 |      The induced subgraph contains each edge in `edges` and each
 |      node incident to any one of those edges.
 |
 |      Parameters
 |      ----------
 |      edges : iterable
 |          An iterable of edges in this graph.
 |
 |      Returns
 |      -------
 |      G : Graph
 |          An edge-induced subgraph of this graph with the same edge
 |          attributes.
 |
 |      Notes
 |      -----
 |      The graph, edge, and node attributes in the returned subgraph
 |      view are references to the corresponding attributes in the original
 |      graph. The view is read-only.
 |
 |      To create a full graph version of the subgraph with its own copy
 |      of the edge or node attributes, use::
 |
 |          G.edge_subgraph(edges).copy()
 |
 |      Examples
 |      --------
 |      >>> G = nx.path_graph(5)
 |      >>> H = G.edge_subgraph([(0, 1), (3, 4)])
 |      >>> list(H.nodes)
 |      [0, 1, 3, 4]
 |      >>> list(H.edges)
 |      [(0, 1), (3, 4)]
 |
 |  edges = <functools.cached_property object>
 |      An EdgeView of the Graph as G.edges or G.edges().
 |
 |      edges(self, nbunch=None, data=False, default=None)
 |
 |      The EdgeView provides set-like operations on the edge-tuples
 |      as well as edge attribute lookup. When called, it also provides
 |      an EdgeDataView object which allows control of access to edge
 |      attributes (but does not provide set-like operations).
 |      Hence, `G.edges[u, v]['color']` provides the value of the color
 |      attribute for edge `(u, v)` while
 |      `for (u, v, c) in G.edges.data('color', default='red'):`
 |      iterates through all the edges yielding the color attribute
 |      with default `'red'` if no color attribute exists.
 |
 |      Parameters
 |      ----------
 |      nbunch : single node, container, or all nodes (default= all nodes)
 |          The view will only report edges from these nodes.
 |      data : string or bool, optional (default=False)
 |          The edge attribute returned in 3-tuple (u, v, ddict[data]).
 |          If True, return edge attribute dict in 3-tuple (u, v, ddict).
 |          If False, return 2-tuple (u, v).
 |      default : value, optional (default=None)
 |          Value used for edges that don't have the requested attribute.
 |          Only relevant if data is not True or False.
 |
 |      Returns
 |      -------
 |      edges : EdgeView
 |          A view of edge attributes, usually it iterates over (u, v)
 |          or (u, v, d) tuples of edges, but can also be used for
 |          attribute lookup as `edges[u, v]['foo']`.
 |
 |      Notes
 |      -----
 |      Nodes in nbunch that are not in the graph will be (quietly) ignored.
 |      For directed graphs this returns the out-edges.
 |
 |      Examples
 |      --------
 |      >>> G = nx.path_graph(3)  # or MultiGraph, etc
 |      >>> G.add_edge(2, 3, weight=5)
 |      >>> [e for e in G.edges]
 |      [(0, 1), (1, 2), (2, 3)]
 |      >>> G.edges.data()  # default data is {} (empty dict)
 |      EdgeDataView([(0, 1, {}), (1, 2, {}), (2, 3, {'weight': 5})])
 |      >>> G.edges.data("weight", default=1)
 |      EdgeDataView([(0, 1, 1), (1, 2, 1), (2, 3, 5)])
 |      >>> G.edges([0, 3])  # only edges from these nodes
 |      EdgeDataView([(0, 1), (3, 2)])
 |      >>> G.edges(0)  # only edges from node 0
 |      EdgeDataView([(0, 1)])
 |
 |  get_edge_data(self, u, v, default=None)
 |      Returns the attribute dictionary associated with edge (u, v).
 |
 |      This is identical to `G[u][v]` except the default is returned
 |      instead of an exception if the edge doesn't exist.
 |
 |      Parameters
 |      ----------
 |      u, v : nodes
 |      default:  any Python object (default=None)
 |          Value to return if the edge (u, v) is not found.
 |
 |      Returns
 |      -------
 |      edge_dict : dictionary
 |          The edge attribute dictionary.
 |
 |      Examples
 |      --------
 |      >>> G = nx.path_graph(4)  # or DiGraph, MultiGraph, MultiDiGraph, etc
 |      >>> G[0][1]
 |      {}
 |
 |      Warning: Assigning to `G[u][v]` is not permitted.
 |      But it is safe to assign attributes `G[u][v]['foo']`
 |
 |      >>> G[0][1]["weight"] = 7
 |      >>> G[0][1]["weight"]
 |      7
 |      >>> G[1][0]["weight"]
 |      7
 |
 |      >>> G = nx.path_graph(4)  # or DiGraph, MultiGraph, MultiDiGraph, etc
 |      >>> G.get_edge_data(0, 1)  # default edge data is {}
 |      {}
 |      >>> e = (0, 1)
 |      >>> G.get_edge_data(*e)  # tuple form
 |      {}
 |      >>> G.get_edge_data("a", "b", default=0)  # edge not in graph, return 0
 |      0
 |
 |  has_edge(self, u, v)
 |      Returns True if the edge (u, v) is in the graph.
 |
 |      This is the same as `v in G[u]` without KeyError exceptions.
 |
 |      Parameters
 |      ----------
 |      u, v : nodes
 |          Nodes can be, for example, strings or numbers.
 |          Nodes must be hashable (and not None) Python objects.
 |
 |      Returns
 |      -------
 |      edge_ind : bool
 |          True if edge is in the graph, False otherwise.
 |
 |      Examples
 |      --------
 |      >>> G = nx.path_graph(4)  # or DiGraph, MultiGraph, MultiDiGraph, etc
 |      >>> G.has_edge(0, 1)  # using two nodes
 |      True
 |      >>> e = (0, 1)
 |      >>> G.has_edge(*e)  #  e is a 2-tuple (u, v)
 |      True
 |      >>> e = (0, 1, {"weight": 7})
 |      >>> G.has_edge(*e[:2])  # e is a 3-tuple (u, v, data_dictionary)
 |      True
 |
 |      The following syntax are equivalent:
 |
 |      >>> G.has_edge(0, 1)
 |      True
 |      >>> 1 in G[0]  # though this gives KeyError if 0 not in G
 |      True
 |
 |  has_node(self, n)
 |      Returns True if the graph contains the node n.
 |
 |      Identical to `n in G`
 |
 |      Parameters
 |      ----------
 |      n : node
 |
 |      Examples
 |      --------
 |      >>> G = nx.path_graph(3)  # or DiGraph, MultiGraph, MultiDiGraph, etc
 |      >>> G.has_node(0)
 |      True
 |
 |      It is more readable and simpler to use
 |
 |      >>> 0 in G
 |      True
 |
 |  is_directed(self)
 |      Returns True if graph is directed, False otherwise.
 |
 |  is_multigraph(self)
 |      Returns True if graph is a multigraph, False otherwise.
 |
 |  nbunch_iter(self, nbunch=None)
 |      Returns an iterator over nodes contained in nbunch that are
 |      also in the graph.
 |
 |      The nodes in an iterable nbunch are checked for membership in the graph
 |      and if not are silently ignored.
 |
 |      Parameters
 |      ----------
 |      nbunch : single node, container, or all nodes (default= all nodes)
 |          The view will only report edges incident to these nodes.
 |
 |      Returns
 |      -------
 |      niter : iterator
 |          An iterator over nodes in nbunch that are also in the graph.
 |          If nbunch is None, iterate over all nodes in the graph.
 |
 |      Raises
 |      ------
 |      NetworkXError
 |          If nbunch is not a node or sequence of nodes.
 |          If a node in nbunch is not hashable.
 |
 |      See Also
 |      --------
 |      Graph.__iter__
 |
 |      Notes
 |      -----
 |      When nbunch is an iterator, the returned iterator yields values
 |      directly from nbunch, becoming exhausted when nbunch is exhausted.
 |
 |      To test whether nbunch is a single node, one can use
 |      "if nbunch in self:", even after processing with this routine.
 |
 |      If nbunch is not a node or a (possibly empty) sequence/iterator
 |      or None, a :exc:`NetworkXError` is raised.  Also, if any object in
 |      nbunch is not hashable, a :exc:`NetworkXError` is raised.
 |
 |  neighbors(self, n)
 |      Returns an iterator over all neighbors of node n.
 |
 |      This is identical to `iter(G[n])`
 |
 |      Parameters
 |      ----------
 |      n : node
 |         A node in the graph
 |
 |      Returns
 |      -------
 |      neighbors : iterator
 |          An iterator over all neighbors of node n
 |
 |      Raises
 |      ------
 |      NetworkXError
 |          If the node n is not in the graph.
 |
 |      Examples
 |      --------
 |      >>> G = nx.path_graph(4)  # or DiGraph, MultiGraph, MultiDiGraph, etc
 |      >>> [n for n in G.neighbors(0)]
 |      [1]
 |
 |      Notes
 |      -----
 |      Alternate ways to access the neighbors are ``G.adj[n]`` or ``G[n]``:
 |
 |      >>> G = nx.Graph()  # or DiGraph, MultiGraph, MultiDiGraph, etc
 |      >>> G.add_edge("a", "b", weight=7)
 |      >>> G["a"]
 |      AtlasView({'b': {'weight': 7}})
 |      >>> G = nx.path_graph(4)
 |      >>> [n for n in G[0]]
 |      [1]
 |
 |  nodes = <functools.cached_property object>
 |      A NodeView of the Graph as G.nodes or G.nodes().
 |
 |      Can be used as `G.nodes` for data lookup and for set-like operations.
 |      Can also be used as `G.nodes(data='color', default=None)` to return a
 |      NodeDataView which reports specific node data but no set operations.
 |      It presents a dict-like interface as well with `G.nodes.items()`
 |      iterating over `(node, nodedata)` 2-tuples and `G.nodes[3]['foo']`
 |      providing the value of the `foo` attribute for node `3`. In addition,
 |      a view `G.nodes.data('foo')` provides a dict-like interface to the
 |      `foo` attribute of each node. `G.nodes.data('foo', default=1)`
 |      provides a default for nodes that do not have attribute `foo`.
 |
 |      Parameters
 |      ----------
 |      data : string or bool, optional (default=False)
 |          The node attribute returned in 2-tuple (n, ddict[data]).
 |          If True, return entire node attribute dict as (n, ddict).
 |          If False, return just the nodes n.
 |
 |      default : value, optional (default=None)
 |          Value used for nodes that don't have the requested attribute.
 |          Only relevant if data is not True or False.
 |
 |      Returns
 |      -------
 |      NodeView
 |          Allows set-like operations over the nodes as well as node
 |          attribute dict lookup and calling to get a NodeDataView.
 |          A NodeDataView iterates over `(n, data)` and has no set operations.
 |          A NodeView iterates over `n` and includes set operations.
 |
 |          When called, if data is False, an iterator over nodes.
 |          Otherwise an iterator of 2-tuples (node, attribute value)
 |          where the attribute is specified in `data`.
 |          If data is True then the attribute becomes the
 |          entire data dictionary.
 |
 |      Notes
 |      -----
 |      If your node data is not needed, it is simpler and equivalent
 |      to use the expression ``for n in G``, or ``list(G)``.
 |
 |      Examples
 |      --------
 |      There are two simple ways of getting a list of all nodes in the graph:
 |
 |      >>> G = nx.path_graph(3)
 |      >>> list(G.nodes)
 |      [0, 1, 2]
 |      >>> list(G)
 |      [0, 1, 2]
 |
 |      To get the node data along with the nodes:
 |
 |      >>> G.add_node(1, time="5pm")
 |      >>> G.nodes[0]["foo"] = "bar"
 |      >>> list(G.nodes(data=True))
 |      [(0, {'foo': 'bar'}), (1, {'time': '5pm'}), (2, {})]
 |      >>> list(G.nodes.data())
 |      [(0, {'foo': 'bar'}), (1, {'time': '5pm'}), (2, {})]
 |
 |      >>> list(G.nodes(data="foo"))
 |      [(0, 'bar'), (1, None), (2, None)]
 |      >>> list(G.nodes.data("foo"))
 |      [(0, 'bar'), (1, None), (2, None)]
 |
 |      >>> list(G.nodes(data="time"))
 |      [(0, None), (1, '5pm'), (2, None)]
 |      >>> list(G.nodes.data("time"))
 |      [(0, None), (1, '5pm'), (2, None)]
 |
 |      >>> list(G.nodes(data="time", default="Not Available"))
 |      [(0, 'Not Available'), (1, '5pm'), (2, 'Not Available')]
 |      >>> list(G.nodes.data("time", default="Not Available"))
 |      [(0, 'Not Available'), (1, '5pm'), (2, 'Not Available')]
 |
 |      If some of your nodes have an attribute and the rest are assumed
 |      to have a default attribute value you can create a dictionary
 |      from node/attribute pairs using the `default` keyword argument
 |      to guarantee the value is never None::
 |
 |          >>> G = nx.Graph()
 |          >>> G.add_node(0)
 |          >>> G.add_node(1, weight=2)
 |          >>> G.add_node(2, weight=3)
 |          >>> dict(G.nodes(data="weight", default=1))
 |          {0: 1, 1: 2, 2: 3}
 |
 |  number_of_edges(self, u=None, v=None)
 |      Returns the number of edges between two nodes.
 |
 |      Parameters
 |      ----------
 |      u, v : nodes, optional (default=all edges)
 |          If u and v are specified, return the number of edges between
 |          u and v. Otherwise return the total number of all edges.
 |
 |      Returns
 |      -------
 |      nedges : int
 |          The number of edges in the graph.  If nodes `u` and `v` are
 |          specified return the number of edges between those nodes. If
 |          the graph is directed, this only returns the number of edges
 |          from `u` to `v`.
 |
 |      See Also
 |      --------
 |      size
 |
 |      Examples
 |      --------
 |      For undirected graphs, this method counts the total number of
 |      edges in the graph:
 |
 |      >>> G = nx.path_graph(4)
 |      >>> G.number_of_edges()
 |      3
 |
 |      If you specify two nodes, this counts the total number of edges
 |      joining the two nodes:
 |
 |      >>> G.number_of_edges(0, 1)
 |      1
 |
 |      For directed graphs, this method can count the total number of
 |      directed edges from `u` to `v`:
 |
 |      >>> G = nx.DiGraph()
 |      >>> G.add_edge(0, 1)
 |      >>> G.add_edge(1, 0)
 |      >>> G.number_of_edges(0, 1)
 |      1
 |
 |  number_of_nodes(self)
 |      Returns the number of nodes in the graph.
 |
 |      Returns
 |      -------
 |      nnodes : int
 |          The number of nodes in the graph.
 |
 |      See Also
 |      --------
 |      order: identical method
 |      __len__: identical method
 |
 |      Examples
 |      --------
 |      >>> G = nx.path_graph(3)  # or DiGraph, MultiGraph, MultiDiGraph, etc
 |      >>> G.number_of_nodes()
 |      3
 |
 |  order(self)
 |      Returns the number of nodes in the graph.
 |
 |      Returns
 |      -------
 |      nnodes : int
 |          The number of nodes in the graph.
 |
 |      See Also
 |      --------
 |      number_of_nodes: identical method
 |      __len__: identical method
 |
 |      Examples
 |      --------
 |      >>> G = nx.path_graph(3)  # or DiGraph, MultiGraph, MultiDiGraph, etc
 |      >>> G.order()
 |      3
 |
 |  remove_edge(self, u, v)
 |      Remove the edge between u and v.
 |
 |      Parameters
 |      ----------
 |      u, v : nodes
 |          Remove the edge between nodes u and v.
 |
 |      Raises
 |      ------
 |      NetworkXError
 |          If there is not an edge between u and v.
 |
 |      See Also
 |      --------
 |      remove_edges_from : remove a collection of edges
 |
 |      Examples
 |      --------
 |      >>> G = nx.path_graph(4)  # or DiGraph, etc
 |      >>> G.remove_edge(0, 1)
 |      >>> e = (1, 2)
 |      >>> G.remove_edge(*e)  # unpacks e from an edge tuple
 |      >>> e = (2, 3, {"weight": 7})  # an edge with attribute data
 |      >>> G.remove_edge(*e[:2])  # select first part of edge tuple
 |
 |  remove_edges_from(self, ebunch)
 |      Remove all edges specified in ebunch.
 |
 |      Parameters
 |      ----------
 |      ebunch: list or container of edge tuples
 |          Each edge given in the list or container will be removed
 |          from the graph. The edges can be:
 |
 |              - 2-tuples (u, v) edge between u and v.
 |              - 3-tuples (u, v, k) where k is ignored.
 |
 |      See Also
 |      --------
 |      remove_edge : remove a single edge
 |
 |      Notes
 |      -----
 |      Will fail silently if an edge in ebunch is not in the graph.
 |
 |      Examples
 |      --------
 |      >>> G = nx.path_graph(4)  # or DiGraph, MultiGraph, MultiDiGraph, etc
 |      >>> ebunch = [(1, 2), (2, 3)]
 |      >>> G.remove_edges_from(ebunch)
 |
 |  remove_node(self, n)
 |      Remove node n.
 |
 |      Removes the node n and all adjacent edges.
 |      Attempting to remove a nonexistent node will raise an exception.
 |
 |      Parameters
 |      ----------
 |      n : node
 |         A node in the graph
 |
 |      Raises
 |      ------
 |      NetworkXError
 |         If n is not in the graph.
 |
 |      See Also
 |      --------
 |      remove_nodes_from
 |
 |      Examples
 |      --------
 |      >>> G = nx.path_graph(3)  # or DiGraph, MultiGraph, MultiDiGraph, etc
 |      >>> list(G.edges)
 |      [(0, 1), (1, 2)]
 |      >>> G.remove_node(1)
 |      >>> list(G.edges)
 |      []
 |
 |  remove_nodes_from(self, nodes)
 |      Remove multiple nodes.
 |
 |      Parameters
 |      ----------
 |      nodes : iterable container
 |          A container of nodes (list, dict, set, etc.).  If a node
 |          in the container is not in the graph it is silently
 |          ignored.
 |
 |      See Also
 |      --------
 |      remove_node
 |
 |      Notes
 |      -----
 |      When removing nodes from an iterator over the graph you are changing,
 |      a `RuntimeError` will be raised with message:
 |      `RuntimeError: dictionary changed size during iteration`. This
 |      happens when the graph's underlying dictionary is modified during
 |      iteration. To avoid this error, evaluate the iterator into a separate
 |      object, e.g. by using `list(iterator_of_nodes)`, and pass this
 |      object to `G.remove_nodes_from`.
 |
 |      Examples
 |      --------
 |      >>> G = nx.path_graph(3)  # or DiGraph, MultiGraph, MultiDiGraph, etc
 |      >>> e = list(G.nodes)
 |      >>> e
 |      [0, 1, 2]
 |      >>> G.remove_nodes_from(e)
 |      >>> list(G.nodes)
 |      []
 |
 |      Evaluate an iterator over a graph if using it to modify the same graph
 |
 |      >>> G = nx.Graph([(0, 1), (1, 2), (3, 4)])
 |      >>> # this command will fail, as the graph's dict is modified during iteration
 |      >>> # G.remove_nodes_from(n for n in G.nodes if n < 2)
 |      >>> # this command will work, since the dictionary underlying graph is not modified
 |      >>> G.remove_nodes_from(list(n for n in G.nodes if n < 2))
 |
 |  size(self, weight=None)
 |      Returns the number of edges or total of all edge weights.
 |
 |      Parameters
 |      ----------
 |      weight : string or None, optional (default=None)
 |          The edge attribute that holds the numerical value used
 |          as a weight. If None, then each edge has weight 1.
 |
 |      Returns
 |      -------
 |      size : numeric
 |          The number of edges or
 |          (if weight keyword is provided) the total weight sum.
 |
 |          If weight is None, returns an int. Otherwise a float
 |          (or more general numeric if the weights are more general).
 |
 |      See Also
 |      --------
 |      number_of_edges
 |
 |      Examples
 |      --------
 |      >>> G = nx.path_graph(4)  # or DiGraph, MultiGraph, MultiDiGraph, etc
 |      >>> G.size()
 |      3
 |
 |      >>> G = nx.Graph()  # or DiGraph, MultiGraph, MultiDiGraph, etc
 |      >>> G.add_edge("a", "b", weight=2)
 |      >>> G.add_edge("b", "c", weight=4)
 |      >>> G.size()
 |      2
 |      >>> G.size(weight="weight")
 |      6.0
 |
 |  subgraph(self, nodes)
 |      Returns a SubGraph view of the subgraph induced on `nodes`.
 |
 |      The induced subgraph of the graph contains the nodes in `nodes`
 |      and the edges between those nodes.
 |
 |      Parameters
 |      ----------
 |      nodes : list, iterable
 |          A container of nodes which will be iterated through once.
 |
 |      Returns
 |      -------
 |      G : SubGraph View
 |          A subgraph view of the graph. The graph structure cannot be
 |          changed but node/edge attributes can and are shared with the
 |          original graph.
 |
 |      Notes
 |      -----
 |      The graph, edge and node attributes are shared with the original graph.
 |      Changes to the graph structure is ruled out by the view, but changes
 |      to attributes are reflected in the original graph.
 |
 |      To create a subgraph with its own copy of the edge/node attributes use:
 |      G.subgraph(nodes).copy()
 |
 |      For an inplace reduction of a graph to a subgraph you can remove nodes:
 |      G.remove_nodes_from([n for n in G if n not in set(nodes)])
 |
 |      Subgraph views are sometimes NOT what you want. In most cases where
 |      you want to do more than simply look at the induced edges, it makes
 |      more sense to just create the subgraph as its own graph with code like:
 |
 |      ::
 |
 |          # Create a subgraph SG based on a (possibly multigraph) G
 |          SG = G.__class__()
 |          SG.add_nodes_from((n, G.nodes[n]) for n in largest_wcc)
 |          if SG.is_multigraph():
 |              SG.add_edges_from(
 |                  (n, nbr, key, d)
 |                  for n, nbrs in G.adj.items()
 |                  if n in largest_wcc
 |                  for nbr, keydict in nbrs.items()
 |                  if nbr in largest_wcc
 |                  for key, d in keydict.items()
 |              )
 |          else:
 |              SG.add_edges_from(
 |                  (n, nbr, d)
 |                  for n, nbrs in G.adj.items()
 |                  if n in largest_wcc
 |                  for nbr, d in nbrs.items()
 |                  if nbr in largest_wcc
 |              )
 |          SG.graph.update(G.graph)
 |
 |      Subgraphs are not guaranteed to preserve the order of nodes or edges
 |      as they appear in the original graph. For example:
 |
 |      >>> G = nx.Graph()
 |      >>> G.add_nodes_from(reversed(range(10)))
 |      >>> list(G)
 |      [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
 |      >>> list(G.subgraph([1, 3, 2]))
 |      [1, 2, 3]
 |
 |      Examples
 |      --------
 |      >>> G = nx.path_graph(4)  # or DiGraph, MultiGraph, MultiDiGraph, etc
 |      >>> H = G.subgraph([0, 1, 2])
 |      >>> list(H.edges)
 |      [(0, 1), (1, 2)]
 |
 |  to_directed(self, as_view=False)
 |      Returns a directed representation of the graph.
 |
 |      Returns
 |      -------
 |      G : DiGraph
 |          A directed graph with the same name, same nodes, and with
 |          each edge (u, v, data) replaced by two directed edges
 |          (u, v, data) and (v, u, data).
 |
 |      Notes
 |      -----
 |      This returns a "deepcopy" of the edge, node, and
 |      graph attributes which attempts to completely copy
 |      all of the data and references.
 |
 |      This is in contrast to the similar D=DiGraph(G) which returns a
 |      shallow copy of the data.
 |
 |      See the Python copy module for more information on shallow
 |      and deep copies, https://docs.python.org/3/library/copy.html.
 |
 |      Warning: If you have subclassed Graph to use dict-like objects
 |      in the data structure, those changes do not transfer to the
 |      DiGraph created by this method.
 |
 |      Examples
 |      --------
 |      >>> G = nx.Graph()  # or MultiGraph, etc
 |      >>> G.add_edge(0, 1)
 |      >>> H = G.to_directed()
 |      >>> list(H.edges)
 |      [(0, 1), (1, 0)]
 |
 |      If already directed, return a (deep) copy
 |
 |      >>> G = nx.DiGraph()  # or MultiDiGraph, etc
 |      >>> G.add_edge(0, 1)
 |      >>> H = G.to_directed()
 |      >>> list(H.edges)
 |      [(0, 1)]
 |
 |  to_directed_class(self)
 |      Returns the class to use for empty directed copies.
 |
 |      If you subclass the base classes, use this to designate
 |      what directed class to use for `to_directed()` copies.
 |
 |  to_undirected(self, as_view=False)
 |      Returns an undirected copy of the graph.
 |
 |      Parameters
 |      ----------
 |      as_view : bool (optional, default=False)
 |        If True return a view of the original undirected graph.
 |
 |      Returns
 |      -------
 |      G : Graph/MultiGraph
 |          A deepcopy of the graph.
 |
 |      See Also
 |      --------
 |      Graph, copy, add_edge, add_edges_from
 |
 |      Notes
 |      -----
 |      This returns a "deepcopy" of the edge, node, and
 |      graph attributes which attempts to completely copy
 |      all of the data and references.
 |
 |      This is in contrast to the similar `G = nx.DiGraph(D)` which returns a
 |      shallow copy of the data.
 |
 |      See the Python copy module for more information on shallow
 |      and deep copies, https://docs.python.org/3/library/copy.html.
 |
 |      Warning: If you have subclassed DiGraph to use dict-like objects
 |      in the data structure, those changes do not transfer to the
 |      Graph created by this method.
 |
 |      Examples
 |      --------
 |      >>> G = nx.path_graph(2)  # or MultiGraph, etc
 |      >>> H = G.to_directed()
 |      >>> list(H.edges)
 |      [(0, 1), (1, 0)]
 |      >>> G2 = H.to_undirected()
 |      >>> list(G2.edges)
 |      [(0, 1)]
 |
 |  to_undirected_class(self)
 |      Returns the class to use for empty undirected copies.
 |
 |      If you subclass the base classes, use this to designate
 |      what directed class to use for `to_directed()` copies.
 |
 |  update(self, edges=None, nodes=None)
 |      Update the graph using nodes/edges/graphs as input.
 |
 |      Like dict.update, this method takes a graph as input, adding the
 |      graph's nodes and edges to this graph. It can also take two inputs:
 |      edges and nodes. Finally it can take either edges or nodes.
 |      To specify only nodes the keyword `nodes` must be used.
 |
 |      The collections of edges and nodes are treated similarly to
 |      the add_edges_from/add_nodes_from methods. When iterated, they
 |      should yield 2-tuples (u, v) or 3-tuples (u, v, datadict).
 |
 |      Parameters
 |      ----------
 |      edges : Graph object, collection of edges, or None
 |          The first parameter can be a graph or some edges. If it has
 |          attributes `nodes` and `edges`, then it is taken to be a
 |          Graph-like object and those attributes are used as collections
 |          of nodes and edges to be added to the graph.
 |          If the first parameter does not have those attributes, it is
 |          treated as a collection of edges and added to the graph.
 |          If the first argument is None, no edges are added.
 |      nodes : collection of nodes, or None
 |          The second parameter is treated as a collection of nodes
 |          to be added to the graph unless it is None.
 |          If `edges is None` and `nodes is None` an exception is raised.
 |          If the first parameter is a Graph, then `nodes` is ignored.
 |
 |      Examples
 |      --------
 |      >>> G = nx.path_graph(5)
 |      >>> G.update(nx.complete_graph(range(4, 10)))
 |      >>> from itertools import combinations
 |      >>> edges = (
 |      ...     (u, v, {"power": u * v})
 |      ...     for u, v in combinations(range(10, 20), 2)
 |      ...     if u * v < 225
 |      ... )
 |      >>> nodes = [1000]  # for singleton, use a container
 |      >>> G.update(edges, nodes)
 |
 |      Notes
 |      -----
 |      It you want to update the graph using an adjacency structure
 |      it is straightforward to obtain the edges/nodes from adjacency.
 |      The following examples provide common cases, your adjacency may
 |      be slightly different and require tweaks of these examples::
 |
 |      >>> # dict-of-set/list/tuple
 |      >>> adj = {1: {2, 3}, 2: {1, 3}, 3: {1, 2}}
 |      >>> e = [(u, v) for u, nbrs in adj.items() for v in nbrs]
 |      >>> G.update(edges=e, nodes=adj)
 |
 |      >>> DG = nx.DiGraph()
 |      >>> # dict-of-dict-of-attribute
 |      >>> adj = {1: {2: 1.3, 3: 0.7}, 2: {1: 1.4}, 3: {1: 0.7}}
 |      >>> e = [
 |      ...     (u, v, {"weight": d})
 |      ...     for u, nbrs in adj.items()
 |      ...     for v, d in nbrs.items()
 |      ... ]
 |      >>> DG.update(edges=e, nodes=adj)
 |
 |      >>> # dict-of-dict-of-dict
 |      >>> adj = {1: {2: {"weight": 1.3}, 3: {"color": 0.7, "weight": 1.2}}}
 |      >>> e = [
 |      ...     (u, v, {"weight": d})
 |      ...     for u, nbrs in adj.items()
 |      ...     for v, d in nbrs.items()
 |      ... ]
 |      >>> DG.update(edges=e, nodes=adj)
 |
 |      >>> # predecessor adjacency (dict-of-set)
 |      >>> pred = {1: {2, 3}, 2: {3}, 3: {3}}
 |      >>> e = [(v, u) for u, nbrs in pred.items() for v in nbrs]
 |
 |      >>> # MultiGraph dict-of-dict-of-dict-of-attribute
 |      >>> MDG = nx.MultiDiGraph()
 |      >>> adj = {
 |      ...     1: {2: {0: {"weight": 1.3}, 1: {"weight": 1.2}}},
 |      ...     3: {2: {0: {"weight": 0.7}}},
 |      ... }
 |      >>> e = [
 |      ...     (u, v, ekey, d)
 |      ...     for u, nbrs in adj.items()
 |      ...     for v, keydict in nbrs.items()
 |      ...     for ekey, d in keydict.items()
 |      ... ]
 |      >>> MDG.update(edges=e)
 |
 |      See Also
 |      --------
 |      add_edges_from: add multiple edges to a graph
 |      add_nodes_from: add multiple nodes to a graph
 |
 |  ----------------------------------------------------------------------
 |  Data descriptors inherited from networkx.classes.graph.Graph:
 |
 |  __dict__
 |      dictionary for instance variables
 |
 |  __weakref__
 |      list of weak references to the object
 |
 |  name
 |      String identifier of the graph.
 |
 |      This graph attribute appears in the attribute dict G.graph
 |      keyed by the string `"name"`. as well as an attribute (technically
 |      a property) `G.name`. This is entirely user controlled.
 |
 |  ----------------------------------------------------------------------
 |  Data and other attributes inherited from networkx.classes.graph.Graph:
 |
 |  __networkx_backend__ = 'networkx'
 |
 |  adjlist_inner_dict_factory = <class 'dict'>
 |      dict() -> new empty dictionary
 |      dict(mapping) -> new dictionary initialized from a mapping object's
 |          (key, value) pairs
 |      dict(iterable) -> new dictionary initialized as if via:
 |          d = {}
 |          for k, v in iterable:
 |              d[k] = v
 |      dict(**kwargs) -> new dictionary initialized with the name=value pairs
 |          in the keyword argument list.  For example:  dict(one=1, two=2)
 |
 |
 |  adjlist_outer_dict_factory = <class 'dict'>
 |      dict() -> new empty dictionary
 |      dict(mapping) -> new dictionary initialized from a mapping object's
 |          (key, value) pairs
 |      dict(iterable) -> new dictionary initialized as if via:
 |          d = {}
 |          for k, v in iterable:
 |              d[k] = v
 |      dict(**kwargs) -> new dictionary initialized with the name=value pairs
 |          in the keyword argument list.  For example:  dict(one=1, two=2)
 |
 |
 |  edge_attr_dict_factory = <class 'dict'>
 |      dict() -> new empty dictionary
 |      dict(mapping) -> new dictionary initialized from a mapping object's
 |          (key, value) pairs
 |      dict(iterable) -> new dictionary initialized as if via:
 |          d = {}
 |          for k, v in iterable:
 |              d[k] = v
 |      dict(**kwargs) -> new dictionary initialized with the name=value pairs
 |          in the keyword argument list.  For example:  dict(one=1, two=2)
 |
 |
 |  graph_attr_dict_factory = <class 'dict'>
 |      dict() -> new empty dictionary
 |      dict(mapping) -> new dictionary initialized from a mapping object's
 |          (key, value) pairs
 |      dict(iterable) -> new dictionary initialized as if via:
 |          d = {}
 |          for k, v in iterable:
 |              d[k] = v
 |      dict(**kwargs) -> new dictionary initialized with the name=value pairs
 |          in the keyword argument list.  For example:  dict(one=1, two=2)
 |
 |
 |  node_attr_dict_factory = <class 'dict'>
 |      dict() -> new empty dictionary
 |      dict(mapping) -> new dictionary initialized from a mapping object's
 |          (key, value) pairs
 |      dict(iterable) -> new dictionary initialized as if via:
 |          d = {}
 |          for k, v in iterable:
 |              d[k] = v
 |      dict(**kwargs) -> new dictionary initialized with the name=value pairs
 |          in the keyword argument list.  For example:  dict(one=1, two=2)
 |
 |
 |  node_dict_factory = <class 'dict'>
 |      dict() -> new empty dictionary
 |      dict(mapping) -> new dictionary initialized from a mapping object's
 |          (key, value) pairs
 |      dict(iterable) -> new dictionary initialized as if via:
 |          d = {}
 |          for k, v in iterable:
 |              d[k] = v
 |      dict(**kwargs) -> new dictionary initialized with the name=value pairs
 |          in the keyword argument list.  For example:  dict(one=1, two=2)

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Class: swctools.SWCParseResult
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Python Library Documentation: class SWCParseResult in module swctools.io

class SWCParseResult(builtins.object)
 |  SWCParseResult(records: 'Dict[int, SWCRecord]', reconnections: 'List[Tuple[int, int]]', header: 'List[str]') -> None
 |
 |  Parsed SWC content.
 |
 |  Methods defined here:
 |
 |  __delattr__(self, name)
 |      Implement delattr(self, name).
 |
 |  __eq__(self, other)
 |      Return self==value.
 |
 |  __hash__(self)
 |      Return hash(self).
 |
 |  __init__(self, records: 'Dict[int, SWCRecord]', reconnections: 'List[Tuple[int, int]]', header: 'List[str]') -> None
 |      Initialize self.  See help(type(self)) for accurate signature.
 |
 |  __repr__(self) -> 'str'
 |      Return repr(self).
 |
 |  __setattr__(self, name, value)
 |      Implement setattr(self, name, value).
 |
 |  __str__(self) -> 'str'
 |      Return str(self).
 |
 |  ----------------------------------------------------------------------
 |  Data descriptors defined here:
 |
 |  __dict__
 |      dictionary for instance variables
 |
 |  __weakref__
 |      list of weak references to the object
 |
 |  ----------------------------------------------------------------------
 |  Data and other attributes defined here:
 |
 |  __annotations__ = {'header': 'List[str]', 'reconnections': 'List[Tuple...
 |
 |  __dataclass_fields__ = {'header': Field(name='header',type='List[str]'...
 |
 |  __dataclass_params__ = _DataclassParams(init=True,repr=True,eq=True,or...
 |
 |  __match_args__ = ('records', 'reconnections', 'header')

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Class: swctools.SWCRecord
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Python Library Documentation: class SWCRecord in module swctools.io

class SWCRecord(builtins.object)
 |  SWCRecord(n: 'int', t: 'int', x: 'float', y: 'float', z: 'float', r: 'float', parent: 'int', line: 'int') -> None
 |
 |  One SWC row.
 |
 |  Attributes
 |  ----------
 |  n: int
 |      Node id (unique within file)
 |  t: int
 |      Tag index
 |  x, y, z: float
 |      Coordinates (usually micrometers)
 |  r: float
 |      Radius
 |  parent: int
 |      Parent id; -1 indicates root
 |  line: int
 |      1-based line number in the source file/string
 |
 |  Methods defined here:
 |
 |  __delattr__(self, name)
 |      Implement delattr(self, name).
 |
 |  __eq__(self, other)
 |      Return self==value.
 |
 |  __hash__(self)
 |      Return hash(self).
 |
 |  __init__(self, n: 'int', t: 'int', x: 'float', y: 'float', z: 'float', r: 'float', parent: 'int', line: 'int') -> None
 |      Initialize self.  See help(type(self)) for accurate signature.
 |
 |  __repr__(self)
 |      Return repr(self).
 |
 |  __setattr__(self, name, value)
 |      Implement setattr(self, name, value).
 |
 |  ----------------------------------------------------------------------
 |  Data descriptors defined here:
 |
 |  __dict__
 |      dictionary for instance variables
 |
 |  __weakref__
 |      list of weak references to the object
 |
 |  ----------------------------------------------------------------------
 |  Data and other attributes defined here:
 |
 |  __annotations__ = {'line': 'int', 'n': 'int', 'parent': 'int', 'r': 'f...
 |
 |  __dataclass_fields__ = {'line': Field(name='line',type='int',default=<...
 |
 |  __dataclass_params__ = _DataclassParams(init=True,repr=True,eq=True,or...
 |
 |  __match_args__ = ('n', 't', 'x', 'y', 'z', 'r', 'parent', 'line')

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Class: swctools.Segment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Python Library Documentation: class Segment in module swctools.geometry

class Segment(builtins.object)
 |  Segment(a: 'Point3', b: 'Point3', ra: 'float', rb: 'float', tag: 'int') -> None
 |
 |  Oriented frustum segment between endpoints `a` and `b`.
 |
 |  Attributes
 |  ----------
 |  a, b: Point3
 |      Endpoints in model/world coordinates.
 |  ra, rb: float
 |      Radii at `a` and `b`.
 |
 |  Methods defined here:
 |
 |  __delattr__(self, name)
 |      Implement delattr(self, name).
 |
 |  __eq__(self, other)
 |      Return self==value.
 |
 |  __hash__(self)
 |      Return hash(self).
 |
 |  __init__(self, a: 'Point3', b: 'Point3', ra: 'float', rb: 'float', tag: 'int') -> None
 |      Initialize self.  See help(type(self)) for accurate signature.
 |
 |  __repr__(self)
 |      Return repr(self).
 |
 |  __setattr__(self, name, value)
 |      Implement setattr(self, name, value).
 |
 |  length(self) -> 'float'
 |
 |  midpoint(self) -> 'Point3'
 |
 |  scale(self, scalar: 'float') -> "'Segment'"
 |      Return a new `Segment` uniformly scaled by `scalar` (positions and radii).
 |
 |  vector(self) -> 'Vec3'
 |
 |  ----------------------------------------------------------------------
 |  Data descriptors defined here:
 |
 |  __dict__
 |      dictionary for instance variables
 |
 |  __weakref__
 |      list of weak references to the object
 |
 |  ----------------------------------------------------------------------
 |  Data and other attributes defined here:
 |
 |  __annotations__ = {'a': 'Point3', 'b': 'Point3', 'ra': 'float', 'rb': ...
 |
 |  __dataclass_fields__ = {'a': Field(name='a',type='Point3',default=<dat...
 |
 |  __dataclass_params__ = _DataclassParams(init=True,repr=True,eq=True,or...
 |
 |  __match_args__ = ('a', 'b', 'ra', 'rb', 'tag')

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Function: swctools.apply_layout
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Python Library Documentation: function apply_layout in module swctools.config

apply_layout(fig, *, title: 'str | None' = None) -> 'None'
    Apply global layout defaults to a Plotly figure in-place.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Function: swctools.batch_frusta
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Python Library Documentation: function batch_frusta in module swctools.geometry

batch_frusta(segments: 'Iterable[Segment]', *, sides: 'int' = 16, end_caps: 'bool' = False) -> 'Tuple[List[Point3], List[Face]]'
    Batch multiple frusta into a single mesh.

    Returns a concatenated list of `vertices` and `faces` with the proper index offsets.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Function: swctools.frustum_mesh
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Python Library Documentation: function frustum_mesh in module swctools.geometry

frustum_mesh(seg: 'Segment', *, sides: 'int' = 16, end_caps: 'bool' = False) -> 'Tuple[List[Point3], List[Face]]'
    Generate a frustum mesh for a single `Segment`.

    Returns
    -------
    (vertices, faces):
        - vertices: List[Point3]
        - faces: List[Face], each = (i, j, k) indexing into `vertices`

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Function: swctools.get_config
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Python Library Documentation: function get_config in module swctools.config

get_config() -> 'VizConfig'
    Return the current visualization configuration (live object).

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Function: swctools.parse_swc
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Python Library Documentation: function parse_swc in module swctools.io

parse_swc(source: 'Union[str, os.PathLike, Iterable[str], io.TextIOBase]', *, strict: 'bool' = True, validate_reconnections: 'bool' = True, float_tol: 'float' = 1e-09) -> 'SWCParseResult'
    Parse an SWC file or text stream.

    Parameters
    ----------
    source
        Path to an SWC file, a file-like object, an iterable of lines, or a string
        containing SWC content.
    strict
        If True, enforce 7-column rows and validate parent references exist.
    validate_reconnections
        If True, ensure reconnection node pairs share identical (x, y, z, r).
    float_tol
        Tolerance used when comparing floating-point coordinates/radii.

    Returns
    -------
    SWCParseResult
        Parsed records, reconnection pairs, and header lines.

    Raises
    ------
    ValueError
        If parsing or validation fails.
    FileNotFoundError
        If a string path is provided that does not exist.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Function: swctools.plot_centroid
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Python Library Documentation: function plot_centroid in module swctools.viz

plot_centroid(swc_model: 'SWCModel', *, marker_size: 'float' = 2.0, line_width: 'float' = 2.0, show_nodes: 'bool' = True, title: 'str | None' = None) -> 'go.Figure'
    Plot centroid skeleton from an `SWCModel`.

    Edges are drawn as line segments in 3D using Scatter3d.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Function: swctools.plot_frusta
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Python Library Documentation: function plot_frusta in module swctools.viz

plot_frusta(frusta: 'FrustaSet', *, color: 'str' = 'lightblue', opacity: 'float' = 0.8, flatshading: 'bool' = True, radius_scale: 'float' = 1.0, tag_colors: 'dict[int, str] | None' = None, title: 'str | None' = None) -> 'go.Figure'
    Plot a FrustaSet as a Mesh3d figure.

    Parameters
    ----------
    frusta: FrustaSet
        Batched frusta mesh to render.
    color: str
        Mesh color.
    opacity: float
        Mesh opacity.
    flatshading: bool
        Whether to enable flat shading.
    radius_scale: float
        Uniform scale applied to all segment radii before meshing (1.0 = no change).
    tag_colors: dict[int, str] | None
        Optional mapping {tag: color}. If provided, each frustum segment is colored
        uniformly according to its tag (fallback to `color` if a tag is missing).

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Function: swctools.plot_frusta_slider
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Python Library Documentation: function plot_frusta_slider in module swctools.viz

plot_frusta_slider(frusta: 'FrustaSet', *, color: 'str' = 'lightblue', opacity: 'float' = 0.8, flatshading: 'bool' = True, tag_colors: 'dict[int, str] | None' = None, min_scale: 'float' = 0.0, max_scale: 'float' = 1.0, steps: 'int' = 21, title: 'str | None' = None) -> 'go.Figure'
    Interactive slider (0..1 default) controlling uniform `radius_scale`.

    Precomputes frames at evenly spaced scales between `min_scale` and `max_scale`.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Function: swctools.plot_frusta_timeseries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Python Library Documentation: function plot_frusta_timeseries in module swctools.viz

plot_frusta_timeseries(frusta: 'FrustaSet', values: 'Sequence[Sequence[float]]', *, colorscale: 'str | list' = 'Viridis', cmin: 'Optional[float]' = None, cmax: 'Optional[float]' = None, opacity: 'float' = 0.8, flatshading: 'bool' = True, radius_scale: 'float' = 1.0, fps: 'int' = 10, title: 'str | None' = None) -> 'go.Figure'
    Animate per-segment values over time by coloring frusta faces.

    Parameters
    ----------
    frusta: FrustaSet
        Batched frusta mesh. Optionally scaled via `radius_scale` before rendering.
    values: Sequence[Sequence[float]]
        Time series V_i(t) shaped [T][N], where N = `frusta.segment_count`.
        Each time step provides one scalar per segment in the current order.
    colorscale: str | list
        Plotly colorscale for mapping intensities.
    cmin, cmax: float | None
        Fixed color range. If omitted, inferred from the data across all frames.
    opacity: float
        Mesh opacity.
    flatshading: bool
        Enable flat shading on the mesh.
    radius_scale: float
        Uniform radius scaling applied before meshing (1.0 = no change).
    fps: int
        Playback frame rate for the Play button.

    Returns
    -------
    plotly.graph_objects.Figure
        A figure with a Mesh3d trace and animation frames, plus a slider and play/pause controls.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Function: swctools.plot_frusta_with_centroid
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Python Library Documentation: function plot_frusta_with_centroid in module swctools.viz

plot_frusta_with_centroid(swc_model: 'SWCModel', frusta: 'FrustaSet', *, color: 'str' = 'lightblue', opacity: 'float' = 0.8, flatshading: 'bool' = True, radius_scale: 'float' = 1.0, tag_colors: 'dict[int, str] | None' = None, centroid_color: 'str' = '#1f77b4', centroid_line_width: 'float' = 2.0, show_nodes: 'bool' = False, node_size: 'float' = 2.0, title: 'str | None' = None) -> 'go.Figure'
    Overlay frusta mesh with centroid skeleton from an `SWCModel`.

    Parameters mirror `plot_centroid` and `plot_frusta` with an extra `radius_scale`.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Function: swctools.plot_model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Python Library Documentation: function plot_model in module swctools.viz

plot_model(*, swc_model: 'SWCModel | None' = None, frusta: 'FrustaSet | None' = None, show_frusta: 'bool' = True, show_centroid: 'bool' = True, title: 'str | None' = None, sides: 'int' = 16, end_caps: 'bool' = False, color: 'str' = 'lightblue', opacity: 'float' = 0.8, flatshading: 'bool' = True, tag_colors: 'dict[int, str] | None' = None, radius_scale: 'float' = 1.0, slider: 'bool' = False, min_scale: 'float' = 0.0, max_scale: 'float' = 1.0, steps: 'int' = 21, centroid_color: 'str' = '#1f77b4', centroid_line_width: 'float' = 2.0, show_nodes: 'bool' = False, node_size: 'float' = 2.0, point_set: 'PointSet | None' = None, point_size: 'float' = 1.0, point_color: 'str' = '#d62728') -> 'go.Figure'
    Master visualization combining centroid, frusta, slider, and overlay points.

    - If `frusta` is not provided and `gm` is, a `FrustaSet` is built from `gm`.
    - If `slider=True` and `show_frusta=True`, a Plotly slider controls `radius_scale`.
    - `points` overlays arbitrary xyz positions as small markers.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Function: swctools.set_config
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Python Library Documentation: function set_config in module swctools.config

set_config(**kwargs: 'Any') -> 'None'
    Update global visualization configuration.

    Example:
        set_config(width=800, height=600, scene_aspectmode="cube")

================================================================================
Module: swctools.config
--------------------------------------------------------------------------------
Python Library Documentation: module swctools.config in swctools

NAME
    swctools.config - Global visualization configuration for swctools Plotly figures.

DESCRIPTION
    Use `set_config(...)` to override defaults in notebooks/apps, and
    `apply_layout(fig, title=...)` to apply them to a Plotly Figure.

CLASSES
    builtins.object
        VizConfig

    class VizConfig(builtins.object)
     |  VizConfig(width: 'int' = 800, height: 'int' = 600, template: 'str' = 'plotly_white', scene_aspectmode: 'str' = 'auto', force_equal_axes: 'bool' = True, scene_aspectratio: 'Dict[str, float]' = <factory>, margin: 'Dict[str, int]' = <factory>, showlegend: 'bool' = False) -> None
     |
     |  Global Plotly layout defaults used by `apply_layout` and `set_config`.
     |
     |  Methods defined here:
     |
     |  __eq__(self, other)
     |      Return self==value.
     |
     |  __init__(self, width: 'int' = 800, height: 'int' = 600, template: 'str' = 'plotly_white', scene_aspectmode: 'str' = 'auto', force_equal_axes: 'bool' = True, scene_aspectratio: 'Dict[str, float]' = <factory>, margin: 'Dict[str, int]' = <factory>, showlegend: 'bool' = False) -> None
     |      Initialize self.  See help(type(self)) for accurate signature.
     |
     |  __repr__(self)
     |      Return repr(self).
     |
     |  ----------------------------------------------------------------------
     |  Data descriptors defined here:
     |
     |  __dict__
     |      dictionary for instance variables
     |
     |  __weakref__
     |      list of weak references to the object
     |
     |  ----------------------------------------------------------------------
     |  Data and other attributes defined here:
     |
     |  __annotations__ = {'force_equal_axes': 'bool', 'height': 'int', 'margi...
     |
     |  __dataclass_fields__ = {'force_equal_axes': Field(name='force_equal_ax...
     |
     |  __dataclass_params__ = _DataclassParams(init=True,repr=True,eq=True,or...
     |
     |  __hash__ = None
     |
     |  __match_args__ = ('width', 'height', 'template', 'scene_aspectmode', '...
     |
     |  force_equal_axes = True
     |
     |  height = 600
     |
     |  scene_aspectmode = 'auto'
     |
     |  showlegend = False
     |
     |  template = 'plotly_white'
     |
     |  width = 800

FUNCTIONS
    apply_layout(fig, *, title: 'str | None' = None) -> 'None'
        Apply global layout defaults to a Plotly figure in-place.

    get_config() -> 'VizConfig'
        Return the current visualization configuration (live object).

    set_config(**kwargs: 'Any') -> 'None'
        Update global visualization configuration.

        Example:
            set_config(width=800, height=600, scene_aspectmode="cube")

DATA
    Dict = typing.Dict
        A generic version of dict.

FILE
    c:\users\mainuser\documents\repos\mascaf\.venv\lib\site-packages\swctools\config.py


~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Class: swctools.config.Any
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Python Library Documentation: class Any in module typing

class Any(builtins.object)
 |  Any(*args, **kwargs)
 |
 |  Special type indicating an unconstrained type.
 |
 |  - Any is compatible with every type.
 |  - Any assumed to have all methods.
 |  - All values assumed to be instances of Any.
 |
 |  Note that all the above statements are true from the point of view of
 |  static type checkers. At runtime, Any should not be used with instance
 |  checks.
 |
 |  Static methods defined here:
 |
 |  __new__(cls, *args, **kwargs)
 |      Create and return a new object.  See help(type) for accurate signature.
 |
 |  ----------------------------------------------------------------------
 |  Data descriptors defined here:
 |
 |  __dict__
 |      dictionary for instance variables
 |
 |  __weakref__
 |      list of weak references to the object

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Class: swctools.config.VizConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Python Library Documentation: class VizConfig in module swctools.config

class VizConfig(builtins.object)
 |  VizConfig(width: 'int' = 800, height: 'int' = 600, template: 'str' = 'plotly_white', scene_aspectmode: 'str' = 'auto', force_equal_axes: 'bool' = True, scene_aspectratio: 'Dict[str, float]' = <factory>, margin: 'Dict[str, int]' = <factory>, showlegend: 'bool' = False) -> None
 |
 |  Global Plotly layout defaults used by `apply_layout` and `set_config`.
 |
 |  Methods defined here:
 |
 |  __eq__(self, other)
 |      Return self==value.
 |
 |  __init__(self, width: 'int' = 800, height: 'int' = 600, template: 'str' = 'plotly_white', scene_aspectmode: 'str' = 'auto', force_equal_axes: 'bool' = True, scene_aspectratio: 'Dict[str, float]' = <factory>, margin: 'Dict[str, int]' = <factory>, showlegend: 'bool' = False) -> None
 |      Initialize self.  See help(type(self)) for accurate signature.
 |
 |  __repr__(self)
 |      Return repr(self).
 |
 |  ----------------------------------------------------------------------
 |  Data descriptors defined here:
 |
 |  __dict__
 |      dictionary for instance variables
 |
 |  __weakref__
 |      list of weak references to the object
 |
 |  ----------------------------------------------------------------------
 |  Data and other attributes defined here:
 |
 |  __annotations__ = {'force_equal_axes': 'bool', 'height': 'int', 'margi...
 |
 |  __dataclass_fields__ = {'force_equal_axes': Field(name='force_equal_ax...
 |
 |  __dataclass_params__ = _DataclassParams(init=True,repr=True,eq=True,or...
 |
 |  __hash__ = None
 |
 |  __match_args__ = ('width', 'height', 'template', 'scene_aspectmode', '...
 |
 |  force_equal_axes = True
 |
 |  height = 600
 |
 |  scene_aspectmode = 'auto'
 |
 |  showlegend = False
 |
 |  template = 'plotly_white'
 |
 |  width = 800

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Function: swctools.config.asdict
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Python Library Documentation: function asdict in module dataclasses

asdict(obj, *, dict_factory=<class 'dict'>)
    Return the fields of a dataclass instance as a new dictionary mapping
    field names to field values.

    Example usage::

      @dataclass
      class C:
          x: int
          y: int

      c = C(1, 2)
      assert asdict(c) == {'x': 1, 'y': 2}

    If given, 'dict_factory' will be used instead of built-in dict.
    The function applies recursively to field values that are
    dataclass instances. This will also look into built-in containers:
    tuples, lists, and dicts. Other objects are copied with 'copy.deepcopy()'.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Function: swctools.config.dataclass
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Python Library Documentation: function dataclass in module dataclasses

dataclass(cls=None, /, *, init=True, repr=True, eq=True, order=False, unsafe_hash=False, frozen=False, match_args=True, kw_only=False, slots=False, weakref_slot=False)
    Add dunder methods based on the fields defined in the class.

    Examines PEP 526 __annotations__ to determine fields.

    If init is true, an __init__() method is added to the class. If repr
    is true, a __repr__() method is added. If order is true, rich
    comparison dunder methods are added. If unsafe_hash is true, a
    __hash__() method is added. If frozen is true, fields may not be
    assigned to after instance creation. If match_args is true, the
    __match_args__ tuple is added. If kw_only is true, then by default
    all fields are keyword-only. If slots is true, a new class with a
    __slots__ attribute is returned.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Function: swctools.config.field
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Python Library Documentation: function field in module dataclasses

field(*, default=<dataclasses._MISSING_TYPE object at 0x000001B46C536B10>, default_factory=<dataclasses._MISSING_TYPE object at 0x000001B46C536B10>, init=True, repr=True, hash=None, compare=True, metadata=None, kw_only=<dataclasses._MISSING_TYPE object at 0x000001B46C536B10>)
    Return an object to identify dataclass fields.

    default is the default value of the field.  default_factory is a
    0-argument function called to initialize a field's value.  If init
    is true, the field will be a parameter to the class's __init__()
    function.  If repr is true, the field will be included in the
    object's repr().  If hash is true, the field will be included in the
    object's hash().  If compare is true, the field will be used in
    comparison functions.  metadata, if specified, must be a mapping
    which is stored but not otherwise examined by dataclass.  If kw_only
    is true, the field will become a keyword-only parameter to
    __init__().

    It is an error to specify both default and default_factory.

================================================================================
Module: swctools.geometry
--------------------------------------------------------------------------------
Python Library Documentation: module swctools.geometry in swctools

NAME
    swctools.geometry - Geometry utilities for segment/frustum mesh generation.

DESCRIPTION
    - Segment: oriented frustum defined by two points with radii
    - frustum_mesh: build vertices/faces for a single frustum
    - batch_frusta: combine multiple frusta into one mesh

    Implementation is pure-Python (standard library math), returning lists
    of vertices and triangular faces suitable for Plotly Mesh3d or other
    renderers after light conversion.

CLASSES
    builtins.object
        FrustaSet
        PointSet
        Segment

    class FrustaSet(builtins.object)
     |  FrustaSet(vertices: 'List[Point3]', faces: 'List[Face]', sides: 'int', end_caps: 'bool', segment_count: 'int', edge_count: 'int', segments: 'List[Segment]', edge_uvs: 'Optional[List[Tuple[int, int]]]' = None) -> None
     |
     |  A batched frusta mesh derived from a `SWCModel`.
     |
     |  Attributes
     |  ----------
     |  vertices: List[Point3]
     |      Concatenated vertices for all frusta.
     |  faces: List[Face]
     |      Triangular faces indexing into `vertices`.
     |  sides: int
     |      Circumferential resolution used per frustum.
     |  end_caps: bool
     |      Whether end caps were included during construction.
     |  segment_count: int
     |      Number of segments used (one per graph edge).
     |  edge_count: int
     |      Alias for `segment_count` for clarity.
     |  segments: List[Segment]
     |      The segments used to construct the frusta.
     |  edge_uvs: Optional[List[Tuple[int, int]]]
     |      Optional labels preserving which (u, v) edge generated each segment, in the same order.
     |
     |  Methods defined here:
     |
     |  __delattr__(self, name)
     |      Implement delattr(self, name).
     |
     |  __eq__(self, other)
     |      Return self==value.
     |
     |  __hash__(self)
     |      Return hash(self).
     |
     |  __init__(self, vertices: 'List[Point3]', faces: 'List[Face]', sides: 'int', end_caps: 'bool', segment_count: 'int', edge_count: 'int', segments: 'List[Segment]', edge_uvs: 'Optional[List[Tuple[int, int]]]' = None) -> None
     |      Initialize self.  See help(type(self)) for accurate signature.
     |
     |  __repr__(self)
     |      Return repr(self).
     |
     |  __setattr__(self, name, value)
     |      Implement setattr(self, name, value).
     |
     |  per_segment_face_slices(self) -> 'List[Tuple[int, int]]'
     |      Return (start, count) face spans for each segment in current mesh order.
     |
     |  print_segment_order(self) -> 'None'
     |      Print the current segment ordering; uses `(u, v)` labels if available.
     |
     |  reordered(self, new_order: 'Sequence[int] | None' = None, *, label_remap: 'Optional[Mapping[Tuple[int, int], int]]' = None) -> "'FrustaSet'"
     |      Return a new set with segments reordered by index or (u, v) label mapping.
     |
     |  scale(self, scalar: 'float') -> "'FrustaSet'"
     |      Return a new `FrustaSet` with coordinates and radii scaled by `scalar`.
     |
     |  scaled(self, radius_scale: 'float') -> "'FrustaSet'"
     |      Return a new FrustaSet with all segment radii scaled by `radius_scale`.
     |
     |      This rebuilds vertices/faces from the stored `segments` list.
     |
     |  to_mesh3d_arrays(self) -> 'Tuple[List[float], List[float], List[float], List[int], List[int], List[int]]'
     |      Return Plotly Mesh3d arrays: x, y, z, i, j, k.
     |
     |  ----------------------------------------------------------------------
     |  Class methods defined here:
     |
     |  from_swc_model(model: 'SWCModel', *, sides: 'int' = 16, end_caps: 'bool' = False, flip_tag_assignment: 'bool' = False) -> "'FrustaSet'"
     |      Build a `FrustaSet` by converting each undirected edge into a `Segment`.
     |
     |      Expects nodes to have attributes `x, y, z, r`.
     |
     |      Parameters
     |      ----------
     |      model: SWCModel
     |          The model to convert.
     |      sides: int
     |          Number of sides per frustum.
     |      end_caps: bool
     |          Whether to include end caps.
     |      flip_tag_assignment: bool
     |          If True, assign tags from the child node to the parent node. Otherwise, assign tags from the parent node to the child node.
     |
     |  ----------------------------------------------------------------------
     |  Data descriptors defined here:
     |
     |  __dict__
     |      dictionary for instance variables
     |
     |  __weakref__
     |      list of weak references to the object
     |
     |  ----------------------------------------------------------------------
     |  Data and other attributes defined here:
     |
     |  __annotations__ = {'edge_count': 'int', 'edge_uvs': 'Optional[List[Tup...
     |
     |  __dataclass_fields__ = {'edge_count': Field(name='edge_count',type='in...
     |
     |  __dataclass_params__ = _DataclassParams(init=True,repr=True,eq=True,or...
     |
     |  __match_args__ = ('vertices', 'faces', 'sides', 'end_caps', 'segment_c...
     |
     |  edge_uvs = None

    class PointSet(builtins.object)
     |  PointSet(vertices: 'List[Point3]', faces: 'List[Face]', points: 'List[Point3]', base_radius: 'float', stacks: 'int', slices: 'int') -> None
     |
     |  A batched mesh of small spheres placed at given 3D points.
     |
     |  Methods defined here:
     |
     |  __delattr__(self, name)
     |      Implement delattr(self, name).
     |
     |  __eq__(self, other)
     |      Return self==value.
     |
     |  __hash__(self)
     |      Return hash(self).
     |
     |  __init__(self, vertices: 'List[Point3]', faces: 'List[Face]', points: 'List[Point3]', base_radius: 'float', stacks: 'int', slices: 'int') -> None
     |      Initialize self.  See help(type(self)) for accurate signature.
     |
     |  __repr__(self)
     |      Return repr(self).
     |
     |  __setattr__(self, name, value)
     |      Implement setattr(self, name, value).
     |
     |  project_onto_frusta(self, frusta: "'FrustaSet'", include_end_caps: 'Optional[bool]' = None) -> "'PointSet'"
     |      Project each point to the nearest surface of the nearest frustum.
     |
     |      Parameters
     |      ----------
     |      frusta: FrustaSet
     |          Set of oriented frusta (as `Segment`s) to project onto.
     |      include_end_caps: Optional[bool]
     |          If None (default), follow `frusta.end_caps`. If True/False, explicitly
     |          include or ignore projections to the circular end caps.
     |
     |      Returns
     |      -------
     |      PointSet
     |          A new `PointSet` whose `points` have been moved onto the closest
     |          surface points of the closest frusta; sphere mesh is rebuilt.
     |
     |      Notes
     |      -----
     |      For each input point, the algorithm iterates all frusta segments and
     |      evaluates the squared distance to:
     |      - The lateral surface: project the point to the segment axis (clamped
     |        t in [0,1]), interpolate radius r(t), then move along the radial
     |        direction to the mantle.
     |      - The end caps (optional): orthogonal distance to each cap plane; if
     |        the projected point falls outside the disk, distance to the rim is used.
     |      Degenerate segments (zero length) are treated as a sphere of radius
     |      max(ra, rb) centered at the endpoint.
     |      Complexity is O(N_points × N_segments), implemented in pure Python.
     |
     |  scale(self, scalar: 'float') -> "'PointSet'"
     |      Return a new `PointSet` with coordinates and radii scaled by `scalar`.
     |
     |  scaled(self, radius_scale: 'float') -> "'PointSet'"
     |      Return a new `PointSet` with all sphere radii scaled by `radius_scale`.
     |
     |  to_mesh3d_arrays(self) -> 'Tuple[List[float], List[float], List[float], List[int], List[int], List[int]]'
     |      Return Plotly `Mesh3d` arrays `(x, y, z, i, j, k)` for this point set.
     |
     |  to_txt_file(self, path: 'Union[str, os.PathLike]') -> 'None'
     |
     |  ----------------------------------------------------------------------
     |  Class methods defined here:
     |
     |  from_points(points: 'Sequence[Point3]', *, base_radius: 'float' = 1.0, stacks: 'int' = 6, slices: 'int' = 12) -> "'PointSet'"
     |      Build a batched low-res spheres mesh from a list of 3D points.
     |
     |      Parameters
     |      ----------
     |      points: sequence of (x, y, z)
     |          Sphere centers.
     |      base_radius: float
     |          Sphere radius used when building the mesh (scaled later via `scaled()`).
     |      stacks, slices: int
     |          Sphere tessellation parameters (>=2 and >=3 respectively).
     |
     |  from_txt_file(path: 'Union[str, os.PathLike]', *, base_radius: 'float' = 1.0, stacks: 'int' = 6, slices: 'int' = 12, comments: 'str' = '#') -> "'PointSet'"
     |
     |  ----------------------------------------------------------------------
     |  Data descriptors defined here:
     |
     |  __dict__
     |      dictionary for instance variables
     |
     |  __weakref__
     |      list of weak references to the object
     |
     |  ----------------------------------------------------------------------
     |  Data and other attributes defined here:
     |
     |  __annotations__ = {'base_radius': 'float', 'faces': 'List[Face]', 'poi...
     |
     |  __dataclass_fields__ = {'base_radius': Field(name='base_radius',type='...
     |
     |  __dataclass_params__ = _DataclassParams(init=True,repr=True,eq=True,or...
     |
     |  __match_args__ = ('vertices', 'faces', 'points', 'base_radius', 'stack...

    class Segment(builtins.object)
     |  Segment(a: 'Point3', b: 'Point3', ra: 'float', rb: 'float', tag: 'int') -> None
     |
     |  Oriented frustum segment between endpoints `a` and `b`.
     |
     |  Attributes
     |  ----------
     |  a, b: Point3
     |      Endpoints in model/world coordinates.
     |  ra, rb: float
     |      Radii at `a` and `b`.
     |
     |  Methods defined here:
     |
     |  __delattr__(self, name)
     |      Implement delattr(self, name).
     |
     |  __eq__(self, other)
     |      Return self==value.
     |
     |  __hash__(self)
     |      Return hash(self).
     |
     |  __init__(self, a: 'Point3', b: 'Point3', ra: 'float', rb: 'float', tag: 'int') -> None
     |      Initialize self.  See help(type(self)) for accurate signature.
     |
     |  __repr__(self)
     |      Return repr(self).
     |
     |  __setattr__(self, name, value)
     |      Implement setattr(self, name, value).
     |
     |  length(self) -> 'float'
     |
     |  midpoint(self) -> 'Point3'
     |
     |  scale(self, scalar: 'float') -> "'Segment'"
     |      Return a new `Segment` uniformly scaled by `scalar` (positions and radii).
     |
     |  vector(self) -> 'Vec3'
     |
     |  ----------------------------------------------------------------------
     |  Data descriptors defined here:
     |
     |  __dict__
     |      dictionary for instance variables
     |
     |  __weakref__
     |      list of weak references to the object
     |
     |  ----------------------------------------------------------------------
     |  Data and other attributes defined here:
     |
     |  __annotations__ = {'a': 'Point3', 'b': 'Point3', 'ra': 'float', 'rb': ...
     |
     |  __dataclass_fields__ = {'a': Field(name='a',type='Point3',default=<dat...
     |
     |  __dataclass_params__ = _DataclassParams(init=True,repr=True,eq=True,or...
     |
     |  __match_args__ = ('a', 'b', 'ra', 'rb', 'tag')

FUNCTIONS
    batch_frusta(segments: 'Iterable[Segment]', *, sides: 'int' = 16, end_caps: 'bool' = False) -> 'Tuple[List[Point3], List[Face]]'
        Batch multiple frusta into a single mesh.

        Returns a concatenated list of `vertices` and `faces` with the proper index offsets.

    batch_spheres(points: 'Iterable[Point3]', *, radius: 'float' = 1.0, stacks: 'int' = 6, slices: 'int' = 12) -> 'Tuple[List[Point3], List[Face]]'
        Batch multiple spheres into a single mesh.

        Returns concatenated `vertices` and reindexed `faces`.

    frustum_mesh(seg: 'Segment', *, sides: 'int' = 16, end_caps: 'bool' = False) -> 'Tuple[List[Point3], List[Face]]'
        Generate a frustum mesh for a single `Segment`.

        Returns
        -------
        (vertices, faces):
            - vertices: List[Point3]
            - faces: List[Face], each = (i, j, k) indexing into `vertices`

    sphere_mesh(center: 'Point3', radius: 'float', *, stacks: 'int' = 6, slices: 'int' = 12) -> 'Tuple[List[Point3], List[Face]]'
        Generate a low-res UV sphere mesh at `center` with given `radius`.

        Parameters
        ----------
        stacks: int
            Number of latitudinal divisions (>= 2).
        slices: int
            Number of longitudinal divisions (>= 3).

DATA
    __all__ = ['Segment', 'frustum_mesh', 'batch_frusta', 'sphere_mesh', '...

FILE
    c:\users\mainuser\documents\repos\mascaf\.venv\lib\site-packages\swctools\geometry.py


~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Function: swctools.geometry._circle_ring
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Python Library Documentation: function _circle_ring in module swctools.geometry

_circle_ring(center: 'Point3', radius: 'float', U: 'Vec3', V: 'Vec3', sides: 'int') -> 'List[Point3]'

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Function: swctools.geometry._orthonormal_frame
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Python Library Documentation: function _orthonormal_frame in module swctools.geometry

_orthonormal_frame(z_axis: 'Vec3') -> 'Tuple[Vec3, Vec3, Vec3]'
    Return (U, V, W) forming a right-handed orthonormal basis with W along z_axis.

    Handles near-colinearity by choosing a stable temporary axis.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Function: swctools.geometry.batch_spheres
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Python Library Documentation: function batch_spheres in module swctools.geometry

batch_spheres(points: 'Iterable[Point3]', *, radius: 'float' = 1.0, stacks: 'int' = 6, slices: 'int' = 12) -> 'Tuple[List[Point3], List[Face]]'
    Batch multiple spheres into a single mesh.

    Returns concatenated `vertices` and reindexed `faces`.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Function: swctools.geometry.sphere_mesh
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Python Library Documentation: function sphere_mesh in module swctools.geometry

sphere_mesh(center: 'Point3', radius: 'float', *, stacks: 'int' = 6, slices: 'int' = 12) -> 'Tuple[List[Point3], List[Face]]'
    Generate a low-res UV sphere mesh at `center` with given `radius`.

    Parameters
    ----------
    stacks: int
        Number of latitudinal divisions (>= 2).
    slices: int
        Number of longitudinal divisions (>= 3).

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Function: swctools.geometry.v_add
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Python Library Documentation: function v_add in module swctools.geometry

v_add(a: 'Vec3', b: 'Vec3') -> 'Vec3'

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Function: swctools.geometry.v_cross
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Python Library Documentation: function v_cross in module swctools.geometry

v_cross(a: 'Vec3', b: 'Vec3') -> 'Vec3'

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Function: swctools.geometry.v_dot
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Python Library Documentation: function v_dot in module swctools.geometry

v_dot(a: 'Vec3', b: 'Vec3') -> 'float'

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Function: swctools.geometry.v_mul
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Python Library Documentation: function v_mul in module swctools.geometry

v_mul(a: 'Vec3', s: 'float') -> 'Vec3'

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Function: swctools.geometry.v_norm
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Python Library Documentation: function v_norm in module swctools.geometry

v_norm(a: 'Vec3') -> 'float'

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Function: swctools.geometry.v_sub
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Python Library Documentation: function v_sub in module swctools.geometry

v_sub(a: 'Vec3', b: 'Vec3') -> 'Vec3'

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Function: swctools.geometry.v_unit
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Python Library Documentation: function v_unit in module swctools.geometry

v_unit(a: 'Vec3', eps: 'float' = 1e-12) -> 'Vec3'

================================================================================
Module: swctools.io
--------------------------------------------------------------------------------
Python Library Documentation: module swctools.io in swctools

NAME
    swctools.io - SWC file parsing utilities.

DESCRIPTION
    This module provides functions to parse SWC morphology files and extract:
    - records for each SWC node (n, T, x, y, z, r, parent)
    - header annotations for cycle break reconnections

    It performs basic validations (unique ids, parent references) and can
    optionally validate that requested reconnection node pairs share identical
    (x, y, z, r) values.

    Notes
    -----
    - The SWC format is documented by the NeuronLand spec and INCF.
    - Header reconnection annotations follow lines like:
      "# CYCLE_BREAK reconnect i j"
      Parsing is case-insensitive for the tokens "CYCLE_BREAK" and "reconnect".
    - Geometry/graph construction are handled elsewhere (e.g., SWCModel / GeneralModel).

    Example
    -------
    >>> from swctools.io import parse_swc
    >>> result = parse_swc("data/example.swc")
    >>> len(result.records) > 0
    True
    >>> result.reconnections  # list of (i, j) tuples (may be empty)
    [(16, 8)]

CLASSES
    builtins.object
        SWCParseResult
        SWCRecord

    class SWCParseResult(builtins.object)
     |  SWCParseResult(records: 'Dict[int, SWCRecord]', reconnections: 'List[Tuple[int, int]]', header: 'List[str]') -> None
     |
     |  Parsed SWC content.
     |
     |  Methods defined here:
     |
     |  __delattr__(self, name)
     |      Implement delattr(self, name).
     |
     |  __eq__(self, other)
     |      Return self==value.
     |
     |  __hash__(self)
     |      Return hash(self).
     |
     |  __init__(self, records: 'Dict[int, SWCRecord]', reconnections: 'List[Tuple[int, int]]', header: 'List[str]') -> None
     |      Initialize self.  See help(type(self)) for accurate signature.
     |
     |  __repr__(self) -> 'str'
     |      Return repr(self).
     |
     |  __setattr__(self, name, value)
     |      Implement setattr(self, name, value).
     |
     |  __str__(self) -> 'str'
     |      Return str(self).
     |
     |  ----------------------------------------------------------------------
     |  Data descriptors defined here:
     |
     |  __dict__
     |      dictionary for instance variables
     |
     |  __weakref__
     |      list of weak references to the object
     |
     |  ----------------------------------------------------------------------
     |  Data and other attributes defined here:
     |
     |  __annotations__ = {'header': 'List[str]', 'reconnections': 'List[Tuple...
     |
     |  __dataclass_fields__ = {'header': Field(name='header',type='List[str]'...
     |
     |  __dataclass_params__ = _DataclassParams(init=True,repr=True,eq=True,or...
     |
     |  __match_args__ = ('records', 'reconnections', 'header')

    class SWCRecord(builtins.object)
     |  SWCRecord(n: 'int', t: 'int', x: 'float', y: 'float', z: 'float', r: 'float', parent: 'int', line: 'int') -> None
     |
     |  One SWC row.
     |
     |  Attributes
     |  ----------
     |  n: int
     |      Node id (unique within file)
     |  t: int
     |      Tag index
     |  x, y, z: float
     |      Coordinates (usually micrometers)
     |  r: float
     |      Radius
     |  parent: int
     |      Parent id; -1 indicates root
     |  line: int
     |      1-based line number in the source file/string
     |
     |  Methods defined here:
     |
     |  __delattr__(self, name)
     |      Implement delattr(self, name).
     |
     |  __eq__(self, other)
     |      Return self==value.
     |
     |  __hash__(self)
     |      Return hash(self).
     |
     |  __init__(self, n: 'int', t: 'int', x: 'float', y: 'float', z: 'float', r: 'float', parent: 'int', line: 'int') -> None
     |      Initialize self.  See help(type(self)) for accurate signature.
     |
     |  __repr__(self)
     |      Return repr(self).
     |
     |  __setattr__(self, name, value)
     |      Implement setattr(self, name, value).
     |
     |  ----------------------------------------------------------------------
     |  Data descriptors defined here:
     |
     |  __dict__
     |      dictionary for instance variables
     |
     |  __weakref__
     |      list of weak references to the object
     |
     |  ----------------------------------------------------------------------
     |  Data and other attributes defined here:
     |
     |  __annotations__ = {'line': 'int', 'n': 'int', 'parent': 'int', 'r': 'f...
     |
     |  __dataclass_fields__ = {'line': Field(name='line',type='int',default=<...
     |
     |  __dataclass_params__ = _DataclassParams(init=True,repr=True,eq=True,or...
     |
     |  __match_args__ = ('n', 't', 'x', 'y', 'z', 'r', 'parent', 'line')

FUNCTIONS
    parse_swc(source: 'Union[str, os.PathLike, Iterable[str], io.TextIOBase]', *, strict: 'bool' = True, validate_reconnections: 'bool' = True, float_tol: 'float' = 1e-09) -> 'SWCParseResult'
        Parse an SWC file or text stream.

        Parameters
        ----------
        source
            Path to an SWC file, a file-like object, an iterable of lines, or a string
            containing SWC content.
        strict
            If True, enforce 7-column rows and validate parent references exist.
        validate_reconnections
            If True, ensure reconnection node pairs share identical (x, y, z, r).
        float_tol
            Tolerance used when comparing floating-point coordinates/radii.

        Returns
        -------
        SWCParseResult
            Parsed records, reconnection pairs, and header lines.

        Raises
        ------
        ValueError
            If parsing or validation fails.
        FileNotFoundError
            If a string path is provided that does not exist.

DATA
    __all__ = ['SWCRecord', 'SWCParseResult', 'parse_swc']

FILE
    c:\users\mainuser\documents\repos\mascaf\.venv\lib\site-packages\swctools\io.py


~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Function: swctools.io._close
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Python Library Documentation: function _close in module swctools.io

_close(a: 'float', b: 'float', tol: 'float') -> 'bool'

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Function: swctools.io._coerce_int
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Python Library Documentation: function _coerce_int in module swctools.io

_coerce_int(value: 'str') -> 'int'
    Coerce an integer possibly represented as float text like '3.0'.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Function: swctools.io._iter_lines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Python Library Documentation: function _iter_lines in module swctools.io

_iter_lines(source: 'Union[str, os.PathLike, Iterable[str], io.TextIOBase]') -> 'Iterator[Tuple[int, str]]'
    Yield (1-based line number, line) from various sources.

    - Path-like or existing string path -> open and read
    - File-like object -> iterate its lines
    - Iterable of strings -> iterate
    - Other strings -> treat as content string

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Function: swctools.io.isclose
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Python Library Documentation: built-in function isclose in module math

isclose(a, b, *, rel_tol=1e-09, abs_tol=0.0)
    Determine whether two floating-point numbers are close in value.

      rel_tol
        maximum difference for being considered "close", relative to the
        magnitude of the input values
      abs_tol
        maximum difference for being considered "close", regardless of the
        magnitude of the input values

    Return True if a is close in value to b, and False otherwise.

    For the values to be considered close, the difference between them
    must be smaller than at least one of the tolerances.

    -inf, inf and NaN behave similarly to the IEEE 754 Standard.  That
    is, NaN is not close to anything, even itself.  inf and -inf are
    only close to themselves.

================================================================================
Module: swctools.model
--------------------------------------------------------------------------------
Python Library Documentation: module swctools.model in swctools

NAME
    swctools.model - SWC graph data model.

DESCRIPTION
    SWCModel stores SWC morphology as an undirected graph where nodes are SWC points,
    and the original directed parent -> child relationships are preserved in an
    internal parent map.

    Notes
    -----
    - Use `SWCModel` for topology and attribute management of parsed SWC, and for
      reconnection merges via `SWCModel.make_cycle_connections()`.

CLASSES
    networkx.classes.graph.Graph(builtins.object)
        SWCModel

    class SWCModel(networkx.classes.graph.Graph)
     |  SWCModel() -> 'None'
     |
     |  SWC morphology graph (undirected storage with directed-tree metadata).
     |
     |  Nodes are keyed by SWC id `n` and store attributes:
     |  - t: int (tag)
     |  - x, y, z: float (coordinates)
     |  - r: float (radius)
     |  - line: int (line number in source; informational)
     |
     |  Directed parent -> child relationships from the SWC are preserved via an
     |  internal parent map (`_parents`). The underlying graph is undirected. Cycle
     |  connections can be applied via `make_cycle_connections()` which may merge nodes.
     |
     |  Method resolution order:
     |      SWCModel
     |      networkx.classes.graph.Graph
     |      builtins.object
     |
     |  Methods defined here:
     |
     |  __init__(self) -> 'None'
     |      Initialize a graph with edges, name, or graph attributes.
     |
     |      Parameters
     |      ----------
     |      incoming_graph_data : input graph (optional, default: None)
     |          Data to initialize graph. If None (default) an empty
     |          graph is created.  The data can be an edge list, or any
     |          NetworkX graph object.  If the corresponding optional Python
     |          packages are installed the data can also be a 2D NumPy array, a
     |          SciPy sparse array, or a PyGraphviz graph.
     |
     |      attr : keyword arguments, optional (default= no attributes)
     |          Attributes to add to graph as key=value pairs.
     |
     |      See Also
     |      --------
     |      convert
     |
     |      Examples
     |      --------
     |      >>> G = nx.Graph()  # or DiGraph, MultiGraph, MultiDiGraph, etc
     |      >>> G = nx.Graph(name="my graph")
     |      >>> e = [(1, 2), (2, 3), (3, 4)]  # list of edges
     |      >>> G = nx.Graph(e)
     |
     |      Arbitrary graph attribute pairs (key=value) may be assigned
     |
     |      >>> G = nx.Graph(e, day="Friday")
     |      >>> G.graph
     |      {'day': 'Friday'}
     |
     |  add_junction(self, node_id: 'int | None' = None, *, t: 'int' = 0, x: 'float' = 0.0, y: 'float' = 0.0, z: 'float' = 0.0, r: 'float' = 0.0, parent: 'int | None' = None, **kwargs: 'Any') -> 'int'
     |      Add a junction (node) to the model.
     |
     |      Parameters
     |      ----------
     |      node_id: int | None
     |          Node ID to use. If None, automatically assigns the next available ID.
     |      t: int
     |          Node tag. Default 0.
     |      x, y, z: float
     |          Node coordinates. Default 0.0.
     |      r: float
     |          Node radius. Default 0.0.
     |      parent: int | None
     |          Parent node ID. If specified, creates an edge to the parent.
     |          Default None (root node).
     |      **kwargs: Any
     |          Additional node attributes.
     |
     |      Returns
     |      -------
     |      int
     |          The ID of the added node.
     |
     |  copy(self) -> "'SWCModel'"
     |      Return a shallow copy of this model (nodes/edges/attributes).
     |
     |  make_cycle_connections(self, *, validate_reconnections: 'bool' = True, float_tol: 'float' = 1e-09) -> "'SWCModel'"
     |      Return a new SWCModel with reconnection pairs merged and undirected edges across merged reps.
     |
     |      Uses reconnection pairs stored under `self.graph['reconnections']` if present.
     |      Node attributes are merged; provenance kept under `merged_ids` and `lines`.
     |      After merges, the original tree parent mapping no longer applies and parents are set to None.
     |
     |  parent_of(self, n: 'int') -> 'int | None'
     |      Return the parent id of node n from the original SWC tree (or None).
     |
     |  path_to_root(self, n: 'int') -> 'list[int]'
     |      Return the path from node n up to its root, inclusive.
     |
     |      Example: For edges 1->2->3, `path_to_root(3)` returns `[3, 2, 1]`.
     |
     |  print_attributes(self, *, node_info: 'bool' = False, edge_info: 'bool' = False) -> 'None'
     |      Print graph attributes and optional node/edge details.
     |
     |      Parameters
     |      ----------
     |      node_info: bool
     |          If True, print per-node attributes (t, x, y, z, r, line where present).
     |      edge_info: bool
     |          If True, print all edges (u -- v) with edge attributes if any.
     |
     |  remove_junction(self, node_id: 'int', *, reconnect_children: 'bool' = False) -> 'None'
     |      Remove a junction (node) from the model.
     |
     |      Parameters
     |      ----------
     |      node_id: int
     |          ID of the node to remove.
     |      reconnect_children: bool
     |          If True, reconnect children of the removed node to its parent.
     |          If False (default), children become orphaned (roots).
     |
     |  roots(self) -> 'list[int]'
     |      Return nodes with no parent in the original SWC tree.
     |
     |  scale(self, scalar: 'float') -> "'SWCModel'"
     |      Return a new model with all node coordinates and radii scaled by `scalar`.
     |
     |      Multiplies each node's `x`, `y`, `z`, and `r` by `scalar` on a copy.
     |
     |  set_tag_by_sphere(self, center: 'tuple[float, float, float] | list[float]', radius: 'float', new_tag: 'int', old_tag: 'int | None' = None, include_boundary: 'bool' = True, copy: 'bool' = False) -> "'SWCModel'"
     |      Override node 't' values for points inside a sphere.
     |
     |      Sets the tag 't' for all nodes whose Euclidean distance from
     |      `center` is less than `radius` (or equal if `include_boundary` is True).
     |
     |      If `old_tag` is specified, only nodes with that tag are modified.
     |
     |      Parameters
     |      ----------
     |      center: tuple[float, float, float] | list[float]
     |          Sphere center as (x, y, z).
     |      radius: float
     |          Sphere radius (same units as coordinates).
     |      new_tag: int
     |          Tag to assign to matching nodes.
     |      old_tag: int | None
     |          If specified, only nodes with this tag are modified.
     |      include_boundary: bool
     |          If True, include nodes exactly at distance == radius. Default True.
     |      copy: bool
     |          If True, operate on and return a copy; otherwise mutate in place and return self.
     |
     |  to_swc_file(self, path: 'str | os.PathLike[str]', *, precision: 'int' = 6, header: 'Iterable[str] | None' = None) -> 'None'
     |      Write the model to an SWC file.
     |
     |      The output uses the standard 7-column SWC format per row:
     |      "n T x y z r parent" with floats formatted to the requested precision.
     |
     |      Parameters
     |      ----------
     |      path: str | os.PathLike[str]
     |          Destination file path.
     |      precision: int
     |          Decimal places for floating-point fields (x, y, z, r). Default 6.
     |      header: Iterable[str] | None
     |          Optional additional header comment lines (without leading '#').
     |
     |  ----------------------------------------------------------------------
     |  Class methods defined here:
     |
     |  from_parse_result(result: 'SWCParseResult') -> "'SWCModel'"
     |      Build a model from a parsed SWC result.
     |
     |  from_records(records: 'Mapping[int, SWCRecord] | Iterable[SWCRecord]') -> "'SWCModel'"
     |      Build a model from SWC records.
     |
     |      Accepts either a mapping of id->record or any iterable of SWCRecord.
     |
     |  from_swc_file(source: 'str | os.PathLike[str] | Iterable[str]', *, strict: 'bool' = True, validate_reconnections: 'bool' = True, float_tol: 'float' = 1e-09) -> "'SWCModel'"
     |      Parse an SWC source then build a model.
     |
     |      The `source` is passed through to `parse_swc`, which supports a path,
     |      a file-like object, a string with the full contents, or an iterable of lines.
     |
     |  ----------------------------------------------------------------------
     |  Methods inherited from networkx.classes.graph.Graph:
     |
     |  __contains__(self, n)
     |      Returns True if n is a node, False otherwise. Use: 'n in G'.
     |
     |      Examples
     |      --------
     |      >>> G = nx.path_graph(4)  # or DiGraph, MultiGraph, MultiDiGraph, etc
     |      >>> 1 in G
     |      True
     |
     |  __getitem__(self, n)
     |      Returns a dict of neighbors of node n.  Use: 'G[n]'.
     |
     |      Parameters
     |      ----------
     |      n : node
     |         A node in the graph.
     |
     |      Returns
     |      -------
     |      adj_dict : dictionary
     |         The adjacency dictionary for nodes connected to n.
     |
     |      Notes
     |      -----
     |      G[n] is the same as G.adj[n] and similar to G.neighbors(n)
     |      (which is an iterator over G.adj[n])
     |
     |      Examples
     |      --------
     |      >>> G = nx.path_graph(4)  # or DiGraph, MultiGraph, MultiDiGraph, etc
     |      >>> G[0]
     |      AtlasView({1: {}})
     |
     |  __iter__(self)
     |      Iterate over the nodes. Use: 'for n in G'.
     |
     |      Returns
     |      -------
     |      niter : iterator
     |          An iterator over all nodes in the graph.
     |
     |      Examples
     |      --------
     |      >>> G = nx.path_graph(4)  # or DiGraph, MultiGraph, MultiDiGraph, etc
     |      >>> [n for n in G]
     |      [0, 1, 2, 3]
     |      >>> list(G)
     |      [0, 1, 2, 3]
     |
     |  __len__(self)
     |      Returns the number of nodes in the graph. Use: 'len(G)'.
     |
     |      Returns
     |      -------
     |      nnodes : int
     |          The number of nodes in the graph.
     |
     |      See Also
     |      --------
     |      number_of_nodes: identical method
     |      order: identical method
     |
     |      Examples
     |      --------
     |      >>> G = nx.path_graph(4)  # or DiGraph, MultiGraph, MultiDiGraph, etc
     |      >>> len(G)
     |      4
     |
     |  __str__(self)
     |      Returns a short summary of the graph.
     |
     |      Returns
     |      -------
     |      info : string
     |          Graph information including the graph name (if any), graph type, and the
     |          number of nodes and edges.
     |
     |      Examples
     |      --------
     |      >>> G = nx.Graph(name="foo")
     |      >>> str(G)
     |      "Graph named 'foo' with 0 nodes and 0 edges"
     |
     |      >>> G = nx.path_graph(3)
     |      >>> str(G)
     |      'Graph with 3 nodes and 2 edges'
     |
     |  add_edge(self, u_of_edge, v_of_edge, **attr)
     |      Add an edge between u and v.
     |
     |      The nodes u and v will be automatically added if they are
     |      not already in the graph.
     |
     |      Edge attributes can be specified with keywords or by directly
     |      accessing the edge's attribute dictionary. See examples below.
     |
     |      Parameters
     |      ----------
     |      u_of_edge, v_of_edge : nodes
     |          Nodes can be, for example, strings or numbers.
     |          Nodes must be hashable (and not None) Python objects.
     |      attr : keyword arguments, optional
     |          Edge data (or labels or objects) can be assigned using
     |          keyword arguments.
     |
     |      See Also
     |      --------
     |      add_edges_from : add a collection of edges
     |
     |      Notes
     |      -----
     |      Adding an edge that already exists updates the edge data.
     |
     |      Many NetworkX algorithms designed for weighted graphs use
     |      an edge attribute (by default `weight`) to hold a numerical value.
     |
     |      Examples
     |      --------
     |      The following all add the edge e=(1, 2) to graph G:
     |
     |      >>> G = nx.Graph()  # or DiGraph, MultiGraph, MultiDiGraph, etc
     |      >>> e = (1, 2)
     |      >>> G.add_edge(1, 2)  # explicit two-node form
     |      >>> G.add_edge(*e)  # single edge as tuple of two nodes
     |      >>> G.add_edges_from([(1, 2)])  # add edges from iterable container
     |
     |      Associate data to edges using keywords:
     |
     |      >>> G.add_edge(1, 2, weight=3)
     |      >>> G.add_edge(1, 3, weight=7, capacity=15, length=342.7)
     |
     |      For non-string attribute keys, use subscript notation.
     |
     |      >>> G.add_edge(1, 2)
     |      >>> G[1][2].update({0: 5})
     |      >>> G.edges[1, 2].update({0: 5})
     |
     |  add_edges_from(self, ebunch_to_add, **attr)
     |      Add all the edges in ebunch_to_add.
     |
     |      Parameters
     |      ----------
     |      ebunch_to_add : container of edges
     |          Each edge given in the container will be added to the
     |          graph. The edges must be given as 2-tuples (u, v) or
     |          3-tuples (u, v, d) where d is a dictionary containing edge data.
     |      attr : keyword arguments, optional
     |          Edge data (or labels or objects) can be assigned using
     |          keyword arguments.
     |
     |      See Also
     |      --------
     |      add_edge : add a single edge
     |      add_weighted_edges_from : convenient way to add weighted edges
     |
     |      Notes
     |      -----
     |      Adding the same edge twice has no effect but any edge data
     |      will be updated when each duplicate edge is added.
     |
     |      Edge attributes specified in an ebunch take precedence over
     |      attributes specified via keyword arguments.
     |
     |      When adding edges from an iterator over the graph you are changing,
     |      a `RuntimeError` can be raised with message:
     |      `RuntimeError: dictionary changed size during iteration`. This
     |      happens when the graph's underlying dictionary is modified during
     |      iteration. To avoid this error, evaluate the iterator into a separate
     |      object, e.g. by using `list(iterator_of_edges)`, and pass this
     |      object to `G.add_edges_from`.
     |
     |      Examples
     |      --------
     |      >>> G = nx.Graph()  # or DiGraph, MultiGraph, MultiDiGraph, etc
     |      >>> G.add_edges_from([(0, 1), (1, 2)])  # using a list of edge tuples
     |      >>> e = zip(range(0, 3), range(1, 4))
     |      >>> G.add_edges_from(e)  # Add the path graph 0-1-2-3
     |
     |      Associate data to edges
     |
     |      >>> G.add_edges_from([(1, 2), (2, 3)], weight=3)
     |      >>> G.add_edges_from([(3, 4), (1, 4)], label="WN2898")
     |
     |      Evaluate an iterator over a graph if using it to modify the same graph
     |
     |      >>> G = nx.Graph([(1, 2), (2, 3), (3, 4)])
     |      >>> # Grow graph by one new node, adding edges to all existing nodes.
     |      >>> # wrong way - will raise RuntimeError
     |      >>> # G.add_edges_from(((5, n) for n in G.nodes))
     |      >>> # correct way - note that there will be no self-edge for node 5
     |      >>> G.add_edges_from(list((5, n) for n in G.nodes))
     |
     |  add_node(self, node_for_adding, **attr)
     |      Add a single node `node_for_adding` and update node attributes.
     |
     |      Parameters
     |      ----------
     |      node_for_adding : node
     |          A node can be any hashable Python object except None.
     |      attr : keyword arguments, optional
     |          Set or change node attributes using key=value.
     |
     |      See Also
     |      --------
     |      add_nodes_from
     |
     |      Examples
     |      --------
     |      >>> G = nx.Graph()  # or DiGraph, MultiGraph, MultiDiGraph, etc
     |      >>> G.add_node(1)
     |      >>> G.add_node("Hello")
     |      >>> K3 = nx.Graph([(0, 1), (1, 2), (2, 0)])
     |      >>> G.add_node(K3)
     |      >>> G.number_of_nodes()
     |      3
     |
     |      Use keywords set/change node attributes:
     |
     |      >>> G.add_node(1, size=10)
     |      >>> G.add_node(3, weight=0.4, UTM=("13S", 382871, 3972649))
     |
     |      Notes
     |      -----
     |      A hashable object is one that can be used as a key in a Python
     |      dictionary. This includes strings, numbers, tuples of strings
     |      and numbers, etc.
     |
     |      On many platforms hashable items also include mutables such as
     |      NetworkX Graphs, though one should be careful that the hash
     |      doesn't change on mutables.
     |
     |  add_nodes_from(self, nodes_for_adding, **attr)
     |      Add multiple nodes.
     |
     |      Parameters
     |      ----------
     |      nodes_for_adding : iterable container
     |          A container of nodes (list, dict, set, etc.).
     |          OR
     |          A container of (node, attribute dict) tuples.
     |          Node attributes are updated using the attribute dict.
     |      attr : keyword arguments, optional (default= no attributes)
     |          Update attributes for all nodes in nodes.
     |          Node attributes specified in nodes as a tuple take
     |          precedence over attributes specified via keyword arguments.
     |
     |      See Also
     |      --------
     |      add_node
     |
     |      Notes
     |      -----
     |      When adding nodes from an iterator over the graph you are changing,
     |      a `RuntimeError` can be raised with message:
     |      `RuntimeError: dictionary changed size during iteration`. This
     |      happens when the graph's underlying dictionary is modified during
     |      iteration. To avoid this error, evaluate the iterator into a separate
     |      object, e.g. by using `list(iterator_of_nodes)`, and pass this
     |      object to `G.add_nodes_from`.
     |
     |      Examples
     |      --------
     |      >>> G = nx.Graph()  # or DiGraph, MultiGraph, MultiDiGraph, etc
     |      >>> G.add_nodes_from("Hello")
     |      >>> K3 = nx.Graph([(0, 1), (1, 2), (2, 0)])
     |      >>> G.add_nodes_from(K3)
     |      >>> sorted(G.nodes(), key=str)
     |      [0, 1, 2, 'H', 'e', 'l', 'o']
     |
     |      Use keywords to update specific node attributes for every node.
     |
     |      >>> G.add_nodes_from([1, 2], size=10)
     |      >>> G.add_nodes_from([3, 4], weight=0.4)
     |
     |      Use (node, attrdict) tuples to update attributes for specific nodes.
     |
     |      >>> G.add_nodes_from([(1, dict(size=11)), (2, {"color": "blue"})])
     |      >>> G.nodes[1]["size"]
     |      11
     |      >>> H = nx.Graph()
     |      >>> H.add_nodes_from(G.nodes(data=True))
     |      >>> H.nodes[1]["size"]
     |      11
     |
     |      Evaluate an iterator over a graph if using it to modify the same graph
     |
     |      >>> G = nx.Graph([(0, 1), (1, 2), (3, 4)])
     |      >>> # wrong way - will raise RuntimeError
     |      >>> # G.add_nodes_from(n + 1 for n in G.nodes)
     |      >>> # correct way
     |      >>> G.add_nodes_from(list(n + 1 for n in G.nodes))
     |
     |  add_weighted_edges_from(self, ebunch_to_add, weight='weight', **attr)
     |      Add weighted edges in `ebunch_to_add` with specified weight attr
     |
     |      Parameters
     |      ----------
     |      ebunch_to_add : container of edges
     |          Each edge given in the list or container will be added
     |          to the graph. The edges must be given as 3-tuples (u, v, w)
     |          where w is a number.
     |      weight : string, optional (default= 'weight')
     |          The attribute name for the edge weights to be added.
     |      attr : keyword arguments, optional (default= no attributes)
     |          Edge attributes to add/update for all edges.
     |
     |      See Also
     |      --------
     |      add_edge : add a single edge
     |      add_edges_from : add multiple edges
     |
     |      Notes
     |      -----
     |      Adding the same edge twice for Graph/DiGraph simply updates
     |      the edge data. For MultiGraph/MultiDiGraph, duplicate edges
     |      are stored.
     |
     |      When adding edges from an iterator over the graph you are changing,
     |      a `RuntimeError` can be raised with message:
     |      `RuntimeError: dictionary changed size during iteration`. This
     |      happens when the graph's underlying dictionary is modified during
     |      iteration. To avoid this error, evaluate the iterator into a separate
     |      object, e.g. by using `list(iterator_of_edges)`, and pass this
     |      object to `G.add_weighted_edges_from`.
     |
     |      Examples
     |      --------
     |      >>> G = nx.Graph()  # or DiGraph, MultiGraph, MultiDiGraph, etc
     |      >>> G.add_weighted_edges_from([(0, 1, 3.0), (1, 2, 7.5)])
     |
     |      Evaluate an iterator over edges before passing it
     |
     |      >>> G = nx.Graph([(1, 2), (2, 3), (3, 4)])
     |      >>> weight = 0.1
     |      >>> # Grow graph by one new node, adding edges to all existing nodes.
     |      >>> # wrong way - will raise RuntimeError
     |      >>> # G.add_weighted_edges_from(((5, n, weight) for n in G.nodes))
     |      >>> # correct way - note that there will be no self-edge for node 5
     |      >>> G.add_weighted_edges_from(list((5, n, weight) for n in G.nodes))
     |
     |  adj = <functools.cached_property object>
     |      Graph adjacency object holding the neighbors of each node.
     |
     |      This object is a read-only dict-like structure with node keys
     |      and neighbor-dict values.  The neighbor-dict is keyed by neighbor
     |      to the edge-data-dict.  So `G.adj[3][2]['color'] = 'blue'` sets
     |      the color of the edge `(3, 2)` to `"blue"`.
     |
     |      Iterating over G.adj behaves like a dict. Useful idioms include
     |      `for nbr, datadict in G.adj[n].items():`.
     |
     |      The neighbor information is also provided by subscripting the graph.
     |      So `for nbr, foovalue in G[node].data('foo', default=1):` works.
     |
     |      For directed graphs, `G.adj` holds outgoing (successor) info.
     |
     |  adjacency(self)
     |      Returns an iterator over (node, adjacency dict) tuples for all nodes.
     |
     |      For directed graphs, only outgoing neighbors/adjacencies are included.
     |
     |      Returns
     |      -------
     |      adj_iter : iterator
     |         An iterator over (node, adjacency dictionary) for all nodes in
     |         the graph.
     |
     |      Examples
     |      --------
     |      >>> G = nx.path_graph(4)  # or DiGraph, MultiGraph, MultiDiGraph, etc
     |      >>> [(n, nbrdict) for n, nbrdict in G.adjacency()]
     |      [(0, {1: {}}), (1, {0: {}, 2: {}}), (2, {1: {}, 3: {}}), (3, {2: {}})]
     |
     |  clear(self)
     |      Remove all nodes and edges from the graph.
     |
     |      This also removes the name, and all graph, node, and edge attributes.
     |
     |      Examples
     |      --------
     |      >>> G = nx.path_graph(4)  # or DiGraph, MultiGraph, MultiDiGraph, etc
     |      >>> G.clear()
     |      >>> list(G.nodes)
     |      []
     |      >>> list(G.edges)
     |      []
     |
     |  clear_edges(self)
     |      Remove all edges from the graph without altering nodes.
     |
     |      Examples
     |      --------
     |      >>> G = nx.path_graph(4)  # or DiGraph, MultiGraph, MultiDiGraph, etc
     |      >>> G.clear_edges()
     |      >>> list(G.nodes)
     |      [0, 1, 2, 3]
     |      >>> list(G.edges)
     |      []
     |
     |  degree = <functools.cached_property object>
     |      A DegreeView for the Graph as G.degree or G.degree().
     |
     |      The node degree is the number of edges adjacent to the node.
     |      The weighted node degree is the sum of the edge weights for
     |      edges incident to that node.
     |
     |      This object provides an iterator for (node, degree) as well as
     |      lookup for the degree for a single node.
     |
     |      Parameters
     |      ----------
     |      nbunch : single node, container, or all nodes (default= all nodes)
     |          The view will only report edges incident to these nodes.
     |
     |      weight : string or None, optional (default=None)
     |         The name of an edge attribute that holds the numerical value used
     |         as a weight.  If None, then each edge has weight 1.
     |         The degree is the sum of the edge weights adjacent to the node.
     |
     |      Returns
     |      -------
     |      DegreeView or int
     |          If multiple nodes are requested (the default), returns a `DegreeView`
     |          mapping nodes to their degree.
     |          If a single node is requested, returns the degree of the node as an integer.
     |
     |      Examples
     |      --------
     |      >>> G = nx.path_graph(4)  # or DiGraph, MultiGraph, MultiDiGraph, etc
     |      >>> G.degree[0]  # node 0 has degree 1
     |      1
     |      >>> list(G.degree([0, 1, 2]))
     |      [(0, 1), (1, 2), (2, 2)]
     |
     |  edge_subgraph(self, edges)
     |      Returns the subgraph induced by the specified edges.
     |
     |      The induced subgraph contains each edge in `edges` and each
     |      node incident to any one of those edges.
     |
     |      Parameters
     |      ----------
     |      edges : iterable
     |          An iterable of edges in this graph.
     |
     |      Returns
     |      -------
     |      G : Graph
     |          An edge-induced subgraph of this graph with the same edge
     |          attributes.
     |
     |      Notes
     |      -----
     |      The graph, edge, and node attributes in the returned subgraph
     |      view are references to the corresponding attributes in the original
     |      graph. The view is read-only.
     |
     |      To create a full graph version of the subgraph with its own copy
     |      of the edge or node attributes, use::
     |
     |          G.edge_subgraph(edges).copy()
     |
     |      Examples
     |      --------
     |      >>> G = nx.path_graph(5)
     |      >>> H = G.edge_subgraph([(0, 1), (3, 4)])
     |      >>> list(H.nodes)
     |      [0, 1, 3, 4]
     |      >>> list(H.edges)
     |      [(0, 1), (3, 4)]
     |
     |  edges = <functools.cached_property object>
     |      An EdgeView of the Graph as G.edges or G.edges().
     |
     |      edges(self, nbunch=None, data=False, default=None)
     |
     |      The EdgeView provides set-like operations on the edge-tuples
     |      as well as edge attribute lookup. When called, it also provides
     |      an EdgeDataView object which allows control of access to edge
     |      attributes (but does not provide set-like operations).
     |      Hence, `G.edges[u, v]['color']` provides the value of the color
     |      attribute for edge `(u, v)` while
     |      `for (u, v, c) in G.edges.data('color', default='red'):`
     |      iterates through all the edges yielding the color attribute
     |      with default `'red'` if no color attribute exists.
     |
     |      Parameters
     |      ----------
     |      nbunch : single node, container, or all nodes (default= all nodes)
     |          The view will only report edges from these nodes.
     |      data : string or bool, optional (default=False)
     |          The edge attribute returned in 3-tuple (u, v, ddict[data]).
     |          If True, return edge attribute dict in 3-tuple (u, v, ddict).
     |          If False, return 2-tuple (u, v).
     |      default : value, optional (default=None)
     |          Value used for edges that don't have the requested attribute.
     |          Only relevant if data is not True or False.
     |
     |      Returns
     |      -------
     |      edges : EdgeView
     |          A view of edge attributes, usually it iterates over (u, v)
     |          or (u, v, d) tuples of edges, but can also be used for
     |          attribute lookup as `edges[u, v]['foo']`.
     |
     |      Notes
     |      -----
     |      Nodes in nbunch that are not in the graph will be (quietly) ignored.
     |      For directed graphs this returns the out-edges.
     |
     |      Examples
     |      --------
     |      >>> G = nx.path_graph(3)  # or MultiGraph, etc
     |      >>> G.add_edge(2, 3, weight=5)
     |      >>> [e for e in G.edges]
     |      [(0, 1), (1, 2), (2, 3)]
     |      >>> G.edges.data()  # default data is {} (empty dict)
     |      EdgeDataView([(0, 1, {}), (1, 2, {}), (2, 3, {'weight': 5})])
     |      >>> G.edges.data("weight", default=1)
     |      EdgeDataView([(0, 1, 1), (1, 2, 1), (2, 3, 5)])
     |      >>> G.edges([0, 3])  # only edges from these nodes
     |      EdgeDataView([(0, 1), (3, 2)])
     |      >>> G.edges(0)  # only edges from node 0
     |      EdgeDataView([(0, 1)])
     |
     |  get_edge_data(self, u, v, default=None)
     |      Returns the attribute dictionary associated with edge (u, v).
     |
     |      This is identical to `G[u][v]` except the default is returned
     |      instead of an exception if the edge doesn't exist.
     |
     |      Parameters
     |      ----------
     |      u, v : nodes
     |      default:  any Python object (default=None)
     |          Value to return if the edge (u, v) is not found.
     |
     |      Returns
     |      -------
     |      edge_dict : dictionary
     |          The edge attribute dictionary.
     |
     |      Examples
     |      --------
     |      >>> G = nx.path_graph(4)  # or DiGraph, MultiGraph, MultiDiGraph, etc
     |      >>> G[0][1]
     |      {}
     |
     |      Warning: Assigning to `G[u][v]` is not permitted.
     |      But it is safe to assign attributes `G[u][v]['foo']`
     |
     |      >>> G[0][1]["weight"] = 7
     |      >>> G[0][1]["weight"]
     |      7
     |      >>> G[1][0]["weight"]
     |      7
     |
     |      >>> G = nx.path_graph(4)  # or DiGraph, MultiGraph, MultiDiGraph, etc
     |      >>> G.get_edge_data(0, 1)  # default edge data is {}
     |      {}
     |      >>> e = (0, 1)
     |      >>> G.get_edge_data(*e)  # tuple form
     |      {}
     |      >>> G.get_edge_data("a", "b", default=0)  # edge not in graph, return 0
     |      0
     |
     |  has_edge(self, u, v)
     |      Returns True if the edge (u, v) is in the graph.
     |
     |      This is the same as `v in G[u]` without KeyError exceptions.
     |
     |      Parameters
     |      ----------
     |      u, v : nodes
     |          Nodes can be, for example, strings or numbers.
     |          Nodes must be hashable (and not None) Python objects.
     |
     |      Returns
     |      -------
     |      edge_ind : bool
     |          True if edge is in the graph, False otherwise.
     |
     |      Examples
     |      --------
     |      >>> G = nx.path_graph(4)  # or DiGraph, MultiGraph, MultiDiGraph, etc
     |      >>> G.has_edge(0, 1)  # using two nodes
     |      True
     |      >>> e = (0, 1)
     |      >>> G.has_edge(*e)  #  e is a 2-tuple (u, v)
     |      True
     |      >>> e = (0, 1, {"weight": 7})
     |      >>> G.has_edge(*e[:2])  # e is a 3-tuple (u, v, data_dictionary)
     |      True
     |
     |      The following syntax are equivalent:
     |
     |      >>> G.has_edge(0, 1)
     |      True
     |      >>> 1 in G[0]  # though this gives KeyError if 0 not in G
     |      True
     |
     |  has_node(self, n)
     |      Returns True if the graph contains the node n.
     |
     |      Identical to `n in G`
     |
     |      Parameters
     |      ----------
     |      n : node
     |
     |      Examples
     |      --------
     |      >>> G = nx.path_graph(3)  # or DiGraph, MultiGraph, MultiDiGraph, etc
     |      >>> G.has_node(0)
     |      True
     |
     |      It is more readable and simpler to use
     |
     |      >>> 0 in G
     |      True
     |
     |  is_directed(self)
     |      Returns True if graph is directed, False otherwise.
     |
     |  is_multigraph(self)
     |      Returns True if graph is a multigraph, False otherwise.
     |
     |  nbunch_iter(self, nbunch=None)
     |      Returns an iterator over nodes contained in nbunch that are
     |      also in the graph.
     |
     |      The nodes in an iterable nbunch are checked for membership in the graph
     |      and if not are silently ignored.
     |
     |      Parameters
     |      ----------
     |      nbunch : single node, container, or all nodes (default= all nodes)
     |          The view will only report edges incident to these nodes.
     |
     |      Returns
     |      -------
     |      niter : iterator
     |          An iterator over nodes in nbunch that are also in the graph.
     |          If nbunch is None, iterate over all nodes in the graph.
     |
     |      Raises
     |      ------
     |      NetworkXError
     |          If nbunch is not a node or sequence of nodes.
     |          If a node in nbunch is not hashable.
     |
     |      See Also
     |      --------
     |      Graph.__iter__
     |
     |      Notes
     |      -----
     |      When nbunch is an iterator, the returned iterator yields values
     |      directly from nbunch, becoming exhausted when nbunch is exhausted.
     |
     |      To test whether nbunch is a single node, one can use
     |      "if nbunch in self:", even after processing with this routine.
     |
     |      If nbunch is not a node or a (possibly empty) sequence/iterator
     |      or None, a :exc:`NetworkXError` is raised.  Also, if any object in
     |      nbunch is not hashable, a :exc:`NetworkXError` is raised.
     |
     |  neighbors(self, n)
     |      Returns an iterator over all neighbors of node n.
     |
     |      This is identical to `iter(G[n])`
     |
     |      Parameters
     |      ----------
     |      n : node
     |         A node in the graph
     |
     |      Returns
     |      -------
     |      neighbors : iterator
     |          An iterator over all neighbors of node n
     |
     |      Raises
     |      ------
     |      NetworkXError
     |          If the node n is not in the graph.
     |
     |      Examples
     |      --------
     |      >>> G = nx.path_graph(4)  # or DiGraph, MultiGraph, MultiDiGraph, etc
     |      >>> [n for n in G.neighbors(0)]
     |      [1]
     |
     |      Notes
     |      -----
     |      Alternate ways to access the neighbors are ``G.adj[n]`` or ``G[n]``:
     |
     |      >>> G = nx.Graph()  # or DiGraph, MultiGraph, MultiDiGraph, etc
     |      >>> G.add_edge("a", "b", weight=7)
     |      >>> G["a"]
     |      AtlasView({'b': {'weight': 7}})
     |      >>> G = nx.path_graph(4)
     |      >>> [n for n in G[0]]
     |      [1]
     |
     |  nodes = <functools.cached_property object>
     |      A NodeView of the Graph as G.nodes or G.nodes().
     |
     |      Can be used as `G.nodes` for data lookup and for set-like operations.
     |      Can also be used as `G.nodes(data='color', default=None)` to return a
     |      NodeDataView which reports specific node data but no set operations.
     |      It presents a dict-like interface as well with `G.nodes.items()`
     |      iterating over `(node, nodedata)` 2-tuples and `G.nodes[3]['foo']`
     |      providing the value of the `foo` attribute for node `3`. In addition,
     |      a view `G.nodes.data('foo')` provides a dict-like interface to the
     |      `foo` attribute of each node. `G.nodes.data('foo', default=1)`
     |      provides a default for nodes that do not have attribute `foo`.
     |
     |      Parameters
     |      ----------
     |      data : string or bool, optional (default=False)
     |          The node attribute returned in 2-tuple (n, ddict[data]).
     |          If True, return entire node attribute dict as (n, ddict).
     |          If False, return just the nodes n.
     |
     |      default : value, optional (default=None)
     |          Value used for nodes that don't have the requested attribute.
     |          Only relevant if data is not True or False.
     |
     |      Returns
     |      -------
     |      NodeView
     |          Allows set-like operations over the nodes as well as node
     |          attribute dict lookup and calling to get a NodeDataView.
     |          A NodeDataView iterates over `(n, data)` and has no set operations.
     |          A NodeView iterates over `n` and includes set operations.
     |
     |          When called, if data is False, an iterator over nodes.
     |          Otherwise an iterator of 2-tuples (node, attribute value)
     |          where the attribute is specified in `data`.
     |          If data is True then the attribute becomes the
     |          entire data dictionary.
     |
     |      Notes
     |      -----
     |      If your node data is not needed, it is simpler and equivalent
     |      to use the expression ``for n in G``, or ``list(G)``.
     |
     |      Examples
     |      --------
     |      There are two simple ways of getting a list of all nodes in the graph:
     |
     |      >>> G = nx.path_graph(3)
     |      >>> list(G.nodes)
     |      [0, 1, 2]
     |      >>> list(G)
     |      [0, 1, 2]
     |
     |      To get the node data along with the nodes:
     |
     |      >>> G.add_node(1, time="5pm")
     |      >>> G.nodes[0]["foo"] = "bar"
     |      >>> list(G.nodes(data=True))
     |      [(0, {'foo': 'bar'}), (1, {'time': '5pm'}), (2, {})]
     |      >>> list(G.nodes.data())
     |      [(0, {'foo': 'bar'}), (1, {'time': '5pm'}), (2, {})]
     |
     |      >>> list(G.nodes(data="foo"))
     |      [(0, 'bar'), (1, None), (2, None)]
     |      >>> list(G.nodes.data("foo"))
     |      [(0, 'bar'), (1, None), (2, None)]
     |
     |      >>> list(G.nodes(data="time"))
     |      [(0, None), (1, '5pm'), (2, None)]
     |      >>> list(G.nodes.data("time"))
     |      [(0, None), (1, '5pm'), (2, None)]
     |
     |      >>> list(G.nodes(data="time", default="Not Available"))
     |      [(0, 'Not Available'), (1, '5pm'), (2, 'Not Available')]
     |      >>> list(G.nodes.data("time", default="Not Available"))
     |      [(0, 'Not Available'), (1, '5pm'), (2, 'Not Available')]
     |
     |      If some of your nodes have an attribute and the rest are assumed
     |      to have a default attribute value you can create a dictionary
     |      from node/attribute pairs using the `default` keyword argument
     |      to guarantee the value is never None::
     |
     |          >>> G = nx.Graph()
     |          >>> G.add_node(0)
     |          >>> G.add_node(1, weight=2)
     |          >>> G.add_node(2, weight=3)
     |          >>> dict(G.nodes(data="weight", default=1))
     |          {0: 1, 1: 2, 2: 3}
     |
     |  number_of_edges(self, u=None, v=None)
     |      Returns the number of edges between two nodes.
     |
     |      Parameters
     |      ----------
     |      u, v : nodes, optional (default=all edges)
     |          If u and v are specified, return the number of edges between
     |          u and v. Otherwise return the total number of all edges.
     |
     |      Returns
     |      -------
     |      nedges : int
     |          The number of edges in the graph.  If nodes `u` and `v` are
     |          specified return the number of edges between those nodes. If
     |          the graph is directed, this only returns the number of edges
     |          from `u` to `v`.
     |
     |      See Also
     |      --------
     |      size
     |
     |      Examples
     |      --------
     |      For undirected graphs, this method counts the total number of
     |      edges in the graph:
     |
     |      >>> G = nx.path_graph(4)
     |      >>> G.number_of_edges()
     |      3
     |
     |      If you specify two nodes, this counts the total number of edges
     |      joining the two nodes:
     |
     |      >>> G.number_of_edges(0, 1)
     |      1
     |
     |      For directed graphs, this method can count the total number of
     |      directed edges from `u` to `v`:
     |
     |      >>> G = nx.DiGraph()
     |      >>> G.add_edge(0, 1)
     |      >>> G.add_edge(1, 0)
     |      >>> G.number_of_edges(0, 1)
     |      1
     |
     |  number_of_nodes(self)
     |      Returns the number of nodes in the graph.
     |
     |      Returns
     |      -------
     |      nnodes : int
     |          The number of nodes in the graph.
     |
     |      See Also
     |      --------
     |      order: identical method
     |      __len__: identical method
     |
     |      Examples
     |      --------
     |      >>> G = nx.path_graph(3)  # or DiGraph, MultiGraph, MultiDiGraph, etc
     |      >>> G.number_of_nodes()
     |      3
     |
     |  order(self)
     |      Returns the number of nodes in the graph.
     |
     |      Returns
     |      -------
     |      nnodes : int
     |          The number of nodes in the graph.
     |
     |      See Also
     |      --------
     |      number_of_nodes: identical method
     |      __len__: identical method
     |
     |      Examples
     |      --------
     |      >>> G = nx.path_graph(3)  # or DiGraph, MultiGraph, MultiDiGraph, etc
     |      >>> G.order()
     |      3
     |
     |  remove_edge(self, u, v)
     |      Remove the edge between u and v.
     |
     |      Parameters
     |      ----------
     |      u, v : nodes
     |          Remove the edge between nodes u and v.
     |
     |      Raises
     |      ------
     |      NetworkXError
     |          If there is not an edge between u and v.
     |
     |      See Also
     |      --------
     |      remove_edges_from : remove a collection of edges
     |
     |      Examples
     |      --------
     |      >>> G = nx.path_graph(4)  # or DiGraph, etc
     |      >>> G.remove_edge(0, 1)
     |      >>> e = (1, 2)
     |      >>> G.remove_edge(*e)  # unpacks e from an edge tuple
     |      >>> e = (2, 3, {"weight": 7})  # an edge with attribute data
     |      >>> G.remove_edge(*e[:2])  # select first part of edge tuple
     |
     |  remove_edges_from(self, ebunch)
     |      Remove all edges specified in ebunch.
     |
     |      Parameters
     |      ----------
     |      ebunch: list or container of edge tuples
     |          Each edge given in the list or container will be removed
     |          from the graph. The edges can be:
     |
     |              - 2-tuples (u, v) edge between u and v.
     |              - 3-tuples (u, v, k) where k is ignored.
     |
     |      See Also
     |      --------
     |      remove_edge : remove a single edge
     |
     |      Notes
     |      -----
     |      Will fail silently if an edge in ebunch is not in the graph.
     |
     |      Examples
     |      --------
     |      >>> G = nx.path_graph(4)  # or DiGraph, MultiGraph, MultiDiGraph, etc
     |      >>> ebunch = [(1, 2), (2, 3)]
     |      >>> G.remove_edges_from(ebunch)
     |
     |  remove_node(self, n)
     |      Remove node n.
     |
     |      Removes the node n and all adjacent edges.
     |      Attempting to remove a nonexistent node will raise an exception.
     |
     |      Parameters
     |      ----------
     |      n : node
     |         A node in the graph
     |
     |      Raises
     |      ------
     |      NetworkXError
     |         If n is not in the graph.
     |
     |      See Also
     |      --------
     |      remove_nodes_from
     |
     |      Examples
     |      --------
     |      >>> G = nx.path_graph(3)  # or DiGraph, MultiGraph, MultiDiGraph, etc
     |      >>> list(G.edges)
     |      [(0, 1), (1, 2)]
     |      >>> G.remove_node(1)
     |      >>> list(G.edges)
     |      []
     |
     |  remove_nodes_from(self, nodes)
     |      Remove multiple nodes.
     |
     |      Parameters
     |      ----------
     |      nodes : iterable container
     |          A container of nodes (list, dict, set, etc.).  If a node
     |          in the container is not in the graph it is silently
     |          ignored.
     |
     |      See Also
     |      --------
     |      remove_node
     |
     |      Notes
     |      -----
     |      When removing nodes from an iterator over the graph you are changing,
     |      a `RuntimeError` will be raised with message:
     |      `RuntimeError: dictionary changed size during iteration`. This
     |      happens when the graph's underlying dictionary is modified during
     |      iteration. To avoid this error, evaluate the iterator into a separate
     |      object, e.g. by using `list(iterator_of_nodes)`, and pass this
     |      object to `G.remove_nodes_from`.
     |
     |      Examples
     |      --------
     |      >>> G = nx.path_graph(3)  # or DiGraph, MultiGraph, MultiDiGraph, etc
     |      >>> e = list(G.nodes)
     |      >>> e
     |      [0, 1, 2]
     |      >>> G.remove_nodes_from(e)
     |      >>> list(G.nodes)
     |      []
     |
     |      Evaluate an iterator over a graph if using it to modify the same graph
     |
     |      >>> G = nx.Graph([(0, 1), (1, 2), (3, 4)])
     |      >>> # this command will fail, as the graph's dict is modified during iteration
     |      >>> # G.remove_nodes_from(n for n in G.nodes if n < 2)
     |      >>> # this command will work, since the dictionary underlying graph is not modified
     |      >>> G.remove_nodes_from(list(n for n in G.nodes if n < 2))
     |
     |  size(self, weight=None)
     |      Returns the number of edges or total of all edge weights.
     |
     |      Parameters
     |      ----------
     |      weight : string or None, optional (default=None)
     |          The edge attribute that holds the numerical value used
     |          as a weight. If None, then each edge has weight 1.
     |
     |      Returns
     |      -------
     |      size : numeric
     |          The number of edges or
     |          (if weight keyword is provided) the total weight sum.
     |
     |          If weight is None, returns an int. Otherwise a float
     |          (or more general numeric if the weights are more general).
     |
     |      See Also
     |      --------
     |      number_of_edges
     |
     |      Examples
     |      --------
     |      >>> G = nx.path_graph(4)  # or DiGraph, MultiGraph, MultiDiGraph, etc
     |      >>> G.size()
     |      3
     |
     |      >>> G = nx.Graph()  # or DiGraph, MultiGraph, MultiDiGraph, etc
     |      >>> G.add_edge("a", "b", weight=2)
     |      >>> G.add_edge("b", "c", weight=4)
     |      >>> G.size()
     |      2
     |      >>> G.size(weight="weight")
     |      6.0
     |
     |  subgraph(self, nodes)
     |      Returns a SubGraph view of the subgraph induced on `nodes`.
     |
     |      The induced subgraph of the graph contains the nodes in `nodes`
     |      and the edges between those nodes.
     |
     |      Parameters
     |      ----------
     |      nodes : list, iterable
     |          A container of nodes which will be iterated through once.
     |
     |      Returns
     |      -------
     |      G : SubGraph View
     |          A subgraph view of the graph. The graph structure cannot be
     |          changed but node/edge attributes can and are shared with the
     |          original graph.
     |
     |      Notes
     |      -----
     |      The graph, edge and node attributes are shared with the original graph.
     |      Changes to the graph structure is ruled out by the view, but changes
     |      to attributes are reflected in the original graph.
     |
     |      To create a subgraph with its own copy of the edge/node attributes use:
     |      G.subgraph(nodes).copy()
     |
     |      For an inplace reduction of a graph to a subgraph you can remove nodes:
     |      G.remove_nodes_from([n for n in G if n not in set(nodes)])
     |
     |      Subgraph views are sometimes NOT what you want. In most cases where
     |      you want to do more than simply look at the induced edges, it makes
     |      more sense to just create the subgraph as its own graph with code like:
     |
     |      ::
     |
     |          # Create a subgraph SG based on a (possibly multigraph) G
     |          SG = G.__class__()
     |          SG.add_nodes_from((n, G.nodes[n]) for n in largest_wcc)
     |          if SG.is_multigraph():
     |              SG.add_edges_from(
     |                  (n, nbr, key, d)
     |                  for n, nbrs in G.adj.items()
     |                  if n in largest_wcc
     |                  for nbr, keydict in nbrs.items()
     |                  if nbr in largest_wcc
     |                  for key, d in keydict.items()
     |              )
     |          else:
     |              SG.add_edges_from(
     |                  (n, nbr, d)
     |                  for n, nbrs in G.adj.items()
     |                  if n in largest_wcc
     |                  for nbr, d in nbrs.items()
     |                  if nbr in largest_wcc
     |              )
     |          SG.graph.update(G.graph)
     |
     |      Subgraphs are not guaranteed to preserve the order of nodes or edges
     |      as they appear in the original graph. For example:
     |
     |      >>> G = nx.Graph()
     |      >>> G.add_nodes_from(reversed(range(10)))
     |      >>> list(G)
     |      [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
     |      >>> list(G.subgraph([1, 3, 2]))
     |      [1, 2, 3]
     |
     |      Examples
     |      --------
     |      >>> G = nx.path_graph(4)  # or DiGraph, MultiGraph, MultiDiGraph, etc
     |      >>> H = G.subgraph([0, 1, 2])
     |      >>> list(H.edges)
     |      [(0, 1), (1, 2)]
     |
     |  to_directed(self, as_view=False)
     |      Returns a directed representation of the graph.
     |
     |      Returns
     |      -------
     |      G : DiGraph
     |          A directed graph with the same name, same nodes, and with
     |          each edge (u, v, data) replaced by two directed edges
     |          (u, v, data) and (v, u, data).
     |
     |      Notes
     |      -----
     |      This returns a "deepcopy" of the edge, node, and
     |      graph attributes which attempts to completely copy
     |      all of the data and references.
     |
     |      This is in contrast to the similar D=DiGraph(G) which returns a
     |      shallow copy of the data.
     |
     |      See the Python copy module for more information on shallow
     |      and deep copies, https://docs.python.org/3/library/copy.html.
     |
     |      Warning: If you have subclassed Graph to use dict-like objects
     |      in the data structure, those changes do not transfer to the
     |      DiGraph created by this method.
     |
     |      Examples
     |      --------
     |      >>> G = nx.Graph()  # or MultiGraph, etc
     |      >>> G.add_edge(0, 1)
     |      >>> H = G.to_directed()
     |      >>> list(H.edges)
     |      [(0, 1), (1, 0)]
     |
     |      If already directed, return a (deep) copy
     |
     |      >>> G = nx.DiGraph()  # or MultiDiGraph, etc
     |      >>> G.add_edge(0, 1)
     |      >>> H = G.to_directed()
     |      >>> list(H.edges)
     |      [(0, 1)]
     |
     |  to_directed_class(self)
     |      Returns the class to use for empty directed copies.
     |
     |      If you subclass the base classes, use this to designate
     |      what directed class to use for `to_directed()` copies.
     |
     |  to_undirected(self, as_view=False)
     |      Returns an undirected copy of the graph.
     |
     |      Parameters
     |      ----------
     |      as_view : bool (optional, default=False)
     |        If True return a view of the original undirected graph.
     |
     |      Returns
     |      -------
     |      G : Graph/MultiGraph
     |          A deepcopy of the graph.
     |
     |      See Also
     |      --------
     |      Graph, copy, add_edge, add_edges_from
     |
     |      Notes
     |      -----
     |      This returns a "deepcopy" of the edge, node, and
     |      graph attributes which attempts to completely copy
     |      all of the data and references.
     |
     |      This is in contrast to the similar `G = nx.DiGraph(D)` which returns a
     |      shallow copy of the data.
     |
     |      See the Python copy module for more information on shallow
     |      and deep copies, https://docs.python.org/3/library/copy.html.
     |
     |      Warning: If you have subclassed DiGraph to use dict-like objects
     |      in the data structure, those changes do not transfer to the
     |      Graph created by this method.
     |
     |      Examples
     |      --------
     |      >>> G = nx.path_graph(2)  # or MultiGraph, etc
     |      >>> H = G.to_directed()
     |      >>> list(H.edges)
     |      [(0, 1), (1, 0)]
     |      >>> G2 = H.to_undirected()
     |      >>> list(G2.edges)
     |      [(0, 1)]
     |
     |  to_undirected_class(self)
     |      Returns the class to use for empty undirected copies.
     |
     |      If you subclass the base classes, use this to designate
     |      what directed class to use for `to_directed()` copies.
     |
     |  update(self, edges=None, nodes=None)
     |      Update the graph using nodes/edges/graphs as input.
     |
     |      Like dict.update, this method takes a graph as input, adding the
     |      graph's nodes and edges to this graph. It can also take two inputs:
     |      edges and nodes. Finally it can take either edges or nodes.
     |      To specify only nodes the keyword `nodes` must be used.
     |
     |      The collections of edges and nodes are treated similarly to
     |      the add_edges_from/add_nodes_from methods. When iterated, they
     |      should yield 2-tuples (u, v) or 3-tuples (u, v, datadict).
     |
     |      Parameters
     |      ----------
     |      edges : Graph object, collection of edges, or None
     |          The first parameter can be a graph or some edges. If it has
     |          attributes `nodes` and `edges`, then it is taken to be a
     |          Graph-like object and those attributes are used as collections
     |          of nodes and edges to be added to the graph.
     |          If the first parameter does not have those attributes, it is
     |          treated as a collection of edges and added to the graph.
     |          If the first argument is None, no edges are added.
     |      nodes : collection of nodes, or None
     |          The second parameter is treated as a collection of nodes
     |          to be added to the graph unless it is None.
     |          If `edges is None` and `nodes is None` an exception is raised.
     |          If the first parameter is a Graph, then `nodes` is ignored.
     |
     |      Examples
     |      --------
     |      >>> G = nx.path_graph(5)
     |      >>> G.update(nx.complete_graph(range(4, 10)))
     |      >>> from itertools import combinations
     |      >>> edges = (
     |      ...     (u, v, {"power": u * v})
     |      ...     for u, v in combinations(range(10, 20), 2)
     |      ...     if u * v < 225
     |      ... )
     |      >>> nodes = [1000]  # for singleton, use a container
     |      >>> G.update(edges, nodes)
     |
     |      Notes
     |      -----
     |      It you want to update the graph using an adjacency structure
     |      it is straightforward to obtain the edges/nodes from adjacency.
     |      The following examples provide common cases, your adjacency may
     |      be slightly different and require tweaks of these examples::
     |
     |      >>> # dict-of-set/list/tuple
     |      >>> adj = {1: {2, 3}, 2: {1, 3}, 3: {1, 2}}
     |      >>> e = [(u, v) for u, nbrs in adj.items() for v in nbrs]
     |      >>> G.update(edges=e, nodes=adj)
     |
     |      >>> DG = nx.DiGraph()
     |      >>> # dict-of-dict-of-attribute
     |      >>> adj = {1: {2: 1.3, 3: 0.7}, 2: {1: 1.4}, 3: {1: 0.7}}
     |      >>> e = [
     |      ...     (u, v, {"weight": d})
     |      ...     for u, nbrs in adj.items()
     |      ...     for v, d in nbrs.items()
     |      ... ]
     |      >>> DG.update(edges=e, nodes=adj)
     |
     |      >>> # dict-of-dict-of-dict
     |      >>> adj = {1: {2: {"weight": 1.3}, 3: {"color": 0.7, "weight": 1.2}}}
     |      >>> e = [
     |      ...     (u, v, {"weight": d})
     |      ...     for u, nbrs in adj.items()
     |      ...     for v, d in nbrs.items()
     |      ... ]
     |      >>> DG.update(edges=e, nodes=adj)
     |
     |      >>> # predecessor adjacency (dict-of-set)
     |      >>> pred = {1: {2, 3}, 2: {3}, 3: {3}}
     |      >>> e = [(v, u) for u, nbrs in pred.items() for v in nbrs]
     |
     |      >>> # MultiGraph dict-of-dict-of-dict-of-attribute
     |      >>> MDG = nx.MultiDiGraph()
     |      >>> adj = {
     |      ...     1: {2: {0: {"weight": 1.3}, 1: {"weight": 1.2}}},
     |      ...     3: {2: {0: {"weight": 0.7}}},
     |      ... }
     |      >>> e = [
     |      ...     (u, v, ekey, d)
     |      ...     for u, nbrs in adj.items()
     |      ...     for v, keydict in nbrs.items()
     |      ...     for ekey, d in keydict.items()
     |      ... ]
     |      >>> MDG.update(edges=e)
     |
     |      See Also
     |      --------
     |      add_edges_from: add multiple edges to a graph
     |      add_nodes_from: add multiple nodes to a graph
     |
     |  ----------------------------------------------------------------------
     |  Data descriptors inherited from networkx.classes.graph.Graph:
     |
     |  __dict__
     |      dictionary for instance variables
     |
     |  __weakref__
     |      list of weak references to the object
     |
     |  name
     |      String identifier of the graph.
     |
     |      This graph attribute appears in the attribute dict G.graph
     |      keyed by the string `"name"`. as well as an attribute (technically
     |      a property) `G.name`. This is entirely user controlled.
     |
     |  ----------------------------------------------------------------------
     |  Data and other attributes inherited from networkx.classes.graph.Graph:
     |
     |  __networkx_backend__ = 'networkx'
     |
     |  adjlist_inner_dict_factory = <class 'dict'>
     |      dict() -> new empty dictionary
     |      dict(mapping) -> new dictionary initialized from a mapping object's
     |          (key, value) pairs
     |      dict(iterable) -> new dictionary initialized as if via:
     |          d = {}
     |          for k, v in iterable:
     |              d[k] = v
     |      dict(**kwargs) -> new dictionary initialized with the name=value pairs
     |          in the keyword argument list.  For example:  dict(one=1, two=2)
     |
     |
     |  adjlist_outer_dict_factory = <class 'dict'>
     |      dict() -> new empty dictionary
     |      dict(mapping) -> new dictionary initialized from a mapping object's
     |          (key, value) pairs
     |      dict(iterable) -> new dictionary initialized as if via:
     |          d = {}
     |          for k, v in iterable:
     |              d[k] = v
     |      dict(**kwargs) -> new dictionary initialized with the name=value pairs
     |          in the keyword argument list.  For example:  dict(one=1, two=2)
     |
     |
     |  edge_attr_dict_factory = <class 'dict'>
     |      dict() -> new empty dictionary
     |      dict(mapping) -> new dictionary initialized from a mapping object's
     |          (key, value) pairs
     |      dict(iterable) -> new dictionary initialized as if via:
     |          d = {}
     |          for k, v in iterable:
     |              d[k] = v
     |      dict(**kwargs) -> new dictionary initialized with the name=value pairs
     |          in the keyword argument list.  For example:  dict(one=1, two=2)
     |
     |
     |  graph_attr_dict_factory = <class 'dict'>
     |      dict() -> new empty dictionary
     |      dict(mapping) -> new dictionary initialized from a mapping object's
     |          (key, value) pairs
     |      dict(iterable) -> new dictionary initialized as if via:
     |          d = {}
     |          for k, v in iterable:
     |              d[k] = v
     |      dict(**kwargs) -> new dictionary initialized with the name=value pairs
     |          in the keyword argument list.  For example:  dict(one=1, two=2)
     |
     |
     |  node_attr_dict_factory = <class 'dict'>
     |      dict() -> new empty dictionary
     |      dict(mapping) -> new dictionary initialized from a mapping object's
     |          (key, value) pairs
     |      dict(iterable) -> new dictionary initialized as if via:
     |          d = {}
     |          for k, v in iterable:
     |              d[k] = v
     |      dict(**kwargs) -> new dictionary initialized with the name=value pairs
     |          in the keyword argument list.  For example:  dict(one=1, two=2)
     |
     |
     |  node_dict_factory = <class 'dict'>
     |      dict() -> new empty dictionary
     |      dict(mapping) -> new dictionary initialized from a mapping object's
     |          (key, value) pairs
     |      dict(iterable) -> new dictionary initialized as if via:
     |          d = {}
     |          for k, v in iterable:
     |              d[k] = v
     |      dict(**kwargs) -> new dictionary initialized with the name=value pairs
     |          in the keyword argument list.  For example:  dict(one=1, two=2)

DATA
    Iterable = typing.Iterable
        A generic version of collections.abc.Iterable.

    Mapping = typing.Mapping
        A generic version of collections.abc.Mapping.

FILE
    c:\users\mainuser\documents\repos\mascaf\.venv\lib\site-packages\swctools\model.py


~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Function: swctools.model._graph_attributes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Python Library Documentation: function _graph_attributes in module swctools.model

_graph_attributes(G: 'nx.Graph | nx.DiGraph') -> 'dict[str, Any]'
    Compute generic attributes for a graph.

    Returns a dictionary including:
    - graph_type: "DiGraph" or "Graph"
    - directed: bool
    - nodes: int
    - edges: int
    - components: int (computed on the undirected view)
    - cycles: int (cyclomatic number on undirected view: E - N + C)
    - branch_points_count: int
      * DiGraph: nodes with out-degree > 1
      * Graph: nodes with degree > 2
    - roots_count: int | None (only for DiGraph; nodes with in-degree == 0)
    - leaves_count: int (DiGraph: out-degree == 0; Graph: degree == 1)
    - self_loops: int
    - density: float (on undirected view)

================================================================================
Module: swctools.viz
--------------------------------------------------------------------------------
Python Library Documentation: module swctools.viz in swctools

NAME
    swctools.viz - Visualization helpers for swctools.

DESCRIPTION
    - plot_centroid: skeleton plotting from SWCModel using Scatter3d
    - plot_frusta: volumetric frusta plotting from FrustaSet using Mesh3d

FUNCTIONS
    plot_centroid(swc_model: 'SWCModel', *, marker_size: 'float' = 2.0, line_width: 'float' = 2.0, show_nodes: 'bool' = True, title: 'str | None' = None) -> 'go.Figure'
        Plot centroid skeleton from an `SWCModel`.

        Edges are drawn as line segments in 3D using Scatter3d.

    plot_frusta(frusta: 'FrustaSet', *, color: 'str' = 'lightblue', opacity: 'float' = 0.8, flatshading: 'bool' = True, radius_scale: 'float' = 1.0, tag_colors: 'dict[int, str] | None' = None, title: 'str | None' = None) -> 'go.Figure'
        Plot a FrustaSet as a Mesh3d figure.

        Parameters
        ----------
        frusta: FrustaSet
            Batched frusta mesh to render.
        color: str
            Mesh color.
        opacity: float
            Mesh opacity.
        flatshading: bool
            Whether to enable flat shading.
        radius_scale: float
            Uniform scale applied to all segment radii before meshing (1.0 = no change).
        tag_colors: dict[int, str] | None
            Optional mapping {tag: color}. If provided, each frustum segment is colored
            uniformly according to its tag (fallback to `color` if a tag is missing).

    plot_frusta_slider(frusta: 'FrustaSet', *, color: 'str' = 'lightblue', opacity: 'float' = 0.8, flatshading: 'bool' = True, tag_colors: 'dict[int, str] | None' = None, min_scale: 'float' = 0.0, max_scale: 'float' = 1.0, steps: 'int' = 21, title: 'str | None' = None) -> 'go.Figure'
        Interactive slider (0..1 default) controlling uniform `radius_scale`.

        Precomputes frames at evenly spaced scales between `min_scale` and `max_scale`.

    plot_frusta_timeseries(frusta: 'FrustaSet', values: 'Sequence[Sequence[float]]', *, colorscale: 'str | list' = 'Viridis', cmin: 'Optional[float]' = None, cmax: 'Optional[float]' = None, opacity: 'float' = 0.8, flatshading: 'bool' = True, radius_scale: 'float' = 1.0, fps: 'int' = 10, title: 'str | None' = None) -> 'go.Figure'
        Animate per-segment values over time by coloring frusta faces.

        Parameters
        ----------
        frusta: FrustaSet
            Batched frusta mesh. Optionally scaled via `radius_scale` before rendering.
        values: Sequence[Sequence[float]]
            Time series V_i(t) shaped [T][N], where N = `frusta.segment_count`.
            Each time step provides one scalar per segment in the current order.
        colorscale: str | list
            Plotly colorscale for mapping intensities.
        cmin, cmax: float | None
            Fixed color range. If omitted, inferred from the data across all frames.
        opacity: float
            Mesh opacity.
        flatshading: bool
            Enable flat shading on the mesh.
        radius_scale: float
            Uniform radius scaling applied before meshing (1.0 = no change).
        fps: int
            Playback frame rate for the Play button.

        Returns
        -------
        plotly.graph_objects.Figure
            A figure with a Mesh3d trace and animation frames, plus a slider and play/pause controls.

    plot_frusta_with_centroid(swc_model: 'SWCModel', frusta: 'FrustaSet', *, color: 'str' = 'lightblue', opacity: 'float' = 0.8, flatshading: 'bool' = True, radius_scale: 'float' = 1.0, tag_colors: 'dict[int, str] | None' = None, centroid_color: 'str' = '#1f77b4', centroid_line_width: 'float' = 2.0, show_nodes: 'bool' = False, node_size: 'float' = 2.0, title: 'str | None' = None) -> 'go.Figure'
        Overlay frusta mesh with centroid skeleton from an `SWCModel`.

        Parameters mirror `plot_centroid` and `plot_frusta` with an extra `radius_scale`.

    plot_model(*, swc_model: 'SWCModel | None' = None, frusta: 'FrustaSet | None' = None, show_frusta: 'bool' = True, show_centroid: 'bool' = True, title: 'str | None' = None, sides: 'int' = 16, end_caps: 'bool' = False, color: 'str' = 'lightblue', opacity: 'float' = 0.8, flatshading: 'bool' = True, tag_colors: 'dict[int, str] | None' = None, radius_scale: 'float' = 1.0, slider: 'bool' = False, min_scale: 'float' = 0.0, max_scale: 'float' = 1.0, steps: 'int' = 21, centroid_color: 'str' = '#1f77b4', centroid_line_width: 'float' = 2.0, show_nodes: 'bool' = False, node_size: 'float' = 2.0, point_set: 'PointSet | None' = None, point_size: 'float' = 1.0, point_color: 'str' = '#d62728') -> 'go.Figure'
        Master visualization combining centroid, frusta, slider, and overlay points.

        - If `frusta` is not provided and `gm` is, a `FrustaSet` is built from `gm`.
        - If `slider=True` and `show_frusta=True`, a Plotly slider controls `radius_scale`.
        - `points` overlays arbitrary xyz positions as small markers.

DATA
    Optional = typing.Optional
        Optional[X] is equivalent to Union[X, None].

    Sequence = typing.Sequence
        A generic version of collections.abc.Sequence.

FILE
    c:\users\mainuser\documents\repos\mascaf\.venv\lib\site-packages\swctools\viz.py


~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Function: swctools.viz.contextmanager
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Python Library Documentation: function contextmanager in module contextlib

contextmanager(func)
    @contextmanager decorator.

    Typical usage:

        @contextmanager
        def some_generator(<arguments>):
            <setup>
            try:
                yield <value>
            finally:
                <cleanup>

    This makes this:

        with some_generator(<arguments>) as <variable>:
            <body>

    equivalent to this:

        <setup>
        try:
            <variable> = <value>
            <body>
        finally:
            <cleanup>
