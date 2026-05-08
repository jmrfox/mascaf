# %%
from mascaf import MeshManager, PolylinesSkeleton

mm = MeshManager()
mm.load_mesh("../data/demo/cylinder.obj")
ps = PolylinesSkeleton().from_txt("../data/demo/cylinder.polylines.txt")
mm.visualize_mesh_3d(polylines=ps)

# %%
mm = MeshManager()
mm.load_mesh("../data/demo/torus.obj")
ps = PolylinesSkeleton().from_txt("../data/demo/torus.polylines.txt")
mm.visualize_mesh_3d(polylines=ps)

# %%
mm = MeshManager()
mm.load_mesh("../data/mesh/processed/TS2.obj")
ps = PolylinesSkeleton().from_txt("../data/polylines/TS2.polylines.txt")
mm.visualize_mesh_3d(polylines=ps)

# %%
mm = MeshManager()
mm.load_mesh("../data/mesh/processed/TS1.obj")
ps = PolylinesSkeleton().from_txt("../data/polylines/TS1_quality0.9.polylines.txt")
mm.visualize_mesh_3d(polylines=ps)

# %%
mm = MeshManager()
mm.load_mesh("../data/mesh/processed/TS3.obj")
ps = PolylinesSkeleton().from_txt("../data/polylines/TS3.polylines.txt")
mm.visualize_mesh_3d(polylines=ps)
