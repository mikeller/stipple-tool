"""Test compound cuts at various batch sizes with dense overlapping spheres."""
import random
import time
from OCP.STEPControl import STEPControl_Reader
from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
from OCP.BRepGProp import BRepGProp
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.BRep import BRep_Tool
from OCP.BRepPrimAPI import BRepPrimAPI_MakeSphere
from OCP.GProp import GProp_GProps
from OCP.TopAbs import TopAbs_SOLID, TopAbs_FACE
from OCP.TopExp import TopExp_Explorer
from OCP.TopLoc import TopLoc_Location
from OCP.TopTools import TopTools_ListOfShape
from OCP.TopoDS import TopoDS, TopoDS_Builder, TopoDS_Compound
from OCP.gp import gp_Pnt, gp_Vec

random.seed(42)

# Load shape
reader = STEPControl_Reader()
reader.ReadFile("sample_part.step")
reader.TransferRoots()
shape = reader.OneShape()
BRepMesh_IncrementalMesh(shape, 0.1, False, 0.5, True).Perform()

sol_exp = TopExp_Explorer(shape, TopAbs_SOLID)
solid = TopoDS.Solid_s(sol_exp.Current())

props = GProp_GProps()
BRepGProp.VolumeProperties_s(solid, props)
orig_vol = abs(props.Mass())
print(f"Original volume: {orig_vol:.1f}")

# Generate dense sphere positions from triangulation of face 2 (large face)
face_idx = 0
target_face = None
exp = TopExp_Explorer(shape, TopAbs_FACE)
while exp.More():
    if face_idx == 2:
        target_face = TopoDS.Face_s(exp.Current())
        break
    exp.Next()
    face_idx += 1

loc = TopLoc_Location()
tri = BRep_Tool.Triangulation_s(target_face, loc)
trsf = loc.Transformation()

triangles = []
total_area = 0.0
for t_idx in range(1, tri.NbTriangles() + 1):
    tri_obj = tri.Triangle(t_idx)
    i1, i2, i3 = tri_obj.Get()
    p1 = tri.Node(i1); p2 = tri.Node(i2); p3 = tri.Node(i3)
    p1.Transform(trsf); p2.Transform(trsf); p3.Transform(trsf)
    v1 = gp_Vec(p1, p2); v2 = gp_Vec(p1, p3)
    area = 0.5 * v1.Crossed(v2).Magnitude()
    if area > 1e-10:
        triangles.append((p1, p2, p3, area))
        total_area += area

cum_areas = []
c = 0.0
for _, _, _, a in triangles:
    c += a
    cum_areas.append(c)

def sample_point():
    r = random.random() * total_area
    idx = 0
    for j, ca in enumerate(cum_areas):
        if ca >= r: idx = j; break
    tp1, tp2, tp3, _ = triangles[idx]
    s = random.random(); t = random.random()
    if s + t > 1: s = 1 - s; t = 1 - t
    w = 1 - s - t
    pnt = gp_Pnt(
        w*tp1.X() + s*tp2.X() + t*tp3.X(),
        w*tp1.Y() + s*tp2.Y() + t*tp3.Y(),
        w*tp1.Z() + s*tp2.Z() + t*tp3.Z(),
    )
    normal = gp_Vec(tp1, tp2).Crossed(gp_Vec(tp1, tp3))
    if normal.Magnitude() > 1e-6:
        normal.Normalize()
    radius = 0.6 + random.random() * 0.8
    offset = max(0, radius - 0.5)
    centre = gp_Pnt(
        pnt.X() - normal.X() * offset,
        pnt.Y() - normal.Y() * offset,
        pnt.Z() - normal.Z() * offset,
    )
    return centre, radius

# Generate lots of spheres
all_spheres = [sample_point() for _ in range(200)]

# Test compound cuts at different batch sizes
for batch_size in [10, 25, 50, 100, 200]:
    print(f"\n--- Batch size {batch_size} ---")
    cur = solid
    cur_vol = orig_vol
    ok = True
    t0 = time.time()
    
    for batch_start in range(0, len(all_spheres), batch_size):
        batch = all_spheres[batch_start:batch_start + batch_size]
        
        builder = TopoDS_Builder()
        compound = TopoDS_Compound()
        builder.MakeCompound(compound)
        for c, r in batch:
            builder.Add(compound, BRepPrimAPI_MakeSphere(c, r).Shape())
        
        cut = BRepAlgoAPI_Cut()
        args = TopTools_ListOfShape()
        args.Append(cur)
        cut.SetArguments(args)
        tools = TopTools_ListOfShape()
        tools.Append(compound)
        cut.SetTools(tools)
        cut.SetFuzzyValue(0.01)
        cut.Build()
        
        if cut.IsDone() and not cut.Shape().IsNull():
            result = cut.Shape()
            vp = GProp_GProps()
            BRepGProp.VolumeProperties_s(result, vp)
            rv = abs(vp.Mass())
            drop = cur_vol - rv
            if rv < 1e-6 or drop > 12 * batch_size * 1.5:
                print(f"  Batch {batch_start}: VOLUME ANOMALY vol={rv:.1f} drop={drop:.1f}")
                ok = False
                break
            cur = result
            cur_vol = rv
        else:
            print(f"  Batch {batch_start}: Cut FAILED")
            ok = False
            break
    
    t1 = time.time()
    if ok:
        removed = orig_vol - cur_vol
        print(f"  OK: vol={cur_vol:.1f}, removed={removed:.1f}, time={t1-t0:.2f}s")
    else:
        print(f"  FAILED at time={t1-t0:.2f}s")
