"""Test face-grouped compound batches on LARGE faces only."""
import random, time, sys
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

# Collect ALL faces with triangulation AND area
faces_data = []
exp = TopExp_Explorer(shape, TopAbs_FACE)
face_idx = 0
while exp.More():
    face = TopoDS.Face_s(exp.Current())
    loc = TopLoc_Location()
    tri = BRep_Tool.Triangulation_s(face, loc)
    if tri is not None:
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
        if triangles:
            cum = []; c = 0.0
            for _, _, _, a in triangles:
                c += a; cum.append(c)
            faces_data.append({'idx': face_idx, 'triangles': triangles, 'total_area': total_area, 'cum': cum})
    exp.Next(); face_idx += 1

# Sort by area and show
faces_data.sort(key=lambda x: x['total_area'], reverse=True)
print(f"\nAll {len(faces_data)} faces by area:")
for fd in faces_data[:20]:
    print(f"  Face {fd['idx']}: {fd['total_area']:.1f} mm²")

# Filter to faces > 50mm² (skip tiny fillets/edges)
MIN_AREA = 50.0
big_faces = [fd for fd in faces_data if fd['total_area'] >= MIN_AREA]
print(f"\n{len(big_faces)} faces with area >= {MIN_AREA} mm²")

def sample_from_face(fd):
    r = random.random() * fd['total_area']
    idx = 0
    for j, ca in enumerate(fd['cum']):
        if ca >= r: idx = j; break
    tp1, tp2, tp3, _ = fd['triangles'][idx]
    s = random.random(); t_ = random.random()
    if s + t_ > 1: s = 1 - s; t_ = 1 - t_
    w = 1 - s - t_
    pnt = gp_Pnt(w*tp1.X()+s*tp2.X()+t_*tp3.X(), w*tp1.Y()+s*tp2.Y()+t_*tp3.Y(), w*tp1.Z()+s*tp2.Z()+t_*tp3.Z())
    normal = gp_Vec(tp1, tp2).Crossed(gp_Vec(tp1, tp3))
    if normal.Magnitude() > 1e-6: normal.Normalize()
    radius = 0.6 + random.random() * 0.8
    offset = max(0, radius - 0.5)
    return gp_Pnt(pnt.X()-normal.X()*offset, pnt.Y()-normal.Y()*offset, pnt.Z()-normal.Z()*offset), radius

# Density-based sphere count per face
DENSITY = float(sys.argv[1]) if len(sys.argv) > 1 else 0.6  # spheres/mm²
BATCH = int(sys.argv[2]) if len(sys.argv) > 2 else 50

total = 0
face_spheres = {}
for i, fd in enumerate(big_faces):
    count = max(1, int(fd['total_area'] * DENSITY))
    face_spheres[i] = [sample_from_face(fd) for _ in range(count)]
    total += count

print(f"\nDensity: {DENSITY}/mm², total spheres: {total}, batch: {BATCH}")

cur = solid; cur_vol = orig_vol
t0 = time.time()
total_applied = 0
total_skipped = 0

for i, fd in enumerate(big_faces):
    spheres = face_spheres[i]
    if not spheres: continue
    face_start = time.time()
    face_applied = 0
    face_skipped = 0
    
    for batch_start in range(0, len(spheres), BATCH):
        batch = spheres[batch_start:batch_start + BATCH]
        builder = TopoDS_Builder(); compound = TopoDS_Compound()
        builder.MakeCompound(compound)
        for c, r in batch:
            builder.Add(compound, BRepPrimAPI_MakeSphere(c, r).Shape())
        
        cut = BRepAlgoAPI_Cut()
        args = TopTools_ListOfShape(); args.Append(cur); cut.SetArguments(args)
        tools = TopTools_ListOfShape(); tools.Append(compound); cut.SetTools(tools)
        cut.SetFuzzyValue(0.01); cut.SetRunParallel(True); cut.Build()
        
        if cut.IsDone() and not cut.Shape().IsNull():
            result = cut.Shape()
            vp = GProp_GProps(); BRepGProp.VolumeProperties_s(result, vp)
            rv = abs(vp.Mass())
            drop = cur_vol - rv
            max_drop = 12 * len(batch) * 1.5
            if rv < 1e-6 or drop > max_drop:
                print(f"  SKIP face {fd['idx']} batch {batch_start} ({len(batch)} sph): "
                      f"vol={rv:.1f} drop={drop:.1f} max_ok={max_drop:.1f}")
                face_skipped += len(batch)
                continue  # skip this batch, don't update solid
            cur = result; cur_vol = rv
            face_applied += len(batch)
        else:
            face_skipped += len(batch)
    
    total_applied += face_applied
    total_skipped += face_skipped
    elapsed = time.time() - t0
    face_time = time.time() - face_start
    print(f"  Face {fd['idx']} ({fd['total_area']:.0f}mm²): "
          f"{face_applied}/{len(spheres)} applied, {face_skipped} skip, "
          f"vol={cur_vol:.1f}, ft={face_time:.1f}s, tot={elapsed:.1f}s")
    sys.stdout.flush()

elapsed = time.time() - t0
removed = orig_vol - cur_vol
pct = (removed / orig_vol) * 100
print(f"\nTotal: {total_applied}/{total} applied, {total_skipped} skipped, "
      f"removed={removed:.1f}mm³ ({pct:.2f}%), time={elapsed:.1f}s")
