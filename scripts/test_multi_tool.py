"""Test multi-tool vs compound vs individual cuts."""
import time
from OCP.STEPControl import STEPControl_Reader
from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
from OCP.BRepGProp import BRepGProp
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.BRepPrimAPI import BRepPrimAPI_MakeSphere
from OCP.GProp import GProp_GProps
from OCP.TopAbs import TopAbs_SOLID
from OCP.TopExp import TopExp_Explorer
from OCP.TopTools import TopTools_ListOfShape
from OCP.TopoDS import TopoDS, TopoDS_Builder, TopoDS_Compound
from OCP.gp import gp_Pnt

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

# Generate 20 sphere positions on one face
spheres = []
for i in range(20):
    x = -50 + (i % 5) * 3
    y = 5 + (i // 5) * 3
    spheres.append((gp_Pnt(x, y, 15), 1.2))

# Approach 1: Compound tool (one entry in tools list)
print("\n--- Approach 1: Compound tool ---")
builder = TopoDS_Builder()
compound = TopoDS_Compound()
builder.MakeCompound(compound)
for centre, radius in spheres:
    builder.Add(compound, BRepPrimAPI_MakeSphere(centre, radius).Shape())

t0 = time.time()
cut = BRepAlgoAPI_Cut()
args = TopTools_ListOfShape()
args.Append(solid)
cut.SetArguments(args)
tools = TopTools_ListOfShape()
tools.Append(compound)
cut.SetTools(tools)
cut.SetFuzzyValue(0.01)
cut.Build()
t1 = time.time()

if cut.IsDone() and not cut.Shape().IsNull():
    vp = GProp_GProps()
    BRepGProp.VolumeProperties_s(cut.Shape(), vp)
    v = abs(vp.Mass())
    print(f"  Volume: {v:.1f}, removed: {orig_vol-v:.1f}, time: {t1-t0:.2f}s")
else:
    print(f"  FAILED")

# Approach 2: Multi-tool (each sphere as separate entry in tools list)
print("\n--- Approach 2: Multi-tool ---")
t0 = time.time()
cut2 = BRepAlgoAPI_Cut()
args2 = TopTools_ListOfShape()
args2.Append(solid)
cut2.SetArguments(args2)
tools2 = TopTools_ListOfShape()
for centre, radius in spheres:
    tools2.Append(BRepPrimAPI_MakeSphere(centre, radius).Shape())
cut2.SetTools(tools2)
cut2.SetFuzzyValue(0.01)
cut2.Build()
t1 = time.time()

if cut2.IsDone() and not cut2.Shape().IsNull():
    vp = GProp_GProps()
    BRepGProp.VolumeProperties_s(cut2.Shape(), vp)
    v = abs(vp.Mass())
    print(f"  Volume: {v:.1f}, removed: {orig_vol-v:.1f}, time: {t1-t0:.2f}s")
else:
    print(f"  FAILED")

# Approach 3: Individual sequential cuts
print("\n--- Approach 3: Individual sequential ---")
t0 = time.time()
cur = solid
for centre, radius in spheres:
    s = BRepPrimAPI_MakeSphere(centre, radius).Shape()
    cut3 = BRepAlgoAPI_Cut()
    a3 = TopTools_ListOfShape()
    a3.Append(cur)
    cut3.SetArguments(a3)
    t3 = TopTools_ListOfShape()
    t3.Append(s)
    cut3.SetTools(t3)
    cut3.SetFuzzyValue(0.01)
    cut3.Build()
    if cut3.IsDone() and not cut3.Shape().IsNull():
        cur = cut3.Shape()
t1 = time.time()

vp = GProp_GProps()
BRepGProp.VolumeProperties_s(cur, vp)
v = abs(vp.Mass())
print(f"  Volume: {v:.1f}, removed: {orig_vol-v:.1f}, time: {t1-t0:.2f}s")

# Approach 4: 2-arg BRepAlgoAPI_Cut with compound
print("\n--- Approach 4: 2-arg Cut(solid, compound) ---")
t0 = time.time()
cut4 = BRepAlgoAPI_Cut(solid, compound)
t1 = time.time()
if cut4.IsDone() and not cut4.Shape().IsNull():
    vp = GProp_GProps()
    BRepGProp.VolumeProperties_s(cut4.Shape(), vp)
    v = abs(vp.Mass())
    print(f"  Volume: {v:.1f}, removed: {orig_vol-v:.1f}, time: {t1-t0:.2f}s")
else:
    print(f"  FAILED")
