#!/usr/bin/env python3
"""Debug boolean cut operations."""

from core.step_loader import STEPLoader
from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
from OCP.BRepPrimAPI import BRepPrimAPI_MakeSphere
from OCP.gp import gp_Pnt
from OCP.TopoDS import TopoDS_Compound, TopoDS_Builder

loader = STEPLoader()
if not loader.load('crosman_1377_grip_base_left.step'):
    print('Error: Failed to load STEP file')
    exit(1)

shape = loader.shape
if shape is None:
    print('Error: Shape is None after loading')
    exit(1)

print(f'Loaded shape, is null: {shape.IsNull()}')

# Try a single sphere cut at origin
center = gp_Pnt(0, 0, 0)
sphere = BRepPrimAPI_MakeSphere(center, 1.0).Shape()

builder = TopoDS_Builder()
compound = TopoDS_Compound()
builder.MakeCompound(compound)
builder.Add(compound, sphere)

cut = BRepAlgoAPI_Cut(shape, compound)
cut.Build()
print(f'Cut done: {cut.IsDone()}')

result = cut.Shape()
print(f'Result shape is null: {result.IsNull()}')

# Save it
loader.save_step('/tmp/test_single_cut.step', result)
print('Saved to /tmp/test_single_cut.step')
