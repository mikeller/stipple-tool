#!/usr/bin/env python3
"""Mesh-based stippling CLI using vertex displacement."""

import argparse
from pathlib import Path

from core.color_analyzer import ColorAnalyzer
from core.mesh_color_analyzer import MeshColorAnalyzer
from core.mesh_loader import MeshLoader
from core.mesh_stipple_engine import MeshStippleEngine
from core.step_loader import STEPLoader
from core.step_mesh_converter import STEPMeshConverter


def _load_step_mesh(input_path: Path, color: str, deflection: float):
    loader = STEPLoader()
    if not loader.load(str(input_path)):
        return None

    step_model = loader.get_model()
    if step_model is None:
        print("Failed to read STEP model")
        return None

    colors = ColorAnalyzer().extract_colors_from_model(step_model)
    target_face_indices = colors.get(color, [])
    if not target_face_indices:
        print(f"No faces found with color {color}")
        return None

    result = STEPMeshConverter.shape_to_mesh_with_face_map(loader.shape, deflection)
    if result is None:
        return None

    mesh, face_map = result
    target_face_set = set(target_face_indices)
    mesh_face_indices = [
        i for i, step_face_idx in enumerate(face_map)
        if step_face_idx in target_face_set
    ]

    if not mesh_face_indices:
        print("No mesh faces mapped to the target color")
        return None

    return mesh, mesh_face_indices


def _load_mesh(input_path: Path, color: str):
    loader = MeshLoader()
    if not loader.load(str(input_path)):
        return None

    model = loader.get_model()
    if model is None:
        print("Failed to read mesh model")
        return None

    color_groups = MeshColorAnalyzer().extract_colors_from_model(model)
    target_faces = color_groups.get(color, [])
    if not target_faces:
        print(f"No faces found with color {color}")
        return None

    return model["mesh"], target_faces


def main():
    parser = argparse.ArgumentParser(description="Mesh-based stippling")
    parser.add_argument("input", help="Input STEP/OBJ/STL file")
    parser.add_argument("-o", "--output", required=True, help="Output mesh file (STL/OBJ)")
    parser.add_argument("-c", "--color", required=True, help="Target color (hex, e.g., #360200)")
    parser.add_argument("--size", type=float, default=2.0, help="Stipple size in mm")
    parser.add_argument("--depth", type=float, default=0.5, help="Max depth in mm")
    parser.add_argument("--density", type=float, default=0.35, help="Density factor (0.01-1.0)")
    parser.add_argument("--coverage", type=float, default=3.0, help="Coverage factor")
    parser.add_argument("--max-points", type=int, default=8000, help="Max stipple points")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--deflection", type=float, default=0.1, help="STEP mesh deflection")
    parser.add_argument(
        "--all-faces",
        action="store_true",
        help="Ignore color filter and stipple all faces",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    print("MESH STIPPLING")
    print("=" * 50)
    print(f"Input:       {input_path}")
    print(f"Output:      {output_path}")
    print(f"Color:       {args.color}")
    print(f"Size:        {args.size} mm")
    print(f"Depth:       {args.depth} mm")
    print(f"Density:     {args.density}")
    print(f"Coverage:    {args.coverage}")
    print(f"Max points:  {args.max_points}")
    print(f"Seed:        {args.seed}")
    if input_path.suffix.lower() in [".step", ".stp"]:
        print(f"Deflection:  {args.deflection}")
    print("=" * 50)

    if input_path.suffix.lower() in [".step", ".stp"]:
        loaded = _load_step_mesh(input_path, args.color, args.deflection)
    else:
        loaded = _load_mesh(input_path, args.color)

    if loaded is None:
        print("\n✗ Processing failed")
        return 1

    mesh, target_faces = loaded

    if args.all_faces:
        target_faces = list(range(len(mesh.faces)))

    print(f"Target faces: {len(target_faces)} / {len(mesh.faces)}")

    engine = MeshStippleEngine()
    engine.random_seed = args.seed
    engine.set_parameters(
        size=args.size,
        depth=args.depth,
        density=args.density,
        coverage_factor=args.coverage,
        max_total_points=args.max_points,
    )

    def progress(done: int, total: int):
        print(f"  Stipple {done}/{total}")

    result = engine.apply_stippling_to_mesh(
        mesh=mesh,
        face_indices=target_faces,
        progress_callback=progress,
        status_callback=print,
    )

    saver = MeshLoader()
    if saver.save_mesh(str(output_path), result):
        print(f"\n✓ Successfully saved to: {output_path}")
        return 0

    print("\n✗ Failed to write output")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
