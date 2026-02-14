#!/usr/bin/env python3
"""Bitmap-based stippling CLI using vertex displacement on mesh."""

import argparse
from pathlib import Path
import numpy as np

from core.color_analyzer import ColorAnalyzer
from core.mesh_color_analyzer import MeshColorAnalyzer
from core.mesh_loader import MeshLoader
from core.step_loader import STEPLoader
from core.step_mesh_converter import STEPMeshConverter


def _generate_blue_noise_bitmap(size: int, seed: int) -> np.ndarray:
    """
    Generate a blue-noise-like bitmap using Gaussian-filtered white noise.

    Args:
        size: Bitmap dimensions (size × size)
        seed: Random seed

    Returns:
        Normalized numpy array [0, 1] of shape (size, size)
    """
    np.random.seed(seed)
    bitmap = np.random.rand(size, size)

    try:
        from scipy.ndimage import gaussian_filter

        bitmap = gaussian_filter(bitmap, sigma=1.5)
    except ImportError:
        # Fallback: simple box filter
        kernel_size = 3
        for _ in range(2):
            new_bitmap = np.zeros_like(bitmap)
            for i in range(size):
                for j in range(size):
                    region = bitmap[
                        max(0, i - kernel_size) : min(size, i + kernel_size + 1),
                        max(0, j - kernel_size) : min(size, j + kernel_size + 1),
                    ]
                    new_bitmap[i, j] = np.mean(region)
            bitmap = new_bitmap

    # Normalize to [0, 1]
    bitmap = (bitmap - bitmap.min()) / (bitmap.max() - bitmap.min() + 1e-9)
    return bitmap


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
        i for i, step_face_idx in enumerate(face_map) if step_face_idx in target_face_set
    ]

    if not mesh_face_indices:
        print("No mesh faces mapped to the target color")
        return None

    return mesh, mesh_face_indices


def _apply_bitmap_stippling(
    mesh,
    face_indices,
    bitmap: np.ndarray,
    max_depth: float,
    threshold: float,
) -> None:
    """
    Apply bitmap-based displacement to mesh vertices.

    Args:
        mesh: trimesh.Trimesh object
        face_indices: Target face indices
        bitmap: Stipple bitmap [0, 1]
        max_depth: Max displacement in mm
        threshold: Cutoff for bitmap values
    """
    import trimesh

    # Build UV-like coordinates from face barycentrics
    vertices = mesh.vertices.copy()
    vertex_normals = mesh.vertex_normals
    faces = mesh.faces
    vertex_displacements = np.zeros(len(vertices), dtype=np.float64)

    bitmap_h, bitmap_w = bitmap.shape

    # For each target face, displace its vertices
    for face_idx in face_indices:
        if face_idx >= len(faces):
            continue

        face = faces[face_idx]
        face_vertices = vertices[face]

        # Compute centroid and bounds for UV mapping
        centroid = face_vertices.mean(axis=0)
        v_min = face_vertices.min(axis=0)
        v_max = face_vertices.max(axis=0)
        v_range = v_max - v_min
        v_range[v_range < 1e-6] = 1e-6  # Avoid division by zero

        for vi, v_idx in enumerate(face):
            vertex = vertices[v_idx]

            # Compute normalized UV (0-1) based on vertex position within face bounds
            u_norm = (vertex[0] - v_min[0]) / v_range[0]
            v_norm = (vertex[1] - v_min[1]) / v_range[1]

            # Clamp to [0, 1]
            u_norm = max(0.0, min(1.0, u_norm))
            v_norm = max(0.0, min(1.0, v_norm))

            # Sample bitmap
            bx = int(u_norm * (bitmap_w - 1))
            by = int(v_norm * (bitmap_h - 1))
            value = bitmap[by, bx]

            if value > threshold:
                displacement = max_depth * value
                if displacement > vertex_displacements[v_idx]:
                    vertex_displacements[v_idx] = displacement

    # Apply displacement
    mesh.vertices = vertices - (vertex_normals * vertex_displacements[:, None])

    moved = int(np.count_nonzero(vertex_displacements > 1e-6))
    print(f"Displacement max: {vertex_displacements.max():.4f} mm on {moved} vertices")


def main():
    parser = argparse.ArgumentParser(description="Bitmap-based stippling")
    parser.add_argument("input", help="Input STEP/OBJ/STL file")
    parser.add_argument("-o", "--output", required=True, help="Output file (STL/OBJ/STEP)")
    parser.add_argument("-c", "--color", required=True, help="Target color (hex, e.g., #360200)")
    parser.add_argument("--depth", type=float, default=0.5, help="Max displacement depth in mm")
    parser.add_argument(
        "--bitmap-size", type=int, default=512, help="Stipple bitmap resolution"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.35,
        help="Bitmap threshold (0.0-1.0; only values > threshold are stippled)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for bitmap")
    parser.add_argument("--deflection", type=float, default=0.1, help="STEP mesh deflection")
    parser.add_argument("--step-output", action="store_true", help="Export as STEP (shell, not solid)")

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    print("BITMAP STIPPLING")
    print("=" * 50)
    print(f"Input:        {input_path}")
    print(f"Output:       {output_path}")
    print(f"Color:        {args.color}")
    print(f"Max depth:    {args.depth} mm")
    print(f"Bitmap size:  {args.bitmap_size}×{args.bitmap_size}")
    print(f"Threshold:    {args.threshold}")
    print(f"Seed:         {args.seed}")
    if input_path.suffix.lower() in [".step", ".stp"]:
        print(f"Deflection:   {args.deflection}")
    print("=" * 50)

    # Load and convert to mesh
    if input_path.suffix.lower() in [".step", ".stp"]:
        loaded = _load_step_mesh(input_path, args.color, args.deflection)
    else:
        loader = MeshLoader()
        if not loader.load(str(input_path)):
            print("\n✗ Processing failed")
            return 1

        model = loader.get_model()
        if model is None:
            print("\n✗ Processing failed")
            return 1

        color_groups = MeshColorAnalyzer().extract_colors_from_model(model)
        target_faces = color_groups.get(args.color, [])
        if not target_faces:
            print(f"No faces found with color {args.color}")
            print("\n✗ Processing failed")
            return 1

        loaded = model["mesh"], target_faces

    if loaded is None:
        print("\n✗ Processing failed")
        return 1

    mesh, target_faces = loaded

    print(f"Target faces: {len(target_faces)} / {len(mesh.faces)}")

    # Generate bitmap
    print(f"Generating {args.bitmap_size}×{args.bitmap_size} blue-noise bitmap...")
    bitmap = _generate_blue_noise_bitmap(args.bitmap_size, args.seed)

    # Apply bitmap stippling
    print("Applying bitmap displacement...")
    _apply_bitmap_stippling(mesh, target_faces, bitmap, args.depth, args.threshold)

    # Save
    if args.step_output or output_path.suffix.lower() in [".step", ".stp"]:
        print("Converting mesh to STEP shell...")
        shape = STEPMeshConverter.mesh_to_shape(mesh)
        if shape is None:
            print("\n✗ Failed to convert mesh to STEP")
            return 1

        loader = STEPLoader()
        if loader.save_step(str(output_path), shape):
            print(f"\n✓ Successfully saved to: {output_path}")
            print("Note: Output is a STEP shell, not a solid")
            return 0
        else:
            print("\n✗ Failed to write STEP file")
            return 1
    else:
        saver = MeshLoader()
        if saver.save_mesh(str(output_path), mesh):
            print(f"\n✓ Successfully saved to: {output_path}")
            return 0

    print("\n✗ Failed to write output")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
