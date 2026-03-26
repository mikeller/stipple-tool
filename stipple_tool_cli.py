#!/usr/bin/env python3
"""CLI for color-aware stippling pipelines (OCC or STL-first hybrid)."""

import argparse
from pathlib import Path

from core.stencil_processor import StencilStippleProcessor


def _run_manifold(args) -> int:
    """Run STEP → mesh → manifold3d boolean difference pipeline."""
    from core.manifold_stipple_processor import ManifoldStippleProcessor

    processor = ManifoldStippleProcessor()
    processor.random_seed = getattr(args, "mesh_seed", 42)
    result = processor.process(
        step_file=args.input,
        output_path=args.output,
        target_color=args.color,
        sphere_radius=args.radius,
        sphere_depth=args.depth,
        spheres_per_mm2=args.spheres_per_mm2,
        size_variation=not args.no_variation,
        size_variation_mode=args.size_variation_mode,
        size_variation_sigma=args.size_variation_sigma,
        size_variation_min=args.size_variation_min,
        size_variation_max=args.size_variation_max,
        deflection=getattr(args, "mesh_deflection", 0.05),
    )
    return 0 if result else 1


def _run_hybrid_stl(args) -> int:
    """Run STEP color-selection + mesh stippling + STL export pipeline."""
    from core.color_analyzer import ColorAnalyzer
    from core.mesh_loader import MeshLoader
    from core.mesh_stipple_engine import MeshStippleEngine
    from core.step_loader import STEPLoader
    from core.step_mesh_converter import STEPMeshConverter

    input_path = Path(args.input)
    out_path = Path(args.export_stl or args.output)
    if out_path.suffix.lower() not in {".stl", ".obj", ".gltf", ".glb", ".vrml"}:
        out_path = out_path.with_suffix(".stl")

    print("HYBRID STL-FIRST STIPPLING")
    print("=" * 50)
    print(f"Input:         {input_path}")
    print(f"Output mesh:   {out_path}")
    print(f"Color:         {args.color}")
    print(f"Mesh size:     {args.mesh_size} mm")
    print(f"Mesh depth:    {args.mesh_depth} mm")
    print(f"Mesh density:  {args.mesh_density}")
    print(f"Coverage:      {args.mesh_coverage}")
    print(f"Max points:    {args.mesh_max_points}")
    print(f"Mesh seed:     {args.mesh_seed}")
    print(f"Deflection:    {args.mesh_deflection}")
    print("=" * 50)

    if input_path.suffix.lower() not in {".step", ".stp"}:
        print("✗ Hybrid STL-first mode requires STEP input for color-aware face selection")
        return 1

    loader = STEPLoader()
    if not loader.load(str(input_path)) or loader.shape is None:
        print("✗ Failed to load STEP input")
        return 1

    step_model = loader.get_model()
    if step_model is None:
        print("✗ Failed to parse STEP model")
        return 1

    colors = ColorAnalyzer().extract_colors_from_model(step_model)
    target_face_indices = colors.get(args.color, [])
    if not target_face_indices:
        print(f"✗ No STEP faces found with color {args.color}")
        return 1

    converted = STEPMeshConverter.shape_to_mesh_with_face_map(
        loader.shape,
        deflection=args.mesh_deflection,
    )
    if converted is None:
        print("✗ STEP → mesh conversion failed")
        return 1

    mesh, face_map = converted
    target_set = set(target_face_indices)
    mesh_face_indices = [
        i for i, step_face_idx in enumerate(face_map)
        if step_face_idx in target_set
    ]
    if not mesh_face_indices:
        print("✗ No mesh faces map to the target STEP color")
        return 1

    print(
        f"Target mesh faces: {len(mesh_face_indices)} / {len(mesh.faces)} "
        f"(from {len(target_face_indices)} STEP faces)"
    )

    engine = MeshStippleEngine()
    engine.random_seed = args.mesh_seed
    engine.set_parameters(
        size=args.mesh_size,
        depth=args.mesh_depth,
        density=args.mesh_density,
        coverage_factor=args.mesh_coverage,
        max_total_points=args.mesh_max_points,
    )

    def progress(done: int, total: int):
        print(f"  Stipple {done}/{total}")

    result_mesh = engine.apply_stippling_to_mesh(
        mesh=mesh,
        face_indices=mesh_face_indices,
        progress_callback=progress,
        status_callback=print,
    )

    saver = MeshLoader()
    if saver.save_mesh(str(out_path), result_mesh):
        print(f"\n✓ Hybrid STL-first complete: {out_path}")
        return 0

    print("\n✗ Failed to write hybrid mesh output")
    return 1


def _run_occ_stencil(args) -> int:
    print("STENCIL STIPPLING")
    print("=" * 50)
    print(f"Input:       {args.input}")
    print(f"Output:      {args.output}")
    if args.export_stl:
        print(f"Export STL:  {args.export_stl}")
    if args.log_file:
        print(f"Log file:    {args.log_file}")
    print(f"Color:       {args.color}")
    print(f"Spheres/mm²: {args.spheres_per_mm2}")
    print(f"Radius:      {args.radius} mm")
    print(f"Depth:       {args.depth} mm")
    print(f"Strips:      {args.strips}")
    print(f"Overlap:     {args.overlap:.0%}")
    if not args.no_variation:
        print(f"Size mode:   {args.size_variation_mode}")
        if args.size_variation_mode == "gaussian":
            print(
                "Size sigma:  "
                f"{args.size_variation_sigma} "
                f"(min {args.size_variation_min}, max {args.size_variation_max})"
            )
    print("=" * 50)

    processor = StencilStippleProcessor()
    result = processor.process_step_with_stencil_stippling(
        step_file=args.input,
        output_path=args.output,
        target_color=args.color,
        sphere_radius=args.radius,
        sphere_depth=args.depth,
        spheres_per_mm2=args.spheres_per_mm2,
        strip_count=args.strips,
        overlap=args.overlap,
        size_variation=not args.no_variation,
        size_variation_mode=args.size_variation_mode,
        size_variation_sigma=args.size_variation_sigma,
        size_variation_min=args.size_variation_min,
        size_variation_max=args.size_variation_max,
        face_order_strategy=args.face_order,
        seed_spheres_per_face=args.seed_spheres_per_face,
        debug_log_path=args.log_file,
    )

    if result:
        print(f"\n✓ Successfully saved to: {result}")

        if args.export_stl:
            print("\nExporting to STL...")
            from core.step_loader import STEPLoader

            loader = STEPLoader()
            if loader.load(result) and loader.shape is not None:
                if processor.export_shape_to_stl(
                    loader.shape,
                    args.export_stl,
                    linear_deflection=0.01,
                    angular_deflection=0.5,
                    status_callback=print,
                ):
                    print("✓ STL export complete")
                else:
                    print("✗ STL export failed")
                    return 1
            else:
                print("✗ STL export failed: could not load STEP for meshing")
                return 1

        return 0

    print("\n✗ Processing failed")
    return 1


def main():
    parser = argparse.ArgumentParser(description="Color-aware stippling")
    parser.add_argument("input", help="Input STEP file")
    parser.add_argument("-o", "--output", required=True, help="Output STEP or mesh file")
    parser.add_argument("--export-stl", help="Also export to STL file (optional)")
    parser.add_argument(
        "--pipeline",
        choices=["occ", "hybrid-stl", "manifold"],
        default="occ",
        help="occ boolean-cuts, hybrid-stl mesh displacement, or manifold boolean-cuts via manifold3d",
    )
    parser.add_argument(
        "--log-file",
        help="Write detailed stippling diagnostics log to this file",
    )
    parser.add_argument("-c", "--color", required=True, help="Target color (hex, e.g., #360200)")

    parser.add_argument(
        "--spheres-per-mm2",
        type=float,
        default=0.34,
        help="(occ) Sphere density per mm² (higher = more stipples)",
    )
    parser.add_argument("--radius", type=float, default=1.0, help="(occ) Sphere radius in mm")
    parser.add_argument("--depth", type=float, default=0.45, help="(occ) Cut depth in mm")
    parser.add_argument("--strips", type=int, default=6, help="(occ) Number of stencil strips")
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.2,
        help="(occ) Overlap ratio between strips (0.0-0.9)",
    )
    parser.add_argument("--no-variation", action="store_true", help="(occ) Disable size variation")
    parser.add_argument(
        "--size-variation-mode",
        choices=["uniform", "gaussian"],
        default="gaussian",
        help="(occ) Size variation distribution (default: gaussian)",
    )
    parser.add_argument(
        "--size-variation-sigma",
        type=float,
        default=0.25,
        help="(occ) Gaussian sigma for size variation (default: 0.25)",
    )
    parser.add_argument(
        "--size-variation-min",
        type=float,
        default=0.60,
        help="(occ) Minimum size scale for variation (default: 0.60)",
    )
    parser.add_argument(
        "--size-variation-max",
        type=float,
        default=1.6,
        help="(occ) Maximum size scale for variation (default: 1.6)",
    )
    parser.add_argument(
        "--face-order",
        choices=["largest_first", "smallest_first", "original"],
        default="largest_first",
        help="(occ) Face processing order for batch cuts (default: largest_first)",
    )
    parser.add_argument(
        "--seed-spheres-per-face",
        type=int,
        default=20,
        help="(occ) Initial per-face seed pass before remainder scheduling (default: 20)",
    )

    parser.add_argument("--mesh-size", type=float, default=2.0, help="(hybrid-stl) Stipple size in mm")
    parser.add_argument("--mesh-depth", type=float, default=0.5, help="(hybrid-stl) Max displacement depth in mm")
    parser.add_argument("--mesh-density", type=float, default=0.35, help="(hybrid-stl) Density factor (0.01-1.0)")
    parser.add_argument("--mesh-coverage", type=float, default=3.0, help="(hybrid-stl) Coverage factor")
    parser.add_argument("--mesh-max-points", type=int, default=8000, help="(hybrid-stl) Maximum stipple points")
    parser.add_argument("--mesh-seed", type=int, default=42, help="(hybrid-stl) Random seed")
    parser.add_argument("--mesh-deflection", type=float, default=0.1, help="(hybrid-stl) STEP→mesh deflection")

    args = parser.parse_args()

    if args.pipeline == "hybrid-stl":
        return _run_hybrid_stl(args)
    if args.pipeline == "manifold":
        return _run_manifold(args)
    return _run_occ_stencil(args)


if __name__ == "__main__":
    raise SystemExit(main())
