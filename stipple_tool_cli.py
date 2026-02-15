#!/usr/bin/env python3
"""Demo for stencil-based stippling approach."""

import argparse
from core.stencil_processor import StencilStippleProcessor


def main():
    parser = argparse.ArgumentParser(description="Stencil-based stippling")
    parser.add_argument("input", help="Input STEP file")
    parser.add_argument("-o", "--output", required=True, help="Output STEP file")
    parser.add_argument("--export-stl", help="Also export to STL file (optional)")
    parser.add_argument("-c", "--color", required=True, help="Target color (hex, e.g., #360200)")
    parser.add_argument(
        "--spheres-per-mm2",
        type=float,
        default=0.12,
        help="Sphere density per mm² (higher = more stipples)",
    )
    parser.add_argument("--radius", type=float, default=1.0, help="Sphere radius in mm")
    parser.add_argument("--depth", type=float, default=0.5, help="Cut depth in mm")
    parser.add_argument("--strips", type=int, default=6, help="Number of stencil strips")
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.2,
        help="Overlap ratio between strips (0.0-0.9)",
    )
    parser.add_argument("--batch", type=int, default=10, help="Spheres per boolean cut (max 10 for reliable output)")
    parser.add_argument("--no-variation", action="store_true", help="Disable size variation")
    parser.add_argument(
        "--size-variation-mode",
        choices=["uniform", "gaussian"],
        default="uniform",
        help="Size variation distribution (default: uniform)",
    )
    parser.add_argument(
        "--size-variation-sigma",
        type=float,
        default=0.2,
        help="Gaussian sigma for size variation (default: 0.2)",
    )
    parser.add_argument(
        "--size-variation-min",
        type=float,
        default=0.7,
        help="Minimum size scale for variation (default: 0.7)",
    )
    parser.add_argument(
        "--size-variation-max",
        type=float,
        default=1.3,
        help="Maximum size scale for variation (default: 1.3)",
    )

    args = parser.parse_args()

    print("STENCIL STIPPLING")
    print("=" * 50)
    print(f"Input:       {args.input}")
    print(f"Output:      {args.output}")
    if args.export_stl:
        print(f"Export STL:  {args.export_stl}")
    print(f"Color:       {args.color}")
    print(f"Spheres/mm²: {args.spheres_per_mm2}")
    print(f"Radius:      {args.radius} mm")
    print(f"Depth:       {args.depth} mm")
    print(f"Strips:      {args.strips}")
    print(f"Overlap:     {args.overlap:.0%}")
    print(f"Batch:       {args.batch}")
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
        batch_size=args.batch,
        size_variation=not args.no_variation,
        size_variation_mode=args.size_variation_mode,
        size_variation_sigma=args.size_variation_sigma,
        size_variation_min=args.size_variation_min,
        size_variation_max=args.size_variation_max,
    )

    if result:
        print(f"\n✓ Successfully saved to: {result}")
        
        # Export STL if requested
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
    else:
        print("\n✗ Processing failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
