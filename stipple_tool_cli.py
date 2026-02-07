#!/usr/bin/env python3
"""Demo for stencil-based stippling approach."""

import argparse
from core.stencil_processor import StencilStippleProcessor


def main():
    parser = argparse.ArgumentParser(description="Stencil-based stippling")
    parser.add_argument("input", help="Input STEP file")
    parser.add_argument("-o", "--output", required=True, help="Output STEP file")
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
    parser.add_argument("--batch", type=int, default=5, help="Batch size per local cut")
    parser.add_argument("--no-variation", action="store_true", help="Disable size variation")

    args = parser.parse_args()

    print("STENCIL STIPPLING")
    print("=" * 50)
    print(f"Input:       {args.input}")
    print(f"Output:      {args.output}")
    print(f"Color:       {args.color}")
    print(f"Spheres/mm²: {args.spheres_per_mm2}")
    print(f"Radius:      {args.radius} mm")
    print(f"Depth:       {args.depth} mm")
    print(f"Strips:      {args.strips}")
    print(f"Overlap:     {args.overlap:.0%}")
    print(f"Batch:       {args.batch}")
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
    )

    if result:
        print(f"\n✓ Successfully saved to: {result}")
        return 0
    else:
        print("\n✗ Processing failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
