#!/usr/bin/env python3
"""CLI for color-aware stippling using manifold3d boolean cuts."""

import argparse

from core.manifold_stipple_processor import ManifoldStippleProcessor


def main():
    parser = argparse.ArgumentParser(description="Color-aware stippling (manifold3d)")
    parser.add_argument("input", help="Input STEP file")
    parser.add_argument("-o", "--output", required=True, help="Output mesh file (STL/3MF/OBJ)")
    parser.add_argument("-c", "--color", required=True, help="Target color (hex, e.g., #360200)")

    parser.add_argument(
        "--spheres-per-mm2",
        type=float,
        default=0.5,
        help="Sphere density per mm² (default: 0.5)",
    )
    parser.add_argument("--radius", type=float, default=1.4, help="Sphere radius in mm (default: 1.4)")
    parser.add_argument("--depth", type=float, default=0.6, help="Cut depth in mm (default: 0.6)")
    parser.add_argument("--no-variation", action="store_true", help="Disable size variation")
    parser.add_argument(
        "--size-variation-mode",
        choices=["uniform", "gaussian"],
        default="gaussian",
        help="Size variation distribution (default: gaussian)",
    )
    parser.add_argument(
        "--size-variation-sigma",
        type=float,
        default=0.25,
        help="Gaussian sigma for size variation (default: 0.25)",
    )
    parser.add_argument(
        "--size-variation-min",
        type=float,
        default=0.60,
        help="Minimum size scale for variation (default: 0.60)",
    )
    parser.add_argument(
        "--size-variation-max",
        type=float,
        default=1.6,
        help="Maximum size scale for variation (default: 1.6)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--deflection", type=float, default=0.05, help="STEP→mesh deflection (default: 0.05)")

    args = parser.parse_args()

    processor = ManifoldStippleProcessor(random_seed=args.seed)
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
        deflection=args.deflection,
    )
    return 0 if result else 1


if __name__ == "__main__":
    raise SystemExit(main())
