#!/usr/bin/env python3
"""CLI for bitmap-based boolean stippling (combines bitmap patterns with STEP boolean cuts)."""
import argparse
import sys
from pathlib import Path

from core.step_loader import STEPLoader
from core.bitmap_boolean_processor import BitmapBooleanProcessor


def main():
    parser = argparse.ArgumentParser(
        description="Apply stippling using bitmap patterns and boolean sphere cuts"
    )
    parser.add_argument("input", type=str, help="Input STEP file")
    parser.add_argument("-o", "--output", type=str, help="Output STEP file")
    parser.add_argument(
        "-c",
        "--color",
        type=str,
        default="#360200",
        help="Target color in hex format (default: #360200)",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=1.0,
        help="Sphere radius in mm (default: 1.0)",
    )
    parser.add_argument(
        "--depth",
        type=float,
        default=0.5,
        help="Sphere cut depth in mm (default: 0.5)",
    )
    parser.add_argument(
        "--bitmap-size",
        type=int,
        default=512,
        help="Bitmap size NxN (default: 512)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.35,
        help="Bitmap threshold for sphere placement 0-1 (default: 0.35)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for bitmap generation (default: 42)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="Timeout for individual cuts in seconds (default: 10.0)",
    )

    args = parser.parse_args()

    # Validate input
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Input file not found: {input_path}")
        sys.exit(1)

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / f"{input_path.stem}_bitmap_boolean.step"

    print("BITMAP BOOLEAN STIPPLING")
    print("=" * 50)
    print(f"Input:        {input_path}")
    print(f"Output:       {output_path}")
    print(f"Color:        {args.color}")
    print(f"Radius:       {args.radius} mm")
    print(f"Depth:        {args.depth} mm")
    print(f"Bitmap size:  {args.bitmap_size}×{args.bitmap_size}")
    print(f"Threshold:    {args.threshold}")
    print(f"Seed:         {args.seed}")
    print(f"Cut timeout:  {args.timeout}s")
    print("=" * 50)

    # Load STEP file
    loader = STEPLoader()
    if not loader.load(str(input_path)):
        print(f"❌ Failed to load: {input_path}")
        sys.exit(1)
    
    shape = loader.shape

    # Process with bitmap boolean processor
    processor = BitmapBooleanProcessor(
        sphere_radius=args.radius,
        sphere_depth=args.depth,
        bitmap_size=args.bitmap_size,
        threshold=args.threshold,
        seed=args.seed,
        cut_timeout=args.timeout,
    )

    result_shape = processor.process(shape, args.color, loader)

    if result_shape is None:
        print("❌ Processing failed")
        sys.exit(1)

    # Save result
    if loader.save_step(str(output_path), result_shape):
        print(f"\n✓ Successfully saved to: {output_path}")
    else:
        print(f"❌ Failed to save: {output_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
