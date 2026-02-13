#!/usr/bin/env python3
"""Calculate stipple coverage from tool parameters.

Uses the Poisson coverage model:
  remaining_fraction = exp(-effective_density * circle_area)

where:
  circle_area = pi * (2*R*d - d^2)   (opening of sphere cut)
  effective_density = applied_spheres / total_area
"""

import math
import sys


def circle_area(radius: float, depth: float) -> float:
    """Area of the circular opening when a sphere of given radius
    is cut to the given depth into a flat surface."""
    return math.pi * (2 * radius * depth - depth ** 2)


def remaining_fraction(density: float, radius: float, depth: float) -> float:
    """Fraction of surface area remaining uncovered (Poisson model)."""
    a = circle_area(radius, depth)
    return math.exp(-density * a)


def density_for_target(target_remaining: float, radius: float, depth: float) -> float:
    """Required density (spheres/mm²) to achieve target remaining fraction."""
    a = circle_area(radius, depth)
    return -math.log(target_remaining) / a


def main():
    R = 1.0    # sphere radius (mm)
    d = 0.5    # cut depth (mm)
    target = 0.05  # 5% remaining

    a = circle_area(R, d)
    required_density = density_for_target(target, R, d)

    print(f"Sphere radius:     {R} mm")
    print(f"Cut depth:         {d} mm")
    print(f"Opening area:      {a:.3f} mm²")
    print(f"Target remaining:  {target*100:.0f}%")
    print(f"Required density:  {required_density:.2f} spheres/mm²")
    print()

    # If command-line args: applied_spheres total_area
    if len(sys.argv) >= 3:
        applied = int(sys.argv[1])
        total_area = float(sys.argv[2])
        eff_density = applied / total_area
        remaining = remaining_fraction(eff_density, R, d)
        print(f"Applied spheres:   {applied}")
        print(f"Total area:        {total_area:.1f} mm²")
        print(f"Effective density: {eff_density:.3f} spheres/mm²")
        print(f"Remaining surface: {remaining*100:.1f}%")
        print()
        needed = int(math.ceil(required_density * total_area))
        print(f"To reach {target*100:.0f}%: need {needed} applied spheres "
              f"({required_density:.2f}/mm²)")
    else:
        # Show a table of densities vs remaining coverage
        print(f"{'Density':>10}  {'Applied*':>10}  {'Remaining':>10}")
        print(f"{'(sph/mm²)':>10}  {'(4109mm²)':>10}  {'(%)':>10}")
        print("-" * 36)
        total_area = 4108.5  # from sample_part
        for rho in [0.05, 0.1, 0.2, 0.3, 0.5, 0.6, 0.8, 1.0, 1.27, 1.5, 2.0]:
            rem = remaining_fraction(rho, R, d)
            n = int(rho * total_area)
            print(f"{rho:>10.2f}  {n:>10}  {rem*100:>9.1f}%")
        print()
        print(f"* Applied count assumes ALL requested spheres succeed.")
        print(f"  At high density, escalation stops early, so effective")
        print(f"  density < requested density.")
        print()
        print(f"Usage: python {sys.argv[0]} <applied_spheres> <total_area_mm2>")
        print(f"  e.g.: python {sys.argv[0]} 748 4108.5")


if __name__ == "__main__":
    main()
