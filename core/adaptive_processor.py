"""
Adaptive stippling processor - applies initial stippling, then fills gaps.

Strategy:
1. Initial pass: Apply stippling at moderate density across all target surfaces
2. Analysis pass: Identify areas with insufficient coverage (too much original surface)
3. Fill pass: Add targeted stippling to under-covered regions
4. Repeat until coverage is uniform

This avoids the exponential complexity of dense single-pass stippling.
"""

import time
import math
from typing import List, Optional, Callable, Tuple
from pathlib import Path

from OCP.TopoDS import TopoDS_Shape, TopoDS
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_FACE
from OCP.GProp import GProp_GProps
from OCP.BRepGProp import BRepGProp

from core.step_loader import STEPLoader
from core.color_analyzer import ColorAnalyzer
from core.incremental_processor import IncrementalStippleProcessor


class AdaptiveStippleProcessor:
    """Adaptive multi-pass stippling for better coverage."""

    def __init__(self):
        self.color_analyzer = ColorAnalyzer()
        self.stipple_engine = IncrementalStippleProcessor()

    def process_step_with_adaptive_stippling(
        self,
        step_file: str,
        output_path: str,
        target_color: str,
        sphere_radius: float = 1.0,
        sphere_depth: float = 0.5,
        target_density: float = 0.8,
        surface_remaining: Optional[float] = None,
        spheres_per_mm2: float = 0.04,
        size_variation: bool = True,
        max_passes: int = 3,
        pass_timeout_seconds: Optional[float] = 60.0,
        progress_callback: Optional[Callable] = None,
        status_callback: Optional[Callable] = None,
        cancel_callback: Optional[Callable] = None,
    ) -> Optional[str]:
        """
        Apply stippling adaptively in multiple passes.
        
        Args:
            step_file: Input STEP file
            output_path: Output STEP file
            target_color: Color to stipple
            sphere_radius: Stipple radius in mm
            sphere_depth: Depth of cuts in mm
            target_density: Target coverage (0.0-1.0)
            size_variation: Vary sphere sizes
            max_passes: Maximum number of refinement passes
            
        Returns:
            Output path on success, None on failure
        """
        def emit_status(msg: str):
            if status_callback:
                status_callback(msg)
            print(msg)

        def check_cancel():
            if cancel_callback:
                cancel_callback()

        try:
            emit_status("=== ADAPTIVE STIPPLING ===")
            if surface_remaining is not None:
                target_density = max(0.0, min(1.0, 1.0 - surface_remaining))
                emit_status(f"Target surface remaining: {surface_remaining:.1%}")
            emit_status(f"Target density: {target_density:.1%}")
            emit_status(f"Sphere density: {spheres_per_mm2:.3f} per mm²")
            emit_status(f"Max passes: {max_passes}")
            
            # Load model
            loader = STEPLoader()
            if not loader.load(step_file):
                emit_status("Failed to load STEP file")
                return None

            # Get target faces
            colors = self.color_analyzer.extract_colors_from_model({
                "faces": loader.faces,
                "shape_tool": loader.shape_tool,
                "color_tool": loader.color_tool,
            })

            target_faces = colors.get(target_color, [])
            if not target_faces:
                emit_status(f"No surfaces found with color {target_color}")
                return None

            emit_status(f"Found {len(target_faces)} target faces")
            
            # Calculate total target surface area
            total_area = self._calculate_surface_area(loader.shape, target_faces)
            emit_status(f"Total target surface area: {total_area:.1f} mm²")
            
            # Pass 1: Initial moderate stippling (30% of target)
            emit_status("\n=== PASS 1: Initial Coverage ===")
            initial_density = min(0.3, target_density)
            initial_spheres = int(total_area * spheres_per_mm2 * (initial_density / target_density))
            emit_status(f"Applying {initial_spheres} spheres ({initial_density:.1%} density)")
            
            current_file = step_file
            result = self.stipple_engine.process_step_with_incremental_stippling(
                step_file=current_file,
                output_path=output_path,
                target_color=target_color,
                sphere_radius=sphere_radius,
                sphere_depth=sphere_depth,
                num_spheres=initial_spheres,
                batch_size=30,
                size_variation=size_variation,
                status_callback=status_callback,
                cancel_callback=cancel_callback,
                return_stats=True,
            )
            
            if not result:
                emit_status("Pass 1 failed")
                return None
            
            current_file, stats = result
            cumulative_successful = stats.get("successful_cuts", 0) if stats else 0
            check_cancel()

            # Note: After pass 1, color information is lost in boolean operations
            # For subsequent passes, just use all visible geometry (target_faces = [])
            # to allow spheres to fill remaining gaps naturally

            # Estimate coverage after pass 1
            covered_ratio = self._estimate_coverage_ratio(
                cumulative_successful,
                sphere_radius,
                total_area,
            )
            remaining_ratio = max(0.0, 1.0 - covered_ratio)
            emit_status(
                f"Estimated surface remaining after pass 1: {remaining_ratio:.1%}"
            )
            
            # Additional passes: Fill in gaps
            pass_num = 2
            while pass_num <= max_passes:
                emit_status(f"\n=== PASS {pass_num}: Gap Filling ===")
                pass_start_time = time.monotonic()

                def pass_cancel():
                    if cancel_callback:
                        cancel_callback()
                    if pass_timeout_seconds is not None:
                        if (time.monotonic() - pass_start_time) > pass_timeout_seconds:
                            raise TimeoutError(
                                f"Pass {pass_num} exceeded {pass_timeout_seconds:.0f}s timeout"
                            )
                
                # Estimate current coverage (simplified - assume linear accumulation)
                if surface_remaining is not None and remaining_ratio <= surface_remaining:
                    emit_status("Target surface remaining reached")
                    break
                if surface_remaining is None and covered_ratio >= target_density:
                    emit_status("Target density reached")
                    break
                
                # Add more spheres proportional to remaining coverage gap
                target_coverage = target_density if surface_remaining is None else (1.0 - surface_remaining)
                remaining_gap = max(0.0, target_coverage - covered_ratio)
                # Scale back sphere count on later passes to avoid exponential slowdown
                # Pass 2: modest reduction, Pass 3+: aggressive reduction
                if pass_num >= 4:
                    pass_reduction_factor = 0.25
                elif pass_num >= 3:
                    pass_reduction_factor = 0.35
                else:
                    pass_reduction_factor = 0.6
                additional_spheres = max(5, int(total_area * spheres_per_mm2 * remaining_gap * pass_reduction_factor))
                # Hard cap to prevent hanging on complex geometry
                if pass_num >= 3:
                    max_spheres_this_pass = 10
                else:
                    max_spheres_this_pass = max(10, int(initial_spheres * 0.15))
                additional_spheres = min(additional_spheres, max_spheres_this_pass)
                emit_status(f"Adding {additional_spheres} more spheres (pass reduction: {pass_reduction_factor:.1%})")
                
                result = self.stipple_engine.process_step_with_incremental_stippling(
                    step_file=current_file,
                    output_path=output_path,
                    target_color=None,  # Use existing file directly
                    sphere_radius=sphere_radius,
                    sphere_depth=sphere_depth,
                    num_spheres=additional_spheres,
                    batch_size=1,  # Single-sphere batches to avoid long boolean ops in pass 2+
                    target_faces=[],  # Empty = use all geometry (color info lost after pass 1)
                    size_variation=size_variation,
                    status_callback=status_callback,
                    cancel_callback=pass_cancel,
                    return_stats=True,
                )
                
                pass_elapsed = time.monotonic() - pass_start_time
                
                if not result:
                    if pass_timeout_seconds is not None and pass_elapsed > pass_timeout_seconds:
                        emit_status(f"Pass {pass_num} timed out after {pass_elapsed:.1f}s - stopping")
                    else:
                        emit_status(f"Pass {pass_num} failed, using previous result")
                    break
                
                current_file, stats = result
                cumulative_successful += stats.get("successful_cuts", 0) if stats else 0
                emit_status(f"Pass {pass_num} completed in {pass_elapsed:.1f}s")

                # If cuts are getting too slow, stop before next pass
                if stats and stats.get("successful_cuts", 0) > 0:
                    avg_cut_time = pass_elapsed / max(1, stats.get("successful_cuts", 0))
                    if pass_num >= 3 and avg_cut_time > 1.0:
                        emit_status(
                            f"Average cut time {avg_cut_time:.2f}s too high - stopping"
                        )
                        break
                
                # Stop if a pass takes way too long (geometry getting unwieldy)
                if pass_elapsed > 120:  # More than 2 minutes per pass
                    emit_status(f"Pass {pass_num} took {pass_elapsed:.1f}s - geometry too complex, stopping")
                    break
                
                check_cancel()

                # Recompute target faces for the next pass by color
                target_faces = self._identify_target_faces_by_color(
                    current_file,
                    target_color,
                    emit_status=emit_status,
                    check_cancel=check_cancel,
                )
                # Note: After boolean operations, color info is lost
                # Just continue with all-geometry mode on pass 2+
                # For pass 2+, use empty list to apply spheres across all geometry
                target_faces_for_pass = []

                # Re-estimate coverage after this pass
                covered_ratio = self._estimate_coverage_ratio(
                    cumulative_successful,
                    sphere_radius,
                    total_area,
                )
                remaining_ratio = max(0.0, 1.0 - covered_ratio)
                emit_status(
                    f"Estimated surface remaining after pass {pass_num}: {remaining_ratio:.1%}"
                )

                pass_num += 1
            
            emit_status(f"\n✓ Adaptive stippling complete: {output_path}")
            return output_path

        except Exception as e:
            import traceback
            emit_status(f"Error during adaptive stippling: {e}")
            traceback.print_exc()
            return None

    def _calculate_surface_area(
        self,
        shape: TopoDS_Shape,
        target_face_indices: List[int],
    ) -> float:
        """Calculate total surface area of target faces."""
        total_area = 0.0
        
        explorer = TopExp_Explorer(shape, TopAbs_FACE)
        face_idx = 0
        
        while explorer.More():
            if face_idx in target_face_indices:
                face = TopoDS.Face_s(explorer.Current())
                props = GProp_GProps()
                BRepGProp.SurfaceProperties_s(face, props)
                total_area += props.Mass()  # Mass() gives surface area for 2D
            
            explorer.Next()
            face_idx += 1
        
        return total_area

    def _estimate_coverage_ratio(
        self,
        successful_cuts: int,
        sphere_radius: float,
        total_area: float,
    ) -> float:
        """Estimate coverage ratio based on number of successful cuts."""
        if total_area <= 0:
            return 0.0
        # Approximate coverage per stipple: area of circle * packing factor
        area_per_sphere = math.pi * (sphere_radius ** 2) * 0.75
        covered_area = successful_cuts * area_per_sphere
        return max(0.0, min(1.0, covered_area / total_area))
    def _identify_target_faces_by_color(
        self,
        step_file: str,
        target_color: str,
        emit_status: Optional[Callable] = None,
        check_cancel: Optional[Callable] = None,
    ) -> List[int]:
        """Re-identify target faces in updated file by color."""
        loader = STEPLoader()
        if not loader.load(step_file):
            if emit_status:
                emit_status("Failed to reload STEP for target remap")
            return []

        try:
            colors = self.color_analyzer.extract_colors_from_model({
                "faces": loader.faces,
                "shape_tool": loader.shape_tool,
                "color_tool": loader.color_tool,
            })
            target_faces = colors.get(target_color, [])
            if emit_status and target_faces:
                emit_status(f"Remapped {len(target_faces)} target faces")
            return target_faces
        except Exception as e:
            if emit_status:
                emit_status(f"Color remapping failed: {e} (will use all geometry)")
            return []