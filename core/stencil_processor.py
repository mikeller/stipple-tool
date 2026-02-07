"""
Stencil-based stippling processor.

Strategy:
1. Identify target colored faces on the original shape.
2. Compute bounding box over target faces.
3. Sweep strips along longest axis.
4. For each strip:
   - Filter target faces whose centroids fall within the strip bounds.
   - Generate sphere positions only for those faces.
   - Apply spheres directly to the main shape (small batch).

This limits complexity by only generating spheres for a subset of faces at a time,
while always operating on the single main shape.
"""

import math
import random
import time
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
from OCP.BRepBndLib import BRepBndLib
from OCP.BRepGProp import BRepGProp
from OCP.BRepPrimAPI import BRepPrimAPI_MakeSphere
from OCP.Bnd import Bnd_Box
from OCP.GProp import GProp_GProps
from OCP.ShapeFix import ShapeFix_Shape
from OCP.TopAbs import TopAbs_FACE
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS, TopoDS_Shape, TopoDS_Compound, TopoDS_Builder
from OCP.gp import gp_Pnt, gp_Vec

from core.color_analyzer import ColorAnalyzer
from core.step_loader import STEPLoader


class StencilStippleProcessor:
    """Stencil-based stippling to reduce boolean complexity."""

    def __init__(self):
        self.color_analyzer = ColorAnalyzer()

    def process_step_with_stencil_stippling(
        self,
        step_file: str,
        output_path: str,
        target_color: str,
        sphere_radius: float = 1.0,
        sphere_depth: float = 0.5,
        spheres_per_mm2: float = 0.12,
        strip_count: int = 6,
        overlap: float = 0.2,
        batch_size: int = 3,
        size_variation: bool = True,
        status_callback: Optional[Callable] = None,
        cancel_callback: Optional[Callable] = None,
    ) -> Optional[str]:
        def emit_status(msg: str):
            if status_callback:
                status_callback(msg)
            else:
                print(msg)

        def check_cancel():
            if cancel_callback:
                cancel_callback()

        try:
            emit_status("=== STENCIL STIPPLING ===")

            loader = STEPLoader()
            if not loader.load(step_file):
                emit_status("Failed to load STEP file")
                return None

            step_model = loader.get_model()
            colors = self.color_analyzer.extract_colors_from_model(step_model)
            target_face_indices = colors.get(target_color, [])
            if not target_face_indices:
                emit_status(f"No surfaces found with color {target_color}")
                return None

            emit_status(f"Found {len(target_face_indices)} target faces")

            # Extract target face objects and their centroids
            target_faces_data = []  # [(face_obj, centroid, area), ...]
            explorer = TopExp_Explorer(loader.shape, TopAbs_FACE)
            face_idx = 0
            while explorer.More():
                if face_idx in target_face_indices:
                    face = TopoDS.Face_s(explorer.Current())
                    props = GProp_GProps()
                    BRepGProp.SurfaceProperties_s(face, props)
                    centroid = props.CentreOfMass()
                    area = props.Mass()
                    target_faces_data.append((face, (centroid.X(), centroid.Y(), centroid.Z()), area))
                explorer.Next()
                face_idx += 1

            emit_status(f"Extracted {len(target_faces_data)} target face geometries")

            # Compute bounding box over target faces
            bbox = Bnd_Box()
            for face, _, _ in target_faces_data:
                BRepBndLib.Add_s(face, bbox)
            min_x, min_y, min_z, max_x, max_y, max_z = bbox.Get()

            # Determine longest axis for strip sweep
            dx, dy, dz = max_x - min_x, max_y - min_y, max_z - min_z
            if dx >= dy and dx >= dz:
                axis = 0  # x
                axis_min, axis_max = min_x, max_x
            elif dy >= dx and dy >= dz:
                axis = 1  # y
                axis_min, axis_max = min_y, max_y
            else:
                axis = 2  # z
                axis_min, axis_max = min_z, max_z

            axis_names = ["x", "y", "z"]
            axis_len = max(1e-6, axis_max - axis_min)
            strip_len = axis_len / max(1, strip_count)
            overlap_len = strip_len * max(0.0, min(0.5, overlap))

            emit_status(
                f"Sweep axis={axis_names[axis]}, range=[{axis_min:.2f}, {axis_max:.2f}], strips={strip_count}"
            )

            current_shape = loader.shape
            total_spheres_applied = 0

            for i in range(strip_count):
                check_cancel()
                strip_start = axis_min + i * strip_len - overlap_len
                strip_end = axis_min + (i + 1) * strip_len + overlap_len

                emit_status(f"\n=== STRIP {i + 1}/{strip_count} [{strip_start:.2f} - {strip_end:.2f}] ===")

                # Filter faces whose centroid falls in this strip
                strip_faces = []
                for face, centroid, area in target_faces_data:
                    c_val = centroid[axis]
                    if strip_start <= c_val <= strip_end:
                        strip_faces.append((face, centroid, area))

                if not strip_faces:
                    emit_status("No target faces in this strip - skipping")
                    continue

                strip_area = sum(area for _, _, area in strip_faces)
                num_spheres = max(3, int(strip_area * spheres_per_mm2))
                # Cap to prevent complexity blowup - allow more per strip for dense coverage
                max_per_strip = 150
                num_spheres = min(num_spheres, max_per_strip)

                emit_status(f"Faces: {len(strip_faces)}, Area: {strip_area:.1f} mm², Spheres: {num_spheres}")

                # Generate sphere positions on these faces
                sphere_positions = self._generate_sphere_positions_on_faces(
                    strip_faces, num_spheres, sphere_radius, size_variation
                )

                if not sphere_positions:
                    emit_status("No sphere positions generated - skipping")
                    continue

                emit_status(f"Generated {len(sphere_positions)} sphere positions")

                # Apply spheres in small batches
                strip_spheres = 0
                failed_batches = 0
                num_batches = (len(sphere_positions) + batch_size - 1) // batch_size
                for batch_idx, batch_start in enumerate(range(0, len(sphere_positions), batch_size)):
                    check_cancel()
                    batch = sphere_positions[batch_start:batch_start + batch_size]

                    # Progress indicator
                    if batch_idx % 5 == 0 or batch_idx == num_batches - 1:
                        emit_status(f"  Batch {batch_idx + 1}/{num_batches}...")

                    # Create compound of spheres
                    builder = TopoDS_Builder()
                    compound = TopoDS_Compound()
                    builder.MakeCompound(compound)

                    spheres_in_batch = 0
                    for center, radius in batch:
                        try:
                            sphere = BRepPrimAPI_MakeSphere(center, radius).Shape()
                            builder.Add(compound, sphere)
                            spheres_in_batch += 1
                        except Exception as e:
                            emit_status(f"Sphere creation failed: {e}")

                    if spheres_in_batch == 0:
                        continue

                    # Cut from current shape with timing
                    try:
                        cut_start = time.time()
                        cut_op = BRepAlgoAPI_Cut(current_shape, compound)
                        cut_op.Build()
                        cut_time = time.time() - cut_start
                        
                        if cut_time > 10:
                            emit_status(f"  Warning: batch {batch_idx + 1} took {cut_time:.1f}s")
                        
                        if cut_op.IsDone():
                            new_shape = cut_op.Shape()
                            if not new_shape.IsNull():
                                # Accept the cut - shape validity is confirmed by IsDone() and IsNull() checks
                                # (cut result might be a shell/face and legitimately have 0 volume)
                                current_shape = new_shape
                                total_spheres_applied += spheres_in_batch
                                strip_spheres += spheres_in_batch
                            else:
                                failed_batches += 1
                        else:
                            failed_batches += 1
                    except Exception as e:
                        emit_status(f"Batch cut failed: {e}")

                emit_status(f"Strip {i + 1} complete: {strip_spheres} spheres, total: {total_spheres_applied}", )
                if failed_batches > 0:
                    emit_status(f"  ({failed_batches} batches could not be applied)")

            # Save final shape
            emit_status(f"\nSaving result: {output_path}")
            emit_status(f"Total spheres applied: {total_spheres_applied}")
            if not loader.save_step(output_path, current_shape):
                emit_status("Failed to write final STEP")
                return None

            emit_status("✓ Stencil stippling complete")
            return output_path

        except Exception as e:
            import traceback
            emit_status(f"Error during stencil stippling: {e}")
            traceback.print_exc()
            return None

    def _generate_sphere_positions_on_faces(
        self,
        faces_data: List[Tuple[TopoDS_Shape, Tuple[float, float, float], float]],
        num_spheres: int,
        base_radius: float,
        size_variation: bool,
    ) -> List[Tuple[gp_Pnt, float]]:
        """Generate sphere positions distributed across faces by area."""
        from OCP.BRepAdaptor import BRepAdaptor_Surface

        positions = []
        total_area = sum(area for _, _, area in faces_data)
        if total_area <= 0:
            return positions

        for face, _, area in faces_data:
            # Allocate spheres proportionally to face area
            face_spheres = max(1, int(num_spheres * (area / total_area)))

            try:
                adaptor = BRepAdaptor_Surface(face)
                u_min, u_max = adaptor.FirstUParameter(), adaptor.LastUParameter()
                v_min, v_max = adaptor.FirstVParameter(), adaptor.LastVParameter()

                for _ in range(face_spheres):
                    # Random UV position
                    u = u_min + random.random() * (u_max - u_min)
                    v = v_min + random.random() * (v_max - v_min)

                    # Get point on surface
                    pnt = adaptor.Value(u, v)

                    # Compute normal via derivative
                    try:
                        d1u = gp_Vec()
                        d1v = gp_Vec()
                        p_temp = gp_Pnt()
                        adaptor.D1(u, v, p_temp, d1u, d1v)
                        normal = d1u.Crossed(d1v)
                        if normal.Magnitude() > 1e-6:
                            normal.Normalize()
                        else:
                            normal = gp_Vec(0, 0, 1)
                    except Exception:
                        normal = gp_Vec(0, 0, 1)

                    # Offset point inward by sphere depth
                    depth = base_radius * 0.5
                    center = gp_Pnt(
                        pnt.X() - normal.X() * depth,
                        pnt.Y() - normal.Y() * depth,
                        pnt.Z() - normal.Z() * depth,
                    )

                    # Vary radius if enabled
                    if size_variation:
                        radius = base_radius * (0.6 + random.random() * 0.8)
                    else:
                        radius = base_radius

                    positions.append((center, radius))

                    if len(positions) >= num_spheres:
                        return positions

            except Exception:
                continue

        return positions
