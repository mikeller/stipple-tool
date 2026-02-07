"""Incremental STEP stippling with batch processing."""
from pathlib import Path
from typing import Dict, List, Optional, Callable
import time
import numpy as np
from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
from OCP.BRepPrimAPI import BRepPrimAPI_MakeSphere

from core.step_loader import STEPLoader
from core.color_analyzer import ColorAnalyzer


class IncrementalStippleProcessor:
    """Apply stippling to STEP files incrementally, one sphere at a time."""

    def __init__(self):
        self.step_loader = STEPLoader()
        self.color_analyzer = ColorAnalyzer()

    def process_step_with_incremental_stippling(
        self,
        step_file: str,
        output_path: str,
        target_color: str,
        sphere_radius: float = 1.0,
        sphere_depth: float = 0.5,
        num_spheres: int = 50,
        batch_size: int = 30,
        target_faces: Optional[List[int]] = None,
        size_variation: bool = False,
        progress_callback: Optional[Callable] = None,
        batch_progress_callback: Optional[Callable] = None,
        status_callback: Optional[Callable] = None,
        cancel_callback: Optional[Callable] = None,
        status_prefix: str = "",
        return_stats: bool = False,
    ) -> Optional[str]:
        """
        Apply stippling to STEP file incrementally.
        
        Args:
            step_file: Path to input STEP file
            output_path: Path to output STEP file
            target_color: Color hex code to stipple
            sphere_radius: Radius of stipple dimples
            sphere_depth: Depth to push spheres
            num_spheres: Total number of stipples
            batch_size: How many spheres to apply per batch
            target_faces: List of face indices to stipple
            size_variation: If True, vary sphere radius from 0.5x to 1.5x
            progress_callback: Callback(current, total)
            status_callback: Callback(message)
            sphere_depth: Depth to push spheres
            num_spheres: Total number of stipples
            batch_size: How many spheres to apply per batch
            progress_callback: Callback(current, total)
            status_callback: Callback(message)
            
        Returns:
            Output file path on success, None on failure
        """
        
        def emit_progress(current, total):
            if progress_callback:
                progress_callback(current, total)

        def emit_status(msg):
            if status_callback:
                status_callback(f"{status_prefix}{msg}")

        def check_cancel():
            if cancel_callback:
                cancel_callback()

        try:
            emit_status("Incremental processor: v2 (adaptive batching, 5% collision sampling, heal every 50 batches)")
            emit_status("Loading STEP file...")
            if not self.step_loader.load(step_file):
                emit_status("Failed to load STEP file")
                return None

            step_model = self.step_loader.get_model()
            emit_status(f"Loaded: {len(step_model.get('faces', []))} faces")
            emit_progress(5, 100)

            # Use provided target_faces or extract from colors
            if target_faces is None:
                # Extract colors
                emit_status("Analyzing colors...")
                color_groups = self.color_analyzer.extract_colors_from_model(step_model)
                if target_color not in color_groups:
                    emit_status(f"Target color {target_color} not found")
                    return None
                
                target_faces = color_groups[target_color]
                emit_status(f"Found {len(target_faces)} faces with color {target_color}")
            elif len(target_faces) == 0:
                # Empty list means "no collision checking" (for pass 2+ of adaptive stippling)
                emit_status(f"No target faces specified - collision checking disabled")
                target_faces = None
            else:
                # Using directly provided face indices (from partition matching)
                emit_status(f"Using {len(target_faces)} pre-identified target faces")

            # Get all face indices for collision detection
            all_face_indices = set(range(len(step_model.get('faces', []))))
            non_target_face_indices = all_face_indices - set(target_faces if target_faces else [])
            collision_checking_enabled = (target_faces is not None and len(target_faces) > 0)
            
            emit_progress(10, 100)

            # Generate sphere centers on target faces
            emit_status("Generating sphere positions...")
            check_cancel()
            # When target_faces is None (pass 2+ of adaptive), use all faces
            faces_for_placement = target_faces if target_faces is not None else list(all_face_indices)
            sphere_data = self._generate_sphere_positions(
                step_model, faces_for_placement, num_spheres, sphere_radius, size_variation
            )
            emit_status(f"Generated {len(sphere_data)} sphere positions")
            emit_progress(15, 100)

            # Apply spheres incrementally in batches
            emit_status("Applying stipples incrementally...")
            current_shape = step_model.get("shape")
            original_shape = current_shape  # Store original for collision detection
            successful_cuts = 0
            failed_cuts = 0
            skipped_collisions = 0

            total_spheres = len(sphere_data)
            last_batch_update = time.monotonic()

            # Precompute non-target face objects + bounding boxes for fast rejection
            from OCP.TopExp import TopExp_Explorer
            from OCP.TopAbs import TopAbs_FACE
            from OCP.TopoDS import TopoDS
            from OCP.Bnd import Bnd_Box
            from OCP.BRepBndLib import BRepBndLib

            non_target_faces = []  # [(face, bbox), ...]
            if collision_checking_enabled:
                explorer = TopExp_Explorer(original_shape, TopAbs_FACE)
                face_idx = 0
                while explorer.More():
                    if face_idx in non_target_face_indices:
                        face = TopoDS.Face_s(explorer.Current())
                        bbox = Bnd_Box()
                        BRepBndLib.Add_s(face, bbox)
                        non_target_faces.append((face, bbox))
                    face_idx += 1
                    explorer.Next()
            
            # Adaptive batch sizing to counteract growing boolean complexity
            initial_batch_size = batch_size
            min_batch_size = max(5, int(batch_size * 0.15))
            max_batch_size = max(50, batch_size * 3)  # Allow much larger batches early
            target_batch_seconds = 10.0  # Even more lenient (10 seconds)
            heal_frequency = 50  # Heal only every 50 batches (10x less healing!)
            processed = 0
            batch_number = 0
            recent_apply_rate = 1.0  # Track recent success rate

            while processed < len(sphere_data):
                check_cancel()
                batch = sphere_data[processed : processed + batch_size]
                batch_total = len(batch)
                batch_number += 1
                batch_start = time.monotonic()

                if batch_progress_callback:
                    batch_progress_callback(0)
                
                # Pre-filter spheres that would collide with non-target faces
                # Sample-based collision detection: only check 10% of spheres (2x faster)
                # This still prevents most bleeding while keeping performance very fast
                import random
                valid_spheres = []
                for sphere_idx, (center, radius) in enumerate(batch):
                    if (sphere_idx % 5) == 0:
                        check_cancel()
                    overall_idx = processed + sphere_idx
                    try:
                        # Create sphere with the specific radius for this sphere
                        sphere = BRepPrimAPI_MakeSphere(center, radius).Shape()
                        
                        # Collision checking: only check 5% for speed
                        # Skip entirely if collision_checking_enabled is False (pass 2+ of adaptive)
                        if collision_checking_enabled and random.random() < 0.05:
                            if self._sphere_intersects_non_target_faces(
                                sphere, center, radius, non_target_faces
                            ):
                                skipped_collisions += 1
                                continue
                        
                        valid_spheres.append((sphere, overall_idx))
                    except Exception:
                        failed_cuts += 1

                    # Update in-batch progress and status periodically
                    if batch_progress_callback:
                        now = time.monotonic()
                        if (sphere_idx + 1) == batch_total or (now - last_batch_update) > 0.5:
                            batch_progress = int(((sphere_idx + 1) / max(1, batch_total)) * 100)
                            batch_progress_callback(batch_progress)
                            emit_status(
                                f"Batch {batch_number}: Checking {sphere_idx + 1}/{batch_total} | Overall: {min(processed + sphere_idx + 1, total_spheres)}/{total_spheres}"
                            )
                            last_batch_update = now
                
                # Signal end of collision checking phase
                if batch_progress_callback:
                    batch_progress_callback(100)
                
                # Apply all valid spheres in this batch as a compound cut
                if valid_spheres:
                    try:
                        from OCP.BRepBuilderAPI import BRepBuilderAPI_Sewing
                        from OCP.TopoDS import TopoDS_Compound, TopoDS_Builder
                        
                        # Create compound of all spheres in batch
                        emit_status(f"Batch {batch_number}: Creating compound of {len(valid_spheres)} spheres | Overall: {processed}/{total_spheres}")
                        if batch_progress_callback:
                            batch_progress_callback(0)
                        builder = TopoDS_Builder()
                        compound = TopoDS_Compound()
                        builder.MakeCompound(compound)
                        
                        for sphere, _ in valid_spheres:
                            builder.Add(compound, sphere)
                        
                        if len(valid_spheres) == 0:
                            failed_cuts += 0
                            continue

                        # Cut all spheres at once (more efficient than sequential cuts)
                        check_cancel()
                        emit_status(f"Batch {batch_number}: Cutting {len(valid_spheres)} spheres (may take a while) | Overall: {processed}/{total_spheres}")
                        if batch_progress_callback:
                            batch_progress_callback(50)
                        cut_op = BRepAlgoAPI_Cut(current_shape, compound)
                        cut_op.Build()
                        
                        if cut_op.IsDone():
                            result_shape = cut_op.Shape()
                            
                            # Heal geometry only every 20 batches to minimize slowdowns
                            # (healing becomes exponentially expensive as geometry grows)
                            if batch_number % heal_frequency == 0:
                                emit_status(f"Batch {batch_number}: Healing geometry | Overall: {processed}/{total_spheres}")
                                if batch_progress_callback:
                                    batch_progress_callback(75)
                                try:
                                    from OCP.ShapeFix import ShapeFix_Shape
                                    healer = ShapeFix_Shape()
                                    healer.Init(result_shape)
                                    healer.Perform()
                                    current_shape = healer.Shape()
                                except:
                                    # If healing fails, just use the raw result
                                    current_shape = result_shape
                            else:
                                # Skip healing to maintain performance
                                current_shape = result_shape
                            
                            if batch_progress_callback:
                                batch_progress_callback(100)
                            
                            successful_cuts += len(valid_spheres)
                        else:
                            failed_cuts += len(valid_spheres)
                    except Exception:
                        failed_cuts += len(valid_spheres)

                # Update processed count and adapt batch size based on timing
                processed += batch_total
                batch_seconds = time.monotonic() - batch_start
                
                # Track recent apply rate to detect diminishing returns
                current_apply_rate = len(valid_spheres) / batch_total if batch_total > 0 else 0
                recent_apply_rate = 0.9 * recent_apply_rate + 0.1 * current_apply_rate
                
                emit_status(f"Batch {batch_number}: Completed in {batch_seconds:.1f}s | Apply rate: {current_apply_rate*100:.0f}% | Overall: {processed}/{total_spheres}")
                
                # Early exit if hitting severe diminishing returns (less than 5% being applied)
                if recent_apply_rate < 0.05 and batch_number > 10:
                    emit_status(f"Stopping at batch {batch_number}: Apply rate dropped to {recent_apply_rate*100:.1f}% (too slow to continue)")
                    break
                
                # Aggressively reduce batch size if processing slows down
                if batch_seconds > target_batch_seconds * 1.5 and batch_size > min_batch_size:
                    new_batch_size = max(min_batch_size, int(batch_size * 0.4))  # Cut to 40%
                    if new_batch_size != batch_size:
                        batch_size = new_batch_size
                        emit_status(f"Reducing batch size to {batch_size} (last batch took {batch_seconds:.1f}s)")
                elif batch_seconds < target_batch_seconds * 0.4 and batch_size < max_batch_size:
                    new_batch_size = min(max_batch_size, int(batch_size * 1.5))  # Increase by 50%
                    if new_batch_size != batch_size:
                        batch_size = new_batch_size
                        emit_status(f"Increasing batch size to {batch_size} (last batch took {batch_seconds:.1f}s)")
                
                # Progress update after batch
                progress = 15 + int((processed / len(sphere_data)) * 75)
                emit_progress(progress, 100)

            emit_status(f"Finished: {successful_cuts} successful cuts, {failed_cuts} failed, {skipped_collisions} skipped (collision)")
            emit_progress(90, 100)

            # Save result
            emit_status(f"Saving to {Path(output_path).name}...")
            success = self.step_loader.save_step(output_path, current_shape)
            
            stats = {
                "successful_cuts": successful_cuts,
                "failed_cuts": failed_cuts,
                "skipped_collisions": skipped_collisions,
                "total_spheres": total_spheres,
            }

            if success:
                emit_status("Saved successfully")
                emit_progress(100, 100)
                return (output_path, stats) if return_stats else output_path
            else:
                emit_status("Failed to save STEP file")
                return (None, stats) if return_stats else None

        except Exception as e:
            emit_status(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def _sphere_intersects_non_target_faces(
        self, sphere, center, radius, non_target_faces
    ) -> bool:
        """Check if a sphere would intersect any non-target faces.
        
        Returns True if intersection detected (sphere should be skipped).
        """
        from OCP.BRepExtrema import BRepExtrema_DistShapeShape
        from OCP.Bnd import Bnd_Box
        
        try:
            # Build a fast bounding box for the sphere
            sphere_bbox = Bnd_Box()
            sphere_bbox.Update(
                center.X() - radius,
                center.Y() - radius,
                center.Z() - radius,
                center.X() + radius,
                center.Y() + radius,
                center.Z() + radius,
            )

            # Iterate through non-target faces and check distance only if bbox overlaps
            for face, face_bbox in non_target_faces:
                if sphere_bbox.IsOut(face_bbox):
                    continue

                # Calculate minimum distance between sphere and face
                dist_calc = BRepExtrema_DistShapeShape()
                dist_calc.LoadS1(sphere)
                dist_calc.LoadS2(face)
                dist_calc.Perform()
                
                # Use slightly negative threshold to only reject actual intersections
                # (positive distance = separated, negative = overlap)
                if dist_calc.IsDone() and dist_calc.Value() < -0.01:
                    # Sphere significantly intersects this non-target face
                    return True
            
            return False
        except:
            # If distance calculation fails, conservatively skip the sphere
            return True

    def _generate_sphere_positions(
        self, step_model: Dict, target_faces: List[int], num_spheres: int, sphere_radius: float = 1.0, size_variation: bool = False
    ) -> List:
        """Generate positions for stipples ON target faces (not just in bounding box).
        
        Returns list of (center_point, radius) tuples.
        """
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopAbs import TopAbs_FACE
        from OCP.TopoDS import TopoDS
        from OCP.BRepAdaptor import BRepAdaptor_Surface
        from OCP.BRepGProp import BRepGProp
        from OCP.GProp import GProp_GProps
        from OCP.gp import gp_Pnt, gp_Vec
        import numpy as np
        
        positions = []
        shape = step_model.get("shape")
        
        # Get target faces as actual face objects and calculate their areas
        explorer = TopExp_Explorer(shape, TopAbs_FACE)
        face_idx = 0
        target_face_data = []  # [(face_obj, area), ...]
        
        while explorer.More():
            if face_idx in target_faces:
                # Cast TopoDS_Shape to TopoDS_Face
                face_obj = TopoDS.Face_s(explorer.Current())
                
                # Calculate face area
                props = GProp_GProps()
                BRepGProp.SurfaceProperties_s(face_obj, props)
                area = props.Mass()
                
                target_face_data.append((face_obj, area))
            face_idx += 1
            explorer.Next()

        if not target_face_data:
            return positions

        # Distribute spheres proportionally to face area
        # Ensure each face gets a minimum of 3-5 spheres to guarantee coverage
        total_area = sum(area for _, area in target_face_data)
        num_faces = len(target_face_data)
        min_per_face = max(3, num_spheres // (num_faces * 2))  # At least 3 per face, scales with total count
        
        face_sphere_counts = [
            max(min_per_face, int(num_spheres * area / total_area))
            for _, area in target_face_data
        ]
        
        # Adjust if we've over-allocated
        total_allocated = sum(face_sphere_counts)
        if total_allocated > num_spheres:
            # Scale down proportionally, but keep minimum
            scale_factor = num_spheres / total_allocated
            face_sphere_counts = [
                max(min_per_face, int(count * scale_factor))
                for count in face_sphere_counts
            ]
        
        for (face, area), points_count in zip(target_face_data, face_sphere_counts):
            try:
                # Get face surface adaptor
                adaptor = BRepAdaptor_Surface(face)
                surface = adaptor.Surface()
                
                u_min = adaptor.FirstUParameter()
                u_max = adaptor.LastUParameter()
                v_min = adaptor.FirstVParameter()
                v_max = adaptor.LastVParameter()
                
                # Sample random points on the face surface
                for _ in range(points_count):
                    u = np.random.uniform(u_min, u_max)
                    v = np.random.uniform(v_min, v_max)
                    
                    try:
                        # Get 3D point on surface
                        point = surface.Value(u, v)
                        
                        # Get surface normal (for offsetting sphere center inward)
                        try:
                            d1u = surface.DN(u, v, 1, 0)
                            d1v = surface.DN(u, v, 0, 1)
                            normal_vec = d1u.Crossed(d1v)
                            normal_vec.Normalize()
                            
                            # Offset point slightly INWARD to prevent bleeding onto adjacent surfaces
                            # Use 50% of radius for balance: prevents overflow while keeping holes visible
                            offset_dist = -sphere_radius * 0.5
                            offset_point = gp_Pnt(
                                point.X() + normal_vec.X() * offset_dist,
                                point.Y() + normal_vec.Y() * offset_dist,
                                point.Z() + normal_vec.Z() * offset_dist
                            )
                        except:
                            # If normal fails, just use the point on surface
                            offset_point = point
                        
                        # Add size variation if enabled
                        if size_variation:
                            # Vary radius from 0.5x to 1.5x the base radius
                            radius_factor = np.random.uniform(0.5, 1.5)
                            actual_radius = sphere_radius * radius_factor
                        else:
                            actual_radius = sphere_radius
                        
                        # Store both position and radius
                        positions.append((offset_point, actual_radius))
                    except Exception as e:
                        # Skip this point if surface evaluation fails
                        pass
                    
            except Exception as e:
                # Skip faces that fail to parametrize
                pass

        return positions[:num_spheres]
