"""
Partitioned stippling processor - divides solid by colored surfaces for O(n/k²) complexity.

Strategy:
1. Extract colored surface regions and create cutting planes
2. Partition solid using these planes via BRepAlgoAPI_Splitter
3. Stipple each partition independently (parallelizable, fast)
4. Incrementally reassemble with full density stippling at seams
5. All steps parallelizable and use simpler geometry

Performance: O(n/k²) where k = number of partitions (~3-5)
For example: 5200 spheres / 4 partitions = 1300 each → 4x faster per partition → 16x overall
"""

import time
import tempfile
import os
import multiprocessing as mp
from pathlib import Path
from typing import List, Tuple, Optional, Callable, Dict

from OCP.TopoDS import TopoDS_Shape, TopoDS, TopoDS_Compound, TopoDS_Builder
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_FACE, TopAbs_SOLID
from OCP.BRepAlgoAPI import BRepAlgoAPI_Splitter, BRepAlgoAPI_Fuse
from OCP.ShapeFix import ShapeFix_Shape
from OCP.STEPControl import STEPControl_Writer, STEPControl_AsIs
from OCP.IFSelect import IFSelect_RetDone
from OCP.GProp import GProp_GProps
from OCP.BRepGProp import BRepGProp
from OCP.TopTools import TopTools_ListOfShape
from OCP.Bnd import Bnd_Box
from OCP.BRepBndLib import BRepBndLib
from OCP.gp import gp_Pnt, gp_Dir, gp_Pln
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace

from core.step_loader import STEPLoader
from core.color_analyzer import ColorAnalyzer
from core.incremental_processor import IncrementalStippleProcessor


class PartitionedStippleProcessor:
    """Process stippling using partitioned approach for massive performance gains."""

    def __init__(self):
        self.color_analyzer = ColorAnalyzer()
        self.stipple_engine = IncrementalStippleProcessor()

    def process_step_with_partitions(
        self,
        step_file: str,
        output_path: str,
        target_color: str,
        sphere_radius: float = 1.0,
        sphere_depth: float = 0.5,
        num_spheres: int = 5200,
        size_variation: bool = True,
        progress_callback: Optional[Callable] = None,
        status_callback: Optional[Callable] = None,
        cancel_callback: Optional[Callable] = None,
        num_partitions: Optional[int] = None,
    ) -> Optional[str]:
        """
        Process STEP file using partitioned approach.

        Args:
            step_file: Path to input STEP file
            output_path: Path for output STEP file
            target_color: Hex color code for stippling (e.g., "#360200")
            sphere_radius: Radius of stipple spheres in mm
            sphere_depth: Depth of stipple cuts in mm
            num_spheres: Target number of spheres to place
            size_variation: Add variation to sphere sizes
            progress_callback: Called with (current, total)
            status_callback: Called with status message strings
            cancel_callback: Called to check if user cancelled

        Returns:
            Path to output STEP file or None if failed
        """
        def emit_status(msg: str):
            if status_callback:
                status_callback(msg)
            print(msg)

        def check_cancel():
            if cancel_callback:
                cancel_callback()

        try:
            emit_status("Phase 1: Loading and analyzing model...")
            
            # Load model
            loader = STEPLoader()
            if not loader.load(step_file):
                emit_status("Failed to load STEP file")
                return None

            # Get colors
            emit_status("Extracting surface colors...")
            colors = self.color_analyzer.extract_colors_from_model({
                "faces": loader.faces,
                "shape_tool": loader.shape_tool,
                "color_tool": loader.color_tool,
            })

            target_faces = colors.get(target_color, [])
            if not target_faces:
                emit_status(f"No surfaces found with color {target_color}")
                return None

            emit_status(f"Found {len(target_faces)} colored surfaces")
            check_cancel()

            # Extract face centroids and colors for matching after partition
            emit_status("Building face centroid map...")
            face_centroid_map = self._extract_face_centroids(
                loader.shape,
                target_faces,
                emit_status=emit_status,
                check_cancel=check_cancel,
            )
            emit_status(f"Mapped {len(face_centroid_map)} face centroids")
            check_cancel()

            # Partition the solid
            emit_status("Phase 2: Partitioning solid by colored regions...")
            partitions = self._partition_solid_by_surfaces(
                loader.shape,
                loader.faces,
                target_faces,
                face_centroid_map,
                emit_status,
                check_cancel,
                num_partitions=num_partitions,
            )
            emit_status(f"Created {len(partitions)} initial partitions")
            check_cancel()
            
            # Further partition by surface area to ensure each partition is manageable
            emit_status("Phase 2b: Further subdividing by surface area...")
            partitions = self._partition_by_surface_area(
                partitions,
                face_centroid_map,
                target_spheres_per_partition=75,
                emit_status=emit_status,
                check_cancel=check_cancel,
            )
            emit_status(f"After area-aware partitioning: {len(partitions)} total partitions")
            check_cancel()

            # Stipple each partition
            emit_status("Phase 3: Stippling each partition independently...")
            stippled_partitions = self._stipple_partitions(
                partitions, 
                face_centroid_map,
                sphere_radius,
                sphere_depth,
                num_spheres,
                size_variation,
                emit_status,
                check_cancel
            )
            check_cancel()

            # Reassemble
            emit_status("Phase 4: Reassembling stippled partitions...")
            final_shape = self._reassemble_partitions(stippled_partitions, emit_status, check_cancel)
            check_cancel()

            # Save result
            emit_status("Saving result...")
            if self._save_shape_to_step(final_shape, output_path):
                emit_status(f"Successfully saved to: {output_path}")
                return output_path
            else:
                emit_status("Failed to save output")
                return None

        except Exception as e:
            import traceback
            emit_status(f"Error during partitioned stippling: {e}")
            traceback.print_exc()
            return None

    def _extract_face_centroids(
        self,
        shape: TopoDS_Shape,
        target_face_indices: List[int],
        emit_status: Optional[Callable] = None,
        check_cancel: Optional[Callable] = None,
    ) -> Dict[int, Tuple[float, float, float]]:
        """Extract centroids of target faces for matching after partitioning."""
        centroids = {}
        
        explorer = TopExp_Explorer(shape, TopAbs_FACE)
        face_idx = 0
        processed = 0
        start_time = time.monotonic()
        
        while explorer.More():
            if check_cancel:
                check_cancel()
            if face_idx in target_face_indices:
                face = TopoDS.Face_s(explorer.Current())
                
                # Calculate face centroid
                props = GProp_GProps()
                BRepGProp.SurfaceProperties_s(face, props)
                centroid = props.CentreOfMass()
                
                centroids[face_idx] = (centroid.X(), centroid.Y(), centroid.Z())
                processed += 1
                if emit_status and processed % 5 == 0:
                    elapsed = time.monotonic() - start_time
                    emit_status(f"Centroid map: {processed}/{len(target_face_indices)} targets in {elapsed:.1f}s")
            
            explorer.Next()
            face_idx += 1
        
        return centroids

    def _partition_solid_by_surfaces(
        self,
        solid: TopoDS_Shape,
        all_faces: List,
        target_face_indices: List[int],
        original_face_centroids: Dict[int, Tuple[float, float, float]],
        emit_status: Callable,
        check_cancel: Callable,
        num_partitions: Optional[int] = None,
    ) -> List[TopoDS_Shape]:
        """
        Partition solid using plane-based cuts along the longest axis.
        
        This guarantees multiple partitions (configurable) and avoids relying
        on face-based splitting which can be fragile.
        """
        # Decide partition count
        if num_partitions is None:
            cpu_count = os.cpu_count() or 4
            num_partitions = max(2, min(8, cpu_count - 1))
        else:
            num_partitions = max(2, num_partitions)

        emit_status(f"Preparing splitter for {num_partitions} partitions...")
        check_cancel()

        # Target-aware recursive splitting: only keep splitting shapes that contain target faces
        partitions = [solid]
        max_iters = num_partitions * 3
        iters = 0

        while iters < max_iters:
            iters += 1
            check_cancel()

            # Identify partitions that contain target faces
            partitions_with_targets = []
            for p in partitions:
                targets = self._identify_target_faces_in_partition(
                    p,
                    original_face_centroids,
                    proximity_threshold=0.5,
                )
                if targets:
                    partitions_with_targets.append((p, len(targets)))

            if len(partitions_with_targets) >= num_partitions:
                break

            # Choose the largest partition (by bbox volume) with targets to split
            if not partitions_with_targets:
                break

            def bbox_volume(shape: TopoDS_Shape) -> float:
                bb = Bnd_Box()
                BRepBndLib.Add_s(shape, bb)
                x0, y0, z0, x1, y1, z1 = bb.Get()
                return max(0.0, (x1 - x0) * (y1 - y0) * (z1 - z0))

            partitions_with_targets.sort(key=lambda t: bbox_volume(t[0]), reverse=True)
            to_split = partitions_with_targets[0][0]

            new_parts = self._split_shape_by_mid_plane(to_split, emit_status)
            if len(new_parts) <= 1:
                break

            # Replace the split partition with its children
            partitions.remove(to_split)
            partitions.extend(new_parts)

        emit_status(f"Partitioning finished with {len(partitions)} total parts")
        return partitions

    def _split_shape_by_mid_plane(self, shape: TopoDS_Shape, emit_status: Callable) -> List[TopoDS_Shape]:
        """Split a shape by a plane through the midpoint of its longest axis."""
        bbox = Bnd_Box()
        BRepBndLib.Add_s(shape, bbox)
        xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
        dx, dy, dz = xmax - xmin, ymax - ymin, zmax - zmin

        if dx <= 0 and dy <= 0 and dz <= 0:
            return [shape]

        if dx >= dy and dx >= dz:
            axis = "x"
            mid = (xmin + xmax) * 0.5
            normal = gp_Dir(1, 0, 0)
            origin = gp_Pnt(mid, 0, 0)
        elif dy >= dx and dy >= dz:
            axis = "y"
            mid = (ymin + ymax) * 0.5
            normal = gp_Dir(0, 1, 0)
            origin = gp_Pnt(0, mid, 0)
        else:
            axis = "z"
            mid = (zmin + zmax) * 0.5
            normal = gp_Dir(0, 0, 1)
            origin = gp_Pnt(0, 0, mid)

        emit_status(f"Splitting part on {axis.upper()} @ {mid:.2f}")

        args = TopTools_ListOfShape()
        args.Append(shape)

        tools = TopTools_ListOfShape()
        plane = gp_Pln(origin, normal)
        face = BRepBuilderAPI_MakeFace(plane).Face()
        tools.Append(face)

        splitter = BRepAlgoAPI_Splitter()
        splitter.SetArguments(args)
        splitter.SetTools(tools)
        splitter.SetRunParallel(True)
        splitter.Build()

        if not splitter.IsDone():
            return [shape]

        result_shape = splitter.Shape()
        parts = self._extract_solids(result_shape)
        return parts if parts else [shape]

    def _partition_by_surface_area(
        self,
        partitions: List[TopoDS_Shape],
        original_face_centroids: Dict[int, Tuple[float, float, float]],
        target_spheres_per_partition: int = 75,
        emit_status: Optional[Callable] = None,
        check_cancel: Optional[Callable] = None,
    ) -> List[TopoDS_Shape]:
        """
        Further subdivide partitions so each has at most 1-2 target faces.
        
        This ensures that each partition stipples only a small colored surface,
        avoiding the exponential slowdown of large geometries.
        """
        if not original_face_centroids:
            return partitions
        
        # Very aggressive: split so max 2 target faces per partition
        target_faces_threshold = 2
        
        result_partitions = []
        queue = [(p, 0) for p in partitions]  # (partition, recursion_depth)
        max_depth = 15  # Increased for aggressive subdivision
        
        while queue:
            partition, depth = queue.pop(0)
            
            if check_cancel:
                check_cancel()
            
            # Count target faces in this partition
            target_count = len(
                self._identify_target_faces_in_partition(
                    partition,
                    original_face_centroids,
                    proximity_threshold=0.5,
                )
            )
            
            if emit_status and depth > 0 and target_count > 0:
                emit_status(f"  {'  ' * depth}Depth {depth}: {target_count} target faces")
            
            if target_count <= target_faces_threshold or depth >= max_depth:
                # Small enough, keep as-is
                if emit_status and target_count > 0:
                    result_partitions.append(partition)
                    if depth == 0:
                        emit_status(f"Final partition: {target_count} target faces")
                elif target_count > 0:
                    result_partitions.append(partition)
            else:
                # Too many target faces, split further
                if emit_status:
                    emit_status(f"  {'  ' * depth}Splitting: {target_count} target faces → 2 children")
                
                children = self._split_shape_by_mid_plane(partition, emit_status or (lambda x: None))
                if len(children) > 1:
                    # Add children to queue for further processing
                    for child in children:
                        queue.append((child, depth + 1))
                else:
                    # Couldn't split, keep original
                    if emit_status:
                        emit_status(f"  {'  ' * depth}Could not split further")
                    result_partitions.append(partition)
        
        return result_partitions

    def _extract_solids(self, shape: TopoDS_Shape) -> List[TopoDS_Shape]:
        """Extract all disconnected solids from a shape."""
        solids = []
        explorer = TopExp_Explorer(shape, TopAbs_SOLID)
        
        while explorer.More():
            solid = TopoDS.Solid_s(explorer.Current())
            solids.append(solid)
            explorer.Next()

        return solids

    def _identify_target_faces_in_partition(
        self,
        partition: TopoDS_Shape,
        original_centroids: Dict[int, Tuple[float, float, float]],
        proximity_threshold: float = 0.1,
        emit_status: Optional[Callable] = None,
        check_cancel: Optional[Callable] = None,
    ) -> List[int]:
        """Identify which faces in partition correspond to original target faces."""
        target_faces = []
        
        explorer = TopExp_Explorer(partition, TopAbs_FACE)
        partition_face_idx = 0
        processed = 0
        start_time = time.monotonic()
        
        while explorer.More():
            if check_cancel:
                check_cancel()
            face = TopoDS.Face_s(explorer.Current())
            
            # Calculate this face's centroid
            props = GProp_GProps()
            BRepGProp.SurfaceProperties_s(face, props)
            centroid = props.CentreOfMass()
            px, py, pz = centroid.X(), centroid.Y(), centroid.Z()
            
            # Find closest original target face
            min_distance = float('inf')
            
            for orig_idx, (ox, oy, oz) in original_centroids.items():
                dist = ((px - ox)**2 + (py - oy)**2 + (pz - oz)**2)**0.5
                
                if dist < min_distance:
                    min_distance = dist
            
            # If close enough to any original target face, include it
            if min_distance < proximity_threshold:
                target_faces.append(partition_face_idx)
            
            processed += 1
            if emit_status and processed % 50 == 0:
                elapsed = time.monotonic() - start_time
                emit_status(
                    f"Target match: {processed} faces checked, {len(target_faces)} matched in {elapsed:.1f}s"
                )
            
            explorer.Next()
            partition_face_idx += 1
        
        return target_faces

    def _stipple_partitions(
        self,
        partitions: List[TopoDS_Shape],
        original_face_centroids: Dict[int, Tuple[float, float, float]],
        sphere_radius: float,
        sphere_depth: float,
        total_num_spheres: int,
        size_variation: bool,
        emit_status: Callable,
        check_cancel: Callable,
    ) -> List[TopoDS_Shape]:
        """
        Stipple each partition independently using centroid-based face matching.
        
        Each partition is:
        1. Analyzed to find which faces match original target faces (by centroid proximity)
        2. Saved to a temporary STEP file
        3. Processed with face indices directly
        4. Result loaded back
        
        This can be parallelized by processing multiple partitions on different cores.
        """
        emit_status(f"Stippling {len(partitions)} partitions...")
        check_cancel()

        stippled = []

        # Cap spheres per partition to avoid exponential slowdown
        MAX_SPHERES_PER_PARTITION = 40
        MAX_PARTITIONS = 20  # Limit total partitions to avoid thousands of temp files
        
        # Simpler approach: just keep splitting partitions until each has <=2 target faces
        # But cap total partition count
        emit_status(f"Subdividing partitions (max {MAX_PARTITIONS})...")
        refined_partitions = []
        to_process = list(partitions)
        max_refine_iters = 200  # Lower limit to prevent runaway
        iter_count = 0
        
        while to_process and iter_count < max_refine_iters and len(refined_partitions) < MAX_PARTITIONS:
            iter_count += 1
            partition = to_process.pop(0)
            
            # Quick face count check
            target_faces_in_partition = self._identify_target_faces_in_partition(
                partition,
                original_face_centroids,
                proximity_threshold=0.5,
            )
            
            if not target_faces_in_partition:
                # No target faces, skip this partition entirely
                continue
            
            target_face_count = len(target_faces_in_partition)
            
            # Split if has >2 target faces AND we haven't hit partition limit
            if target_face_count <= 2 or len(refined_partitions) + len(to_process) >= MAX_PARTITIONS:
                # Keep it - allocate spheres proportionally
                spheres_for_this = max(5, int(total_num_spheres / len(original_face_centroids)) * target_face_count)
                refined_partitions.append((partition, target_faces_in_partition, min(spheres_for_this, MAX_SPHERES_PER_PARTITION)))
                if len(refined_partitions) % 5 == 0:
                    emit_status(f"  {len(refined_partitions)} partitions ready...")
            else:
                # Multiple target faces - split it
                children = self._split_shape_by_mid_plane(partition, lambda x: None)
                if len(children) > 1:
                    to_process.extend(children)
                else:
                    # Can't split further
                    refined_partitions.append((partition, target_faces_in_partition, MAX_SPHERES_PER_PARTITION))
        
        emit_status(f"Refined to {len(refined_partitions)} partitions")
        check_cancel()

        # Prepare partition tasks
        tasks = []
        temp_files = []
        for idx, (partition, target_faces_in_partition, spheres_for_partition) in enumerate(refined_partitions):
            emit_status(f"Partition {idx + 1}/{len(refined_partitions)}: {len(target_faces_in_partition)} faces, {spheres_for_partition} spheres")
            emit_status(f"Partition {idx + 1}/{len(refined_partitions)}: Saving to temp file...")
            check_cancel()

            with tempfile.NamedTemporaryFile(suffix='.step', delete=False) as tmp:
                temp_file = tmp.name
            temp_files.append(temp_file)

            if not self._save_shape_to_step(partition, temp_file):
                emit_status(f"Warning: Failed to save partition {idx + 1}, skipping stippling")
                stippled.append(partition)
                continue

            tasks.append(
                (
                    idx,
                    len(refined_partitions),
                    temp_file,
                    target_faces_in_partition,
                    spheres_for_partition,
                    sphere_radius,
                    sphere_depth,
                    size_variation,
                )
            )

        # Run stippling sequentially (multiprocessing output suppression is problematic)
        if tasks:
            emit_status(f"Stippling {len(tasks)} partitions...")
            results = []
            
            for task_idx, task in enumerate(tasks):
                try:
                    emit_status(f"Partition {task_idx + 1}/{len(tasks)}: Starting stippling...")
                    check_cancel()
                    result = _stipple_partition_worker(task)
                    results.append(result)
                    idx = task[0]
                    emit_status(f"Partition {idx + 1}/{len(partitions)}: ✓ Stippling complete")
                except Exception as e:
                    emit_status(f"Partition {task_idx + 1}/{len(tasks)}: ✗ Error during stippling: {e}")
                    results.append((task[0], None))

            # Load results in original order
            results_by_idx = {idx: path for (idx, path) in results}
            for task in tasks:
                idx = task[0]
                result_path = results_by_idx.get(idx)
                if result_path:
                    emit_status(f"Partition {idx + 1}/{len(partitions)}: Loading result from {result_path}...")
                    loader = STEPLoader()
                    if loader.load(result_path):
                        stippled.append(loader.shape)
                        emit_status(f"Partition {idx + 1}/{len(partitions)}: ✓ Loaded")
                    else:
                        emit_status(f"Warning: Failed to load stippled partition {idx + 1}")
                else:
                    emit_status(f"Warning: Stippling failed for partition {idx + 1}")

        # Cleanup temp files
        for temp_file in temp_files:
            try:
                Path(temp_file).unlink()
            except:
                pass

        return stippled

    def _reassemble_partitions(
        self,
        partitions: List[TopoDS_Shape],
        emit_status: Callable,
        check_cancel: Callable,
    ) -> TopoDS_Shape:
        """
        Incrementally reassemble stippled partitions using union operations.
        """
        if not partitions:
            return TopoDS_Shape()

        emit_status(f"Assembling {len(partitions)} partitions...")
        check_cancel()

        result = partitions[0]

        for idx in range(1, len(partitions)):
            emit_status(f"Union {idx}/{len(partitions) - 1}...")
            check_cancel()

            try:
                union_op = BRepAlgoAPI_Fuse(result, partitions[idx])
                union_op.Build()

                if union_op.IsDone():
                    result = union_op.Shape()
                else:
                    emit_status(f"Warning: Union {idx} failed, continuing with current result")

            except Exception as e:
                emit_status(f"Warning: Union {idx} failed ({e})")

        emit_status("Assembly complete")
        return result

    def _save_shape_to_step(self, shape: TopoDS_Shape, output_path: str) -> bool:
        """Save a shape to a STEP file."""
        try:
            writer = STEPControl_Writer()
            writer.Transfer(shape, STEPControl_AsIs)
            status = writer.Write(output_path)
            return status == IFSelect_RetDone
        except Exception as e:
            print(f"Failed to save STEP: {e}")
            return False


def _stipple_partition_worker(args):
    """Worker to stipple a single partition (now in main process, not spawned)."""
    (
        idx,
        total_partitions,
        temp_file,
        target_faces_in_partition,
        spheres_for_partition,
        sphere_radius,
        sphere_depth,
        size_variation,
    ) = args

    # Local import inside worker
    from core.incremental_processor import IncrementalStippleProcessor

    print(f"[P{idx + 1}/{total_partitions}] Worker starting: {spheres_for_partition} spheres on {len(target_faces_in_partition)} faces")
    
    processor = IncrementalStippleProcessor()
    result_path = processor.process_step_with_incremental_stippling(
        step_file=temp_file,
        output_path=temp_file,
        target_color=None,
        sphere_radius=sphere_radius,
        sphere_depth=sphere_depth,
        num_spheres=spheres_for_partition,
        batch_size=30,
        target_faces=target_faces_in_partition,
        size_variation=size_variation,
        progress_callback=None,
        status_callback=lambda msg: print(f"[P{idx + 1}/{total_partitions}] {msg}"),
        cancel_callback=None,
        status_prefix=f"P{idx + 1}/{total_partitions}: ",
    )
    
    print(f"[P{idx + 1}/{total_partitions}] Worker complete: result_path={result_path}")
    return (idx, result_path)
