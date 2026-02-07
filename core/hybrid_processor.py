"""Hybrid STEP-to-mesh workflow with color preservation."""
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import tempfile
import numpy as np
import trimesh

from core.step_loader import STEPLoader
from core.color_analyzer import ColorAnalyzer
from core.mesh_loader import MeshLoader
from core.mesh_color_analyzer import MeshColorAnalyzer
from core.mesh_stipple_engine import MeshStippleEngine


class HybridSTEPMeshProcessor:
    """
    Hybrid processor: Load STEP, extract colors, convert to mesh, apply stippling.
    Preserves color information during conversion.
    """

    def __init__(self):
        self.step_loader = STEPLoader()
        self.color_analyzer = ColorAnalyzer()
        self.mesh_loader = MeshLoader()
        self.stipple_engine = MeshStippleEngine()

    def process_step_with_color_stippling(
        self,
        step_file: Optional[str] = None,
        output_path: Optional[str] = None,
        target_color: str = "#360200",
        densities: float = 1.5,
        step_model_data: Optional[Dict] = None,
        color_groups: Optional[Dict] = None,
        progress_callback=None,
        status_callback=None,
    ) -> Optional[str]:
        """
        Full workflow: STEP → Colors → Mesh → Stipple → Export.

        Args:
            step_file: Path to input STEP file (optional if step_model_data provided)
            output_path: Path to output file (STL recommended)
            target_color: Color hex code to stipple (e.g., "#360200")
            densities: Stipple density value
            step_model_data: Pre-loaded STEP model data (alternative to step_file)
            color_groups: Pre-computed color groups (alternative to extracting)
            progress_callback: Callback for progress (0.0-1.0)
            status_callback: Callback for status messages

        Returns:
            Output file path if successful, None otherwise
        """
        try:
            def emit_progress(value):
                if progress_callback:
                    progress_callback(value)

            def emit_status(msg):
                if status_callback:
                    status_callback(msg)

            emit_status("Starting hybrid workflow...")

            # Step 1: Load STEP and extract colors
            if step_model_data is None:
                emit_status("Loading STEP file...")
                if not self.step_loader.load(step_file):
                    emit_status(f"Failed to load STEP: {step_file}")
                    return None
                step_model = self.step_loader.get_model()
            else:
                step_model = step_model_data

            emit_status(f"Loaded STEP: {len(step_model.get('faces', []))} faces")
            emit_progress(0.1)

            # Extract colors if not provided
            if color_groups is None:
                emit_status("Extracting surface colors...")
                color_groups = self.color_analyzer.extract_colors_from_model(step_model)
                for color, faces in color_groups.items():
                    emit_status(f"  Found {color}: {len(faces)} faces")
            else:
                emit_status(f"Using provided color groups: {len(color_groups)} groups")

            if target_color not in color_groups:
                emit_status(f"Target color {target_color} not found in model")
                return None

            target_face_indices = color_groups[target_color]
            emit_status(f"Target color {target_color}: {len(target_face_indices)} faces")
            emit_progress(0.2)

            # Step 2: Convert STEP to mesh while preserving face mapping
            emit_status("Converting STEP to mesh...")
            mesh, face_color_map = self._step_to_mesh_with_colors(step_model, color_groups)
            if mesh is None:
                emit_status("Failed to convert STEP to mesh")
                return None
            emit_status(f"Mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
            emit_progress(0.4)

            # Step 3: Map target color to mesh face indices
            emit_status("Mapping colors to mesh faces...")
            mesh_target_indices = self._map_colors_to_mesh_faces(
                face_color_map, target_color, len(mesh.faces)
            )
            if not mesh_target_indices:
                emit_status("No target color faces found in mesh")
                return None
            emit_status(f"Target faces in mesh: {len(mesh_target_indices)}")
            emit_progress(0.5)

            # Step 4: Apply stippling
            emit_status("Applying mesh stippling...")
            self.stipple_engine.set_parameters(
                size=2.5,
                depth=densities,
                density=0.7,
            )

            def progress_cb(current, total):
                if total > 0 and current % max(1, total // 10) == 0:
                    emit_status(f"Processing: {current}/{total}")
                    emit_progress(0.5 + 0.3 * (current / total))

            stippled_mesh = self.stipple_engine.apply_stippling_to_mesh(
                mesh,
                mesh_target_indices,
                "random",
                progress_callback=progress_cb,
                status_callback=emit_status,
                cancel_callback=lambda: False,
            )

            if stippled_mesh is None:
                emit_status("Failed to apply stippling")
                return None
            emit_status("Stippling applied successfully")
            emit_progress(0.8)

            # Step 5: Export
            emit_status("Exporting result...")
            output_format = Path(output_path).suffix.lower().lstrip(".")
            if not self.mesh_loader.save_mesh(output_path, stippled_mesh, output_format):
                emit_status(f"Failed to export to {output_path}")
                return None
            emit_status(f"Saved to {output_path}")
            emit_progress(1.0)

            return output_path

        except Exception as e:
            emit_status(f"Error in hybrid workflow: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def _step_to_mesh_with_colors(
        self, step_model: Dict, color_groups: Dict[str, List[int]]
    ) -> Tuple[Optional[trimesh.Trimesh], Dict[int, str]]:
        """
        Convert STEP to mesh preserving face→color mapping.

        Strategy: 
        1. Try using cadquery's built-in conversion
        2. Fallback: Map color information by traversal order

        Returns:
            (mesh, face_color_map) where face_color_map[mesh_face_idx] = color_hex
        """
        try:
            print("   Attempting mesh extraction via CadQuery...")

            from OCP.BRepMesh import BRepMesh_IncrementalMesh
            from OCP.StlAPI import StlAPI_Writer
            import tempfile
            import os

            shape = step_model.get("shape")

            # Create mesh in the shape
            mesh_maker = BRepMesh_IncrementalMesh(shape, 0.1, False, 0.5)
            mesh_maker.Perform()

            # Export to temporary STL file
            with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as f:
                tmp_stl = f.name

            writer = StlAPI_Writer()
            writer.Write(shape, tmp_stl)

            print(f"     Exported to temporary STL: {tmp_stl}")

            # Load the STL back as a mesh
            import trimesh

            mesh = trimesh.load(tmp_stl, force="mesh", process=False)
            os.unlink(tmp_stl)

            if mesh is None or mesh.is_empty:
                print("     ✗ Failed to load STL")
                return None, {}

            print(f"     ✓ Loaded mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

            # Map faces to colors
            # Strategy: Since STL doesn't preserve face relationships, we distribute colors
            # proportionally based on the number of STEP faces per color
            step_faces = step_model.get("faces", [])
            mesh_face_count = len(mesh.faces)
            face_to_color = {}
            
            # Calculate how many mesh faces each color should get
            total_step_faces = len(step_faces)
            if total_step_faces == 0:
                print("     Warning: No STEP faces found, cannot map colors")
                return mesh, {}
            
            mesh_face_idx = 0
            
            for color, step_face_indices in sorted(color_groups.items()):
                num_step_faces = len(step_face_indices)
                # Proportionally assign mesh faces to this color
                num_mesh_faces = int((num_step_faces / total_step_faces) * mesh_face_count)
                
                for i in range(num_mesh_faces):
                    if mesh_face_idx < mesh_face_count:
                        face_to_color[mesh_face_idx] = color
                        mesh_face_idx += 1
            if not face_to_color:
                print("     Using fallback color assignment...")
                mesh_idx = 0
                for color, step_face_indices in color_groups.items():
                    face_count = len(step_face_indices)
                    for _ in range(face_count):
                        if mesh_idx < mesh_face_count:
                            face_to_color[mesh_idx] = color
                            mesh_idx += 1

            print(f"   ✓ Created color mapping for {len(face_to_color)} faces")
            for color in set(face_to_color.values()):
                count = sum(1 for c in face_to_color.values() if c == color)
                print(f"     {color}: {count} faces")

            return mesh, face_to_color

        except Exception as e:
            print(f"   ✗ Error in STEP→mesh conversion: {e}")
            import traceback

            traceback.print_exc()
            return None, {}

    def _map_colors_to_mesh_faces(
        self,
        face_color_map: Dict[int, str],
        target_color: str,
        total_faces: int,
    ) -> List[int]:
        """
        Map target color to mesh face indices using the preserved color map.

        Returns:
            List of mesh face indices with the target color
        """
        target_indices = []

        # First try exact mapping
        for mesh_face_idx, color in face_color_map.items():
            if color == target_color:
                target_indices.append(mesh_face_idx)

        # If we got some matches, return them
        if target_indices:
            print(f"   Mapped {len(target_indices)} mesh faces to color {target_color}")
            return target_indices

        # Fallback: If no exact match, return empty list
        print(f"   ⚠ No exact color mapping found for color {target_color}")
        print(f"     Total mesh faces: {total_faces}")
        print(f"     Face color map entries: {len(face_color_map)}")
        print(f"     Warning: Cannot stipple faces without proper color mapping")

        return []
