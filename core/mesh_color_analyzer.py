"""Surface color analysis for mesh models using per-face colors and MTL materials."""
from typing import Dict, List
from collections import defaultdict
from pathlib import Path

import numpy as np
import trimesh


class MeshColorAnalyzer:
    """Analyzes and detects surface colors in mesh models."""

    def __init__(self):
        self.color_groups = defaultdict(list)
        self.material_info = {}

    def extract_colors_from_model(self, model: Dict) -> Dict[str, List[int]]:
        """
        Extract colors from mesh faces (from materials or face colors).

        Args:
            model: Model dictionary from MeshLoader

        Returns:
            Dictionary mapping color hex to lists of face indices
        """
        try:
            mesh: trimesh.Trimesh = model.get("mesh")
            file_path = model.get("file_path")
            
            if mesh is None or mesh.is_empty:
                return {"default": []}

            # First try to extract colors directly from mesh visual
            colors = self._extract_face_colors(mesh)
            
            # If no colors found and it's an OBJ file, try to load from MTL
            if colors is None or (isinstance(colors, np.ndarray) and len(colors) == 0):
                if file_path and Path(file_path).suffix.lower() == ".obj":
                    colors = self._extract_mtl_colors(file_path, mesh)
            
            # If still no colors, fall back to default
            if colors is None or (isinstance(colors, np.ndarray) and len(colors) == 0):
                self.color_groups = {"default": list(range(len(mesh.faces)))}
                print("⚠️  No colors found in mesh or MTL, using default (single color)")
                if file_path and Path(file_path).suffix.lower() == ".obj":
                    print("💡 Tip: If the original STEP file has multiple colors, use that instead")
                return dict(self.color_groups)

            colors_dict = defaultdict(list)
            for idx, color in enumerate(colors):
                if len(color) >= 3:
                    r, g, b = int(color[0]), int(color[1]), int(color[2])
                    color_key = f"#{r:02x}{g:02x}{b:02x}"
                else:
                    color_key = "default"
                colors_dict[color_key].append(idx)

            self.color_groups = colors_dict
            print(f"✓ Extracted {len(colors_dict)} color(s) from model")
            return dict(colors_dict)

        except Exception as e:
            print(f"Error extracting mesh colors: {e}")
            return {"default": list(range(len(model.get("faces", []))))}

    def _extract_face_colors(self, mesh: trimesh.Trimesh) -> np.ndarray:
        """Extract face colors from mesh visual."""
        try:
            if not hasattr(mesh, 'visual') or mesh.visual is None:
                return np.array([])

            # Try direct face colors
            if hasattr(mesh.visual, "face_colors") and mesh.visual.face_colors is not None:
                colors = np.asarray(mesh.visual.face_colors)
                if len(colors) > 0:
                    print(f"Found {len(colors)} face colors from mesh")
                    return colors

            # Try vertex colors as fallback
            if hasattr(mesh.visual, "vertex_colors") and mesh.visual.vertex_colors is not None:
                vertex_colors = np.asarray(mesh.visual.vertex_colors)
                if len(vertex_colors) > 0:
                    faces = np.asarray(mesh.faces)
                    colors = vertex_colors[faces[:, 0]]
                    print(f"Found {len(colors)} vertex colors from mesh")
                    return colors

            return np.array([])

        except Exception as e:
            print(f"Error extracting face colors: {e}")
            return np.array([])

    def _extract_mtl_colors(self, file_path: str, mesh: trimesh.Trimesh) -> np.ndarray:
        """Extract colors from MTL file for OBJ."""
        try:
            obj_path = Path(file_path)
            mtl_path = obj_path.with_suffix(".mtl")

            if not mtl_path.exists():
                print(f"MTL file not found: {mtl_path}")
                return np.array([])

            print(f"Parsing MTL file: {mtl_path}")

            # Parse MTL file to extract material colors and track usemtl statements
            material_colors = {}
            mtl_order = []  # Track order of materials in MTL
            current_material = None

            with open(mtl_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue

                    parts = line.split()
                    if not parts:
                        continue

                    cmd = parts[0]

                    if cmd == "newmtl":
                        current_material = parts[1] if len(parts) > 1 else "default"
                        if current_material not in mtl_order:
                            mtl_order.append(current_material)
                    elif cmd == "Kd" and current_material:
                        # Diffuse color (RGB)
                        if len(parts) >= 4:
                            r = int(float(parts[1]) * 255)
                            g = int(float(parts[2]) * 255)
                            b = int(float(parts[3]) * 255)
                            color = (r, g, b, 255)
                            material_colors[current_material] = color
                            hex_color = f"#{r:02x}{g:02x}{b:02x}"
                            print(f"  Material '{current_material}': {hex_color}")

            if not material_colors:
                print("❌ No material colors found in MTL")
                return np.array([])

            # Parse OBJ to map materials to faces
            material_face_map = self._parse_obj_materials(file_path)
            
            if material_face_map:
                # Build color array using material assignments
                colors = np.zeros((len(mesh.faces), 4), dtype=np.uint8)
                for mat_name, face_indices in material_face_map.items():
                    if mat_name in material_colors:
                        for face_idx in face_indices:
                            colors[face_idx] = material_colors[mat_name]
                
                unique_materials = len(material_face_map)
                print(f"✓ Mapped {unique_materials} material(s) to {len(mesh.faces)} faces")
                return colors
            else:
                # Fallback: apply first material color to all faces
                if material_colors:
                    first_color = list(material_colors.values())[0]
                    colors = np.tile(first_color, (len(mesh.faces), 1))
                    print(f"⚠️  Could not map materials to faces, applying single material to all {len(mesh.faces)} faces")
                    return colors

            return np.array([])

        except Exception as e:
            print(f"Error extracting MTL colors: {e}")
            return np.array([])

    def _parse_obj_materials(self, file_path: str) -> Dict[str, List[int]]:
        """Parse OBJ file to map materials to face indices."""
        try:
            with open(file_path, 'r') as f:
                material_faces = {}
                current_material = None
                face_index = 0

                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue

                    parts = line.split()
                    if not parts:
                        continue

                    cmd = parts[0]

                    if cmd == "usemtl":
                        current_material = parts[1] if len(parts) > 1 else "default"
                        if current_material not in material_faces:
                            material_faces[current_material] = []
                    elif cmd == "f":
                        if current_material is not None:
                            material_faces[current_material].append(face_index)
                        face_index += 1

                return material_faces if material_faces else {}

        except Exception as e:
            print(f"Error parsing OBJ materials: {e}")
            return {}

    def get_faces_by_color(self, color: str) -> List[int]:
        """Get all face indices with a specific color."""
        return self.color_groups.get(color, [])

    def get_all_color_groups(self) -> Dict[str, List[int]]:
        """Get all color groups."""
        return dict(self.color_groups)

