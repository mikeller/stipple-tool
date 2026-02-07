"""Mesh loader for OBJ/STL using trimesh."""
from pathlib import Path
from typing import Optional, Dict, List

import trimesh


class MeshLoader:
    """Loads and manages mesh format files (OBJ/STL)."""

    def __init__(self):
        self.file_path: Optional[Path] = None
        self.mesh: Optional[trimesh.Trimesh] = None

    def load(self, file_path: str) -> bool:
        """
        Load an OBJ or STL file.

        Args:
            file_path: Path to the .obj or .stl file

        Returns:
            True if successful, False otherwise
        """
        try:
            path = Path(file_path)

            if not path.exists():
                print(f"File not found: {path}")
                return False

            if path.suffix.lower() not in [".obj", ".gltf", ".glb", ".vrml", ".stl"]:
                print(f"Invalid file format. Expected .obj/.gltf/.glb/.vrml/.stl, got {path.suffix}")
                return False

            # For OBJ files, ensure MTL is loaded from the same directory
            if path.suffix.lower() == ".obj":
                mtl_path = path.with_suffix(".mtl")
                if mtl_path.exists():
                    print(f"Found associated MTL file: {mtl_path}")
                else:
                    print(f"No MTL file found for: {path}")

            loaded = trimesh.load(str(path), force="mesh", process=False)
            if isinstance(loaded, trimesh.Scene):
                mesh = trimesh.util.concatenate(loaded.dump())
            else:
                mesh = loaded

            if mesh is None or mesh.is_empty:
                print("Failed to load mesh or mesh is empty")
                return False

            self.mesh = mesh
            self.file_path = path

            print(f"Successfully loaded: {path}")
            print(f"Faces found: {len(mesh.faces)}")
            if hasattr(mesh.visual, "face_colors") and len(mesh.visual.face_colors) > 0:
                print(f"Face colors found: {len(mesh.visual.face_colors)}")
            return True

        except Exception as e:
            print(f"Error loading mesh file: {e}")
            return False

    def get_model(self) -> Optional[Dict]:
        """Get the loaded mesh model data."""
        if self.mesh is None:
            return None

        return {
            "file_path": self.file_path,
            "mesh": self.mesh,
            "faces": list(range(len(self.mesh.faces))),
            "type": "mesh",
        }

    def save_mesh(self, output_path: str, mesh: trimesh.Trimesh, output_format: Optional[str] = None) -> bool:
        """
        Save a mesh to STL, glTF, GLB, or VRML.

        Args:
            output_path: Path where to save
            mesh: Mesh to export
            output_format: "stl", "gltf", "glb", or "vrml"; if None, uses file extension

        Returns:
            True if successful, False otherwise
        """
        try:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)

            fmt = output_format or path.suffix.lower().lstrip(".")
            if fmt not in ["obj", "stl", "gltf", "glb", "vrml"]:
                print(f"Invalid output format: {fmt}")
                return False

            if path.suffix.lower() != f".{fmt}":
                path = path.with_suffix(f".{fmt}")

            mesh.export(str(path), file_type=fmt)
            print(f"Successfully saved: {path}")
            return True

        except Exception as e:
            print(f"Error saving mesh file: {e}")
            return False

    def get_model_info(self) -> dict:
        """Get information about the loaded mesh."""
        if self.mesh is None:
            return {}

        return {
            "file": str(self.file_path),
            "faces": len(self.mesh.faces),
            "type": "mesh",
        }
