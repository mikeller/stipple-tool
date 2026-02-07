"""Mesh-based stippling engine using vertex displacement."""
from typing import List, Optional, Callable

import numpy as np
import trimesh
from scipy.spatial import cKDTree


class MeshStippleEngine:
    """Applies stippling to mesh surfaces by displacing vertices inward."""

    def __init__(self):
        self.stipple_size = 2.5  # mm
        self.stipple_depth = 1.5  # mm
        self.stipple_density = 0.70  # 0.0 to 1.0
        self.coverage_factor = 3.0
        self.max_total_points = 2500
        self.random_seed = 42

    def set_parameters(
        self,
        size: float,
        depth: float,
        density: float,
        coverage_factor: float = 3.0,
        max_total_points: int = 2500,
    ):
        self.stipple_size = max(0.1, size)
        self.stipple_depth = max(0.1, depth)
        self.stipple_density = max(0.01, min(1.0, density))
        self.coverage_factor = max(0.1, coverage_factor)
        self.max_total_points = max(100, min(20000, int(max_total_points)))

    def apply_stippling_to_mesh(
        self,
        mesh: trimesh.Trimesh,
        face_indices: List[int],
        pattern: str = "random",
        progress_callback: Optional[Callable[[int, int], None]] = None,
        status_callback: Optional[Callable[[str], None]] = None,
        cancel_callback: Optional[Callable[[], None]] = None,
    ) -> trimesh.Trimesh:
        """
        Apply stippling to selected faces by displacing nearby vertices.

        Args:
            mesh: Input mesh
            face_indices: Target face indices
            pattern: Currently unused (random sampling)

        Returns:
            Modified mesh
        """
        if mesh is None or mesh.is_empty or not face_indices:
            return mesh

        np.random.seed(self.random_seed)

        mesh_out = mesh.copy()
        vertices = mesh_out.vertices.copy()
        faces = mesh_out.faces
        vertex_normals = mesh_out.vertex_normals

        face_indices = np.array(face_indices, dtype=int)
        face_indices = face_indices[(face_indices >= 0) & (face_indices < len(faces))]
        if len(face_indices) == 0:
            return mesh_out

        total_area = mesh_out.area_faces[face_indices].sum()
        count = int((total_area / (self.stipple_size ** 2)) * self.stipple_density * self.coverage_factor)
        count = max(1, min(count, self.max_total_points))

        if status_callback:
            status_callback(f"Sampling {count} stipple points on mesh...")

        submesh = mesh_out.submesh([face_indices], append=True, repair=False)
        points, face_index = trimesh.sample.sample_surface(submesh, count)
        face_normals = submesh.face_normals[face_index]

        target_vertex_indices = np.unique(faces[face_indices].reshape(-1))
        target_positions = vertices[target_vertex_indices]
        tree = cKDTree(target_positions)

        radius = max(0.5, self.stipple_size * 0.6)
        sigma = max(0.2, radius * 0.5)

        displacement = np.zeros(len(vertices), dtype=np.float64)

        for i, point in enumerate(points):
            if cancel_callback:
                try:
                    cancel_callback()
                except RuntimeError:
                    if status_callback:
                        status_callback("Cancelling...")
                    return mesh

            idxs = tree.query_ball_point(point, radius)
            if not idxs:
                continue

            idxs_global = target_vertex_indices[idxs]
            pos = vertices[idxs_global]
            d = np.linalg.norm(pos - point, axis=1)
            weights = np.exp(-((d ** 2) / (2 * sigma ** 2)))
            delta = self.stipple_depth * weights

            for j, v_idx in enumerate(idxs_global):
                if delta[j] > displacement[v_idx]:
                    displacement[v_idx] = delta[j]

            if progress_callback and (i % max(1, count // 50) == 0):
                progress_callback(i + 1, count)

        vertices = vertices - (vertex_normals * displacement[:, None])
        mesh_out.vertices = vertices
        mesh_out.rezero()
        mesh_out.remove_degenerate_faces()
        mesh_out.remove_duplicate_faces()
        mesh_out.remove_unreferenced_vertices()

        if status_callback:
            status_callback("Mesh stippling complete")

        return mesh_out
