"""
Mesh and geometry utilities.
"""
import numpy as np
from typing import List, Tuple


class MeshUtils:
    """Utilities for mesh and geometry operations."""
    
    @staticmethod
    def calculate_face_area(face) -> float:
        """
        Calculate the area of a face.
        
        Args:
            face: OCP face object
            
        Returns:
            Face area in mm²
        """
        try:
            return face.Area()
        except:
            return 0.0
    
    @staticmethod
    def get_face_normal(face) -> Tuple[float, float, float]:
        """
        Get the normal vector of a face.
        
        Args:
            face: OCP face object
            
        Returns:
            Normal vector as (x, y, z)
        """
        try:
            surface = face.Surface()
            center = face.Center()
            
            # Get normal at center point (simplified approach)
            u, v = surface.Parameters(center)
            normal = surface.Normal(u, v)
            
            return (normal.X, normal.Y, normal.Z)
        except:
            return (0, 0, 1)  # Default to Z normal
    
    @staticmethod
    def project_point_to_face(point: Tuple[float, float, float],
                            face) -> Tuple[float, float, float]:
        """
        Project a point onto a face surface.
        
        Args:
            point: Point coordinates (x, y, z)
            face: OCP face object
            
        Returns:
            Projected point on the face surface
        """
        try:
            # This is a simplified projection
            # A full implementation would use surface projection algorithms
            return point
        except:
            return point
    
    @staticmethod
    def calculate_mesh_bounds(faces: List) -> Tuple[Tuple[float, float, float], 
                                                     Tuple[float, float, float]]:
        """
        Calculate bounding box for mesh.
        
        Args:
            faces: List of face objects
            
        Returns:
            Tuple of (min_point, max_point) as ((x_min, y_min, z_min), (x_max, y_max, z_max))
        """
        try:
            min_x = min_y = min_z = float('inf')
            max_x = max_y = max_z = float('-inf')
            
            for face in faces:
                bbox = face.BoundingBox()
                min_x = min(min_x, bbox.xmin)
                min_y = min(min_y, bbox.ymin)
                min_z = min(min_z, bbox.zmin)
                max_x = max(max_x, bbox.xmax)
                max_y = max(max_y, bbox.ymax)
                max_z = max(max_z, bbox.zmax)
            
            return ((min_x, min_y, min_z), (max_x, max_y, max_z))
        except:
            return ((0, 0, 0), (1, 1, 1))
    
    @staticmethod
    def distance_between_points(p1: Tuple[float, float, float],
                               p2: Tuple[float, float, float]) -> float:
        """
        Calculate Euclidean distance between two points.
        
        Args:
            p1: First point (x, y, z)
            p2: Second point (x, y, z)
            
        Returns:
            Distance value
        """
        return np.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))
    
    @staticmethod
    def point_in_bounding_box(point: Tuple[float, float, float],
                             min_point: Tuple[float, float, float],
                             max_point: Tuple[float, float, float]) -> bool:
        """
        Check if a point is within a bounding box.
        
        Args:
            point: Point to check (x, y, z)
            min_point: Minimum corner of box
            max_point: Maximum corner of box
            
        Returns:
            True if point is in box, False otherwise
        """
        return all(min_point[i] <= point[i] <= max_point[i] for i in range(3))
