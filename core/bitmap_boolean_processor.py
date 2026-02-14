"""Boolean stippling using bitmap patterns for controlled distribution."""
import math
import time
from typing import List, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter

try:
    from OCP.gp import gp_Pnt, gp_Vec
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeSphere
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
    from OCP.gp import gp_Trsf
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopoDS import TopoDS
    from OCP.BRep import BRep_Tool
    from OCP.GeomLProp import GeomLProp_SLProps
    from OCP.BRepClass3d import BRepClass3d_SolidClassifier
    from OCP.TopAbs import TopAbs_IN, TopAbs_OUT
    HAS_OCP = True
except ImportError:
    HAS_OCP = False

from core.step_loader import STEPLoader
from core.color_analyzer import ColorAnalyzer


class BitmapBooleanProcessor:
    """Apply stippling using bitmap patterns and boolean sphere cuts."""

    def __init__(
        self,
        sphere_radius: float = 1.0,
        sphere_depth: float = 0.5,
        bitmap_size: int = 512,
        threshold: float = 0.35,
        seed: int = 42,
        cut_timeout: float = 10.0,
    ):
        """
        Initialize bitmap boolean processor.

        Args:
            sphere_radius: Radius of stipple spheres in mm
            sphere_depth: Depth of sphere cuts in mm
            bitmap_size: Size of bitmap texture (NxN)
            threshold: Bitmap threshold for sphere placement (0-1)
            seed: Random seed for bitmap generation
            cut_timeout: Timeout for individual boolean cuts in seconds
        """
        if not HAS_OCP:
            raise ImportError("cadquery-ocp is required")

        self.sphere_radius = sphere_radius
        self.sphere_depth = sphere_depth
        self.bitmap_size = bitmap_size
        self.threshold = threshold
        self.seed = seed
        self.cut_timeout = cut_timeout  # Kept for potential future use

    def _generate_blue_noise_bitmap(self) -> np.ndarray:
        """Generate a blue-noise-like bitmap using Gaussian filtering."""
        np.random.seed(self.seed)
        noise = np.random.random((self.bitmap_size, self.bitmap_size))
        # Apply Gaussian filter to create low-frequency noise
        filtered = gaussian_filter(noise, sigma=1.5)
        # Normalize to 0-1
        filtered = (filtered - filtered.min()) / (filtered.max() - filtered.min())
        return filtered

    def _sample_bitmap_points(self, bitmap: np.ndarray, face_area: float) -> List[Tuple[float, float]]:
        """
        Sample UV points from bitmap where values exceed threshold.

        Args:
            bitmap: 2D array with values 0-1
            face_area: Area of the face in mm²

        Returns:
            List of (u, v) coordinates in 0-1 range
        """
        points = []
        height, width = bitmap.shape

        # Sample at regular grid and check bitmap values
        for i in range(height):
            for j in range(width):
                if bitmap[i, j] > self.threshold:
                    u = j / width
                    v = i / height
                    points.append((u, v))

        return points

    def _uv_to_surface_point(self, face, u: float, v: float) -> Tuple[gp_Pnt, gp_Vec]:
        """
        Convert UV coordinates to 3D point on face surface.

        Args:
            face: OCP face
            u, v: Parameters in 0-1 range

        Returns:
            (point, normal) tuple
        """
        surface = BRep_Tool.Surface_s(face)
        u_min, u_max, v_min, v_max = BRep_Tool.Surface_s(face).Bounds()

        # Map 0-1 to actual parameter range
        u_param = u_min + u * (u_max - u_min)
        v_param = v_min + v * (v_max - v_min)

        props = GeomLProp_SLProps(surface, u_param, v_param, 1, 1e-6)

        if not props.IsNormalDefined():
            # Fallback: use approximate normal
            point = surface.Value(u_param, v_param)
            normal = gp_Vec(0, 0, 1)
        else:
            point = props.Value()
            normal = props.Normal()

        # Ensure normal points outward
        face_orientation = face.Orientation()
        from OCP.TopAbs import TopAbs_REVERSED
        if face_orientation == TopAbs_REVERSED:
            normal.Reverse()

        return point, normal

    def _get_face_area(self, face) -> float:
        """Calculate face area in mm²."""
        props = GProp_GProps()
        BRepGProp.SurfaceProperties_s(face, props)
        return props.Mass()

    def _make_sphere_at_point(self, center: gp_Pnt, radius: float) -> any:
        """Create a sphere at the given point."""
        sphere_maker = BRepPrimAPI_MakeSphere(center, radius)
        return sphere_maker.Shape()

    def _try_boolean_cut(self, current_shape, sphere):
        """Attempt boolean cut with exception handling."""
        try:
            cutter = BRepAlgoAPI_Cut(current_shape, sphere)
            cutter.SetFuzzyValue(0.01)
            cutter.Build()
            if cutter.IsDone():
                return cutter.Shape()
        except Exception:
            pass
        return None

    def _is_point_inside_solid(self, shape, point: gp_Pnt) -> bool:
        """Check if a point is inside the solid."""
        classifier = BRepClass3d_SolidClassifier(shape, point, 1e-6)
        state = classifier.State()
        return state == TopAbs_IN

    def process(
        self,
        shape,
        target_color: str,
        loader: STEPLoader,
    ):
        """
        Apply bitmap-based boolean stippling to colored faces.

        Args:
            shape: OCP shape to process
            target_color: Hex color string (e.g., "#360200")
            loader: STEPLoader instance for color information

        Returns:
            Modified shape or None if processing fails
        """
        print("\nBITMAP BOOLEAN STIPPLING")
        print("=" * 50)
        print(f"Sphere radius: {self.sphere_radius} mm")
        print(f"Sphere depth:  {self.sphere_depth} mm")
        print(f"Bitmap size:   {self.bitmap_size}×{self.bitmap_size}")
        print(f"Threshold:     {self.threshold}")
        print(f"Cut timeout:   {self.cut_timeout}s")
        print("=" * 50)

        # Find target faces by color
        color_analyzer = ColorAnalyzer()
        model = loader.get_model()
        if model is None:
            print("❌ Failed to get model from loader")
            return None
        
        shape_tool = model.get("shape_tool")
        color_tool = model.get("color_tool")
        
        target_faces = []
        
        explorer = TopExp_Explorer(shape, TopAbs_FACE)
        face_idx = 0
        while explorer.More():
            face = TopoDS.Face_s(explorer.Current())
            color = color_analyzer._find_face_color(face, shape_tool, color_tool)
            if color is not None:
                face_color = color_analyzer._color_to_hex(color)
                if face_color.upper() == target_color.upper():
                    target_faces.append((face, face_idx))
            explorer.Next()
            face_idx += 1

        if not target_faces:
            print(f"❌ No faces found with color {target_color}")
            return None

        print(f"Found {len(target_faces)} target faces")

        # Generate bitmap once
        print("Generating bitmap pattern...")
        bitmap = self._generate_blue_noise_bitmap()

        # Collect all sphere positions across all faces
        all_spheres = []
        for face, face_idx in target_faces:
            face_area = self._get_face_area(face)
            uv_points = self._sample_bitmap_points(bitmap, face_area)
            
            for u, v in uv_points:
                try:
                    point, normal = self._uv_to_surface_point(face, u, v)
                    
                    # Position sphere so it cuts to specified depth
                    # Center is at: surface_point + normal * (radius - depth)
                    offset = self.sphere_radius - self.sphere_depth
                    center = gp_Pnt(
                        point.X() + normal.X() * offset,
                        point.Y() + normal.Y() * offset,
                        point.Z() + normal.Z() * offset,
                    )
                    
                    all_spheres.append((center, self.sphere_radius, point, normal))
                except Exception as e:
                    continue

        print(f"Generated {len(all_spheres)} sphere positions")

        if not all_spheres:
            print("❌ No valid sphere positions generated")
            return None

        # Limit total spheres to something reasonable
        max_spheres = 5000
        if len(all_spheres) > max_spheres:
            print(f"⚠️  Limiting from {len(all_spheres)} to {max_spheres} spheres")
            import random
            random.shuffle(all_spheres)
            all_spheres = all_spheres[:max_spheres]

        # Apply boolean cuts
        current_shape = shape
        successful_cuts = 0
        failed_cuts = 0
        timeout_cuts = 0
        consecutive_failures = 0
        max_consecutive_failures = 100
        start_time = time.time()

        print(f"\nApplying {len(all_spheres)} boolean cuts...")
        
        for i, (center, radius, surface_point, normal) in enumerate(all_spheres):
            if (i + 1) % 100 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                print(f"  Progress: {i+1}/{len(all_spheres)} ({successful_cuts} successful, {failed_cuts} failed, {timeout_cuts} timeout) [{rate:.1f} cuts/s]")

            # Early stop if too many consecutive failures
            if consecutive_failures >= max_consecutive_failures:
                print(f"\n⚠️  Stopping early: {max_consecutive_failures} consecutive failures")
                break

            try:
                # Create sphere
                sphere = self._make_sphere_at_point(center, radius)

                # Attempt cut
                result = self._try_boolean_cut(current_shape, sphere)
                
                if result is not None:
                    current_shape = result
                    successful_cuts += 1
                    consecutive_failures = 0  # Reset on success
                else:
                    failed_cuts += 1
                    consecutive_failures += 1

            except Exception as e:
                failed_cuts += 1
                consecutive_failures += 1
                continue

        elapsed = time.time() - start_time
        print(f"\n{'=' * 50}")
        print(f"Completed in {elapsed:.1f}s")
        print(f"Successful cuts: {successful_cuts}/{len(all_spheres)}")
        print(f"Failed cuts:     {failed_cuts}")
        print(f"Timeout cuts:    {timeout_cuts}")
        print(f"{'=' * 50}")

        return current_shape if successful_cuts > 0 else None
