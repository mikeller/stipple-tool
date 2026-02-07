"""Stippling texture generation engine using OCP geometry operations."""
from typing import List, Tuple, Optional, Callable
import numpy as np

from OCP.BRep import BRep_Tool, BRep_Builder
from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
from OCP.BRepGProp import BRepGProp
from OCP.BRepTools import BRepTools
from OCP.BRepClass import BRepClass_FaceClassifier
from OCP.GeomLProp import GeomLProp_SLProps
from OCP.GProp import GProp_GProps
from OCP.gp import gp_Pnt, gp_Vec, gp_Pnt2d
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_FACE, TopAbs_IN, TopAbs_ON
from OCP.TopoDS import TopoDS_Shape, TopoDS, TopoDS_Compound
from OCP.BRepPrimAPI import BRepPrimAPI_MakeSphere


class StippleEngine:
    """Generates and applies stippling textures to surfaces."""

    def __init__(self):
        self.stipple_size = 2.5  # mm
        self.stipple_depth = 1.5  # mm
        self.stipple_density = 0.70  # 0.0 to 1.0
        self.depth_variation = 0.6  # 0.0 to 1.0
        self.size_variation = 0.35  # 0.0 to 1.0
        self.coverage_factor = 4.0
        self.layers = 3
        self.layer_density_decay = 0.7
        self.layer_size_decay = 0.8
        self.layer_depth_decay = 0.6
        self.cut_batch_size = 150
        self.max_total_spheres = 2500
        self.random_seed = 42

    def set_parameters(
        self,
        size: float,
        depth: float,
        density: float,
        depth_variation: float = 0.6,
        size_variation: float = 0.35,
        coverage_factor: float = 4.0,
        layers: int = 3,
        layer_density_decay: float = 0.7,
        layer_size_decay: float = 0.8,
        layer_depth_decay: float = 0.6,
        cut_batch_size: int = 150,
        max_total_spheres: int = 2500,
    ):
        self.stipple_size = max(0.1, size)
        self.stipple_depth = max(0.1, depth)
        self.stipple_density = max(0.01, min(1.0, density))
        self.depth_variation = max(0.0, min(1.0, depth_variation))
        self.size_variation = max(0.0, min(1.0, size_variation))
        self.coverage_factor = max(0.1, coverage_factor)
        self.layers = max(1, min(6, int(layers)))
        self.layer_density_decay = max(0.1, min(1.0, layer_density_decay))
        self.layer_size_decay = max(0.1, min(1.0, layer_size_decay))
        self.layer_depth_decay = max(0.1, min(1.0, layer_depth_decay))
        self.cut_batch_size = max(25, min(1000, int(cut_batch_size)))
        self.max_total_spheres = max(100, min(10000, int(max_total_spheres)))

    def _generate_spheres(
        self,
        shape: TopoDS_Shape,
        faces: List,
        pattern: str = "random",
        max_stipples_per_face: int = 6000,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        status_callback: Optional[Callable[[str], None]] = None,
        cancel_callback: Optional[Callable[[], None]] = None,
    ) -> List:
        """
        Generate sphere geometry for stippling. Can be run in main thread.
        Returns list of sphere shapes to be used in boolean cut.
        """
        if shape.IsNull() or not faces:
            return []

        np.random.seed(self.random_seed)

        spheres = []

        effective_max_stipples = max_stipples_per_face
        if self.stipple_density > 0.9:
            effective_max_stipples = min(max_stipples_per_face, 2000)
            if status_callback:
                status_callback(f"Very high density: capping stipples to {effective_max_stipples}")
        elif self.stipple_density > 0.8:
            effective_max_stipples = min(max_stipples_per_face, 3000)

        effective_max_total = self.max_total_spheres
        if self.stipple_density > 0.9:
            effective_max_total = min(effective_max_total, 1500)
        elif self.stipple_density > 0.8:
            effective_max_total = min(effective_max_total, 2000)

        total_steps = max(1, len(faces) * self.layers)
        completed_steps = 0

        effective_layers = self.layers
        if self.stipple_density > 0.85:
            effective_layers = max(1, self.layers - 1)
            if status_callback:
                status_callback(f"High density: reducing to {effective_layers} layer(s) for speed")

        for face in faces:
            for layer in range(effective_layers):
                if cancel_callback:
                    try:
                        cancel_callback()
                    except RuntimeError:
                        return spheres

                density_scale = self.layer_density_decay ** layer
                size_scale = self.layer_size_decay ** layer
                depth_scale = self.layer_depth_decay ** layer

                face_points = self._generate_stipple_points_on_face(
                    face,
                    pattern=pattern,
                    max_points=effective_max_stipples,
                    density_scale=density_scale,
                    size_for_count=self.stipple_size * size_scale,
                )

                for point, normal in face_points:
                    depth = self._sample_depth() * depth_scale
                    size = self._sample_size() * size_scale
                    center = gp_Pnt(
                        point.X() - normal.X() * depth,
                        point.Y() - normal.Y() * depth,
                        point.Z() - normal.Z() * depth,
                    )
                    sphere = BRepPrimAPI_MakeSphere(center, size / 2).Shape()
                    spheres.append(sphere)

                    if len(spheres) >= effective_max_total:
                        if status_callback:
                            status_callback(
                                f"Reached max stipples ({effective_max_total}); stopping generation"
                            )
                        return spheres

                completed_steps += 1
                if progress_callback:
                    progress_callback(completed_steps, total_steps)

        return spheres

    def apply_stippling_to_shape(
        self,
        shape: TopoDS_Shape,
        faces: List,
        pattern: str = "random",
        max_stipples_per_face: int = 6000,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        status_callback: Optional[Callable[[str], None]] = None,
        cancel_callback: Optional[Callable[[], None]] = None,
    ) -> TopoDS_Shape:
        """
        Apply stippling to selected faces by cutting small spheres.
        This method now generates spheres and applies the cut.

        Args:
            shape: Shape to modify
            faces: List of face objects to stipple
            pattern: "random", "grid", or "hexagon"
            max_stipples_per_face: Cap to avoid extreme operation counts

        Returns:
            Modified shape
        """
        if shape.IsNull() or not faces:
            return shape

        # Generate spheres first
        spheres = self._generate_spheres(
            shape,
            faces,
            pattern,
            max_stipples_per_face,
            progress_callback,
            status_callback,
            cancel_callback,
        )

        if not spheres:
            return shape

        # Perform boolean cut with status updates
        return self._apply_boolean_cut(shape, spheres, status_callback, cancel_callback)

    def _apply_boolean_cut(
        self,
        shape: TopoDS_Shape,
        spheres: List,
        status_callback: Optional[Callable[[str], None]],
        cancel_callback: Optional[Callable[[], None]],
    ) -> TopoDS_Shape:
        """
        Apply boolean cut operation. This can be run in a separate thread.
        Uses compound approach (more reliable than fusing all spheres).
        """
        if not spheres:
            if status_callback:
                status_callback("No spheres to cut")
            return shape

        if cancel_callback:
            try:
                cancel_callback()
            except RuntimeError:
                if status_callback:
                    status_callback("Cancelling...")
                return shape

        if status_callback:
            status_callback(f"Creating compound with {len(spheres)} spheres...")

        # Create compound of all spheres
        compound = TopoDS_Compound()
        builder = BRep_Builder()
        builder.MakeCompound(compound)
        
        for i, sphere in enumerate(spheres):
            if sphere.IsNull():
                if status_callback:
                    status_callback(f"Warning: Skipping null sphere at index {i}")
                continue
            builder.Add(compound, sphere)
            if i > 0 and i % 500 == 0 and status_callback:
                status_callback(f"Building compound: {i}/{len(spheres)} spheres")

        if compound.IsNull():
            if status_callback:
                status_callback("Compound is empty, returning original shape")
            return shape

        if status_callback:
            status_callback(f"Performing boolean cut with {len(spheres)} spheres (this may take a while)...")

        if cancel_callback:
            try:
                cancel_callback()
            except RuntimeError:
                if status_callback:
                    status_callback("Cancelling...")
                return shape

        try:
            cut_op = BRepAlgoAPI_Cut(shape, compound)
            result = cut_op.Shape()
            
            if result.IsNull():
                if status_callback:
                    status_callback("Boolean cut returned null shape, returning original")
                return shape
            
            if status_callback:
                status_callback("Boolean cut complete - stippling applied")
            return result
            
        except Exception as e:
            if status_callback:
                status_callback(f"Cut error: {str(e)}, returning original")
            return shape

    def _generate_stipple_points_on_face(
        self,
        face,
        pattern: str,
        max_points: int,
        density_scale: float = 1.0,
        size_for_count: Optional[float] = None,
    ) -> List[Tuple[gp_Pnt, gp_Vec]]:
        """Generate points and normals on a face surface."""
        props = GProp_GProps()
        BRepGProp.SurfaceProperties_s(face, props)
        area = max(props.Mass(), 1.0)

        effective_size = size_for_count or self.stipple_size
        count = int((area / (effective_size ** 2)) * self.stipple_density * self.coverage_factor * density_scale)
        count = max(1, min(count, max_points))

        umin, umax, vmin, vmax = BRepTools.UVBounds_s(face)
        points = []
        classifier = BRepClass_FaceClassifier()

        attempts = 0
        max_attempts = count * 20

        while len(points) < count and attempts < max_attempts:
            attempts += 1
            u = np.random.uniform(umin, umax)
            v = np.random.uniform(vmin, vmax)

            classifier.Perform(face, gp_Pnt2d(u, v), 1e-6)
            state = classifier.State()
            if state not in (TopAbs_IN, TopAbs_ON):
                continue

            surface = BRep_Tool.Surface_s(face)
            props = GeomLProp_SLProps(surface, u, v, 1, 1e-6)
            if not props.IsNormalDefined():
                continue

            point = props.Value()
            normal_dir = props.Normal()
            normal = gp_Vec(normal_dir.X(), normal_dir.Y(), normal_dir.Z())
            points.append((point, normal))

        return points

    def _sample_depth(self) -> float:
        if self.depth_variation <= 0.0:
            return self.stipple_depth
        factor = (1.0 - self.depth_variation / 2.0) + self.depth_variation * np.random.rand()
        return max(0.05, self.stipple_depth * factor)

    def _sample_size(self) -> float:
        if self.size_variation <= 0.0:
            return self.stipple_size
        factor = (1.0 - self.size_variation / 2.0) + self.size_variation * np.random.rand()
        return max(0.05, self.stipple_size * factor)
