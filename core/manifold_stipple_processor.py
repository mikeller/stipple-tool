"""Manifold-based stipple processor using trimesh + manifold3d for robust boolean cuts.

Pipeline: STEP → mesh → generate sphere positions → union all spheres →
single boolean difference → export STL/3MF.

Uses Google's manifold3d library (via trimesh) for guaranteed-manifold
boolean operations, avoiding the topology corruption that plagues
OCC BRepAlgoAPI_Cut on complex parts with thousands of sequential cuts.
"""
import time
import random
from typing import List, Tuple, Optional, Callable, Dict
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree


class ManifoldStippleProcessor:
    """Stipple processor using manifold3d boolean engine via trimesh."""

    def __init__(self):
        self.random_seed = 42

    def process(
        self,
        step_file: str,
        output_path: str,
        target_color: str,
        sphere_radius: float = 1.0,
        sphere_depth: float = 0.45,
        spheres_per_mm2: float = 0.35,
        size_variation: bool = True,
        size_variation_mode: str = "gaussian",
        size_variation_sigma: float = 0.25,
        size_variation_min: float = 0.60,
        size_variation_max: float = 1.6,
        deflection: float = 0.05,
        status_callback: Optional[Callable[[str], None]] = None,
    ) -> Optional[str]:
        """Run the full STEP → mesh → manifold boolean → export pipeline.

        Args:
            step_file: Path to input STEP file
            output_path: Path for output mesh (STL/3MF/OBJ)
            target_color: Hex color of faces to stipple (e.g. "#360200")
            sphere_radius: Base sphere radius in mm
            sphere_depth: How deep the sphere cap cuts into the surface (mm)
            spheres_per_mm2: Target sphere density
            size_variation: Enable random size variation
            size_variation_mode: "gaussian" or "uniform"
            size_variation_sigma: Gaussian sigma for size variation
            size_variation_min: Minimum scale factor
            size_variation_max: Maximum scale factor
            deflection: Mesh deflection for STEP→mesh conversion
            status_callback: Optional progress reporting function

        Returns:
            Output file path on success, None on failure.
        """
        def emit(msg: str):
            if status_callback:
                status_callback(msg)
            print(msg)

        emit("=" * 60)
        emit("MANIFOLD BOOLEAN STIPPLING")
        emit("=" * 60)
        emit(f"Input:         {step_file}")
        emit(f"Output:        {output_path}")
        emit(f"Color:         {target_color}")
        emit(f"Radius:        {sphere_radius} mm")
        emit(f"Depth:         {sphere_depth} mm")
        emit(f"Density:       {spheres_per_mm2} spheres/mm²")
        emit(f"Deflection:    {deflection}")
        emit(f"Size var:      {size_variation} ({size_variation_mode})")
        emit("=" * 60)

        t_start = time.time()
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

        # --- Step 1: Load STEP and identify target faces ---
        emit("\n[1/5] Loading STEP and identifying colored faces...")
        from core.step_loader import STEPLoader
        from core.color_analyzer import ColorAnalyzer
        from core.step_mesh_converter import STEPMeshConverter

        loader = STEPLoader()
        if not loader.load(step_file) or loader.shape is None:
            emit("✗ Failed to load STEP file")
            return None

        model = loader.get_model()
        if model is None:
            emit("✗ Failed to parse STEP model")
            return None

        colors = ColorAnalyzer().extract_colors_from_model(model)
        target_face_indices = colors.get(target_color, [])
        if not target_face_indices:
            emit(f"✗ No faces with color {target_color}")
            available = list(colors.keys())
            emit(f"  Available colors: {available}")
            return None

        emit(f"  Found {len(target_face_indices)} STEP faces with target color")

        # --- Step 2: Convert STEP to mesh ---
        emit("\n[2/5] Converting STEP to mesh...")
        converted = STEPMeshConverter.shape_to_mesh_with_face_map(
            loader.shape, deflection=deflection,
        )
        if converted is None:
            emit("✗ STEP → mesh conversion failed")
            return None

        part_mesh, face_map = converted
        emit(f"  Part mesh: {len(part_mesh.vertices)} vertices, {len(part_mesh.faces)} triangles")
        emit(f"  Watertight: {part_mesh.is_watertight}, volume: {part_mesh.is_volume}")

        # Map STEP face indices → mesh triangle indices BEFORE vertex merging.
        # The raw mesh has duplicate vertices at shared STEP-face edges
        # (process=False), so each STEP face is a disconnected patch.
        target_set = set(target_face_indices)
        target_tri_mask = np.array([
            step_idx in target_set for step_idx in face_map
        ], dtype=bool)
        if not target_tri_mask.any():
            emit("✗ No mesh triangles map to target STEP faces")
            return None

        # Record target vertex POSITIONS (not indices) — positions survive
        # vertex merging while indices do not.
        target_tri_indices_pre = np.where(target_tri_mask)[0]
        target_vert_indices_pre = np.unique(part_mesh.faces[target_tri_indices_pre].ravel())
        target_positions = part_mesh.vertices[target_vert_indices_pre]

        # Merge coincident vertices & repair to make manifold
        emit("  Merging vertices and repairing mesh...")
        part_mesh.merge_vertices(merge_tex=True, merge_norm=True)
        part_mesh.process(validate=True)
        trimesh.repair.fix_winding(part_mesh)
        trimesh.repair.fix_normals(part_mesh, multibody=True)
        if not part_mesh.is_watertight:
            trimesh.repair.fill_holes(part_mesh)
            part_mesh.process(validate=True)
        emit(f"  After trimesh repair: {len(part_mesh.vertices)} verts, "
             f"{len(part_mesh.faces)} tris, "
             f"watertight={part_mesh.is_watertight}, volume={part_mesh.is_volume}")

        # If still not watertight, use pymeshfix (more robust repair)
        if not part_mesh.is_watertight:
            try:
                import pymeshfix
                emit("  Attempting pymeshfix repair...")
                meshfix = pymeshfix.MeshFix(
                    np.asarray(part_mesh.vertices),
                    np.asarray(part_mesh.faces),
                )
                meshfix.repair(verbose=False)
                part_mesh = trimesh.Trimesh(
                    vertices=meshfix.v, faces=meshfix.f, process=True,
                )
                emit(f"  After pymeshfix: {len(part_mesh.vertices)} verts, "
                     f"{len(part_mesh.faces)} tris, "
                     f"watertight={part_mesh.is_watertight}, "
                     f"volume={part_mesh.is_volume}")
            except Exception as e:
                emit(f"  pymeshfix failed: {e}")

        # Fix inverted normals — if volume is negative, flip all faces
        if part_mesh.is_watertight and part_mesh.volume < 0:
            emit("  Flipping normals (negative volume detected)...")
            part_mesh.invert()

        emit(f"  euler={part_mesh.euler_number}, volume={abs(part_mesh.volume):.1f}")

        # Re-identify target vertices by position lookup (KDTree).
        # pymeshfix may slightly shift vertices, so use a reasonable tolerance.
        from scipy.spatial import cKDTree
        tree = cKDTree(part_mesh.vertices)
        tol = 0.01  # 0.01mm tolerance for vertex matching
        dists, indices = tree.query(target_positions, distance_upper_bound=tol)
        matched = indices[dists < tol]
        target_vertex_set = set(matched)
        emit(f"  Matched {len(target_vertex_set)} target vertices "
             f"(from {len(target_positions)} original, tol={tol}mm)")

        # Target triangles: all 3 vertices are in the target set
        target_tri_indices = np.array([
            i for i in range(len(part_mesh.faces))
            if all(v in target_vertex_set for v in part_mesh.faces[i])
        ])

        # If too few triangles found, also match by face centroid proximity
        # to the original target surface
        if len(target_tri_indices) < len(target_tri_indices_pre) * 0.5:
            emit(f"  Vertex match got only {len(target_tri_indices)} tris "
                 f"(expected ~{len(target_tri_indices_pre)}), "
                 f"falling back to centroid proximity...")
            # Build a reference submesh from original target triangles
            orig_verts = np.asarray(converted[0].vertices)
            orig_faces = np.asarray(converted[0].faces)
            orig_target_faces = orig_faces[target_tri_indices_pre]
            # Get centroids of original target triangles
            orig_centroids = orig_verts[orig_target_faces].mean(axis=1)
            ref_tree = cKDTree(orig_centroids)

            # For each repaired mesh face, check if its centroid is close
            # to any original target centroid
            repaired_centroids = part_mesh.vertices[part_mesh.faces].mean(axis=1)
            dists2, _ = ref_tree.query(repaired_centroids)
            # A face is "target" if its centroid is within 0.5mm of an
            # original target centroid (generous for mesh restructuring)
            target_tri_indices = np.where(dists2 < 0.5)[0]
            emit(f"  Centroid proximity matched {len(target_tri_indices)} tris")

        if len(target_tri_indices) == 0:
            emit("✗ No mesh triangles survive repair for target faces")
            return None

        target_area = part_mesh.area_faces[target_tri_indices].sum()
        emit(f"  Target triangles: {len(target_tri_indices)}, area: {target_area:.1f} mm²")

        # --- Step 3: Generate sphere positions on target faces ---
        emit("\n[3/5] Generating sphere positions...")
        sphere_specs = self._generate_sphere_positions(
            mesh=part_mesh,
            target_tri_indices=target_tri_indices,
            spheres_per_mm2=spheres_per_mm2,
            base_radius=sphere_radius,
            sphere_depth=sphere_depth,
            size_variation=size_variation,
            size_variation_mode=size_variation_mode,
            size_variation_sigma=size_variation_sigma,
            size_variation_min=size_variation_min,
            size_variation_max=size_variation_max,
        )
        if not sphere_specs:
            emit("✗ No sphere positions generated")
            return None
        emit(f"  Generated {len(sphere_specs)} sphere positions")

        # --- Step 4: Build sphere union & boolean difference ---
        emit("\n[4/5] Boolean difference via manifold3d...")
        result_mesh = self._manifold_boolean_cut(
            part_mesh, sphere_specs, emit,
        )
        if result_mesh is None:
            emit("✗ Boolean operation failed")
            return None

        # --- Step 5: Export ---
        emit("\n[5/5] Exporting result...")
        out = Path(output_path)
        if out.suffix.lower() not in {".stl", ".3mf", ".obj", ".ply", ".glb", ".gltf"}:
            out = out.with_suffix(".stl")

        result_mesh.export(str(out))
        elapsed = time.time() - t_start
        emit(f"\n✓ Done in {elapsed:.1f}s → {out}")
        emit(f"  Result: {len(result_mesh.vertices)} vertices, {len(result_mesh.faces)} triangles")
        emit(f"  Watertight: {result_mesh.is_watertight}")
        return str(out)

    def _generate_sphere_positions(
        self,
        mesh: trimesh.Trimesh,
        target_tri_indices: np.ndarray,
        spheres_per_mm2: float,
        base_radius: float,
        sphere_depth: float,
        size_variation: bool,
        size_variation_mode: str,
        size_variation_sigma: float,
        size_variation_min: float,
        size_variation_max: float,
    ) -> List[Tuple[np.ndarray, float]]:
        """Sample sphere centre positions on the target mesh faces.

        Returns list of (centre_xyz, radius) tuples.
        Each sphere centre is offset outward from the surface so that only
        a shallow cap (depth) intersects the solid.
        """
        target_area = mesh.area_faces[target_tri_indices].sum()
        num_spheres = max(3, int(target_area * spheres_per_mm2))

        # Build a submesh of target triangles for sampling
        submesh = mesh.submesh([target_tri_indices], append=True, repair=False)
        points, face_ids = trimesh.sample.sample_surface(submesh, num_spheres)

        # Get normals at sampled points (from the triangle they fell on)
        normals = submesh.face_normals[face_ids]

        # Determine outward vs inward direction by checking if the normal
        # points away from the mesh interior. We use the mesh centroid as
        # a rough "inside" reference.
        centroid = mesh.centroid
        to_centroid = centroid - points
        # If normal · (centroid - point) > 0, normal points inward → flip
        dots = np.einsum("ij,ij->i", normals, to_centroid)
        outward_normals = normals.copy()
        flip_mask = dots > 0
        outward_normals[flip_mask] *= -1

        sphere_specs = []
        for i in range(len(points)):
            if size_variation:
                if size_variation_mode == "gaussian":
                    scale = random.gauss(1.0, size_variation_sigma)
                    scale = max(size_variation_min, min(size_variation_max, scale))
                else:
                    scale = 0.6 + random.random() * 0.8
                radius = base_radius * scale
            else:
                radius = base_radius

            # Effective depth scales slightly with radius
            scale_factor = 1.0 + 0.15 * ((radius / base_radius) - 1.0)
            effective_depth = min(sphere_depth * scale_factor, sphere_depth * 1.5, radius)

            # Offset centre outward so only a cap of `effective_depth` intersects
            offset = radius - effective_depth
            if offset < 0:
                offset = 0

            centre = points[i] + outward_normals[i] * offset
            sphere_specs.append((centre, radius))

        # Poisson-disk-like rejection: remove spheres whose centres are
        # too close together (< base_radius * 0.8)
        if len(sphere_specs) > 1:
            centres = np.array([s[0] for s in sphere_specs])
            tree = cKDTree(centres)
            min_dist = base_radius * 0.8
            keep = np.ones(len(sphere_specs), dtype=bool)
            for idx in range(len(sphere_specs)):
                if not keep[idx]:
                    continue
                neighbors = tree.query_ball_point(centres[idx], min_dist)
                for n in neighbors:
                    if n != idx and keep[n]:
                        keep[n] = False
            sphere_specs = [s for s, k in zip(sphere_specs, keep) if k]

        return sphere_specs

    def _manifold_boolean_cut(
        self,
        part_mesh: trimesh.Trimesh,
        sphere_specs: List[Tuple[np.ndarray, float]],
        emit: Callable[[str], None],
    ) -> Optional[trimesh.Trimesh]:
        """Union all spheres, then subtract from the part in one operation."""
        try:
            from manifold3d import Manifold, Mesh as ManifoldMesh
        except ImportError:
            emit("✗ manifold3d not installed. Run: pip install manifold3d")
            return None

        t0 = time.time()

        # Build all sphere meshes and union them into one manifold
        emit(f"  Building {len(sphere_specs)} sphere meshes...")
        batch_size = 500
        sphere_batches = []

        for batch_start in range(0, len(sphere_specs), batch_size):
            batch_end = min(batch_start + batch_size, len(sphere_specs))
            batch_spheres = []
            for centre, radius in sphere_specs[batch_start:batch_end]:
                sphere = trimesh.creation.icosphere(subdivisions=2, radius=radius)
                sphere.apply_translation(centre)
                batch_spheres.append(sphere)

            # Concatenate batch into one mesh, then use manifold union
            batch_mesh = trimesh.util.concatenate(batch_spheres)
            sphere_batches.append(batch_mesh)
            if len(sphere_specs) > batch_size:
                emit(f"  Built batch {batch_start//batch_size + 1}/"
                     f"{(len(sphere_specs) + batch_size - 1)//batch_size} "
                     f"({batch_end - batch_start} spheres)")

        # Union all batches
        if len(sphere_batches) == 1:
            tool_mesh = sphere_batches[0]
        else:
            tool_mesh = trimesh.util.concatenate(sphere_batches)

        t1 = time.time()
        emit(f"  Sphere mesh built: {len(tool_mesh.vertices)} vertices, "
             f"{len(tool_mesh.faces)} triangles ({t1-t0:.1f}s)")

        # Use manifold3d directly for maximum control
        emit("  Converting to Manifold objects...")
        try:
            part_manifold = self._trimesh_to_manifold(part_mesh, Manifold, ManifoldMesh)
            if part_manifold.is_empty():
                emit(f"  ✗ Part manifold is empty (status={part_manifold.status()})")
                emit("    The mesh is likely not manifold even after repair")
                return None
            emit(f"  Part manifold: OK ({part_manifold.num_vert()} verts, "
                 f"{part_manifold.num_tri()} tris, vol={part_manifold.volume():.1f})")
        except Exception as e:
            emit(f"  ✗ Part mesh → Manifold failed: {e}")
            return None

        # For spheres: union them using manifold's batch_boolean
        # Each icosphere is already manifold, so build individual manifolds and union
        emit("  Building sphere manifolds and unioning...")
        t_union_start = time.time()
        sphere_manifolds = []
        for i, (centre, radius) in enumerate(sphere_specs):
            m = Manifold.sphere(radius, circular_segments=16)
            m = m.translate([centre[0], centre[1], centre[2]])
            sphere_manifolds.append(m)

        # Batch union using tree reduction for O(n log n) instead of O(n²)
        while len(sphere_manifolds) > 1:
            next_level = []
            for i in range(0, len(sphere_manifolds), 2):
                if i + 1 < len(sphere_manifolds):
                    next_level.append(sphere_manifolds[i] + sphere_manifolds[i+1])
                else:
                    next_level.append(sphere_manifolds[i])
            sphere_manifolds = next_level

        tool_manifold = sphere_manifolds[0]
        t_union_end = time.time()
        emit(f"  Sphere union complete ({t_union_end - t_union_start:.1f}s)")

        # Boolean difference
        emit("  Running manifold boolean difference...")
        try:
            result_manifold = part_manifold - tool_manifold
        except Exception as e:
            emit(f"  ✗ Boolean difference failed: {e}")
            return None

        t2 = time.time()
        emit(f"  Boolean complete ({t2 - t_union_end:.1f}s)")

        # Convert back to trimesh
        try:
            result_mesh = self._manifold_to_trimesh(result_manifold)
        except Exception as e:
            emit(f"  ✗ Manifold → trimesh conversion failed: {e}")
            return None

        if result_mesh is None or result_mesh.is_empty:
            emit("  ✗ Boolean returned empty result")
            return None

        # Volume sanity check
        original_vol = abs(part_mesh.volume) if part_mesh.is_watertight else 0
        result_vol = abs(result_mesh.volume) if result_mesh.is_watertight else 0
        if original_vol > 0 and result_vol > 0:
            removed_pct = (original_vol - result_vol) / original_vol * 100
            emit(f"  Volume: {original_vol:.1f} → {result_vol:.1f} mm³ "
                 f"(removed {removed_pct:.1f}%)")
            if removed_pct > 50:
                emit("  ⚠ WARNING: >50% volume removed — check parameters")

        return result_mesh

    @staticmethod
    def _trimesh_to_manifold(mesh: trimesh.Trimesh, Manifold, ManifoldMesh):
        """Convert a trimesh to a manifold3d Manifold object."""
        verts = np.asarray(mesh.vertices, dtype=np.float32)
        tris = np.asarray(mesh.faces, dtype=np.uint32)
        m_mesh = ManifoldMesh(vert_properties=verts, tri_verts=tris)
        return Manifold(m_mesh)

    @staticmethod
    def _manifold_to_trimesh(manifold) -> trimesh.Trimesh:
        """Convert a manifold3d Manifold back to a trimesh."""
        m_mesh = manifold.to_mesh()
        verts = np.array(m_mesh.vert_properties[:, :3])
        tris = np.array(m_mesh.tri_verts)
        return trimesh.Trimesh(vertices=verts, faces=tris, process=True)
