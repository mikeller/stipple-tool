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
        sphere_radius: float = 1.4,
        sphere_depth: float = 0.6,
        spheres_per_mm2: float = 0.5,
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

        # Record target triangle CENTROIDS (not vertex indices) for robust
        # re-identification after mesh repair. Centroids are unique per-face
        # and avoid the boundary-sharing problem of vertex-based matching.
        target_tri_indices_pre = np.where(target_tri_mask)[0]
        orig_verts = np.asarray(part_mesh.vertices)
        orig_faces = np.asarray(part_mesh.faces)

        # Compute centroids for ALL original triangles, plus a label
        # array marking which are target vs non-target.
        all_orig_centroids = orig_verts[orig_faces].mean(axis=1)
        orig_is_target = target_tri_mask.copy()  # bool array, length = num original tris

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

        # Re-identify target triangles by nearest-centroid classification.
        # For each repaired triangle, find the closest original centroid
        # (target or non-target). If it belonged to a target triangle, the
        # repaired triangle is classified as target.
        from scipy.spatial import cKDTree
        ref_tree = cKDTree(all_orig_centroids)
        repaired_centroids = part_mesh.vertices[part_mesh.faces].mean(axis=1)
        dists, nearest_idx = ref_tree.query(repaired_centroids)
        # A repaired triangle is "target" if its nearest original centroid
        # was a target triangle AND the distance is reasonable.
        orig_edge_len = np.sqrt(part_mesh.area_faces.mean()) * 1.0
        centroid_tol = max(0.1, min(orig_edge_len, 0.5))
        is_matched = (dists < centroid_tol) & orig_is_target[nearest_idx]
        target_tri_indices = np.where(is_matched)[0]
        matched_area = part_mesh.area_faces[target_tri_indices].sum() if len(target_tri_indices) > 0 else 0
        emit(f"  Matched {len(target_tri_indices)} target triangles "
             f"(centroid tol={centroid_tol:.3f}mm, "
             f"orig {len(target_tri_indices_pre)} tris, "
             f"matched area={matched_area:.0f}mm²)")

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
        # The repaired mesh has consistent outward-facing normals (positive
        # volume was verified after repair/flip).  submesh preserves face
        # winding from the parent mesh, so submesh.face_normals already
        # point outward — no centroid-based heuristic needed.
        outward_normals = submesh.face_normals[face_ids].copy()

        # --- Compute boundary edges and edge margin ---
        # Find the boundary of the target submesh: edges that appear in only
        # one triangle. Sphere positions near this boundary will "run over"
        # into non-target surfaces.
        boundary_points = self._get_submesh_boundary_points(submesh)

        # The lateral radius of the sphere cap on the surface:
        # r_lateral = sqrt(2*R*d - d²) where R=radius, d=depth
        base_lateral = np.sqrt(max(0, 2 * base_radius * sphere_depth - sphere_depth**2))

        # Pre-build KDTree of boundary points for fast proximity queries
        boundary_tree = cKDTree(boundary_points) if len(boundary_points) > 0 else None

        # --- Curvature for depth compensation ---
        # On concave surfaces a sphere cuts deeper than the nominal depth
        # because the surface curves toward the sphere centre.  We measure
        # curvature so we can *reduce* the depth on concave areas and
        # *increase* it slightly on convex areas, keeping effective depth
        # approximately constant.
        point_curvatures = self._compute_point_curvatures(mesh, submesh, points, face_ids)
        nz = point_curvatures[point_curvatures != 0]
        if len(nz) > 0:
            print(f"  Curvature stats: min={nz.min():.4f}, median={np.median(nz):.4f}, "
                  f"max={nz.max():.4f}, nonzero={len(nz)}/{len(point_curvatures)}")

        # --- Thin-wall check: use ray casting on the full mesh ---
        ray_caster = mesh  # trimesh has built-in ray casting

        sphere_specs = []
        edge_rejected = 0
        thin_rejected = 0
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

            # --- Curvature-compensated depth ---
            # trimesh: κ > 0 = convex, κ < 0 = concave.
            # On a concave surface (κ < 0) the sphere sinks deeper than
            # nominal because the surface curves toward the sphere.
            # The extra depth at the edge of the dimple ≈ r_lat² · |κ| / 2.
            # We subtract this from the nominal depth to keep actual max
            # depth approximately equal to sphere_depth on any curvature.
            kappa = point_curvatures[i]
            r_lat = np.sqrt(max(0, 2 * radius * sphere_depth - sphere_depth**2))
            curvature_extra = r_lat * r_lat * abs(kappa) / 2.0
            if kappa < 0:
                # Concave: reduce depth to compensate for the extra cut
                effective_depth = sphere_depth - curvature_extra
            else:
                # Convex: surface curves away, sphere cuts shallower than
                # nominal — we can add a bit but cap it conservatively
                effective_depth = sphere_depth + curvature_extra * 0.5

            # Clamp depth to a safe range
            effective_depth = max(sphere_depth * 0.3, min(sphere_depth, effective_depth))
            effective_depth = min(effective_depth, radius)

            # Edge margin check: reject if sphere cap would extend past
            # the target region boundary
            lateral_r = np.sqrt(max(0, 2 * radius * effective_depth - effective_depth**2))
            if boundary_tree is not None:
                dist_to_edge, _ = boundary_tree.query(points[i])
                if dist_to_edge < lateral_r:
                    edge_rejected += 1
                    continue

            # Thin-wall check: cast ray inward, reject if the wall is
            # thinner than the sphere depth (would punch through)
            inward = -outward_normals[i]
            # Start ray slightly inside the surface to avoid self-intersection
            ray_origin = points[i] + inward * 0.01
            locations, _, _ = ray_caster.ray.intersects_location(
                ray_origins=[ray_origin],
                ray_directions=[inward],
            )
            if len(locations) > 0:
                # Distance to first intersection = wall thickness
                wall_dists = np.linalg.norm(locations - ray_origin, axis=1)
                wall_thickness = wall_dists.min()
                if wall_thickness < effective_depth * 3.0:
                    thin_rejected += 1
                    continue

            # Offset centre outward so only a cap of `effective_depth` intersects
            offset = radius - effective_depth
            if offset < 0:
                offset = 0

            centre = points[i] + outward_normals[i] * offset
            sphere_specs.append((centre, radius))

        # Poisson-disk-like rejection: remove spheres whose centres are
        # too close together (< base_radius * 0.8)
        pre_poisson = len(sphere_specs)
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

        poisson_rejected = pre_poisson - len(sphere_specs)
        print(f"  Sphere placement: {len(points)} sampled, "
              f"{edge_rejected} edge-rejected, {thin_rejected} thin-wall-rejected, "
              f"{poisson_rejected} Poisson-rejected → {len(sphere_specs)} kept")

        return sphere_specs

    @staticmethod
    def _get_submesh_boundary_points(submesh: trimesh.Trimesh, num_samples: int = 500) -> np.ndarray:
        """Return points along the boundary edges of a submesh.

        Boundary edges are edges that belong to only one triangle
        (i.e. the open boundary of the submesh).  We sample points
        along these edges so we can quickly measure distance from any
        surface point to the nearest boundary.
        """
        # facets_unique_edges: for each face, its 3 edge indices
        # edges_unique: the actual vertex-pair for each edge index
        edges = submesh.edges_sorted
        # Count how many faces reference each edge
        edge_tuples = [tuple(e) for e in edges]
        from collections import Counter
        edge_counts = Counter(edge_tuples)
        boundary_edges = np.array([list(e) for e, c in edge_counts.items() if c == 1])

        if len(boundary_edges) == 0:
            return np.zeros((0, 3))

        # Sample points along boundary edges
        v0 = submesh.vertices[boundary_edges[:, 0]]
        v1 = submesh.vertices[boundary_edges[:, 1]]

        # Distribute samples proportional to edge length
        edge_lengths = np.linalg.norm(v1 - v0, axis=1)
        total_length = edge_lengths.sum()
        if total_length < 1e-12:
            return np.zeros((0, 3))

        all_pts = []
        for i in range(len(boundary_edges)):
            n_samples = max(1, int(num_samples * edge_lengths[i] / total_length))
            for t in np.linspace(0, 1, n_samples, endpoint=False):
                all_pts.append(v0[i] + t * (v1[i] - v0[i]))

        return np.array(all_pts) if all_pts else np.zeros((0, 3))

    @staticmethod
    def _compute_point_curvatures(
        full_mesh: trimesh.Trimesh,
        submesh: trimesh.Trimesh,
        points: np.ndarray,
        face_ids: np.ndarray,
    ) -> np.ndarray:
        """Estimate mean curvature at each sampled surface point.

        Returns an array of signed mean curvatures (per point), where
        positive = concave (dimple appears bigger) and negative = convex.
        The sign convention matches outward-normal: concave surfaces have
        normals pointing away from the centre of curvature.

        Uses the discrete mean curvature measure (angle-deficit method)
        on the full mesh, then interpolates per-vertex values to each
        sample point via barycentric weights on the submesh triangle.
        """
        n_pts = len(points)
        curvatures = np.zeros(n_pts)

        try:
            # Discrete mean curvature per vertex (trimesh built-in)
            # The radius parameter defines the integration ball size;
            # use ~2× average edge length for a good local estimate.
            avg_edge = np.mean(full_mesh.edges_unique_length)
            curv_radius = avg_edge * 2.0
            vertex_mc = trimesh.curvature.discrete_mean_curvature_measure(
                full_mesh, full_mesh.vertices, radius=curv_radius
            )
            # Normalize: the measure is an integral over a ball of radius r,
            # divide by π·r² to get curvature in 1/mm units.
            ball_area = np.pi * curv_radius**2
            vertex_mc = vertex_mc / ball_area

            # Map full-mesh vertex curvatures to submesh vertices.
            # submesh shares vertex positions but may have different indices.
            from scipy.spatial import cKDTree
            full_tree = cKDTree(full_mesh.vertices)
            _, v_map = full_tree.query(submesh.vertices)
            sub_vertex_mc = vertex_mc[v_map]

            # Interpolate to each sample point using barycentric coords
            # on the submesh triangle it was sampled from.
            for i in range(n_pts):
                fid = face_ids[i]
                tri_verts = submesh.faces[fid]  # 3 vertex indices
                v0, v1, v2 = submesh.vertices[tri_verts]
                # Barycentric coordinates via area method
                p = points[i]
                area_full = np.linalg.norm(np.cross(v1 - v0, v2 - v0))
                if area_full < 1e-15:
                    curvatures[i] = 0.0
                    continue
                w0 = np.linalg.norm(np.cross(v1 - p, v2 - p)) / area_full
                w1 = np.linalg.norm(np.cross(v2 - p, v0 - p)) / area_full
                w2 = 1.0 - w0 - w1
                curvatures[i] = (
                    w0 * sub_vertex_mc[tri_verts[0]]
                    + w1 * sub_vertex_mc[tri_verts[1]]
                    + w2 * sub_vertex_mc[tri_verts[2]]
                )
        except Exception as e:
            # If curvature computation fails, return zeros (no compensation)
            print(f"  Warning: curvature computation failed ({e}), skipping compensation")

        return curvatures

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
