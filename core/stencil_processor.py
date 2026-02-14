"""
Stencil-based stippling processor.

Strategy:
1. Identify target colored faces on the original shape.
2. Triangulate the shape so sphere positions lie on actual trimmed faces.
3. Generate sphere positions per face, proportional to area.
4. Interleave positions from all faces and cut individually.
   Each cut.Build() runs in a thread with a timeout to catch hangs.
5. Escalation detection stops early if cuts get too slow.
6. Save the stippled solid — partial coverage is saved if stopped early.

Fuzzy booleans (SetFuzzyValue) relax geometric tolerance so near-tangent
intersections are resolved reliably.
"""

import random
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from typing import Callable, List, Optional, Tuple

from OCP.BRepClass3d import BRepClass3d_SolidClassifier
from OCP.BRep import BRep_Tool
from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
from OCP.BRepGProp import BRepGProp
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.BRepPrimAPI import BRepPrimAPI_MakeSphere
from OCP.GProp import GProp_GProps
from OCP.TopAbs import TopAbs_FACE, TopAbs_IN, TopAbs_SOLID
from OCP.TopExp import TopExp_Explorer
from OCP.TopLoc import TopLoc_Location
from OCP.TopTools import TopTools_ListOfShape
from OCP.TopoDS import TopoDS, TopoDS_Shape
from OCP.gp import gp_Pnt, gp_Vec

from core.color_analyzer import ColorAnalyzer
from core.step_loader import STEPLoader


class StencilStippleProcessor:
    """Stencil-based stippling using fuzzy boolean sphere cuts."""

    def __init__(self):
        self.color_analyzer = ColorAnalyzer()

    def _count_solid_components(self, shape: TopoDS_Shape) -> int:
        """Count the number of disconnected SOLID components in a shape.
        
        Returns the count of solid components. If the shape is disconnected
        (multiple solids), we should reject it to avoid isolated "filled" geometry.
        """
        try:
            count = 0
            explorer = TopExp_Explorer(shape, TopAbs_SOLID)
            while explorer.More():
                count += 1
                explorer.Next()
            return count
        except Exception:
            return 0

    def process_step_with_stencil_stippling(
        self,
        step_file: str,
        output_path: str,
        target_color: str,
        sphere_radius: float = 1.0,
        sphere_depth: float = 0.5,
        spheres_per_mm2: float = 0.12,
        strip_count: int = 6,
        overlap: float = 0.2,
        batch_size: int = 10,
        size_variation: bool = True,
        size_variation_mode: str = "uniform",
        size_variation_sigma: float = 0.2,
        size_variation_min: float = 0.7,
        size_variation_max: float = 1.3,
        status_callback: Optional[Callable] = None,
        cancel_callback: Optional[Callable] = None,
    ) -> Optional[str]:
        def emit_status(msg: str):
            if status_callback:
                status_callback(msg)
            else:
                print(msg)

        def check_cancel():
            if cancel_callback:
                cancel_callback()

        try:
            emit_status("=== STENCIL STIPPLING ===")

            loader = STEPLoader()
            if not loader.load(step_file):
                emit_status("Failed to load STEP file")
                return None

            step_model = loader.get_model()
            colors = self.color_analyzer.extract_colors_from_model(step_model)
            target_face_indices = colors.get(target_color, [])
            if not target_face_indices:
                emit_status(f"No surfaces found with color {target_color}")
                return None

            emit_status(f"Found {len(target_face_indices)} target faces")

            # Extract target face objects and their centroids
            target_faces_data = []  # [(face_obj, centroid, area), ...]
            explorer = TopExp_Explorer(loader.shape, TopAbs_FACE)
            face_idx = 0
            while explorer.More():
                if face_idx in target_face_indices:
                    face = TopoDS.Face_s(explorer.Current())
                    props = GProp_GProps()
                    BRepGProp.SurfaceProperties_s(face, props)
                    centroid = props.CentreOfMass()
                    area = props.Mass()
                    target_faces_data.append((face, (centroid.X(), centroid.Y(), centroid.Z()), area))
                explorer.Next()
                face_idx += 1

            emit_status(f"Extracted {len(target_faces_data)} target face geometries")

            # Triangulate the shape so _generate_sphere_positions_on_faces
            # can sample points from the mesh.  This guarantees every
            # sampled point lies on the actual trimmed face (unlike raw
            # UV sampling, which can produce points outside trim curves).
            emit_status("Triangulating shape...")
            mesh = BRepMesh_IncrementalMesh(loader.shape, 0.1, False, 0.5, True)
            mesh.Perform()

            # Extract parent solid for point-in-solid classification
            # (used to verify sphere centres are positioned inside the shape)
            parent_solid = None
            sol_exp = TopExp_Explorer(loader.shape, TopAbs_SOLID)
            if sol_exp.More():
                parent_solid = TopoDS.Solid_s(sol_exp.Current())

            # Phase 1: Generate sphere positions across ALL target faces
            # Each face gets spheres proportional to its area — no per-strip caps
            total_area = sum(area for _, _, area in target_faces_data)
            total_spheres_target = max(3, int(total_area * spheres_per_mm2))

            emit_status(f"Total target area: {total_area:.1f} mm², target spheres: {total_spheres_target}")

            sphere_positions = self._generate_sphere_positions_on_faces(
                target_faces_data,
                total_spheres_target,
                sphere_radius,
                sphere_depth,
                size_variation,
                size_variation_mode,
                size_variation_sigma,
                size_variation_min,
                size_variation_max,
                parent_solid,
            )

            # sphere_positions is now a list of per-face groups
            face_groups = sphere_positions

            if not face_groups:
                emit_status("No sphere positions generated")
                return None

            total = sum(len(g) for g in face_groups)
            emit_status(
                f"Generated {total} sphere positions across "
                f"{len(face_groups)} faces"
            )

            # Phase 2: Individual fuzzy boolean cuts, face by face.
            #
            # Process spheres in face order (not shuffled) so every
            # face gets at least some coverage before the shape becomes
            # too complex.  Each cut.Build() runs in a thread with a
            # timeout.  An escalation detector stops processing when
            # average cut time exceeds a threshold — this prevents the
            # tool from hanging indefinitely as accumulated complexity
            # makes each subsequent boolean operation slower.
            emit_status("Cutting spheres...")
            cut_start = time.time()
            current_shape = parent_solid if parent_solid else loader.shape
            applied = 0
            skipped = 0
            timed_out = 0

            # Measure initial volume
            init_props = GProp_GProps()
            BRepGProp.VolumeProperties_s(current_shape, init_props)
            current_volume = abs(init_props.Mass())
            initial_volume = current_volume

            # Flatten face_groups into ordered list, interleaving faces
            # so early cuts are distributed across all faces
            max_face_len = max(len(g) for g in face_groups)
            ordered_positions: List[Tuple[gp_Pnt, float, gp_Vec]] = []
            for slot in range(max_face_len):
                for group in face_groups:
                    if slot < len(group):
                        ordered_positions.append(group[slot])

            total = len(ordered_positions)
            report_interval = max(1, total // 20)

            # Per-cut timeout
            cut_timeout = 15.0  # seconds
            # Escalation detection
            recent_times: List[float] = []
            max_recent = 10
            escalation_threshold = 10.0  # Increased for deeper cuts (0.5mm depth)
            consecutive_timeouts = 0
            max_consecutive_timeouts = 5
            # Consecutive validation failures — stop when no cuts succeed
            consecutive_validation_fails = 0
            max_consecutive_validation_fails = 150  # Try harder
            # Adaptive back-off for repeated failures
            global_scale = 1.0
            min_global_scale = 0.4  # Back off more aggressively
            backoff_trigger = 40  # More permissive before backing off
            backoff_factor = 0.85  # Smaller steps preserve more geometry

            if size_variation and size_variation_mode == "gaussian":
                max_sphere_radius = sphere_radius * size_variation_max
            else:
                max_sphere_radius = sphere_radius * (1.4 if size_variation else 1.0)
            max_single_sphere_vol = (
                (4.0 / 3.0) * 3.14159 * max_sphere_radius ** 3
            )

            executor = ThreadPoolExecutor(max_workers=1)

            try:
                for i, (centre, radius, outward_dir) in enumerate(ordered_positions):
                    check_cancel()

                    t0 = time.time()
                    try:
                        attempt_scales = (1.0, 0.85, 0.7)
                        success = False
                        timed_out_this_cut = False

                        for scale in attempt_scales:
                            attempt_radius = radius * global_scale * scale
                            if attempt_radius <= 0:
                                continue

                            sphere = BRepPrimAPI_MakeSphere(
                                centre, attempt_radius
                            ).Shape()

                            cut = BRepAlgoAPI_Cut()
                            args = TopTools_ListOfShape()
                            args.Append(current_shape)
                            cut.SetArguments(args)
                            tools = TopTools_ListOfShape()
                            tools.Append(sphere)
                            cut.SetTools(tools)
                            cut.SetFuzzyValue(0.01)

                            future = executor.submit(cut.Build)
                            try:
                                future.result(timeout=cut_timeout)
                            except FutureTimeout:
                                timed_out += 1
                                consecutive_timeouts += 1
                                dt = time.time() - t0
                                emit_status(
                                    f"  Sphere {i+1}: timed out ({dt:.1f}s)"
                                )
                                executor.shutdown(wait=False)
                                executor = ThreadPoolExecutor(max_workers=1)
                                timed_out_this_cut = True
                                if consecutive_timeouts >= max_consecutive_timeouts:
                                    emit_status(
                                        f"  {max_consecutive_timeouts} "
                                        f"consecutive timeouts — stopping"
                                    )
                                    break
                                break

                            consecutive_timeouts = 0

                            if cut.IsDone() and not cut.Shape().IsNull():
                                result = cut.Shape()

                                # Volume validation: ensure the cut didn't
                                # produce degenerate geometry
                                vp = GProp_GProps()
                                BRepGProp.VolumeProperties_s(result, vp)
                                new_vol = abs(vp.Mass())

                                if new_vol < 1e-3:
                                    continue
                                if new_vol > current_volume * 1.001:
                                    continue
                                removed_vol = current_volume - new_vol
                                if removed_vol <= 0:
                                    continue
                                min_cut_vol = (
                                    (4.0 / 3.0) * 3.14159
                                    * attempt_radius ** 3 * 0.002
                                )
                                if removed_vol < min_cut_vol:
                                    continue
                                if removed_vol > max_single_sphere_vol * 3:
                                    continue

                                # Reject if boolean cut created disconnected solids
                                # (isolated "filled" geometry around the cut area)
                                solid_count = self._count_solid_components(result)
                                if solid_count > 1:
                                    continue

                                # Strict validation: the outward sphere cap
                                # must be outside the resulting solid.
                                if outward_dir.Magnitude() > 1e-6:
                                    outward_dir.Normalize()
                                    probe = gp_Pnt(
                                        centre.X() + outward_dir.X() * attempt_radius,
                                        centre.Y() + outward_dir.Y() * attempt_radius,
                                        centre.Z() + outward_dir.Z() * attempt_radius,
                                    )
                                    try:
                                        post_classifier = BRepClass3d_SolidClassifier(result)
                                        post_classifier.Perform(probe, 1e-4)
                                        if post_classifier.State() == TopAbs_IN:
                                            continue
                                    except Exception:
                                        continue

                                current_shape = result
                                current_volume = new_vol
                                applied += 1
                                consecutive_validation_fails = 0
                                success = True
                                break

                        if timed_out_this_cut:
                            if consecutive_timeouts >= max_consecutive_timeouts:
                                break
                            continue

                        if not success:
                            skipped += 1
                            consecutive_validation_fails += 1
                    except Exception:
                        skipped += 1
                        consecutive_validation_fails += 1

                    if (consecutive_validation_fails
                            >= backoff_trigger
                            and global_scale > min_global_scale):
                        global_scale *= backoff_factor
                        if global_scale < min_global_scale:
                            global_scale = min_global_scale
                        emit_status(
                            f"  Back-off: reducing sphere scale to "
                            f"{global_scale:.2f}"
                        )
                        consecutive_validation_fails = 0
                    elif (consecutive_validation_fails
                            >= max_consecutive_validation_fails):
                        emit_status(
                            f"  {max_consecutive_validation_fails} "
                            f"consecutive failed cuts — shape too "
                            f"complex, stopping at {applied} spheres"
                        )
                        break

                    dt = time.time() - t0
                    recent_times.append(dt)
                    if len(recent_times) > max_recent:
                        recent_times.pop(0)

                    # Escalation detection
                    if (len(recent_times) == max_recent
                            and sum(recent_times) / max_recent
                            > escalation_threshold):
                        avg = sum(recent_times) / max_recent
                        emit_status(
                            f"  Cuts slowing (avg {avg:.1f}s/cut) — "
                            f"stopping at {applied} spheres"
                        )
                        break

                    # Progress
                    if (i + 1) % report_interval == 0 or i == total - 1:
                        elapsed = time.time() - cut_start
                        rate = (i + 1) / elapsed if elapsed > 0 else 0
                        eta = (total - i - 1) / rate if rate > 0 else 0
                        emit_status(
                            f"  Sphere {i+1}/{total} "
                            f"({applied} ok, {skipped} skip) "
                            f"[{elapsed:.0f}s, ~{eta:.0f}s left]"
                        )
            finally:
                executor.shutdown(wait=False)

            skipped = total - applied

            cut_time = time.time() - cut_start

            # Final volume report
            final_props = GProp_GProps()
            BRepGProp.VolumeProperties_s(current_shape, final_props)
            final_volume = abs(final_props.Mass())
            removed = initial_volume - final_volume

            emit_status(
                f"Stippling complete: {applied} spheres applied, "
                f"{skipped} skipped"
                + (f" ({timed_out} timed out)" if timed_out else "")
                + f" in {cut_time:.1f}s"
            )
            emit_status(
                f"Volume: {initial_volume:.1f} → {final_volume:.1f} mm³ "
                f"(removed {removed:.1f} mm³, "
                f"{removed/initial_volume*100:.2f}%)"
            )

            # Save final shape
            emit_status(f"Saving result: {output_path}")
            if not loader.save_step(output_path, current_shape):
                emit_status("Failed to write final STEP")
                return None

            emit_status("✓ Stencil stippling complete")
            return output_path

        except Exception as e:
            import traceback
            emit_status(f"Error during stencil stippling: {e}")
            traceback.print_exc()
            return None

    def _generate_sphere_positions_on_faces(
        self,
        faces_data: List[Tuple[TopoDS_Shape, Tuple[float, float, float], float]],
        num_spheres: int,
        base_radius: float,
        sphere_depth: float,
        size_variation: bool,
        size_variation_mode: str,
        size_variation_sigma: float,
        size_variation_min: float,
        size_variation_max: float,
        parent_solid: Optional[TopoDS_Shape] = None,
    ) -> List[List[Tuple[gp_Pnt, float, gp_Vec]]]:
        """Generate sphere positions distributed across faces by area.

        Returns a list of per-face groups, where each group is a list
        of (centre, radius) tuples for that face.  This grouping allows
        the caller to cut all spheres for one face in a single boolean
        operation.

        Uses face triangulation to sample points that are guaranteed to
        lie on the actual trimmed face (not outside it, as raw UV sampling
        can produce on faces with large untrimmed parameter domains).

        For each sampled point, probes along the triangle normal to
        determine the inward direction, then offsets the sphere centre
        to create an inward indentation.
        """
        face_groups: List[List[Tuple[gp_Pnt, float, gp_Vec]]] = []
        total_placed = 0
        total_area = sum(area for _, _, area in faces_data)
        if total_area <= 0:
            return face_groups

        # Build classifier for inside/outside checks
        classifier = None
        if parent_solid is not None:
            try:
                classifier = BRepClass3d_SolidClassifier(parent_solid)
            except Exception:
                pass

        for face, _, area in faces_data:
            # Allocate spheres proportionally to face area
            face_spheres = max(1, int(num_spheres * (area / total_area)))
            face_positions: List[Tuple[gp_Pnt, float, gp_Vec]] = []

            try:
                # Get the face triangulation (already meshed on the shape)
                loc = TopLoc_Location()
                tri = BRep_Tool.Triangulation_s(face, loc)
                if tri is None:
                    continue

                trsf = loc.Transformation()

                # Collect triangles with vertices and areas
                triangles = []  # [(p1, p2, p3, area), ...]
                total_tri_area = 0.0
                for t_idx in range(1, tri.NbTriangles() + 1):
                    tri_obj = tri.Triangle(t_idx)
                    i1, i2, i3 = tri_obj.Get()
                    p1 = tri.Node(i1)
                    p2 = tri.Node(i2)
                    p3 = tri.Node(i3)
                    p1.Transform(trsf)
                    p2.Transform(trsf)
                    p3.Transform(trsf)

                    v1 = gp_Vec(p1, p2)
                    v2 = gp_Vec(p1, p3)
                    tri_area = 0.5 * v1.Crossed(v2).Magnitude()
                    if tri_area > 1e-10:
                        triangles.append((p1, p2, p3, tri_area))
                        total_tri_area += tri_area

                if not triangles:
                    continue

                # Build cumulative area array for weighted sampling
                cum_areas = []
                cumulative = 0.0
                for _, _, _, t_area in triangles:
                    cumulative += t_area
                    cum_areas.append(cumulative)

                # Determine inward normal direction for this face ONCE
                # using the centroid of a representative triangle
                inward_sign = -1.0  # default
                if classifier is not None:
                    rep = triangles[len(triangles) // 2]
                    rp1, rp2, rp3, _ = rep
                    mid_pnt = gp_Pnt(
                        (rp1.X() + rp2.X() + rp3.X()) / 3,
                        (rp1.Y() + rp2.Y() + rp3.Y()) / 3,
                        (rp1.Z() + rp2.Z() + rp3.Z()) / 3,
                    )
                    probe_normal = gp_Vec(rp1, rp2).Crossed(gp_Vec(rp1, rp3))
                    if probe_normal.Magnitude() > 1e-6:
                        probe_normal.Normalize()
                        probe_dist = max(base_radius, 1.0)
                        probe_pos = gp_Pnt(
                            mid_pnt.X() + probe_normal.X() * probe_dist,
                            mid_pnt.Y() + probe_normal.Y() * probe_dist,
                            mid_pnt.Z() + probe_normal.Z() * probe_dist,
                        )
                        probe_neg = gp_Pnt(
                            mid_pnt.X() - probe_normal.X() * probe_dist,
                            mid_pnt.Y() - probe_normal.Y() * probe_dist,
                            mid_pnt.Z() - probe_normal.Z() * probe_dist,
                        )
                        classifier.Perform(probe_pos, 1e-4)
                        pos_inside = classifier.State() == TopAbs_IN
                        classifier.Perform(probe_neg, 1e-4)
                        neg_inside = classifier.State() == TopAbs_IN

                        if pos_inside and not neg_inside:
                            inward_sign = 1.0
                        elif neg_inside and not pos_inside:
                            inward_sign = -1.0

                for _ in range(face_spheres):
                    # Pick a random triangle weighted by area
                    r = random.random() * total_tri_area
                    t_idx = 0
                    for j, ca in enumerate(cum_areas):
                        if ca >= r:
                            t_idx = j
                            break

                    tp1, tp2, tp3, _ = triangles[t_idx]

                    # Random barycentric coordinates
                    s = random.random()
                    t_val = random.random()
                    if s + t_val > 1.0:
                        s = 1.0 - s
                        t_val = 1.0 - t_val
                    w = 1.0 - s - t_val

                    pnt = gp_Pnt(
                        w * tp1.X() + s * tp2.X() + t_val * tp3.X(),
                        w * tp1.Y() + s * tp2.Y() + t_val * tp3.Y(),
                        w * tp1.Z() + s * tp2.Z() + t_val * tp3.Z(),
                    )

                    # Normal from triangle edges
                    normal = gp_Vec(tp1, tp2).Crossed(gp_Vec(tp1, tp3))
                    if normal.Magnitude() > 1e-6:
                        normal.Normalize()
                    else:
                        continue

                    # Determine actual sphere radius
                    if size_variation:
                        if size_variation_mode == "gaussian":
                            scale = random.gauss(1.0, size_variation_sigma)
                            if scale < size_variation_min:
                                scale = size_variation_min
                            elif scale > size_variation_max:
                                scale = size_variation_max
                            radius = base_radius * scale
                        else:
                            radius = base_radius * (0.6 + random.random() * 0.8)
                    else:
                        radius = base_radius

                    # Scale depth slightly with sphere size (radius ratio).
                    # For larger spheres, use ~15% deeper cuts to improve coverage.
                    # This balances geometric complexity with visual density.
                    scale_factor = 1.0 + 0.15 * ((radius / base_radius) - 1.0)
                    effective_depth = sphere_depth * scale_factor
                    if effective_depth > sphere_depth * 1.5:
                        effective_depth = sphere_depth * 1.5
                    if effective_depth > radius:
                        effective_depth = radius

                    # Centre goes outward by (radius - effective_depth) so
                    # only a shallow cap intersects the solid.
                    offset = radius - effective_depth
                    if offset < 0:
                        offset = 0

                    center = gp_Pnt(
                        pnt.X() - inward_sign * normal.X() * offset,
                        pnt.Y() - inward_sign * normal.Y() * offset,
                        pnt.Z() - inward_sign * normal.Z() * offset,
                    )

                    # Ensure the center lands outside the solid.
                    if classifier is not None:
                        classifier.Perform(center, 1e-4)
                        if classifier.State() == TopAbs_IN:
                            center = gp_Pnt(
                                pnt.X() + inward_sign * normal.X() * offset,
                                pnt.Y() + inward_sign * normal.Y() * offset,
                                pnt.Z() + inward_sign * normal.Z() * offset,
                            )
                            classifier.Perform(center, 1e-4)
                            if classifier.State() == TopAbs_IN:
                                continue

                    outward_dir = gp_Vec(normal.X(), normal.Y(), normal.Z())
                    outward_dir.Multiply(-inward_sign)
                    face_positions.append((center, radius, outward_dir))
                    total_placed += 1

                    if total_placed >= num_spheres:
                        face_groups.append(face_positions)
                        return face_groups

            except Exception:
                continue

            if face_positions:
                face_groups.append(face_positions)

        return face_groups
