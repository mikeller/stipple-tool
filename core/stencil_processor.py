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
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from typing import Callable, Dict, List, Optional, Tuple

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
from OCP.TopoDS import TopoDS, TopoDS_Builder, TopoDS_Compound, TopoDS_Shape
from OCP.gp import gp_Pnt, gp_Vec

from core.color_analyzer import ColorAnalyzer
from core.step_loader import STEPLoader


class StencilStippleProcessor:
    """Stencil-based stippling using fuzzy boolean sphere cuts."""

    def __init__(self):
        self.color_analyzer = ColorAnalyzer()

    def _heal_shape_for_export(
        self,
        shape: TopoDS_Shape,
        emit_status: Callable[[str], None],
    ) -> TopoDS_Shape:
        """Heal and fix orientation of shape before export."""
        try:
            from OCP.BRepCheck import BRepCheck_Analyzer
            from OCP.ShapeFix import ShapeFix_Shape, ShapeFix_Solid

            # Always fix solid face orientation, even on topologically valid shapes.
            # BRepCheck_Analyzer reports "valid" for shapes that still have inverted
            # face normals (pointing inward), which FreeCAD renders as transparent
            # surfaces or convex protrusions instead of concave stipple dimples.
            try:
                sf = ShapeFix_Solid(TopoDS.Solid_s(shape))
                sf.Perform()
                oriented = sf.Solid()
                if oriented is not None and not oriented.IsNull():
                    shape = oriented
            except Exception:
                pass  # shape may be a compound; orientation fix is best-effort

            # If now valid, no further healing needed.
            pre_analyzer = BRepCheck_Analyzer(shape, True)
            if pre_analyzer.IsValid():
                return shape

            # Full healing pass for shapes still reporting errors.
            emit_status("Healing and fixing shape orientation...")
            healer = ShapeFix_Shape()
            healer.Init(shape)
            healer.SetPrecision(1e-5)
            healer.SetMaxTolerance(0.1)
            healer.SetMinTolerance(1e-6)
            healer.Perform()
            healed = healer.Shape()

            analyzer = BRepCheck_Analyzer(healed, True)
            if not analyzer.IsValid():
                emit_status("Warning: Shape still invalid after healing")
                emit_status("Healing did not improve validity; keeping original shape")
                return shape

            return healed
        except Exception as e:
            emit_status(f"Healing encountered issue: {e} (using original)")
            return shape

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

    def _largest_solid_component(
        self, shape: TopoDS_Shape
    ) -> Tuple[Optional[TopoDS_Shape], float, float, int]:
        """Return largest solid, its volume, total solids volume, and count."""
        try:
            largest_solid = None
            largest_volume = 0.0
            total_volume = 0.0
            solid_count = 0

            explorer = TopExp_Explorer(shape, TopAbs_SOLID)
            while explorer.More():
                solid = TopoDS.Solid_s(explorer.Current())
                props = GProp_GProps()
                BRepGProp.VolumeProperties_s(solid, props)
                vol = abs(props.Mass())
                total_volume += vol
                solid_count += 1
                if vol > largest_volume:
                    largest_volume = vol
                    largest_solid = solid
                explorer.Next()

            return largest_solid, largest_volume, total_volume, solid_count
        except Exception:
            return None, 0.0, 0.0, 0

    def _solid_contains_point(self, solid: TopoDS_Shape, point: gp_Pnt) -> bool:
        """Check whether a point lies inside a solid."""
        try:
            classifier = BRepClass3d_SolidClassifier(TopoDS.Solid_s(solid))
            classifier.Perform(point, 1e-4)
            return classifier.State() == TopAbs_IN
        except Exception:
            return False

    def _select_main_solid_component(
        self, shape: TopoDS_Shape, reference_point: Optional[gp_Pnt]
    ) -> Tuple[Optional[TopoDS_Shape], float, float, int]:
        """Select main solid component using a stable interior reference point.

        Returns (selected_solid, selected_volume, total_volume, solid_count).
        """
        try:
            solids: List[Tuple[TopoDS_Shape, float]] = []
            total_volume = 0.0
            explorer = TopExp_Explorer(shape, TopAbs_SOLID)
            while explorer.More():
                solid = TopoDS.Solid_s(explorer.Current())
                props = GProp_GProps()
                BRepGProp.VolumeProperties_s(solid, props)
                vol = abs(props.Mass())
                solids.append((solid, vol))
                total_volume += vol
                explorer.Next()

            if not solids:
                return None, 0.0, 0.0, 0

            if reference_point is not None:
                for solid, vol in solids:
                    if self._solid_contains_point(solid, reference_point):
                        return solid, vol, total_volume, len(solids)

            solid, vol = max(solids, key=lambda item: item[1])
            return solid, vol, total_volume, len(solids)
        except Exception:
            return None, 0.0, 0.0, 0

    def _extract_solids_only_shape(
        self,
        shape: TopoDS_Shape,
        reference_point: Optional[gp_Pnt],
    ) -> Optional[TopoDS_Shape]:
        """Drop detached sheets/shells by rebuilding the shape from solids only.

        If there is one solid, return it directly. If there are multiple solids,
        prefer the solid containing the reference interior point; otherwise use
        the largest solid. This is intentionally conservative because the target
        output is a single stippled part, not a mixed solid+sheet compound.
        """
        try:
            solids: List[TopoDS_Shape] = []
            explorer = TopExp_Explorer(shape, TopAbs_SOLID)
            while explorer.More():
                solids.append(TopoDS.Solid_s(explorer.Current()))
                explorer.Next()

            if not solids:
                return None

            if len(solids) == 1:
                return solids[0]

            selected_solid, _, _, _ = self._select_main_solid_component(
                shape, reference_point
            )
            if selected_solid is not None:
                return selected_solid

            return solids[0]
        except Exception:
            return None

    def _is_shape_valid(self, shape: TopoDS_Shape) -> bool:
        """Return True when OCC reports a topologically valid shape."""
        try:
            from OCP.BRepCheck import BRepCheck_Analyzer

            analyzer = BRepCheck_Analyzer(shape, True)
            return bool(analyzer.IsValid())
        except Exception:
            # If validation itself fails, do not block processing.
            return True

    def process_step_with_stencil_stippling(
        self,
        step_file: str,
        output_path: str,
        target_color: str,
        sphere_radius: float = 1.0,
        sphere_depth: float = 0.45,
        spheres_per_mm2: float = 0.34,
        strip_count: int = 6,
        overlap: float = 0.2,
        size_variation: bool = True,
        size_variation_mode: str = "gaussian",
        size_variation_sigma: float = 0.25,
        size_variation_min: float = 0.60,
        size_variation_max: float = 1.6,
        face_order_strategy: str = "largest_first",
        seed_spheres_per_face: int = 20,
        debug_log_path: Optional[str] = None,
        status_callback: Optional[Callable] = None,
        cancel_callback: Optional[Callable] = None,
    ) -> Optional[str]:
        status_log_lines: List[str] = []

        def emit_status(msg: str):
            timestamped = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
            if debug_log_path:
                status_log_lines.append(timestamped)
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
            target_face_ids: List[int] = []
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
                    target_face_ids.append(face_idx)
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

            # Phase 2: Batched boolean cuts per face.
            #
            # We batch all spheres for each face into an OCC compound
            # and cut them in one boolean operation, reducing ~5000 ops
            # to ~165 (one per face).  This avoids O(n²) complexity
            # accumulation that slows and eventually corrupts the shape.
            emit_status("Cutting spheres...")
            cut_start = time.time()
            current_shape = parent_solid if parent_solid else loader.shape
            last_valid_shape = current_shape
            applied = 0
            skipped = 0

            # Measure initial volume
            init_props = GProp_GProps()
            BRepGProp.VolumeProperties_s(current_shape, init_props)
            current_volume = abs(init_props.Mass())
            initial_volume = current_volume
            reference_inside_point = init_props.CentreOfMass()

            # Phase 2: Batched boolean cuts per face.
            #
            # Instead of cutting spheres one-by-one (which accumulates
            # topological complexity O(n²) and eventually corrupts the
            # shape), we batch all spheres for each face into a single
            # OCC compound and cut them in one boolean operation.
            # This reduces ~5000 boolean ops to ~165 (one per face),
            # dramatically improving speed and eliminating accumulated
            # corruption.  If a batch fails, it is split in half and
            # retried recursively down to a minimum batch size.

            face_stats: Dict[int, Dict[str, float]] = {}
            skip_reasons = Counter()
            total = sum(len(g) for g in face_groups)

            # Build face processing order. Prioritizing larger groups tends to
            # hit harder/larger regions earlier; a small seed pass can preserve
            # minimum coverage across all faces when runtime budget is tight.
            face_entries = []
            for group_index, group in enumerate(face_groups):
                face_id = (
                    target_face_ids[group_index]
                    if group_index < len(target_face_ids)
                    else group_index
                )
                face_entries.append((group_index, int(face_id), group))

            if face_order_strategy == "smallest_first":
                face_entries.sort(key=lambda item: len(item[2]))
            elif face_order_strategy == "original":
                pass
            else:
                face_entries.sort(key=lambda item: len(item[2]), reverse=True)

            seed_count = max(0, int(seed_spheres_per_face))
            if seed_count > 0:
                seeded_entries = []
                remainder_entries = []
                for group_index, face_id, group in face_entries:
                    if len(group) <= seed_count:
                        seeded_entries.append((group_index, face_id, group))
                        continue
                    seeded_entries.append((group_index, face_id, group[:seed_count]))
                    remainder_entries.append((group_index, face_id, group[seed_count:]))
                face_entries = seeded_entries + remainder_entries
                emit_status(
                    f"Face ordering: {face_order_strategy}, seed pass {seed_count} spheres/face"
                )
            else:
                emit_status(f"Face ordering: {face_order_strategy}")

            # Hard runtime budget to guarantee completion.
            max_cut_runtime_seconds = (
                24 * 3600 if spheres_per_mm2 >= 0.40 else 18 * 3600
            )

            # Initialize per-face stats
            for group_index, face_id, group in face_entries:
                if group_index not in face_stats:
                    face_stats[group_index] = {
                        "face_id": face_id,
                        "planned": len(group),
                        "applied": 0,
                        "skipped": 0,
                        "timeouts": 0,
                    }
                else:
                    face_stats[group_index]["planned"] += len(group)

            # Per-face: largest batch size that succeeded (seed pass → hint for remainder).
            face_seed_max_success: Dict[int, int] = {}
            # Map group_index → original face data for fresh-position retries.
            origin_face_data_map = {
                i: target_faces_data[i] for i in range(len(target_faces_data))
            }

            executor = ThreadPoolExecutor(max_workers=1)
            interrupted = False
            timed_out = 0

            def _make_compound(
                positions: List[Tuple[gp_Pnt, float, gp_Vec]],
            ) -> TopoDS_Compound:
                """Build a compound shape from a list of sphere positions."""
                builder = TopoDS_Builder()
                compound = TopoDS_Compound()
                builder.MakeCompound(compound)
                for centre, r, _ in positions:
                    builder.Add(
                        compound,
                        BRepPrimAPI_MakeSphere(centre, r).Shape(),
                    )
                return compound

            def _try_batch_cut(
                shape: TopoDS_Shape,
                compound: TopoDS_Compound,
                timeout_s: float,
            ) -> Optional[TopoDS_Shape]:
                """Cut compound from shape with timeout.  Returns result or None."""
                nonlocal executor
                try:
                    cut = BRepAlgoAPI_Cut()
                    a = TopTools_ListOfShape()
                    a.Append(shape)
                    cut.SetArguments(a)
                    t = TopTools_ListOfShape()
                    t.Append(compound)
                    cut.SetTools(t)
                    cut.SetFuzzyValue(0.01)

                    future = executor.submit(cut.Build)
                    try:
                        future.result(timeout=timeout_s)
                    except FutureTimeout:
                        executor.shutdown(wait=False)
                        executor = ThreadPoolExecutor(max_workers=1)
                        return None

                    if cut.IsDone() and not cut.Shape().IsNull():
                        return cut.Shape()
                except Exception:
                    pass
                return None

            def _validate_batch_result(
                result: Optional[TopoDS_Shape],
                prev_volume: float,
                batch_positions: List[Tuple[gp_Pnt, float, gp_Vec]],
            ) -> Tuple[Optional[TopoDS_Shape], float]:
                """Validate a batch cut result.

                Returns (accepted_shape, new_volume) or (None, 0) on rejection.
                """
                if result is None:
                    return None, 0.0

                vp = GProp_GProps()
                BRepGProp.VolumeProperties_s(result, vp)
                new_vol = abs(vp.Mass())

                # Shape destroyed
                if new_vol < 1e-3:
                    return None, 0.0
                # Volume increased — boolean corruption
                if new_vol > prev_volume * 1.001:
                    return None, 0.0

                removed = prev_volume - new_vol
                # Guard against unstable boolean outcomes that effectively add
                # material. A tiny negative value can be numerical noise.
                if removed < -0.05:
                    return None, 0.0
                # Guard against zero-intersection cuts that produce only a
                # circular seam curve (circle-only artifact) without actually
                # removing any cap material.
                if removed < 0.001:
                    return None, 0.0
                # Sanity: removal should not vastly exceed the total sphere
                # volume (with generous 3x headroom for partial overlaps and
                # thin-wall merging).
                total_sphere_vol = sum(
                    (4.0 / 3.0) * 3.14159 * r ** 3
                    for _, r, _ in batch_positions
                )
                if total_sphere_vol > 0 and removed > total_sphere_vol * 3.0:
                    return None, 0.0

                # Handle split solids
                solid_count = self._count_solid_components(result)
                if solid_count > 1:
                    largest, largest_vol, total_vol, _ = (
                        self._select_main_solid_component(
                            result, reference_inside_point
                        )
                    )
                    if (
                        largest is not None
                        and total_vol > 0
                        and (largest_vol / total_vol) >= 0.90
                    ):
                        return largest, largest_vol

                    # Solid split without a dominant component — reject the
                    # batch so it gets split or retried with fresh positions.
                    # clip_to_boundary was removed here: the BRepAlgoAPI_Common
                    # it used produced stray sheet geometry and impossible volume
                    # anomalies (see v6/v7 run notes).
                    return None, 0.0

                return result, new_vol

            def _try_fresh_cut(
                g_index: int,
                n: int,
                timeout_s: float,
                reason: str,
                max_retries: int = 3,
            ) -> int:
                """Retry a small batch with freshly sampled sphere positions.

                Returns the number of spheres successfully applied (0 on total failure).
                Each retry independently regenerates n random positions for the face
                so a bad placement cluster can be avoided.
                """
                nonlocal current_shape, current_volume, last_valid_shape
                face_data = origin_face_data_map.get(g_index)
                if face_data is None:
                    return 0
                for attempt in range(max_retries):
                    try:
                        fresh_groups = self._generate_sphere_positions_on_faces(
                            [face_data],
                            n,
                            sphere_radius,
                            sphere_depth,
                            size_variation,
                            size_variation_mode,
                            size_variation_sigma,
                            size_variation_min,
                            size_variation_max,
                            parent_solid,
                        )
                    except Exception:
                        continue
                    if not fresh_groups or not fresh_groups[0]:
                        continue
                    fresh_batch = fresh_groups[0][:n]
                    fresh_compound = _make_compound(fresh_batch)
                    fresh_result = _try_batch_cut(
                        current_shape, fresh_compound, timeout_s
                    )
                    if fresh_result is None:
                        continue
                    fv_shape, fv_vol = _validate_batch_result(
                        fresh_result, current_volume, fresh_batch
                    )
                    if fv_shape is None:
                        continue
                    removed_f = current_volume - fv_vol
                    current_shape = fv_shape
                    current_volume = fv_vol
                    if self._count_solid_components(current_shape) >= 1:
                        last_valid_shape = current_shape
                    emit_status(
                        f"    Fresh-retry {attempt + 1}/{max_retries}: "
                        f"{len(fresh_batch)} spheres ok "
                        f"(removed {removed_f:.2f} mm³, was {reason})"
                    )
                    return len(fresh_batch)
                return 0

            # Minimum batch size below which we stop splitting and switch to
            # fresh-position retries. Keeping this at 5 avoids deep split cascades.
            min_batch_size = 5

            # Per-face guardrails to prevent pathological runs from appearing hung.
            max_face_runtime_seconds = 900.0
            max_face_consecutive_failures = 40

            try:
                for group_index, face_id, group in face_entries:
                    check_cancel()

                    elapsed = time.time() - cut_start
                    if elapsed >= max_cut_runtime_seconds:
                        emit_status(
                            f"  Runtime budget reached ({elapsed:.0f}s) "
                            f"— stopping at {applied} spheres"
                        )
                        break

                    if not group:
                        continue

                    face_count = len(group)
                    face_start = time.time()

                    emit_status(
                        f"  Face {face_id}: cutting {face_count} spheres "
                        f"as batch..."
                    )

                    # Queue-based adaptive splitting.
                    # If this face had a successful batch size in the seed pass,
                    # pre-chunk the remainder at that size to skip the expensive
                    # halving cascade that would otherwise start at the full group.
                    _hint = face_seed_max_success.get(group_index, len(group))
                    if 0 < _hint < len(group):
                        queue = [
                            group[i : i + _hint]
                            for i in range(0, len(group), _hint)
                        ]
                        # Avoid tiny trailing chunks (e.g. 2-4) from pre-chunking,
                        # which correlate with circle-only/degenerate local artifacts.
                        # Merge remainder into previous chunk so downstream logic
                        # only sees robust batch sizes or min-batch fresh retries.
                        if len(queue) > 1 and len(queue[-1]) < min_batch_size:
                            queue[-2].extend(queue[-1])
                            queue.pop()
                    else:
                        queue = [group]
                    face_applied = 0
                    face_timed_out = 0
                    face_consecutive_failures = 0

                    while queue:
                        check_cancel()

                        elapsed = time.time() - cut_start
                        if elapsed >= max_cut_runtime_seconds:
                            break

                        face_elapsed = time.time() - face_start
                        if face_elapsed >= max_face_runtime_seconds:
                            skip_reasons["face_runtime_guard"] += len(queue)
                            emit_status(
                                f"    Face {face_id}: runtime guard hit "
                                f"({face_elapsed:.0f}s) - skipping remaining batches"
                            )
                            break

                        if face_consecutive_failures >= max_face_consecutive_failures:
                            skip_reasons["face_failure_guard"] += len(queue)
                            emit_status(
                                f"    Face {face_id}: failure guard hit "
                                f"({face_consecutive_failures} consecutive) - "
                                f"skipping remaining batches"
                            )
                            break

                        batch = queue.pop(0)
                        if not batch:
                            continue

                        compound = _make_compound(batch)
                        batch_timeout = max(120.0, len(batch) * 2.0)

                        t0 = time.time()
                        result = _try_batch_cut(
                            current_shape, compound, batch_timeout
                        )
                        dt = time.time() - t0

                        if result is None:
                            # Timed out or failed
                            face_timed_out += len(batch)
                            face_consecutive_failures += 1
                            if len(batch) >= 2 * min_batch_size:
                                mid = len(batch) // 2
                                queue.append(batch[:mid])
                                queue.append(batch[mid:])
                                emit_status(
                                    f"    Batch of {len(batch)} failed/timed-out "
                                    f"({dt:.1f}s) — splitting"
                                )
                            else:
                                placed = _try_fresh_cut(
                                    group_index, len(batch), batch_timeout,
                                    "timed-out",
                                )
                                if placed > 0:
                                    face_applied += placed
                                    face_consecutive_failures = 0
                                else:
                                    skip_reasons["batch_failed"] += len(batch)
                            continue

                        valid_shape, new_vol = _validate_batch_result(
                            result, current_volume, batch
                        )

                        if valid_shape is None:
                            # Validation failed — split and retry
                            face_consecutive_failures += 1
                            if len(batch) >= 2 * min_batch_size:
                                mid = len(batch) // 2
                                queue.append(batch[:mid])
                                queue.append(batch[mid:])
                                emit_status(
                                    f"    Batch of {len(batch)} failed validation "
                                    f"({dt:.1f}s) — splitting"
                                )
                            else:
                                placed = _try_fresh_cut(
                                    group_index, len(batch), batch_timeout,
                                    "validation",
                                )
                                if placed > 0:
                                    face_applied += placed
                                    face_consecutive_failures = 0
                                else:
                                    skip_reasons["batch_validation"] += len(batch)
                            continue

                        # Batch accepted
                        removed = current_volume - new_vol
                        current_shape = valid_shape
                        current_volume = new_vol
                        face_applied += len(batch)
                        face_consecutive_failures = 0
                        # Record largest success for this face (used as pre-chunk hint).
                        if len(batch) > face_seed_max_success.get(group_index, 0):
                            face_seed_max_success[group_index] = len(batch)
                        emit_status(
                            f"    Batch of {len(batch)} ok ({dt:.1f}s, "
                            f"removed {removed:.2f} mm³)"
                        )

                    applied += face_applied
                    face_skipped = face_count - face_applied
                    skipped += face_skipped
                    face_stats[group_index]["applied"] += face_applied
                    face_stats[group_index]["skipped"] += face_skipped
                    face_stats[group_index]["timeouts"] += face_timed_out
                    timed_out += face_timed_out

                    elapsed = time.time() - cut_start
                    emit_status(
                        f"  Face {face_id}: {face_applied}/{face_count} applied "
                        f"[{elapsed:.0f}s elapsed, {applied} total]"
                    )

                    # Persist a known-good checkpoint for final export fallback.
                    if face_applied > 0 and self._count_solid_components(current_shape) >= 1:
                        last_valid_shape = current_shape

            except KeyboardInterrupt:
                interrupted = True
                emit_status(
                    "  Interrupted by user — finalizing partial result"
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
            if interrupted:
                emit_status("Run ended early due to user interrupt.")
            emit_status(
                f"Volume: {initial_volume:.1f} → {final_volume:.1f} mm³ "
                f"(removed {removed:.1f} mm³, "
                f"{removed/initial_volume*100:.2f}%)"
            )

            if debug_log_path:
                coverage_lines = []
                for stats in face_stats.values():
                    planned = int(stats["planned"])
                    applied_face = int(stats["applied"])
                    skipped_face = int(stats["skipped"])
                    coverage_pct = (
                        (applied_face / planned) * 100.0 if planned > 0 else 0.0
                    )
                    coverage_lines.append(
                        (
                            coverage_pct,
                            f"face={int(stats['face_id'])} "
                            f"planned={planned} applied={applied_face} "
                            f"skipped={skipped_face} "
                            f"timeouts={int(stats['timeouts'])} "
                            f"coverage={coverage_pct:.1f}%"
                        )
                    )

                low_coverage = [line for cov, line in coverage_lines if cov < 50.0]
                zero_coverage = [line for cov, line in coverage_lines if cov == 0.0]

                summary_lines = [
                    "",
                    "===== DEBUG SUMMARY =====",
                    f"input={step_file}",
                    f"output={output_path}",
                    f"target_faces={len(face_groups)}",
                    f"total_spheres={total}",
                    f"applied={applied}",
                    f"skipped={skipped}",
                    f"timed_out={timed_out}",
                    f"cut_time_seconds={cut_time:.1f}",
                    f"volume_removed_percent={removed/initial_volume*100:.4f}",
                    "",
                    "Top skip reasons:",
                ]
                for reason, count in skip_reasons.most_common(15):
                    summary_lines.append(f"- {reason}: {count}")

                summary_lines.append("")
                summary_lines.append(
                    f"Faces with zero coverage ({len(zero_coverage)}):"
                )
                summary_lines.extend(f"- {line}" for line in zero_coverage[:50])

                summary_lines.append("")
                summary_lines.append(
                    f"Faces with coverage <50% ({len(low_coverage)}):"
                )
                summary_lines.extend(f"- {line}" for line in low_coverage[:100])

                summary_lines.append("")
                summary_lines.append("Per-face coverage:")
                for _, line in sorted(coverage_lines, key=lambda x: x[0]):
                    summary_lines.append(f"- {line}")

                with open(debug_log_path, "w", encoding="utf-8") as debug_file:
                    if status_log_lines:
                        debug_file.write("\n".join(status_log_lines))
                        debug_file.write("\n")
                    debug_file.write("\n".join(summary_lines))
                    debug_file.write("\n")

                emit_status(f"Debug log saved: {debug_log_path}")

            solids_only_shape = self._extract_solids_only_shape(
                current_shape,
                reference_inside_point,
            )
            if solids_only_shape is not None:
                current_shape = solids_only_shape

            current_shape = self._heal_shape_for_export(
                current_shape,
                emit_status,
            )

            if not self._is_shape_valid(current_shape):
                emit_status("Final shape invalid after healing, reverting to last valid checkpoint")
                if self._is_shape_valid(last_valid_shape):
                    current_shape = last_valid_shape

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

    def export_shape_to_stl(
        self,
        shape: TopoDS_Shape,
        output_path: str,
        linear_deflection: float = 0.01,
        angular_deflection: float = 0.5,
        status_callback: Optional[Callable] = None,
    ) -> bool:
        """Export a shape to STL using OCP's mesher for reliable complex geometry.
        
        Args:
            shape: The OCP shape to mesh
            output_path: Path to save STL file
            linear_deflection: Mesh accuracy (lower = finer, ~0.01 recommended)
            angular_deflection: Angular deflection in degrees (~0.5)
            status_callback: Optional callback for status messages
        
        Returns:
            True if successful, False otherwise
        """
        def emit_status(msg: str):
            if status_callback:
                status_callback(msg)
            else:
                print(msg)
        
        try:
            from OCP.BRepMesh import BRepMesh_IncrementalMesh
            from OCP.StlAPI import StlAPI_Writer
            
            emit_status(f"Meshing shape with deflection {linear_deflection}mm...")
            mesher = BRepMesh_IncrementalMesh(
                shape,
                linear_deflection,
                False,
                angular_deflection,
            )
            mesher.Perform()
            
            if not mesher.IsDone():
                emit_status("Meshing failed")
                return False
            
            emit_status(f"Writing STL: {output_path}")
            writer = StlAPI_Writer()
            writer.Write(shape, output_path)
            
            emit_status(f"✓ STL exported to: {output_path}")
            return True
            
        except Exception as e:
            emit_status(f"STL export failed: {e}")
            import traceback
            traceback.print_exc()
            return False

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

        # Allocate an exact sphere budget across faces using largest-remainder
        # apportionment so low-density runs don't bias toward early face order.
        face_count = len(faces_data)
        base_targets: List[int] = [0] * face_count
        remainders: List[Tuple[float, int]] = []

        if num_spheres >= face_count and face_count > 0:
            for idx in range(face_count):
                base_targets[idx] = 1
            extra_budget = num_spheres - face_count
        else:
            extra_budget = num_spheres

        assigned_extra = 0
        for idx, (_, _, area) in enumerate(faces_data):
            raw_extra = extra_budget * (area / total_area)
            extra_floor = int(raw_extra)
            base_targets[idx] += extra_floor
            remainders.append((raw_extra - extra_floor, idx))
            assigned_extra += extra_floor

        remaining_extra = extra_budget - assigned_extra
        if remaining_extra > 0:
            remainders.sort(key=lambda item: item[0], reverse=True)
            for _, face_idx in remainders[:remaining_extra]:
                base_targets[face_idx] += 1

        # Build classifier for inside/outside checks
        classifier = None
        if parent_solid is not None:
            try:
                classifier = BRepClass3d_SolidClassifier(parent_solid)
            except Exception:
                pass

        for face_idx, (face, _, _area) in enumerate(faces_data):
            face_spheres = base_targets[face_idx]
            if face_spheres <= 0:
                continue
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

                # Determine a fallback outward normal sign for this face.
                # A per-point sign is still resolved below for robustness.
                outward_sign_face = 1.0  # +normal means outward
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
                            outward_sign_face = -1.0
                        elif neg_inside and not pos_inside:
                            outward_sign_face = 1.0

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

                    # Resolve outward direction per-point to avoid relying on
                    # a single face-level normal orientation on complex faces.
                    outward_sign = outward_sign_face
                    if classifier is not None:
                        probe_dist = max(radius * 0.6, 0.35)
                        probe_pos = gp_Pnt(
                            pnt.X() + normal.X() * probe_dist,
                            pnt.Y() + normal.Y() * probe_dist,
                            pnt.Z() + normal.Z() * probe_dist,
                        )
                        probe_neg = gp_Pnt(
                            pnt.X() - normal.X() * probe_dist,
                            pnt.Y() - normal.Y() * probe_dist,
                            pnt.Z() - normal.Z() * probe_dist,
                        )
                        classifier.Perform(probe_pos, 1e-4)
                        pos_inside = classifier.State() == TopAbs_IN
                        classifier.Perform(probe_neg, 1e-4)
                        neg_inside = classifier.State() == TopAbs_IN

                        if pos_inside and not neg_inside:
                            outward_sign = -1.0
                        elif neg_inside and not pos_inside:
                            outward_sign = 1.0

                    center = gp_Pnt(
                        pnt.X() + normal.X() * offset * outward_sign,
                        pnt.Y() + normal.Y() * offset * outward_sign,
                        pnt.Z() + normal.Z() * offset * outward_sign,
                    )

                    # Ensure center is outside current solid; if not, try flipped.
                    if classifier is not None:
                        classifier.Perform(center, 1e-4)
                        if classifier.State() == TopAbs_IN:
                            center_alt = gp_Pnt(
                                pnt.X() - normal.X() * offset * outward_sign,
                                pnt.Y() - normal.Y() * offset * outward_sign,
                                pnt.Z() - normal.Z() * offset * outward_sign,
                            )
                            classifier.Perform(center_alt, 1e-4)
                            if classifier.State() == TopAbs_IN:
                                continue
                            center = center_alt
                            outward_sign *= -1.0

                    outward_dir = gp_Vec(normal.X(), normal.Y(), normal.Z())
                    outward_dir.Multiply(outward_sign)
                    
                    face_positions.append((center, radius, outward_dir))
                    total_placed += 1

                    if total_placed >= num_spheres:
                        break

            except Exception:
                continue

            if face_positions:
                face_groups.append(face_positions)

        return face_groups
