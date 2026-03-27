"""Convert STEP shapes to mesh with configurable resolution."""
from typing import Optional
import numpy as np
import trimesh

try:
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    HAS_OCP = True
except ImportError:
    HAS_OCP = False


class STEPMeshConverter:
    """Converts STEP shapes to mesh format using CadQuery/OCP."""

    @staticmethod
    def _extract_triangulation_nodes(triangulation):
        nodes_attr = getattr(triangulation, "Nodes", None)
        if nodes_attr is not None:
            return nodes_attr() if callable(nodes_attr) else nodes_attr

        if hasattr(triangulation, "NbNodes") and hasattr(triangulation, "Node"):
            count = triangulation.NbNodes()
            return [triangulation.Node(i) for i in range(1, count + 1)]

        return None

    @staticmethod
    def _extract_triangulation_triangles(triangulation):
        triangles_attr = getattr(triangulation, "Triangles", None)
        if triangles_attr is not None:
            return triangles_attr() if callable(triangles_attr) else triangles_attr

        if hasattr(triangulation, "NbTriangles") and hasattr(triangulation, "Triangle"):
            count = triangulation.NbTriangles()
            return [triangulation.Triangle(i) for i in range(1, count + 1)]

        return None

    @staticmethod
    def shape_to_mesh_with_face_map(
        shape,
        deflection: float = 0.1,
    ) -> Optional[tuple[trimesh.Trimesh, list[int]]]:
        """
        Convert a CadQuery/OCP shape to a trimesh.Trimesh object and
        return a mapping from mesh face index to STEP face index.

        Args:
            shape: OCP TopoDS_Shape object
            deflection: Mesh deflection (lower = finer mesh)

        Returns:
            (trimesh.Trimesh, face_map) or None if conversion fails
        """
        if not HAS_OCP:
            print("❌ cadquery-ocp libraries not available")
            return None

        try:
            from OCP.BRep import BRep_Tool
            from OCP.TopExp import TopExp_Explorer
            from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED
            from OCP.TopLoc import TopLoc_Location
            from OCP.TopoDS import TopoDS
        except ImportError as e:
            print(f"❌ OCP import failed: {e}")
            return None

        try:
            deflection = max(0.001, min(1.0, deflection))
            print(f"Converting STEP shape to mesh (deflection={deflection})...")

            mesh_maker = BRepMesh_IncrementalMesh(shape, float(deflection), False, 0.5)
            mesh_maker.Perform()

            if not mesh_maker.IsDone():
                print("❌ Mesh creation failed")
                return None

            vertices = []
            faces = []
            face_map: list[int] = []

            explorer = TopExp_Explorer(shape, TopAbs_FACE)
            face_count = 0
            triangulated_count = 0

            while explorer.More():
                face = TopoDS.Face_s(explorer.Current())

                location = TopLoc_Location()
                triangulation = None
                try:
                    triangulation = BRep_Tool.Triangulation_s(face, location)
                except Exception:
                    triangulation = None

                if triangulation is not None:
                    triangulated_count += 1
                    face_triangles = STEPMeshConverter._extract_triangulation_triangles(
                        triangulation
                    )
                    face_nodes = STEPMeshConverter._extract_triangulation_nodes(
                        triangulation
                    )

                    if face_triangles is None or face_nodes is None:
                        explorer.Next()
                        face_count += 1
                        continue

                    trsf = None
                    if not location.IsIdentity():
                        trsf = location.Transformation()

                    face_vertex_offset = len(vertices)
                    if hasattr(face_nodes, "Length") and hasattr(face_nodes, "Value"):
                        node_count = face_nodes.Length()
                        node_get = face_nodes.Value
                    else:
                        node_count = len(face_nodes)
                        def node_get(idx, _nodes=face_nodes):
                            return _nodes[idx - 1]

                    for i in range(1, node_count + 1):
                        node = node_get(i)
                        if trsf is not None:
                            node = node.Transformed(trsf)
                        vertices.append([node.X(), node.Y(), node.Z()])

                    if hasattr(face_triangles, "Length") and hasattr(face_triangles, "Value"):
                        tri_count = face_triangles.Length()
                        tri_get = face_triangles.Value
                    else:
                        tri_count = len(face_triangles)
                        def tri_get(idx, _tris=face_triangles):
                            return _tris[idx - 1]

                    for i in range(1, tri_count + 1):
                        triangle = tri_get(i)
                        if hasattr(triangle, "Value"):
                            v1 = triangle.Value(1) - 1 + face_vertex_offset
                            v2 = triangle.Value(2) - 1 + face_vertex_offset
                            v3 = triangle.Value(3) - 1 + face_vertex_offset
                        else:
                            v1, v2, v3 = triangle.Get()
                            v1 = v1 - 1 + face_vertex_offset
                            v2 = v2 - 1 + face_vertex_offset
                            v3 = v3 - 1 + face_vertex_offset
                        # Flip winding for reversed faces to preserve
                        # outward-facing normals
                        if face.Orientation() == TopAbs_REVERSED:
                            v2, v3 = v3, v2
                        faces.append([v1, v2, v3])
                        face_map.append(face_count)

                explorer.Next()
                face_count += 1

            print(
                f"  Explored {face_count} faces, triangulated {triangulated_count}, "
                f"extracted {len(faces)} triangles"
            )

            if not vertices or not faces:
                print("❌ No mesh data extracted")
                return None

            mesh = trimesh.Trimesh(
                vertices=np.array(vertices),
                faces=np.array(faces),
                process=False,
            )

            print(f"✓ Converted to mesh: {len(vertices)} vertices, {len(faces)} faces")
            return mesh, face_map

        except Exception as e:
            print(f"❌ Error converting shape to mesh: {e}")
            import traceback
            traceback.print_exc()
            return None

    @staticmethod
    def shape_to_mesh(shape, deflection: float = 0.1) -> Optional[trimesh.Trimesh]:
        """
        Convert a CadQuery/OCP shape to a trimesh.Trimesh object.

        Args:
            shape: OCP TopoDS_Shape object
            deflection: Mesh deflection (lower = finer mesh, higher = coarser)
                       Typical range: 0.01 (very fine) to 1.0 (coarse)
                       Default: 0.1 (good balance)

        Returns:
            trimesh.Trimesh object or None if conversion fails
        """
        result = STEPMeshConverter.shape_to_mesh_with_face_map(shape, deflection)
        if result is None:
            return None
        mesh, _face_map = result
        return mesh

    @staticmethod
    def mesh_to_shape(mesh: trimesh.Trimesh):
        """
        Convert a trimesh back to an OCP shape.

        Note: This creates a shell, not a solid. For 3D printing, STL is recommended.

        Args:
            mesh: trimesh.Trimesh object

        Returns:
            OCP TopoDS_Shell or None if conversion fails
        """
        if not HAS_OCP:
            print("❌ cadquery-ocp libraries not available")
            return None

        try:
            from OCP.BRepBuilderAPI import BRepBuilderAPI_MakePolygon, BRepBuilderAPI_MakeFace, BRepBuilderAPI_Sewing
            from OCP.gp import gp_Pnt

            print("Converting mesh back to STEP shape...")

            sewing = BRepBuilderAPI_Sewing()
            face_count = 0

            for face_indices in mesh.faces:
                polygon_maker = BRepBuilderAPI_MakePolygon()

                for idx in face_indices:
                    vertex = mesh.vertices[idx]
                    polygon_maker.Add(gp_Pnt(float(vertex[0]), float(vertex[1]), float(vertex[2])))

                polygon_maker.Close()
                if polygon_maker.IsDone():
                    # Create a face from the polygon wire
                    face_maker = BRepBuilderAPI_MakeFace(polygon_maker.Wire())
                    if face_maker.IsDone():
                        sewing.Add(face_maker.Face())
                        face_count += 1

            print(f"  Sewing {face_count} triangular faces...")
            sewing.Perform()
            shell = sewing.SewedShape()

            print(f"✓ Converted to STEP shape with {face_count} faces")
            return shell

        except Exception as e:
            print(f"❌ Error converting mesh to shape: {e}")
            return None

    @staticmethod
    def get_recommended_deflection(bounding_box_size: float) -> float:
        """
        Get recommended deflection value based on bounding box size.

        Args:
            bounding_box_size: Size of model bounding box

        Returns:
            Recommended deflection value
        """
        # Deflection is typically 0.1-1% of model size
        recommended = bounding_box_size * 0.001  # 0.1% of model size
        # Clamp to reasonable range
        return max(0.01, min(1.0, recommended))
