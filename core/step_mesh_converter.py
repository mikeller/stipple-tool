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
        if not HAS_OCP:
            print("❌ cadquery-ocp libraries not available")
            return None

        try:
            from OCP.BRepMesh import BRepMesh_IncrementalMesh
            from OCP.TopExp import TopExp_Explorer
            from OCP.TopAbs import TopAbs_FACE
            from OCP.TopLoc import TopLoc_Location
        except ImportError as e:
            print(f"❌ OCP import failed: {e}")
            return None

        try:
            # Clamp deflection to valid range
            deflection = max(0.001, min(1.0, deflection))
            print(f"Converting STEP shape to mesh (deflection={deflection})...")

            # Create mesh from shape
            mesh_maker = BRepMesh_IncrementalMesh(shape, float(deflection), False, 0.5)
            mesh_maker.Perform()

            if not mesh_maker.IsDone():
                print("❌ Mesh creation failed")
                return None

            # Extract vertices and faces
            vertices = []
            faces = []

            explorer = TopExp_Explorer(shape, TopAbs_FACE)
            
            face_count = 0
            triangulated_count = 0
            while explorer.More():
                face_count += 1
                face = explorer.Current()
                
                # Get triangulation from face - BRepMesh stores it in the face itself
                location = TopLoc_Location()
                triangulation = None
                
                # First try with location (Standard method)
                try:
                    # This is the correct way to get triangulation from a meshed shape
                    from OCP.Handle import Handle_Poly_Triangulation
                    triangulation = face.Triangulation(location)
                except Exception:
                    # Some faces may legitimately lack triangulation
                    pass

                if triangulation is not None:
                    triangulated_count += 1
                    face_triangles = triangulation.Triangles()
                    face_nodes = triangulation.Nodes()

                    # Precompute transformation from location, if any
                    trsf = None
                    if not location.IsIdentity():
                        trsf = location.Transformation()

                    # Add vertices for this face
                    face_vertex_offset = len(vertices)
                    for i in range(1, face_nodes.Length() + 1):
                        node = face_nodes.Value(i)
                        if trsf is not None:
                            node = node.Transformed(trsf)
                        vertices.append([node.X(), node.Y(), node.Z()])

                    # Add faces
                    for i in range(1, face_triangles.Length() + 1):
                        triangle = face_triangles.Value(i)
                        v1 = triangle.Value(1) - 1 + face_vertex_offset
                        v2 = triangle.Value(2) - 1 + face_vertex_offset
                        v3 = triangle.Value(3) - 1 + face_vertex_offset
                        faces.append([v1, v2, v3])

                explorer.Next()
            
            print(f"  Explored {face_count} faces, triangulated {triangulated_count}, extracted {len(faces)} triangles")

            if not vertices or not faces:
                print("❌ No mesh data extracted")
                return None

            # Create trimesh
            mesh = trimesh.Trimesh(
                vertices=np.array(vertices),
                faces=np.array(faces),
                process=False,
            )

            print(f"✓ Converted to mesh: {len(vertices)} vertices, {len(faces)} faces")
            return mesh

        except Exception as e:
            print(f"❌ Error converting shape to mesh: {e}")
            import traceback
            traceback.print_exc()
            return None

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
            from OCP.BRepBuilderAPI import BRepBuilderAPI_MakePolygon, BRepBuilderAPI_Sewing
            from OCP.gp import gp_Pnt

            print("Converting mesh back to STEP shape...")

            sewing = BRepBuilderAPI_Sewing()

            for face_indices in mesh.faces:
                polygon_maker = BRepBuilderAPI_MakePolygon()

                for idx in face_indices:
                    vertex = mesh.vertices[idx]
                    polygon_maker.Add(gp_Pnt(float(vertex[0]), float(vertex[1]), float(vertex[2])))

                polygon_maker.Close()
                if polygon_maker.IsDone():
                    sewing.Add(polygon_maker.Shape())

            sewing.Perform()
            shell = sewing.SewedShape()

            print(f"✓ Converted to STEP shape")
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
