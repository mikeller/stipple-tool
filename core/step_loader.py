"""STEP file loader and converter using OCP (Open Cascade)."""
from pathlib import Path
from typing import Optional, Dict, List

from OCP.IFSelect import IFSelect_RetDone
from OCP.STEPCAFControl import STEPCAFControl_Reader, STEPCAFControl_Writer
from OCP.STEPControl import STEPControl_AsIs
from OCP.TDocStd import TDocStd_Document
from OCP.TCollection import TCollection_ExtendedString
from OCP.TDF import TDF_LabelSequence
from OCP.XCAFDoc import XCAFDoc_DocumentTool
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_FACE
from OCP.TopoDS import TopoDS_Shape, TopoDS


class STEPLoader:
    """Loads and manages STEP format files using OCP."""

    def __init__(self):
        self.file_path: Optional[Path] = None
        self.document: Optional[TDocStd_Document] = None
        self.shape: Optional[TopoDS_Shape] = None
        self.shape_tool = None
        self.color_tool = None
        self.faces: List = []

    def load(self, file_path: str) -> bool:
        """
        Load a STEP file with colors.

        Args:
            file_path: Path to the .step or .stp file

        Returns:
            True if successful, False otherwise
        """
        try:
            path = Path(file_path)

            if not path.exists():
                print(f"File not found: {path}")
                return False

            if path.suffix.lower() not in [".step", ".stp"]:
                print(f"Invalid file format. Expected .step or .stp, got {path.suffix}")
                return False

            self.document = TDocStd_Document(TCollection_ExtendedString("step"))
            reader = STEPCAFControl_Reader()

            status = reader.ReadFile(str(path))
            if status != IFSelect_RetDone:
                print("Failed to read STEP file")
                return False

            if not reader.Transfer(self.document):
                print("Failed to transfer STEP data")
                return False

            self.shape_tool = XCAFDoc_DocumentTool.ShapeTool_s(self.document.Main())
            self.color_tool = XCAFDoc_DocumentTool.ColorTool_s(self.document.Main())

            self.shape = self._get_root_shape()
            if self.shape.IsNull():
                print("No shapes found in STEP file")
                return False

            self.faces = self._extract_faces(self.shape)
            self.file_path = path

            print(f"Successfully loaded: {path}")
            print(f"Faces found: {len(self.faces)}")
            return True

        except Exception as e:
            print(f"Error loading STEP file: {e}")
            return False

    def _get_root_shape(self) -> TopoDS_Shape:
        """Get the root shape from the document."""
        seq = TDF_LabelSequence()
        self.shape_tool.GetFreeShapes(seq)

        if seq.Length() == 0:
            return TopoDS_Shape()

        root_label = seq.Value(1)
        return self.shape_tool.GetShape_s(root_label)

    def _extract_faces(self, shape: TopoDS_Shape) -> List:
        """Extract all faces from a shape."""
        faces = []
        explorer = TopExp_Explorer(shape, TopAbs_FACE)
        while explorer.More():
            faces.append(TopoDS.Face_s(explorer.Current()))
            explorer.Next()
        return faces

    def get_model(self) -> Optional[Dict]:
        """Get the loaded model data and tools."""
        if self.shape is None or self.document is None:
            return None

        return {
            "file_path": self.file_path,
            "document": self.document,
            "shape": self.shape,
            "shape_tool": self.shape_tool,
            "color_tool": self.color_tool,
            "faces": self.faces,
        }

    def get_faces(self) -> List:
        """Get all faces from the loaded model."""
        return self.faces

    def save_step(self, output_path: str, shape: TopoDS_Shape) -> bool:
        """
        Save a shape to a STEP file.

        Args:
            output_path: Path where to save the STEP file
            shape: Shape to export

        Returns:
            True if successful, False otherwise
        """
        try:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)

            doc = TDocStd_Document(TCollection_ExtendedString("export"))
            shape_tool = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
            shape_tool.AddShape(shape)

            writer = STEPCAFControl_Writer()
            writer.SetColorMode(True)
            writer.SetNameMode(True)
            writer.SetLayerMode(True)

            if not writer.Transfer(doc, STEPControl_AsIs):
                print("Failed to transfer model for export")
                return False

            status = writer.Write(str(path))
            if status != IFSelect_RetDone:
                print("Failed to write STEP file")
                return False

            print(f"Successfully saved: {path}")
            return True

        except Exception as e:
            print(f"Error saving STEP file: {e}")
            return False

    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        if self.shape is None:
            return {}

        return {
            "file": str(self.file_path),
            "faces": len(self.faces),
            "type": "STEP",
        }
