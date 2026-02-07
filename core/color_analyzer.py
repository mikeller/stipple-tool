"""Surface color analysis for STEP models using OCP XCAF tools."""
from typing import List, Tuple, Dict, Optional
from collections import defaultdict

from OCP.Quantity import Quantity_Color
from OCP.XCAFDoc import XCAFDoc_ColorType
from OCP.TDF import TDF_Label


class ColorAnalyzer:
    """Analyzes and detects surface colors in 3D models."""

    def __init__(self):
        self.color_groups = defaultdict(list)

    def extract_colors_from_model(self, model: Dict) -> Dict[str, List[int]]:
        """
        Extract colors from model surfaces.

        Args:
            model: Model dictionary from STEPLoader

        Returns:
            Dictionary mapping color names to lists of face indices
        """
        try:
            faces = model.get("faces", [])
            shape_tool = model.get("shape_tool")
            color_tool = model.get("color_tool")

            colors_dict = defaultdict(list)

            for idx, face in enumerate(faces):
                color_key = "default"

                color = self._find_face_color(face, shape_tool, color_tool)
                if color is not None:
                    color_key = self._color_to_hex(color)

                colors_dict[color_key].append(idx)

            self.color_groups = colors_dict
            return dict(colors_dict)

        except Exception as e:
            print(f"Error extracting colors: {e}")
            return {"default": list(range(len(model.get("faces", []))))}

    def _find_face_color(self, face, shape_tool, color_tool) -> Optional[Quantity_Color]:
        if shape_tool is None or color_tool is None:
            return None

        color = Quantity_Color()

        main_label = shape_tool.FindMainShape(face)
        if not main_label.IsNull():
            sub_label = TDF_Label()
            if shape_tool.FindSubShape(main_label, face, sub_label):
                if self._get_color_from_label(sub_label, color_tool, color):
                    return color

            if self._get_color_from_label(main_label, color_tool, color):
                return color

        if color_tool.GetInstanceColor(face, XCAFDoc_ColorType.XCAFDoc_ColorSurf, color):
            return color

        if color_tool.GetInstanceColor(face, XCAFDoc_ColorType.XCAFDoc_ColorGen, color):
            return color

        return None

    @staticmethod
    def _get_color_from_label(label: TDF_Label, color_tool, color: Quantity_Color) -> bool:
        if color_tool.GetColor_s(label, XCAFDoc_ColorType.XCAFDoc_ColorSurf, color):
            return True
        if color_tool.GetColor_s(label, XCAFDoc_ColorType.XCAFDoc_ColorGen, color):
            return True
        if color_tool.GetColor_s(label, XCAFDoc_ColorType.XCAFDoc_ColorCurv, color):
            return True
        return False

    def detect_distinct_colors(self, model: Dict) -> List[str]:
        """Detect distinct colors in the model."""
        color_dict = self.extract_colors_from_model(model)
        return list(color_dict.keys())

    def get_faces_by_color(self, color: str) -> List[int]:
        """Get all face indices with a specific color."""
        return self.color_groups.get(color, [])

    def get_all_color_groups(self) -> Dict[str, List[int]]:
        """Get all color groups."""
        return dict(self.color_groups)

    @staticmethod
    def _color_to_hex(color: Quantity_Color) -> str:
        r, g, b = color.Red(), color.Green(), color.Blue()
        return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"
