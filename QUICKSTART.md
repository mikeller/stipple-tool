# STEP Stippling Tool - Quick Start

## Installation

1. Create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage - GUI Mode

```bash
python main.py
```

This opens an interactive interface where you can:
- Load STEP files (.step or .stp)
- View and select surface colors
- Configure stippling parameters (sphere radius, depth, density)
- Apply stippling to selected surfaces
- Export as STL, 3MF, or OBJ

## Usage - CLI

```bash
python stipple_tool_cli.py model.step -o output.stl -c "#360200"
```

Parameters:
```bash
python stipple_tool_cli.py model.step \
  -o output.stl \
  -c "#360200" \
  --radius 1.4 \
  --depth 0.6 \
  --spheres-per-mm2 0.5
```

Run `python stipple_tool_cli.py --help` for all options.

## File Structure

```text
main.py              - Launch GUI application
stipple_tool_cli.py  - Command-line interface

core/
  step_loader.py               - STEP file I/O (OCP-based)
  color_analyzer.py            - Surface color detection
  step_mesh_converter.py       - STEP → trimesh conversion
  manifold_stipple_processor.py - Stippling pipeline (manifold3d booleans)

ui/
  main_window.py     - GUI implementation (PyQt6)

requirements.txt     - Dependencies
README.md            - Full documentation
```

## Features

✓ Load and parse STEP files
✓ Detect and analyze surface colors
✓ Sphere-based stippling with manifold3d boolean operations
✓ Curvature-compensated depth for uniform indentations
✓ Size variation (Gaussian or uniform)
✓ Export to STL, 3MF, or OBJ mesh formats
✓ Interactive GUI with progress reporting

## Stippling Parameters

- **Sphere Radius**: Radius of each spherical indentation (mm)
- **Depth**: How deep the sphere cap cuts into the surface (mm)
- **Spheres per mm²**: Target density of stipples on the surface

## Typical Workflow

1. Run: `python main.py`
2. Click "Load STEP File" and select your model
3. Click "Analyze Colors" to detect colored surfaces
4. Select the color you want to stipple
5. Adjust stippling parameters
6. Click "Browse..." to select output file (STL/3MF/OBJ)
7. Click "Apply Stippling & Export" to process

## Output

A mesh file (STL/3MF/OBJ) with stippling applied to the selected colored surfaces. Ready for 3D printing!

## Notes

- Processing time depends on model complexity and sphere count
- Stippling creates spherical indentations using manifold3d boolean operations
- Adds gripping texture suitable for handles
- Maintains overall model geometry

For more information, see [README.md](README.md)
