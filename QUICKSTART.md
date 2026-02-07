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

3. Verify installation:
   ```bash
   python3 scripts/test_installation.py
   ```

## Usage - GUI Mode

```bash
python main.py
```

This opens an interactive interface where you can:
- Load STEP files (.step or .stp)
- View and select surface colors
- Configure stippling parameters (size, depth, density, pattern)
- Apply stippling to selected surfaces
- Export as a new STEP file

## Usage - Programmatic

```bash
python demo.py
```

Example code:
```python
from core.step_loader import STEPLoader
from core.color_analyzer import ColorAnalyzer
from core.stipple_engine import StippleEngine

loader = STEPLoader()
loader.load("model.step")

analyzer = ColorAnalyzer()
model_data = loader.get_model()
colors = analyzer.detect_distinct_colors(model_data)

engine = StippleEngine()
engine.set_parameters(size=1.0, depth=0.5, density=0.3)

faces = model_data["faces"]
modified = engine.apply_stippling_to_shape(
    model_data["shape"], faces, pattern="random"
)

loader.save_step("output.step", modified)
```

## File Structure

```
main.py              - Launch GUI application
demo.py              - Programmatic example
test_installation.py - Verify dependencies

core/
  step_loader.py     - STEP file I/O (OCP-based)
  color_analyzer.py  - Surface color detection
  stipple_engine.py  - Stippling algorithm

ui/
  main_window.py     - GUI implementation (PyQt6)

requirements.txt     - Dependencies
README.md            - Full documentation
INSTALLATION.md      - Setup guide
```

## Features

✓ Load and parse STEP files  
✓ Detect and analyze surface colors  
✓ Apply customizable stippling patterns  
✓ Export modified models to STEP format  
✓ Interactive GUI with real-time preview  
✓ Batch processing support  

## Stippling Parameters

- **Size**: Diameter of each indentation (mm)
- **Depth**: How deep the indentations go (mm)
- **Density**: Number of stipples per unit area (0-1)
- **Pattern**: random, grid, or hexagonal distribution

## Typical Workflow

1. Run: `python main.py`
2. Click "Load STEP File" and select your model
3. Click "Analyze Colors" to detect colored surfaces
4. Select the color you want to stipple
5. Adjust stippling parameters in the Stipple tab
6. Click "Browse..." to select output file
7. Click "Apply Stippling & Export" to process

## Output

A new STEP file with stippling applied to the selected colored surfaces. Ready for 3D printing!

## Notes

- Processing time depends on model complexity
- Stippling creates hemispheric indentations
- Adds gripping texture suitable for handles
- Maintains overall model geometry
- Backward compatible with STEP standard

For more information, see [README.md](README.md) and INSTALLATION.md
