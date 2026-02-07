# STEP Stippling Tool

A 3D CAD application that applies stippling texture to colored surfaces in STEP files for enhanced grippiness.

## Features

- Read 3D models in STEP format (.step, .stp)
- Analyze and display surface colors
- Select which colored surfaces to apply stippling to
- Generate 3D stippling texture patterns
- Export modified model to STEP format
- Interactive GUI for easy workflow

## Requirements

- Python 3.9+
- OCP (Open Cascade Technology Python bindings)
- PyQt6 for GUI
- NumPy for numerical operations

## Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Run the application
python main.py
```

### Workflow

1. **Load STEP File**: Open a .step or .stp file
2. **Analyze Colors**: The app automatically detects surface colors
3. **Select Target Color**: Choose which colored surfaces to add stippling to
4. **Configure Stippling**: Adjust stipple size, density, and depth
5. **Apply Stippling**: Generate the stippled texture on selected surfaces
6. **Export**: Save the modified model as a new STEP file

## Project Structure

```
stipple_tool/
├── main.py                 # Application entry point
├── ui/
│   ├── __init__.py
│   └── main_window.py     # Main GUI window
├── core/
│   ├── __init__.py
│   ├── step_loader.py     # STEP file reading (OCP)
│   ├── color_analyzer.py  # Surface color detection (XCAF)
│   └── stipple_engine.py  # Stippling algorithm (OCP)
├── utils/
│   ├── __init__.py
│   └── mesh_utils.py      # Mesh manipulation utilities
├── requirements.txt
└── README.md
```

## How Stippling Works

The stippling algorithm:
1. Analyzes selected surfaces
2. Generates random or patterned dimple distributions
3. Cuts small spherical indentations into the surface
4. Adjustable parameters for size, density, and depth
5. Preserves overall model geometry while adding texture

## License

MIT
