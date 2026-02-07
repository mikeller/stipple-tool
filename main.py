#!/usr/bin/env python3
"""
STEP Stippling Tool - Main Entry Point

This application allows users to:
1. Load 3D models in STEP format
2. Analyze surface colors
3. Apply stippling texture to selected colored surfaces
4. Export the modified model back to STEP format

Stippling adds gripping texture similar to tool handles and sports equipment.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ui.main_window import MainWindow
from PyQt6.QtWidgets import QApplication


def main():
    """Main application entry point."""
    try:
        # Create output directory if it doesn't exist
        output_dir = project_root / "output_models"
        output_dir.mkdir(exist_ok=True)
        
        # Create and run the application
        app = QApplication(sys.argv)
        window = MainWindow()
        window.show()
        
        sys.exit(app.exec())
        
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
