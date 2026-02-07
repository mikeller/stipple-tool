#!/usr/bin/env python3
"""
Test script to verify all dependencies are installed correctly.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))


def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        print("  ✓ importing numpy...", end=" ")
        import numpy
        print("OK")
        
        print("  ✓ importing OCP...", end=" ")
        from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
        print("OK")
        
        print("  ✓ importing PyQt6...", end=" ")
        from PyQt6.QtWidgets import QApplication, QMainWindow
        print("OK")
        
        return True
        
    except ImportError as e:
        print(f"\n  ✗ Import failed: {e}")
        return False


def test_local_modules():
    """Test that local modules can be imported."""
    print("\nTesting local modules...")
    
    try:
        print("  ✓ importing core.step_loader...", end=" ")
        from core.step_loader import STEPLoader
        print("OK")
        
        print("  ✓ importing core.color_analyzer...", end=" ")
        from core.color_analyzer import ColorAnalyzer
        print("OK")
        
        print("  ✓ importing core.stipple_engine...", end=" ")
        from core.stipple_engine import StippleEngine
        print("OK")
        
        print("  ✓ importing utils.mesh_utils...", end=" ")
        from utils.mesh_utils import MeshUtils
        print("OK")
        
        print("  ✓ importing ui.main_window...", end=" ")
        from ui.main_window import MainWindow
        print("OK")
        
        return True
        
    except ImportError as e:
        print(f"\n  ✗ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_basic_functionality():
    """Test basic functionality of core modules."""
    print("\nTesting basic functionality...")
    
    try:
        print("  ✓ Creating STEPLoader...", end=" ")
        from core.step_loader import STEPLoader
        loader = STEPLoader()
        print("OK")
        
        print("  ✓ Creating ColorAnalyzer...", end=" ")
        from core.color_analyzer import ColorAnalyzer
        analyzer = ColorAnalyzer()
        print("OK")
        
        print("  ✓ Creating StippleEngine...", end=" ")
        from core.stipple_engine import StippleEngine
        engine = StippleEngine()
        print("OK")
        
        print("  ✓ Creating MeshUtils...", end=" ")
        from utils.mesh_utils import MeshUtils
        utils = MeshUtils()
        print("OK")
        
        return True
        
    except Exception as e:
        print(f"\n  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("STEP Stippling Tool - Dependency Test")
    print("=" * 60)
    
    results = []
    
    results.append(("Python imports", test_imports()))
    results.append(("Local modules", test_local_modules()))
    results.append(("Basic functionality", test_basic_functionality()))
    
    print("\n" + "=" * 60)
    print("Test Results:")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("\n✓ All tests passed! Installation is successful.")
        print("\nYou can now run the application with:")
        print("  python main.py")
        return 0
    else:
        print("\n✗ Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
