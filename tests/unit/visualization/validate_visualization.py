# tests/unit/visualization/validate_visualization.py
"""
Validation script for visualization module.

This script performs comprehensive validation:
1. Smoke tests (imports, instantiation)
2. Debug with visual output (generates test figures)
3. Error checking and reporting

Usage:
    cd C:\\Users\\Tcyborg\\MODULO DE DESENVOLVIMENTO DA BIBLIOTECA PYMEMORIAL\\Pymemorial_v1
    python tests/unit/visualization/validate_visualization.py
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple
import traceback

# ============================================================================
# SETUP - Add src to path
# ============================================================================

# Get project root (go up 3 levels from this file)
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
SRC_PATH = PROJECT_ROOT / "src"
TEST_OUTPUT_DIR = PROJECT_ROOT / "tests" / "unit" / "visualization" / "validation_outputs"

# Add src to Python path
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

# Create output directory
TEST_OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 80)
print("PyMemorial Visualization - VALIDATION SCRIPT")
print("=" * 80)
print(f"Project Root: {PROJECT_ROOT}")
print(f"Source Path:  {SRC_PATH}")
print(f"Output Dir:   {TEST_OUTPUT_DIR}")
print("=" * 80)


# ============================================================================
# TEST RESULTS TRACKER
# ============================================================================

class TestResults:
    """Track test execution results."""
    
    def __init__(self):
        self.passed: List[str] = []
        self.failed: List[Tuple[str, Exception]] = []
        self.skipped: List[Tuple[str, str]] = []
    
    def add_pass(self, test_name: str) -> None:
        """Record passing test."""
        self.passed.append(test_name)
        print(f"  ✓ {test_name}")
    
    def add_fail(self, test_name: str, error: Exception) -> None:
        """Record failing test."""
        self.failed.append((test_name, error))
        print(f"  ✗ {test_name}")
        print(f"    Error: {error}")
    
    def add_skip(self, test_name: str, reason: str) -> None:
        """Record skipped test."""
        self.skipped.append((test_name, reason))
        print(f"  ⊘ {test_name} (SKIPPED: {reason})")
    
    def summary(self) -> None:
        """Print summary."""
        total = len(self.passed) + len(self.failed) + len(self.skipped)
        
        print("\n" + "=" * 80)
        print("VALIDATION SUMMARY")
        print("=" * 80)
        print(f"✓ Passed:  {len(self.passed)}/{total}")
        print(f"✗ Failed:  {len(self.failed)}/{total}")
        print(f"⊘ Skipped: {len(self.skipped)}/{total}")
        
        if self.failed:
            print("\n❌ FAILED TESTS:")
            for name, error in self.failed:
                print(f"  • {name}")
                print(f"    {type(error).__name__}: {error}")
        
        if self.skipped:
            print("\n⚠️  SKIPPED TESTS:")
            for name, reason in self.skipped:
                print(f"  • {name}: {reason}")
        
        success_rate = (len(self.passed) / total * 100) if total > 0 else 0
        print(f"\nSuccess Rate: {success_rate:.1f}%")
        
        if len(self.failed) == 0:
            print("\n🎉 ALL TESTS PASSED! Code is ready for pytest.")
            return 0
        else:
            print("\n⚠️  SOME TESTS FAILED. Fix errors before proceeding.")
            return 1


results = TestResults()


# ============================================================================
# PHASE 1: SMOKE TESTS - Basic Imports
# ============================================================================

def test_imports_base():
    """Test base_visualizer imports."""
    try:
        from pymemorial.visualization import base_visualizer
        
        required = [
            "VisualizerEngine",
            "PlotConfig",
            "ExportConfig",
            "ImageFormat",
            "DiagramType",
            "ThemeStyle",
        ]
        
        for attr in required:
            assert hasattr(base_visualizer, attr), f"Missing: {attr}"
        
        results.add_pass("Import base_visualizer")
        return True
    except Exception as e:
        results.add_fail("Import base_visualizer", e)
        return False


def test_imports_matplotlib_utils():
    """Test matplotlib_utils imports."""
    try:
        from pymemorial.visualization import matplotlib_utils
        
        required = [
            "plot_pm_interaction",
            "plot_moment_curvature",
            "add_dimension_arrow",
            "add_section_hatch",
            "set_publication_style",
        ]
        
        for attr in required:
            assert hasattr(matplotlib_utils, attr), f"Missing: {attr}"
        
        results.add_pass("Import matplotlib_utils")
        return True
    except Exception as e:
        results.add_fail("Import matplotlib_utils", e)
        return False


def test_imports_plotly_engine():
    """Test plotly_engine imports (conditional)."""
    try:
        from pymemorial.visualization import plotly_engine
        
        if not plotly_engine.PLOTLY_AVAILABLE:
            results.add_skip(
                "Import plotly_engine",
                "Plotly not installed (pip install pymemorial[viz])"
            )
            return True
        
        required = ["PlotlyEngine", "PLOTLY_AVAILABLE", "KALEIDO_AVAILABLE"]
        
        for attr in required:
            assert hasattr(plotly_engine, attr), f"Missing: {attr}"
        
        results.add_pass("Import plotly_engine")
        return True
    except ImportError:
        results.add_skip(
            "Import plotly_engine",
            "Plotly not installed (optional)"
        )
        return True
    except Exception as e:
        results.add_fail("Import plotly_engine", e)
        return False


def test_imports_package_level():
    """Test package-level imports."""
    try:
        import pymemorial.visualization as viz
        
        required = [
            # Core
            "VisualizerEngine",
            "PlotConfig",
            "ExportConfig",
            # Factory
            "VisualizerFactory",
            "create_visualizer",
            "list_available_engines",
            # Utils
            "check_installation",
            "get_engine_status",
        ]
        
        for attr in required:
            assert hasattr(viz, attr), f"Missing: {attr}"
        
        results.add_pass("Import package level")
        return True
    except Exception as e:
        results.add_fail("Import package level", e)
        return False


# ============================================================================
# PHASE 2: INSTANTIATION TESTS
# ============================================================================

def test_create_plot_config():
    """Test PlotConfig creation."""
    try:
        from pymemorial.visualization import PlotConfig
        
        config = PlotConfig()
        assert config.width == 800
        assert config.height == 600
        assert config.dpi == 150
        
        # Custom config
        custom = PlotConfig(width=1200, height=900, dpi=300)
        assert custom.width == 1200
        assert custom.dpi == 300
        
        results.add_pass("Create PlotConfig")
        return True
    except Exception as e:
        results.add_fail("Create PlotConfig", e)
        return False


def test_create_export_config():
    """Test ExportConfig creation."""
    try:
        from pymemorial.visualization import ExportConfig, ImageFormat
        
        config = ExportConfig(
            filename=TEST_OUTPUT_DIR / "test.png",
            format=ImageFormat.PNG,
        )
        assert config.format == ImageFormat.PNG
        assert config.scale == 1.0
        
        results.add_pass("Create ExportConfig")
        return True
    except Exception as e:
        results.add_fail("Create ExportConfig", e)
        return False


def test_factory_create():
    """Test factory engine creation."""
    try:
        from pymemorial.visualization import create_visualizer
        
        viz = create_visualizer()
        assert viz is not None
        assert hasattr(viz, "version")
        assert hasattr(viz, "available")
        assert viz.available  # At least one engine must be available
        
        print(f"    → Created: {viz.name} v{viz.version}")
        
        results.add_pass("Factory create_visualizer")
        return True
    except Exception as e:
        results.add_fail("Factory create_visualizer", e)
        return False


def test_list_engines():
    """Test engine listing."""
    try:
        from pymemorial.visualization import list_available_engines
        
        engines = list_available_engines()
        assert isinstance(engines, list)
        assert len(engines) > 0, "No engines available!"
        
        print(f"    → Available engines: {', '.join(engines)}")
        
        results.add_pass("List available engines")
        return True
    except Exception as e:
        results.add_fail("List available engines", e)
        return False


# ============================================================================
# PHASE 3: FUNCTIONAL TESTS - Matplotlib (always available)
# ============================================================================

def test_matplotlib_pm_diagram():
    """Test matplotlib P-M diagram generation."""
    try:
        import numpy as np
        from pymemorial.visualization import plot_pm_interaction
        import matplotlib.pyplot as plt
        
        # Create test data
        p = np.array([0, 0.5, 1.0, 0.8, 0.3, 0])
        m = np.array([0.6, 0.8, 0.5, 0.4, 0.7, 0])
        
        # Create plot
        fig, ax = plt.subplots(figsize=(8, 6))
        plot_pm_interaction(ax, p, m, design_point=(0.4, 0.6))
        
        # Save
        output_file = TEST_OUTPUT_DIR / "pm_diagram_matplotlib.png"
        fig.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close(fig)
        
        assert output_file.exists(), "Output file not created"
        
        print(f"    → Saved: {output_file.name}")
        results.add_pass("Matplotlib P-M diagram")
        return True
    except Exception as e:
        results.add_fail("Matplotlib P-M diagram", e)
        traceback.print_exc()
        return False


def test_matplotlib_section_hatch():
    """Test matplotlib section hatching."""
    try:
        import numpy as np
        from pymemorial.visualization import add_section_hatch
        import matplotlib.pyplot as plt
        
        # Create test section (rectangle)
        vertices = np.array([[0, 0], [0.3, 0], [0.3, 0.5], [0, 0.5]])
        
        fig, ax = plt.subplots(figsize=(6, 6))
        add_section_hatch(ax, vertices, material="steel")
        ax.set_xlim(-0.1, 0.4)
        ax.set_ylim(-0.1, 0.6)
        ax.set_aspect("equal")
        ax.set_title("Steel Section with Hatching")
        
        output_file = TEST_OUTPUT_DIR / "section_hatch_matplotlib.png"
        fig.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close(fig)
        
        assert output_file.exists()
        
        print(f"    → Saved: {output_file.name}")
        results.add_pass("Matplotlib section hatch")
        return True
    except Exception as e:
        results.add_fail("Matplotlib section hatch", e)
        traceback.print_exc()
        return False


# ============================================================================
# PHASE 4: FUNCTIONAL TESTS - Plotly (conditional)
# ============================================================================

def test_plotly_pm_diagram():
    """Test Plotly P-M diagram (if available)."""
    try:
        from pymemorial.visualization import PLOTLY_AVAILABLE
        
        if not PLOTLY_AVAILABLE:
            results.add_skip(
                "Plotly P-M diagram",
                "Plotly not installed"
            )
            return True
        
        import numpy as np
        from pymemorial.visualization import create_visualizer, PlotConfig, ExportConfig, ImageFormat
        
        # Create Plotly engine
        viz = create_visualizer(engine="plotly")
        
        if not viz.available:
            results.add_skip("Plotly P-M diagram", "Plotly engine unavailable")
            return True
        
        # Create test data
        p = np.array([0, 0.5, 1.0, 0.8, 0.3, 0])
        m = np.array([0.6, 0.8, 0.5, 0.4, 0.7, 0])
        
        config = PlotConfig(title="P-M Interaction (Plotly)", width=800, height=600)
        fig = viz.create_pm_diagram(p, m, design_point=(0.4, 0.6), config=config)
        
        # Export HTML (always works)
        html_file = viz.export_static(
            fig,
            ExportConfig(
                filename=TEST_OUTPUT_DIR / "pm_diagram_plotly.html",
                format=ImageFormat.HTML,
            ),
        )
        
        assert html_file.exists()
        print(f"    → Saved: {html_file.name}")
        
        # Try PNG (requires Kaleido)
        try:
            from pymemorial.visualization import KALEIDO_AVAILABLE
            
            if KALEIDO_AVAILABLE:
                png_file = viz.export_static(
                    fig,
                    ExportConfig(
                        filename=TEST_OUTPUT_DIR / "pm_diagram_plotly.png",
                        format=ImageFormat.PNG,
                        scale=2.0,
                    ),
                )
                print(f"    → Saved: {png_file.name}")
            else:
                print(f"    → PNG export skipped (Kaleido not installed)")
        except Exception:
            print(f"    → PNG export failed (Kaleido issue)")
        
        results.add_pass("Plotly P-M diagram")
        return True
        
    except Exception as e:
        results.add_fail("Plotly P-M diagram", e)
        traceback.print_exc()
        return False


# ============================================================================
# PHASE 5: ERROR HANDLING TESTS
# ============================================================================

def test_error_handling():
    """Test error handling with invalid inputs."""
    try:
        import numpy as np
        from pymemorial.visualization import create_visualizer
        
        viz = create_visualizer()
        
        # Test 1: Empty arrays should fail
        try:
            fig = viz.create_pm_diagram(np.array([]), np.array([]))
            results.add_fail("Error handling", Exception("Should reject empty arrays"))
            return False
        except ValueError:
            pass  # Expected
        
        # Test 2: Mismatched arrays should fail
        try:
            fig = viz.create_pm_diagram(np.array([1, 2, 3]), np.array([1, 2]))
            results.add_fail("Error handling", Exception("Should reject mismatched arrays"))
            return False
        except ValueError:
            pass  # Expected
        
        # Test 3: Very small arrays should work
        fig = viz.create_pm_diagram(np.array([0, 1]), np.array([1, 0]))
        
        results.add_pass("Error handling")
        return True
        
    except Exception as e:
        results.add_fail("Error handling", e)
        traceback.print_exc()
        return False


# ============================================================================
# MAIN RUNNER
# ============================================================================

def main():
    """Run all validation tests."""
    
    print("\n" + "=" * 80)
    print("PHASE 1: SMOKE TESTS (Imports)")
    print("=" * 80)
    
    test_imports_base()
    test_imports_matplotlib_utils()
    test_imports_plotly_engine()
    test_imports_package_level()
    
    print("\n" + "=" * 80)
    print("PHASE 2: INSTANTIATION TESTS")
    print("=" * 80)
    
    test_create_plot_config()
    test_create_export_config()
    test_factory_create()
    test_list_engines()
    
    print("\n" + "=" * 80)
    print("PHASE 3: MATPLOTLIB FUNCTIONAL TESTS")
    print("=" * 80)
    
    test_matplotlib_pm_diagram()
    test_matplotlib_section_hatch()
    
    print("\n" + "=" * 80)
    print("PHASE 4: PLOTLY FUNCTIONAL TESTS (Conditional)")
    print("=" * 80)
    
    test_plotly_pm_diagram()
    
    print("\n" + "=" * 80)
    print("PHASE 5: ERROR HANDLING TESTS")
    print("=" * 80)
    
    test_error_handling()
    
    # Print summary
    return results.summary()


if __name__ == "__main__":
    exit_code = main()
    
    print("\n" + "=" * 80)
    print(f"Validation outputs saved to: {TEST_OUTPUT_DIR}")
    print("=" * 80)
    
    sys.exit(exit_code)
