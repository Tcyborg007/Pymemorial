# tests/unit/visualization/test_00_smoke.py
"""
Smoke tests for visualization module.

These tests validate that basic imports work and modules are loadable.
Should run in <2 seconds total.
"""

import sys
import pytest


@pytest.mark.smoke
class TestSmokeImports:
    """Test that all modules can be imported without errors."""

    def test_import_base_visualizer(self, ensure_imports):
        """Base visualizer module must import."""
        from pymemorial.visualization import base_visualizer
        
        assert hasattr(base_visualizer, "VisualizerEngine")
        assert hasattr(base_visualizer, "PlotConfig")
        assert hasattr(base_visualizer, "ExportConfig")
        assert hasattr(base_visualizer, "ImageFormat")
        assert hasattr(base_visualizer, "DiagramType")

    def test_import_matplotlib_utils(self, ensure_imports):
        """Matplotlib utils must always be available."""
        from pymemorial.visualization import matplotlib_utils
        
        assert hasattr(matplotlib_utils, "plot_pm_interaction")
        assert hasattr(matplotlib_utils, "plot_moment_curvature")
        assert hasattr(matplotlib_utils, "add_dimension_arrow")
        assert hasattr(matplotlib_utils, "add_section_hatch")

    def test_import_plotly_engine_conditional(self, ensure_imports):
        """Plotly engine may or may not be available."""
        try:
            from pymemorial.visualization import plotly_engine
            # If import succeeds, check required attributes
            assert hasattr(plotly_engine, "PlotlyEngine")
            assert hasattr(plotly_engine, "PLOTLY_AVAILABLE")
        except ImportError:
            # OK if not installed
            pass

    def test_import_package_level(self, ensure_imports):
        """Package-level imports must work."""
        import pymemorial.visualization as viz
        
        # Core interfaces
        assert hasattr(viz, "VisualizerEngine")
        assert hasattr(viz, "PlotConfig")
        assert hasattr(viz, "ExportConfig")
        
        # Factory
        assert hasattr(viz, "VisualizerFactory")
        assert hasattr(viz, "create_visualizer")
        assert hasattr(viz, "list_available_engines")
        
        # Utils
        assert hasattr(viz, "check_installation")
        assert hasattr(viz, "get_engine_status")


@pytest.mark.smoke
class TestSmokeInstantiation:
    """Test that basic objects can be instantiated."""

    def test_create_plot_config(self, ensure_imports):
        """PlotConfig must be creatable with defaults."""
        from pymemorial.visualization import PlotConfig
        
        config = PlotConfig()
        assert config.width == 800
        assert config.height == 600
        assert config.dpi == 150

    def test_create_export_config(self, ensure_imports, tmp_path):
        """ExportConfig must be creatable."""
        from pymemorial.visualization import ExportConfig, ImageFormat
        
        config = ExportConfig(
            filename=tmp_path / "test.png",
            format=ImageFormat.PNG,
        )
        assert config.format == ImageFormat.PNG
        assert config.scale == 1.0

    def test_factory_create_default(self, ensure_imports):
        """Factory must create some engine."""
        from pymemorial.visualization import create_visualizer
        
        viz = create_visualizer()
        assert viz is not None
        assert hasattr(viz, "version")
        assert hasattr(viz, "available")

    def test_list_engines(self, ensure_imports):
        """Engine listing must work."""
        from pymemorial.visualization import list_available_engines
        
        engines = list_available_engines()
        assert isinstance(engines, list)
        assert len(engines) > 0  # At least one engine available


@pytest.mark.smoke
class TestSmokeStatus:
    """Test status checking functions."""

    def test_check_installation_runs(self, ensure_imports, capsys):
        """check_installation() must run without crash."""
        from pymemorial.visualization import check_installation
        
        check_installation()
        
        # Capture output
        captured = capsys.readouterr()
        assert "PyMemorial Visualization" in captured.out
        assert "Installation Status" in captured.out

    def test_get_engine_status(self, ensure_imports):
        """get_engine_status() must return dict."""
        from pymemorial.visualization import get_engine_status
        
        status = get_engine_status()
        assert isinstance(status, dict)
        assert "engines" in status
        assert "plotly" in status
        assert "default_engine" in status


# ============================================================================
# RUN SMOKE TESTS ONLY
# ============================================================================

if __name__ == "__main__":
    """
    Quick smoke test runner.
    
    Usage:
        python -m pytest tests/unit/visualization/test_00_smoke.py -v
    """
    pytest.main([__file__, "-v", "-m", "smoke"])
