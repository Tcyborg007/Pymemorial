# tests/unit/visualization/test_03_plotly_engine.py
"""
Unit tests for plotly_engine module.

Tests Plotly engine functionality (conditional on Plotly availability).
"""

import numpy as np
import pytest


# ============================================================================
# PLOTLY ENGINE TESTS (Conditional)
# ============================================================================


@pytest.mark.unit
@pytest.mark.requires_plotly
class TestPlotlyEngine:
    """Test PlotlyEngine class."""

    def test_plotly_engine_creation(self, ensure_imports, skip_if_no_plotly):
        """PlotlyEngine should be creatable."""
        from pymemorial.visualization import PlotlyEngine

        engine = PlotlyEngine()
        assert engine.name == "plotly"
        assert engine.available is True

    def test_plotly_properties(self, ensure_imports, skip_if_no_plotly):
        """PlotlyEngine properties should be correct."""
        from pymemorial.visualization import PlotlyEngine

        engine = PlotlyEngine()
        assert engine.supports_3d is True
        assert engine.supports_interactive is True
        assert len(engine.version) > 0
        assert "plotly" in str(engine).lower()

    def test_supported_formats(self, ensure_imports, skip_if_no_plotly):
        """PlotlyEngine should list supported formats."""
        from pymemorial.visualization import PlotlyEngine, ImageFormat

        engine = PlotlyEngine()
        formats = engine.supported_formats

        assert ImageFormat.HTML in formats  # Always supported
        # PNG, SVG require Kaleido (may or may not be available)

    def test_create_pm_diagram(
        self, ensure_imports, skip_if_no_plotly, simple_pm_data
    ):
        """PlotlyEngine should create P-M diagram."""
        from pymemorial.visualization import PlotlyEngine, PlotConfig

        engine = PlotlyEngine()
        p, m = simple_pm_data

        config = PlotConfig(title="Test P-M")
        fig = engine.create_pm_diagram(p, m, config=config)

        assert fig is not None
        assert hasattr(fig, "data")  # Plotly Figure
        assert len(fig.data) >= 1  # At least one trace

    def test_create_pm_diagram_with_design_point(
        self, ensure_imports, skip_if_no_plotly, pm_data_with_points
    ):
        """PlotlyEngine should handle design points."""
        from pymemorial.visualization import PlotlyEngine

        engine = PlotlyEngine()
        fig = engine.create_pm_diagram(
            pm_data_with_points["p"],
            pm_data_with_points["m"],
            design_point=pm_data_with_points["design_point"],
        )

        assert len(fig.data) >= 2  # Envelope + design point

    def test_create_moment_curvature(
        self, ensure_imports, skip_if_no_plotly, moment_curvature_data
    ):
        """PlotlyEngine should create M-κ diagram."""
        from pymemorial.visualization import PlotlyEngine

        engine = PlotlyEngine()
        curvature, moment = moment_curvature_data

        fig = engine.create_moment_curvature(curvature, moment)

        assert fig is not None
        assert len(fig.data) >= 1

    def test_create_3d_structure(
        self, ensure_imports, skip_if_no_plotly, structure_3d_data
    ):
        """PlotlyEngine should create 3D structure."""
        from pymemorial.visualization import PlotlyEngine

        engine = PlotlyEngine()
        fig = engine.create_3d_structure(
            structure_3d_data["nodes"], structure_3d_data["elements"]
        )

        assert fig is not None
        assert len(fig.data) >= 1

    def test_create_section_2d(
        self, ensure_imports, skip_if_no_plotly, section_2d_data
    ):
        """PlotlyEngine should create 2D section."""
        from pymemorial.visualization import PlotlyEngine

        engine = PlotlyEngine()
        fig = engine.create_section_2d(
            section_2d_data["vertices"],
            section_2d_data["facets"],
            section_2d_data["materials"],
        )

        assert fig is not None

    def test_create_stress_contour(
        self, ensure_imports, skip_if_no_plotly, stress_contour_data
    ):
        """PlotlyEngine should create stress contour."""
        from pymemorial.visualization import PlotlyEngine

        engine = PlotlyEngine()
        fig = engine.create_stress_contour(
            stress_contour_data["x"],
            stress_contour_data["y"],
            stress_contour_data["stress"],
        )

        assert fig is not None

    def test_export_html(
        self, ensure_imports, skip_if_no_plotly, simple_pm_data, test_output_dir
    ):
        """PlotlyEngine should export HTML."""
        from pymemorial.visualization import (
            PlotlyEngine,
            ExportConfig,
            ImageFormat,
        )

        engine = PlotlyEngine()
        p, m = simple_pm_data
        fig = engine.create_pm_diagram(p, m)

        output_file = test_output_dir / "test_plotly.html"
        export_config = ExportConfig(filename=output_file, format=ImageFormat.HTML)

        result_path = engine.export_static(fig, export_config)

        assert result_path.exists()
        assert result_path.suffix == ".html"

    @pytest.mark.requires_kaleido
    def test_export_png(
        self,
        ensure_imports,
        skip_if_no_plotly,
        skip_if_no_kaleido,
        simple_pm_data,
        test_output_dir,
    ):
        """PlotlyEngine should export PNG (requires Kaleido)."""
        from pymemorial.visualization import (
            PlotlyEngine,
            ExportConfig,
            ImageFormat,
        )

        engine = PlotlyEngine()
        p, m = simple_pm_data
        fig = engine.create_pm_diagram(p, m)

        output_file = test_output_dir / "test_plotly.png"
        export_config = ExportConfig(
            filename=output_file, format=ImageFormat.PNG, scale=2.0
        )

        result_path = engine.export_static(fig, export_config)

        assert result_path.exists()
        assert result_path.suffix == ".png"

    def test_validate_arrays(self, ensure_imports, skip_if_no_plotly):
        """PlotlyEngine should validate input arrays."""
        from pymemorial.visualization import PlotlyEngine

        engine = PlotlyEngine()

        # Valid case
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        engine.validate_arrays(a, b, same_length=True)  # Should not raise

        # Invalid case
        c = np.array([7, 8])
        with pytest.raises(ValueError):
            engine.validate_arrays(a, c, same_length=True)

    def test_error_empty_arrays(self, ensure_imports, skip_if_no_plotly):
        """PlotlyEngine should reject empty arrays."""
        from pymemorial.visualization import PlotlyEngine

        engine = PlotlyEngine()

        with pytest.raises(ValueError):
            engine.create_pm_diagram(np.array([]), np.array([]))


# ============================================================================
# KALEIDO TESTS
# ============================================================================


@pytest.mark.unit
class TestKaleidoAvailability:
    """Test Kaleido availability checking."""

    def test_kaleido_status(self, ensure_imports):
        """KALEIDO_AVAILABLE should be boolean."""
        from pymemorial.visualization import KALEIDO_AVAILABLE

        assert isinstance(KALEIDO_AVAILABLE, bool)

    def test_check_plotly_installation(self, ensure_imports):
        """check_plotly_installation should return status dict."""
        from pymemorial.visualization import check_plotly_installation

        status = check_plotly_installation()

        assert isinstance(status, dict)
        assert "plotly" in status
        assert "kaleido" in status
        assert "status" in status


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "unit and requires_plotly"])
