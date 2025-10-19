# tests/unit/visualization/test_01_base_visualizer.py
"""
Unit tests for base_visualizer module.

Tests ABCs, dataclasses, enums, and utility functions.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# ============================================================================
# DATACLASS TESTS - PlotConfig
# ============================================================================


@pytest.mark.unit
class TestPlotConfig:
    """Test PlotConfig dataclass."""

    def test_default_values(self, ensure_imports):
        """PlotConfig should have sensible defaults."""
        from pymemorial.visualization import PlotConfig

        config = PlotConfig()
        assert config.width == 800
        assert config.height == 600
        assert config.dpi == 150
        assert config.grid is True
        assert config.legend is True

    def test_custom_values(self, ensure_imports):
        """PlotConfig should accept custom values."""
        from pymemorial.visualization import PlotConfig, ThemeStyle

        config = PlotConfig(
            title="Custom Title",
            width=1200,
            height=900,
            dpi=300,
            theme=ThemeStyle.PUBLICATION,
        )
        assert config.title == "Custom Title"
        assert config.width == 1200
        assert config.height == 900
        assert config.dpi == 300
        assert config.theme == ThemeStyle.PUBLICATION

    def test_immutable(self, ensure_imports):
        """PlotConfig should be immutable (frozen)."""
        from pymemorial.visualization import PlotConfig

        config = PlotConfig(width=800)

        with pytest.raises(AttributeError):
            config.width = 1000  # Should fail - frozen dataclass

    def test_validation_width_height(self, ensure_imports):
        """PlotConfig should validate width/height."""
        from pymemorial.visualization import PlotConfig

        with pytest.raises(ValueError, match="Width and height must be positive"):
            PlotConfig(width=0, height=600)

        with pytest.raises(ValueError, match="Width and height must be positive"):
            PlotConfig(width=800, height=-100)

    def test_validation_dpi(self, ensure_imports):
        """PlotConfig should validate DPI range."""
        from pymemorial.visualization import PlotConfig

        with pytest.raises(ValueError, match="DPI must be between"):
            PlotConfig(dpi=10)  # Too low

        with pytest.raises(ValueError, match="DPI must be between"):
            PlotConfig(dpi=1000)  # Too high

    def test_validation_font_size(self, ensure_imports):
        """PlotConfig should validate font size."""
        from pymemorial.visualization import PlotConfig

        with pytest.raises(ValueError, match="Font size must be between"):
            PlotConfig(font_size=2)  # Too small

        with pytest.raises(ValueError, match="Font size must be between"):
            PlotConfig(font_size=100)  # Too large


# ============================================================================
# DATACLASS TESTS - ExportConfig
# ============================================================================


@pytest.mark.unit
class TestExportConfig:
    """Test ExportConfig dataclass."""

    def test_default_values(self, ensure_imports, tmp_path):
        """ExportConfig should have sensible defaults."""
        from pymemorial.visualization import ExportConfig, ImageFormat

        config = ExportConfig(filename=tmp_path / "test.png")
        assert config.format == ImageFormat.PNG
        assert config.scale == 1.0
        assert config.transparent_bg is False
        assert config.quality == 95

    def test_corrected_filename(self, ensure_imports, tmp_path):
        """ExportConfig should auto-correct file extension."""
        from pymemorial.visualization import ExportConfig, ImageFormat

        # Wrong extension
        config = ExportConfig(
            filename=tmp_path / "test.jpg", format=ImageFormat.PNG
        )

        corrected = config.corrected_filename
        assert corrected.suffix == ".png"

    def test_validation_scale(self, ensure_imports, tmp_path):
        """ExportConfig should validate scale range."""
        from pymemorial.visualization import ExportConfig

        with pytest.raises(ValueError, match="Scale must be between"):
            ExportConfig(filename=tmp_path / "test.png", scale=0)

        with pytest.raises(ValueError, match="Scale must be between"):
            ExportConfig(filename=tmp_path / "test.png", scale=20)

    def test_validation_quality(self, ensure_imports, tmp_path):
        """ExportConfig should validate quality range."""
        from pymemorial.visualization import ExportConfig

        with pytest.raises(ValueError, match="Quality must be between"):
            ExportConfig(filename=tmp_path / "test.jpg", quality=0)

        with pytest.raises(ValueError, match="Quality must be between"):
            ExportConfig(filename=tmp_path / "test.jpg", quality=101)

    def test_transparent_bg_validation(self, ensure_imports, tmp_path):
        """ExportConfig should validate transparent bg for format."""
        from pymemorial.visualization import ExportConfig, ImageFormat

        # Transparent BG not supported for JPEG
        with pytest.raises(ValueError, match="Transparent background not supported"):
            ExportConfig(
                filename=tmp_path / "test.jpg",
                format=ImageFormat.JPEG,
                transparent_bg=True,
            )

        # Should work for PNG
        config = ExportConfig(
            filename=tmp_path / "test.png",
            format=ImageFormat.PNG,
            transparent_bg=True,
        )
        assert config.transparent_bg is True


# ============================================================================
# ENUM TESTS
# ============================================================================


@pytest.mark.unit
class TestImageFormat:
    """Test ImageFormat enum."""

    def test_all_formats_present(self, ensure_imports):
        """ImageFormat should have all expected formats."""
        from pymemorial.visualization import ImageFormat

        expected = ["PNG", "SVG", "JPEG", "PDF", "HTML", "WEBP", "EPS"]
        for fmt in expected:
            assert hasattr(ImageFormat, fmt)

    def test_is_vector(self, ensure_imports):
        """Test is_vector property."""
        from pymemorial.visualization import ImageFormat

        assert ImageFormat.SVG.is_vector is True
        assert ImageFormat.PDF.is_vector is True
        assert ImageFormat.EPS.is_vector is True
        assert ImageFormat.PNG.is_vector is False
        assert ImageFormat.JPEG.is_vector is False

    def test_is_raster(self, ensure_imports):
        """Test is_raster property."""
        from pymemorial.visualization import ImageFormat

        assert ImageFormat.PNG.is_raster is True
        assert ImageFormat.JPEG.is_raster is True
        assert ImageFormat.WEBP.is_raster is True
        assert ImageFormat.SVG.is_raster is False
        assert ImageFormat.PDF.is_raster is False

    def test_is_interactive(self, ensure_imports):
        """Test is_interactive property."""
        from pymemorial.visualization import ImageFormat

        assert ImageFormat.HTML.is_interactive is True
        assert ImageFormat.PNG.is_interactive is False
        assert ImageFormat.SVG.is_interactive is False

    def test_requires_kaleido(self, ensure_imports):
        """Test requires_kaleido property."""
        from pymemorial.visualization import ImageFormat

        assert ImageFormat.PNG.requires_kaleido is True
        assert ImageFormat.SVG.requires_kaleido is True
        assert ImageFormat.PDF.requires_kaleido is True
        assert ImageFormat.HTML.requires_kaleido is False


@pytest.mark.unit
class TestDiagramType:
    """Test DiagramType enum."""

    def test_all_types_present(self, ensure_imports):
        """DiagramType should have all expected types."""
        from pymemorial.visualization import DiagramType

        expected = [
            "INTERACTION_PM",
            "MOMENT_CURVATURE",
            "SHEAR_MOMENT",
            "STRUCTURE_3D",
            "SECTION_2D",
            "STRESS_CONTOUR",
        ]
        for dtype in expected:
            assert hasattr(DiagramType, dtype)


# ============================================================================
# UTILITY FUNCTION TESTS
# ============================================================================


@pytest.mark.unit
class TestUtilityFunctions:
    """Test utility functions."""

    def test_get_default_config_pm(self, ensure_imports):
        """get_default_config should return appropriate config for P-M."""
        from pymemorial.visualization import get_default_config, DiagramType

        config = get_default_config(DiagramType.INTERACTION_PM)
        assert "P-M" in config.title
        assert "M/M_n" in config.xlabel
        assert "P/P_n" in config.ylabel

    def test_get_default_config_moment_curvature(self, ensure_imports):
        """get_default_config should return appropriate config for M-κ."""
        from pymemorial.visualization import get_default_config, DiagramType

        config = get_default_config(DiagramType.MOMENT_CURVATURE)
        assert "Moment-Curvature" in config.title
        assert "κ" in config.xlabel or "Curvature" in config.xlabel

    def test_get_default_config_section(self, ensure_imports):
        """get_default_config should set equal_aspect for sections."""
        from pymemorial.visualization import get_default_config, DiagramType

        config = get_default_config(DiagramType.SECTION_2D)
        assert config.equal_aspect is True

    def test_validate_colormap_matplotlib(self, ensure_imports):
        """validate_colormap should check matplotlib colormaps."""
        from pymemorial.visualization import validate_colormap

        # Valid matplotlib colormaps
        assert validate_colormap("viridis", engine="matplotlib") is True
        assert validate_colormap("plasma", engine="matplotlib") is True

        # Invalid colormap
        assert validate_colormap("nonexistent_cmap", engine="matplotlib") is False

    def test_validate_colormap_plotly(self, ensure_imports):
        """validate_colormap should check plotly colormaps."""
        from pymemorial.visualization import validate_colormap

        # Valid plotly colormaps
        assert validate_colormap("viridis", engine="plotly") is True
        assert validate_colormap("RdBu", engine="plotly") is True

        # Invalid colormap
        assert validate_colormap("nonexistent_cmap", engine="plotly") is False


# ============================================================================
# ABC TESTS - VisualizerEngine
# ============================================================================


@pytest.mark.unit
class TestVisualizerEngine:
    """Test VisualizerEngine ABC."""

    def test_cannot_instantiate_abc(self, ensure_imports):
        """VisualizerEngine is abstract and cannot be instantiated."""
        from pymemorial.visualization import VisualizerEngine

        with pytest.raises(TypeError):
            VisualizerEngine("test")  # Should fail - abstract class

    def test_validate_arrays_same_length(self, ensure_imports):
        """Test validate_arrays with same_length=True."""
        from pymemorial.visualization import create_visualizer

        viz = create_visualizer()

        # Valid case
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        viz.validate_arrays(a, b, same_length=True)  # Should not raise

        # Invalid case - different lengths
        c = np.array([7, 8])
        with pytest.raises(ValueError, match="same length"):
            viz.validate_arrays(a, c, same_length=True)

    def test_validate_arrays_min_length(self, ensure_imports):
        """Test validate_arrays with min_length."""
        from pymemorial.visualization import create_visualizer

        viz = create_visualizer()

        # Valid case
        a = np.array([1, 2, 3])
        viz.validate_arrays(a, min_length=2)  # Should not raise

        # Invalid case - too short
        b = np.array([1])
        with pytest.raises(ValueError, match="at least"):
            viz.validate_arrays(b, min_length=2)

    def test_repr_and_str(self, ensure_imports):
        """Test __repr__ and __str__ methods."""
        from pymemorial.visualization import create_visualizer

        viz = create_visualizer()

        repr_str = repr(viz)
        assert viz.name in repr_str
        assert viz.version in repr_str

        str_str = str(viz)
        assert viz.name in str_str


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "unit"])
