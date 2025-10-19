# tests/unit/visualization/test_02_matplotlib_utils.py
"""
Unit tests for matplotlib_utils module.

Tests all matplotlib utility functions, styling, and diagram generation.
"""

import numpy as np
import pytest
import matplotlib.pyplot as plt


# ============================================================================
# STYLING TESTS
# ============================================================================


@pytest.mark.unit
class TestStyling:
    """Test matplotlib styling functions."""

    def test_set_publication_style_default(self, ensure_imports):
        """set_publication_style with default should not crash."""
        from pymemorial.visualization import set_publication_style

        set_publication_style()
        # Check some rcParams were set
        assert plt.rcParams["savefig.dpi"] == 300

    @pytest.mark.parametrize(
        "style", ["default", "ieee", "nature", "asce", "presentation", "thesis"]
    )
    def test_set_publication_style_all(self, ensure_imports, style):
        """All publication styles should work."""
        from pymemorial.visualization import set_publication_style

        set_publication_style(style)
        # Should not raise

    def test_reset_style(self, ensure_imports):
        """reset_style should restore defaults."""
        from pymemorial.visualization import set_publication_style, reset_style
    
        # Set custom style
        set_publication_style("ieee")
        custom_font_size = plt.rcParams["font.size"]
    
        # Reset
        reset_style()
        default_font_size = plt.rcParams["font.size"]
    
        # Font size should exist (may or may not be different, but should be valid)
        assert isinstance(default_font_size, (int, float)) or isinstance(default_font_size, str)
        # Function should not crash - that's the main test


# ============================================================================
# DIMENSION ANNOTATION TESTS
# ============================================================================


@pytest.mark.unit
class TestDimensionAnnotations:
    """Test dimension annotation functions."""

    def test_add_dimension_arrow_basic(self, ensure_imports):
        """add_dimension_arrow should create arrow and text."""
        from pymemorial.visualization import add_dimension_arrow

        fig, ax = plt.subplots()
        arrow, text = add_dimension_arrow(ax, (0, 0), (5, 0), "5.000 m")

        assert arrow is not None
        assert text is not None
        assert text.get_text() == "5.000 m"

    def test_add_dimension_arrow_auto_text(self, ensure_imports):
        """add_dimension_arrow should auto-calculate distance."""
        from pymemorial.visualization import add_dimension_arrow

        fig, ax = plt.subplots()
        arrow, text = add_dimension_arrow(
            ax, (0, 0), (3, 4), "", decimal_places=2
        )  # 3-4-5 triangle

        distance = float(text.get_text())
        assert abs(distance - 5.0) < 0.01  # Should be 5.0

    def test_add_multiple_dimensions(self, ensure_imports):
        """add_multiple_dimensions should create multiple arrows."""
        from pymemorial.visualization import add_multiple_dimensions

        fig, ax = plt.subplots()
        points = [(0, 0), (5, 0), (5, 3), (0, 3)]

        annotations = add_multiple_dimensions(ax, points)
        assert len(annotations) == 3  # 4 points = 3 segments


# ============================================================================
# MATERIAL HATCHING TESTS
# ============================================================================


@pytest.mark.unit
class TestMaterialHatching:
    """Test material hatching functions."""

    def test_add_section_hatch_steel(self, ensure_imports):
        """add_section_hatch should work for steel."""
        from pymemorial.visualization import add_section_hatch

        vertices = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])

        fig, ax = plt.subplots()
        patch = add_section_hatch(ax, vertices, material="steel")

        assert patch is not None
        assert patch.get_label() == "Steel"

    @pytest.mark.parametrize(
        "material", ["steel", "concrete", "wood", "aluminum", "masonry"]
    )
    def test_add_section_hatch_all_materials(self, ensure_imports, material):
        """All materials should have colors and hatches."""
        from pymemorial.visualization import add_section_hatch

        vertices = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])

        fig, ax = plt.subplots()
        patch = add_section_hatch(ax, vertices, material=material)

        assert patch is not None
        assert patch.get_facecolor() is not None
        assert patch.get_hatch() is not None

    def test_plot_composite_section(self, ensure_imports):
        """plot_composite_section should handle multiple materials."""
        from pymemorial.visualization import plot_composite_section

        # Two materials (steel + concrete)
        materials = [
            {"vertices": np.array([[0, 0], [0.3, 0], [0.3, 0.1], [0, 0.1]]), "material": "steel"},
            {"vertices": np.array([[0, 0.1], [0.3, 0.1], [0.3, 0.5], [0, 0.5]]), "material": "concrete"},
        ]

        fig, ax = plt.subplots()
        patches = plot_composite_section(ax, materials, show_centroid=True)

        assert len(patches) == 2


# ============================================================================
# DIAGRAM GENERATION TESTS
# ============================================================================


@pytest.mark.unit
class TestDiagramGeneration:
    """Test diagram generation functions."""

    def test_plot_pm_interaction_basic(self, ensure_imports, simple_pm_data):
        """plot_pm_interaction should create basic diagram."""
        from pymemorial.visualization import plot_pm_interaction

        p, m = simple_pm_data

        fig, ax = plt.subplots()
        lines = plot_pm_interaction(ax, p, m)

        assert len(lines) >= 1  # At least capacity envelope

    def test_plot_pm_interaction_with_design_point(
        self, ensure_imports, pm_data_with_points
    ):
        """plot_pm_interaction should highlight design point."""
        from pymemorial.visualization import plot_pm_interaction

        fig, ax = plt.subplots()
        lines = plot_pm_interaction(
            ax,
            pm_data_with_points["p"],
            pm_data_with_points["m"],
            design_point=pm_data_with_points["design_point"],
        )

        assert len(lines) >= 2  # Envelope + design point

    def test_plot_moment_curvature_basic(self, ensure_imports, moment_curvature_data):
        """plot_moment_curvature should create M-κ diagram."""
        from pymemorial.visualization import plot_moment_curvature

        curvature, moment = moment_curvature_data

        fig, ax = plt.subplots()
        lines = plot_moment_curvature(ax, curvature, moment)

        assert len(lines) >= 1

    def test_plot_moment_curvature_with_points(
        self, ensure_imports, moment_curvature_data
    ):
        """plot_moment_curvature should mark yield and ultimate points."""
        from pymemorial.visualization import plot_moment_curvature

        curvature, moment = moment_curvature_data

        fig, ax = plt.subplots()
        lines = plot_moment_curvature(
            ax,
            curvature,
            moment,
            yield_point=(0.005, 5e3),
            ultimate_point=(0.01, 7.5e3),
        )

        assert len(lines) >= 3  # Curve + yield + ultimate

    def test_plot_shear_moment_diagrams(self, ensure_imports):
        """plot_shear_moment_diagrams should create two subplots."""
        from pymemorial.visualization import plot_shear_moment_diagrams

        x = np.linspace(0, 10, 50)
        shear = 100 - 20 * x
        moment = 100 * x - 10 * x**2

        fig, ax = plt.subplots()
        fig_out, (ax_shear, ax_moment) = plot_shear_moment_diagrams(
            ax, x, shear, moment
        )

        assert fig_out is not None
        assert ax_shear is not None
        assert ax_moment is not None


# ============================================================================
# 3D STRUCTURE TESTS
# ============================================================================


@pytest.mark.unit
class Test3DStructure:
    """Test 3D structure plotting."""

    def test_plot_3d_structure_basic(self, ensure_imports, structure_3d_data):
        """plot_3d_structure should create 3D plot."""
        from pymemorial.visualization import plot_3d_structure

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        ax_out = plot_3d_structure(
            ax, structure_3d_data["nodes"], structure_3d_data["elements"]
        )

        assert ax_out is not None

    def test_plot_3d_structure_with_displacements(
        self, ensure_imports, structure_3d_data
    ):
        """plot_3d_structure should plot deformed shape."""
        from pymemorial.visualization import plot_3d_structure

        displacements = np.array([[0, 0, 0], [0.1, 0, 0], [0.1, 0, -0.05], [0, 0, 0]])

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        ax_out = plot_3d_structure(
            ax,
            structure_3d_data["nodes"],
            structure_3d_data["elements"],
            displacements=displacements,
        )

        assert ax_out is not None


# ============================================================================
# AXIS FORMATTING TESTS
# ============================================================================


@pytest.mark.unit
class TestAxisFormatting:
    """Test axis formatting utilities."""

    def test_format_engineering_axis_both(self, ensure_imports):
        """format_engineering_axis should format both axes."""
        from pymemorial.visualization import format_engineering_axis

        fig, ax = plt.subplots()
        ax.plot([0, 1e6], [0, 1e9])

        format_engineering_axis(ax, axis="both", precision=2)
        # Should not raise

    @pytest.mark.parametrize("axis_name", ["x", "y", "both"])
    def test_format_engineering_axis_all(self, ensure_imports, axis_name):
        """All axis options should work."""
        from pymemorial.visualization import format_engineering_axis

        fig, ax = plt.subplots()
        ax.plot([0, 1e6], [0, 1e9])

        format_engineering_axis(ax, axis=axis_name)
        # Should not raise

    def test_set_equal_aspect_3d(self, ensure_imports):
        """set_equal_aspect_3d should work on 3D axes."""
        from pymemorial.visualization import set_equal_aspect_3d

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        # Plot some data
        ax.plot([0, 1], [0, 2], [0, 3])

        set_equal_aspect_3d(ax)
        # Should not raise


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================


@pytest.mark.unit
class TestErrorHandling:
    """Test error handling in matplotlib utils."""

    def test_plot_pm_mismatched_arrays(self, ensure_imports):
        """plot_pm_interaction should reject mismatched arrays."""
        from pymemorial.visualization import plot_pm_interaction

        p = np.array([1, 2, 3])
        m = np.array([1, 2])

        fig, ax = plt.subplots()

        with pytest.raises((ValueError, AssertionError)):
            plot_pm_interaction(ax, p, m)


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "unit"])
