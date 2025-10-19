# tests/unit/visualization/test_05_integration.py
"""
Integration tests for visualization module.

Tests end-to-end workflows combining multiple components.
"""

import numpy as np
import pytest


# ============================================================================
# END-TO-END WORKFLOW TESTS
# ============================================================================


@pytest.mark.integration
class TestEndToEndWorkflows:
    """Test complete visualization workflows."""

    def test_workflow_matplotlib_pm_diagram(
        self, ensure_imports, simple_pm_data, test_output_dir
    ):
        """Complete workflow: create P-M diagram with matplotlib."""
        from pymemorial.visualization import plot_pm_interaction
        import matplotlib.pyplot as plt

        p, m = simple_pm_data

        # Create plot
        fig, ax = plt.subplots(figsize=(8, 6))
        plot_pm_interaction(ax, p, m, design_point=(0.4, 0.6))

        # Save
        output_file = test_output_dir / "integration_pm_matplotlib.png"
        fig.savefig(output_file, dpi=150)
        plt.close(fig)

        assert output_file.exists()

    @pytest.mark.requires_plotly
    def test_workflow_plotly_full_cycle(
        self, ensure_imports, skip_if_no_plotly, simple_pm_data, test_output_dir
    ):
        """Complete workflow: Factory → Create → Configure → Export."""
        from pymemorial.visualization import (
            create_visualizer,
            PlotConfig,
            ExportConfig,
            ImageFormat,
        )

        # Step 1: Create engine via factory
        viz = create_visualizer(engine="plotly")

        # Step 2: Configure
        config = PlotConfig(
            title="Integration Test P-M Diagram",
            width=1000,
            height=800,
            dpi=200,
        )
        viz.set_config(config)

        # Step 3: Create diagram
        p, m = simple_pm_data
        fig = viz.create_pm_diagram(p, m, design_point=(0.4, 0.6))

        # Step 4: Export HTML
        html_file = viz.export_static(
            fig,
            ExportConfig(
                filename=test_output_dir / "integration_pm_plotly.html",
                format=ImageFormat.HTML,
            ),
        )

        assert html_file.exists()

    def test_workflow_mixed_engines(
        self, ensure_imports, simple_pm_data, test_output_dir
    ):
        """Workflow using both matplotlib and Plotly (if available)."""
        from pymemorial.visualization import (
            create_visualizer,
            plot_pm_interaction,
            list_available_engines,
        )
        import matplotlib.pyplot as plt

        p, m = simple_pm_data
        engines = list_available_engines()

        # Matplotlib (always available)
        fig, ax = plt.subplots()
        plot_pm_interaction(ax, p, m)
        mpl_file = test_output_dir / "integration_mixed_matplotlib.png"
        fig.savefig(mpl_file)
        plt.close(fig)
        assert mpl_file.exists()

        # Plotly (if available)
        if "plotly" in engines:
            viz = create_visualizer(engine="plotly")
            fig = viz.create_pm_diagram(p, m)
            # Just check creation worked
            assert fig is not None


# ============================================================================
# MULTI-DIAGRAM TESTS
# ============================================================================


@pytest.mark.integration
class TestMultipleDiagrams:
    """Test creating multiple diagrams in sequence."""

    def test_multiple_matplotlib_diagrams(
        self, ensure_imports, simple_pm_data, moment_curvature_data, test_output_dir
    ):
        """Create multiple matplotlib diagrams in sequence."""
        from pymemorial.visualization import (
            plot_pm_interaction,
            plot_moment_curvature,
        )
        import matplotlib.pyplot as plt

        # Diagram 1: P-M
        p, m = simple_pm_data
        fig1, ax1 = plt.subplots()
        plot_pm_interaction(ax1, p, m)
        file1 = test_output_dir / "multi_pm.png"
        fig1.savefig(file1)
        plt.close(fig1)

        # Diagram 2: M-κ
        curvature, moment = moment_curvature_data
        fig2, ax2 = plt.subplots()
        plot_moment_curvature(ax2, curvature, moment)
        file2 = test_output_dir / "multi_mk.png"
        fig2.savefig(file2)
        plt.close(fig2)

        assert file1.exists()
        assert file2.exists()

    @pytest.mark.requires_plotly
    def test_multiple_plotly_diagrams(
        self,
        ensure_imports,
        skip_if_no_plotly,
        simple_pm_data,
        moment_curvature_data,
        test_output_dir,
    ):
        """Create multiple Plotly diagrams in sequence."""
        from pymemorial.visualization import create_visualizer, ExportConfig, ImageFormat

        viz = create_visualizer(engine="plotly")

        # Diagram 1: P-M
        p, m = simple_pm_data
        fig1 = viz.create_pm_diagram(p, m)
        file1 = viz.export_static(
            fig1,
            ExportConfig(
                filename=test_output_dir / "multi_plotly_pm.html",
                format=ImageFormat.HTML,
            ),
        )

        # Diagram 2: M-κ
        curvature, moment = moment_curvature_data
        fig2 = viz.create_moment_curvature(curvature, moment)
        file2 = viz.export_static(
            fig2,
            ExportConfig(
                filename=test_output_dir / "multi_plotly_mk.html",
                format=ImageFormat.HTML,
            ),
        )

        assert file1.exists()
        assert file2.exists()


# ============================================================================
# CROSS-ENGINE COMPATIBILITY TESTS
# ============================================================================


@pytest.mark.integration
class TestCrossEngineCompatibility:
    """Test that same data works across engines."""

    def test_same_data_all_engines(self, ensure_imports, simple_pm_data):
        """Same data should work with all available engines."""
        from pymemorial.visualization import list_available_engines, create_visualizer

        p, m = simple_pm_data
        engines = list_available_engines()

        for engine_name in engines:
            viz = create_visualizer(engine=engine_name)
            fig = viz.create_pm_diagram(p, m)
            assert fig is not None


# ============================================================================
# ERROR RECOVERY TESTS
# ============================================================================


@pytest.mark.integration
class TestErrorRecovery:
    """Test error handling and recovery."""

    def test_invalid_data_recovery(self, ensure_imports):
        """Engine should reject invalid data but remain functional."""
        from pymemorial.visualization import create_visualizer

        viz = create_visualizer()

        # Try invalid data
        try:
            viz.create_pm_diagram(np.array([]), np.array([]))
        except ValueError:
            pass  # Expected

        # Engine should still work after error
        p = np.array([0, 1])
        m = np.array([1, 0])
        fig = viz.create_pm_diagram(p, m)
        assert fig is not None


# ============================================================================
# PERFORMANCE TESTS (Optional - marked as slow)
# ============================================================================


@pytest.mark.integration
@pytest.mark.slow
class TestPerformance:
    """Test performance with larger datasets."""

    def test_large_pm_diagram(self, ensure_imports):
        """Engine should handle large P-M diagrams."""
        from pymemorial.visualization import create_visualizer

        viz = create_visualizer()

        # Large dataset (100 points)
        p = np.linspace(0, 1, 100)
        m = np.sqrt(1 - p**2)  # Quarter circle

        fig = viz.create_pm_diagram(p, m)
        assert fig is not None

    def test_large_3d_structure(self, ensure_imports):
        """Engine should handle larger 3D structures."""
        from pymemorial.visualization import create_visualizer

        viz = create_visualizer()

        # 10x10 grid (100 nodes, 180 elements)
        n = 10
        nodes = np.array([[i, j, 0] for i in range(n) for j in range(n)])

        elements = []
        for i in range(n - 1):
            for j in range(n - 1):
                node = i * n + j
                elements.extend(
                    [(node, node + 1), (node, node + n), (node, node + n + 1)]
                )

        fig = viz.create_3d_structure(nodes, elements)
        assert fig is not None


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])
