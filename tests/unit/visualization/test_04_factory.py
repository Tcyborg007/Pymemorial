# tests/unit/visualization/test_04_factory.py
"""
Unit tests for VisualizerFactory and factory pattern.

Tests engine creation, fallbacks, and capability-based selection.
"""

import pytest


# ============================================================================
# FACTORY PATTERN TESTS
# ============================================================================


@pytest.mark.unit
class TestVisualizerFactory:
    """Test VisualizerFactory class."""

    def test_create_default(self, ensure_imports):
        """Factory should create default engine."""
        from pymemorial.visualization import VisualizerFactory

        viz = VisualizerFactory.create()
        assert viz is not None
        assert viz.available is True

    def test_create_with_fallback(self, ensure_imports):
        """Factory should fallback if preferred unavailable."""
        from pymemorial.visualization import VisualizerFactory

        # Request engine that may not exist, with fallback
        viz = VisualizerFactory.create(engine="pyvista", fallback=True)
        assert viz is not None
        assert viz.available is True

    def test_create_no_fallback_fails(self, ensure_imports, plotly_available):
        """Factory should fail without fallback if engine unavailable."""
        from pymemorial.visualization import VisualizerFactory

        if plotly_available:
            pytest.skip("Plotly available, can't test failure case")

        # Request Plotly without fallback - should fail
        with pytest.raises(ValueError, match="not available"):
            VisualizerFactory.create(engine="plotly", fallback=False)

    def test_create_with_config(self, ensure_imports, basic_plot_config):
        """Factory should apply custom config."""
        from pymemorial.visualization import VisualizerFactory

        viz = VisualizerFactory.create(config=basic_plot_config)
        assert viz.get_config() is not None
        assert viz.get_config().title == "Test Plot"

    def test_create_for_3d(self, ensure_imports):
        """Factory should create engine with 3D support."""
        from pymemorial.visualization import VisualizerFactory

        viz = VisualizerFactory.create_for_3d()
        assert viz.supports_3d is True

    def test_create_for_interactive(self, ensure_imports):
        """Factory should create interactive-capable engine."""
        from pymemorial.visualization import VisualizerFactory

        viz = VisualizerFactory.create_for_interactive()
        assert viz.supports_interactive is True

    @pytest.mark.parametrize(
        "style", ["ieee", "nature", "asce", "thesis"]
    )
    def test_create_for_publication(self, ensure_imports, style):
        """Factory should create publication-configured engines."""
        from pymemorial.visualization import VisualizerFactory

        viz = VisualizerFactory.create_for_publication(style=style)
        assert viz is not None

    def test_get_available_engines(self, ensure_imports):
        """Factory should list available engines."""
        from pymemorial.visualization import VisualizerFactory

        engines = VisualizerFactory.get_available_engines()
        assert isinstance(engines, list)
        assert len(engines) > 0

    def test_get_default_engine(self, ensure_imports):
        """Factory should return default engine name."""
        from pymemorial.visualization import VisualizerFactory

        default = VisualizerFactory.get_default_engine()
        assert isinstance(default, str)
        assert len(default) > 0

    def test_get_engine_info(self, ensure_imports):
        """Factory should return engine information."""
        from pymemorial.visualization import VisualizerFactory

        info = VisualizerFactory.get_engine_info()
        assert isinstance(info, dict)
        # Should have at least one engine
        assert len(info) > 0

        # Each engine should have required keys
        for engine_name, engine_info in info.items():
            if engine_info.get("available"):
                assert "version" in engine_info
                assert "supports_3d" in engine_info
                assert "supports_interactive" in engine_info


# ============================================================================
# CONVENIENCE FUNCTION TESTS
# ============================================================================


@pytest.mark.unit
class TestConvenienceFunctions:
    """Test convenience wrapper functions."""

    def test_create_visualizer_default(self, ensure_imports):
        """create_visualizer should work with defaults."""
        from pymemorial.visualization import create_visualizer

        viz = create_visualizer()
        assert viz is not None
        assert viz.available is True

    def test_create_visualizer_with_config(self, ensure_imports, basic_plot_config):
        """create_visualizer should accept config."""
        from pymemorial.visualization import create_visualizer

        viz = create_visualizer(config=basic_plot_config)
        assert viz.get_config() is not None

    def test_list_available_engines(self, ensure_imports):
        """list_available_engines should return list."""
        from pymemorial.visualization import list_available_engines

        engines = list_available_engines()
        assert isinstance(engines, list)
        assert len(engines) > 0

    def test_get_engine_status(self, ensure_imports):
        """get_engine_status should return comprehensive status."""
        from pymemorial.visualization import get_engine_status

        status = get_engine_status()
        assert isinstance(status, dict)
        assert "engines" in status
        assert "plotly" in status
        assert "default_engine" in status
        assert "recommended_action" in status

    def test_check_installation(self, ensure_imports, capsys):
        """check_installation should print status."""
        from pymemorial.visualization import check_installation

        check_installation()

        captured = capsys.readouterr()
        assert "PyMemorial Visualization" in captured.out
        assert "Installation Status" in captured.out


# ============================================================================
# MODULE-LEVEL CONFIGURATION TESTS
# ============================================================================


@pytest.mark.unit
class TestModuleConfiguration:
    """Test module-level configuration."""

    def test_set_default_engine(self, ensure_imports):
        """set_default_engine should configure default."""
        from pymemorial.visualization import (
            set_default_engine,
            get_default_engine_instance,
            reset_default_engine,
            list_available_engines,
        )

        engines = list_available_engines()
        if not engines:
            pytest.skip("No engines available")

        # Set default
        engine_name = engines[0]
        set_default_engine(engine_name)

        # Get instance
        viz = get_default_engine_instance()
        assert viz.name == engine_name

        # Reset
        reset_default_engine()

    def test_get_default_engine_instance_lazy(self, ensure_imports):
        """get_default_engine_instance should lazily create."""
        from pymemorial.visualization import (
            get_default_engine_instance,
            reset_default_engine,
        )

        # Reset to ensure clean state
        reset_default_engine()

        # Get instance (should create)
        viz = get_default_engine_instance()
        assert viz is not None

    def test_reset_default_engine(self, ensure_imports):
        """reset_default_engine should clear default."""
        from pymemorial.visualization import (
            set_default_engine,
            reset_default_engine,
            list_available_engines,
        )

        engines = list_available_engines()
        if not engines:
            pytest.skip("No engines available")

        # Set and reset
        set_default_engine(engines[0])
        reset_default_engine()
        # Should not crash


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================


@pytest.mark.unit
class TestFactoryErrorHandling:
    """Test factory error handling."""

    def test_no_engines_available_error(self, ensure_imports, monkeypatch):
        """Factory should fail gracefully if no engines available."""
        from pymemorial.visualization import VisualizerFactory
        import pymemorial.visualization as viz_module

        # Mock the module-level registry (not class attribute)
        monkeypatch.setattr(viz_module, "_ENGINE_REGISTRY", {})

        with pytest.raises(RuntimeError, match="No visualization engines"):
            VisualizerFactory.get_default_engine()



# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "unit"])
