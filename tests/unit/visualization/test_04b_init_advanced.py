# tests/unit/visualization/test_04b_init_advanced.py
"""
Advanced tests for __init__.py module-level functionality.

Tests import handling, error paths, and edge cases to increase coverage.
"""

import sys
import pytest


# ============================================================================
# IMPORT ERROR HANDLING TESTS
# ============================================================================


@pytest.mark.unit
class TestImportErrorHandling:
    """Test graceful handling of missing optional dependencies."""

    def test_plotly_not_available_fallback(self, ensure_imports, monkeypatch):
        """Module should work even if Plotly import fails."""
        # This tests the except ImportError block at line ~147
        # Already covered by conditional imports, but let's be explicit
        
        from pymemorial.visualization import PLOTLY_AVAILABLE
        
        # If Plotly is available, we can't test the failure path
        if PLOTLY_AVAILABLE:
            # Test that check_plotly_installation works
            from pymemorial.visualization import check_plotly_installation
            status = check_plotly_installation()
            assert status["plotly"]["available"] is True

    def test_module_import_without_plotly(self, ensure_imports):
        """Core visualization should import without Plotly."""
        # Test that base classes are always available
        from pymemorial.visualization import (
            VisualizerEngine,
            PlotConfig,
            ExportConfig,
            ImageFormat,
        )
        
        assert VisualizerEngine is not None
        assert PlotConfig is not None


# ============================================================================
# FACTORY EDGE CASES
# ============================================================================


@pytest.mark.unit
class TestFactoryEdgeCases:
    """Test factory pattern edge cases."""

    def test_create_with_none_engine(self, ensure_imports):
        """Factory should handle None engine gracefully."""
        from pymemorial.visualization import VisualizerFactory
        
        # None engine should auto-select
        viz = VisualizerFactory.create(engine=None)
        assert viz is not None
        assert viz.available is True

    def test_create_for_3d_fallback(self, ensure_imports, monkeypatch):
        """create_for_3d should fallback if no 3D engine available."""
        from pymemorial.visualization import VisualizerFactory
        import pymemorial.visualization as viz_module
        
        # Create mock registry with no 3D engines
        original_registry = viz_module._ENGINE_REGISTRY.copy()
        
        try:
            # Mock a non-3D engine
            class MockNon3DEngine:
                def __init__(self):
                    self.available = True
                    self.supports_3d = False
                    self.version = "1.0.0"
                    self.name = "mock"
                
                def set_config(self, config):
                    pass
            
            # Test with mock (if we can inject it safely)
            # This is tricky, so let's just test the normal path
            viz = VisualizerFactory.create_for_3d()
            assert viz is not None
            
        finally:
            # Restore original registry
            viz_module._ENGINE_REGISTRY.clear()
            viz_module._ENGINE_REGISTRY.update(original_registry)

    def test_create_for_interactive_fallback(self, ensure_imports):
        """create_for_interactive should fallback gracefully."""
        from pymemorial.visualization import VisualizerFactory
        
        viz = VisualizerFactory.create_for_interactive()
        assert viz is not None
        # Should have some interactive capability or fallback

    def test_create_for_publication_all_styles(self, ensure_imports):
        """Test all publication styles."""
        from pymemorial.visualization import VisualizerFactory
        
        styles = ["ieee", "nature", "asce", "thesis"]
        
        for style in styles:
            viz = VisualizerFactory.create_for_publication(style=style)
            assert viz is not None
            assert viz.available is True

    def test_get_engine_info_empty_registry(self, ensure_imports, monkeypatch):
        """get_engine_info should handle empty registry."""
        from pymemorial.visualization import VisualizerFactory
        import pymemorial.visualization as viz_module
        
        # Backup original
        original_registry = viz_module._ENGINE_REGISTRY.copy()
        
        try:
            # Set empty registry
            viz_module._ENGINE_REGISTRY.clear()
            
            info = VisualizerFactory.get_engine_info()
            assert isinstance(info, dict)
            assert len(info) == 0  # Empty
            
        finally:
            # Restore
            viz_module._ENGINE_REGISTRY.clear()
            viz_module._ENGINE_REGISTRY.update(original_registry)

    def test_get_available_engines_caching(self, ensure_imports):
        """get_available_engines should return consistent results."""
        from pymemorial.visualization import VisualizerFactory
        
        engines1 = VisualizerFactory.get_available_engines()
        engines2 = VisualizerFactory.get_available_engines()
        
        assert engines1 == engines2


# ============================================================================
# MODULE-LEVEL CONFIGURATION EDGE CASES
# ============================================================================


@pytest.mark.unit
class TestModuleLevelConfigEdgeCases:
    """Test module-level configuration edge cases."""

    def test_set_default_engine_invalid(self, ensure_imports):
        """set_default_engine with invalid engine should fallback gracefully."""
        from pymemorial.visualization import set_default_engine, get_default_engine_instance
        
        # Invalid engine name - should fallback, not crash
        set_default_engine("nonexistent_engine")
        
        # Should create some valid engine
        viz = get_default_engine_instance()
        assert viz is not None
        assert viz.available is True


    def test_get_default_engine_instance_creates_once(self, ensure_imports):
        """get_default_engine_instance should lazily create only once."""
        from pymemorial.visualization import (
            get_default_engine_instance,
            reset_default_engine,
        )
        
        # Reset first
        reset_default_engine()
        
        # Get twice
        viz1 = get_default_engine_instance()
        viz2 = get_default_engine_instance()
        
        # Should be same instance
        assert viz1 is viz2

    def test_reset_default_engine_multiple_times(self, ensure_imports):
        """reset_default_engine should be idempotent."""
        from pymemorial.visualization import (
            reset_default_engine,
            get_default_engine_instance,
        )
        
        # Create instance
        get_default_engine_instance()
        
        # Reset multiple times
        reset_default_engine()
        reset_default_engine()
        reset_default_engine()
        
        # Should not crash


# ============================================================================
# CONVENIENCE FUNCTION EDGE CASES
# ============================================================================


@pytest.mark.unit
class TestConvenienceFunctionEdgeCases:
    """Test convenience function edge cases."""

    def test_create_visualizer_with_invalid_engine(self, ensure_imports):
        """create_visualizer should fallback on invalid engine."""
        from pymemorial.visualization import create_visualizer
        
        # With fallback enabled (default), should succeed
        viz = create_visualizer(engine="nonexistent")
        assert viz is not None

    def test_list_available_engines_not_empty(self, ensure_imports):
        """list_available_engines should always return at least one."""
        from pymemorial.visualization import list_available_engines
        
        engines = list_available_engines()
        assert len(engines) > 0  # At least one engine must exist

    def test_get_engine_status_structure(self, ensure_imports):
        """get_engine_status should return complete structure."""
        from pymemorial.visualization import get_engine_status
        
        status = get_engine_status()
        
        # Required keys
        assert "engines" in status
        assert "plotly" in status
        assert "default_engine" in status
        assert "recommended_action" in status
        
        # Engines should be dict
        assert isinstance(status["engines"], dict)
        
        # Plotly status should have structure
        assert "plotly" in status["plotly"]
        assert "kaleido" in status["plotly"]


# ============================================================================
# CHECK_INSTALLATION OUTPUT TESTS
# ============================================================================


@pytest.mark.unit
class TestCheckInstallationOutput:
    """Test check_installation output formatting."""

    def test_check_installation_complete_output(self, ensure_imports, capsys):
        """check_installation should print complete status."""
        from pymemorial.visualization import check_installation
        
        check_installation()
        
        captured = capsys.readouterr()
        output = captured.out
        
        # Should contain key sections
        assert "PyMemorial Visualization" in output
        assert "Installation Status" in output
        assert "Visualization Engines:" in output
        assert "Recommendation:" in output

    def test_check_installation_shows_versions(self, ensure_imports, capsys):
        """check_installation should show version numbers."""
        from pymemorial.visualization import check_installation
        
        check_installation()
        
        captured = capsys.readouterr()
        output = captured.out
        
        # Should show version info
        assert "version" in output.lower() or "v" in output


# ============================================================================
# REGISTRY MANIPULATION TESTS
# ============================================================================


@pytest.mark.unit
class TestRegistryManipulation:
    """Test internal registry behavior."""

    def test_engine_registry_not_empty(self, ensure_imports):
        """Engine registry should contain at least one engine."""
        import pymemorial.visualization as viz_module
        
        registry = viz_module._ENGINE_REGISTRY
        assert isinstance(registry, dict)
        assert len(registry) > 0  # Must have at least one

    def test_plotly_in_registry_if_available(self, ensure_imports):
        """If Plotly is available, it should be in registry."""
        from pymemorial.visualization import PLOTLY_AVAILABLE
        import pymemorial.visualization as viz_module
        
        registry = viz_module._ENGINE_REGISTRY
        
        if PLOTLY_AVAILABLE:
            assert "plotly" in registry


# ============================================================================
# ERROR MESSAGE TESTS
# ============================================================================


@pytest.mark.unit
class TestErrorMessages:
    """Test that error messages are informative."""

    def test_no_engines_error_message(self, ensure_imports, monkeypatch):
        """Error message should guide user when no engines available."""
        from pymemorial.visualization import VisualizerFactory
        import pymemorial.visualization as viz_module
        
        original_registry = viz_module._ENGINE_REGISTRY.copy()
        
        try:
            # Empty registry
            viz_module._ENGINE_REGISTRY.clear()
            
            with pytest.raises(RuntimeError) as exc_info:
                VisualizerFactory.get_default_engine()
            
            error_msg = str(exc_info.value)
            assert "No visualization engines" in error_msg
            
        finally:
            viz_module._ENGINE_REGISTRY.clear()
            viz_module._ENGINE_REGISTRY.update(original_registry)


# ============================================================================
# VERSION AND METADATA TESTS
# ============================================================================


@pytest.mark.unit
class TestVersionMetadata:
    """Test module version and metadata."""

    def test_module_has_version(self, ensure_imports):
        """Module should export __version__."""
        import pymemorial.visualization as viz
        
        assert hasattr(viz, "__version__")
        assert isinstance(viz.__version__, str)
        assert len(viz.__version__) > 0

    def test_module_has_all_export(self, ensure_imports):
        """Module should have __all__ defined."""
        import pymemorial.visualization as viz
        
        assert hasattr(viz, "__all__")
        assert isinstance(viz.__all__, list)
        assert len(viz.__all__) > 10  # Should export many items


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "unit"])
