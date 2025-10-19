# tests/smoke/test_smoke_visualization.py
"""Smoke tests - validação básica de imports e instanciação."""

def test_imports_base():
    """Base module deve importar sem erros."""
    from pymemorial.visualization import base_visualizer
    assert hasattr(base_visualizer, 'VisualizerEngine')

def test_imports_matplotlib_utils():
    """Matplotlib utils sempre disponível."""
    from pymemorial.visualization import matplotlib_utils
    assert hasattr(matplotlib_utils, 'plot_pm_interaction')

def test_imports_plotly_optional():
    """Plotly pode ou não estar disponível."""
    try:
        from pymemorial.visualization import plotly_engine
        assert hasattr(plotly_engine, 'PlotlyEngine')
    except ImportError:
        pass  # OK se não instalado

def test_factory_create_without_crash():
    """Factory deve criar engine sem crash."""
    from pymemorial.visualization import create_visualizer
    viz = create_visualizer()
    assert viz is not None
    assert hasattr(viz, 'version')
