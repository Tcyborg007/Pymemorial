# tests/unit/visualization/conftest.py
"""
Shared fixtures for visualization tests.

This conftest provides reusable test data, mocks, and configurations
for all visualization test modules.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# ============================================================================
# MARKERS CONFIGURATION
# ============================================================================


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "smoke: Smoke tests (fast, basic imports)"
    )
    config.addinivalue_line(
        "markers", "unit: Unit tests (isolated, mocked)"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests (multi-module)"
    )
    config.addinivalue_line(
        "markers", "requires_plotly: Requires Plotly installed"
    )
    config.addinivalue_line(
        "markers", "requires_kaleido: Requires Kaleido installed"
    )
    config.addinivalue_line(
        "markers", "slow: Tests that take >5 seconds"
    )


# ============================================================================
# PATH FIXTURES
# ============================================================================


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Return project root directory."""
    # From tests/unit/visualization/ -> go up 3 levels
    return Path(__file__).parent.parent.parent.parent


@pytest.fixture(scope="session")
def src_path(project_root) -> Path:
    """Return src directory path."""
    return project_root / "src"


@pytest.fixture(scope="session")
def test_output_dir(tmp_path_factory) -> Path:
    """Create temporary directory for test outputs."""
    output_dir = tmp_path_factory.mktemp("viz_test_outputs")
    return output_dir


# ============================================================================
# IMPORT FIXTURES
# ============================================================================


@pytest.fixture(scope="session")
def ensure_imports(src_path):
    """Ensure src is in Python path for imports."""
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    return src_path


# ============================================================================
# DATA FIXTURES - Reusable test data
# ============================================================================


@pytest.fixture
def simple_pm_data() -> tuple:
    """Simple P-M interaction data for testing."""
    p = np.array([0, 0.5, 1.0, 0.8, 0.3, 0])
    m = np.array([0.6, 0.8, 0.5, 0.4, 0.7, 0])
    return p, m


@pytest.fixture
def pm_data_with_points(simple_pm_data) -> dict:
    """P-M data with design and capacity points."""
    p, m = simple_pm_data
    return {
        "p": p,
        "m": m,
        "design_point": (0.4, 0.6),
        "capacity_point": (0.5, 1.0),
    }


@pytest.fixture
def moment_curvature_data() -> tuple:
    """Moment-curvature data for testing."""
    curvature = np.linspace(0, 0.01, 50)
    # Bilinear response
    moment = np.where(
        curvature < 0.005,
        1e6 * curvature,  # Elastic
        1e6 * (0.005 + 0.5 * (curvature - 0.005)),  # Plastic
    )
    return curvature, moment


@pytest.fixture
def structure_3d_data() -> dict:
    """3D structure data (simple frame)."""
    nodes = np.array([
        [0, 0, 0],
        [5, 0, 0],
        [5, 0, 3],
        [0, 0, 3],
    ])
    elements = [(0, 1), (1, 2), (2, 3), (3, 0)]
    return {"nodes": nodes, "elements": elements}


@pytest.fixture
def section_2d_data() -> dict:
    """2D section data (rectangular)."""
    vertices = np.array([
        [0, 0],
        [0.3, 0],
        [0.3, 0.5],
        [0, 0.5],
    ])
    facets = [[0, 1, 2, 3]]
    materials = ["steel"]
    return {"vertices": vertices, "facets": facets, "materials": materials}


@pytest.fixture
def stress_contour_data() -> dict:
    """Stress contour meshgrid data."""
    x = np.linspace(0, 1, 30)
    y = np.linspace(0, 1, 30)
    X, Y = np.meshgrid(x, y)
    # Simple radial stress field
    stress = 1e6 * (X**2 + Y**2)
    return {"x": X, "y": Y, "stress": stress}


# ============================================================================
# CONFIGURATION FIXTURES
# ============================================================================


@pytest.fixture
def basic_plot_config(ensure_imports):
    """Basic PlotConfig for testing."""
    from pymemorial.visualization import PlotConfig
    
    return PlotConfig(
        title="Test Plot",
        xlabel="X Axis",
        ylabel="Y Axis",
        width=800,
        height=600,
    )


@pytest.fixture
def export_config_png(test_output_dir, ensure_imports):
    """ExportConfig for PNG output."""
    from pymemorial.visualization import ExportConfig, ImageFormat
    
    return ExportConfig(
        filename=test_output_dir / "test_output.png",
        format=ImageFormat.PNG,
        scale=1.0,
    )


@pytest.fixture
def export_config_html(test_output_dir, ensure_imports):
    """ExportConfig for HTML output."""
    from pymemorial.visualization import ExportConfig, ImageFormat
    
    return ExportConfig(
        filename=test_output_dir / "test_output.html",
        format=ImageFormat.HTML,
    )


# ============================================================================
# ENGINE FIXTURES - Conditional availability
# ============================================================================


@pytest.fixture
def plotly_available(ensure_imports) -> bool:
    """Check if Plotly is available."""
    try:
        from pymemorial.visualization import PLOTLY_AVAILABLE
        return PLOTLY_AVAILABLE
    except ImportError:
        return False


@pytest.fixture
def kaleido_available(ensure_imports) -> bool:
    """Check if Kaleido is available."""
    try:
        from pymemorial.visualization import KALEIDO_AVAILABLE
        return KALEIDO_AVAILABLE
    except ImportError:
        return False


@pytest.fixture
def skip_if_no_plotly(plotly_available):
    """Skip test if Plotly not available."""
    if not plotly_available:
        pytest.skip("Plotly not installed (pip install pymemorial[viz])")


@pytest.fixture
def skip_if_no_kaleido(kaleido_available):
    """Skip test if Kaleido not available."""
    if not kaleido_available:
        pytest.skip("Kaleido not installed (pip install kaleido)")


# ============================================================================
# MOCK FIXTURES - For isolated testing
# ============================================================================


@pytest.fixture
def mock_matplotlib_figure(monkeypatch):
    """Mock matplotlib figure for testing without actual plotting."""
    import matplotlib.pyplot as plt
    from unittest.mock import MagicMock
    
    mock_fig = MagicMock()
    mock_ax = MagicMock()
    
    def mock_subplots(*args, **kwargs):
        return mock_fig, mock_ax
    
    monkeypatch.setattr(plt, "subplots", mock_subplots)
    return mock_fig, mock_ax


# ============================================================================
# CLEANUP FIXTURES
# ============================================================================


@pytest.fixture(autouse=True)
def cleanup_matplotlib():
    """Cleanup matplotlib state after each test."""
    yield
    import matplotlib.pyplot as plt
    plt.close("all")


# ============================================================================
# PARAMETRIZE HELPERS
# ============================================================================


# Invalid array sizes for parametrized tests
INVALID_ARRAY_CASES = [
    pytest.param(np.array([]), np.array([]), id="empty_arrays"),
    pytest.param(np.array([1]), np.array([1, 2]), id="mismatched_length"),
    pytest.param(np.array([1]), np.array([]), id="one_empty"),
]

# Valid small array cases (edge cases)
VALID_SMALL_CASES = [
    pytest.param(np.array([0, 1]), np.array([1, 0]), id="two_points"),
    pytest.param(np.array([0, 1, 0]), np.array([1, 0, 1]), id="three_points"),
]
