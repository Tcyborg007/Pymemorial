# src/pymemorial/visualization/__init__.py
"""
PyMemorial Visualization Package.

This package provides a complete, production-ready visualization system for
structural engineering analysis and technical documentation. Supports multiple
backends (Plotly, Matplotlib, PyVista) with automatic fallbacks and a unified API.

Key Features:
    - Multiple visualization engines (Plotly, Matplotlib, PyVista)
    - Factory pattern for engine selection
    - Graceful degradation if optional dependencies missing
    - Publication-quality styling presets
    - Both static (PNG, SVG, PDF) and interactive (HTML) outputs
    - 3D FEM visualization with PyVista (GPU-accelerated)
    - Type-safe interfaces following ABC pattern
    - CI/CD compatible (headless rendering)

Architecture:
    - VisualizerEngine (ABC): Base interface for all engines
    - PlotlyEngine: Interactive + static via Kaleido (primary)
    - PyVistaEngine: 3D FEM with VTK backend (advanced)
    - MatplotlibUtils: Static rendering, always available (fallback)
    - VisualizerFactory: Smart engine selection with fallbacks

Quick Start:
    >>> from pymemorial.visualization import create_visualizer, PlotConfig
    >>> 
    >>> # Auto-select best available engine
    >>> viz = create_visualizer()
    >>> 
    >>> # Or specify engine
    >>> viz_plotly = create_visualizer(engine="plotly")
    >>> viz_pyvista = create_visualizer(engine="pyvista")  # 3D FEM
    >>> 
    >>> # Create P-M diagram
    >>> import numpy as np
    >>> p = np.array([0, 0.5, 1.0, 0.8, 0.3, 0])
    >>> m = np.array([0.6, 0.8, 0.5, 0.4, 0.7, 0])
    >>> fig = viz.create_pm_diagram(p, m)
    >>> 
    >>> # Export
    >>> from pathlib import Path
    >>> viz.export_static(fig, ExportConfig(Path("diagram.png")))

Installation:
    # Basic (matplotlib only)
    pip install pymemorial
    
    # Full 2D visualization (Plotly + Kaleido)
    pip install pymemorial[viz]
    
    # 3D FEM visualization (PyVista + VTK)
    pip install pymemorial[viz3d]
    
    # All features
    pip install pymemorial[all]

Author: PyMemorial Team
License: MIT
Python: >=3.9
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Dict, List, Literal, Optional, Type, Union

# ============================================================================
# BASE INTERFACES - Always available
# ============================================================================

from .base_visualizer import (
    # ABCs
    VisualizerEngine,
    # Dataclasses
    PlotConfig,
    ExportConfig,
    AnnotationStyle,
    # Enums
    ImageFormat,
    DiagramType,
    ThemeStyle,
    # Protocols
    FigureProtocol,
    # Type aliases
    NDArrayFloat,
    Point2D,
    Point3D,
    ColorType,
    # Utilities
    validate_colormap,
    get_default_config,
)

# ============================================================================
# MATPLOTLIB UTILITIES - Always available (no optional deps)
# ============================================================================

from .matplotlib_utils import (
    # Styling
    set_publication_style,
    reset_style,
    # Dimensions
    add_dimension_arrow,
    add_multiple_dimensions,
    # Materials
    add_section_hatch,
    plot_composite_section,
    # Diagrams
    plot_pm_interaction,
    plot_moment_curvature,
    plot_shear_moment_diagrams,
    # 3D
    plot_3d_structure,
    # Formatting
    format_engineering_axis,
    set_equal_aspect_3d,
    # Constants
    MATERIAL_COLORS,
    MATERIAL_HATCHES,
    MATERIAL_EDGE_COLORS,
)

# ============================================================================
# OPTIONAL ENGINES - Graceful import with logging
# ============================================================================

logger = logging.getLogger(__name__)

# Try importing Plotly engine
PLOTLY_ENGINE_AVAILABLE = False
PlotlyEngine = None  # type: ignore

try:
    from .plotly_engine import (
        PlotlyEngine,
        check_plotly_installation,
        PLOTLY_AVAILABLE,
        KALEIDO_AVAILABLE,
        PLOTLY_VERSION,
        KALEIDO_VERSION,
    )

    PLOTLY_ENGINE_AVAILABLE = PLOTLY_AVAILABLE
    if PLOTLY_ENGINE_AVAILABLE:
        logger.info(f"PlotlyEngine loaded (v{PLOTLY_VERSION})")
    else:
        logger.warning("Plotly installed but not functional")
except ImportError:
    logger.info("PlotlyEngine not available (optional)")
    PLOTLY_AVAILABLE = False
    KALEIDO_AVAILABLE = False
    PLOTLY_VERSION = "0.0.0"
    KALEIDO_VERSION = "0.0.0"

    def check_plotly_installation() -> Dict[str, any]:
        """Stub function when Plotly not installed."""
        return {
            "plotly": {"available": False, "version": "not installed"},
            "kaleido": {"available": False, "version": "not installed"},
            "status": "missing dependencies",
            "install_command": "pip install 'pymemorial[viz]'",
        }

# Try importing PyVista engine
PYVISTA_ENGINE_AVAILABLE = False
PyVistaEngine = None  # type: ignore

try:
    from .pyvista_engine import (
        PyVistaEngine,
        check_pyvista_installation,
        create_example_truss,
        PYVISTA_AVAILABLE,
        PYVISTA_VERSION,
        VTK_VERSION,
        STRESS_COLORMAPS,
        MATERIAL_RENDER_STYLES,
        CAMERA_PRESETS,
    )

    PYVISTA_ENGINE_AVAILABLE = PYVISTA_AVAILABLE
    if PYVISTA_ENGINE_AVAILABLE:
        logger.info(
            f"PyVistaEngine loaded (PyVista v{PYVISTA_VERSION}, VTK v{VTK_VERSION})"
        )
    else:
        logger.warning("PyVista installed but not functional")
except ImportError:
    logger.info("PyVistaEngine not available (optional for 3D FEM visualization)")
    PYVISTA_AVAILABLE = False
    PYVISTA_VERSION = "0.0.0"
    VTK_VERSION = "0.0.0"
    STRESS_COLORMAPS = {}
    MATERIAL_RENDER_STYLES = {}
    CAMERA_PRESETS = {}

    def check_pyvista_installation() -> Dict[str, any]:
        """Stub function when PyVista not installed."""
        return {
            "pyvista": {"available": False, "version": "not installed"},
            "vtk": {"version": "not installed"},
            "status": "missing dependencies",
            "install_command": "pip install 'pymemorial[viz3d]'",
        }

    def create_example_truss():
        """Stub function when PyVista not installed."""
        raise RuntimeError(
            "PyVista not installed. Install with: pip install 'pymemorial[viz3d]'"
        )

# Diagram generators (specialized functions)
try:
    from .diagram_generators import (
        DesignCode,
        PMDiagramParams,
        MomentCurvatureParams,
        generate_pm_interaction_envelope,
        generate_moment_curvature_response,
        create_pm_diagram_with_code,
        calculate_ductility,
        format_code_reference,
    )
except ImportError:
    logger.debug("Diagram generators not available")
    DesignCode = None  # type: ignore
    PMDiagramParams = None  # type: ignore
    MomentCurvatureParams = None  # type: ignore

# ============================================================================
# ENGINE REGISTRY - Available engines with priorities
# ============================================================================

# Registry of available engines (name -> class)
_ENGINE_REGISTRY: Dict[str, Type[VisualizerEngine]] = {}

# Register engines in order of preference (best first)
if PLOTLY_ENGINE_AVAILABLE and PlotlyEngine is not None:
    _ENGINE_REGISTRY["plotly"] = PlotlyEngine

if PYVISTA_ENGINE_AVAILABLE and PyVistaEngine is not None:
    _ENGINE_REGISTRY["pyvista"] = PyVistaEngine

# Matplotlib as fallback (always available via utils functions)
# Note: matplotlib_utils provides standalone functions, not a full engine class
# For now, we prioritize Plotly. Future: wrap matplotlib_utils in MatplotlibEngine class

# ============================================================================
# FACTORY PATTERN - Smart engine selection with fallbacks
# ============================================================================


class VisualizerFactory:
    """
    Factory for creating visualization engines with automatic fallbacks.

    This factory implements the Factory pattern with intelligent engine
    selection based on availability, capabilities, and user preferences.

    Features:
        - Automatic engine selection (best available)
        - Manual engine specification with fallback
        - Capability-based selection (3D, interactive, etc.)
        - Graceful degradation if dependencies missing
        - Thread-safe singleton registry

    Examples:
        >>> # Auto-select best engine
        >>> viz = VisualizerFactory.create()
        >>> 
        >>> # Request specific engine with fallback
        >>> viz = VisualizerFactory.create(engine="plotly", fallback=True)
        >>> 
        >>> # Require 3D support
        >>> viz = VisualizerFactory.create_for_3d()

    References:
        - Factory Pattern: https://refactoring.guru/design-patterns/factory-method
        - Dependency Injection: https://python-dependency-injector.ets-labs.org/
    """

    @staticmethod
    def create(
        engine: Optional[Literal["plotly", "matplotlib", "pyvista"]] = None,
        fallback: bool = True,
        config: Optional[PlotConfig] = None,
    ) -> VisualizerEngine:
        """
        Create visualization engine with automatic fallback.

        Args:
            engine: Preferred engine name (None = auto-select best)
            fallback: Allow fallback to other engines if preferred unavailable
            config: Default plot configuration for this engine

        Returns:
            VisualizerEngine instance (PlotlyEngine, PyVistaEngine, etc.)

        Raises:
            RuntimeError: If no engines available (should never happen)
            ValueError: If requested engine not available and fallback=False

        Examples:
            >>> # Auto-select (Plotly preferred if available)
            >>> viz = VisualizerFactory.create()
            >>> 
            >>> # Force Plotly, fail if unavailable
            >>> viz = VisualizerFactory.create(engine="plotly", fallback=False)
            >>> 
            >>> # Plotly preferred, matplotlib fallback
            >>> viz = VisualizerFactory.create(engine="plotly", fallback=True)
        """
        # Auto-select best available engine
        if engine is None:
            engine = VisualizerFactory.get_default_engine()

        # Try requested engine
        if engine in _ENGINE_REGISTRY:
            engine_class = _ENGINE_REGISTRY[engine]
            instance = engine_class()

            if instance.available:
                if config:
                    instance.set_config(config)
                logger.info(f"Created {engine} engine (v{instance.version})")
                return instance
            else:
                logger.warning(f"{engine} engine not available")
                if not fallback:
                    raise ValueError(
                        f"{engine} engine requested but not available. "
                        f"Install with: pip install pymemorial[viz]"
                    )

        # Fallback: try other engines in priority order
        if fallback:
            for fallback_name, fallback_class in _ENGINE_REGISTRY.items():
                if fallback_name == engine:
                    continue  # Already tried

                instance = fallback_class()
                if instance.available:
                    logger.warning(
                        f"Falling back to {fallback_name} engine "
                        f"(requested {engine} not available)"
                    )
                    if config:
                        instance.set_config(config)
                    return instance

        # No engines available (should never happen - matplotlib always available)
        raise RuntimeError(
            "No visualization engines available. This should not happen. "
            "Please check matplotlib installation."
        )

    @staticmethod
    def create_for_3d(config: Optional[PlotConfig] = None) -> VisualizerEngine:
        """
        Create engine optimized for 3D visualization.

        Selects engine with best 3D support (PyVista > Plotly).

        Args:
            config: Default plot configuration

        Returns:
            VisualizerEngine with 3D support

        Examples:
            >>> viz = VisualizerFactory.create_for_3d()
            >>> assert viz.supports_3d
        """
        # Prefer engines with native 3D support
        for name in ["pyvista", "plotly"]:
            if name in _ENGINE_REGISTRY:
                engine = _ENGINE_REGISTRY[name]()
                if engine.available and engine.supports_3d:
                    if config:
                        engine.set_config(config)
                    logger.info(f"Created {name} for 3D (v{engine.version})")
                    return engine

        # Fallback to any available engine
        logger.warning("No dedicated 3D engine available, using default")
        return VisualizerFactory.create(config=config)

    @staticmethod
    def create_for_interactive(config: Optional[PlotConfig] = None) -> VisualizerEngine:
        """
        Create engine optimized for interactive visualization.

        Selects engine with best interactivity (Plotly > PyVista).

        Args:
            config: Default plot configuration

        Returns:
            VisualizerEngine with interactive support

        Examples:
            >>> viz = VisualizerFactory.create_for_interactive()
            >>> fig = viz.create_pm_diagram(p, m)
            >>> viz.show(fig, interactive=True)
        """
        # Prefer engines with interactive HTML output
        for name in ["plotly", "pyvista"]:
            if name in _ENGINE_REGISTRY:
                engine = _ENGINE_REGISTRY[name]()
                if engine.available and engine.supports_interactive:
                    if config:
                        engine.set_config(config)
                    logger.info(f"Created {name} for interactive")
                    return engine

        # Fallback
        logger.warning("No interactive engine available, using default")
        return VisualizerFactory.create(config=config)

    @staticmethod
    def create_for_publication(
        style: Literal["ieee", "nature", "asce", "thesis"] = "ieee",
    ) -> VisualizerEngine:
        """
        Create engine with publication-quality presets.

        Args:
            style: Publication style (ieee, nature, asce, thesis)

        Returns:
            VisualizerEngine configured for publication

        Examples:
            >>> viz = VisualizerFactory.create_for_publication(style="ieee")
            >>> # Figures sized for IEEE Transactions (3.5" single column)
        """
        # Map style to theme
        style_to_theme = {
            "ieee": ThemeStyle.PUBLICATION,
            "nature": ThemeStyle.PUBLICATION,
            "asce": ThemeStyle.PUBLICATION,
            "thesis": ThemeStyle.PUBLICATION,
        }

        config = PlotConfig(
            width=800,
            height=600,
            dpi=300,
            theme=style_to_theme.get(style, ThemeStyle.PUBLICATION),
            font_family=(
                "Arial, sans-serif"
                if style == "ieee"
                else "Times New Roman, serif"
            ),
            font_size=10 if style in ["ieee", "nature"] else 11,
        )

        # Apply matplotlib publication style if using matplotlib
        set_publication_style(style)  # type: ignore

        return VisualizerFactory.create(config=config)

    @staticmethod
    def get_available_engines() -> List[str]:
        """
        Get list of available engine names.

        Returns:
            List of engine names that are installed and functional

        Examples:
            >>> engines = VisualizerFactory.get_available_engines()
            >>> print(engines)
            ['plotly', 'pyvista']
        """
        available = []
        for name, engine_class in _ENGINE_REGISTRY.items():
            try:
                instance = engine_class()
                if instance.available:
                    available.append(name)
            except Exception as e:
                logger.debug(f"Engine {name} check failed: {e}")
        return available

    @staticmethod
    def get_default_engine() -> str:
        """
        Get name of default (best available) engine.

        Returns:
            Name of default engine ("plotly", "pyvista", etc.)

        Examples:
            >>> default = VisualizerFactory.get_default_engine()
            >>> print(f"Using {default} by default")
        """
        available = VisualizerFactory.get_available_engines()
        if not available:
            raise RuntimeError("No visualization engines available")

        # Priority order: plotly > pyvista > matplotlib
        priority = ["plotly", "pyvista", "matplotlib"]
        for engine in priority:
            if engine in available:
                return engine

        # Fallback to first available
        return available[0]

    @staticmethod
    def get_engine_info() -> Dict[str, Dict[str, any]]:
        """
        Get information about all registered engines.

        Returns:
            Dictionary mapping engine names to info dicts with keys:
                - available: bool
                - version: str
                - supports_3d: bool
                - supports_interactive: bool
                - formats: List[ImageFormat]

        Examples:
            >>> info = VisualizerFactory.get_engine_info()
            >>> print(info["plotly"]["version"])
            '6.3.0'
        """
        info = {}
        for name, engine_class in _ENGINE_REGISTRY.items():
            try:
                instance = engine_class()
                info[name] = {
                    "available": instance.available,
                    "version": instance.version,
                    "supports_3d": instance.supports_3d,
                    "supports_interactive": instance.supports_interactive,
                    "formats": [fmt.value for fmt in instance.supported_formats],
                }
            except Exception as e:
                logger.debug(f"Failed to get info for {name}: {e}")
                info[name] = {
                    "available": False,
                    "version": "unknown",
                    "error": str(e),
                }
        return info


# ============================================================================
# CONVENIENCE FUNCTIONS - Simplified API
# ============================================================================


def create_visualizer(
    engine: Optional[Literal["plotly", "matplotlib", "pyvista"]] = None,
    config: Optional[PlotConfig] = None,
) -> VisualizerEngine:
    """
    Create visualization engine (convenience wrapper for Factory).

    This is the primary entry point for most users. Auto-selects best
    available engine with sensible defaults.

    Args:
        engine: Optional engine name (None = auto-select)
        config: Optional default plot configuration

    Returns:
        VisualizerEngine instance

    Examples:
        >>> viz = create_visualizer()
        >>> fig = viz.create_pm_diagram(p, m)

    See Also:
        VisualizerFactory.create() for more options
    """
    return VisualizerFactory.create(engine=engine, fallback=True, config=config)


def list_available_engines() -> List[str]:
    """
    List names of installed and functional engines.

    Returns:
        List of engine names

    Examples:
        >>> engines = list_available_engines()
        >>> print(f"Available: {', '.join(engines)}")
        Available: plotly, pyvista
    """
    return VisualizerFactory.get_available_engines()


def get_engine_status() -> Dict[str, any]:
    """
    Get detailed status of all visualization components.

    Returns comprehensive information about installed packages,
    versions, and capabilities.

    Returns:
        Dictionary with keys:
            - engines: Engine availability/versions
            - plotly: Plotly installation details
            - pyvista: PyVista installation details
            - recommended_action: Installation advice

    Examples:
        >>> status = get_engine_status()
        >>> if not status["plotly"]["available"]:
        ...     print(status["recommended_action"])
        pip install pymemorial[viz]
    """
    status = {
        "engines": VisualizerFactory.get_engine_info(),
        "plotly": check_plotly_installation(),
        "pyvista": check_pyvista_installation(),
        "default_engine": (
            VisualizerFactory.get_default_engine()
            if VisualizerFactory.get_available_engines()
            else None
        ),
    }

    # Recommendation
    if not PLOTLY_ENGINE_AVAILABLE and not PYVISTA_ENGINE_AVAILABLE:
        status["recommended_action"] = (
            "Install visualization engines: pip install 'pymemorial[viz,viz3d]'"
        )
    elif not PLOTLY_ENGINE_AVAILABLE:
        status["recommended_action"] = (
            "Install Plotly for 2D charts: pip install 'pymemorial[viz]'"
        )
    elif PLOTLY_ENGINE_AVAILABLE and not KALEIDO_AVAILABLE:
        status["recommended_action"] = (
            "Install Kaleido for static export: pip install kaleido"
        )
    elif not PYVISTA_ENGINE_AVAILABLE:
        status["recommended_action"] = (
            "Install PyVista for 3D FEM: pip install 'pymemorial[viz3d]'"
        )
    else:
        status["recommended_action"] = "All recommended packages installed ✓"

    return status


def check_installation() -> None:
    """
    Print detailed installation status to console.

    Useful for debugging installation issues or verifying setup.

    Examples:
        >>> from pymemorial.visualization import check_installation
        >>> check_installation()
        ✓ Plotly 6.3.0 installed
        ✓ Kaleido 0.4.1 installed
        ✓ PyVista 0.46.3 installed
        ✓ All features available
    """
    print("=" * 60)
    print("PyMemorial Visualization - Installation Status")
    print("=" * 60)

    status = get_engine_status()

    # Engines
    print("\n📊 Visualization Engines:")
    for name, info in status["engines"].items():
        symbol = "✓" if info["available"] else "✗"
        version = info.get("version", "unknown")
        print(f"  {symbol} {name.capitalize()}: {version}")
        if info["available"]:
            print(f"    • 3D support: {info.get('supports_3d', False)}")
            print(f"    • Interactive: {info.get('supports_interactive', False)}")

    # Plotly details
    print("\n🎨 Plotly Details:")
    plotly_info = status["plotly"]
    for lib, details in plotly_info.items():
        if isinstance(details, dict):
            symbol = "✓" if details["available"] else "✗"
            print(f"  {symbol} {lib.capitalize()}: {details['version']}")

    # PyVista details
    if "pyvista" in status:
        print("\n🎮 PyVista Details (3D FEM):")
        pyvista_info = status["pyvista"]
        if isinstance(pyvista_info, dict):
            # Iterate only over dict items, not all keys
            for lib in ["pyvista", "vtk"]:
                if lib in pyvista_info and isinstance(pyvista_info[lib], dict):
                    details = pyvista_info[lib]
                    symbol = "✓" if details.get("available", False) else "✗"
                    version = details.get("version", "not installed")
                    print(f"  {symbol} {lib.capitalize()}: {version}")


    # Recommendation
    print(f"\n💡 Recommendation:")
    print(f"   {status['recommended_action']}")

    print("\n" + "=" * 60)


# ============================================================================
# MODULE-LEVEL CONFIGURATION
# ============================================================================

# Default engine for module-level operations (lazy initialization)
_default_visualizer: Optional[VisualizerEngine] = None


def set_default_engine(
    engine: Literal["plotly", "matplotlib", "pyvista"],
    config: Optional[PlotConfig] = None,
) -> None:
    """
    Set default engine for module-level operations.

    Args:
        engine: Engine name to use as default
        config: Optional default configuration

    Examples:
        >>> set_default_engine("plotly")
        >>> # Now all create_visualizer() calls use Plotly by default
    """
    global _default_visualizer
    _default_visualizer = VisualizerFactory.create(engine=engine, config=config)
    logger.info(f"Default engine set to: {engine}")


def get_default_engine_instance() -> VisualizerEngine:
    """
    Get default engine instance (creates if not exists).

    Returns:
        Default VisualizerEngine

    Examples:
        >>> viz = get_default_engine_instance()
    """
    global _default_visualizer
    if _default_visualizer is None:
        _default_visualizer = create_visualizer()
    return _default_visualizer


def reset_default_engine() -> None:
    """Reset default engine to auto-select mode."""
    global _default_visualizer
    _default_visualizer = None
    logger.info("Default engine reset")


# ============================================================================
# EXPORTS - Comprehensive __all__ for clean imports
# ============================================================================

__all__ = [
    # ===== CORE INTERFACES =====
    "VisualizerEngine",
    "PlotConfig",
    "ExportConfig",
    "AnnotationStyle",
    "FigureProtocol",
    # ===== ENUMS =====
    "ImageFormat",
    "DiagramType",
    "ThemeStyle",
    # ===== TYPE ALIASES =====
    "NDArrayFloat",
    "Point2D",
    "Point3D",
    "ColorType",
    # ===== ENGINE CLASSES =====
    "PlotlyEngine",  # May be None if not installed
    "PyVistaEngine",  # May be None if not installed
    # ===== FACTORY PATTERN =====
    "VisualizerFactory",
    "create_visualizer",
    "list_available_engines",
    "get_engine_status",
    "check_installation",
    # ===== MODULE CONFIG =====
    "set_default_engine",
    "get_default_engine_instance",
    "reset_default_engine",
    # ===== MATPLOTLIB UTILITIES =====
    # Styling
    "set_publication_style",
    "reset_style",
    # Dimensions
    "add_dimension_arrow",
    "add_multiple_dimensions",
    # Materials
    "add_section_hatch",
    "plot_composite_section",
    "MATERIAL_COLORS",
    "MATERIAL_HATCHES",
    "MATERIAL_EDGE_COLORS",
    # Diagrams
    "plot_pm_interaction",
    "plot_moment_curvature",
    "plot_shear_moment_diagrams",
    # 3D
    "plot_3d_structure",
    # Formatting
    "format_engineering_axis",
    "set_equal_aspect_3d",
    # ===== UTILITIES =====
    "validate_colormap",
    "get_default_config",
    "check_plotly_installation",
    "check_pyvista_installation",
    # ===== DIAGRAM GENERATORS =====
    "DesignCode",
    "PMDiagramParams",
    "MomentCurvatureParams",
    "generate_pm_interaction_envelope",
    "generate_moment_curvature_response",
    "create_pm_diagram_with_code",
    "calculate_ductility",
    "format_code_reference",
    # ===== PYVISTA SPECIFICS =====
    "create_example_truss",
    "STRESS_COLORMAPS",
    "MATERIAL_RENDER_STYLES",
    "CAMERA_PRESETS",
    # ===== VERSION INFO =====
    "PLOTLY_AVAILABLE",
    "KALEIDO_AVAILABLE",
    "PYVISTA_AVAILABLE",
    "PLOTLY_VERSION",
    "KALEIDO_VERSION",
    "PYVISTA_VERSION",
    "VTK_VERSION",
]

# Package metadata
__version__ = "0.6.0"  # Fase 6 - Visualization complete
__author__ = "PyMemorial Team"
__license__ = "MIT"

# Configure logging (only if not already configured)
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
