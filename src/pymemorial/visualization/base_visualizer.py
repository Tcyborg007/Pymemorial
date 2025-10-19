# src/pymemorial/visualization/base_visualizer.py
"""
Base Abstract Classes for Visualization Engines - PyMemorial.

This module defines the complete interface for all visualization engines.
Every engine (Plotly, PyVista, Matplotlib) must implement these ABCs.

Features:
    - Immutable configuration dataclasses (frozen=True)
    - Type-safe enums for formats and diagram types
    - Complete ABC interface with 100% type coverage
    - Extensible design for future engines
    - Production-ready error handling
    - Support for static and interactive outputs

Examples:
    >>> from pymemorial.visualization import PlotConfig, ImageFormat
    >>> config = PlotConfig(width=1200, height=800, dpi=300)
    >>> export_cfg = ExportConfig(
    ...     filename=Path("output.png"),
    ...     format=ImageFormat.PNG,
    ...     scale=2.0
    ... )

Author: PyMemorial Team
License: MIT
Python: >=3.9
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, unique
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Protocol,
    Tuple,
    Union,
    TypedDict,
    runtime_checkable,
)

import numpy as np
import numpy.typing as npt

# Type aliases para clareza
NDArrayFloat = npt.NDArray[np.floating[Any]]
Point2D = Tuple[float, float]
Point3D = Tuple[float, float, float]
ColorType = Union[str, Tuple[int, int, int], Tuple[int, int, int, int]]


# ============================================================================
# ENUMS - Tipos seguros para formatos e diagramas
# ============================================================================


@unique
class ImageFormat(Enum):
    """
    Supported static and interactive image formats.

    Attributes:
        PNG: Portable Network Graphics (raster, lossless)
        SVG: Scalable Vector Graphics (vector, lossless)
        JPEG: Joint Photographic Experts Group (raster, lossy)
        PDF: Portable Document Format (vector, document)
        HTML: Interactive HTML with embedded JavaScript
        WEBP: Modern raster format with better compression
        EPS: Encapsulated PostScript (vector, publication)
    """

    PNG = "png"
    SVG = "svg"
    JPEG = "jpeg"
    PDF = "pdf"
    HTML = "html"
    WEBP = "webp"
    EPS = "eps"

    @property
    def is_vector(self) -> bool:
        """Check if format is vector-based."""
        return self in (ImageFormat.SVG, ImageFormat.PDF, ImageFormat.EPS)

    @property
    def is_raster(self) -> bool:
        """Check if format is raster-based."""
        return self in (ImageFormat.PNG, ImageFormat.JPEG, ImageFormat.WEBP)

    @property
    def is_interactive(self) -> bool:
        """Check if format supports interactivity."""
        return self == ImageFormat.HTML

    @property
    def requires_kaleido(self) -> bool:
        """Check if format requires Kaleido for Plotly export."""
        return self != ImageFormat.HTML


@unique
class DiagramType(Enum):
    """
    Types of structural engineering diagrams.

    Attributes:
        INTERACTION_PM: P-M interaction diagram (axial-flexure)
        MOMENT_CURVATURE: M-κ diagram (moment-curvature)
        SHEAR_MOMENT: Shear and moment diagrams (beam analysis)
        STRUCTURE_3D: 3D structural frame visualization
        SECTION_2D: 2D cross-section view
        STRESS_CONTOUR: Stress field contour plot
        STRAIN_CONTOUR: Strain field contour plot
        DISPLACEMENT_3D: 3D displacement field visualization
        MODE_SHAPE: Modal analysis shape visualization
        LOAD_PATH: Load path visualization
        INFLUENCE_LINE: Influence line diagram
        BUCKLING_MODE: Buckling mode shape
    """

    INTERACTION_PM = "pm_interaction"
    MOMENT_CURVATURE = "moment_curvature"
    SHEAR_MOMENT = "shear_moment"
    STRUCTURE_3D = "structure_3d"
    SECTION_2D = "section_2d"
    STRESS_CONTOUR = "stress_contour"
    STRAIN_CONTOUR = "strain_contour"
    DISPLACEMENT_3D = "displacement_3d"
    MODE_SHAPE = "mode_shape"
    LOAD_PATH = "load_path"
    INFLUENCE_LINE = "influence_line"
    BUCKLING_MODE = "buckling_mode"


@unique
class ThemeStyle(Enum):
    """
    Visualization theme styles.

    Attributes:
        DEFAULT: Clean white background with black text
        DARK: Dark background for presentations
        PLOTLY: Plotly's default theme
        SEABORN: Seaborn statistical theme
        PUBLICATION: Publication-quality styling (Nature/IEEE)
        PRESENTATION: High-contrast for presentations
        ENGINEERING: Engineering drawing style (ISO/ABNT)
    """

    DEFAULT = "default"
    DARK = "dark"
    PLOTLY = "plotly"
    SEABORN = "seaborn"
    PUBLICATION = "publication"
    PRESENTATION = "presentation"
    ENGINEERING = "engineering"


# ============================================================================
# DATACLASSES - Configurações imutáveis (frozen=True para type-safety)
# ============================================================================


@dataclass(frozen=True)
class PlotConfig:
    """
    Immutable configuration for plot styling and behavior.

    All fields are immutable (frozen=True) for thread-safety and
    hashability. Use `dataclasses.replace()` to create modified copies.

    Attributes:
        title: Plot title (supports LaTeX: r"$\\sigma_{max}$")
        xlabel: X-axis label
        ylabel: Y-axis label
        zlabel: Z-axis label (for 3D plots)
        width: Figure width in pixels
        height: Figure height in pixels
        dpi: Dots per inch for raster export
        grid: Show grid lines
        legend: Show legend
        interactive: Enable interactive features (zoom, pan, hover)
        theme: Visual theme style
        font_family: Font family (e.g., "Arial", "Times New Roman")
        font_size: Base font size in points
        line_width: Default line width
        marker_size: Default marker size
        colormap: Colormap name (e.g., "viridis", "jet", "coolwarm")
        background_color: Background color
        show_axes: Show axes
        equal_aspect: Force equal aspect ratio (important for sections)
        tight_layout: Use tight layout (removes white space)

    Examples:
        >>> config = PlotConfig(title="P-M Diagram", width=1200, dpi=300)
        >>> modified = dataclasses.replace(config, height=900)
    """

    # Text labels
    title: str = ""
    xlabel: str = ""
    ylabel: str = ""
    zlabel: str = ""

    # Size and resolution
    width: int = 800
    height: int = 600
    dpi: int = 150

    # Display options
    grid: bool = True
    legend: bool = True
    interactive: bool = False
    theme: ThemeStyle = ThemeStyle.DEFAULT

    # Typography
    font_family: str = "Arial, sans-serif"
    font_size: int = 11
    title_size: int = 14
    label_size: int = 12

    # Styling
    line_width: float = 2.0
    marker_size: float = 8.0
    colormap: str = "viridis"
    background_color: ColorType = "white"

    # Layout
    show_axes: bool = True
    equal_aspect: bool = False
    tight_layout: bool = True

    # Advanced
    show_toolbar: bool = True
    show_watermark: bool = False
    watermark_text: str = "PyMemorial"

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Width and height must be positive")
        if self.dpi < 50 or self.dpi > 600:
            raise ValueError("DPI must be between 50 and 600")
        if self.font_size < 6 or self.font_size > 72:
            raise ValueError("Font size must be between 6 and 72 points")


@dataclass(frozen=True)
class ExportConfig:
    """
    Immutable configuration for image export.

    Attributes:
        filename: Output file path (extension auto-corrected if needed)
        format: Output image format
        scale: Scaling factor for resolution (2.0 = double resolution)
        transparent_bg: Use transparent background (PNG, SVG only)
        quality: JPEG quality 1-100 (higher = better)
        optimize: Optimize file size (slower but smaller files)
        embed_fonts: Embed fonts in PDF/SVG (for portability)
        metadata: Custom metadata dict (author, title, subject, etc.)

    Examples:
        >>> export = ExportConfig(
        ...     filename=Path("diagram.png"),
        ...     format=ImageFormat.PNG,
        ...     scale=2.0,
        ...     transparent_bg=True
        ... )
    """

    filename: Path
    format: ImageFormat = ImageFormat.PNG
    scale: float = 1.0
    transparent_bg: bool = False
    quality: int = 95  # For JPEG/WEBP
    optimize: bool = True
    embed_fonts: bool = True
    metadata: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate export configuration."""
        if self.scale <= 0 or self.scale > 10:
            raise ValueError("Scale must be between 0 and 10")
        if not 1 <= self.quality <= 100:
            raise ValueError("Quality must be between 1 and 100")
        if self.transparent_bg and self.format not in (
            ImageFormat.PNG,
            ImageFormat.SVG,
            ImageFormat.WEBP,
        ):
            raise ValueError(
                f"Transparent background not supported for {self.format.value}"
            )

    @property
    def corrected_filename(self) -> Path:
        """Return filename with correct extension for format."""
        return self.filename.with_suffix(f".{self.format.value}")


@dataclass(frozen=True)
class AnnotationStyle:
    """
    Styling for annotations and dimensions.

    Attributes:
        color: Annotation color
        font_size: Font size for text
        arrow_width: Arrow line width
        arrow_style: Arrow style ('<|-|>', '<->', '->', etc.)
        background: Background color for text box
        border_color: Border color for text box
        border_width: Border width
        opacity: Opacity 0-1
    """

    color: ColorType = "red"
    font_size: int = 10
    arrow_width: float = 2.0
    arrow_style: str = "<|-|>"
    background: ColorType = "white"
    border_color: Optional[ColorType] = None
    border_width: float = 1.0
    opacity: float = 0.9


# ============================================================================
# PROTOCOLS - Type-safe interfaces (PEP 544)
# ============================================================================


@runtime_checkable
class FigureProtocol(Protocol):
    """
    Protocol for figure objects from different libraries.

    Allows type-safe duck typing for matplotlib.Figure, plotly.Figure, etc.
    """

    def savefig(self, filename: Union[str, Path], **kwargs: Any) -> None:
        """Save figure to file."""
        ...

    def show(self) -> None:
        """Display figure."""
        ...


# ============================================================================
# ABSTRACT BASE CLASS - Interface principal para engines
# ============================================================================


class VisualizerEngine(ABC):
    """
    Abstract base class for all visualization engines.

    Every visualization engine (Plotly, PyVista, Matplotlib, etc.) must
    inherit from this class and implement all abstract methods.

    This ensures a consistent API across all engines and allows for
    easy swapping of visualization backends.

    Attributes:
        name: Unique identifier for this engine
        version: Version of underlying library
        supports_3d: Whether engine supports 3D visualization
        supports_interactive: Whether engine supports interactivity
        available: Whether engine dependencies are installed

    Examples:
        >>> engine = PlotlyEngine()
        >>> if engine.available:
        ...     fig = engine.create_pm_diagram(p_vals, m_vals)
        ...     engine.export_static(fig, ExportConfig(Path("pm.png")))
    """

    def __init__(self, name: str) -> None:
        """
        Initialize visualization engine.

        Args:
            name: Unique identifier (e.g., "plotly", "matplotlib", "pyvista")
        """
        self.name = name
        self._config: Optional[PlotConfig] = None
        self._cache: Dict[str, Any] = {}  # For caching computed data

    # ------------------------------------------------------------------------
    # PROPERTIES (must be implemented by subclasses)
    # ------------------------------------------------------------------------

    @property
    @abstractmethod
    def version(self) -> str:
        """Return version of underlying visualization library."""
        pass

    @property
    @abstractmethod
    def supports_3d(self) -> bool:
        """Return whether this engine supports 3D visualization."""
        pass

    @property
    @abstractmethod
    def supports_interactive(self) -> bool:
        """Return whether this engine supports interactive plots."""
        pass

    @property
    @abstractmethod
    def available(self) -> bool:
        """
        Check if engine is available (dependencies installed).

        Returns:
            True if all required dependencies are installed and functional.

        Examples:
            >>> engine = PlotlyEngine()
            >>> if not engine.available:
            ...     print("Install with: pip install pymemorial[viz]")
        """
        pass

    @property
    def supported_formats(self) -> List[ImageFormat]:
        """
        Return list of supported export formats for this engine.

        Override in subclass to specify engine capabilities.

        Returns:
            List of supported ImageFormat enums.
        """
        return [ImageFormat.PNG, ImageFormat.SVG, ImageFormat.PDF, ImageFormat.HTML]

    # ------------------------------------------------------------------------
    # ABSTRACT METHODS - Diagram creation (must implement ALL)
    # ------------------------------------------------------------------------

    @abstractmethod
    def create_pm_diagram(
        self,
        p_values: NDArrayFloat,
        m_values: NDArrayFloat,
        design_point: Optional[Point2D] = None,
        capacity_point: Optional[Point2D] = None,
        config: Optional[PlotConfig] = None,
    ) -> Any:
        """
        Create P-M interaction diagram for column/beam-column design.

        Args:
            p_values: Axial force values (N), typically normalized P/P_n
            m_values: Bending moment values (N⋅m), typically normalized M/M_n
            design_point: Optional (P_design, M_design) point to highlight
            capacity_point: Optional (P_capacity, M_capacity) maximum capacity
            config: Plot configuration (uses default if None)

        Returns:
            Figure object (engine-specific type: plotly.Figure, matplotlib.Figure, etc.)

        Raises:
            RuntimeError: If engine not available
            ValueError: If input arrays have mismatched shapes

        Examples:
            >>> p = np.array([0, 0.5, 1.0, 0.8, 0.3, 0])
            >>> m = np.array([0.6, 0.8, 0.5, 0.4, 0.7, 0])
            >>> fig = engine.create_pm_diagram(p, m, design_point=(0.6, 0.4))
        """
        pass

    @abstractmethod
    def create_moment_curvature(
        self,
        curvature: NDArrayFloat,
        moment: NDArrayFloat,
        yield_point: Optional[Point2D] = None,
        ultimate_point: Optional[Point2D] = None,
        config: Optional[PlotConfig] = None,
    ) -> Any:
        """
        Create moment-curvature (M-κ) diagram for section analysis.

        Args:
            curvature: Curvature values (1/m)
            moment: Moment values (N⋅m)
            yield_point: Optional (κ_y, M_y) first yield point
            ultimate_point: Optional (κ_u, M_u) ultimate capacity
            config: Plot configuration

        Returns:
            Figure object (engine-specific)

        Examples:
            >>> kappa = np.linspace(0, 0.01, 100)
            >>> m = 1e6 * kappa * (1 - 0.5 * kappa / 0.01)  # Bilinear
            >>> fig = engine.create_moment_curvature(kappa, m)
        """
        pass

    @abstractmethod
    def create_3d_structure(
        self,
        nodes: NDArrayFloat,
        elements: List[Tuple[int, int]],
        displacements: Optional[NDArrayFloat] = None,
        loads: Optional[NDArrayFloat] = None,
        config: Optional[PlotConfig] = None,
    ) -> Any:
        """
        Create 3D visualization of structural frame.

        Args:
            nodes: Node coordinates [N x 3] in meters
            elements: Element connectivity list [(node_i, node_j), ...]
            displacements: Optional nodal displacements [N x 3] in meters
            loads: Optional nodal loads [N x 6] (Fx, Fy, Fz, Mx, My, Mz)
            config: Plot configuration

        Returns:
            Figure object (engine-specific)

        Examples:
            >>> nodes = np.array([[0,0,0], [5,0,0], [5,0,3], [0,0,3]])
            >>> elements = [(0,1), (1,2), (2,3), (3,0)]
            >>> fig = engine.create_3d_structure(nodes, elements)
        """
        pass

    @abstractmethod
    def create_section_2d(
        self,
        vertices: NDArrayFloat,
        facets: Optional[List[List[int]]] = None,
        materials: Optional[List[str]] = None,
        show_centroid: bool = True,
        show_dimensions: bool = True,
        config: Optional[PlotConfig] = None,
    ) -> Any:
        """
        Create 2D cross-section visualization with materials.

        Args:
            vertices: Vertex coordinates [N x 2] in meters
            facets: Optional facet connectivity [[v1, v2, v3, ...], ...]
            materials: Material names for each facet ["steel", "concrete", ...]
            show_centroid: Show centroid marker
            show_dimensions: Show dimension annotations
            config: Plot configuration

        Returns:
            Figure object (engine-specific)
        """
        pass

    @abstractmethod
    def create_stress_contour(
        self,
        x: NDArrayFloat,
        y: NDArrayFloat,
        stress: NDArrayFloat,
        stress_type: Literal["von_mises", "sigma_x", "sigma_y", "tau_xy"] = "von_mises",
        config: Optional[PlotConfig] = None,
    ) -> Any:
        """
        Create stress contour plot for section analysis.

        Args:
            x: X coordinates [N x M] meshgrid
            y: Y coordinates [N x M] meshgrid
            stress: Stress values [N x M] in Pa
            stress_type: Type of stress to plot
            config: Plot configuration

        Returns:
            Figure object (engine-specific)
        """
        pass

    # ------------------------------------------------------------------------
    # EXPORT AND DISPLAY METHODS
    # ------------------------------------------------------------------------

    @abstractmethod
    def export_static(
        self,
        figure: Any,
        export_config: ExportConfig,
    ) -> Path:
        """
        Export figure to static image file.

        Args:
            figure: Figure object to export (engine-specific type)
            export_config: Export configuration

        Returns:
            Path to saved file (with corrected extension)

        Raises:
            RuntimeError: If export fails or format not supported
            ValueError: If figure is invalid

        Examples:
            >>> fig = engine.create_pm_diagram(p, m)
            >>> path = engine.export_static(
            ...     fig,
            ...     ExportConfig(Path("output.png"), format=ImageFormat.PNG)
            ... )
        """
        pass

    @abstractmethod
    def show(self, figure: Any, interactive: bool = False) -> None:
        """
        Display figure in appropriate viewer.

        Args:
            figure: Figure object to display
            interactive: Show interactive version if supported

        Examples:
            >>> fig = engine.create_pm_diagram(p, m)
            >>> engine.show(fig, interactive=True)
        """
        pass

    # ------------------------------------------------------------------------
    # CONFIGURATION METHODS
    # ------------------------------------------------------------------------

    def set_config(self, config: PlotConfig) -> None:
        """
        Set default plot configuration for this engine.

        Args:
            config: Default configuration to use

        Examples:
            >>> config = PlotConfig(width=1200, dpi=300, theme=ThemeStyle.PUBLICATION)
            >>> engine.set_config(config)
        """
        self._config = config

    def get_config(self) -> Optional[PlotConfig]:
        """
        Get current default plot configuration.

        Returns:
            Current default PlotConfig or None if not set
        """
        return self._config

    def reset_config(self) -> None:
        """Reset to default configuration."""
        self._config = None

    # ------------------------------------------------------------------------
    # UTILITY METHODS (concrete implementations)
    # ------------------------------------------------------------------------

    def validate_arrays(
        self,
        *arrays: NDArrayFloat,
        min_length: int = 2,
        same_length: bool = True,
    ) -> None:
        """
        Validate input arrays for plotting.

        Args:
            *arrays: Variable number of numpy arrays to validate
            min_length: Minimum required array length
            same_length: Whether all arrays must have same length

        Raises:
            ValueError: If validation fails
        """
        if not arrays:
            raise ValueError("At least one array must be provided")

        lengths = [len(arr) for arr in arrays]

        if min(lengths) < min_length:
            raise ValueError(f"Arrays must have at least {min_length} elements")

        if same_length and len(set(lengths)) > 1:
            raise ValueError(
                f"All arrays must have same length. Got lengths: {lengths}"
            )

    def clear_cache(self) -> None:
        """Clear internal cache."""
        self._cache.clear()

    def __repr__(self) -> str:
        """Return string representation."""
        status = "available" if self.available else "unavailable"
        return (
            f"{self.__class__.__name__}("
            f"name='{self.name}', "
            f"version='{self.version}', "
            f"status='{status}')"
        )

    def __str__(self) -> str:
        """Return human-readable string."""
        return f"{self.name} v{self.version} ({self.available and 'ready' or 'not installed'})"

    @abstractmethod
    def export(
        self,
        fig: Any,
        filename: str | Path,
        **kwargs
    ) -> Path:
        """
        Export figure to file.
        
        Args:
            fig: Figure object
            filename: Output filename
            **kwargs: Additional export parameters
        
        Returns:
            Path to exported file
        """
        pass




# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def validate_colormap(colormap: str, engine: str = "matplotlib") -> bool:
    """
    Validate if colormap name exists for given engine.

    Args:
        colormap: Colormap name to validate
        engine: Engine name ("matplotlib", "plotly", etc.)

    Returns:
        True if valid, False otherwise
    """
    if engine == "matplotlib":
        try:
            import matplotlib.pyplot as plt

            return colormap in plt.colormaps()
        except ImportError:
            return False
    elif engine == "plotly":
        # Plotly colormaps
        plotly_cmaps = [
            "viridis",
            "plasma",
            "inferno",
            "magma",
            "cividis",
            "turbo",
            "jet",
            "hot",
            "cool",
            "coolwarm",
            "RdBu",
            "YlOrRd",
        ]
        return colormap in plotly_cmaps
    return False


def get_default_config(diagram_type: DiagramType) -> PlotConfig:
    """
    Get sensible default configuration for diagram type.

    Args:
        diagram_type: Type of diagram to configure

    Returns:
        PlotConfig with appropriate defaults

    Examples:
        >>> config = get_default_config(DiagramType.INTERACTION_PM)
        >>> config.xlabel
        'M/M_n (Normalized Moment)'
    """
    defaults = {
        DiagramType.INTERACTION_PM: PlotConfig(
            title="P-M Interaction Diagram",
            xlabel="M/M_n (Normalized Moment)",
            ylabel="P/P_n (Normalized Axial Load)",
            equal_aspect=False,
        ),
        DiagramType.MOMENT_CURVATURE: PlotConfig(
            title="Moment-Curvature Diagram",
            xlabel="Curvature κ (1/km)",
            ylabel="Moment M (MN⋅m)",
        ),
        DiagramType.SECTION_2D: PlotConfig(
            title="Cross-Section",
            xlabel="Width (m)",
            ylabel="Height (m)",
            equal_aspect=True,
            grid=False,
        ),
        DiagramType.STRUCTURE_3D: PlotConfig(
            title="3D Structural Model",
            xlabel="X (m)",
            ylabel="Y (m)",
            zlabel="Z (m)",
            equal_aspect=True,
        ),
    }
    return defaults.get(diagram_type, PlotConfig())


# ============================================================================
# TYPE HINTS FOR EXTERNAL USE
# ============================================================================

__all__ = [
    # Enums
    "ImageFormat",
    "DiagramType",
    "ThemeStyle",
    # Dataclasses
    "PlotConfig",
    "ExportConfig",
    "AnnotationStyle",
    # ABC
    "VisualizerEngine",
    # Protocols
    "FigureProtocol",
    # Type aliases
    "NDArrayFloat",
    "Point2D",
    "Point3D",
    "ColorType",
    # Utilities
    "validate_colormap",
    "get_default_config",
]
