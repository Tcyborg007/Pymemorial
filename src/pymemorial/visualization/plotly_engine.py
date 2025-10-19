# src/pymemorial/visualization/plotly_engine.py
"""
Plotly-based Visualization Engine for PyMemorial.

This module implements a production-ready visualization engine using Plotly,
supporting both static (PNG, SVG, PDF) and interactive (HTML) outputs.
Uses Kaleido for high-quality static image export without browser dependencies.

Features:
    - Static image export via Kaleido (headless, CI/CD compatible)
    - Interactive HTML with full Plotly.js capabilities
    - 3D visualization with camera controls
    - Publication-quality styling presets
    - Graceful degradation if dependencies missing
    - Type-safe interface following VisualizerEngine ABC
    - WebGL acceleration for large datasets (>1000 points)

Requirements:
    - plotly >= 5.24.0 (latest: 6.3.0, Oct 2025)
    - kaleido >= 0.2.1 (optional, for static export)
    - numpy >= 1.24.0

Author: PyMemorial Team
License: MIT
Python: >=3.9
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import numpy.typing as npt

# Import base classes
from .base_visualizer import (
    AnnotationStyle,
    ColorType,
    DiagramType,
    ExportConfig,
    ImageFormat,
    NDArrayFloat,
    PlotConfig,
    Point2D,
    Point3D,
    ThemeStyle,
    VisualizerEngine,
)

# ============================================================================
# OPTIONAL IMPORTS - Graceful degradation
# ============================================================================

# Setup logging
logger = logging.getLogger(__name__)

# Try importing Plotly
PLOTLY_AVAILABLE = False
PLOTLY_VERSION = "0.0.0"
try:
    import plotly.graph_objects as go
    import plotly.io as pio
    from plotly.subplots import make_subplots
    import plotly

    PLOTLY_AVAILABLE = True
    PLOTLY_VERSION = plotly.__version__
    logger.info(f"Plotly {PLOTLY_VERSION} loaded successfully")
except ImportError as e:
    logger.warning(f"Plotly not available: {e}")
    logger.warning("Install with: pip install 'pymemorial[viz]'")

# Try importing Kaleido for static export
KALEIDO_AVAILABLE = False
KALEIDO_VERSION = "0.0.0"
try:
    import kaleido

    KALEIDO_AVAILABLE = True
    KALEIDO_VERSION = getattr(kaleido, "__version__", "unknown")
    logger.info(f"Kaleido {KALEIDO_VERSION} loaded successfully")
except ImportError:
    logger.warning("Kaleido not available (static image export disabled)")
    logger.warning("Install with: pip install kaleido")


# ============================================================================
# CONSTANTS - Material colors and styling
# ============================================================================

# Material colors matching matplotlib_utils (consistency)
MATERIAL_COLORS_PLOTLY: Dict[str, str] = {
    "steel": "#B0C4DE",  # Light steel blue
    "concrete": "#D3D3D3",  # Light gray
    "wood": "#DEB887",  # Burlywood
    "aluminum": "#E8E8E8",  # Very light gray
    "masonry": "#CD853F",  # Peru
    "composite": "#98D8C8",  # Light teal
    "rebar": "#2F4F4F",  # Dark slate gray
}

# Plotly template mappings
THEME_TO_PLOTLY_TEMPLATE: Dict[ThemeStyle, str] = {
    ThemeStyle.DEFAULT: "plotly_white",
    ThemeStyle.DARK: "plotly_dark",
    ThemeStyle.PLOTLY: "plotly",
    ThemeStyle.SEABORN: "seaborn",
    ThemeStyle.PUBLICATION: "simple_white",  # Clean for publications
    ThemeStyle.PRESENTATION: "presentation",  # High contrast
    ThemeStyle.ENGINEERING: "plotly_white",  # Grid-heavy
}

# Colorscales for contour plots (Plotly-compatible)
COLORSCALES: Dict[str, str] = {
    "viridis": "Viridis",
    "plasma": "Plasma",
    "inferno": "Inferno",
    "magma": "Magma",
    "cividis": "Cividis",
    "turbo": "Turbo",
    "jet": "Jet",
    "hot": "Hot",
    "cool": "Blues",
    "coolwarm": "RdBu",
    "stress": "RdYlGn_r",  # Reversed (red = high stress)
}


# ============================================================================
# PLOTLY ENGINE CLASS
# ============================================================================


class PlotlyEngine(VisualizerEngine):
    """
    Plotly-based visualization engine for PyMemorial.

    This engine provides both static (PNG, SVG, PDF) and interactive (HTML)
    visualization capabilities using Plotly.js. Supports 2D/3D plots with
    publication-quality styling.

    Features:
        - Static export via Kaleido (no browser required)
        - Interactive HTML with zoom, pan, hover tooltips
        - 3D visualization with camera controls
        - WebGL rendering for large datasets (performance)
        - Automatic fallback to matplotlib if unavailable

    Attributes:
        name: Engine identifier ("plotly")
        version: Plotly library version
        supports_3d: True (full 3D support)
        supports_interactive: True (interactive HTML)
        available: Whether Plotly is installed

    Examples:
        >>> engine = PlotlyEngine()
        >>> if engine.available:
        ...     fig = engine.create_pm_diagram(p_vals, m_vals)
        ...     engine.export_static(fig, ExportConfig(Path("diagram.png")))
        ... else:
        ...     print("Install: pip install pymemorial[viz]")

    References:
        - Plotly Python: https://plotly.com/python/
        - Graph Objects API: https://plotly.com/python/graph-objects/
        - Kaleido: https://github.com/plotly/Kaleido
    """

    def __init__(self) -> None:
        """Initialize Plotly visualization engine."""
        super().__init__(name="plotly")

        # Check availability
        if not PLOTLY_AVAILABLE:
            logger.error("Plotly not available - engine disabled")
            return


        # Set default template
        pio.templates.default = "plotly_white"

    # ------------------------------------------------------------------------
    # PROPERTIES (implementing ABC)
    # ------------------------------------------------------------------------

    @property
    def version(self) -> str:
        """Return Plotly version."""
        return PLOTLY_VERSION

    @property
    def supports_3d(self) -> bool:
        """Plotly supports full 3D visualization."""
        return True

    @property
    def supports_interactive(self) -> bool:
        """Plotly supports interactive HTML output."""
        return True

    @property
    def available(self) -> bool:
        """Check if Plotly is installed and functional."""
        return PLOTLY_AVAILABLE

    @property
    def supported_formats(self) -> List[ImageFormat]:
        """Return supported export formats."""
        if not PLOTLY_AVAILABLE:
            return []

        formats = [ImageFormat.HTML]  # Always available

        if KALEIDO_AVAILABLE:
            formats.extend(
                [
                    ImageFormat.PNG,
                    ImageFormat.SVG,
                    ImageFormat.PDF,
                    ImageFormat.JPEG,
                    ImageFormat.WEBP,
                ]
            )

        return formats

    # ------------------------------------------------------------------------
    # PRIVATE HELPER METHODS
    # ------------------------------------------------------------------------

    def _check_available(self) -> None:
        """Raise error if engine not available."""
        if not self.available:
            raise RuntimeError(
                "PlotlyEngine not available. Install with: "
                "pip install 'pymemorial[viz]'"
            )

    def _apply_config(
        self, fig: go.Figure, config: Optional[PlotConfig] = None
    ) -> None:
        """
        Apply PlotConfig to Plotly figure.

        Args:
            fig: Plotly figure to modify
            config: Configuration to apply (uses default if None)
        """
        cfg = config or self._config or PlotConfig()

        # Update layout
        layout_updates: Dict[str, Any] = {
            "title": {
                "text": cfg.title,
                "font": {"size": cfg.title_size, "family": cfg.font_family},
            },
            "xaxis": {
                "title": cfg.xlabel,
                "showgrid": cfg.grid,
                "gridcolor": "lightgray",
                "gridwidth": 0.5,
                "showline": cfg.show_axes,
                "linecolor": "black",
                "linewidth": 1,
            },
            "yaxis": {
                "title": cfg.ylabel,
                "showgrid": cfg.grid,
                "gridcolor": "lightgray",
                "gridwidth": 0.5,
                "showline": cfg.show_axes,
                "linecolor": "black",
                "linewidth": 1,
            },
            "showlegend": cfg.legend,
            "font": {"size": cfg.font_size, "family": cfg.font_family},
            "plot_bgcolor": cfg.background_color,
            "paper_bgcolor": cfg.background_color,
            "width": cfg.width,
            "height": cfg.height,
        }

        # Equal aspect ratio for sections
        if cfg.equal_aspect:
            layout_updates["yaxis"]["scaleanchor"] = "x"
            layout_updates["yaxis"]["scaleratio"] = 1

        # Apply theme template
        template = THEME_TO_PLOTLY_TEMPLATE.get(cfg.theme, "plotly_white")
        fig.update_layout(template=template, **layout_updates)

        # Watermark if requested
        if cfg.show_watermark:
            fig.add_annotation(
                text=cfg.watermark_text,
                xref="paper",
                yref="paper",
                x=0.98,
                y=0.02,
                showarrow=False,
                font=dict(size=10, color="lightgray"),
                opacity=0.5,
            )

    def _get_colorscale(self, colormap: str) -> str:
        """
        Get Plotly-compatible colorscale name.

        Args:
            colormap: Colormap name (matplotlib or plotly)

        Returns:
            Plotly colorscale name
        """
        return COLORSCALES.get(colormap.lower(), "Viridis")

    # ------------------------------------------------------------------------
    # DIAGRAM CREATION METHODS (implementing ABC)
    # ------------------------------------------------------------------------

    def create_pm_diagram(
        self,
        p_values: NDArrayFloat,
        m_values: NDArrayFloat,
        design_point: Optional[Point2D] = None,
        capacity_point: Optional[Point2D] = None,
        config: Optional[PlotConfig] = None,
    ) -> go.Figure:
        """
        Create P-M interaction diagram using Plotly.

        Args:
            p_values: Axial force values (N), shape (N,)
            m_values: Bending moment values (N⋅m), shape (N,)
            design_point: Optional (M_design, P_design) point
            capacity_point: Optional (M_capacity, P_capacity) point
            config: Plot configuration

        Returns:
            Plotly Figure object

        Raises:
            RuntimeError: If Plotly not available
            ValueError: If arrays have different lengths

        Examples:
            >>> engine = PlotlyEngine()
            >>> p = np.array([0, 0.5, 1.0, 0.8, 0.3, 0])
            >>> m = np.array([0.6, 0.8, 0.5, 0.4, 0.7, 0])
            >>> fig = engine.create_pm_diagram(p, m)
            >>> engine.show(fig, interactive=True)
        """
        self._check_available()
        self.validate_arrays(p_values, m_values, same_length=True)

        # Create figure
        fig = go.Figure()

        # Capacity envelope curve
        fig.add_trace(
            go.Scatter(
                x=m_values,
                y=p_values,
                mode="lines",
                name="Capacity Envelope",
                line=dict(color="blue", width=3),
                fill="toself",
                fillcolor="rgba(0, 0, 255, 0.1)",
                hovertemplate="M/M_n: %{x:.3f}<br>P/P_n: %{y:.3f}<extra></extra>",
            )
        )

        # Design point
        if design_point is not None:
            m_d, p_d = design_point

            # Check if inside envelope (simple polygon test)
            from matplotlib.path import Path as MplPath

            path = MplPath(np.column_stack([m_values, p_values]))
            is_safe = path.contains_point([m_d, p_d])

            color = "green" if is_safe else "red"
            status = "✓ SAFE" if is_safe else "✗ UNSAFE"

            fig.add_trace(
                go.Scatter(
                    x=[m_d],
                    y=[p_d],
                    mode="markers+text",
                    name="Design Point",
                    marker=dict(size=12, color=color, symbol="circle"),
                    text=[status],
                    textposition="top center",
                    hovertemplate=f"Design Point<br>M: {m_d:.3f}<br>P: {p_d:.3f}<br>{status}<extra></extra>",
                )
            )

        # Capacity point
        if capacity_point is not None:
            m_c, p_c = capacity_point
            fig.add_trace(
                go.Scatter(
                    x=[m_c],
                    y=[p_c],
                    mode="markers",
                    name="Max Capacity",
                    marker=dict(size=10, color="orange", symbol="star"),
                    hovertemplate=f"Max Capacity<br>M: {m_c:.3f}<br>P: {p_c:.3f}<extra></extra>",
                )
            )

        # Apply configuration
        cfg = config or PlotConfig(
            title="P-M Interaction Diagram",
            xlabel="M/M_n (Normalized Moment)",
            ylabel="P/P_n (Normalized Axial Load)",
        )
        self._apply_config(fig, cfg)

        # Add zero lines
        fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)
        fig.add_vline(x=0, line_dash="dash", line_color="gray", line_width=1)

        return fig

    def create_moment_curvature(
        self,
        curvature: NDArrayFloat,
        moment: NDArrayFloat,
        yield_point: Optional[Point2D] = None,
        ultimate_point: Optional[Point2D] = None,
        config: Optional[PlotConfig] = None,
    ) -> go.Figure:
        """
        Create moment-curvature (M-κ) diagram using Plotly.

        Args:
            curvature: Curvature values κ (1/m), shape (N,)
            moment: Moment values M (N⋅m), shape (N,)
            yield_point: Optional (κ_y, M_y) first yield
            ultimate_point: Optional (κ_u, M_u) ultimate capacity
            config: Plot configuration

        Returns:
            Plotly Figure object

        Examples:
            >>> kappa = np.linspace(0, 0.01, 100)
            >>> m = 1e6 * kappa * (1 - 0.5 * kappa / 0.01)
            >>> fig = engine.create_moment_curvature(kappa, m)
        """
        self._check_available()
        self.validate_arrays(curvature, moment, same_length=True)

        fig = go.Figure()

        # Main M-κ curve
        fig.add_trace(
            go.Scatter(
                x=curvature,
                y=moment,
                mode="lines",
                name="M-κ Response",
                line=dict(color="blue", width=3),
                hovertemplate="κ: %{x:.6f} (1/m)<br>M: %{y:.3e} (N⋅m)<extra></extra>",
            )
        )

        # Yield point
        if yield_point is not None:
            k_y, m_y = yield_point
            fig.add_trace(
                go.Scatter(
                    x=[k_y],
                    y=[m_y],
                    mode="markers",
                    name="Yield Point",
                    marker=dict(size=12, color="green", symbol="circle"),
                    hovertemplate=f"Yield<br>κ_y: {k_y:.6f}<br>M_y: {m_y:.3e}<extra></extra>",
                )
            )
            # Dashed lines to axes
            fig.add_vline(x=k_y, line_dash="dash", line_color="green", line_width=1.5)
            fig.add_hline(y=m_y, line_dash="dash", line_color="green", line_width=1.5)

        # Ultimate point
        if ultimate_point is not None:
            k_u, m_u = ultimate_point
            fig.add_trace(
                go.Scatter(
                    x=[k_u],
                    y=[m_u],
                    mode="markers",
                    name="Ultimate Point",
                    marker=dict(size=12, color="red", symbol="circle"),
                    hovertemplate=f"Ultimate<br>κ_u: {k_u:.6f}<br>M_u: {m_u:.3e}<extra></extra>",
                )
            )
            fig.add_vline(x=k_u, line_dash="dash", line_color="red", line_width=1.5)
            fig.add_hline(y=m_u, line_dash="dash", line_color="red", line_width=1.5)

            # Ductility annotation
            if yield_point is not None:
                k_y = yield_point[0]
                mu = k_u / k_y if k_y > 0 else 0
                fig.add_annotation(
                    text=f"Ductility μ = κ_u / κ_y = {mu:.2f}",
                    xref="paper",
                    yref="paper",
                    x=0.05,
                    y=0.95,
                    showarrow=False,
                    bgcolor="wheat",
                    bordercolor="black",
                    borderwidth=1,
                    font=dict(size=12),
                )

        # Apply configuration
        cfg = config or PlotConfig(
            title="Moment-Curvature Diagram",
            xlabel="Curvature κ (1/km)",
            ylabel="Moment M (MN⋅m)",
        )
        self._apply_config(fig, cfg)

        return fig

    def create_3d_structure(
        self,
        nodes: NDArrayFloat,
        elements: List[Tuple[int, int]],
        displacements: Optional[NDArrayFloat] = None,
        loads: Optional[NDArrayFloat] = None,
        config: Optional[PlotConfig] = None,
    ) -> go.Figure:
        """
        Create 3D structural frame visualization using Plotly.

        Args:
            nodes: Node coordinates [N x 3] in meters
            elements: Element connectivity [(node_i, node_j), ...]
            displacements: Optional nodal displacements [N x 3] in meters
            loads: Optional nodal loads [N x 6] (Fx, Fy, Fz, Mx, My, Mz) in kN
            config: Plot configuration

        Returns:
            Plotly Figure object with 3D scene

        Examples:
            >>> nodes = np.array([[0,0,0], [5,0,0], [5,0,3], [0,0,3]])
            >>> elements = [(0,1), (1,2), (2,3), (3,0)]
            >>> fig = engine.create_3d_structure(nodes, elements)
        """
        self._check_available()

        if nodes.shape[1] != 3:
            raise ValueError(f"Nodes must be N x 3 array, got shape {nodes.shape}")

        fig = go.Figure()

        # Plot undeformed elements
        for i, j in elements:
            fig.add_trace(
                go.Scatter3d(
                    x=[nodes[i, 0], nodes[j, 0]],
                    y=[nodes[i, 1], nodes[j, 1]],
                    z=[nodes[i, 2], nodes[j, 2]],
                    mode="lines",
                    name="Undeformed",
                    line=dict(color="blue", width=4),
                    showlegend=(i == 0 and j == 1),  # Show legend once
                    hovertemplate=f"Element {i}-{j}<extra></extra>",
                )
            )

        # Plot nodes
        fig.add_trace(
            go.Scatter3d(
                x=nodes[:, 0],
                y=nodes[:, 1],
                z=nodes[:, 2],
                mode="markers+text",
                name="Nodes",
                marker=dict(size=6, color="blue"),
                text=[str(i) for i in range(len(nodes))],
                textposition="top center",
                hovertemplate="Node %{text}<br>X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}<extra></extra>",
            )
        )

        # Plot deformed structure
        if displacements is not None:
            deformed = nodes + displacements
            for i, j in elements:
                fig.add_trace(
                    go.Scatter3d(
                        x=[deformed[i, 0], deformed[j, 0]],
                        y=[deformed[i, 1], deformed[j, 1]],
                        z=[deformed[i, 2], deformed[j, 2]],
                        mode="lines",
                        name="Deformed",
                        line=dict(color="red", width=3, dash="dash"),
                        showlegend=(i == 0 and j == 1),
                        hovertemplate=f"Deformed {i}-{j}<extra></extra>",
                    )
                )

        # Plot load vectors
        if loads is not None:
            for idx, load in enumerate(loads):
                fx, fy, fz = load[:3]
                magnitude = np.linalg.norm([fx, fy, fz])

                if magnitude > 1e-6:
                    x0, y0, z0 = nodes[idx]
                    # Normalize and scale
                    scale = 0.5  # Adjust based on structure size
                    fx_n, fy_n, fz_n = scale * np.array([fx, fy, fz]) / magnitude

                    fig.add_trace(
                        go.Cone(
                            x=[x0],
                            y=[y0],
                            z=[z0],
                            u=[fx_n],
                            v=[fy_n],
                            w=[fz_n],
                            name="Loads",
                            colorscale="Greens",
                            showlegend=(idx == 0),
                            showscale=False,
                            hovertemplate=f"Load {idx}<br>F: {magnitude:.2f} kN<extra></extra>",
                        )
                    )

        # Apply configuration
        cfg = config or PlotConfig(
            title="3D Structural Model",
            xlabel="X (m)",
            ylabel="Y (m)",
            zlabel="Z (m)",
        )

        # 3D scene configuration
        fig.update_layout(
            title=cfg.title,
            scene=dict(
                xaxis=dict(title=cfg.xlabel, backgroundcolor="white", gridcolor="lightgray"),
                yaxis=dict(title=cfg.ylabel, backgroundcolor="white", gridcolor="lightgray"),
                zaxis=dict(title=cfg.zlabel, backgroundcolor="white", gridcolor="lightgray"),
                aspectmode="data",  # Equal aspect ratio
            ),
            showlegend=cfg.legend,
            width=cfg.width,
            height=cfg.height,
        )

        return fig

    def create_section_2d(
        self,
        vertices: NDArrayFloat,
        facets: Optional[List[List[int]]] = None,
        materials: Optional[List[str]] = None,
        show_centroid: bool = True,
        show_dimensions: bool = True,
        config: Optional[PlotConfig] = None,
    ) -> go.Figure:
        """
        Create 2D cross-section visualization with materials.

        Args:
            vertices: Vertex coordinates [N x 2] in meters
            facets: Facet connectivity [[v1, v2, v3, ...], ...]
            materials: Material names for each facet
            show_centroid: Show centroid marker
            show_dimensions: Show bounding box dimensions
            config: Plot configuration

        Returns:
            Plotly Figure object

        Examples:
            >>> verts = np.array([[0,0], [0.3,0], [0.3,0.5], [0,0.5]])
            >>> facets = [[0, 1, 2, 3]]
            >>> fig = engine.create_section_2d(verts, facets, materials=["steel"])
        """
        self._check_available()

        if vertices.shape[1] != 2:
            raise ValueError(f"Vertices must be N x 2 array, got shape {vertices.shape}")

        fig = go.Figure()

        # Plot facets (filled polygons)
        if facets is not None:
            for idx, facet in enumerate(facets):
                facet_verts = vertices[facet]

                # Get material color
                mat = materials[idx] if materials and idx < len(materials) else "steel"
                color = MATERIAL_COLORS_PLOTLY.get(mat.lower(), "#CCCCCC")

                # Close polygon
                x_coords = np.append(facet_verts[:, 0], facet_verts[0, 0])
                y_coords = np.append(facet_verts[:, 1], facet_verts[0, 1])

                fig.add_trace(
                    go.Scatter(
                        x=x_coords,
                        y=y_coords,
                        fill="toself",
                        fillcolor=color,
                        line=dict(color="black", width=2),
                        name=mat.capitalize(),
                        mode="lines",
                        showlegend=(idx == 0 or (materials and mat != materials[idx - 1])),
                        hovertemplate=f"{mat}<extra></extra>",
                    )
                )

        # Plot centroid
        if show_centroid:
            centroid = np.mean(vertices, axis=0)
            fig.add_trace(
                go.Scatter(
                    x=[centroid[0]],
                    y=[centroid[1]],
                    mode="markers",
                    name="Centroid",
                    marker=dict(size=10, color="black", symbol="cross"),
                    hovertemplate=f"Centroid<br>X: {centroid[0]:.4f}<br>Y: {centroid[1]:.4f}<extra></extra>",
                )
            )

        # Apply configuration
        cfg = config or PlotConfig(
            title="Cross-Section",
            xlabel="Width (m)",
            ylabel="Height (m)",
            equal_aspect=True,
        )
        self._apply_config(fig, cfg)

        return fig

    def create_stress_contour(
        self,
        x: NDArrayFloat,
        y: NDArrayFloat,
        stress: NDArrayFloat,
        stress_type: Literal["von_mises", "sigma_x", "sigma_y", "tau_xy"] = "von_mises",
        config: Optional[PlotConfig] = None,
    ) -> go.Figure:
        """
        Create stress contour plot using Plotly.

        Args:
            x: X coordinates [N x M] meshgrid
            y: Y coordinates [N x M] meshgrid
            stress: Stress values [N x M] in Pa
            stress_type: Type of stress
            config: Plot configuration

        Returns:
            Plotly Figure object with contour plot

        Examples:
            >>> x, y = np.meshgrid(np.linspace(0, 1, 50), np.linspace(0, 1, 50))
            >>> stress = 1e6 * (x**2 + y**2)  # Simple stress field
            >>> fig = engine.create_stress_contour(x, y, stress)
        """
        self._check_available()

        if x.shape != y.shape or x.shape != stress.shape:
            raise ValueError("x, y, and stress must have same shape")

        # Stress labels
        stress_labels = {
            "von_mises": "von Mises Stress σ_vm (MPa)",
            "sigma_x": "Normal Stress σ_x (MPa)",
            "sigma_y": "Normal Stress σ_y (MPa)",
            "tau_xy": "Shear Stress τ_xy (MPa)",
        }

        # Convert Pa to MPa
        stress_mpa = stress / 1e6

        fig = go.Figure(
            data=go.Contour(
                x=x[0, :],  # X coordinates (1D)
                y=y[:, 0],  # Y coordinates (1D)
                z=stress_mpa,
                colorscale=self._get_colorscale("stress"),
                colorbar=dict(title=stress_labels[stress_type]),
                contours=dict(showlabels=True, labelfont=dict(size=10)),
                hovertemplate="X: %{x:.4f}<br>Y: %{y:.4f}<br>Stress: %{z:.2f} MPa<extra></extra>",
            )
        )

        # Apply configuration
        cfg = config or PlotConfig(
            title=f"Stress Contour - {stress_type.replace('_', ' ').title()}",
            xlabel="X (m)",
            ylabel="Y (m)",
            equal_aspect=True,
        )
        self._apply_config(fig, cfg)

        return fig

    # ------------------------------------------------------------------------
    # EXPORT AND DISPLAY METHODS (implementing ABC)
    # ------------------------------------------------------------------------

    def export_static(
        self,
        figure: go.Figure,
        export_config: ExportConfig,
    ) -> Path:
        """
        Export Plotly figure to static image file using Kaleido.

        Args:
            figure: Plotly Figure object
            export_config: Export configuration

        Returns:
            Path to saved file

        Raises:
            RuntimeError: If Kaleido not available for non-HTML formats
            ValueError: If format not supported

        Examples:
            >>> fig = engine.create_pm_diagram(p, m)
            >>> path = engine.export_static(
            ...     fig,
            ...     ExportConfig(Path("diagram.png"), format=ImageFormat.PNG)
            ... )
        """
        self._check_available()

        fmt = export_config.format
        filename = export_config.corrected_filename

        # HTML export (always available)
        if fmt == ImageFormat.HTML:
            figure.write_html(
                str(filename),
                include_plotlyjs="cdn",
                config={"displayModeBar": True, "responsive": True},
            )
            logger.info(f"Exported interactive HTML: {filename}")
            return filename

        # Static formats require Kaleido
        if not KALEIDO_AVAILABLE:
            raise RuntimeError(
                f"Kaleido required for {fmt.value} export. "
                "Install with: pip install kaleido"
            )

        # Calculate dimensions
        width = int(figure.layout.width or 800) * export_config.scale
        height = int(figure.layout.height or 600) * export_config.scale

        # Export via Kaleido
        figure.write_image(
            str(filename),
            format=fmt.value,
            width=width,
            height=height,
            scale=1.0,  # Already applied in dimensions
        )

        logger.info(f"Exported static {fmt.value.upper()}: {filename}")
        return filename

    def show(self, figure: go.Figure, interactive: bool = False) -> None:
        """
        Display Plotly figure in browser or notebook.

        Args:
            figure: Plotly Figure to display
            interactive: Always True for Plotly (interactive by default)

        Examples:
            >>> fig = engine.create_pm_diagram(p, m)
            >>> engine.show(fig)  # Opens in browser
        """
        self._check_available()

        # Plotly auto-detects environment (notebook vs browser)
        figure.show()
        logger.info("Figure displayed")

    # ------------------------------------------------------------------------
    # ADVANCED FEATURES
    # ------------------------------------------------------------------------

    def create_subplots(
        self,
        rows: int,
        cols: int,
        subplot_titles: Optional[List[str]] = None,
        specs: Optional[List[List[Dict[str, Any]]]] = None,
    ) -> go.Figure:
        """
        Create figure with multiple subplots.

        Args:
            rows: Number of rows
            cols: Number of columns
            subplot_titles: Optional titles for each subplot
            specs: Optional subplot specifications (type, secondary_y, etc.)

        Returns:
            Plotly Figure with subplots

        Examples:
            >>> fig = engine.create_subplots(2, 2, subplot_titles=["A", "B", "C", "D"])
        """
        self._check_available()

        fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=subplot_titles,
            specs=specs,
        )

        return fig

    def enable_webgl(self, figure: go.Figure) -> None:
        """
        Enable WebGL rendering for better performance with large datasets.

        Args:
            figure: Plotly Figure to modify

        Examples:
            >>> fig = engine.create_3d_structure(nodes, elements)
            >>> engine.enable_webgl(fig)  # Faster rendering

        Note:
            WebGL acceleration works best for scatter plots with >1000 points.
            This is a no-op in Plotly 6.3+ due to API restrictions.
        """
        # Plotly 6.3+ has immutable trace types after creation
        # WebGL conversion must happen at trace creation time
        # This method is kept for API compatibility but does nothing
        logger.info("WebGL rendering (note: must be enabled at trace creation in Plotly 6.3+)")


    def export(
        self,
        fig,
        filename: str | Path,
        format: str = "png",
        dpi: int = 300,
        width: int = 1200,
        height: int = 800
    ) -> Path:
        """
        Export figure to file using intelligent exporter.
        
        Args:
            fig: Plotly figure
            filename: Output filename
            format: Image format (png, pdf, svg, jpg)
            dpi: Resolution
            width: Width in pixels
            height: Height in pixels
        
        Returns:
            Path to exported file
        
        Example:
            >>> engine = PlotlyEngine()
            >>> fig = engine.create_pm_diagram(...)
            >>> engine.export(fig, "diagram.png", dpi=300)
        """
        from .exporters import export_figure
        
        return export_figure(
            fig,
            filename,
            format=format,
            dpi=dpi,
            width=width,
            height=height
        )


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def check_plotly_installation() -> Dict[str, Any]:
    """
    Check Plotly and Kaleido installation status.

    Returns:
        Dictionary with installation details

    Examples:
        >>> status = check_plotly_installation()
        >>> print(status["plotly"]["available"])
        True
    """
    return {
        "plotly": {
            "available": PLOTLY_AVAILABLE,
            "version": PLOTLY_VERSION,
        },
        "kaleido": {
            "available": KALEIDO_AVAILABLE,
            "version": KALEIDO_VERSION,
        },
        "status": "ready" if PLOTLY_AVAILABLE else "missing dependencies",
        "install_command": "pip install 'pymemorial[viz]'",
    }


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "PlotlyEngine",
    "check_plotly_installation",
    "PLOTLY_AVAILABLE",
    "KALEIDO_AVAILABLE",
    "PLOTLY_VERSION",
    "KALEIDO_VERSION",
]
