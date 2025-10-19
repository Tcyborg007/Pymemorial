# src/pymemorial/visualization/matplotlib_utils.py
"""
Matplotlib Utilities and Extensions for PyMemorial.

This module provides high-level utilities and helper functions for creating
publication-quality engineering plots using matplotlib. All functions are
CI/CD compatible (headless mode via Agg backend).

Features:
    - Publication-quality styling presets
    - Engineering dimension annotations with arrows
    - Material hatching patterns (steel, concrete, wood, etc.)
    - P-M and M-κ diagram generators with code compliance
    - Smart axis formatting (engineering notation, SI units)
    - 3D structure plotting with loads and displacements
    - Stress contour plots with custom colormaps
    - Automatic legend positioning and styling
    - LaTeX rendering for equations and symbols

Requirements:
    - matplotlib >= 3.9.0 (latest: 3.10.0)
    - numpy >= 1.24.0

Author: PyMemorial Team
License: MIT
Python: >=3.9
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import numpy.typing as npt

# Force Agg backend for CI/CD compatibility (no GUI required)
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, Circle, Polygon, Rectangle
from matplotlib.collections import LineCollection, PolyCollection
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# Type aliases
NDArrayFloat = npt.NDArray[np.floating[Any]]
ColorType = Union[str, Tuple[float, float, float], Tuple[float, float, float, float]]

# ============================================================================
# CONSTANTS - Material properties and engineering standards
# ============================================================================

# Material colors following international standards
MATERIAL_COLORS: Dict[str, ColorType] = {
    "steel": "#B0C4DE",  # Light steel blue
    "concrete": "#D3D3D3",  # Light gray
    "wood": "#DEB887",  # Burlywood
    "aluminum": "#E8E8E8",  # Very light gray
    "masonry": "#CD853F",  # Peru
    "composite": "#98D8C8",  # Light teal
    "rebar": "#2F4F4F",  # Dark slate gray
}

# Edge colors for materials
MATERIAL_EDGE_COLORS: Dict[str, ColorType] = {
    "steel": "#000080",  # Navy
    "concrete": "#505050",  # Dark gray
    "wood": "#8B4513",  # Saddle brown
    "aluminum": "#696969",  # Dim gray
    "masonry": "#8B4513",  # Saddle brown
    "composite": "#2F4F4F",  # Dark slate gray
    "rebar": "#000000",  # Black
}

# Hatch patterns for materials (following ISO/ABNT standards)
MATERIAL_HATCHES: Dict[str, str] = {
    "steel": "//",  # Diagonal lines (ISO 128)
    "concrete": "..",  # Dotted pattern (ABNT NBR 6492)
    "wood": "||",  # Vertical lines (grain direction)
    "aluminum": "\\\\",  # Reverse diagonal
    "masonry": "++",  # Crosshatch
    "composite": "xx",  # Dense crosshatch
    "rebar": "oo",  # Circles
}

# Line styles for different load types
LOAD_LINE_STYLES: Dict[str, str] = {
    "dead": "-",  # Solid (permanent loads)
    "live": "--",  # Dashed (variable loads)
    "wind": "-.",  # Dash-dot (wind loads)
    "seismic": ":",  # Dotted (seismic loads)
    "thermal": (0, (5, 5)),  # Custom dash pattern
}


# ============================================================================
# STYLING FUNCTIONS - Publication-quality presets
# ============================================================================


def set_publication_style(
    style: Literal[
        "default", "ieee", "nature", "asce", "presentation", "thesis"
    ] = "default",
) -> None:
    """
    Set matplotlib rcParams for publication-quality figures.

    This function configures matplotlib with sensible defaults for different
    publication venues. Settings persist for the current session.

    Args:
        style: Publication style preset:
            - "default": General-purpose clean style
            - "ieee": IEEE Transactions format (6.5" wide, 2-column)
            - "nature": Nature journal format (89mm single, 183mm double)
            - "asce": ASCE journal format (3.5" single, 7" double column)
            - "presentation": High-contrast for slides (16:9 ratio)
            - "thesis": Dissertation/thesis format (A4 width)

    Examples:
        >>> set_publication_style("ieee")
        >>> fig, ax = plt.subplots(figsize=(3.5, 2.5))
        >>> # Figure is now sized for IEEE single-column

    References:
        - IEEE: https://journals.ieeeauthorcenter.ieee.org/
        - Nature: https://www.nature.com/nature/for-authors/formatting-guide
        - ASCE: https://ascelibrary.org/page/informationforauthors

    Notes:
        Call this BEFORE creating figures. To reset, call matplotlib.rcdefaults()
    """
    # Base settings common to all styles
    base_params = {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "lines.linewidth": 1.5,
        "lines.markersize": 6,
        "axes.linewidth": 0.8,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
        "axes.grid": True,
        "axes.axisbelow": True,  # Grid behind plot elements
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "text.usetex": False,  # Use mathtext instead of LaTeX (faster)
        "mathtext.fontset": "dejavusans",
    }

    # Style-specific overrides
    style_params: Dict[str, Dict[str, Any]] = {
        "ieee": {
            "figure.figsize": (3.5, 2.625),  # Single column (2:3 aspect)
            "font.size": 8,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 7,
        },
        "nature": {
            "figure.figsize": (3.5, 2.625),  # 89mm single column
            "font.family": "sans-serif",
            "font.size": 7,
            "axes.labelsize": 8,
            "legend.fontsize": 6,
        },
        "asce": {
            "figure.figsize": (3.5, 2.625),  # Single column
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 9,
        },
        "presentation": {
            "figure.figsize": (10, 5.625),  # 16:9 aspect ratio
            "font.size": 14,
            "axes.labelsize": 16,
            "axes.titlesize": 18,
            "legend.fontsize": 12,
            "lines.linewidth": 2.5,
            "axes.linewidth": 1.2,
        },
        "thesis": {
            "figure.figsize": (6, 4),  # A4-friendly
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 14,
            "legend.fontsize": 10,
        },
    }

    # Apply base + style-specific params
    params = {**base_params, **style_params.get(style, {})}
    plt.rcParams.update(params)


def reset_style() -> None:
    """Reset matplotlib to default style."""
    matplotlib.rcdefaults()


# ============================================================================
# DIMENSION ANNOTATION FUNCTIONS
# ============================================================================


def add_dimension_arrow(
    ax: Axes,
    start: Tuple[float, float],
    end: Tuple[float, float],
    text: str,
    offset: float = 0.05,
    color: str = "red",
    fontsize: int = 10,
    arrow_style: str = "<|-|>",
    text_position: Literal["above", "below", "left", "right", "center"] = "center",
    decimal_places: int = 3,
) -> Tuple[FancyArrowPatch, plt.Text]:
    """
    Add dimension arrow with text annotation (engineering drawing style).

    Creates a bidirectional arrow with centered text, following ISO 129-1
    and ABNT NBR 10126 standards for technical drawings.

    Args:
        ax: Matplotlib axes to draw on
        start: Starting point (x, y) in data coordinates
        end: Ending point (x, y) in data coordinates
        text: Dimension text (use empty string for auto-calculation)
        offset: Text offset from arrow line (fraction of figure height)
        color: Arrow and text color
        fontsize: Font size for dimension text
        arrow_style: Matplotlib arrowstyle ('<|-|>', '<->', '->', etc.)
        text_position: Text placement relative to arrow
        decimal_places: Decimal places for auto-calculated dimensions

    Returns:
        Tuple of (arrow_patch, text_object) for further customization

    Examples:
        >>> fig, ax = plt.subplots()
        >>> arrow, txt = add_dimension_arrow(
        ...     ax, (0, 0), (5, 0), "5.000 m", offset=0.1
        ... )
        >>> # Auto-calculate dimension
        >>> add_dimension_arrow(ax, (0, 0), (0, 3), "", offset=0.1)
    """
    x1, y1 = start
    x2, y2 = end

    # Calculate distance if text is empty
    if not text:
        distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        text = f"{distance:.{decimal_places}f}"

    # Create arrow annotation
    arrow = ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle=arrow_style,
            color=color,
            lw=2.5,
            shrinkA=0,
            shrinkB=0,
        ),
    )

    # Calculate text position based on arrow orientation
    mid_x = (x1 + x2) / 2
    mid_y = (y1 + y2) / 2

    # Get figure bbox for offset calculation
    fig_height = ax.get_figure().get_figheight()
    offset_val = offset * fig_height

    # Determine text offset direction
    angle = np.arctan2(y2 - y1, x2 - x1)
    is_horizontal = abs(np.cos(angle)) > 0.7
    is_vertical = abs(np.sin(angle)) > 0.7

    if text_position == "above":
        text_x, text_y = mid_x, mid_y + offset_val
    elif text_position == "below":
        text_x, text_y = mid_x, mid_y - offset_val
    elif text_position == "left":
        text_x, text_y = mid_x - offset_val, mid_y
    elif text_position == "right":
        text_x, text_y = mid_x + offset_val, mid_y
    else:  # center
        if is_horizontal:
            text_x, text_y = mid_x, mid_y + offset_val * np.sign(y2 - y1 + 0.01)
        elif is_vertical:
            text_x, text_y = mid_x + offset_val, mid_y
        else:
            text_x, text_y = mid_x, mid_y + offset_val

    # Add text annotation
    text_obj = ax.text(
        text_x,
        text_y,
        text,
        color=color,
        fontsize=fontsize,
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=color, alpha=0.8),
    )

    return arrow, text_obj


def add_multiple_dimensions(
    ax: Axes,
    points: List[Tuple[float, float]],
    labels: Optional[List[str]] = None,
    **kwargs: Any,
) -> List[Tuple[FancyArrowPatch, plt.Text]]:
    """
    Add multiple dimension arrows connecting sequential points.

    Args:
        ax: Matplotlib axes
        points: List of (x, y) coordinates to connect
        labels: Optional labels for each segment (auto-calculated if None)
        **kwargs: Additional arguments passed to add_dimension_arrow()

    Returns:
        List of (arrow, text) tuples

    Examples:
        >>> points = [(0, 0), (5, 0), (5, 3), (0, 3)]
        >>> add_multiple_dimensions(ax, points, color="blue")
    """
    annotations = []
    for i in range(len(points) - 1):
        label = labels[i] if labels and i < len(labels) else ""
        ann = add_dimension_arrow(ax, points[i], points[i + 1], label, **kwargs)
        annotations.append(ann)
    return annotations


# ============================================================================
# MATERIAL HATCHING FUNCTIONS
# ============================================================================


def add_section_hatch(
    ax: Axes,
    vertices: NDArrayFloat,
    material: str = "steel",
    alpha: float = 0.7,
    edgecolor: Optional[ColorType] = None,
    facecolor: Optional[ColorType] = None,
    linewidth: float = 1.5,
    hatch_density: int = 3,
) -> Polygon:
    """
    Add hatched polygon for material representation (ISO/ABNT standards).

    Creates a filled polygon with appropriate hatching pattern for the
    specified material type, following engineering drawing conventions.

    Args:
        ax: Matplotlib axes
        vertices: Array of shape (N, 2) with polygon vertices
        material: Material type (steel, concrete, wood, aluminum, masonry)
        alpha: Face transparency (0-1)
        edgecolor: Edge color (defaults to material standard)
        facecolor: Fill color (defaults to material standard)
        linewidth: Edge line width
        hatch_density: Hatch pattern density (1-5)

    Returns:
        Polygon patch object

    Examples:
        >>> vertices = np.array([[0,0], [1,0], [1,1], [0,1]])
        >>> patch = add_section_hatch(ax, vertices, material="concrete")

    References:
        - ISO 128-50:2001 (Technical drawings - Hatching)
        - ABNT NBR 8403:1984 (Technical drawing conventions)
    """
    material_lower = material.lower()

    # Get default colors
    if facecolor is None:
        facecolor = MATERIAL_COLORS.get(material_lower, "#CCCCCC")
    if edgecolor is None:
        edgecolor = MATERIAL_EDGE_COLORS.get(material_lower, "#000000")

    # Get hatch pattern with density
    base_hatch = MATERIAL_HATCHES.get(material_lower, "//")
    hatch = base_hatch * hatch_density

    # Create polygon patch
    patch = Polygon(
        vertices,
        closed=True,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        alpha=alpha,
        hatch=hatch,
        label=material.capitalize(),
    )

    ax.add_patch(patch)
    return patch


def plot_composite_section(
    ax: Axes,
    materials: List[Dict[str, Any]],
    show_centroid: bool = True,
    show_dimensions: bool = True,
    centroid: Optional[Tuple[float, float]] = None,
) -> List[Polygon]:
    """
    Plot composite section with multiple materials.

    Args:
        ax: Matplotlib axes
        materials: List of material dicts with keys:
            - "vertices": NDArrayFloat (N, 2)
            - "material": str
            - "alpha": Optional[float]
        show_centroid: Show centroid marker
        show_dimensions: Show bounding box dimensions
        centroid: Manual centroid position (auto-calculated if None)

    Returns:
        List of polygon patches

    Examples:
        >>> materials = [
        ...     {"vertices": steel_verts, "material": "steel"},
        ...     {"vertices": concrete_verts, "material": "concrete"}
        ... ]
        >>> patches = plot_composite_section(ax, materials)
    """
    patches = []

    # Plot each material
    for mat_data in materials:
        vertices = mat_data["vertices"]
        material = mat_data.get("material", "steel")
        alpha = mat_data.get("alpha", 0.7)

        patch = add_section_hatch(ax, vertices, material=material, alpha=alpha)
        patches.append(patch)

    # Calculate and plot centroid
    if show_centroid:
        if centroid is None:
            # Calculate centroid from all vertices
            all_verts = np.vstack([m["vertices"] for m in materials])
            centroid = (np.mean(all_verts[:, 0]), np.mean(all_verts[:, 1]))

        ax.plot(
            centroid[0],
            centroid[1],
            "ko",
            markersize=8,
            label="Centroid",
            zorder=10,
        )
        ax.plot(centroid[0], centroid[1], "k+", markersize=12, zorder=10)

    # Add bounding box dimensions
    if show_dimensions:
        all_verts = np.vstack([m["vertices"] for m in materials])
        xmin, ymin = all_verts.min(axis=0)
        xmax, ymax = all_verts.max(axis=0)

        # Width dimension
        add_dimension_arrow(
            ax,
            (xmin, ymin - 0.05),
            (xmax, ymin - 0.05),
            f"{xmax - xmin:.3f} m",
            offset=0.02,
        )

        # Height dimension
        add_dimension_arrow(
            ax,
            (xmax + 0.05, ymin),
            (xmax + 0.05, ymax),
            f"{ymax - ymin:.3f} m",
            offset=0.02,
        )

    ax.set_aspect("equal")
    ax.legend(loc="best")

    return patches


# ============================================================================
# ENGINEERING DIAGRAM GENERATORS
# ============================================================================


def plot_pm_interaction(
    ax: Axes,
    p_values: NDArrayFloat,
    m_values: NDArrayFloat,
    design_point: Optional[Tuple[float, float]] = None,
    capacity_envelope: bool = True,
    code: Literal["NBR8800", "EN1993", "AISC360"] = "NBR8800",
    safety_factor: float = 1.0,
    normalized: bool = True,
) -> Tuple[plt.Line2D, ...]:
    """
    Plot P-M interaction diagram with code-specific formatting.

    Creates publication-quality P-M diagram following structural code
    conventions (NBR 8800, Eurocode 3, AISC 360).

    Args:
        ax: Matplotlib axes
        p_values: Axial force values (positive = compression)
        m_values: Bending moment values
        design_point: Optional (M_d, P_d) design point to highlight
        capacity_envelope: Draw capacity envelope curve
        code: Design code standard
        safety_factor: Safety/resistance factor (γ or φ)
        normalized: Whether values are normalized (P/P_n, M/M_n)

    Returns:
        Tuple of line objects for customization

    Examples:
        >>> p = np.array([0, 0.5, 1.0, 0.8, 0.3, 0])
        >>> m = np.array([0.6, 0.8, 0.5, 0.4, 0.7, 0])
        >>> plot_pm_interaction(ax, p, m, design_point=(0.4, 0.6))

    References:
        - NBR 8800:2024 - Brazilian steel structures code
        - EN 1993-1-1:2020 - Eurocode 3 Part 1-1
        - AISC 360-22 - US steel construction manual
    """
    # Validate inputs
    if len(p_values) != len(m_values):
        raise ValueError("P and M arrays must have same length")

    # Apply safety factor
    p_plot = p_values / safety_factor
    m_plot = m_values / safety_factor

    # Plot capacity envelope
    (line,) = ax.plot(
        m_plot,
        p_plot,
        "b-",
        linewidth=2.5,
        label=f"Capacity Envelope ({code})",
        zorder=5,
    )

    # Fill capacity region
    ax.fill(m_plot, p_plot, alpha=0.1, color="blue", label="Safe Region")

    # Plot design point
    design_line = None
    if design_point is not None:
        m_d, p_d = design_point
        design_line = ax.plot(
            m_d,
            p_d,
            "ro",
            markersize=10,
            label="Design Point",
            zorder=10,
        )[0]

        # Check if point is inside envelope
        from matplotlib.path import Path

        path = Path(np.column_stack([m_plot, p_plot]))
        is_safe = path.contains_point([m_d, p_d])

        color = "green" if is_safe else "red"
        status = "SAFE" if is_safe else "UNSAFE"

        ax.annotate(
            f"Design Point\n({m_d:.3f}, {p_d:.3f})\n{status}",
            xy=(m_d, p_d),
            xytext=(10, 10),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.5", facecolor=color, alpha=0.3),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
        )

    # Formatting
    if normalized:
        ax.set_xlabel("M / M$_n$ (Normalized Moment)", fontsize=12)
        ax.set_ylabel("P / P$_n$ (Normalized Axial Load)", fontsize=12)
    else:
        ax.set_xlabel("Bending Moment M (kN·m)", fontsize=12)
        ax.set_ylabel("Axial Load P (kN)", fontsize=12)

    ax.set_title(f"P-M Interaction Diagram ({code})", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    ax.axhline(0, color="k", linewidth=0.5)
    ax.axvline(0, color="k", linewidth=0.5)

    return (line, design_line) if design_line else (line,)


def plot_moment_curvature(
    ax: Axes,
    curvature: NDArrayFloat,
    moment: NDArrayFloat,
    yield_point: Optional[Tuple[float, float]] = None,
    ultimate_point: Optional[Tuple[float, float]] = None,
    show_ductility: bool = True,
) -> Tuple[plt.Line2D, ...]:
    """
    Plot moment-curvature (M-κ) diagram for section analysis.

    Args:
        ax: Matplotlib axes
        curvature: Curvature values κ (1/m or 1/km)
        moment: Moment values M (kN·m or MN·m)
        yield_point: Optional (κ_y, M_y) first yield
        ultimate_point: Optional (κ_u, M_u) ultimate capacity
        show_ductility: Show ductility ratio μ = κ_u / κ_y

    Returns:
        Tuple of line objects

    Examples:
        >>> kappa = np.linspace(0, 0.01, 100)
        >>> m = 1000 * kappa * (1 - 0.5 * kappa / 0.01)
        >>> plot_moment_curvature(ax, kappa, m)
    """
    # Main curve
    (line,) = ax.plot(
        curvature,
        moment,
        "b-",
        linewidth=2.5,
        label="M-κ Response",
    )

    # Yield point
    yield_line = None
    if yield_point:
        kappa_y, m_y = yield_point
        yield_line = ax.plot(
            kappa_y, m_y, "go", markersize=10, label="Yield Point", zorder=10
        )[0]
        ax.axvline(kappa_y, color="green", linestyle="--", alpha=0.5)
        ax.axhline(m_y, color="green", linestyle="--", alpha=0.5)

    # Ultimate point
    ult_line = None
    if ultimate_point:
        kappa_u, m_u = ultimate_point
        ult_line = ax.plot(
            kappa_u, m_u, "ro", markersize=10, label="Ultimate Point", zorder=10
        )[0]
        ax.axvline(kappa_u, color="red", linestyle="--", alpha=0.5)
        ax.axhline(m_u, color="red", linestyle="--", alpha=0.5)

        # Ductility ratio
        if show_ductility and yield_point:
            kappa_y = yield_point[0]
            mu = kappa_u / kappa_y if kappa_y > 0 else 0
            ax.text(
                0.05,
                0.95,
                f"Ductility μ = κ$_u$ / κ$_y$ = {mu:.2f}",
                transform=ax.transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

    # Formatting
    ax.set_xlabel("Curvature κ (1/km)", fontsize=12)
    ax.set_ylabel("Moment M (MN·m)", fontsize=12)
    ax.set_title("Moment-Curvature Diagram", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    return (line, yield_line, ult_line)


def plot_shear_moment_diagrams(
    ax: Axes,
    x: NDArrayFloat,
    shear: NDArrayFloat,
    moment: NDArrayFloat,
    load_positions: Optional[List[float]] = None,
) -> Tuple[Figure, Tuple[Axes, Axes]]:
    """
    Plot shear force and bending moment diagrams for beam analysis.

    Args:
        ax: Matplotlib axes (will be split into 2 subplots)
        x: Position along beam (m)
        shear: Shear force values (kN)
        moment: Bending moment values (kN·m)
        load_positions: Optional list of concentrated load positions

    Returns:
        Tuple of (figure, (ax_shear, ax_moment))

    Examples:
        >>> x = np.linspace(0, 10, 100)
        >>> v = 100 - 20*x  # Linear shear
        >>> m = 100*x - 10*x**2  # Parabolic moment
        >>> plot_shear_moment_diagrams(ax, x, v, m)
    """
    # Clear existing axes and create subplots
    fig = ax.get_figure()
    ax.remove()

    ax_shear = fig.add_subplot(211)
    ax_moment = fig.add_subplot(212)

    # Shear diagram
    ax_shear.plot(x, shear, "b-", linewidth=2)
    ax_shear.fill_between(x, shear, alpha=0.2, color="blue")
    ax_shear.axhline(0, color="k", linewidth=0.5)
    ax_shear.set_ylabel("Shear Force V (kN)", fontsize=11)
    ax_shear.set_title("Shear Force Diagram", fontsize=12, fontweight="bold")
    ax_shear.grid(True, alpha=0.3)

    # Moment diagram
    ax_moment.plot(x, moment, "r-", linewidth=2)
    ax_moment.fill_between(x, moment, alpha=0.2, color="red")
    ax_moment.axhline(0, color="k", linewidth=0.5)
    ax_moment.set_xlabel("Position x (m)", fontsize=11)
    ax_moment.set_ylabel("Bending Moment M (kN·m)", fontsize=11)
    ax_moment.set_title("Bending Moment Diagram", fontsize=12, fontweight="bold")
    ax_moment.grid(True, alpha=0.3)

    # Mark load positions
    if load_positions:
        for xp in load_positions:
            ax_shear.axvline(xp, color="gray", linestyle=":", alpha=0.5)
            ax_moment.axvline(xp, color="gray", linestyle=":", alpha=0.5)

    fig.tight_layout()
    return fig, (ax_shear, ax_moment)


# ============================================================================
# 3D STRUCTURE VISUALIZATION
# ============================================================================


def plot_3d_structure(
    ax: Union[Axes, Axes3D],
    nodes: NDArrayFloat,
    elements: List[Tuple[int, int]],
    displacements: Optional[NDArrayFloat] = None,
    loads: Optional[NDArrayFloat] = None,
    scale_displacement: float = 1.0,
    scale_load: float = 1.0,
    show_node_labels: bool = True,
) -> Axes3D:
    """
    Plot 3D structural frame with optional displacements and loads.

    Args:
        ax: Matplotlib 3D axes (or 2D axes to be converted)
        nodes: Node coordinates [N x 3] in meters
        elements: Element connectivity [(node_i, node_j), ...]
        displacements: Optional nodal displacements [N x 3] in meters
        loads: Optional nodal loads [N x 6] (Fx, Fy, Fz, Mx, My, Mz) in kN
        scale_displacement: Displacement magnification factor
        scale_load: Load arrow scale factor
        show_node_labels: Show node numbers

    Returns:
        Axes3D object

    Examples:
        >>> nodes = np.array([[0,0,0], [5,0,0], [5,0,3], [0,0,3]])
        >>> elements = [(0,1), (1,2), (2,3), (3,0)]
        >>> fig = plt.figure()
        >>> ax = fig.add_subplot(111, projection='3d')
        >>> plot_3d_structure(ax, nodes, elements)
    """
    # Convert to 3D axes if needed
    if not isinstance(ax, Axes3D):
        fig = ax.get_figure()
        ax.remove()
        ax = fig.add_subplot(111, projection="3d")

    # Plot undeformed structure
    for i, j in elements:
        xs = [nodes[i, 0], nodes[j, 0]]
        ys = [nodes[i, 1], nodes[j, 1]]
        zs = [nodes[i, 2], nodes[j, 2]]
        ax.plot(xs, ys, zs, "b-", linewidth=2, alpha=0.7, label="Undeformed")

    # Plot nodes
    ax.scatter(
        nodes[:, 0],
        nodes[:, 1],
        nodes[:, 2],
        c="blue",
        s=50,
        marker="o",
        label="Nodes",
    )

    # Plot node labels
    if show_node_labels:
        for idx, (x, y, z) in enumerate(nodes):
            ax.text(x, y, z, f"  {idx}", fontsize=9)

    # Plot deformed structure
    if displacements is not None:
        deformed = nodes + displacements * scale_displacement
        for i, j in elements:
            xs = [deformed[i, 0], deformed[j, 0]]
            ys = [deformed[i, 1], deformed[j, 1]]
            zs = [deformed[i, 2], deformed[j, 2]]
            ax.plot(xs, ys, zs, "r--", linewidth=1.5, alpha=0.7, label="Deformed")

    # Plot load vectors
    if loads is not None:
        for idx, load in enumerate(loads):
            fx, fy, fz = load[:3]  # Force components
            if np.linalg.norm([fx, fy, fz]) > 1e-6:
                x, y, z = nodes[idx]
                ax.quiver(
                    x,
                    y,
                    z,
                    fx,
                    fy,
                    fz,
                    length=scale_load,
                    color="green",
                    arrow_length_ratio=0.3,
                    linewidth=2,
                )

    # Formatting
    ax.set_xlabel("X (m)", fontsize=11)
    ax.set_ylabel("Y (m)", fontsize=11)
    ax.set_zlabel("Z (m)", fontsize=11)
    ax.set_title("3D Structural Model", fontsize=13, fontweight="bold")

    # Equal aspect ratio
    max_range = (
        np.array(
            [
                nodes[:, 0].max() - nodes[:, 0].min(),
                nodes[:, 1].max() - nodes[:, 1].min(),
                nodes[:, 2].max() - nodes[:, 2].min(),
            ]
        ).max()
        / 2.0
    )
    mid_x = (nodes[:, 0].max() + nodes[:, 0].min()) * 0.5
    mid_y = (nodes[:, 1].max() + nodes[:, 1].min()) * 0.5
    mid_z = (nodes[:, 2].max() + nodes[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # Remove duplicate labels
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="best")

    return ax


# ============================================================================
# AXIS FORMATTING UTILITIES
# ============================================================================


def format_engineering_axis(
    ax: Axes,
    axis: Literal["x", "y", "both"] = "both",
    precision: int = 2,
    use_si_prefix: bool = True,
) -> None:
    """
    Format axis with engineering notation (powers of 1000) and SI prefixes.

    Args:
        ax: Matplotlib axes
        axis: Which axis to format ("x", "y", or "both")
        precision: Number of decimal places
        use_si_prefix: Use SI prefixes (k, M, G) instead of scientific notation

    Examples:
        >>> format_engineering_axis(ax, "y", precision=3)
        >>> # Y-axis now shows: 1.234M instead of 1.234e6
    """
    from matplotlib.ticker import EngFormatter, FuncFormatter

    if use_si_prefix:
        formatter = EngFormatter(unit="", places=precision)
    else:
        formatter = FuncFormatter(lambda x, p: f"{x:.{precision}e}")

    if axis in ("x", "both"):
        ax.xaxis.set_major_formatter(formatter)
    if axis in ("y", "both"):
        ax.yaxis.set_major_formatter(formatter)


def set_equal_aspect_3d(ax: Axes3D) -> None:
    """
    Set equal aspect ratio for 3D plots.

    Args:
        ax: Matplotlib 3D axes

    Examples:
        >>> fig = plt.figure()
        >>> ax = fig.add_subplot(111, projection='3d')
        >>> plot_3d_structure(ax, nodes, elements)
        >>> set_equal_aspect_3d(ax)
    """
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    spans = np.abs(limits[:, 1] - limits[:, 0])
    centers = np.mean(limits, axis=1)
    radius = 0.5 * max(spans)

    ax.set_xlim3d([centers[0] - radius, centers[0] + radius])
    ax.set_ylim3d([centers[1] - radius, centers[1] + radius])
    ax.set_zlim3d([centers[2] - radius, centers[2] + radius])


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Styling
    "set_publication_style",
    "reset_style",
    # Dimensions
    "add_dimension_arrow",
    "add_multiple_dimensions",
    # Materials
    "add_section_hatch",
    "plot_composite_section",
    # Diagrams
    "plot_pm_interaction",
    "plot_moment_curvature",
    "plot_shear_moment_diagrams",
    # 3D
    "plot_3d_structure",
    # Formatting
    "format_engineering_axis",
    "set_equal_aspect_3d",
    # Constants
    "MATERIAL_COLORS",
    "MATERIAL_HATCHES",
    "MATERIAL_EDGE_COLORS",
]
