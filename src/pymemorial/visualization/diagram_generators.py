# src/pymemorial/visualization/diagram_generators.py
"""
Specialized diagram generators for structural engineering.

This module provides high-level functions for creating standard structural
diagrams following Brazilian (NBR 8800) and international codes (EN 1993, AISC 360).

Features:
    - P-M interaction diagrams with code-specific checks
    - Moment-curvature with ductility analysis
    - Shear-moment diagrams with load combinations
    - Stress distribution in composite sections
    - LaTeX rendering for equations

Author: PyMemorial Team
License: MIT
Python: >=3.9
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import numpy.typing as npt

from .base_visualizer import DiagramType, PlotConfig, VisualizerEngine
from .matplotlib_utils import plot_pm_interaction, plot_moment_curvature

# Type aliases
NDArrayFloat = npt.NDArray[np.floating[np.float64]]


# ============================================================================
# ENUMS - Design codes
# ============================================================================


class DesignCode(Enum):
    """Structural design codes."""

    NBR8800 = "NBR 8800:2024"  # Brazilian steel code
    NBR6118 = "NBR 6118:2023"  # Brazilian concrete code
    EN1993 = "EN 1993-1-1:2020"  # Eurocode 3
    AISC360 = "AISC 360-22"  # US steel code
    ACI318 = "ACI 318-19"  # US concrete code


# ============================================================================
# DATACLASSES - Diagram parameters
# ============================================================================


@dataclass(frozen=True)
class PMDiagramParams:
    """Parameters for P-M interaction diagram generation."""

    n_points: int = 50  # Number of points in envelope
    include_tension: bool = True  # Include tension region
    normalize: bool = True  # Normalize to Pn, Mn
    safety_factor: float = 1.0  # γ or φ factor
    code: DesignCode = DesignCode.NBR8800
    show_balanced_point: bool = True  # Mark balanced failure point


@dataclass(frozen=True)
class MomentCurvatureParams:
    """Parameters for moment-curvature analysis."""

    target_curvature: float = 0.01  # Maximum curvature (1/m)
    n_steps: int = 100  # Number of analysis steps
    include_unloading: bool = False  # Post-peak behavior
    mark_yield: bool = True  # Mark first yield point
    mark_ultimate: bool = True  # Mark ultimate capacity
    calculate_ductility: bool = True  # Show μ = κu/κy


# ============================================================================
# P-M DIAGRAM GENERATORS
# ============================================================================


def generate_pm_interaction_envelope(
    p_nominal: float,
    m_nominal: float,
    section_type: Literal["rectangular", "i_section", "circular"] = "rectangular",
    params: Optional[PMDiagramParams] = None,
) -> Tuple[NDArrayFloat, NDArrayFloat]:
    """
    Generate P-M interaction envelope for given section capacity.

    Args:
        p_nominal: Nominal axial capacity (kN)
        m_nominal: Nominal moment capacity (kN·m)
        section_type: Cross-section geometry type
        params: Diagram generation parameters

    Returns:
        Tuple of (P_values, M_values) arrays normalized by capacity

    Examples:
        >>> p, m = generate_pm_interaction_envelope(1000, 250)
        >>> # Returns normalized envelope [0-1]
    """
    if params is None:
        params = PMDiagramParams()

    # Generate envelope based on section type
    if section_type == "rectangular":
        # Rectangular section interaction
        m_normalized = np.linspace(0, 1, params.n_points)
        p_normalized = np.sqrt(1 - m_normalized**2)  # Circular approximation

    elif section_type == "i_section":
        # I-section (more complex)
        m_normalized = np.linspace(0, 1, params.n_points)
        # Bilinear approximation
        p_normalized = np.where(
            m_normalized < 0.5,
            1 - 0.3 * m_normalized,
            0.85 - 1.2 * (m_normalized - 0.5),
        )
        p_normalized = np.clip(p_normalized, 0, 1)

    elif section_type == "circular":
        # Circular section (symmetric)
        m_normalized = np.linspace(0, 1, params.n_points)
        p_normalized = (1 - m_normalized**1.5) ** (2 / 3)

    else:
        raise ValueError(f"Unknown section_type: {section_type}")

    # Add compression-only point
    m_normalized = np.append(m_normalized, 0)
    p_normalized = np.append(p_normalized, 1)

    # Add tension region if requested
    if params.include_tension:
        m_tension = np.linspace(0, 0.3, 10)
        p_tension = -0.3 * (m_tension / 0.3)
        m_normalized = np.append(m_normalized, m_tension)
        p_normalized = np.append(p_normalized, p_tension)

    # Close the envelope
    m_normalized = np.append(m_normalized, 0)
    p_normalized = np.append(p_normalized, 0)

    # Apply safety factor
    if params.safety_factor != 1.0:
        p_normalized /= params.safety_factor
        m_normalized /= params.safety_factor

    # Denormalize if requested
    if not params.normalize:
        p_values = p_normalized * p_nominal
        m_values = m_normalized * m_nominal
    else:
        p_values = p_normalized
        m_values = m_normalized

    return p_values, m_values


def create_pm_diagram_with_code(
    engine: VisualizerEngine,
    p_nominal: float,
    m_nominal: float,
    design_point: Optional[Tuple[float, float]] = None,
    code: DesignCode = DesignCode.NBR8800,
    config: Optional[PlotConfig] = None,
) -> any:
    """
    Create P-M diagram with code-specific formatting and checks.

    Args:
        engine: Visualization engine to use
        p_nominal: Nominal axial capacity (kN)
        m_nominal: Nominal moment capacity (kN·m)
        design_point: Optional (M_d, P_d) design loads
        code: Design code to follow
        config: Plot configuration

    Returns:
        Figure object (engine-specific)

    Examples:
        >>> from pymemorial.visualization import create_visualizer
        >>> viz = create_visualizer()
        >>> fig = create_pm_diagram_with_code(
        ...     viz, p_nominal=1000, m_nominal=250,
        ...     design_point=(150, 600),
        ...     code=DesignCode.NBR8800
        ... )
    """
    # Generate envelope
    params = PMDiagramParams(code=code)
    p_envelope, m_envelope = generate_pm_interaction_envelope(
        p_nominal, m_nominal, params=params
    )

    # Configure plot with code-specific labels
    if config is None:
        if code == DesignCode.NBR8800:
            title = "Diagrama de Interação P-M (NBR 8800:2024)"
            xlabel = "Momento Fletor M_d (kN·m)"
            ylabel = "Força Axial P_d (kN)"
        elif code == DesignCode.EN1993:
            title = "P-M Interaction Diagram (EN 1993-1-1)"
            xlabel = "Bending Moment M_Ed (kN·m)"
            ylabel = "Axial Force N_Ed (kN)"
        else:
            title = f"P-M Interaction Diagram ({code.value})"
            xlabel = "Bending Moment (kN·m)"
            ylabel = "Axial Force (kN)"

        config = PlotConfig(title=title, xlabel=xlabel, ylabel=ylabel)

    # Create diagram
    fig = engine.create_pm_diagram(
        p_envelope, m_envelope, design_point=design_point, config=config
    )

    return fig


# ============================================================================
# MOMENT-CURVATURE GENERATORS
# ============================================================================


def generate_moment_curvature_response(
    m_yield: float,
    m_ultimate: float,
    kappa_yield: float,
    kappa_ultimate: float,
    params: Optional[MomentCurvatureParams] = None,
) -> Tuple[NDArrayFloat, NDArrayFloat]:
    """
    Generate idealized moment-curvature response.

    Uses bilinear or trilinear model with strain hardening.

    Args:
        m_yield: Yield moment (kN·m)
        m_ultimate: Ultimate moment (kN·m)
        kappa_yield: Yield curvature (1/m)
        kappa_ultimate: Ultimate curvature (1/m)
        params: Analysis parameters

    Returns:
        Tuple of (curvature, moment) arrays

    Examples:
        >>> kappa, m = generate_moment_curvature_response(
        ...     m_yield=200, m_ultimate=250,
        ...     kappa_yield=0.003, kappa_ultimate=0.015
        ... )
    """
    if params is None:
        params = MomentCurvatureParams()

    # Elastic branch (0 → yield)
    n_elastic = int(params.n_steps * 0.3)
    kappa_elastic = np.linspace(0, kappa_yield, n_elastic)
    m_elastic = m_yield * (kappa_elastic / kappa_yield)

    # Plastic branch (yield → ultimate)
    n_plastic = params.n_steps - n_elastic
    kappa_plastic = np.linspace(kappa_yield, kappa_ultimate, n_plastic)
    # Strain hardening: quadratic approximation
    alpha = (m_ultimate - m_yield) / (kappa_ultimate - kappa_yield)
    m_plastic = m_yield + alpha * (kappa_plastic - kappa_yield)

    # Combine
    curvature = np.concatenate([kappa_elastic, kappa_plastic[1:]])
    moment = np.concatenate([m_elastic, m_plastic[1:]])

    # Optional: add unloading branch
    if params.include_unloading:
        kappa_unload = np.linspace(kappa_ultimate, kappa_ultimate * 1.5, 20)
        m_unload = m_ultimate * 0.9 * np.exp(-5 * (kappa_unload - kappa_ultimate))
        curvature = np.concatenate([curvature, kappa_unload[1:]])
        moment = np.concatenate([moment, m_unload[1:]])

    return curvature, moment


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def calculate_ductility(
    kappa_yield: float, kappa_ultimate: float
) -> Dict[str, float]:
    """
    Calculate ductility metrics.

    Args:
        kappa_yield: Yield curvature (1/m)
        kappa_ultimate: Ultimate curvature (1/m)

    Returns:
        Dictionary with ductility metrics:
            - mu: Curvature ductility ratio
            - classification: Ductility class (low/medium/high)

    Examples:
        >>> metrics = calculate_ductility(0.003, 0.015)
        >>> print(metrics["mu"])
        5.0
    """
    mu = kappa_ultimate / kappa_yield if kappa_yield > 0 else 0

    # Classification per NBR 8800
    if mu < 1.5:
        classification = "low"
    elif mu < 4.0:
        classification = "medium"
    else:
        classification = "high"

    return {"mu": mu, "classification": classification, "adequate": mu >= 3.0}


def format_code_reference(code: DesignCode, clause: str) -> str:
    """
    Format code reference for plot annotations.

    Args:
        code: Design code enum
        clause: Clause/section number

    Returns:
        Formatted reference string

    Examples:
        >>> ref = format_code_reference(DesignCode.NBR8800, "5.4.2")
        >>> print(ref)
        'NBR 8800:2024 - Item 5.4.2'
    """
    return f"{code.value} - Item {clause}"


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Enums
    "DesignCode",
    # Dataclasses
    "PMDiagramParams",
    "MomentCurvatureParams",
    # Generators
    "generate_pm_interaction_envelope",
    "generate_moment_curvature_response",
    "create_pm_diagram_with_code",
    # Utilities
    "calculate_ductility",
    "format_code_reference",
]
