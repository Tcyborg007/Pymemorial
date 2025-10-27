# src/pymemorial/document/__init__.py
"""
PyMemorial Document Generation Module (PHASE 7) - MVP Compatible.

This module provides a complete system for generating professional technical
documents (calculation memorials, reports, scientific articles) with:

Key Features
------------
- **Smart Text Processing**: Write in plain text with auto-detection
- **Multi-Norm Support**: NBR 6118, NBR 8800, AISC 360, Eurocode
- **Multiple Output Formats**: PDF, HTML5, DOCX, LaTeX
- **Automatic Numbering**: Sections, figures, tables, equations
- **Cross-References**: Intelligent linking between elements
- **Technical Verifications**: Pass/fail checks with norm compliance
- **Full Integration**: Works seamlessly with PHASES 1-6

Quick Start
-----------
>>> from pymemorial.document import Memorial, DocumentMetadata, NormCode
>>> 
>>> # Create metadata
>>> metadata = DocumentMetadata(
...     title="Cálculo do Pilar PM-1",
...     author="Eng. João Silva",
...     company="Estrutural Engenharia LTDA",
...     norm_code=NormCode.NBR8800_2024  # MVP: norm_code (não code)
... )
>>> 
>>> # Create memorial
>>> memorial = Memorial(metadata)
>>> 
>>> # Add content
>>> section = memorial.add_section("Introdução", level=1)
>>> memorial.add_paragraph("Este memorial apresenta...", parent=section)
>>> 
>>> # Render to PDF
>>> memorial.render("pilar_PM1.pdf")

Author: PyMemorial Team
Date: 2025-10-21
Version: 0.7.0 (MVP Compatible)
Phase: 7 (Document Generation)
Status: Alpha
"""

from __future__ import annotations

import warnings

# ============================================================================
# VERSION INFO
# ============================================================================

__version__ = '0.7.0'
__author__ = 'PyMemorial Team'
__phase__ = 7
__status__ = 'alpha'

# ============================================================================
# IMPORTS - Base Document (MVP Compatible)
# ============================================================================

try:
    # Base classes
    from .base_document import BaseDocument
    
    # Dataclasses (MVP Compatible - Remove classes que não existem)
    from .base_document import (
        # Metadata
        DocumentMetadata,
        DocumentLanguage,
        
        # Content elements (Fallbacks definidos em base_document.py)
        Section,
        Figure,
        Table,
        EquationDoc,
        Verification,
        CrossReference,
        
        # Enums
        NormCode,
        CrossReferenceType,
        TableStyle,
        
        # Results
        ValidationError,
        
        # Exceptions
        DocumentError,
        DocumentValidationError,
        RenderError,
        CrossReferenceError,
        NormComplianceError,  # MVP: novo
        
        # Compliance
        NormCompliance,  # MVP: substitui NormReference
    )
    
    BASE_DOCUMENT_AVAILABLE = True

except ImportError as e:
    BASE_DOCUMENT_AVAILABLE = False
    warnings.warn(
        f"Failed to import base_document: {e}\n"
        "Document generation will not be available.",
        ImportWarning
    )
    
    # Create placeholder to avoid NameError
    class BaseDocument:
        """Placeholder when base_document not available."""
        def __init__(self, *args, **kwargs):
            raise ImportError("base_document module not available")
    
    # Placeholder for other classes
    DocumentMetadata = None
    NormCode = None

# ============================================================================
# IMPORTS - Document Types
# ============================================================================

# Memorial - NOW AVAILABLE ✅
try:
    from .memorial import Memorial
    MEMORIAL_AVAILABLE = True
except ImportError as e:
    MEMORIAL_AVAILABLE = False
    warnings.warn(f"Memorial not available: {e}", ImportWarning)
    
    # Placeholder
    class Memorial:
        """Placeholder when memorial not available."""
        def __init__(self, *args, **kwargs):
            raise ImportError("Memorial module not available. Check if memorial.py exists.")

# TODO: Report (PHASE 7.2)
REPORT_AVAILABLE = False

# TODO: Article (PHASE 7.3)
ARTICLE_AVAILABLE = False

# TODO: Generators (PHASE 7.4)
GENERATORS_AVAILABLE = False

# ============================================================================
# PUBLIC API (MVP Compatible)
# ============================================================================

__all__ = [
    # Base
    'BaseDocument',
    
    # Documents
    'Memorial',      # ✅ AVAILABLE
    
    # Metadata
    'DocumentMetadata',
    'DocumentLanguage',
    
    # Content elements
    'Section',
    'Figure',
    'Table',
    'EquationDoc',
    'Verification',
    'CrossReference',
    
    # Enums
    'NormCode',
    'CrossReferenceType',
    'TableStyle',
    
    # Results
    'ValidationError',
    
    # Exceptions
    'DocumentError',
    'DocumentValidationError',
    'RenderError',
    'CrossReferenceError',
    'NormComplianceError',  # MVP
    
    # Compliance
    'NormCompliance',  # MVP
]

# ============================================================================
# MODULE-LEVEL FUNCTIONS
# ============================================================================

def get_version() -> str:
    """
    Get module version.
    
    Returns
    -------
    str
        Version string (e.g., "0.7.0")
    
    Examples
    --------
    >>> from pymemorial.document import get_version
    >>> print(get_version())
    '0.7.0'
    """
    return __version__

def get_available_norms() -> list:
    """
    Get list of available engineering norms.
    
    Returns
    -------
    List[NormCode]
        List of supported norms
    
    Examples
    --------
    >>> from pymemorial.document import get_available_norms
    >>> norms = get_available_norms()
    >>> print(norms)
    [<NormCode.NBR6118_2023>, <NormCode.NBR8800_2024>, ...]
    """
    if BASE_DOCUMENT_AVAILABLE and NormCode:
        return list(NormCode)
    else:
        return []

def get_available_languages() -> list:
    """
    Get list of available document languages.
    
    Returns
    -------
    List[DocumentLanguage]
        List of supported languages
    
    Examples
    --------
    >>> from pymemorial.document import get_available_languages
    >>> langs = get_available_languages()
    >>> print(langs)
    [<DocumentLanguage.PT_BR>, <DocumentLanguage.EN_US>, ...]
    """
    if BASE_DOCUMENT_AVAILABLE and DocumentLanguage:
        return list(DocumentLanguage)
    else:
        return []

def check_dependencies() -> dict:
    """
    Check availability of optional dependencies.
    
    Returns
    -------
    Dict[str, bool]
        Dictionary with dependency status:
        - base_document: Core document functionality
        - memorial: Memorial document type
        - phase1_2: Core & Equations (PHASE 1-2)
        - phase3_5: Sections (PHASE 3-5)
        - phase6: Visualization & Exporters (PHASE 6)
    
    Examples
    --------
    >>> from pymemorial.document import check_dependencies
    >>> deps = check_dependencies()
    >>> print(deps)
    {
        'base_document': True,
        'memorial': True,
        'phase1_2': True,
        'phase3_5': True,
        'phase6': True
    }
    """
    deps = {
        'base_document': BASE_DOCUMENT_AVAILABLE,
        'memorial': MEMORIAL_AVAILABLE,
        'report': REPORT_AVAILABLE,
        'article': ARTICLE_AVAILABLE,
        'generators': GENERATORS_AVAILABLE,
    }
    
    # Check PHASE 1-6 dependencies
    try:
        from pymemorial.core import Equation
        deps['phase1_2'] = True
    except ImportError:
        deps['phase1_2'] = False
    
    try:
        from pymemorial.sections import SteelSection
        deps['phase3_5'] = True
    except ImportError:
        deps['phase3_5'] = False
    
    try:
        from pymemorial.visualization import PlotlyEngine
        deps['phase6'] = True
    except ImportError:
        deps['phase6'] = False
    
    return deps

# ============================================================================
# MODULE INITIALIZATION
# ============================================================================

def _check_and_warn():
    """Check dependencies and warn if missing."""
    deps = check_dependencies()
    
    missing = [name for name, available in deps.items() if not available]
    
    if missing:
        warnings.warn(
            f"PyMemorial Document module loaded with missing dependencies: {', '.join(missing)}\n"
            f"Some features may not be available.\n"
            f"Install missing packages or check if files exist.",
            ImportWarning
        )

# Run check on import
_check_and_warn()

# ============================================================================
# CONVENIENCE IMPORTS
# ============================================================================

# Users can do: from pymemorial.document import BaseDocument, Memorial, etc.
# Or: import pymemorial.document as doc; doc.Memorial(...)
