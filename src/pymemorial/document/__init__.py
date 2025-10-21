# src/pymemorial/document/__init__.py
"""
PyMemorial Document Generation Module (PHASE 7).

This module provides a complete system for generating professional technical
documents (calculation memorials, reports, scientific articles) with:

Key Features
------------
- **Natural Language Processing**: Write in plain text, not LaTeX
- **Smart Variable Detection**: Automatic recognition of engineering symbols
- **Multi-Norm Support**: NBR 6118, NBR 8800, AISC 360, Eurocode
- **Multiple Output Formats**: PDF (WeasyPrint), HTML5, DOCX, LaTeX
- **Beautiful Templates**: Professional, norm-compliant designs
- **Full Integration**: Works seamlessly with PHASES 1-6
- **Automatic Numbering**: Sections, figures, tables, equations
- **Cross-References**: Intelligent linking between elements
- **Technical Verifications**: Pass/fail checks with norm compliance
- **Revision Control**: Track document changes and approvals

Quick Start
-----------
>>> from pymemorial.document import Memorial, DocumentMetadata, NormCode
>>> 
>>> # Create metadata
>>> metadata = DocumentMetadata(
...     title="Cálculo do Pilar PM-1",
...     author="Eng. João Silva, CREA: 12345/SP",
...     company="Estrutural Engenharia LTDA",
...     code=NormCode.NBR8800_2024
... )
>>> 
>>> # Create memorial
>>> memorial = Memorial(metadata)
>>> 
>>> # NEW FEATURE: Global context
>>> memorial.set_context(N_Sd=2500, N_Rd=3650, chi=0.877)
>>> 
>>> # NEW FEATURE: Natural language paragraph
>>> memorial.add_paragraph(\"\"\"
... Este memorial apresenta o dimensionamento do pilar PM-1.
... A resistência N_Rd = {N_Rd:.2f} kN supera a solicitação N_Sd = {N_Sd:.2f} kN.
... Com fator de redução chi = {chi:.3f}, o pilar está aprovado.
... \"\"\")
>>> 
>>> # Add figure
>>> memorial.add_figure(Path("diagram.png"), "Diagrama P-M")
>>> 
>>> # Render to PDF
>>> memorial.render("pilar_PM1.pdf")

Module Structure
----------------
**Core Components:**

- ``BaseDocument``: Abstract base class for all documents
- ``Memorial``: Calculation memorials (NBR, AISC, Eurocode)
- ``Report``: Technical reports (TODO)
- ``Article``: Scientific articles (TODO)

**Data Classes:**

- ``DocumentMetadata``: Author, company, norm, revision info
- ``Section``: Hierarchical document sections with auto-numbering
- ``Figure``: Images/plots with captions and auto-numbering
- ``Table``: Tabular data with styling
- ``EquationDoc``: LaTeX equations with numbering
- ``Verification``: Technical pass/fail checks
- ``Revision``: Revision control records

**Enums:**

- ``NormCode``: Engineering standards (NBR, AISC, Eurocode, ACI)
- ``CrossReferenceType``: Types of cross-references
- ``TableStyle``: Table styling presets
- ``DocumentLanguage``: Supported languages (PT-BR, EN-US, ES-ES)

**Generators (Output Formats):**

- ``WeasyPrintGenerator``: PDF generation (primary) - TODO
- ``HTMLGenerator``: Interactive HTML5 - TODO
- ``QuartoGenerator``: Scientific documents - TODO

**Internal Tools:**

- ``_internal/text_processing/``: Natural language processing - TODO
- ``_internal/symbol_management/``: Symbol database (NBR/AISC) - TODO
- ``_internal/html_utils/``: HTML rendering utilities - TODO

Examples
--------
**Example 1: Simple Memorial**

>>> from pymemorial.document import Memorial, DocumentMetadata, NormCode
>>> 
>>> metadata = DocumentMetadata(
...     title="Dimensionamento de Viga",
...     author="Eng. Maria Santos",
...     company="Construtora ABC",
...     code=NormCode.NBR6118_2023
... )
>>> 
>>> memorial = Memorial(metadata)
>>> memorial.add_section("Introdução", "Este memorial apresenta...", level=1)
>>> memorial.add_section("Dados de Entrada", "Vão: 6.0 m\\nCarga: 50 kN/m", level=2)
>>> memorial.render("viga.pdf")

**Example 2: With Natural Language (NEW)**

>>> memorial = Memorial(metadata)
>>> 
>>> # Set global context once
>>> memorial.set_context(
...     L=6.0,      # span
...     q=50,       # load
...     M_d=225,    # design moment
...     M_Rd=280    # resistance
... )
>>> 
>>> # Use variables throughout document
>>> memorial.add_paragraph(\"\"\"
... ## Dimensionamento
... 
... Para vão L = {L:.1f} m e carga q = {q:.0f} kN/m:
... 
... Momento fletor de cálculo: M_d = {M_d:.2f} kNm
... Momento resistente: M_Rd = {M_Rd:.2f} kNm
... 
... Verificação: M_d <= M_Rd OK!
... \"\"\")
>>> 
>>> memorial.render("viga.pdf")

**Example 3: With Figures from PHASE 6**

>>> from pymemorial.visualization import PlotlyEngine
>>> 
>>> memorial = Memorial(metadata)
>>> 
>>> # Create diagram with PHASE 6
>>> engine = PlotlyEngine()
>>> fig = engine.create_pm_diagram(section, N_values, M_values)
>>> 
>>> # Add to memorial (auto-exports and numbers)
>>> memorial.add_figure(fig, "Diagrama de Interação P-M")
>>> 
>>> memorial.render("memorial.pdf")

**Example 4: With Technical Verifications**

>>> memorial = Memorial(metadata)
>>> 
>>> # Add verification
>>> memorial.add_verification(
...     expression="N_Sd <= N_Rd",
...     passed=True,
...     description="Resistência à compressão",
...     norm=NormCode.NBR8800_2024,
...     item="5.3.2.1",
...     calculated_values={'N_Sd': 2500, 'N_Rd': 3650}
... )
>>> 
>>> memorial.render("verification.pdf")

Integration with Previous Phases
---------------------------------
**PHASE 1-2 (Core & Equations):**

>>> from pymemorial.core import Equation, Variable
>>> eq = Equation(...)
>>> memorial.add_equation(eq)  # Direct integration

**PHASE 3-5 (Sections):**

>>> from pymemorial.sections import SteelSection
>>> section = SteelSection("W200x46.1")
>>> memorial.set_context(A_g=section.properties.area)

**PHASE 6 (Visualization):**

>>> from pymemorial.visualization import PlotlyEngine
>>> engine = PlotlyEngine()
>>> fig = engine.create_pm_diagram(...)
>>> memorial.add_figure(fig, "P-M Diagram")  # Auto-exports

See Also
--------
- pymemorial.core : Core equation engine (PHASE 1-2)
- pymemorial.sections : Steel/concrete/composite sections (PHASE 3-5)
- pymemorial.backends : FEM analysis (PHASE 5)
- pymemorial.visualization : 2D/3D plotting (PHASE 6)

Notes
-----
**Natural Language Processing (NEW in PHASE 7):**

The module now supports writing paragraphs in natural language:
- Auto-detects variables (N_Rd, chi, f_y)
- Auto-formats LaTeX inline
- Substitutes placeholders {var:.2f}
- Uses global context + local variables

**Thread Safety:**

READ operations are thread-safe. WRITE operations (add_*) require
external synchronization if called from multiple threads.

**Performance:**

- Document creation: O(n) where n = number of elements
- TOC generation: O(n log n)
- Rendering: O(n) + generator overhead

Author: PyMemorial Team
Date: 2025-10-19
Version: 0.7.0
Phase: 7 (Document Generation)
Status: Alpha (base_document.py ✅, memorial.py ✅, generators TODO)
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
# IMPORTS - Base Document
# ============================================================================

try:
    # Base classes
    from .base_document import BaseDocument
    
    # Dataclasses
    from .base_document import (
        # Metadata
        DocumentMetadata,
        Revision,
        DocumentLanguage,
        
        # Content elements
        Section,
        Figure,
        Table,
        EquationDoc,
        Verification,
        NormReference,
        CrossReference,
        
        # Enums
        NormCode,
        CrossReferenceType,
        TableStyle,
        
        # Results
        ValidationResult,
        ValidationError,
        
        # Exceptions
        DocumentError,
        DocumentValidationError,
        RenderError,
        CrossReferenceError,
        
        # Helpers (useful for advanced users)
        SectionNumbering,
        ElementNumbering,
    )
    
    BASE_DOCUMENT_AVAILABLE = True

except ImportError as e:
    BASE_DOCUMENT_AVAILABLE = False
    warnings.warn(
        f"Failed to import base_document: {e}\n"
        "Document generation will not be available."
    )
    
    # Create placeholder to avoid NameError
    class BaseDocument:
        """Placeholder when base_document not available."""
        def __init__(self, *args, **kwargs):
            raise ImportError("base_document module not available")

# ============================================================================
# IMPORTS - Document Types
# ============================================================================

# Memorial - NOW AVAILABLE ✅
try:
    from .memorial import Memorial
    MEMORIAL_AVAILABLE = True
except ImportError as e:
    MEMORIAL_AVAILABLE = False
    warnings.warn(f"Memorial not available: {e}")
    
    # Placeholder
    class Memorial:
        """Placeholder when memorial not available."""
        def __init__(self, *args, **kwargs):
            raise ImportError("Memorial module not available. Check if memorial.py exists.")

# TODO: Report (PHASE 7.2)
# try:
#     from .report import Report
#     REPORT_AVAILABLE = True
# except ImportError:
#     REPORT_AVAILABLE = False
#     warnings.warn("Report not available (not yet implemented)")

# TODO: Article (PHASE 7.3)
# try:
#     from .article import Article
#     ARTICLE_AVAILABLE = True
# except ImportError:
#     ARTICLE_AVAILABLE = False
#     warnings.warn("Article not available (not yet implemented)")

# ============================================================================
# IMPORTS - Generators (TODO: PHASE 7.4)
# ============================================================================

# TODO: Generators
# try:
#     from .generators import (
#         BaseGenerator,
#         WeasyPrintGenerator,
#         HTMLGenerator,
#         QuartoGenerator,
#     )
#     GENERATORS_AVAILABLE = True
# except ImportError:
#     GENERATORS_AVAILABLE = False
#     warnings.warn("Generators not available (not yet implemented)")

# ============================================================================
# PUBLIC API
# ============================================================================

__all__ = [
    # Base
    'BaseDocument',
    
    # Documents
    'Memorial',      # ✅ AVAILABLE
    # 'Report',      # TODO: PHASE 7.2
    # 'Article',     # TODO: PHASE 7.3
    
    # Metadata
    'DocumentMetadata',
    'Revision',
    'DocumentLanguage',
    
    # Content elements
    'Section',
    'Figure',
    'Table',
    'EquationDoc',
    'Verification',
    'NormReference',
    'CrossReference',
    
    # Enums
    'NormCode',
    'CrossReferenceType',
    'TableStyle',
    
    # Results
    'ValidationResult',
    'ValidationError',
    
    # Exceptions
    'DocumentError',
    'DocumentValidationError',
    'RenderError',
    'CrossReferenceError',
    
    # Helpers
    'SectionNumbering',
    'ElementNumbering',
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
    if BASE_DOCUMENT_AVAILABLE:
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
    if BASE_DOCUMENT_AVAILABLE:
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
        - report: Report document type (TODO)
        - article: Article document type (TODO)
        - generators: PDF/HTML/DOCX generators (TODO)
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
        'generators': False,
        'phase1_2': True,
        'phase3_5': True,
        'phase6': True
    }
    """
    deps = {
        'base_document': BASE_DOCUMENT_AVAILABLE,
        'memorial': MEMORIAL_AVAILABLE,
        # 'report': REPORT_AVAILABLE,     # TODO
        # 'article': ARTICLE_AVAILABLE,   # TODO
        # 'generators': GENERATORS_AVAILABLE,  # TODO
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
            f"Install missing packages or check if files exist."
        )

# Run check on import
_check_and_warn()

# ============================================================================
# CONVENIENCE IMPORTS (for advanced users)
# ============================================================================

# Users can do: from pymemorial.document import BaseDocument, Memorial, etc.
# Or: import pymemorial.document as doc; doc.Memorial(...)
