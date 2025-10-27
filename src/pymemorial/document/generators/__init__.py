# src/pymemorial/document/generators/__init__.py
"""
Document Generators Module - PyMemorial v2.0 (Production Ready)

Professional document generation for structural calculation memorials:
- WeasyPrintGenerator: Fast PDF generation (10x faster than Quarto)
- HTMLGenerator: Standalone HTML generation
- QuartoGenerator: Academic documents with ABNT support (optional)
- LaTeXGenerator: Professional typesetting with ABNTeX2 (future)

Author: PyMemorial Team
Date: 2025-10-21
Version: 2.0.0
Phase: 7.3
"""

from __future__ import annotations

import logging
from typing import Optional

# ============================================================================
# LOGGER
# ============================================================================

logger = logging.getLogger(__name__)

# ============================================================================
# BASE GENERATOR
# ============================================================================

try:
    from .base_generator import (
        BaseGenerator,
        GeneratorConfig,
        PageConfig,
        PDFMetadata,
        GenerationError,
        ValidationError,
    )
    logger.debug("BaseGenerator loaded")
except ImportError as e:
    logger.warning(f"BaseGenerator import failed: {e}")
    BaseGenerator = None
    GeneratorConfig = None
    PageConfig = None
    PDFMetadata = None
    GenerationError = Exception
    ValidationError = Exception

# ============================================================================
# WEASYPRINT GENERATOR
# ============================================================================

try:
    from .weasyprint_generator import WeasyPrintGenerator, generate_pdf
    WEASYPRINT_AVAILABLE = True
    logger.debug("WeasyPrintGenerator loaded")
except ImportError as e:
    logger.debug(f"WeasyPrint not available: {e}")
    WEASYPRINT_AVAILABLE = False
    WeasyPrintGenerator = None
    generate_pdf = None

# ============================================================================
# HTML GENERATOR
# ============================================================================

try:
    from .html_generator import HTMLGenerator
    HTML_AVAILABLE = True
    logger.debug("HTMLGenerator loaded")
except ImportError as e:
    logger.debug(f"HTMLGenerator not available: {e}")
    HTML_AVAILABLE = False
    HTMLGenerator = None

# ============================================================================
# QUARTO GENERATOR (OPTIONAL)
# ============================================================================

try:
    from .quarto_generator import QuartoGenerator
    QUARTO_AVAILABLE = True
    logger.debug("QuartoGenerator loaded")
except ImportError as e:
    logger.debug(f"Quarto not available: {e}")
    QUARTO_AVAILABLE = False
    QuartoGenerator = None

# ============================================================================
# LATEX GENERATOR (FUTURE)
# ============================================================================

try:
    from .latex_generator import LaTeXGenerator
    LATEX_AVAILABLE = True
    logger.debug("LaTeXGenerator loaded")
except ImportError as e:
    logger.debug(f"LaTeX not available: {e}")
    LATEX_AVAILABLE = False
    LaTeXGenerator = None

# ============================================================================
# FACTORY
# ============================================================================

def get_generator(generator_type: str = 'weasyprint', **kwargs):
    """
    Factory function to get generator instance by type.
    
    Args:
        generator_type: Generator type ('weasyprint', 'html', 'quarto', 'latex')
        **kwargs: Generator-specific configuration
    
    Returns:
        Generator instance
    
    Raises:
        ValueError: If generator type is unknown
        ImportError: If generator dependencies are not available
    
    Examples:
    --------
    >>> from pymemorial.document.generators import get_generator
    >>> 
    >>> # Get WeasyPrint generator
    >>> generator = get_generator('weasyprint')
    >>> generator.generate(memorial, 'output.pdf', style='nbr')
    >>> 
    >>> # Get HTML generator
    >>> html_gen = get_generator('html')
    >>> html_gen.generate(memorial, 'output.html', style='modern')
    """
    generators = {
        'weasyprint': (WeasyPrintGenerator, WEASYPRINT_AVAILABLE, 'weasyprint'),
        'html': (HTMLGenerator, HTML_AVAILABLE, None),
        'quarto': (QuartoGenerator, QUARTO_AVAILABLE, 'quarto'),
        'latex': (LaTeXGenerator, LATEX_AVAILABLE, 'texlive or miktex'),
    }
    
    if generator_type not in generators:
        raise ValueError(
            f"Unknown generator type: '{generator_type}'. "
            f"Available: {list(generators.keys())}"
        )
    
    generator_class, available, install_cmd = generators[generator_type]
    
    if not available or generator_class is None:
        install_msg = f"Install via: pip install {install_cmd}" if install_cmd else "Dependencies missing"
        raise ImportError(
            f"Generator '{generator_type}' is not available. {install_msg}"
        )
    
    return generator_class(**kwargs)


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

def generate_memorial_pdf(memorial, output_path: str, style: str = 'nbr', **kwargs):
    """
    Convenience function to generate memorial PDF.
    
    Args:
        memorial: Memorial document instance
        output_path: Output PDF file path
        style: CSS style ('nbr', 'modern', 'aisc')
        **kwargs: Additional generator options
    
    Examples:
    --------
    >>> from pymemorial.document import Memorial
    >>> from pymemorial.document.generators import generate_memorial_pdf
    >>> 
    >>> memorial = Memorial(title="Test Memorial")
    >>> generate_memorial_pdf(memorial, 'output.pdf', style='nbr')
    """
    if not WEASYPRINT_AVAILABLE:
        raise ImportError(
            "WeasyPrint is not available. Install via: pip install weasyprint"
        )
    
    generator = WeasyPrintGenerator()
    generator.generate(memorial, output_path, style=style, **kwargs)


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Base classes
    'BaseGenerator',
    'GeneratorConfig',
    'PageConfig',
    'PDFMetadata',
    
    # Exceptions
    'GenerationError',
    'ValidationError',
    
    # Generators
    'WeasyPrintGenerator',
    'HTMLGenerator',
    'QuartoGenerator',
    'LaTeXGenerator',
    
    # Factory and convenience
    'get_generator',
    'generate_pdf',
    'generate_memorial_pdf',
    
    # Availability flags
    'WEASYPRINT_AVAILABLE',
    'HTML_AVAILABLE',
    'QUARTO_AVAILABLE',
    'LATEX_AVAILABLE',
]

# Version
__version__ = '2.0.0'
