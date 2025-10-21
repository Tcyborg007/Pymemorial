# src/pymemorial/document/generators/__init__.py
"""
Document Generators Module - PDF, HTML, LaTeX generation.

This module provides multiple generators for creating professional documents:
- WeasyPrintGenerator: Fast PDF generation (10x faster than Quarto)
- QuartoGenerator: Academic documents with ABNT support
- PlaywrightGenerator: 3D visualization support
- LaTeXGenerator: Professional typesetting with ABNTeX2

Author: PyMemorial Team
Date: 2025-10-20
Phase: 7.3
"""

from pymemorial.document.generators.base_generator import (
    BaseGenerator,
    GeneratorConfig,
    PageConfig,
    PDFMetadata,
    GenerationError,
    ValidationError,
)

from pymemorial.document.generators.weasyprint_generator import (
    WeasyPrintGenerator,
    generate_pdf,
)


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
    
    # Convenience functions
    'generate_pdf',
]


__version__ = '2.0.0'
