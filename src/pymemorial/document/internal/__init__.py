# src/pymemorial/document/internal/__init__.py
"""
Internal utilities for PyMemorial document module.

Author: PyMemorial Team
Date: 2025-10-21
"""

from .text_processing.text_utils import (
    TemplateProcessor,
    process_template,
    extract_variables,
)

__all__ = [
    'TemplateProcessor',
    'process_template',
    'extract_variables',
]
