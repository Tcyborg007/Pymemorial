# src/pymemorial/builder/__init__.py
"""
API pública do builder (v2.0: Enhanced Exports + Bundle).

Exports legacy + novos (validate_chain). Bundle factory para quick setup.
Fallbacks robustos. Compatível 100%.

Example Bundle:
    bundle = get_builder_bundle()
    builder = bundle['builder']
    builder.add_variable("M_k", 150).add_section("Análise")
"""

import logging
from typing import Dict, Any

# Legacy imports with fallbacks
try:
    from .memorial import MemorialBuilder, MemorialMetadata
except ImportError as e:
    logging.warning(f"Memorial import falhou: {e}.")
    MemorialBuilder = MemorialMetadata = None

try:
    from .section import Section
except ImportError as e:
    logging.warning(f"Section import falhou: {e}.")
    Section = None

try:
    from .content import (
        ContentBlock,
        ContentType,
        create_text_block,
        create_equation_block,
        create_figure_block,
        create_table_block,
    )
except ImportError as e:
    logging.warning(f"Content import falhou: {e}.")
    ContentBlock = ContentType = None
    create_text_block = create_equation_block = None
    create_figure_block = create_table_block = None

try:
    from .validators import (
        MemorialValidator,
        ValidationError,
        ValidationReport,
    )
except ImportError as e:
    logging.warning(f"Validators import falhou: {e}.")
    MemorialValidator = ValidationError = ValidationReport = None


# NEW: Bundle Factory (one-stop setup)
def get_builder_bundle(nlp: bool = False) -> Dict[str, Any]:
    """
    Bundle completo de ferramentas builder.
    
    Args:
        nlp: Habilita NLP (não implementado no MVP, ignorado)
    
    Returns:
        Dict com: builder, validator, content creators
    
    Example:
        >>> bundle = get_builder_bundle()
        >>> builder = bundle['builder']("Memorial de Cálculo")
        >>> builder.add_variable("M_k", 150)
    """
    return {
        'builder': MemorialBuilder if MemorialBuilder else None,
        'validator': MemorialValidator if MemorialValidator else None,
        'content': {
            'text': create_text_block,
            'equation': create_equation_block,
            'figure': create_figure_block,
            'table': create_table_block,
        },
    }


__all__ = [
    # Legacy
    "MemorialBuilder",
    "MemorialMetadata",
    "Section",
    "ContentBlock",
    "ContentType",
    "create_text_block",
    "create_equation_block",
    "create_figure_block",
    "create_table_block",
    "MemorialValidator",
    "ValidationError",
    "ValidationReport",
    
    # Enhanced
    "get_builder_bundle",
]
