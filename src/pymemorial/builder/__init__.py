"""API pública do módulo builder."""
from .memorial import MemorialBuilder, MemorialMetadata
from .section import Section
from .content import (
    ContentBlock,
    ContentType,
    create_text_block,
    create_equation_block,
    create_figure_block,
    create_table_block,
)
from .validators import MemorialValidator, ValidationError, validate_template

__all__ = [
    # Builder principal
    "MemorialBuilder",
    "MemorialMetadata",
    # Estrutura
    "Section",
    # Conteúdo
    "ContentBlock",
    "ContentType",
    "create_text_block",
    "create_equation_block",
    "create_figure_block",
    "create_table_block",
    # Validação
    "MemorialValidator",
    "ValidationError",
    "validate_template",
]
