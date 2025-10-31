# src/pymemorial/builder/__init__.py

"""
PyMemorial Builder - High-Level API (v3.0)

Arquitetura em 3 níveis:
    NÍVEL 1 (HIGH-LEVEL): MatrixMemorial - API simplificada para usuários
    NÍVEL 2 (LEGACY): MemorialBuilder - API detalhada existente
    NÍVEL 3 (CORE): Matrix, Variable - Engine de baixo nível

Examples:
    >>> # HIGH-LEVEL API (NOVO - 3 linhas!)
    >>> from pymemorial.builder import MatrixMemorial
    >>> m = MatrixMemorial("Edifício Residencial", "Eng. João Silva")
    >>> m.add_beam("viga", L=6, E=21e6, I=0.0008)
    >>> m.generate("portico.md")
    
    >>> # LEGACY API (compatibilidade total)
    >>> from pymemorial.builder import MemorialBuilder
    >>> builder = MemorialBuilder("Memorial de Cálculo")
    >>> builder.add_section("Introdução")
"""

import logging
from typing import Dict, Any

# ============================================================================
# NÍVEL 1: HIGH-LEVEL API (NOVO)
# ============================================================================

try:
    from .matrix_memorial import (
        MatrixMemorial,
        MemorialConfig
    )
except ImportError as e:
    logging.warning(f"MatrixMemorial import falhou: {e}. High-level API não disponível.")
    MatrixMemorial = MemorialConfig = None


# ============================================================================
# NÍVEL 2: LEGACY BUILDERS (COMPATIBILIDADE)
# ============================================================================

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


# ============================================================================
# BUNDLE FACTORIES (QUICK SETUP)
# ============================================================================

def get_builder_bundle(nlp: bool = False) -> Dict[str, Any]:
    """
    Bundle completo de ferramentas builder (LEGACY).
    
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


def get_matrix_memorial(project: str = None, author: str = None, **kwargs) -> 'MatrixMemorial':
    """
    Factory para criação rápida de MatrixMemorial (HIGH-LEVEL API).
    
    Args:
        project: Nome do projeto (opcional)
        author: Responsável técnico (opcional)
        **kwargs: Configurações adicionais (norm, precision, etc)
    
    Returns:
        MatrixMemorial: Instância configurada
    
    Example:
        >>> m = get_matrix_memorial("Ponte Metálica", "Eng. Maria Santos")
        >>> m.add_beam("principal", L=12, E=200e6, I=0.002)
        >>> m.generate("ponte.md")
    
    Raises:
        ImportError: Se MatrixMemorial não estiver disponível
    """
    if MatrixMemorial is None:
        raise ImportError(
            "MatrixMemorial não disponível. "
            "Verifique se matrix_memorial.py está no diretório builder/"
        )
    
    project = project or "Projeto Estrutural"
    author = author or "Engenheiro Responsável"
    
    return MatrixMemorial(project, author, **kwargs)


# ============================================================================
# EXPORTS PÚBLICOS
# ============================================================================

__all__ = [
    # ========================================================================
    # HIGH-LEVEL API (v3.0) - RECOMENDADO PARA NOVOS PROJETOS
    # ========================================================================
    "MatrixMemorial",           # Builder simplificado para análise estrutural
    "MemorialConfig",           # Configuração global do memorial
    "get_matrix_memorial",      # Factory para MatrixMemorial
    
    # ========================================================================
    # LEGACY API (v2.0) - COMPATIBILIDADE RETROATIVA
    # ========================================================================
    "MemorialBuilder",          # Builder detalhado existente
    "MemorialMetadata",         # Metadados do memorial
    "Section",                  # Seções do memorial
    "ContentBlock",             # Blocos de conteúdo
    "ContentType",              # Tipos de conteúdo
    "create_text_block",        # Helper: bloco de texto
    "create_equation_block",    # Helper: bloco de equação
    "create_figure_block",      # Helper: bloco de figura
    "create_table_block",       # Helper: bloco de tabela
    "MemorialValidator",        # Validador de memoriais
    "ValidationError",          # Erro de validação
    "ValidationReport",         # Relatório de validação
    
    # ========================================================================
    # BUNDLE FACTORIES
    # ========================================================================
    "get_builder_bundle",       # Bundle de ferramentas legacy
]


# ============================================================================
# VERSÃO E METADADOS
# ============================================================================

__version__ = "3.0.0"
__author__ = "PyMemorial Team"
__description__ = "High-Level API para geração automatizada de memoriais de cálculo estrutural"


# ============================================================================
# HELPER: Detectar módulos disponíveis
# ============================================================================

def get_available_modules() -> Dict[str, bool]:
    """
    Retorna status de disponibilidade dos módulos.
    
    Returns:
        Dict com True/False para cada módulo
    
    Example:
        >>> from pymemorial.builder import get_available_modules
        >>> status = get_available_modules()
        >>> print(status)
        {
            'matrix_memorial': True,
            'memorial_builder': True,
            'validators': True,
            ...
        }
    """
    return {
        # High-Level
        'matrix_memorial': MatrixMemorial is not None,
        'memorial_config': MemorialConfig is not None,
        
        # Legacy
        'memorial_builder': MemorialBuilder is not None,
        'memorial_metadata': MemorialMetadata is not None,
        'section': Section is not None,
        'content_block': ContentBlock is not None,
        'validators': MemorialValidator is not None,
    }


# ============================================================================
# LOG DE INICIALIZAÇÃO
# ============================================================================

logger = logging.getLogger(__name__)
logger.info(f"PyMemorial Builder v{__version__} carregado")

# Log de módulos disponíveis
modules_status = get_available_modules()
available = [k for k, v in modules_status.items() if v]
unavailable = [k for k, v in modules_status.items() if not v]

if available:
    logger.info(f"Módulos disponíveis: {', '.join(available)}")
if unavailable:
    logger.warning(f"Módulos não disponíveis: {', '.join(unavailable)}")
