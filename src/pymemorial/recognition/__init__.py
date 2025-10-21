# src/pymemorial/recognition/__init__.py
"""
API pública do recognition (v2.0: MVP simplificado).

Exports legacy + novo (SmartTextEngine, get_engine). 
Bundle factory para quick setup.
Compatibilidade 100%; lazy imports para evitar erros.

Exemplo Bundle:
    bundle = get_recognition_bundle(nlp=False)
    processed = bundle['processor'].process_text(text, context)
    
Exemplo Direto:
    from pymemorial.recognition import SmartTextEngine
    engine = SmartTextEngine()
    result = engine.process_text("M_k = 150 kN", {'M_k': 150})
"""

import logging
from typing import Dict, Any


# ============================================================================
# GREEK SYMBOLS (com fallback robusto)
# ============================================================================
try:
    from .greek import GreekSymbols, ASCII_TO_GREEK
    # FIX: GREEK_TO_ASCII pode não existir em greek.py
    try:
        from .greek import GREEK_TO_ASCII
    except ImportError:
        # Cria reverso se não existir
        GREEK_TO_ASCII = {v: k for k, v in ASCII_TO_GREEK.items()} if ASCII_TO_GREEK else {}
except ImportError as e:
    logging.warning(f"Greek import falhou: {e}. Usando fallback.")
    # Stub simples
    class GreekSymbols:
        @staticmethod
        def to_unicode(text: str) -> str:
            return text
    ASCII_TO_GREEK = {}
    GREEK_TO_ASCII = {}


# ============================================================================
# PARSER (com fallback)
# ============================================================================
try:
    from .parser import VariableParser, ParsedVariable
except ImportError as e:
    logging.warning(f"Parser import falhou: {e}. Parser não disponível.")
    VariableParser = None
    ParsedVariable = None


# ============================================================================
# TEXT PROCESSOR (MVP v2.0) - IMPORTS CORRETOS
# ============================================================================
from .text_processor import (
    TextProcessor,        # Legacy {{var}}
    SmartTextEngine,      # MVP natural (novo)
    DetectedVariable,     # FIX: Nome correto
    get_engine,           # Factory singleton
)


# ============================================================================
# PATTERNS UTILS (com fallback)
# ============================================================================
try:
    from .patterns import (
        find_variables,
        find_numbers,
        find_placeholders,
        has_greek_letters,
    )
except ImportError as e:
    logging.warning(f"Patterns import falhou: {e}. Usando fallback regex.")
    import re
    # Fallbacks simples
    find_variables = lambda t: re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', t)
    find_numbers = lambda t: [float(m) for m in re.findall(r'[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?', t)]
    find_placeholders = lambda t: re.findall(r'\{\{([a-zA-Z_][a-zA-Z0-9_]*)\}\}', t)
    has_greek_letters = lambda t: bool(re.search(r'[\u0370-\u03FF]', t))


# ============================================================================
# BUNDLE FACTORY (inovador: one-stop setup)
# ============================================================================
def get_recognition_bundle(nlp: bool = False, legacy: bool = False) -> Dict[str, Any]:
    """
    Bundle completo de ferramentas de reconhecimento.
    
    Args:
        nlp: Habilita NLP (não implementado na versão MVP, ignorado)
        legacy: Se True, usa TextProcessor (compatibilidade v1.0)
    
    Returns:
        Dict com: processor, parser, greek, patterns
    
    Example:
        >>> bundle = get_recognition_bundle()
        >>> engine = bundle['processor']
        >>> result = engine.process_text("M_k = 150 kN", {'M_k': 150})
    """
    # Escolhe processor (legacy ou novo)
    if legacy:
        processor = TextProcessor()
    else:
        processor = get_engine(auto_detect=True)  # FIX: Parâmetro correto
    
    return {
        'processor': processor,
        'parser': VariableParser() if VariableParser else None,
        'greek': GreekSymbols if GreekSymbols else None,
        'patterns': {
            'find_variables': find_variables,
            'find_placeholders': find_placeholders,
            'find_numbers': find_numbers,
            'has_greek_letters': has_greek_letters,
        },
    }


# ============================================================================
# EXPORTS
# ============================================================================
__all__ = [
    # Legacy (v1.0)
    "GreekSymbols",
    "VariableParser",
    "ParsedVariable",
    "TextProcessor",
    "GREEK_TO_ASCII",
    "ASCII_TO_GREEK",
    "find_variables",
    "find_numbers",
    "find_placeholders",
    "has_greek_letters",
    
    # MVP (v2.0) - FIX: Somente o que existe
    "SmartTextEngine",
    "DetectedVariable",  # FIX: Nome correto
    "get_engine",
    
    # Novo
    "get_recognition_bundle",
]

__version__ = '2.0.0'
