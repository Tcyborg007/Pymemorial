# src/pymemorial/recognition/__init__.py
"""
Recognition Module - API Pública v3.0

Combina funcionalidades v2.0 (SmartTextEngine) + v3.0 (SmartTextProcessor).

Author: PyMemorial Team
Date: October 2025
Version: 3.0.0
"""

import logging
from typing import Dict, Any, Optional

_logger = logging.getLogger(__name__)

# ============================================================================
# GREEK SYMBOLS
# ============================================================================
try:
    from .greek import GreekSymbols, ASCII_TO_GREEK
    try:
        from .greek import GREEK_TO_ASCII
    except ImportError:
        GREEK_TO_ASCII = {v: k for k, v in ASCII_TO_GREEK.items()} if ASCII_TO_GREEK else {}
except ImportError as e:
    _logger.warning(f"Greek import failed: {e}")
    class GreekSymbols:
        @staticmethod
        def to_unicode(text: str) -> str:
            return text
        @staticmethod
        def to_latex(text: str) -> str:
            return text
    ASCII_TO_GREEK = {}
    GREEK_TO_ASCII = {}

# ============================================================================
# PARSER
# ============================================================================
try:
    from .parser import VariableParser, ParsedVariable
except ImportError as e:
    _logger.warning(f"Parser import failed: {e}")
    VariableParser = None
    ParsedVariable = None

# ============================================================================
# TEXT PROCESSOR v2.0 + v3.0
# ============================================================================
try:
    from .text_processor import (
        TextProcessor,
        SmartTextEngine,
        DetectedVariable,
        get_engine,
        SmartTextProcessor,
        VariableRegistry,
        EquationParser,
        LaTeXRenderer,
        ProcessingOptions,
        DocumentType,
        RenderMode,
        CitationStyle,
        VariableContext,
        EquationContext,
    )
    TEXT_PROCESSOR_V3_AVAILABLE = True
except ImportError as e:
    _logger.warning(f"Text processor v3.0 import failed: {e}")
    try:
        from .text_processor import (
            TextProcessor,
            SmartTextEngine,
            DetectedVariable,
            get_engine,
        )
        SmartTextProcessor = None
        VariableRegistry = None
        EquationParser = None
        LaTeXRenderer = None
        ProcessingOptions = None
        DocumentType = None
        RenderMode = None
        CitationStyle = None
        VariableContext = None
        EquationContext = None
        TEXT_PROCESSOR_V3_AVAILABLE = False
    except ImportError as e2:
        _logger.error(f"Critical: {e2}")
        raise

# ============================================================================
# PATTERNS
# ============================================================================
try:
    from .patterns import (
        # Patterns v2.0
        PLACEHOLDER,
        VARNAME,
        GREEKLETTER,
        NUMBER,
        # Patterns v3.0
        VALUEDISPLAYPATTERN,
        FORMULADISPLAYPATTERN,
        EQUATIONBLOCKPATTERN,
        # Functions
        find_variables,
        find_numbers,
        find_placeholders,
        has_greek_letters,
    )
    PATTERNS_V3_AVAILABLE = True
except ImportError as e:
    _logger.warning(f"Patterns import failed: {e}")
    import re
    find_variables = lambda t: re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', t)
    find_numbers = lambda t: [float(m) for m in re.findall(r'[-+]?\d+\.?\d*', t)]
    find_placeholders = lambda t: re.findall(r'\{\{([a-zA-Z_][a-zA-Z0-9_]*)\}\}', t)
    has_greek_letters = lambda t: bool(re.search(r'[\u0370-\u03FF]', t))
    PATTERNS_V3_AVAILABLE = False

# ============================================================================
# BUNDLE FACTORY
# ============================================================================
def get_recognition_bundle(nlp: bool = False, legacy: bool = False, version: str = '3.0') -> Dict[str, Any]:
    """Factory para criar bundle de reconhecimento."""
    if legacy:
        processor = TextProcessor()
    elif version == '3.0' and TEXT_PROCESSOR_V3_AVAILABLE and SmartTextProcessor:
        processor = SmartTextProcessor()
    else:
        processor = get_engine(auto_detect=True)
    
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
        'version': version,
        'v3_available': TEXT_PROCESSOR_V3_AVAILABLE,
    }

def create_smart_processor(document_type: Optional[str] = 'memorial', render_mode: Optional[str] = 'full') -> Optional['SmartTextProcessor']:
    """Factory para SmartTextProcessor v3.0."""
    if not TEXT_PROCESSOR_V3_AVAILABLE or not SmartTextProcessor:
        _logger.warning("SmartTextProcessor v3.0 not available")
        return None
    
    try:
        options = ProcessingOptions(
            document_type=DocumentType[document_type.upper()],
            render_mode=RenderMode[render_mode.upper()],
        )
        return SmartTextProcessor(options=options)
    except Exception as e:
        _logger.error(f"Failed to create SmartTextProcessor: {e}")
        return None

# ============================================================================
# EXPORTS (FIX COMPLETO - ESTAVA INCOMPLETO!)
# ============================================================================
__all__ = [
    # v1.0
    "TextProcessor",
    # v2.0
    "SmartTextEngine",
    "DetectedVariable",
    "get_engine",
    "get_recognition_bundle",
    # v3.0
    "SmartTextProcessor",
    "VariableRegistry",
    "EquationParser",
    "LaTeXRenderer",
    "ProcessingOptions",
    "DocumentType",
    "RenderMode",
    "CitationStyle",
    "VariableContext",
    "EquationContext",
    "create_smart_processor",
    # Utils
    "GreekSymbols",
    "find_variables",
    "find_numbers",
    "find_placeholders",
    "has_greek_letters",
    # Flags
    "TEXT_PROCESSOR_V3_AVAILABLE",
    "PATTERNS_V3_AVAILABLE",
]

__version__ = '3.0.0'
