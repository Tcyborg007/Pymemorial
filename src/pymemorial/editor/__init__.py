# src/pymemorial/editor/__init__.py
"""
PyMemorial Editor Module v5.0 - Core Integrated
"""
from __future__ import annotations
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# --- CORE IMPORTS (Do sub-módulo) ---
from .smart_parser import ( SmartVariableParser, SmartSymbol, GREEK_LETTERS, KNOWN_UNITS )

# --- Tenta importar o editor principal ---
try:
    from .natural_engine import ( NaturalMemorialEditor, DocumentType, RenderMode ) # ✅ Apenas o que existe
    EDITOR_AVAILABLE = True
    logger.debug("Natural editor (v5.0+ Core Integrated) loaded successfully")
except ImportError as e:
    logger.error(f"Failed to load natural editor: {e}")
    EDITOR_AVAILABLE = False; NaturalMemorialEditor = None; DocumentType = None; RenderMode = None
except Exception as e:
     logger.error(f"Unexpected error loading natural editor: {e}")
     EDITOR_AVAILABLE = False; NaturalMemorialEditor = None; DocumentType = None; RenderMode = None

# --- CONVENIENCE FUNCTIONS ---
def create_editor(document_type: str = 'memorial') -> Optional[NaturalMemorialEditor]:
    if not EDITOR_AVAILABLE or NaturalMemorialEditor is None:
        raise ImportError("NaturalMemorialEditor (v5.0+) not available. Check Core install.")
    return NaturalMemorialEditor(document_type=document_type)

def quick_process(text: str, document_type: str = 'memorial', output_file: Optional[str] = None) -> str:
    editor = create_editor(document_type=document_type)
    if editor is None: return "[ERRO: Editor não criado]"
    result = editor.process(text)
    # ... (lógica de salvar arquivo) ...
    if output_file:
        try:
            with open(output_file, 'w', encoding='utf-8') as f: f.write(result)
            logger.info(f"Output saved to {output_file}")
        except IOError as e: logger.error(f"Failed to write output file {output_file}: {e}")
    return result

def get_version() -> str: return "5.0.0" # Versão da Interface do Editor

def validate_editor() -> Dict[str, Any]:
    core_ok, sympy_ok = False, False
    try: from pymemorial.core import SYMPY_AVAILABLE as CORE_SYMPY_OK; core_ok=True; sympy_ok=CORE_SYMPY_OK
    except ImportError: pass
    return {'editor_available': EDITOR_AVAILABLE, 'core_available': core_ok,
            'sympy_via_core': sympy_ok, 'version': get_version(),
            'ready': EDITOR_AVAILABLE and core_ok and sympy_ok}

# ============================================================================
# EXPORTS
# ============================================================================
__all__ = [
    'NaturalMemorialEditor', 'DocumentType', 'RenderMode', # Do natural_engine
    'SmartVariableParser', # Do smart_parser
    'create_editor', 'quick_process', 'get_version', 'validate_editor', # Funções daqui
    'EDITOR_AVAILABLE', # Flag
]
__version__ = "5.0.0"; __author__ = "PyMemorial Team"; __email__ = "contact@pymemorial.org"

# --- MODULE INITIALIZATION ---
if EDITOR_AVAILABLE: logger.info(f"PyMemorial Editor Interface v{__version__} (Core Integrated) loaded.")
else: logger.warning(f"PyMemorial Editor Interface v{__version__} failed to load. Check Core install/logs.")