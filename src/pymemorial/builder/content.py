# src/pymemorial/builder/content.py
"""
Blocos de conteúdo aprimorados (v2.0: LaTeX + Serialization).

Genérico com auto-LaTeX em TEXT/EQUATION. Helpers fluent.
Compatível 100%; to_dict robusto.

Example:
    block = create_text_block("Resistência f_ck = 30 MPa")
    block_json = block.to_dict()
"""

from enum import Enum
from typing import Any, Dict, Optional
from dataclasses import dataclass, field
import logging

# MVP recognition tie-in (lazy) - FIX: Import correto
try:
    from ..recognition import get_engine
    RECOGNITION_AVAILABLE = True
except ImportError:
    RECOGNITION_AVAILABLE = False
    get_engine = None

# Pandas fallback for tables
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

_logger = logging.getLogger(__name__)


class ContentType(Enum):
    """Tipos de conteúdo."""
    TEXT = "text"
    EQUATION = "equation"
    FIGURE = "figure"
    TABLE = "table"
    CODE = "code"


@dataclass
class ContentBlock:
    """
    Bloco genérico de conteúdo (enhanced: LaTeX serialize).
    """
    type: ContentType
    content: Any
    caption: Optional[str] = None
    label: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Serializa (LaTeX auto se text/equation)."""
        data = {
            "type": self.type.value,
            "content": self._serialize_content(),
        }
        
        if self.caption:
            data["caption"] = self.caption
        if self.label:
            data["label"] = self.label
        
        return data
    
    def _serialize_content(self) -> Any:
        """Serialize type-specific (LaTeX tie-in safe)."""
        if self.type == ContentType.TEXT:
            if RECOGNITION_AVAILABLE and isinstance(self.content, str):
                try:
                    engine = get_engine(auto_detect=True)
                    # FIX: Usa process_text ao invés de process_natural_text
                    return engine.process_text(self.content, {})
                except:
                    pass
            return str(self.content)
        
        elif self.type == ContentType.EQUATION:
            # SymPy latex()
            if hasattr(self.content, 'latex'):
                return self.content.latex()
            
            # Recognition LaTeX
            if RECOGNITION_AVAILABLE and isinstance(self.content, str):
                try:
                    engine = get_engine()
                    return engine.to_latex(self.content)
                except:
                    pass
            
            return str(self.content)
        
        elif self.type == ContentType.FIGURE:
            if isinstance(self.content, dict):
                return self.content
            return {"path": str(self.content)}
        
        elif self.type == ContentType.TABLE:
            if PANDAS_AVAILABLE and hasattr(self.content, 'to_dict'):
                return self.content.to_dict('records')
            return self.content if isinstance(self.content, list) else str(self.content)
        
        elif self.type == ContentType.CODE:
            return str(self.content)
        
        return str(self.content)


# ============================================================================
# HELPERS (fluent + LaTeX/NLP)
# ============================================================================

def create_text_block(text: str, use_natural: bool = True) -> ContentBlock:
    """Text block (LaTeX auto-process if enabled)."""
    return ContentBlock(type=ContentType.TEXT, content=text)


def create_equation_block(
    equation: Any,
    label: Optional[str] = None,
    use_latex: bool = True
) -> ContentBlock:
    """Equation block (LaTeX auto)."""
    return ContentBlock(
        type=ContentType.EQUATION,
        content=equation,
        label=label
    )


def create_figure_block(
    path: str,
    caption: str,
    label: Optional[str] = None
) -> ContentBlock:
    """Figure block."""
    return ContentBlock(
        type=ContentType.FIGURE,
        content={"path": path},
        caption=caption,
        label=label
    )


def create_table_block(
    data: list,
    caption: str,
    label: Optional[str] = None
) -> ContentBlock:
    """Table block (Pandas fallback JSON)."""
    return ContentBlock(
        type=ContentType.TABLE,
        content=data,
        caption=caption,
        label=label
    )


__all__ = [
    'ContentType',
    'ContentBlock',
    'create_text_block',
    'create_equation_block',
    'create_figure_block',
    'create_table_block',
]
