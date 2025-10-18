"""
Blocos de conteúdo para memoriais de cálculo.
"""
from enum import Enum
from typing import Any, Dict, Optional
from dataclasses import dataclass


class ContentType(Enum):
    """Tipos de conteúdo suportados."""
    TEXT = "text"
    EQUATION = "equation"
    FIGURE = "figure"
    TABLE = "table"
    CODE = "code"


@dataclass
class ContentBlock:
    """
    Bloco de conteúdo genérico.
    
    Attributes:
        type: tipo do conteúdo
        content: conteúdo propriamente dito
        caption: legenda (para figuras/tabelas)
        label: rótulo para referência cruzada
    """
    type: ContentType
    content: Any
    caption: Optional[str] = None
    label: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Serializa para dicionário."""
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
        """Serializa o conteúdo de acordo com o tipo."""
        if self.type == ContentType.TEXT:
            return str(self.content)
        
        elif self.type == ContentType.EQUATION:
            # Se for objeto Equation, extrair LaTeX
            if hasattr(self.content, 'latex'):
                return self.content.latex()
            return str(self.content)
        
        elif self.type == ContentType.FIGURE:
            # Caminho da figura ou objeto
            if isinstance(self.content, dict):
                return self.content
            return {"path": str(self.content)}
        
        elif self.type == ContentType.TABLE:
            # Tabela como lista de listas ou dict
            return self.content
        
        elif self.type == ContentType.CODE:
            return str(self.content)
        
        return str(self.content)


def create_text_block(text: str) -> ContentBlock:
    """Helper para criar bloco de texto."""
    return ContentBlock(type=ContentType.TEXT, content=text)


def create_equation_block(equation: Any, label: Optional[str] = None) -> ContentBlock:
    """Helper para criar bloco de equação."""
    return ContentBlock(type=ContentType.EQUATION, content=equation, label=label)


def create_figure_block(path: str, caption: str, label: Optional[str] = None) -> ContentBlock:
    """Helper para criar bloco de figura."""
    return ContentBlock(
        type=ContentType.FIGURE,
        content={"path": path},
        caption=caption,
        label=label
    )


def create_table_block(data: list, caption: str, label: Optional[str] = None) -> ContentBlock:
    """Helper para criar bloco de tabela."""
    return ContentBlock(
        type=ContentType.TABLE,
        content=data,
        caption=caption,
        label=label
    )
