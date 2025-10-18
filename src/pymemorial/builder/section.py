"""
Representação de seções hierárquicas do memorial.
"""
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field

from .content import ContentBlock


@dataclass
class Section:
    """
    Seção do memorial com suporte a hierarquia.
    
    Attributes:
        title: título da seção
        level: nível hierárquico (1, 2, 3...)
        numbered: se deve ser numerada
        content: blocos de conteúdo
        subsections: subseções
    """
    title: str
    level: int = 1
    numbered: bool = True
    content: List[ContentBlock] = field(default_factory=list)
    subsections: List['Section'] = field(default_factory=list)
    
    def add_content(self, block: ContentBlock):
        """Adiciona bloco de conteúdo à seção."""
        self.content.append(block)
    
    def add_subsection(self, section: 'Section'):
        """Adiciona subseção."""
        self.subsections.append(section)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serializa para dicionário."""
        return {
            "title": self.title,
            "level": self.level,
            "numbered": self.numbered,
            "content": [block.to_dict() for block in self.content],
            "subsections": [sub.to_dict() for sub in self.subsections],
        }
