# src/pymemorial/builder/section.py
"""
Representação de seções hierárquicas aprimorada (v2.0: Auto-Number).

Hierarquia com numbering inteligente (1.1.2), integração com recognition MVP.
Compatível 100%; to_dict LaTeX/norm-ready.

Example:
    section = Section("Flambagem")
    section.add_content(create_text_block("Análise conforme NBR 8800"))
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
import uuid
import logging

# Local imports
try:
    from .content import ContentBlock, ContentType, create_equation_block
except ImportError:
    # Stub
    class ContentType(Enum):
        TEXT = "text"
        EQUATION = "equation"
    ContentBlock = None
    create_equation_block = lambda eq, label: None

# MVP recognition tie-in (lazy) - FIX: Imports corretos
try:
    from ..recognition import get_engine, SmartTextEngine, DetectedVariable
    RECOGNITION_AVAILABLE = True
except ImportError:
    RECOGNITION_AVAILABLE = False
    get_engine = SmartTextEngine = DetectedVariable = None

_logger = logging.getLogger(__name__)


@dataclass
class Section:
    """
    Seção aprimorada com auto-numbering e integração recognition.
    """
    title: str
    level: int = 1
    numbered: bool = True
    content: List[ContentBlock] = field(default_factory=list)
    subsections: List['Section'] = field(default_factory=list)
    number: str = field(init=False)
    id: str = field(default_factory=lambda: str(uuid.uuid4()), init=False)
    parent: Optional['Section'] = None
    
    def __post_init__(self):
        self._auto_number()
        if RECOGNITION_AVAILABLE:
            self.process_title()
    
    def add_content(self, block: ContentBlock):
        """Adiciona bloco de conteúdo."""
        if block is None:
            return
        
        # FIX: Não tenta processar NLP (não existe process_natural_text)
        self.content.append(block)
        _logger.debug(f"Content added to {self.title}: {block.type if block else 'None'}")
    
    def add_subsection(self, section: 'Section'):
        """Adiciona subseção (re-number, set parent)."""
        section.level = self.level + 1
        section.parent = self
        section._auto_number()
        self.subsections.append(section)
        _logger.debug(f"Subsection added to {self.title}: {section.title}")
    
    def process_title(self, use_natural: bool = True):
        """Processa title com recognition MVP (LaTeX apenas)."""
        if not RECOGNITION_AVAILABLE:
            return
        
        try:
            engine = get_engine(auto_detect=True)
            original_title = self.title
            # FIX: Usa process_text ao invés de process_natural_text
            self.title = engine.process_text(self.title, {})
            
            if self.title != original_title:
                _logger.info(f"Title processed for {self.id}: {original_title} → {self.title}")
        except Exception as e:
            _logger.warning(f"Failed to process title: {e}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Serializa (enhanced: + latex_title, norm_hint)."""
        data = {
            "title": self.title,
            "level": self.level,
            "numbered": self.numbered,
            "number": self.number,
            "content": [block.to_dict() for block in self.content if block],
            "subsections": [sub.to_dict() for sub in self.subsections],
            "id": self.id,
        }
        
        # NEW: LaTeX title if recognition available
        if RECOGNITION_AVAILABLE:
            try:
                engine = get_engine()
                data["latex_title"] = engine.to_latex(self.title)
                data["norm_hint"] = self._infer_norm_from_title()
            except:
                pass
        
        return data
    
    def _infer_norm_from_title(self) -> str:
        """Infer norm from title (ex: 'NBR' → concrete)."""
        title_lower = self.title.lower()
        if 'nbr' in title_lower:
            return 'concrete'
        elif 'aisc' in title_lower:
            return 'steel'
        return 'unknown'
    
    def _auto_number(self):
        """Auto-number hierarchical (1.1.2 from parent)."""
        if self.numbered:
            if self.level == 1:
                # Global counter seria melhor, mas stub por ora
                self.number = "1"
            else:
                parent_num = self.parent.number if self.parent else "0"
                sibling_count = len([s for s in (self.parent.subsections if self.parent else [])]) + 1
                self.number = f"{parent_num}.{sibling_count}"
        else:
            self.number = ""
    
    def validate_hierarchy(self) -> bool:
        """Validate no cycles (DFS)."""
        visited = set()
        
        def dfs(node):
            if node.id in visited:
                return False
            visited.add(node.id)
            for sub in node.subsections:
                if not dfs(sub):
                    return False
            return True
        
        valid = dfs(self)
        if not valid:
            _logger.warning(f"Hierarchy cycle in {self.title}")
        return valid


__all__ = ["Section"]
