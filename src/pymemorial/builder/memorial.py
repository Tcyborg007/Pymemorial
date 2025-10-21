"""
Builder principal para construção de memoriais de cálculo.
"""
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime

from ..core import Variable, VariableFactory, Equation, Calculator
from ..recognition import VariableParser, TextProcessor
from .section import Section
from .content import ContentBlock, ContentType


@dataclass
class MemorialMetadata:
    """Metadados do memorial."""
    title: str
    author: str = ""
    date: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    project: str = ""
    revision: str = "1.0"
    norm: str = ""


class MemorialBuilder:
    """
    Builder fluente para memoriais de cálculo estruturais.
    
    Examples:
        >>> memorial = MemorialBuilder("Viga Biapoiada")
        >>> memorial.add_variable("L", value=6.0, unit="m", description="Vão")
        >>> memorial.add_section("Geometria", level=1)
        >>> memorial.add_text("O vão da viga é {{L}}.")
        >>> data = memorial.build()
    """
    
    def __init__(self, title: str, author: str = ""):
        """
        Args:
            title: título do memorial
            author: autor do memorial
        """
        self.metadata = MemorialMetadata(title=title, author=author)
        self.variables: Dict[str, Variable] = {}
        self.equations: List[Equation] = []
        self.sections: List[Section] = []
        self.calculator = Calculator()
        self.parser = VariableParser()
        self.processor = TextProcessor()
        
        # Controle de contexto (seção atual)
        self._current_section: Optional[Section] = None
    
    @property
    def current_section(self) -> Section:
        """Retorna seção atual."""
        return self._current_section
   
    def add_variable(
        self,
        name: str,
        value: Any = None,
        unit: Optional[str] = None,
        description: str = ""
    ) -> 'MemorialBuilder':
        """
        Adiciona variável ao memorial.
        
        Args:
            name: nome da variável
            value: valor numérico
            unit: unidade física
            description: descrição
        
        Returns:
            self (para encadeamento)
        """
        var = VariableFactory.create(name, value, unit, description)
        self.variables[name] = var
        return self
    
    def add_equation(
        self,
        expression: str,
        description: str = ""
    ) -> 'MemorialBuilder':
        """
        Adiciona equação ao memorial.
        
        Args:
            expression: expressão no formato "var = expr"
            description: descrição da equação
        
        Returns:
            self (para encadeamento)
        """
        import sympy as sp
        
        # Parsear equação
        var_name, expr_str = self.parser.parse_equation(expression)
        if not var_name or not expr_str:
            raise ValueError(f"Equação inválida: {expression}")
        
        # Converter string para expressão SymPy
        # Substituir nomes de variáveis por símbolos
        symbols = {name: var.symbol for name, var in self.variables.items()}
        expr = sp.sympify(expr_str, locals=symbols)
        
        # Criar equação
        eq = Equation(
            expression=expr,
            variables=self.variables,
            description=description
        )
        self.equations.append(eq)
        self.calculator.add_equation(eq)
        
        return self
    
    def add_section(
        self,
        title: str,
        level: int = 1,
        numbered: bool = True
    ) -> 'MemorialBuilder':
        """
        Adiciona nova seção ao memorial.
        
        Args:
            title: título da seção
            level: nível hierárquico (1, 2, 3...)
            numbered: se True, numera automaticamente
        
        Returns:
            self (para encadeamento)
        """
        section = Section(title=title, level=level, numbered=numbered)
        
        # Se houver seção atual e o nível for maior, adiciona como subseção
        if self._current_section and level > self._current_section.level:
            self._current_section.add_subsection(section)
        else:
            self.sections.append(section)
        
        self._current_section = section
        return self
    
    def add_text(self, text: str) -> 'MemorialBuilder':
        """
        Adiciona texto à seção atual.
        Suporta templates com {{var}}.
        
        Args:
            text: texto com placeholders opcionais
        
        Returns:
            self (para encadeamento)
        """
        if not self._current_section:
            raise ValueError("Nenhuma seção ativa. Use add_section() primeiro.")
        
        # Processar template
        context = {name: str(var.value) for name, var in self.variables.items() if var.value}
        processed_text = self.processor.render(text, context)
        
        block = ContentBlock(type=ContentType.TEXT, content=processed_text)
        self._current_section.add_content(block)
        return self
    
    def add_equation_block(self, equation_id: int) -> 'MemorialBuilder':
        """
        Adiciona bloco de equação renderizada.
        
        Args:
            equation_id: índice da equação em self.equations
        
        Returns:
            self (para encadeamento)
        """
        if not self._current_section:
            raise ValueError("Nenhuma seção ativa.")
        
        if equation_id >= len(self.equations):
            raise IndexError(f"Equação {equation_id} não existe.")
        
        eq = self.equations[equation_id]
        block = ContentBlock(type=ContentType.EQUATION, content=eq)
        self._current_section.add_content(block)
        return self
    
    def compute(self) -> Dict[int, float]:
        """
        Calcula todas as equações.
        
        Returns:
            Dicionário id_equação -> resultado
        """
        return self.calculator.evaluate_all()
    
    def build(self) -> Dict[str, Any]:
        """
        Constrói representação final do memorial.
        
        Returns:
            Dicionário com estrutura completa
        """
        return {
            "metadata": {
                "title": self.metadata.title,
                "author": self.metadata.author,
                "date": self.metadata.date,
                "project": self.metadata.project,
                "revision": self.metadata.revision,
                "norm": self.metadata.norm,
            },
            "variables": {
                name: {
                    "value": var.value.magnitude if var.value else None,
                    "unit": str(var.value.units) if var.value else None,
                    "description": var.description,
                }
                for name, var in self.variables.items()
            },
            "equations": [
                {
                    "expression": eq.latex(),
                    "result": eq.result,
                    "description": eq.description,
                }
                for eq in self.equations
            ],
            "sections": [section.to_dict() for section in self.sections],
        }
