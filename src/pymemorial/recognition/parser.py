"""
Parser de variáveis para extração automática de memoriais de cálculo.
"""
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import re
from .greek import GreekSymbols
from .patterns import (
    VAR_NAME,
    NUMBER,
    EQUATION_PATTERN,
    has_greek_letters,
)


@dataclass
class ParsedVariable:
    """Resultado do parsing de uma variável."""
    name: str
    value: Optional[float] = None
    unit: Optional[str] = None
    description: str = ""
    has_greek: bool = False


class VariableParser:
    """
    Parser de variáveis a partir de texto técnico.
    
    Examples:
        >>> parser = VariableParser()
        >>> result = parser.parse_line("fck = 30 MPa (resistência do concreto)")
        >>> result.name
        'fck'
        >>> result.value
        30.0
        >>> result.unit
        'MPa'
    """
    
    def __init__(self):
        # Padrão: nome = valor unidade (descrição)
        self.full_pattern = re.compile(
            r'([a-zA-Z_α-ωΑ-Ω][a-zA-Z0-9_α-ωΑ-Ω]*)\s*=\s*'
            r'([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)\s*'
            r'([a-zA-Z/²³°]+)?'
            r'(?:\s*\(([^)]+)\))?'
        )
    
    def parse_line(self, line: str) -> Optional[ParsedVariable]:
        """
        Extrai variável de uma linha de texto.
        
        Args:
            line: linha contendo definição de variável
        
        Returns:
            ParsedVariable ou None se não encontrar padrão
        """
        # Normalizar símbolos gregos para ASCII
        normalized = GreekSymbols.to_ascii(line)
        has_greek = has_greek_letters(line)
        
        match = self.full_pattern.search(normalized)
        if not match:
            return None
        
        name, value_str, unit, description = match.groups()
        
        return ParsedVariable(
            name=name.strip(),
            value=float(value_str) if value_str else None,
            unit=unit.strip() if unit else None,
            description=description.strip() if description else "",
            has_greek=has_greek,
        )
    
    def parse_text(self, text: str) -> List[ParsedVariable]:
        """
        Extrai todas as variáveis de um texto multi-linha.
        
        Args:
            text: texto completo
        
        Returns:
            Lista de variáveis encontradas
        """
        variables = []
        for line in text.split('\n'):
            parsed = self.parse_line(line)
            if parsed:
                variables.append(parsed)
        return variables
    
    def parse_equation(self, equation: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Extrai nome e expressão de uma equação.
        
        Args:
            equation: string no formato "var = expr"
        
        Returns:
            Tupla (nome_var, expressão) ou (None, None)
        
        Examples:
            >>> parser.parse_equation("M = P * L / 4")
            ('M', 'P * L / 4')
        """
        match = EQUATION_PATTERN.match(equation.strip())
        if match:
            return match.group(1), match.group(2).strip()
        return None, None
    
    def extract_variable_names(self, text: str) -> List[str]:
        """
        Extrai apenas os nomes de variáveis (sem valores).
        
        Args:
            text: texto para analisar
        
        Returns:
            Lista de nomes únicos
        """
        normalized = GreekSymbols.to_ascii(text)
        names = VAR_NAME.findall(normalized)
        # Filtrar palavras comuns que não são variáveis
        excluded = {'e', 'a', 'o', 'de', 'para', 'em', 'com', 'no', 'na'}
        return [n for n in set(names) if n not in excluded and len(n) > 1]
