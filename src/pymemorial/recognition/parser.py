# src/pymemorial/recognition/parser.py
"""
Parser de variáveis e expressões (v2.0: MVP Lite).

Parse basic de expressões (M_k = 150 kN) sem SymPy. Tie-in com recognition MVP.
Compat 100%; lazy init.

Example:
    parser = VariableParser()
    parsed = parser.parse("M_k = 150 kN")
    # → ParsedVariable(name='M_k', value=150, unit='kN')
"""

from typing import Optional, Any, Dict
from dataclasses import dataclass
import re
import logging

# MVP recognition tie-in (lazy) - FIX: Parâmetro correto
try:
    from .text_processor import get_engine
    RECOGNITION_AVAILABLE = True
except ImportError:
    RECOGNITION_AVAILABLE = False
    get_engine = None

_logger = logging.getLogger(__name__)


@dataclass
class ParsedVariable:
    """Variável parseada (básico)."""
    name: str
    value: Optional[Any] = None
    unit: Optional[str] = None
    description: str = ""


class VariableParser:
    """Parser de variáveis (MVP: regex básico sem SymPy)."""
    
    def __init__(self):
        """Inicializa parser (lazy tie-in com recognition)."""
        # FIX: Parâmetro correto é auto_detect, não nlp
        if RECOGNITION_AVAILABLE:
            try:
                self.engine = get_engine(auto_detect=True)  # FIX: auto_detect
            except Exception as e:
                _logger.warning(f"Recognition engine falhou: {e}")
                self.engine = None
        else:
            self.engine = None
        
        # Pattern básico: var = valor unidade
        self.pattern = re.compile(
            r'([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*([0-9.]+)\s*([a-zA-Z./]+)?'
        )
    
    def parse(self, text: str) -> Optional[ParsedVariable]:
        """
        Parse expressão simples.
        
        Args:
            text: Expressão (ex: "M_k = 150 kN")
        
        Returns:
            ParsedVariable ou None se inválido
        
        Example:
            >>> parser.parse("M_k = 150 kN")
            ParsedVariable(name='M_k', value=150.0, unit='kN')
        """
        match = self.pattern.search(text)
        if not match:
            _logger.warning(f"Failed to parse: {text}")
            return None
        
        name = match.group(1)
        value_str = match.group(2)
        unit = match.group(3) or ""
        
        # Converte valor
        try:
            value = float(value_str)
        except ValueError:
            value = None
        
        return ParsedVariable(
            name=name,
            value=value,
            unit=unit.strip()
        )
    
    def parse_multiple(self, text: str) -> list[ParsedVariable]:
        """Parse múltiplas variáveis."""
        results = []
        for match in self.pattern.finditer(text):
            name = match.group(1)
            value_str = match.group(2)
            unit = match.group(3) or ""
            
            try:
                value = float(value_str)
            except ValueError:
                value = None
            
            results.append(ParsedVariable(
                name=name,
                value=value,
                unit=unit.strip()
            ))
        
        return results


# Helper standalone
def parse_variable(text: str) -> Optional[ParsedVariable]:
    """Helper standalone para parse rápido."""
    parser = VariableParser()
    return parser.parse(text)


def parse_expression(text: str) -> Dict[str, Any]:
    """Parse expressão (stub: retorna texto por ora)."""
    return {"raw": text, "parsed": False}


__all__ = [
    'VariableParser',
    'ParsedVariable',
    'parse_variable',
    'parse_expression',
]
