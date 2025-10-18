"""Representação de variáveis de cálculo."""
from typing import Optional, Any
from dataclasses import dataclass
import sympy as sp
from .units import parse_quantity, Quantity

@dataclass
class Variable:
    """
    Variável de memorial de cálculo.
    
    Attributes:
        name: nome da variável (ex: "fck")
        symbol: símbolo SymPy
        value: valor numérico com unidade
        description: descrição textual
    """
    name: str
    symbol: sp.Symbol
    value: Optional[Quantity] = None
    description: str = ""
    
    def __post_init__(self):
        """Valida após inicialização."""
        if not isinstance(self.symbol, sp.Symbol):
            raise TypeError("symbol deve ser sympy.Symbol")
    
    @property
    def magnitude(self) -> float:
        """Retorna magnitude numérica (sem unidade)."""
        if self.value is None:
            raise ValueError(f"Variable {self.name} não tem valor atribuído")
        return float(self.value.magnitude)
    
    def __repr__(self):
        unit_str = f" [{self.value.units}]" if self.value else ""
        return f"Variable({self.name} = {self.value}{unit_str})"

class VariableFactory:
    """Factory para criação padronizada de variáveis."""
    
    @staticmethod
    def create(name: str, value: Any = None, unit: str = None, description: str = "") -> Variable:
        """
        Cria variável com parsing automático.
        
        Args:
            name: nome da variável
            value: valor numérico ou string
            unit: unidade (se value não contém)
            description: descrição
        
        Returns:
            Variable
        """
        symbol = sp.Symbol(name, real=True)
        qty = parse_quantity(value, unit) if value is not None else None
        return Variable(name=name, symbol=symbol, value=qty, description=description)
