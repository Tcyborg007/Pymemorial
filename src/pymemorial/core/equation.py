"""Representação de equações simbólicas e numéricas."""
from typing import Dict, Optional
from dataclasses import dataclass, field
import sympy as sp
from .variable import Variable

@dataclass
class Equation:
    """
    Equação de memorial de cálculo.
    
    Attributes:
        expression: expressão SymPy
        variables: dicionário de variáveis
        result: resultado numérico (após avaliação)
        description: descrição da equação
    """
    expression: sp.Expr
    variables: Dict[str, Variable] = field(default_factory=dict)
    result: Optional[float] = None
    description: str = ""
    
    def substitute(self, **kwargs) -> sp.Expr:
        """
        Substitui variáveis na expressão.
        
        Args:
            **kwargs: pares nome=valor
        
        Returns:
            Expressão com substituições
        """
        subs_dict = {}
        for name, value in kwargs.items():
            if name in self.variables:
                var = self.variables[name]
                subs_dict[var.symbol] = value
        
        return self.expression.subs(subs_dict)
    
    def evaluate(self) -> float:
        """
        Avalia numericamente usando valores das variáveis.
        
        Returns:
            Resultado numérico
        """
        subs = {var.symbol: var.magnitude for var in self.variables.values() if var.value is not None}
        expr_sub = self.expression.subs(subs)
        self.result = float(expr_sub.evalf())
        return self.result
    
    def simplify(self) -> sp.Expr:
        """Retorna expressão simplificada."""
        return sp.simplify(self.expression)
    
    def latex(self) -> str:
        """Retorna representação LaTeX da expressão."""
        return sp.latex(self.expression)
