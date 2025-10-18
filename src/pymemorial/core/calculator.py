"""Motor de cálculo do PyMemorial."""
from typing import Dict, Callable, List
import sympy as sp
from .equation import Equation
from .variable import Variable

class Calculator:
    """
    Motor de cálculo com cache e rastreabilidade.
    """
    
    def __init__(self):
        self.equations: List[Equation] = []
        self._compiled_cache: Dict[int, Callable] = {}
    
    def add_equation(self, equation: Equation):
        """Adiciona equação ao calculador."""
        self.equations.append(equation)
    
    def compile(self, equation: Equation) -> Callable:
        """
        Compila equação para função Python rápida (lambdify).
        
        Returns:
            Função compilada
        """
        eq_id = id(equation)
        if eq_id in self._compiled_cache:
            return self._compiled_cache[eq_id]
        
        symbols = [var.symbol for var in equation.variables.values()]
        func = sp.lambdify(symbols, equation.expression, "numpy")
        self._compiled_cache[eq_id] = func
        return func
    
    def evaluate_all(self) -> Dict[int, float]:
        """
        Avalia todas as equações registradas.
        
        Returns:
            Dicionário id_equação -> resultado
        """
        results = {}
        for eq in self.equations:
            results[id(eq)] = eq.evaluate()
        return results
