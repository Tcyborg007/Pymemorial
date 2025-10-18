"""Testes de equações."""
import sympy as sp
from pymemorial.core.equation import Equation
from pymemorial.core.variable import VariableFactory

def test_equation_creation():
    """Testa criação de equação."""
    x = sp.Symbol("x")
    expr = x**2 + 2*x + 1
    eq = Equation(expression=expr, description="Equação quadrática")
    assert eq.expression == expr

def test_equation_evaluate():
    """Testa avaliação numérica."""
    a = VariableFactory.create("a", value=3.0)
    b = VariableFactory.create("b", value=4.0)
    
    expr = a.symbol**2 + b.symbol**2
    eq = Equation(expression=expr, variables={"a": a, "b": b})
    
    result = eq.evaluate()
    assert result == 25.0  # 3² + 4² = 25

def test_equation_simplify():
    """Testa simplificação."""
    x = sp.Symbol("x")
    expr = (x + 1)**2 - (x**2 + 2*x + 1)
    eq = Equation(expression=expr)
    simplified = eq.simplify()
    assert simplified == 0

def test_equation_latex():
    """Testa geração LaTeX."""
    x = sp.Symbol("x")
    expr = x**2
    eq = Equation(expression=expr)
    latex = eq.latex()
    assert "x^{2}" in latex
def test_equation_substitute():
    """Testa substituição de variáveis."""
    a = VariableFactory.create("a", value=2.0)
    b = VariableFactory.create("b", value=3.0)
    
    expr = a.symbol + b.symbol
    eq = Equation(expression=expr, variables={"a": a, "b": b})
    
    # Testa substituição
    result_expr = eq.substitute(a=5.0)
    expected = 5.0 + b.symbol
    assert result_expr == expected

def test_equation_empty_variables():
    """Testa equação sem variáveis definidas."""
    x = sp.Symbol("x")
    eq = Equation(expression=x**2)
    
    # Com o novo substitute(), agora PODE substituir símbolos na expressão
    # mesmo que não estejam em variables
    result = eq.substitute(x=4.0)
    assert result == 16.0  # x=4 => x^2 = 16
