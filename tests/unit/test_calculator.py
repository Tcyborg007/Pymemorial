"""Testes do motor de cálculo."""
import sympy as sp
from pymemorial.core.calculator import Calculator
from pymemorial.core.equation import Equation
from pymemorial.core.variable import VariableFactory

def test_calculator_add_equation():
    """Testa adição de equações."""
    calc = Calculator()
    eq = Equation(expression=sp.Symbol("x"))
    calc.add_equation(eq)
    assert len(calc.equations) == 1

def test_calculator_evaluate_all():
    """Testa avaliação em lote."""
    calc = Calculator()
    
    x = VariableFactory.create("x", value=10.0)
    eq1 = Equation(expression=x.symbol * 2, variables={"x": x})
    
    calc.add_equation(eq1)
    results = calc.evaluate_all()
    
    assert results[id(eq1)] == 20.0

def test_calculator_compile():
    """Testa compilação lambdify."""
    calc = Calculator()
    x = VariableFactory.create("x", value=5.0)
    eq = Equation(expression=x.symbol**2, variables={"x": x})
    
    func = calc.compile(eq)
    result = func(5.0)
    assert result == 25.0
def test_calculator_compile_cache():
    """Testa que compilação usa cache."""
    calc = Calculator()
    x = VariableFactory.create("x", value=3.0)
    eq = Equation(expression=x.symbol**2, variables={"x": x})
    
    # Primeira compilação
    func1 = calc.compile(eq)
    # Segunda compilação (deve usar cache)
    func2 = calc.compile(eq)
    
    # Deve retornar a mesma função (mesmo objeto)
    assert func1 is func2
