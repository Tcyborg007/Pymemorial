"""Testes abrangentes e robustos para cobrir 100% do equation.py"""

import pytest
import sympy as sp
from pymemorial.core import Equation, VariableFactory


# ============================================================================
# 1. TESTES BÁSICOS DE MÉTODOS
# ============================================================================

def test_substitute_method():
    """Testa método substitute() com substituição manual."""
    variables = {
        "x": VariableFactory.create("x", 10.0),
        "y": VariableFactory.create("y", 5.0),
    }
    
    eq = Equation("result = x + y", variables=variables)
    
    # Substituir x por 20
    expr = eq.substitute(x=20)
    
    # A expressão resultante é: 20 + y
    y_symbol = variables["y"].symbol
    result = float(expr.subs({y_symbol: 5}))
    assert result == 25.0


def test_substitute_invalid_variable():
    """Testa substituição de variável inexistente."""
    variables = {
        "x": VariableFactory.create("x", 10.0),
    }
    
    eq = Equation("result = x", variables=variables)
    
    # Tentar substituir variável que não existe
    with pytest.raises(KeyError) as exc_info:
        eq.substitute(z=100)
    
    assert "z" in str(exc_info.value)


def test_evaluate_without_values():
    """Testa evaluate() quando variável não tem valor."""
    variables = {
        "a": VariableFactory.create("a", 3.0),
        "b": VariableFactory.create("b", 2.0),
    }
    
    # Remover o valor de 'a'
    variables["a"].value = None
    
    eq = Equation("K = a + b", variables=variables)
    
    # Deve lançar erro informativo
    with pytest.raises(ValueError) as exc_info:
        eq.evaluate()
    
    assert "sem valor" in str(exc_info.value).lower()


def test_simplify_method_robust():
    """Testa método simplify() de forma robusta."""
    variables = {
        "x": VariableFactory.create("x", 3.0),
    }
    
    eq = Equation("result = x + x", variables=variables)
    simplified = eq.simplify()
    
    # Usar o símbolo correto da variável
    x_symbol = variables["x"].symbol
    test_value = 7.0
    
    # Avaliar ambas as expressões com o mesmo valor
    result_simplified = float(simplified.subs(x_symbol, test_value))
    result_expected = float((2 * x_symbol).subs(x_symbol, test_value))
    
    assert abs(result_simplified - result_expected) < 1e-10


def test_simplify_complex_expression():
    """Testa simplificação de expressão mais complexa."""
    variables = {
        "a": VariableFactory.create("a", 2.0),
    }
    
    # a^2 - a^2 deve simplificar para 0
    eq = Equation("result = a**2 - a**2", variables=variables)
    simplified = eq.simplify()
    
    # Deve ser 0 ou equivalente
    assert simplified == 0 or simplified == sp.Integer(0)


def test_latex_method():
    """Testa método latex()."""
    variables = {
        "a": VariableFactory.create("a", 2.0),
        "b": VariableFactory.create("b", 3.0),
    }
    
    eq = Equation("result = a**2 + b", variables=variables)
    latex = eq.latex()
    
    assert "a" in latex
    assert "b" in latex
    assert isinstance(latex, str)
    assert len(latex) > 0


def test_invalid_expression():
    """Testa criação de equação com expressão inválida."""
    variables = {"x": VariableFactory.create("x", 1.0)}
    
    with pytest.raises(ValueError) as exc_info:
        Equation("result = x + + +", variables=variables)
    
    assert "Erro ao converter" in str(exc_info.value)


# ============================================================================
# 2. TESTES DE GRANULARIDADE
# ============================================================================

def test_steps_minimal_granularity():
    """Testa granularidade minimal."""
    variables = {
        "x": VariableFactory.create("x", 5.0),
        "y": VariableFactory.create("y", 3.0),
    }
    
    eq = Equation("result = x * y", variables=variables)
    steps = eq.steps(granularity="minimal")
    
    assert len(steps) == 2
    assert steps[0]["operation"] == "symbolic"
    assert steps[1]["operation"] == "result"
    assert steps[1]["numeric"] == 15.0


def test_steps_normal_granularity():
    """Testa granularidade normal."""
    variables = {
        "a": VariableFactory.create("a", 4.0),
        "b": VariableFactory.create("b", 2.0),
    }
    
    eq = Equation("result = a + b", variables=variables)
    steps = eq.steps(granularity="normal")
    
    assert len(steps) >= 3
    assert steps[0]["operation"] == "symbolic"
    assert steps[1]["operation"] == "substitution"
    assert steps[-1]["numeric"] == 6.0


def test_steps_invalid_granularity():
    """Testa granularidade inválida."""
    variables = {"x": VariableFactory.create("x", 1.0)}
    eq = Equation("result = x", variables=variables)
    
    with pytest.raises(ValueError) as exc_info:
        eq.steps(granularity="super_detailed")
    
    assert "inválida" in str(exc_info.value).lower()


def test_steps_all_granularity():
    """Testa granularidade 'all'."""
    variables = {
        "x": VariableFactory.create("x", 2.0),
        "y": VariableFactory.create("y", 3.0),
    }
    
    eq = Equation("result = x**2 + y**2", variables=variables)
    steps = eq.steps(granularity="all")
    
    assert len(steps) >= 4


# ============================================================================
# 3. EXPRESSÕES MATEMÁTICAS COMPLEXAS
# ============================================================================

def test_trigonometric_functions():
    """Testa funções trigonométricas."""
    variables = {
        "theta": VariableFactory.create("theta", float(sp.pi/4)),
    }
    
    eq = Equation("result = sin(theta)**2 + cos(theta)**2", variables=variables)
    result = eq.evaluate()
    
    # sin^2 + cos^2 = 1
    assert abs(result - 1.0) < 0.01


def test_exponential_and_logarithm():
    """Testa exponenciais e logaritmos."""
    variables = {
        "x": VariableFactory.create("x", 2.0),
    }
    
    eq = Equation("result = exp(log(x))", variables=variables)
    result = eq.evaluate()
    
    # exp(log(x)) = x
    assert abs(result - 2.0) < 0.01


def test_square_root():
    """Testa raízes quadradas."""
    variables = {
        "a": VariableFactory.create("a", 16.0),
    }
    
    eq = Equation("result = sqrt(a)", variables=variables)
    result = eq.evaluate()
    
    assert result == 4.0


def test_constants_pi_e():
    """Testa constantes matemáticas π e e."""
    eq = Equation("result = pi + E", variables={})
    result = eq.evaluate()
    
    # π + e ≈ 5.859
    assert abs(result - 5.859) < 0.01


# ============================================================================
# 4. ESTRUTURAS ANINHADAS PROFUNDAS
# ============================================================================

def test_pow_add_pow_nested():
    """Testa Pow(Add(Pow(...))).."""
    variables = {
        "x": VariableFactory.create("x", 2.0),
        "y": VariableFactory.create("y", 2.0),
    }
    
    eq = Equation("result = (x**3 + y**3)**0.5", variables=variables)
    steps = eq.steps(granularity="detailed")
    
    # 2^3 + 2^3 = 8 + 8 = 16, sqrt(16) = 4
    assert steps[-1]["numeric"] == 4.0


def test_mul_pow_add_deeply_nested():
    """Testa Mul(Pow(Add(...)))."""
    variables = {
        "a": VariableFactory.create("a", 1.0),
        "b": VariableFactory.create("b", 2.0),
        "c": VariableFactory.create("c", 3.0),
    }
    
    eq = Equation("result = ((a**2 + b**2)**0.5) * c", variables=variables)
    steps = eq.steps(granularity="detailed")
    
    # sqrt(1 + 4) * 3 = sqrt(5) * 3 ≈ 6.708
    assert abs(steps[-1]["numeric"] - 6.708) < 0.01


def test_triple_nested_mul_pow():
    """Testa Mul(Pow(Mul(Pow(...))))."""
    variables = {
        "x": VariableFactory.create("x", 8.0),
        "y": VariableFactory.create("y", 2.0),
        "z": VariableFactory.create("z", 3.0),
    }
    
    eq = Equation("result = x / (y * z)", variables=variables)
    steps = eq.steps(granularity="detailed")
    
    assert abs(steps[-1]["numeric"] - 1.333) < 0.01


# ============================================================================
# 5. CASOS EXTREMOS
# ============================================================================

def test_division_by_zero_handling():
    """Testa divisão por zero."""
    variables = {
        "a": VariableFactory.create("a", 10.0),
        "b": VariableFactory.create("b", 0.0),
    }
    
    eq = Equation("result = a / b", variables=variables)
    
    # Deve lançar erro ou retornar infinito
    with pytest.raises(ValueError):
        eq.evaluate()


def test_very_large_numbers():
    """Testa números muito grandes."""
    variables = {
        "big": VariableFactory.create("big", 1e100),
    }
    
    eq = Equation("result = big * 2", variables=variables)
    result = eq.evaluate()
    
    assert result == 2e100


def test_very_small_numbers():
    """Testa números muito pequenos."""
    variables = {
        "tiny": VariableFactory.create("tiny", 1e-100),
    }
    
    eq = Equation("result = tiny * 1e50", variables=variables)
    result = eq.evaluate()
    
    assert result == 1e-50


def test_already_simplified_expression():
    """Testa expressão já simplificada."""
    variables = {
        "n": VariableFactory.create("n", 42.0),
    }
    
    eq = Equation("result = n", variables=variables)
    steps = eq.steps(granularity="detailed")
    
    # SEMPRE deve ter pelo menos: simbólica, substituição, resultado
    assert len(steps) >= 2
    
    # Primeiro passo é sempre simbólico
    assert steps[0]["operation"] == "symbolic"
    assert steps[0]["numeric"] is None
    
    # Último passo SEMPRE deve ser resultado com valor numérico
    assert steps[-1]["numeric"] == 42.0
    assert steps[-1]["description"] == "Resultado final"
    
    # Se tem apenas 2 passos, o segundo deve ser result
    # Se tem 3+ passos, o último deve ser result
    if len(steps) == 2:
        # Expressão tão simples que substituição = resultado
        assert steps[-1]["operation"] in ["substitution", "result"]
    else:
        # Expressão com passos intermediários
        assert steps[1]["operation"] == "substitution"
        assert steps[-1]["operation"] == "result"




def test_expression_without_variables():
    """Testa expressão com apenas constantes."""
    eq = Equation("result = 2 + 3 * 4", variables={})
    result = eq.evaluate()
    
    assert result == 14.0


def test_unused_variables():
    """Testa variáveis não usadas."""
    variables = {
        "x": VariableFactory.create("x", 5.0),
        "y": VariableFactory.create("y", 10.0),
        "z": VariableFactory.create("z", 15.0),
    }
    
    eq = Equation("result = x * 2", variables=variables)
    result = eq.evaluate()
    
    assert result == 10.0


# ============================================================================
# 6. TESTES DE ROBUSTEZ
# ============================================================================

def test_operation_description_unmapped():
    """Testa operação não mapeada."""
    variables = {"x": VariableFactory.create("x", 5.0)}
    eq = Equation("result = x", variables=variables)
    
    description = eq._operation_description("unknown_op")
    
    assert description == "Passo de cálculo"


def test_steps_without_units():
    """Testa steps com show_units=False."""
    variables = {
        "length": VariableFactory.create("length", 10.0, "m"),
        "width": VariableFactory.create("width", 5.0, "m"),
    }
    
    eq = Equation("area = length * width", variables=variables)
    steps = eq.steps(granularity="detailed", show_units=False)
    
    assert steps[-1]["numeric"] == 50.0


def test_max_steps_protection():
    """Testa proteção de max_steps."""
    variables = {
        "a": VariableFactory.create("a", 2.0),
        "b": VariableFactory.create("b", 3.0),
        "c": VariableFactory.create("c", 4.0),
    }
    
    eq = Equation("result = (a**2 + b**2 + c**2)**0.5", variables=variables)
    steps = eq.steps(granularity="detailed", max_steps=3)
    
    assert len(steps) <= 5


def test_equation_with_description():
    """Testa equação com descrição."""
    variables = {"x": VariableFactory.create("x", 10.0)}
    
    eq = Equation(
        "F = x * 2",
        variables=variables,
        description="Cálculo da força"
    )
    
    assert eq.description == "Cálculo da força"
    assert eq.evaluate() == 20.0


def test_equation_result_caching():
    """Testa cache do resultado."""
    variables = {"value": VariableFactory.create("value", 7.0)}
    
    eq = Equation("output = value * 3", variables=variables)
    
    assert eq.result is None
    
    result = eq.evaluate()
    assert eq.result == 21.0
    assert result == 21.0


def test_equation_repr():
    """Testa representação string da equação."""
    variables = {
        "x": VariableFactory.create("x", 5.0),
        "y": VariableFactory.create("y", 3.0),
    }
    
    eq = Equation("result = x + y", variables=variables)
    repr_str = repr(eq)
    
    assert "Equation" in repr_str
    assert "2 vars" in repr_str


# ============================================================================
# 7. TESTES ADICIONAIS
# ============================================================================

def test_negative_numbers():
    """Testa números negativos."""
    variables = {
        "neg": VariableFactory.create("neg", -5.0),
        "pos": VariableFactory.create("pos", 10.0),
    }
    
    eq = Equation("result = neg + pos", variables=variables)
    result = eq.evaluate()
    
    assert result == 5.0


def test_fractional_exponents():
    """Testa expoentes fracionários."""
    variables = {
        "base": VariableFactory.create("base", 27.0),
    }
    
    eq = Equation("result = base**(1/3)", variables=variables)
    result = eq.evaluate()
    
    assert abs(result - 3.0) < 0.01


def test_negative_exponents():
    """Testa expoentes negativos."""
    variables = {
        "num": VariableFactory.create("num", 2.0),
    }
    
    eq = Equation("result = num**(-2)", variables=variables)
    result = eq.evaluate()
    
    assert result == 0.25


def test_mixed_operations():
    """Testa mistura de operações."""
    variables = {
        "a": VariableFactory.create("a", 3.0),
        "b": VariableFactory.create("b", 4.0),
        "c": VariableFactory.create("c", 5.0),
    }
    
    eq = Equation("result = (a**2 + b**2)**0.5 / c", variables=variables)
    steps = eq.steps(granularity="detailed")
    
    assert steps[-1]["numeric"] == 1.0


def test_absolute_value():
    """Testa valor absoluto."""
    variables = {
        "neg_val": VariableFactory.create("neg_val", -10.0),
    }
    
    eq = Equation("result = Abs(neg_val)", variables=variables)
    result = eq.evaluate()
    
    assert result == 10.0


def test_modulo_operation():
    """Testa operação módulo."""
    variables = {
        "dividend": VariableFactory.create("dividend", 17.0),
        "divisor": VariableFactory.create("divisor", 5.0),
    }
    
    eq = Equation("result = dividend % divisor", variables=variables)
    result = eq.evaluate()
    
    assert result == 2.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
