"""Testes avançados e robustos para casos especiais"""

import pytest
import sympy as sp
from pymemorial.core import Equation, VariableFactory


# ============================================================================
# OPERAÇÕES COM MATRIZES
# ============================================================================

def test_matrix_determinant():
    """Testa cálculo de determinante de matriz."""
    M = sp.Matrix([[1, 2], [3, 4]])
    det_M = M.det()
    
    # det([[1,2],[3,4]]) = 1*4 - 2*3 = -2
    assert det_M == -2


def test_matrix_multiplication():
    """Testa multiplicação de matrizes."""
    A = sp.Matrix([[1, 2], [3, 4]])
    B = sp.Matrix([[5, 6], [7, 8]])
    C = A * B
    
    # Verificar resultado da multiplicação
    assert C[0, 0] == 19  # 1*5 + 2*7
    assert C[0, 1] == 22  # 1*6 + 2*8
    assert C[1, 0] == 43  # 3*5 + 4*7
    assert C[1, 1] == 50  # 3*6 + 4*8


def test_matrix_transpose():
    """Testa transposição de matriz."""
    A = sp.Matrix([[1, 2, 3], [4, 5, 6]])
    A_T = A.T
    
    assert A_T.shape == (3, 2)
    assert A_T[0, 0] == 1
    assert A_T[1, 0] == 2
    assert A_T[2, 1] == 6


def test_matrix_inverse():
    """Testa inversão de matriz."""
    A = sp.Matrix([[1, 2], [3, 4]])
    A_inv = A.inv()
    
    # A * A^(-1) = I
    I = A * A_inv
    assert I == sp.eye(2)


def test_matrix_eigenvalues():
    """Testa cálculo de autovalores."""
    A = sp.Matrix([[3, -1], [-1, 3]])
    eigenvals = A.eigenvals()
    
    # Autovalores são 2 e 4
    eigenval_list = list(eigenvals.keys())
    assert 2 in eigenval_list or sp.Integer(2) in eigenval_list
    assert 4 in eigenval_list or sp.Integer(4) in eigenval_list


def test_matrix_identity():
    """Testa matriz identidade."""
    I = sp.eye(3)
    
    assert I.det() == 1
    assert I[0, 0] == 1
    assert I[0, 1] == 0
    assert I[1, 1] == 1


# ============================================================================
# CÁLCULO DIFERENCIAL E INTEGRAL
# ============================================================================

def test_derivative_simple():
    """Testa derivada simples."""
    x = sp.Symbol('x')
    expr = x**2
    derivative = sp.diff(expr, x)
    
    # d/dx(x^2) = 2x
    assert derivative == 2*x


def test_derivative_evaluation():
    """Testa avaliação de derivada em ponto específico."""
    x = sp.Symbol('x')
    expr = x**3
    derivative = sp.diff(expr, x)
    
    # d/dx(x^3) = 3x^2, em x=2: 3*4 = 12
    result = derivative.subs(x, 2)
    assert result == 12


def test_integral_indefinite():
    """Testa integral indefinida."""
    x = sp.Symbol('x')
    
    # ∫x dx = x²/2
    integral = sp.integrate(x, x)
    expected = x**2 / 2
    
    # Comparar removendo constante de integração
    assert sp.simplify(integral - expected) == 0


def test_integral_definite():
    """Testa integral definida."""
    x = sp.Symbol('x')
    
    # ∫₀² x dx = [x²/2]₀² = 2
    definite = sp.integrate(x, (x, 0, 2))
    assert definite == 2


def test_partial_derivative():
    """Testa derivada parcial."""
    x, y = sp.symbols('x y')
    expr = x**2 + y**2
    
    # ∂/∂x(x²+y²) = 2x
    partial_x = sp.diff(expr, x)
    assert partial_x == 2*x
    
    # ∂/∂y(x²+y²) = 2y
    partial_y = sp.diff(expr, y)
    assert partial_y == 2*y


# ============================================================================
# FUNÇÕES ESPECIAIS
# ============================================================================

def test_piecewise_functions():
    """Testa funções por partes."""
    x = sp.Symbol('x')
    
    # f(x) = x se x > 0, senão -x
    pw = sp.Piecewise((x, x > 0), (-x, True))
    
    # Testar em diferentes valores
    assert pw.subs(x, 5) == 5
    assert pw.subs(x, -5) == 5
    assert pw.subs(x, 0) == 0


def test_factorial():
    """Testa função fatorial."""
    variables = {
        "n": VariableFactory.create("n", 5)
    }
    
    eq = Equation("result = factorial(n)", variables=variables)
    result = eq.evaluate()
    
    # 5! = 120
    assert result == 120


def test_binomial_coefficient():
    """Testa coeficiente binomial."""
    variables = {
        "n": VariableFactory.create("n", 5),
        "k": VariableFactory.create("k", 2),
    }
    
    # C(5,2) = 10
    eq = Equation("result = binomial(n, k)", variables=variables)
    assert eq.evaluate() == 10


def test_floor_ceiling():
    """Testa funções floor e ceiling."""
    variables = {
        "val": VariableFactory.create("val", 3.7),
    }
    
    # floor(3.7) = 3
    eq_floor = Equation("result = floor(val)", variables=variables)
    assert eq_floor.evaluate() == 3.0
    
    # ceiling(3.7) = 4
    eq_ceil = Equation("result = ceiling(val)", variables=variables)
    assert eq_ceil.evaluate() == 4.0


# ============================================================================
# SOMATÓRIOS E PRODUTÓRIOS
# ============================================================================

def test_summation():
    """Testa somatório."""
    i = sp.Symbol('i')
    
    # Sum(i, (i, 1, 5)) = 1+2+3+4+5 = 15
    summation = sp.Sum(i, (i, 1, 5))
    result = summation.doit()
    assert result == 15


def test_summation_formula():
    """Testa fórmula de somatório."""
    i, n = sp.symbols('i n')
    
    # Sum(i, (i, 1, n)) = n(n+1)/2
    summation = sp.Sum(i, (i, 1, n))
    formula = summation.doit()
    
    # Avaliar para n=10: 10*11/2 = 55
    result = formula.subs(n, 10)
    assert result == 55


def test_product():
    """Testa produtório."""
    i = sp.Symbol('i')
    
    # Product(i, (i, 1, 4)) = 1*2*3*4 = 24
    product = sp.Product(i, (i, 1, 4))
    result = product.doit()
    assert result == 24


# ============================================================================
# LIMITES
# ============================================================================

def test_limit_simple():
    """Testa limite simples."""
    x = sp.Symbol('x')
    
    # lim(x->0) sin(x)/x = 1
    limit = sp.limit(sp.sin(x)/x, x, 0)
    assert limit == 1


def test_limit_infinity():
    """Testa limite no infinito."""
    x = sp.Symbol('x')
    
    # lim(x->∞) 1/x = 0
    limit = sp.limit(1/x, x, sp.oo)
    assert limit == 0


def test_limit_lateral():
    """Testa limites laterais."""
    x = sp.Symbol('x')
    
    # lim(x->0⁺) 1/x = +∞
    limit_right = sp.limit(1/x, x, 0, '+')
    assert limit_right == sp.oo
    
    # lim(x->0⁻) 1/x = -∞
    limit_left = sp.limit(1/x, x, 0, '-')
    assert limit_left == -sp.oo


# ============================================================================
# RESOLUÇÃO DE EQUAÇÕES
# ============================================================================

def test_solve_linear():
    """Testa resolução de equação linear."""
    x = sp.Symbol('x')
    
    # 2x + 3 = 7 => x = 2
    solutions = sp.solve(2*x + 3 - 7, x)
    assert solutions == [2]


def test_solve_quadratic():
    """Testa resolução de equação quadrática."""
    x = sp.Symbol('x')
    
    # x^2 - 4 = 0 => x = ±2
    solutions = sp.solve(x**2 - 4, x)
    assert set(solutions) == {-2, 2}


def test_solve_system():
    """Testa resolução de sistema de equações."""
    x, y = sp.symbols('x y')
    
    # x + y = 3
    # x - y = 1
    # Solução: x=2, y=1
    solutions = sp.solve([x + y - 3, x - y - 1], [x, y])
    assert solutions[x] == 2
    assert solutions[y] == 1


# ============================================================================
# TESTES DE SUBSTITUIÇÃO AVANÇADA
# ============================================================================

def test_substitution_with_expression():
    """Testa substituição com expressão simbólica."""
    variables = {
        "x": VariableFactory.create("x", 3.0),
        "y": VariableFactory.create("y", 2.0),
    }
    
    eq = Equation("result = x + y", variables=variables)
    
    # Substituir x por y*2
    y_symbol = variables["y"].symbol
    expr = eq.substitute(x=y_symbol * 2)
    
    # Agora expr = y*2 + y = 3y
    result = float(expr.subs({y_symbol: 2}))
    assert result == 6.0


def test_substitution_multiple():
    """Testa substituição de múltiplas variáveis."""
    variables = {
        "a": VariableFactory.create("a", 1.0),
        "b": VariableFactory.create("b", 2.0),
        "c": VariableFactory.create("c", 3.0),
    }
    
    eq = Equation("result = a + b + c", variables=variables)
    
    # Substituir a e b
    expr = eq.substitute(a=10, b=20)
    
    # expr = 10 + 20 + c = 30 + c
    c_symbol = variables["c"].symbol
    result = float(expr.subs({c_symbol: 3}))
    assert result == 33.0


# ============================================================================
# NÚMEROS ESPECIAIS
# ============================================================================

def test_rational_numbers():
    """Testa números racionais."""
    variables = {
        "a": VariableFactory.create("a", float(sp.Rational(1, 3))),
        "b": VariableFactory.create("b", float(sp.Rational(2, 3))),
    }
    
    eq = Equation("result = a + b", variables=variables)
    result = eq.evaluate()
    
    # 1/3 + 2/3 = 1
    assert abs(result - 1.0) < 0.01


def test_infinity_operations():
    """Testa operações com infinito."""
    # Infinito
    inf = sp.oo
    
    assert inf + 1 == sp.oo
    assert inf * 2 == sp.oo
    assert 1 / inf == 0
    assert inf > 1000


def test_complex_numbers():
    """Testa números complexos no SymPy."""
    # i = sqrt(-1)
    i = sp.I
    
    # i^2 = -1
    assert i**2 == -1
    
    # (1+i)(1-i) = 1 - i^2 = 1 - (-1) = 2
    result = (1 + i) * (1 - i)
    
    # SymPy pode não simplificar automaticamente, então expandir
    result_simplified = sp.expand(result)
    assert result_simplified == 2



# ============================================================================
# CASOS DE BORDA E ROBUSTEZ
# ============================================================================

def test_equation_without_equals():
    """Testa equação sem sinal de igual."""
    variables = {"x": VariableFactory.create("x", 5.0)}
    
    eq = Equation("x * 2", variables=variables)
    result = eq.evaluate()
    
    assert result == 10.0


def test_equation_with_whitespace():
    """Testa equação com muito espaço em branco."""
    variables = {
        "a": VariableFactory.create("a", 3.0),
        "b": VariableFactory.create("b", 4.0),
    }
    
    eq = Equation("  result   =   a   +   b  ", variables=variables)
    result = eq.evaluate()
    
    assert result == 7.0


def test_equation_with_parentheses():
    """Testa equação com parênteses redundantes."""
    variables = {"x": VariableFactory.create("x", 2.0)}
    
    eq = Equation("result = (((x))) * 2", variables=variables)
    result = eq.evaluate()
    
    assert result == 4.0


def test_numeric_stability():
    """Testa estabilidade numérica."""
    variables = {
        "big": VariableFactory.create("big", 1e10),
        "small": VariableFactory.create("small", 1e-10),
    }
    
    eq = Equation("result = (big + small) - big", variables=variables)
    result = eq.evaluate()
    
    # Pode ter erro numérico, mas ordem de magnitude deve estar certa
    assert abs(result - 1e-10) < 1e-9


def test_greek_letters():
    """Testa símbolos com letras gregas."""
    variables = {
        "alpha": VariableFactory.create("alpha", 30.0),
        "beta": VariableFactory.create("beta", 60.0),
    }
    
    eq = Equation("result = alpha + beta", variables=variables)
    result = eq.evaluate()
    
    assert result == 90.0


def test_subscripted_variables():
    """Testa variáveis com subscritos."""
    variables = {
        "x_1": VariableFactory.create("x_1", 5.0),
        "x_2": VariableFactory.create("x_2", 10.0),
    }
    
    eq = Equation("result = x_1 + x_2", variables=variables)
    result = eq.evaluate()
    
    assert result == 15.0


def test_equation_string_representation():
    """Testa representação em string."""
    variables = {
        "x": VariableFactory.create("x", 5.0),
    }
    
    eq = Equation("F = x * 2", variables=variables)
    
    # Avaliar para ter resultado
    eq.evaluate()
    
    # Verificar repr
    repr_str = repr(eq)
    assert "Equation" in repr_str
    assert "result=" in repr_str


# ============================================================================
# TESTES DE DESEMPENHO E COMPLEXIDADE
# ============================================================================

def test_large_polynomial():
    """Testa polinômio de grau alto."""
    variables = {
        "x": VariableFactory.create("x", 2.0),
    }
    
    # x^5 + x^4 + x^3 + x^2 + x + 1
    eq = Equation("result = x**5 + x**4 + x**3 + x**2 + x + 1", variables=variables)
    result = eq.evaluate()
    
    # Para x=2: 32 + 16 + 8 + 4 + 2 + 1 = 63
    assert result == 63.0


def test_deeply_nested_expression():
    """Testa expressão profundamente aninhada."""
    variables = {
        "a": VariableFactory.create("a", 2.0),
    }
    
    # ((((a))))
    eq = Equation("result = ((((a))))", variables=variables)
    result = eq.evaluate()
    
    assert result == 2.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
