"""Testes robustos para Equation.steps()"""
import pytest
import sympy as sp

# Tentar importar do caminho 'src'
try:
    from pymemorial.core import Equation, VariableFactory
# Tentar importar do pacote instalado (fallback)
except ImportError:
    from pymemorial.core import Equation, VariableFactory


@pytest.fixture
def simple_variables():
    """Variáveis para testes simples."""
    return {
        "a": VariableFactory.create("a", 4.0),
        "b": VariableFactory.create("b", 2.0),
        "c": VariableFactory.create("c", 2.0),
        "d": VariableFactory.create("d", 2.0),
    }


@pytest.fixture
def structural_variables():
    """Variáveis para teste estrutural."""
    return {
        "q": VariableFactory.create("q", 15.0, "kN/m"),
        "L": VariableFactory.create("L", 6.0, "m"),
    }


def test_steps_minimal_granularity(simple_variables):
    """Testa granularidade mínima (apenas expressão e resultado)."""
    eq = Equation("K = a*b/c + d", variables=simple_variables)
    steps = eq.steps(granularity="minimal")
    
    assert len(steps) == 2
    assert steps[0]['operation'] == "symbolic"
    assert steps[1]['operation'] == "result"
    assert steps[1]['numeric'] == 6.0


def test_steps_normal_granularity(simple_variables):
    """Testa granularidade normal (substituição + resultado)."""
    eq = Equation("K = a*b/c + d", variables=simple_variables)
    steps = eq.steps(granularity="normal")
    
    assert len(steps) >= 2
    assert steps[0]['operation'] == "symbolic"
    assert steps[1]['operation'] == "substitution"
    assert steps[-1]['numeric'] == 6.0


def test_steps_detailed_granularity(simple_variables):
    """Testa granularidade detalhada (cada operação separada)."""
    eq = Equation("K = a*b/c + d", variables=simple_variables)
    steps = eq.steps(granularity="detailed")
    
    # Deve ter pelo menos:
    # 1. Simbólica
    # 2. Substituição
    # 3. Multiplicação (a*b) -> 8.0
    # 4. Multiplicação (8.0/c) -> 4.0
    # 5. Adição (+d) -> 6.0
    
    # A minha análise manual do seu código CORRIGIDO mostra 5 passos:
    # 1. symbolic
    # 2. substitution (2.0 + 4.0*2.0/2.0)
    # 3. multiply (2.0 + 8.0/2.0)
    # 4. multiply (2.0 + 4.0)
    # 5. add (6.0)
    assert len(steps) >= 5
    
    # Verificar sequência de operações
    operations = [s['operation'] for s in steps]
    assert "symbolic" in operations
    assert "substitution" in operations
    assert "multiply" in operations
    assert "add" in operations
    
    # Resultado final correto
    assert steps[-1]['numeric'] == 6.0


def test_steps_power_operation(simple_variables):
    """Testa detecção e cálculo de potências."""
    eq = Equation("result = a**2", variables=simple_variables)
    steps = eq.steps(granularity="detailed")
    
    # Encontrar passo de potência
    power_steps = [s for s in steps if s['operation'] == "power"]
    assert len(power_steps) >= 1
    
    # Verificar resultado
    assert steps[-1]['numeric'] == 16.0


def test_steps_structural_moment(structural_variables):
    """Testa equação estrutural realista."""
    eq = Equation("M = q * L**2 / 8", variables=structural_variables)
    steps = eq.steps(granularity="detailed")
    
    # Verificar número de passos
    assert len(steps) >= 5
    
    # Verificar operações
    operations = [s['operation'] for s in steps]
    assert "power" in operations
    assert "multiply" in operations
    
    # Resultado (67.5 kN·m)
    assert abs(steps[-1]['numeric'] - 67.5) < 0.01


def test_steps_complex_expression():
    """Testa expressão complexa com múltiplas operações."""
    variables = {
        "x": VariableFactory.create("x", 3.0),
        "y": VariableFactory.create("y", 4.0),
        "z": VariableFactory.create("z", 2.0),
    }
    
    eq = Equation("result = (x**2 + y**2)**0.5 * z", variables=variables)
    steps = eq.steps(granularity="detailed")
    
    # Deve ter vários passos
    assert len(steps) >= 6
    
    # Resultado = sqrt(9 + 16) * 2 = 5 * 2 = 10
    assert abs(steps[-1]['numeric'] - 10.0) < 0.01


def test_steps_step_numbers():
    """Testa que números de passo são sequenciais."""
    variables = {
        "a": VariableFactory.create("a", 5.0),
        "b": VariableFactory.create("b", 3.0),
    }
    
    eq = Equation("c = a + b", variables=variables)
    steps = eq.steps(granularity="detailed")
    
    # Verificar sequência
    for i, step in enumerate(steps, 1):
        assert step['step_number'] == i


def test_steps_max_steps_protection():
    """Testa proteção contra loops infinitos."""
    variables = {
        "x": VariableFactory.create("x", 1.0),
    }
    
    # Expressão que poderia causar loop (mas não deveria)
    eq = Equation("y = x", variables=variables)
    steps = eq.steps(granularity="detailed", max_steps=5)
    
    # Deve parar antes de max_steps
    assert len(steps) <= 5


def test_steps_division_by_fraction():
    """Testa divisão com frações."""
    variables = {
        "a": VariableFactory.create("a", 10.0),
        "b": VariableFactory.create("b", 5.0),
        "c": VariableFactory.create("c", 2.0),
    }
    
    eq = Equation("result = a / (b / c)", variables=variables)
    steps = eq.steps(granularity="detailed")
    
    # Resultado = 10 / (5/2) = 10 / 2.5 = 4
    assert abs(steps[-1]['numeric'] - 4.0) < 0.01


def test_steps_all_granularity():
    """Testa granularidade máxima."""
    variables = {
        "a": VariableFactory.create("a", 2.0),
        "b": VariableFactory.create("b", 3.0),
    }
    
    eq = Equation("c = a * b", variables=variables)
    steps_all = eq.steps(granularity="all")
    steps_detailed = eq.steps(granularity="detailed")
    
    # "all" deve ter pelo menos tantos passos quanto "detailed"
    assert len(steps_all) >= len(steps_detailed)


def test_steps_description_in_portuguese():
    """Testa que descrições estão em português."""
    variables = {
        "a": VariableFactory.create("a", 4.0),
        "b": VariableFactory.create("b", 2.0),
    }
    
    eq = Equation("c = a + b", variables=variables)
    steps = eq.steps(granularity="detailed")
    
    # Verificar algumas descrições em português
    descriptions = [s['description'] for s in steps]
    assert any("Substituição" in d for d in descriptions)
    assert any("final" in d.lower() for d in descriptions)


def test_steps_latex_formatting():
    """Testa que todos os passos têm LaTeX válido."""
    variables = {
        "x": VariableFactory.create("x", 5.0),
        "y": VariableFactory.create("y", 3.0),
    }
    
    eq = Equation("z = x * y", variables=variables)
    steps = eq.steps(granularity="detailed")
    
    # Todos os passos devem ter LaTeX
    for step in steps:
        assert 'latex' in step
        assert isinstance(step['latex'], str)
        assert len(step['latex']) > 0


def test_steps_addition_subtraction():
    """Testa operações de adição e subtração."""
    variables = {
        "a": VariableFactory.create("a", 10.0),
        "b": VariableFactory.create("b", 5.0),
        "c": VariableFactory.create("c", 3.0),
    }
    
    eq = Equation("result = a - b + c", variables=variables)
    steps = eq.steps(granularity="detailed")
    
    # Verificar operação de adição/subtração
    operations = [s['operation'] for s in steps]
    assert "add" in operations
    
    # Resultado = 10 - 5 + 3 = 8
    assert abs(steps[-1]['numeric'] - 8.0) < 0.01