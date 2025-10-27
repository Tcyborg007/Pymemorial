"""
test_calculator_smoke.py - VERSÃO FINAL CORRIGIDA
"""

import pytest
import numpy as np
# ================== CORREÇÃO AQUI ==================
from pymemorial.core.calculator import (
    Calculator,
    CalculationResult,
    CalculatorError,
    EvaluationError  # <--- Adicione esta linha
)
# ===================================================
from pymemorial.core.variable import Variable
from pymemorial.core.equation import Equation
from pymemorial.core.config import get_config, reset_config

try:
    import scipy
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


@pytest.fixture(autouse=True)
def reset_config_after_test():
    yield
    reset_config()


@pytest.fixture
def calc():
    return Calculator()


def test_smoke_simple_beam_analysis(calc):
    """SMOKE TEST: Análise completa de viga simples."""
    calc.add_variable("L", 6.0, unit="m")
    calc.add_variable("q", 15.0, unit="kN/m")
    calc.add_variable("fck", 25.0, unit="MPa")
    calc.add_variable("gamma_f", 1.4)
    
    Mk = calc.compute("q * L**2 / 8")
    calc.add_variable("Mk", Mk.value, unit="kN.m")
    
    Md = calc.compute("Mk * gamma_f")
    calc.add_variable("Md", Md.value, unit="kN.m")
    
    assert abs(Mk.value - 67.5) < 0.1
    assert abs(Md.value - 94.5) < 0.1


def test_smoke_parametric_analysis_vectorized(calc):
    """SMOKE TEST: Análise paramétrica vetorizada."""
    L_values = np.array([4, 5, 6, 7, 8])
    q_value = 15.0
    
    calc.add_variable("L", L_values)
    calc.add_variable("q", q_value)
    
    Mk = calc.compute("q * L**2 / 8", vectorized=True)
    
    expected = np.array([30, 46.875, 67.5, 91.875, 120])
    assert isinstance(Mk.value, np.ndarray)
    assert np.allclose(Mk.value, expected)


@pytest.mark.skipif(not SCIPY_AVAILABLE, reason="SciPy não disponível")
def test_smoke_monte_carlo_reliability_analysis(calc):
    """SMOKE TEST: Análise de confiabilidade com Monte Carlo."""
    variables = {
        "q": {"mean": 15, "std": 2, "dist": "normal"},
        "L": {"mean": 6, "std": 0.1, "dist": "normal"}
    }
    
    result = calc.monte_carlo(
        expression="q * L**2 / 8",
        variables=variables,
        n_samples=5000,
        confidence_levels=[0.05, 0.95],
        seed=42
    )
    
    assert "mean" in result.metadata
    assert abs(result.metadata["mean"] - 67.5) < 3.0
    assert result.metadata["std"] > 0


@pytest.mark.skipif(not SCIPY_AVAILABLE, reason="SciPy não disponível")
def test_smoke_solve_simple_equation(calc):
    """SMOKE TEST: Resolver equação simples."""
    result = calc.solve_equation_numerical(
        "x**2 - 4",
        "x",
        initial_guess=1.0
    )
    
    assert abs(result.value - 2.0) < 0.01
    assert result.metadata["converged"] is True


def test_smoke_equation_calculator_workflow(calc):
    """SMOKE TEST: Workflow completo com Equation objects."""
    eq1 = Equation(
        expression="q * L**2 / 8",
        name="Mk"
    )
    
    calc.add_variable("q", 15)
    calc.add_variable("L", 6)
    
    result = calc.compute(eq1)
    
    assert isinstance(result, CalculationResult)
    assert abs(result.value - 67.5) < 0.1


@pytest.mark.skipif(not SCIPY_AVAILABLE, reason="SciPy não disponível")
def test_smoke_integration_optimization_workflow(calc):
    """SMOKE TEST: Integração numérica + Otimização."""
    integral = calc.integrate_numerical("x**2", "x", 0, 1)
    assert abs(integral.value - 1/3) < 0.001
    
    opt_result = calc.optimize_function(
        "x**2 - 4*x + 4",
        "x",
        initial_guess=0,
        bounds=(-10, 10)
    )
    
    assert abs(opt_result.value - 2.0) < 0.01


def test_smoke_performance_large_batch(calc):
    """SMOKE TEST: Performance em batch processing."""
    calc.add_variable("x", 10)
    
    expressions = [f"x**{i}" for i in range(1, 11)]
    results = calc.batch_compute(expressions)
    
    assert len(results) == 10
    assert results[0].value == 10
    assert results[9].value == 10**10


def test_smoke_robust_error_handling(calc):
    """SMOKE TEST: Tratamento robusto de erros."""
    # Teste simples: variável não definida
    
    # ================== CORREÇÃO AQUI ==================
    # Adicionar EvaluationError à lista de exceções esperadas
    with pytest.raises((CalculatorError, EvaluationError, NameError, KeyError)):
    # ===================================================
        calc.compute("undefined_variable + 10")


@pytest.mark.skipif(not SCIPY_AVAILABLE, reason="SciPy não disponível")
def test_smoke_complete_structural_design(calc):
    """SMOKE TEST: Design estrutural completo end-to-end."""
    calc.add_variable("L", 6.0, unit="m")
    calc.add_variable("q", 15.0, unit="kN/m")
    calc.add_variable("fck", 25.0, unit="MPa")
    calc.add_variable("fyk", 500.0, unit="MPa")
    
    Mk = calc.compute("q * L**2 / 8")
    calc.add_variable("Mk", Mk.value)
    calc.add_variable("gamma_f", 1.4)
    calc.add_variable("gamma_c", 1.4)
    calc.add_variable("gamma_s", 1.15)
    
    Md = calc.compute("Mk * gamma_f")
    fcd = calc.compute("fck / gamma_c")
    fyd = calc.compute("fyk / gamma_s")
    
    variables_mc = {
        "Mk": {"mean": Mk.value, "std": 5, "dist": "normal"},
        "gamma_f": {"mean": 1.4, "std": 0.05, "dist": "normal"}
    }
    
    Md_dist = calc.monte_carlo(
        "Mk * gamma_f",
        variables=variables_mc,
        n_samples=1000,
        seed=42
    )
    
    assert Mk.value > 0
    assert Md.value > Mk.value
    assert fcd.value < 25.0
    assert fyd.value > 400.0
    assert Md_dist.metadata["mean"] > 0
