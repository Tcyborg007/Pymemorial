"""
test_calculator_advanced.py

Testes TDD para Calculator - Parte 3/3: Vetorização + Monte Carlo + Performance

Seguindo filosofia PyMemorial:
- Análises probabilísticas para engenharia estrutural
- Vetorização eficiente com NumPy
- Performance otimizada
"""

import pytest
import numpy as np
from pymemorial.core.calculator import Calculator, CalculationResult, CalculatorError

# Verificar disponibilidade
try:
    import scipy.stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def calc():
    """Calculator limpo para cada teste."""
    return Calculator()


# ============================================================
# TEST CLASS 1: VECTORIZED COMPUTATION
# ============================================================

class TestCalculatorVectorized:
    """Testes para computação vetorizada NumPy."""
    
    def test_compute_array_simple(self, calc):
        """Deve computar array simples."""
        x_values = np.array([1, 2, 3, 4, 5])
        calc.add_variable("x", x_values)
        
        result = calc.compute("x**2", vectorized=True)
        
        expected = np.array([1, 4, 9, 16, 25])
        assert isinstance(result, CalculationResult)
        assert np.allclose(result.value, expected)
    
    def test_compute_array_multiple_variables(self, calc):
        """Deve computar com múltiplos arrays."""
        calc.add_variable("F", np.array([10, 20, 30]))
        calc.add_variable("L", np.array([2, 4, 6]))
        
        # M = F * L / 4
        result = calc.compute("F * L / 4", vectorized=True)
        
        expected = np.array([5.0, 20.0, 45.0])
        assert np.allclose(result.value, expected)
    
    def test_compute_broadcasting(self, calc):
        """Deve suportar broadcasting NumPy."""
        calc.add_variable("F", np.array([[1, 2], [3, 4]]))
        calc.add_variable("factor", 10)
        
        result = calc.compute("F * factor", vectorized=True)
        
        expected = np.array([[10, 20], [30, 40]])
        assert np.allclose(result.value, expected)
    
    def test_compute_trigonometric_vectorized(self, calc):
        """Deve computar funções trigonométricas vetorizadas."""
        angles = np.array([0, np.pi/4, np.pi/2, np.pi])
        calc.add_variable("theta", angles)
        
        result = calc.compute("sin(theta)", vectorized=True)
        
        expected = np.array([0, np.sqrt(2)/2, 1, 0])
        assert np.allclose(result.value, expected, atol=1e-10)


# ============================================================
# TEST CLASS 2: MONTE CARLO SIMULATION
# ============================================================

@pytest.mark.skipif(not SCIPY_AVAILABLE, reason="SciPy não disponível")
class TestCalculatorMonteCarlo:
    """Testes para simulação Monte Carlo."""
    
    def test_monte_carlo_simple(self, calc):
        """Deve executar simulação Monte Carlo simples."""
        # Variáveis com distribuição normal
        variables = {
            "fck": {"mean": 25, "std": 3, "dist": "normal"},
            "A": {"mean": 0.25, "std": 0.01, "dist": "normal"}
        }
        
        # Resistência característica
        result = calc.monte_carlo(
            expression="fck * A",
            variables=variables,
            n_samples=1000,
            seed=42
        )
        
        assert isinstance(result, CalculationResult)
        assert "mean" in result.metadata
        assert "std" in result.metadata
        assert "percentile_5" in result.metadata
        assert "percentile_95" in result.metadata
        
        # Valor esperado ~ 25 * 0.25 = 6.25
        assert abs(result.metadata["mean"] - 6.25) < 0.5
    
    def test_monte_carlo_structural_load(self, calc):
        """Deve simular carga estrutural com incertezas."""
        variables = {
            "q": {"mean": 15, "std": 2, "dist": "normal"},  # kN/m
            "L": {"mean": 6, "std": 0.1, "dist": "normal"}  # m
        }
        
        # Momento máximo: M = q*L²/8
        result = calc.monte_carlo(
            expression="q * L**2 / 8",
            variables=variables,
            n_samples=5000,
            seed=42
        )
        
        # Valor esperado ~ 15 * 36 / 8 = 67.5
        assert abs(result.metadata["mean"] - 67.5) < 2.0
        assert result.metadata["std"] > 0
    
    def test_monte_carlo_uniform_distribution(self, calc):
        """Deve suportar distribuição uniforme."""
        variables = {
            "x": {"min": 0, "max": 10, "dist": "uniform"}
        }
        
        result = calc.monte_carlo(
            expression="x**2",
            variables=variables,
            n_samples=10000,
            seed=42
        )
        
        # Média de x² com x~U(0,10) é 100/3 ~ 33.33
        assert abs(result.metadata["mean"] - 33.33) < 2.0
    
    def test_monte_carlo_confidence_intervals(self, calc):
        """Deve calcular intervalos de confiança."""
        variables = {
            "x": {"mean": 100, "std": 10, "dist": "normal"}
        }
        
        result = calc.monte_carlo(
            expression="x",
            variables=variables,
            n_samples=10000,
            confidence_levels=[0.05, 0.95],
            seed=42
        )
        
        # Intervalo 90% para N(100, 10)
        assert result.metadata["percentile_5"] < 100
        assert result.metadata["percentile_95"] > 100


# ============================================================
# TEST CLASS 3: ERROR PROPAGATION
# ============================================================

class TestCalculatorErrorPropagation:
    """Testes para propagação de incertezas."""
    
    def test_error_propagation_addition(self, calc):
        """Deve propagar erro em adição."""
        calc.add_variable("x", 10, uncertainty=0.5)
        calc.add_variable("y", 5, uncertainty=0.3)
        
        result = calc.compute("x + y", propagate_error=True)
        
        # σ(x+y) = √(σx² + σy²) = √(0.25 + 0.09) ≈ 0.583
        assert "uncertainty" in result.metadata
        assert abs(result.metadata["uncertainty"] - 0.583) < 0.01
    
    def test_error_propagation_multiplication(self, calc):
        """Deve propagar erro em multiplicação."""
        calc.add_variable("F", 100, uncertainty=5)
        calc.add_variable("L", 2, uncertainty=0.1)
        
        result = calc.compute("F * L", propagate_error=True)
        
        # Erro relativo: √((5/100)² + (0.1/2)²) * 200
        assert "uncertainty" in result.metadata
        assert result.metadata["uncertainty"] > 0


# ============================================================
# TEST CLASS 4: BATCH PROCESSING ADVANCED
# ============================================================

class TestCalculatorBatchAdvanced:
    """Testes de batch processing avançado."""
    
    def test_batch_with_different_variables(self, calc):
        """Deve processar batch com diferentes variáveis."""
        calc.add_variable("L", 6)
        calc.add_variable("q", 15)
        
        batch_specs = [
            {"expression": "q * L**2 / 8", "label": "Mk"},
            {"expression": "Mk * 1.4", "label": "Md", "depends_on": ["Mk"]},
            {"expression": "L / 2", "label": "x_max"}
        ]
        
        results = calc.batch_compute_advanced(batch_specs)
        
        assert len(results) == 3
        assert results["Mk"].value == 67.5
        assert results["Md"].value == 94.5
        assert results["x_max"].value == 3.0
    
    def test_batch_parallel_execution(self, calc):
        """Deve executar batch em paralelo."""
        calc.add_variable("x", 10)
        
        expressions = [f"x**{i}" for i in range(1, 11)]
        
        results = calc.batch_compute(expressions, parallel=True, workers=4)
        
        assert len(results) == 10
        assert results[0].value == 10
        assert results[9].value == 10**10


# ============================================================
# TEST CLASS 5: PERFORMANCE OPTIMIZATION
# ============================================================

class TestCalculatorPerformance:
    """Testes de otimização de performance."""
    
    def test_vectorize_decorator(self, calc):
        """Deve vetorizar automaticamente expressões."""
        x_values = np.linspace(0, 10, 1000)
        calc.add_variable("x", x_values)
        
        # Expressão complexa
        result = calc.compute(
            "sin(x) * cos(x) + exp(-x/10)",
            vectorized=True,
            optimize=True
        )
        
        assert result.value.shape == (1000,)
        assert "optimization_used" in result.metadata
    
    def test_cache_compiled_functions(self, calc):
        """Deve cachear funções compiladas."""
        calc.add_variable("x", 10)
        
        # Primeira execução - popula cache
        result1 = calc.compute("x**2 + 2*x + 1")
        
        # Trocar valor da variável
        calc.add_variable("x", 20)
        
        # Segunda execução - MESMA EXPRESSÃO, mas valor diferente
        result2 = calc.compute("x**2 + 2*x + 1")
        
        # Verificar que resultados são diferentes (valores diferentes)
        assert result1.value != result2.value
        assert result1.value == 121  # 10² + 2*10 + 1
        assert result2.value == 441  # 20² + 2*20 + 1
        
        # Cache stats - verificar que cache interno foi usado
        # (mesmo que CalculationResult seja novo)
        stats = calc.cache_stats()
        assert "total" in stats
        assert "size" in stats



# ============================================================
# TEST CLASS 6: INTEGRATION TEST
# ============================================================

class TestCalculatorAdvancedIntegration:
    """Testes de integração completa."""
    
    @pytest.mark.skipif(not SCIPY_AVAILABLE, reason="SciPy não disponível")
    def test_complete_structural_analysis(self, calc):
        """Smoke test: Análise estrutural completa com incertezas."""
        # Dados de entrada com incertezas
        calc.add_variable("L", 6.0, uncertainty=0.1)
        calc.add_variable("q", 15.0, uncertainty=2.0)
        calc.add_variable("fck", 25.0, uncertainty=3.0)
        
        # Momento característico
        Mk = calc.compute("q * L**2 / 8")
        calc.add_variable("Mk", Mk.value)
        
        # Monte Carlo para momento de cálculo
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
        
        assert Md_dist.metadata["mean"] > 90
        assert Md_dist.metadata["percentile_95"] > Md_dist.metadata["mean"]
