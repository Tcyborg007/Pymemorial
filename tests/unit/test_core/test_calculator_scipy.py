"""
test_calculator_scipy.py

Testes TDD para Calculator - Parte 2/3: SciPy Numerical Methods

Seguindo filosofia PyMemorial:
- Sintaxe natural
- Métodos numéricos robustos
- Graceful degradation (skip se SciPy não disponível)
"""

import pytest
import math
from pymemorial.core.calculator import Calculator, CalculationResult, CalculatorError

# Verificar disponibilidade do SciPy
try:
    import scipy
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
# TEST CLASS 1: SOLVE EQUATION NUMERICAL
# ============================================================

@pytest.mark.skipif(not SCIPY_AVAILABLE, reason="SciPy não disponível")
class TestCalculatorSolveEquationNumerical:
    """Testes para solve_equation_numerical() - Resolver equações."""
    
    def test_solve_quadratic_equation(self, calc):
        """Deve resolver equação quadrática x^2 - 4 = 0."""
        # RED: Este teste FALHARÁ primeiro
        result = calc.solve_equation_numerical("x**2 - 4", "x", initial_guess=1.0)
        
        assert isinstance(result, CalculationResult)
        assert abs(result.value - 2.0) < 1e-5, "Raiz deve ser x=2"
        assert result.metadata['converged'] is True
        assert abs(result.metadata['residual']) < 1e-6
    
    def test_solve_transcendental_equation(self, calc):
        """Deve resolver equação transcendental x - cos(x) = 0."""
        result = calc.solve_equation_numerical("x - cos(x)", "x", initial_guess=0.5)
        
        # x ≈ 0.7390851332151607
        assert abs(result.value - 0.7390851) < 1e-5
        assert result.metadata['converged'] is True
    
    def test_solve_cubic_equation(self, calc):
        """Deve resolver x^3 - 2*x - 5 = 0."""
        result = calc.solve_equation_numerical("x**3 - 2*x - 5", "x", initial_guess=2.0)
        
        # Raiz real ≈ 2.0946
        assert abs(result.value - 2.0946) < 1e-3
    
    def test_solve_with_different_initial_guess(self, calc):
        """Deve encontrar diferentes raízes com chutes iniciais diferentes."""
        # x^2 - 1 tem raízes em x=-1 e x=1
        result1 = calc.solve_equation_numerical("x**2 - 1", "x", initial_guess=-2.0)
        result2 = calc.solve_equation_numerical("x**2 - 1", "x", initial_guess=2.0)
        
        assert abs(result1.value - (-1.0)) < 1e-5
        assert abs(result2.value - 1.0) < 1e-5
    
    def test_solve_raises_error_without_scipy(self, calc, monkeypatch):
        """Deve levantar erro se SciPy não disponível."""
        # Simular ausência de SciPy
        monkeypatch.setattr('pymemorial.core.calculator.SCIPY_AVAILABLE', False)
        
        with pytest.raises(CalculatorError, match="SciPy não disponível"):
            calc.solve_equation_numerical("x**2 - 4", "x", initial_guess=1.0)


# ============================================================
# TEST CLASS 2: FIND ROOTS
# ============================================================

@pytest.mark.skipif(not SCIPY_AVAILABLE, reason="SciPy não disponível")
class TestCalculatorFindRoots:
    """Testes para find_roots() - Encontrar múltiplas raízes."""
    
    def test_find_roots_cubic(self, calc):
        """Deve encontrar 3 raízes de x^3 - x = 0."""
        roots = calc.find_roots("x**3 - x", "x", bounds=(-2, 2), num_points=20)
        
        assert len(roots) == 3, "Deve encontrar 3 raízes"
        values = sorted([r.value for r in roots])
        assert abs(values[0] - (-1.0)) < 1e-5
        assert abs(values[1] - 0.0) < 1e-5
        assert abs(values[2] - 1.0) < 1e-5
    
    def test_find_roots_sin(self, calc):
        """Deve encontrar raízes de sin(x) em [0, 2π]."""
        roots = calc.find_roots("sin(x)", "x", bounds=(0, 2*math.pi), num_points=50)
        
        # sin(x) = 0 em x=π (0 e 2π são bordas, não conta)
        assert len(roots) >= 1, "Deve encontrar pelo menos 1 raiz (π)"

    
    def test_find_roots_no_roots(self, calc):
        """Deve retornar lista vazia se não houver raízes."""
        roots = calc.find_roots("x**2 + 1", "x", bounds=(-10, 10))
        
        assert len(roots) == 0, "Não deve encontrar raízes reais"
    
    def test_find_roots_returns_calculation_results(self, calc):
        """Deve retornar lista de CalculationResult."""
        roots = calc.find_roots("x**2 - 4", "x", bounds=(-3, 3))
        
        assert all(isinstance(r, CalculationResult) for r in roots)
        assert all(r.metadata['method'] == 'brentq' for r in roots)


# ============================================================
# TEST CLASS 3: OPTIMIZE FUNCTION
# ============================================================

@pytest.mark.skipif(not SCIPY_AVAILABLE, reason="SciPy não disponível")
class TestCalculatorOptimizeFunction:
    """Testes para optimize_function() - Otimização."""
    
    def test_minimize_quadratic(self, calc):
        """Deve minimizar x^2 + 2*x + 1 (mínimo em x=-1)."""
        result = calc.optimize_function("x**2 + 2*x + 1", "x", initial_guess=0)
        
        assert abs(result.value - (-1.0)) < 1e-5
        assert abs(result.metadata['optimal_value'] - 0.0) < 1e-5
        assert result.metadata['success'] is True
    
    def test_maximize_function(self, calc):
        """Deve maximizar -(x-3)^2 + 5 (máximo em x=3)."""
        result = calc.optimize_function(
            "-(x-3)**2 + 5", "x", initial_guess=0, maximize=True
        )
        
        assert abs(result.value - 3.0) < 1e-5
        assert abs(result.metadata['optimal_value'] - 5.0) < 1e-5
    
    def test_optimize_with_bounds(self, calc):
        """Deve respeitar limites na otimização."""
        result = calc.optimize_function(
            "x**2", "x", initial_guess=5, bounds=(2, 10)
        )
        
        # Mínimo seria x=0, mas está limitado a [2,10]
        assert result.value >= 2.0
        assert result.value <= 10.0
        assert abs(result.value - 2.0) < 1e-5  # Deve ir para o limite inferior
    
    def test_optimize_cubic_function(self, calc):
        """Deve encontrar mínimo local de x^3 - 3*x."""
        result = calc.optimize_function(
            "x**3 - 3*x", "x", initial_guess=0.5, bounds=(-5, 5)
        )
        
        # Mínimo local em x=1
        assert abs(result.value - 1.0) < 1e-2



# ============================================================
# TEST CLASS 4: INTEGRATE NUMERICAL
# ============================================================

@pytest.mark.skipif(not SCIPY_AVAILABLE, reason="SciPy não disponível")
class TestCalculatorIntegrateNumerical:
    """Testes para integrate_numerical() - Integração numérica."""
    
    def test_integrate_polynomial(self, calc):
        """Deve integrar ∫[0,1] x^2 dx = 1/3."""
        result = calc.integrate_numerical("x**2", "x", lower=0, upper=1)
        
        assert abs(result.value - (1/3)) < 1e-6
        assert result.metadata['method'] == 'quad'
        assert result.metadata['error_estimate'] is not None
    
    def test_integrate_trigonometric(self, calc):
        """Deve integrar ∫[0,π] sin(x) dx = 2."""
        result = calc.integrate_numerical("sin(x)", "x", lower=0, upper=math.pi)
        
        assert abs(result.value - 2.0) < 1e-6
    
    def test_integrate_exponential(self, calc):
        """Deve integrar ∫[0,1] e^x dx = e - 1."""
        result = calc.integrate_numerical("exp(x)", "x", lower=0, upper=1)
        
        expected = math.e - 1
        assert abs(result.value - expected) < 1e-6
    
    def test_integrate_with_high_tolerance(self, calc):
        """Deve respeitar tolerância especificada."""
        result = calc.integrate_numerical(
            "x**2", "x", lower=0, upper=1, epsabs=1e-10, epsrel=1e-10
        )
        
        assert abs(result.value - (1/3)) < 1e-9


# ============================================================
# TEST CLASS 5: DIFFERENTIATE NUMERICAL
# ============================================================

@pytest.mark.skipif(not SCIPY_AVAILABLE, reason="SciPy não disponível")
class TestCalculatorDifferentiateNumerical:
    """Testes para differentiate_numerical() - Derivação numérica."""
    
    def test_differentiate_polynomial(self, calc):
        """Deve derivar d/dx(x^2) no ponto x=3 = 6."""
        result = calc.differentiate_numerical("x**2", "x", at_point=3)
        
        assert abs(result.value - 6.0) < 1e-3
        assert result.metadata['order'] == 1
    
    def test_differentiate_sin(self, calc):
        """Deve derivar d/dx(sin(x)) no ponto x=0 = cos(0) = 1."""
        result = calc.differentiate_numerical("sin(x)", "x", at_point=0)
        
        assert abs(result.value - 1.0) < 1e-3
    
    def test_differentiate_second_order(self, calc):
        """Deve calcular segunda derivada d²/dx²(x^3) em x=2 = 12."""
        result = calc.differentiate_numerical("x**3", "x", at_point=2, order=2)
        
        # d²/dx²(x³) = 6x → 6*2 = 12
        assert abs(result.value - 12.0) < 1e-2
        assert result.metadata['order'] == 2
    
    def test_differentiate_exponential(self, calc):
        """Deve derivar d/dx(e^x) em x=1 = e."""
        result = calc.differentiate_numerical("exp(x)", "x", at_point=1)
        
        assert abs(result.value - math.e) < 1e-3


# ============================================================
# TEST CLASS 6: SCIPY ERROR HANDLING
# ============================================================

class TestCalculatorScipyErrorHandling:
    """Testes de tratamento de erros para métodos SciPy."""
    
    def test_scipy_not_available_error_message(self, calc, monkeypatch):
        """Deve dar mensagem clara se SciPy não disponível."""
        monkeypatch.setattr('pymemorial.core.calculator.SCIPY_AVAILABLE', False)
        
        with pytest.raises(CalculatorError, match="SciPy não disponível"):
            calc.solve_equation_numerical("x**2 - 1", "x", initial_guess=0)
    
    @pytest.mark.skipif(not SCIPY_AVAILABLE, reason="SciPy não disponível")
    def test_convergence_failure_handled(self, calc):
        """Deve lidar graciosamente com falha de convergência."""
        # Equação sem solução real: exp(x) + 1 = 0
        with pytest.raises(CalculatorError):
            calc.solve_equation_numerical(
                "exp(x) + 1", "x", initial_guess=0.0, tol=1e-10
            )



# ============================================================
# TEST CLASS 7: SCIPY INTEGRATION WITH PYMEMORIAL
# ============================================================

@pytest.mark.skipif(not SCIPY_AVAILABLE, reason="SciPy não disponível")
class TestCalculatorScipyIntegration:
    """Testes de integração entre SciPy e resto do PyMemorial."""
    
    def test_solve_then_verify_symbolic(self, calc):
        """Deve resolver numericamente e verificar com SymPy."""
        # Resolver numericamente
        result = calc.solve_equation_numerical("x**2 - 4", "x", initial_guess=1.0)
        
        # Verificar simbolicamente
        x_solution = result.value
        calc.add_variable("x", x_solution)
        verification = calc.compute("x**2 - 4")
        
        # FIX: compute() retorna CalculationResult, não float
        if isinstance(verification, CalculationResult):
            assert abs(verification.value) < 1e-5, "Solução deve satisfazer equação"
        else:
            assert abs(verification) < 1e-5, "Solução deve satisfazer equação"

    
    def test_optimize_then_compute_derivative(self, calc):
        """Deve otimizar e verificar que derivada é zero no ótimo."""
        # Otimizar
        result = calc.optimize_function("x**2 - 4*x + 3", "x", initial_guess=0)
        x_opt = result.value
        
        # Derivada simbólica no ponto ótimo deve ser ~0
        deriv = calc.diff("x**2 - 4*x + 3", "x")
        deriv_at_opt = deriv.subs('x', x_opt)
        
        assert abs(float(deriv_at_opt)) < 1e-5
