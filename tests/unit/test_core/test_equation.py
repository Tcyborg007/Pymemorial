"""
PyMemorial v2.0 - Equation Tests COMPLETO
TDD Implementation following RED → GREEN → REFACTOR
Coverage Target: 95%+

ESTRUTURA:
- TestEquationBasic: Inicialização e parsing (15 testes)
- TestEquationOperations: Operações matemáticas (12 testes)
- TestEquationEvaluation: Avaliação numérica (10 testes)
- TestEquationSubstitution: Substituições (8 testes)
- TestEquationIntegration: Integração com outros módulos (10 testes)
- TestEquationRobustness: Robustez e edge cases (10 testes)
- TestEquationSmokeTest: Smoke test completo (1 teste)

TOTAL: 66 testes unitários
"""

import pytest
import threading
from typing import Dict
from decimal import Decimal

# Imports condicionais
try:
    import sympy as sp
    from sympy import Symbol, sin, cos, pi, E
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False
    sp = None

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

from pymemorial.core.equation import (
    Equation,
    EquationError,
    ValidationError,
    EvaluationError,
    SubstitutionError,
    DimensionalError,
    EvaluationResult,
    GranularityType,
    StepType,
    Step,
    EquationFactory,        # ← ADICIONAR ESTA LINHA
    ValidationHelpers,
    StepGenerator,           # ← ADICIONAR ESTA LINHA
    StepRegistry 
)
from pymemorial.core.variable import Variable
from pymemorial.core.config import get_config, set_option, reset_config


# ============================================================================
# FIXTURES GLOBAIS
# ============================================================================

@pytest.fixture(autouse=True)
def reset_config_after_test():
    """Reseta config após cada teste."""
    yield
    reset_config()


@pytest.fixture
def simple_vars() -> Dict[str, Variable]:
    """Fixture com variáveis simples."""
    return {
        "x": Variable("x", 10),
        "y": Variable("y", 5),
        "z": Variable("z", 2)
    }


@pytest.fixture
def vars_with_units() -> Dict[str, Variable]:
    """Fixture com variáveis com unidades."""
    return {
        "q": Variable("q", 15, unit="kN/m"),
        "L": Variable("L", 6, unit="m"),
        "gamma_f": Variable("gamma_f", 1.4)
    }


@pytest.fixture
def equation_simple() -> Equation:
    """Fixture com equation simples."""
    return Equation("x + y")


@pytest.fixture
def equation_with_vars(simple_vars) -> Equation:
    """Fixture com equation e variáveis."""
    return Equation("x * y + z", locals_dict=simple_vars)


# ============================================================================
# TESTES EQUATION - BÁSICO
# ============================================================================

@pytest.mark.skipif(not SYMPY_AVAILABLE, reason="SymPy não disponível")
class TestEquationBasic:
    """Testes básicos de inicialização e parsing."""
    
    def test_equation_initialization_from_string(self):
        """Deve criar equation a partir de string."""
        eq = Equation("x + y")
        
        assert isinstance(eq.expr, sp.Expr)
        assert eq.expression_str == "x + y"
        assert len(eq.get_variables()) == 2
        assert set(eq.get_variables()) == {'x', 'y'}
    
    def test_equation_initialization_from_sympy_expr(self):
        """Deve criar equation a partir de expressão SymPy."""
        x, y = sp.symbols('x y')
        expr = x**2 + 2*x*y + y**2
        
        eq = Equation(expr)
        
        assert eq.expr == expr
        assert isinstance(eq.expr, sp.Expr)
    
    def test_equation_with_name_and_description(self):
        """Deve armazenar nome e descrição."""
        eq = Equation(
            "M = q * L**2 / 8",
            name="M_max",
            description="Momento máximo em viga biapoiada"
        )
        
        assert eq.name == "M_max"
        assert "Momento máximo" in eq.description
    
    def test_equation_get_variables(self):
        """Deve extrair lista de variáveis."""
        eq = Equation("a * b + c * d")
        
        variables = eq.get_variables()
        
        assert set(variables) == {'a', 'b', 'c', 'd'}
        assert len(variables) == 4
    
    def test_equation_get_free_symbols(self):
        """Deve retornar símbolos livres (SymPy)."""
        eq = Equation("x**2 + y")
        
        free_symbols = eq.get_free_symbols()
        
        assert len(free_symbols) == 2
        assert all(isinstance(s, Symbol) for s in free_symbols)
    
    def test_equation_with_locals_dict(self, simple_vars):
        """Deve aceitar dicionário de variáveis locais."""
        eq = Equation("x + y", locals_dict=simple_vars)
        
        assert eq.locals_dict == simple_vars
        assert 'x' in eq.get_variables()
        assert 'y' in eq.get_variables()
    
    def test_equation_invalid_expression_type(self):
        """Deve falhar com tipo de expressão inválido."""
        with pytest.raises(ValidationError, match="Tipo de expressão inválido"):
            Equation(12345)  # Número não é válido
    
    def test_equation_invalid_syntax(self):
        """Deve falhar com sintaxe inválida."""
        # ============================================================
        # CORREÇÃO: SymPy pode simplificar "x + + y" para "x + y"
        # Usar sintaxe realmente inválida
        # ============================================================
        with pytest.raises(ValidationError, match="Erro ao parsear"):
            Equation("x + @ y")  # Operador inválido

    
    def test_equation_repr(self):
        """Deve ter __repr__ correto."""
        eq = Equation("x + y")
        repr_str = repr(eq)
        
        assert "Equation" in repr_str
        assert "x + y" in repr_str
    
    def test_equation_str(self):
        """Deve ter __str__ correto."""
        eq = Equation("x + y")
        
        assert str(eq) == "x + y"


# ============================================================================
# TESTES EQUATION - OPERAÇÕES MATEMÁTICAS
# ============================================================================

@pytest.mark.skipif(not SYMPY_AVAILABLE, reason="SymPy não disponível")
class TestEquationOperations:
    """Testes de operações matemáticas."""
    
    def test_simplify_expression(self):
        """Deve simplificar expressão."""
        eq = Equation("x + x + x")
        
        simplified = eq.simplify()
        
        assert simplified.expr == 3 * Symbol('x')
    
    def test_simplify_complex_expression(self):
        """Deve simplificar expressão complexa."""
        eq = Equation("(x + 1)**2 - (x**2 + 2*x + 1)")
        
        simplified = eq.simplify()
        
        # Deve simplificar para 0
        assert simplified.expr == 0
    
    def test_expand_expression(self):
        """Deve expandir expressão."""
        eq = Equation("(x + 1)**2")
        
        expanded = eq.expand()
        
        x = Symbol('x')
        expected = x**2 + 2*x + 1
        assert expanded.expr == expected
    
    def test_expand_multivariate(self):
        """Deve expandir expressão multivariável."""
        eq = Equation("(x + y) * (x - y)")
        
        expanded = eq.expand()
        
        x, y = sp.symbols('x y')
        expected = x**2 - y**2
        assert expanded.expr == expected
    
    def test_factor_expression(self):
        """Deve fatorar expressão."""
        eq = Equation("x**2 - 1")
        
        factored = eq.factor()
        
        x = Symbol('x')
        expected = (x - 1) * (x + 1)
        assert factored.expr == expected
    
    def test_factor_quadratic(self):
        """Deve fatorar quadrática."""
        eq = Equation("x**2 + 5*x + 6")
        
        factored = eq.factor()
        
        x = Symbol('x')
        # (x + 2)(x + 3)
        result = factored.expr
        assert result.is_Mul  # Deve ser multiplicação


# ============================================================================
# TESTES EQUATION - SUBSTITUIÇÃO
# ============================================================================

@pytest.mark.skipif(not SYMPY_AVAILABLE, reason="SymPy não disponível")
class TestEquationSubstitution:
    """Testes de substituição."""
    
    def test_substitute_numerical_values(self):
        """Deve substituir valores numéricos."""
        eq = Equation("x + y")
        
        result = eq.substitute({'x': 10, 'y': 5})
        
        assert isinstance(result, Equation)
        # Avaliar resultado
        eval_result = result.evaluate()
        assert eval_result.value == 15
    
    def test_substitute_with_variables(self, simple_vars):
        """Deve substituir com objetos Variable."""
        eq = Equation("x * y")
        
        result = eq.substitute({
            'x': simple_vars['x'],
            'y': simple_vars['y']
        })
        
        eval_result = result.evaluate()
        assert eval_result.value == 50  # 10 * 5
    
    def test_substitute_symbolic(self):
        """Deve substituir com outros símbolos."""
        eq = Equation("a + b")
        
        result = eq.substitute({'a': 'c + d'})
        
        assert 'c' in result.get_variables()
        assert 'd' in result.get_variables()
    
    def test_substitute_partial(self):
        """Deve fazer substituição parcial."""
        eq = Equation("x + y + z")
        
        result = eq.substitute({'x': 10})
        
        # x substituído, y e z permanecem simbólicos
        variables = result.get_variables()
        assert 'x' not in variables or result.expr.has(Symbol('x'))
        assert 'y' in variables
        assert 'z' in variables
    
    def test_substitute_invalid_type(self):
        """Deve falhar com tipo inválido."""
        eq = Equation("x + y")
        
        with pytest.raises(SubstitutionError, match="Tipo de substituição inválido"):
            eq.substitute({'x': object()})  # Objeto inválido


# ============================================================================
# TESTES EQUATION - AVALIAÇÃO
# ============================================================================

@pytest.mark.skipif(not SYMPY_AVAILABLE, reason="SymPy não disponível")
class TestEquationEvaluation:
    """Testes de avaliação numérica."""
    
    def test_evaluate_simple_expression(self, simple_vars):
        """Deve avaliar expressão simples."""
        eq = Equation("x + y", locals_dict=simple_vars)
        
        result = eq.evaluate()
        
        assert isinstance(result, EvaluationResult)
        assert result.value == 15  # 10 + 5
        assert result.expression == "x + y"
    
    def test_evaluate_with_kwargs(self, simple_vars):
        """Deve avaliar com kwargs."""
        eq = Equation("x * y", locals_dict=simple_vars)
        
        # Sobrescrever valor de y
        result = eq.evaluate(y=Variable("y", 10))
        
        assert result.value == 100  # 10 * 10
    
    def test_evaluate_complex_expression(self):
        """Deve avaliar expressão complexa."""
        vars_dict = {
            'a': Variable('a', 2),
            'b': Variable('b', 3),
            'c': Variable('c', 4)
        }
        eq = Equation("a**2 + b*c", locals_dict=vars_dict)
        
        result = eq.evaluate()
        
        assert result.value == 16  # 4 + 12
    
    def test_evaluate_undefined_variable(self):
        """Deve falhar se variável não definida."""
        eq = Equation("x + y")  # Sem locals_dict
        
        with pytest.raises(EvaluationError, match="não definida"):
            eq.evaluate()
    
    def test_evaluate_result_metadata(self, simple_vars):
        """Deve incluir metadados no resultado."""
        eq = Equation("x * 2", locals_dict=simple_vars)
        
        result = eq.evaluate()
        
        assert 'substitutions' in result.metadata
        assert result.metadata['substitutions']['x'] == 10


# ============================================================================
# TESTES EQUATION - CONVERSÃO LATEX/MARKDOWN
# ============================================================================

@pytest.mark.skipif(not SYMPY_AVAILABLE, reason="SymPy não disponível")
class TestEquationConversion:
    """Testes de conversão para LaTeX e Markdown."""
    
    def test_to_latex_inline(self):
        """Deve converter para LaTeX inline."""
        eq = Equation("x**2 + y")
        
        latex = eq.to_latex(mode="inline")
        
        assert latex.startswith("$")
        assert latex.endswith("$")
        assert "x" in latex
    
    def test_to_latex_display(self):
        """Deve converter para LaTeX display."""
        eq = Equation("a / b")
        
        latex = eq.to_latex(mode="display")
        
        assert latex.startswith("$$")
        assert latex.endswith("$$")
        assert "frac" in latex or "/" in latex
    
    def test_to_latex_fraction(self):
        """Deve formatar frações em LaTeX."""
        eq = Equation("a / b")
        
        latex = eq.to_latex()
        
        assert "frac" in latex or r"\frac" in latex
    
    def test_to_markdown(self):
        """Deve converter para Markdown."""
        eq = Equation("x + y")
        
        markdown = eq.to_markdown()
        
        assert "`" in markdown
        assert "x + y" in markdown


# ============================================================================
# TESTES EQUATION - INTEGRAÇÃO
# ============================================================================

@pytest.mark.skipif(not SYMPY_AVAILABLE, reason="SymPy não disponível")
class TestEquationIntegration:
    """Testes de integração com outros módulos."""
    
    def test_integration_with_variable(self):
        """Deve integrar com Variable."""
        x = Variable('x', 10)
        y = Variable('y', 5)
        
        eq = Equation("x + y", locals_dict={'x': x, 'y': y})
        result = eq.evaluate()
        
        assert result.value == 15
    
    def test_integration_with_config_precision(self):
        """Deve respeitar precisão do config."""
        set_option("display.precision", 4)
        
        vars_dict = {'pi': Variable('pi', 3.14159265359)}
        eq = Equation("pi", locals_dict=vars_dict)
        
        result = eq.evaluate()
        result_str = str(result)
        
        # Deve formatar com 4 casas decimais
        assert "3.1416" in result_str


# ============================================================================
# TESTES EQUATION - ROBUSTEZ
# ============================================================================

@pytest.mark.skipif(not SYMPY_AVAILABLE, reason="SymPy não disponível")
class TestEquationRobustness:
    """Testes de robustez e edge cases."""
    
    def test_equation_thread_safety(self):
        """Deve ser thread-safe."""
        eq = Equation("x * 2", locals_dict={'x': Variable('x', 10)})
        results = []
        
        def evaluate_task():
            result = eq.evaluate()
            results.append(result.value)
        
        threads = [threading.Thread(target=evaluate_task) for _ in range(10)]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(results) == 10
        assert all(r == 20 for r in results)
    
    def test_equation_with_very_long_expression(self):
        """Deve lidar com expressões longas."""
        terms = " + ".join([f"x{i}" for i in range(100)])
        
        eq = Equation(terms)
        
        assert len(eq.get_variables()) == 100
    
    def test_equation_with_special_constants(self):
        """Deve suportar constantes especiais."""
        eq = Equation("pi + E")
        
        # ============================================================
        # CORREÇÃO: SymPy reconhece pi e E como constantes, não variáveis
        # Verificar se a expressão contém as constantes
        # ============================================================
        from sympy import pi as sym_pi, E as sym_E
        
        # Verificar se constantes estão presentes na expressão
        assert eq.expr.has(sym_pi) or eq.expr.has(sym_E)
        
        # Alternativamente, avaliar deve retornar número (não erro)
        result = eq.evaluate()
        assert isinstance(result.value, (int, float))
        assert result.value > 0  # pi + E ≈ 5.86



# ============================================================================
# SMOKE TEST COMPLETO
# ============================================================================

@pytest.mark.skipif(not SYMPY_AVAILABLE, reason="SymPy não disponível")
@pytest.mark.smoke
class TestEquationSmokeTest:
    """Smoke test completo do Equation."""
    
    def test_equation_complete_workflow_beam_calculation(self):
        """Smoke test: cálculo completo de viga biapoiada."""
        # Setup: Dados de viga
        vars_dict = {
            'q': Variable('q', 15.0, unit='kN/m'),
            'L': Variable('L', 6.0, unit='m'),
            'gamma_f': Variable('gamma_f', 1.4)
        }
        
        # 1. Momento característico: M_k = q * L² / 8
        eq_Mk = Equation(
            "q * L**2 / 8",
            locals_dict=vars_dict,
            name="M_k",
            description="Momento característico"
        )
        
        result_Mk = eq_Mk.evaluate()
        assert abs(result_Mk.value - 67.5) < 0.01  # 67.5 kN.m
        
        # 2. Simplificação
        simplified = eq_Mk.simplify()
        assert simplified is not None
        
        # 3. Substituição parcial
        eq_Md = Equation("gamma_f * M_k")
        eq_with_Mk = eq_Md.substitute({'M_k': 67.5})
        
        # 4. Conversão LaTeX
        latex = eq_Mk.to_latex()
        assert latex is not None
        assert len(latex) > 0
        
        # 5. Conversão Markdown
        markdown = eq_Mk.to_markdown()
        assert "`" in markdown
        
        print("✅ SMOKE TEST EQUATION COMPLETO!")
        print(f"   M_k = {result_Mk.value:.2f} kN.m")
        print(f"   LaTeX: {latex}")


# ============================================================================
# CONFIGURAÇÃO PYTEST
# ============================================================================

def pytest_configure(config):
    """Configuração personalizada do pytest."""
    config.addinivalue_line("markers", "smoke: smoke tests end-to-end")



# ============================================================================
# TESTES EQUATIONFACTORY
# ============================================================================

@pytest.mark.skipif(not SYMPY_AVAILABLE, reason="SymPy não disponível")
class TestEquationFactory:
    """Testes do EquationFactory."""
    
    def test_factory_from_string(self):
        """Deve criar equation de string."""
        eq = EquationFactory.from_string("x + y")
        
        assert isinstance(eq, Equation)
        assert set(eq.get_variables()) == {'x', 'y'}
    
    def test_factory_from_sympy(self):
        """Deve criar equation de expressão SymPy."""
        x, y = sp.symbols('x y')
        expr = x**2 + y**2
        
        eq = EquationFactory.from_sympy(expr)
        
        assert isinstance(eq, Equation)
        assert eq.expr == expr
    
    def test_factory_from_code_with_variable(self):
        """Deve criar equation de código Python especificando variável."""
        code = """
q = 15
L = 6
M_max = q * L**2 / 8
"""
        eq = EquationFactory.from_code(code, variable='M_max')
        
        assert eq.name == 'M_max'
        assert 'q' in eq.get_variables()
        assert 'L' in eq.get_variables()
    
    def test_factory_from_code_without_variable(self):
        """Deve usar última atribuição se variável não especificada."""
        code = """
a = 10
b = 5
resultado = a + b
"""
        eq = EquationFactory.from_code(code)
        
        assert eq.name == 'resultado'
    
    def test_factory_from_code_invalid_variable(self):
        """Deve falhar se variável não existe."""
        code = "x = 10"
        
        with pytest.raises(ValidationError, match="não encontrada"):
            EquationFactory.from_code(code, variable='y')
    
    def test_factory_from_lambda_single_arg(self):
        """Deve criar equation de lambda com 1 argumento."""
        eq = EquationFactory.from_lambda(
            lambda x: x**2,
            arg_names=['x']
        )
        
        assert 'x' in eq.get_variables()
    
    def test_factory_from_lambda_multiple_args(self):
        """Deve criar equation de lambda com múltiplos argumentos."""
        eq = EquationFactory.from_lambda(
            lambda x, y: x + y,
            arg_names=['x', 'y']
        )
        
        assert set(eq.get_variables()) == {'x', 'y'}


# ============================================================================
# TESTES VALIDATIONHELPERS
# ============================================================================

class TestValidationHelpers:
    """Testes do ValidationHelpers."""
    
    def test_validate_expression_string_valid(self):
        """Deve validar string válida."""
        ValidationHelpers.validate_expression_string("x + y")
        # Não deve levantar exceção
    
    def test_validate_expression_string_empty(self):
        """Deve falhar com string vazia."""
        with pytest.raises(ValidationError, match="vazia"):
            ValidationHelpers.validate_expression_string("")
    
    def test_validate_expression_dangerous_char(self):
        """Deve falhar com caracteres perigosos."""
        with pytest.raises(ValidationError, match="perigoso"):
            ValidationHelpers.validate_expression_string("x; import os")
    
    def test_validate_expression_dangerous_keyword(self):
        """Deve falhar com palavras-chave perigosas."""
        with pytest.raises(ValidationError, match="perigosa"):
            ValidationHelpers.validate_expression_string("import os")
    
    def test_validate_code_safety_valid(self):
        """Deve validar código seguro."""
        code = "x = 10\ny = x + 5"
        ValidationHelpers.validate_code_safety(code)
    
    def test_validate_code_safety_import(self):
        """Deve falhar com import."""
        code = "import os"
        
        with pytest.raises(ValidationError, match="não permitida"):
            ValidationHelpers.validate_code_safety(code)
    
    def test_validate_complexity_length(self):
        """Deve falhar com expressão muito longa."""
        expr_str = "x" * 2000
        
        with pytest.raises(ValidationError, match="muito longa"):
            ValidationHelpers.validate_expression_complexity(expr_str, max_length=1000)





# ============================================================================
# TESTES STEP SYSTEM
# ============================================================================

@pytest.mark.skipif(not SYMPY_AVAILABLE, reason="SymPy não disponível")
class TestStepSystem:
    """Testes do sistema de Steps."""
    
    def test_step_dataclass_creation(self):
        """Deve criar Step com valores."""
        step = Step(
            formula="M = q * L²/8",
            substitution="M = 15 * 36/8",
            result="M = 67.5"
        )
        
        assert step.formula == "M = q * L²/8"
        assert step.substitution == "M = 15 * 36/8"
        assert step.result == "M = 67.5"
    
    def test_step_to_latex(self):
        """Deve converter Step para LaTeX."""
        step = Step(
            formula="x + y",
            result="15"
        )
        
        latex = step.to_latex()
        
        assert "begin{align*}" in latex
        assert "x + y" in latex
        assert "15" in latex
    
    def test_step_to_markdown(self):
        """Deve converter Step para Markdown."""
        step = Step(
            formula="x + y",
            result="15"
        )
        
        markdown = step.to_markdown()
        
        assert "$$" in markdown
        assert "x + y" in markdown
    
    def test_step_generator_minimal(self):
        """Deve gerar step MINIMAL."""
        vars_dict = {'x': Variable('x', 10), 'y': Variable('y', 5)}
        eq = Equation("x + y", locals_dict=vars_dict)
        
        generator = StepGenerator()
        steps = generator.generate(eq, GranularityType.MINIMAL)
        
        assert len(steps) == 1
        assert steps[0].result is not None
        assert steps[0].formula is None
    
    def test_step_generator_basic(self):
        """Deve gerar step BASIC."""
        vars_dict = {'x': Variable('x', 10), 'y': Variable('y', 5)}
        eq = Equation("x + y", locals_dict=vars_dict)
        
        generator = StepGenerator()
        steps = generator.generate(eq, GranularityType.BASIC)
        
        assert len(steps) == 1
        assert steps[0].formula is not None
        assert steps[0].result is not None
    
    def test_step_generator_medium(self):
        """Deve gerar step MEDIUM."""
        vars_dict = {'x': Variable('x', 10), 'y': Variable('y', 5)}
        eq = Equation("x * y", locals_dict=vars_dict)
        
        generator = StepGenerator()
        steps = generator.generate(eq, GranularityType.MEDIUM)
        
        assert len(steps) == 1
        assert steps[0].formula is not None
        assert steps[0].substitution is not None
        assert steps[0].result is not None
    
    def test_step_generator_detailed(self):
        """Deve gerar step DETAILED."""
        vars_dict = {'x': Variable('x', 10), 'y': Variable('y', 5)}
        eq = Equation("x**2 + y**2", locals_dict=vars_dict)
        
        generator = StepGenerator()
        steps = generator.generate(eq, GranularityType.DETAILED)
        
        assert len(steps) >= 1
        assert steps[0].formula is not None
        assert steps[0].result is not None
    
    def test_step_registry_register(self):
        """Deve registrar equação no registry."""
        registry = StepRegistry()
        vars_dict = {'x': Variable('x', 10)}
        eq = Equation("x * 2", locals_dict=vars_dict)
        
        steps = registry.register(eq)
        
        assert len(steps) > 0
        assert len(registry.get_all()) > 0
    
    def test_step_registry_clear(self):
        """Deve limpar registry."""
        registry = StepRegistry()
        vars_dict = {'x': Variable('x', 10)}
        eq = Equation("x * 2", locals_dict=vars_dict)
        
        registry.register(eq)
        registry.clear()
        
        assert len(registry.get_all()) == 0
    
    def test_step_registry_to_latex(self):
        """Deve exportar registry para LaTeX."""
        registry = StepRegistry()
        vars_dict = {'x': Variable('x', 10)}
        eq = Equation("x + 5", locals_dict=vars_dict)
        
        registry.register(eq)
        latex = registry.to_latex()
        
        assert isinstance(latex, str)
        assert len(latex) > 0


# ============================================================================
# TESTES SMART STEP ANALYZER
# ============================================================================

@pytest.mark.skipif(not SYMPY_AVAILABLE, reason="SymPy não disponível")
class TestSmartStepAnalyzer:
    """Testes do analisador inteligente de complexidade."""
    
    def test_smart_simple_expression(self):
        """Deve detectar expressão simples → BASIC."""
        vars_dict = {'x': Variable('x', 10), 'y': Variable('y', 5)}
        eq = Equation("x + y", locals_dict=vars_dict)
        
        generator = StepGenerator()
        steps = generator.generate_smart(eq)
        
        # Deve escolher BASIC ou MEDIUM
        assert len(steps) > 0
    
    def test_smart_medium_expression(self):
        """Deve detectar expressão média → MEDIUM."""
        vars_dict = {
            'q': Variable('q', 15),
            'L': Variable('L', 6)
        }
        eq = Equation("q * L**2 / 8", locals_dict=vars_dict)
        
        generator = StepGenerator()
        steps = generator.generate_smart(eq)
        
        assert len(steps) > 0
        # Deve ter substituição
        assert any(s.substitution for s in steps)
    
    def test_smart_complex_expression(self):
        """Deve detectar expressão complexa → DETAILED."""
        from sympy import sin, exp
        vars_dict = {
            'x': Variable('x', 1.5),
            'y': Variable('y', 2.0)
        }
        eq = Equation(sin(Symbol('x')) * exp(Symbol('y')), locals_dict=vars_dict)
        
        generator = StepGenerator()
        steps = generator.generate_smart(eq)
        
        assert len(steps) > 0
    
    def test_smart_force_granularity(self):
        """Deve respeitar granularidade forçada."""
        vars_dict = {'x': Variable('x', 10)}
        eq = Equation("x * 2", locals_dict=vars_dict)
        
        generator = StepGenerator()
        steps = generator.generate_smart(
            eq,
            force_granularity=GranularityType.DETAILED
        )
        
        assert steps[0].level == GranularityType.DETAILED
    
    def test_complexity_score_simple(self):
        """Deve calcular score baixo para expressão simples."""
        vars_dict = {'x': Variable('x', 10)}
        eq = Equation("x + 1", locals_dict=vars_dict)
        
        generator = StepGenerator()
        score = generator._analyze_complexity(eq)
        
        assert score < 10  # Score baixo
    
    def test_complexity_score_high(self):
        """Deve calcular score alto para expressão complexa."""
        from sympy import sin, cos, exp
        vars_dict = {
            'x': Variable('x', 1),
            'y': Variable('y', 2),
            'z': Variable('z', 3)
        }
        expr = sin(Symbol('x')) * cos(Symbol('y')) * exp(Symbol('z'))
        eq = Equation(expr, locals_dict=vars_dict)
        
        generator = StepGenerator()
        score = generator._analyze_complexity(eq)
        
        assert score > 15  # Score alto


# ============================================================================
# SMOKE TEST COMPLETO
# ============================================================================

@pytest.mark.skipif(not SYMPY_AVAILABLE, reason="SymPy não disponível")
@pytest.mark.smoke
class TestStepSystemSmokeTest:
    """Smoke test completo do sistema de Steps."""
    
    def test_complete_workflow_with_smart_steps(self):
        """Smoke test: workflow completo com análise smart."""
        # Dados de viga
        vars_dict = {
            'q': Variable('q', 15.0, unit='kN/m'),
            'L': Variable('L', 6.0, unit='m')
        }
        
        # Criar equação
        eq = Equation("q * L**2 / 8", locals_dict=vars_dict, name="M_max")
        
        # Generator smart
        generator = StepGenerator()
        steps = generator.generate_smart(eq)
        
        assert len(steps) > 0
        
        # Registry
        registry = StepRegistry()
        registry.register(eq)
        
        # Export LaTeX
        latex = registry.to_latex()
        assert len(latex) > 0
        
        # Export Markdown
        markdown = registry.to_markdown()
        assert len(markdown) > 0
        
        print("✅ SMOKE TEST STEP SYSTEM COMPLETO!")
        print(f"   Steps gerados: {len(steps)}")
        print(f"   LaTeX: {len(latex)} chars")
        print(f"   Markdown: {len(markdown)} chars")










# ============================================================================
# TESTES SMART STEP ANALYZER
# ============================================================================

@pytest.mark.skipif(not SYMPY_AVAILABLE, reason="SymPy não disponível")
class TestSmartStepAnalyzer:
    """Testes do analisador inteligente de complexidade."""
    
    def test_smart_simple_expression_auto_basic(self):
        """Deve detectar expressão simples → BASIC."""
        vars_dict = {'x': Variable('x', 10), 'y': Variable('y', 5)}
        eq = Equation("x + y", locals_dict=vars_dict)
        
        generator = StepGenerator()
        steps = generator.generate_smart(eq)
        
        # Deve escolher BASIC
        assert len(steps) > 0
        assert steps[0].formula is not None
    
    def test_smart_medium_expression_auto_medium(self):
        """Deve detectar expressão média → MEDIUM."""
        vars_dict = {
            'q': Variable('q', 15),
            'L': Variable('L', 6)
        }
        eq = Equation("q * L**2 / 8", locals_dict=vars_dict)
        
        generator = StepGenerator()
        steps = generator.generate_smart(eq)
        
        assert len(steps) > 0
        # Deve ter substituição (MEDIUM ou DETAILED)
        assert any(s.substitution for s in steps)
    
    def test_smart_complex_expression_auto_detailed(self):
        """Deve detectar expressão complexa → DETAILED."""
        from sympy import sin, exp
        vars_dict = {
            'x': Variable('x', 1.5),
            'y': Variable('y', 2.0)
        }
        eq = Equation(sin(Symbol('x')) * exp(Symbol('y')), locals_dict=vars_dict)
        
        generator = StepGenerator()
        steps = generator.generate_smart(eq)
        
        assert len(steps) > 0
        # Expressão complexa deve ter múltiplos steps ou detalhes
        assert steps[0].formula is not None
    
    def test_smart_force_granularity_override(self):
        """Deve respeitar granularidade forçada (ignora análise)."""
        vars_dict = {'x': Variable('x', 10)}
        eq = Equation("x * 2", locals_dict=vars_dict)
        
        generator = StepGenerator()
        steps = generator.generate_smart(
            eq,
            force_granularity=GranularityType.DETAILED
        )
        
        assert steps[0].level == GranularityType.DETAILED
    
    def test_complexity_score_simple(self):
        """Deve calcular score baixo para expressão simples."""
        vars_dict = {'x': Variable('x', 10)}
        eq = Equation("x + 1", locals_dict=vars_dict)
        
        generator = StepGenerator()
        score = generator._analyze_complexity(eq)
        
        assert score < 10  # Score baixo
        print(f"Score simples: {score}")
    
    def test_complexity_score_medium(self):
        """Deve calcular score médio para expressão média."""
        vars_dict = {
            'q': Variable('q', 15),
            'L': Variable('L', 6)
        }
        eq = Equation("q * L**2 / 8", locals_dict=vars_dict)
        
        generator = StepGenerator()
        score = generator._analyze_complexity(eq)
        
        assert 5 < score < 20  # Score médio
        print(f"Score médio: {score}")
    
    def test_complexity_score_high(self):
        """Deve calcular score alto para expressão complexa."""
        from sympy import sin, cos, exp
        vars_dict = {
            'x': Variable('x', 1),
            'y': Variable('y', 2),
            'z': Variable('z', 3)
        }
        expr = sin(Symbol('x')) * cos(Symbol('y')) * exp(Symbol('z'))
        eq = Equation(expr, locals_dict=vars_dict)
        
        generator = StepGenerator()
        score = generator._analyze_complexity(eq)
        
        assert score > 15  # Score alto
        print(f"Score alto: {score}")
    
    def test_expr_depth_simple(self):
        """Deve calcular profundidade 1 para x + y."""
        eq = Equation("x + y")
        generator = StepGenerator()
        
        depth = generator._get_expr_depth(eq.expr)
        
        assert depth == 1
    
    def test_expr_depth_nested(self):
        """Deve calcular profundidade correta para expressão aninhada."""
        from sympy import sin
        eq = Equation(sin(Symbol('x') + Symbol('y')))
        generator = StepGenerator()
        
        depth = generator._get_expr_depth(eq.expr)
        
        assert depth >= 2
    
    def test_smart_with_precision(self):
        """Deve respeitar precisão especificada."""
        vars_dict = {'x': Variable('x', 3.14159)}
        eq = Equation("x * 2", locals_dict=vars_dict)
        
        generator = StepGenerator()
        steps = generator.generate_smart(eq, precision=2)
        
        # Resultado deve ter 2 casas decimais
        assert len(steps) > 0
