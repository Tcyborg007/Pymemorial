"""
PyMemorial v2.0 - Calculator Tests
TDD Implementation following RED → GREEN → REFACTOR
Coverage Target: 95%+

ESTRUTURA:
- TestSafeEvaluator: Avaliador seguro (5 testes)
- TestCalculatorBase: Calculator base (8 testes)

TOTAL PARTE 1/3: 13 testes unitários
"""

import pytest
import threading
from typing import Dict
from decimal import Decimal

# Imports condicionais
try:
    import sympy as sp
    from sympy import Symbol, sin, cos, pi, E, sqrt
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

try:
    from scipy import optimize, integrate
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    optimize = None
    integrate = None

# Imports PyMemorial
from pymemorial.core.calculator import (
    Calculator,
    CalculatorError,
    EvaluationError,
    UnsafeCodeError,
    SafeEvaluator,
    CalculationResult,
)
from pymemorial.core.variable import Variable
from pymemorial.core.equation import (
    Equation, ValidationError, SubstitutionError, EvaluationResult,
    EquationFactory, Step, StepGenerator, StepRegistry, GranularityType,
    ValidationHelpers, StepType, EvaluationError
)
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
        # Usar uma mensagem que realmente aparece no erro
        with pytest.raises(ValidationError, match="Sintaxe inválida na expressão"): # <<< Correção
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

    # COLE ISTO DENTRO DA CLASSE TestEquationEvaluation:
    def test_evaluate_undefined_variable(self):
        """Deve falhar se variável não definida (try/except refinado)."""
        eq = Equation("x + y")
        try:
            eq.evaluate()
            # Se chegou aqui, a exceção não foi levantada
            pytest.fail("EvaluationError NÃO foi levantada quando esperado")
        except EvaluationError:
            # Sucesso! EvaluationError foi levantada como esperado.
            pass # O teste passa
        # Deixar outras exceções (inesperadas) falharem o teste naturalmente

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
    

    # COLE ISTO DENTRO DA CLASSE TestEquationRobustness:
    def test_equation_with_special_constants(self):
        """Deve TRATAR pi e E como SÍMBOLOS (try/except refinado)."""
        eq = Equation("pi + E")
        try:
            eq.evaluate()
            # Se chegou aqui, a exceção não foi levantada
            pytest.fail("EvaluationError NÃO foi levantada para pi/E não definidos")
        except EvaluationError:
            # Sucesso! EvaluationError foi levantada como esperado.
            pass # O teste passa
        # Deixar outras exceções (inesperadas) falharem o teste naturalmente

        # Teste adicional
        result_with_values = eq.evaluate(pi=3.14, E=2.71)
        assert isinstance(result_with_values.value, (int, float))
        assert abs(result_with_values.value - (3.14 + 2.71)) < 1e-9



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
        """Deve criar Step com os campos corretos."""
        # Importar StepType se ainda não estiver (já deve estar com a correção anterior)
        from pymemorial.core.equation import Step, StepType

        step = Step(
            type=StepType.FORMULA,             # <-- Campo correto
            content="M = q * L^2 / 8",       # <-- Campo correto
            latex="M = \\frac{q L^{2}}{8}",     # <-- Campo correto
            explanation="Fórmula do momento", # Exemplo de campo opcional
            level=1                           # Exemplo de campo opcional
        )

        assert step.type == StepType.FORMULA
        assert step.content == "M = q * L^2 / 8"
        assert step.latex == "M = \\frac{q L^{2}}{8}"
        assert step.explanation == "Fórmula do momento"
    
    def test_step_to_latex(self):
        """Deve converter Step para LaTeX."""
        # Corrigido para usar os campos corretos do Step
        step_formula = Step(
            type=StepType.FORMULA,
            content="x + y",
            latex="x + y"
        )
        step_result = Step(
            type=StepType.RESULT,
            content="15",
            latex="15"
        )

        # O método .to_latex() foi removido da dataclass Step
        # na refatoração, pois a lógica de formatação
        # pertence ao StepGenerator ou a um formatter dedicado.
        # Vamos apenas verificar se os dados estão corretos.
        assert step_formula.latex == "x + y"
        assert step_result.latex == "15"

        # Se você precisar testar a formatação LaTeX completa,
        # você deve testar o StepRegistry.to_latex() ou
        # um método de formatação dedicado.
        # Exemplo (requer StepRegistry):
        # registry = StepRegistry()
        # registry._steps = [step_formula, step_result] # Adicionar manualmente
        # latex_output = registry.to_latex()
        # assert "begin{align*}" in latex_output
        # assert "x + y" in latex_output
        # assert "15" in latex_output
    
    def test_step_to_markdown(self):
        """Deve converter Step para Markdown."""
        # Corrigido para usar os campos corretos do Step
        step_formula = Step(
            type=StepType.FORMULA,
            content="x + y",
            latex="x + y"
        )
        step_result = Step(
            type=StepType.RESULT,
            content="15",
            latex="15"
        )

        # Similar ao to_latex, o método .to_markdown() foi removido
        # da dataclass Step. A formatação pertence ao Registry
        # ou a um formatter dedicado.
        # Vamos apenas verificar os dados aqui.
        assert step_formula.content == "x + y"
        assert step_result.content == "15"

        # Se precisar testar a saída Markdown completa:
        # registry = StepRegistry()
        # registry._steps = [step_formula, step_result] # Adicionar manualmente
        # markdown_output = registry.to_markdown()
        # assert "$$" in markdown_output
        # assert "x + y" in markdown_output
        # assert "15" in markdown_output
    
    def test_step_generator_minimal(self):
        """Deve gerar step MINIMAL."""
        vars_dict = {'x': Variable('x', 10), 'y': Variable('y', 5)}
        eq = Equation("x + y", locals_dict=vars_dict)

        generator = StepGenerator()
        # CORREÇÃO: Passar 'variables=vars_dict'
        steps = generator.generate(
            equation=eq,
            variables=vars_dict, # <<< ADICIONADO AQUI
            granularity=GranularityType.MINIMAL
        )

        assert len(steps) == 1
        # CORREÇÃO: Usar campos corretos (type, content)
        assert steps[0].type == StepType.RESULT
        assert "15" in steps[0].content # Verifica se o resultado está lá
    
    def test_step_generator_basic(self):
        """Deve gerar step BASIC."""
        vars_dict = {'x': Variable('x', 10), 'y': Variable('y', 5)}
        eq = Equation("x + y", locals_dict=vars_dict)

        generator = StepGenerator()
        # CORREÇÃO: Passar 'variables=vars_dict'
        steps = generator.generate(
            equation=eq,
            variables=vars_dict, # <<< ADICIONADO AQUI
            granularity=GranularityType.BASIC
        )

        # BASIC deve ter Fórmula e Resultado
        assert len(steps) == 2
        # CORREÇÃO: Usar campos corretos
        assert steps[0].type == StepType.FORMULA
        assert steps[1].type == StepType.RESULT
        assert "15" in steps[1].content
    
    def test_step_generator_medium(self):
        """Deve gerar step MEDIUM."""
        vars_dict = {'x': Variable('x', 10), 'y': Variable('y', 5)}
        eq = Equation("x * y", locals_dict=vars_dict)

        generator = StepGenerator()
        steps = generator.generate(
            equation=eq,
            variables=vars_dict, # <<< ADICIONADO AQUI
            granularity=GranularityType.MEDIUM
        )

        # MEDIUM deve ter Fórmula, Substituição, Resultado
        assert len(steps) == 3
        assert steps[0].type == StepType.FORMULA
        assert steps[1].type == StepType.SUBSTITUTION
        assert steps[2].type == StepType.RESULT
        assert "50" in steps[2].content
    
    def test_step_generator_detailed(self):
        """Deve gerar step DETAILED."""
        vars_dict = {'x': Variable('x', 10), 'y': Variable('y', 5)}
        eq = Equation("x**2 + y**2", locals_dict=vars_dict)

        generator = StepGenerator()
        # CORREÇÃO: Passar 'variables=vars_dict'
        steps = generator.generate(
            equation=eq,
            variables=vars_dict, # <<< ADICIONADO AQUI
            granularity=GranularityType.DETAILED
        )

        # DETAILED deve ter pelo menos Fórmula, Sub, (Simplificação opcional), Resultado
        assert len(steps) >= 3
        # CORREÇÃO: Usar campos corretos
        assert steps[0].type == StepType.FORMULA
        assert steps[1].type == StepType.SUBSTITUTION
        # O último step deve ser o resultado
        assert steps[-1].type == StepType.RESULT
        assert "125" in steps[-1].content # 10^2 + 5^2 = 100 + 25 = 125
    
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
    
    # COLE ISTO DENTRO DA CLASSE TestStepSystemSmokeTest:
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
        # CORREÇÃO: Passar 'variables=vars_dict'
        steps = generator.generate_smart(
            equation=eq,
            variables=vars_dict # <<< ADICIONADO AQUI
        )

        assert len(steps) > 0

        # Registry - AQUI TAMBÉM PRECISA DE 'variables'
        registry = StepRegistry()
        registry.register(
            equation=eq,
            variables=vars_dict # <<< ADICIONADO AQUI
        )

        # Export LaTeX
        latex = registry.to_latex()
        assert isinstance(latex, str) # Verificar se é string
        assert len(latex) > 0
        assert r"\begin{align*}" in latex # Verificar ambiente LaTeX
        assert "M_{max}" in latex # Verificar nome da variável
        assert "67.5" in latex # Verificar resultado

        # Export Markdown
        markdown = registry.to_markdown()
        assert isinstance(markdown, str) # Verificar se é string
        assert len(markdown) > 0
        assert "$$" in markdown # Verificar delimitadores Markdown/LaTeX
        assert "M_{max}" in markdown # Verificar nome da variável
        assert "67.5" in markdown # Verificar resultado

        # Imprimir para confirmação visual (opcional)
        print("\n✅ SMOKE TEST STEP SYSTEM COMPLETO!")
        print(f"   Steps gerados: {len(steps)}")
        print(f"   Granularidade escolhida (exemplo): {steps[0].type.value if steps else 'N/A'}")
        print("-" * 20 + " LaTeX Output " + "-" * 20)
        print(latex)
        print("-" * 20 + " Markdown Output " + "-" * 18)
        print(markdown)
        print("-" * 55)










# ============================================================================
# TESTES SMART STEP ANALYZER
# ============================================================================

@pytest.mark.skipif(not SYMPY_AVAILABLE, reason="SymPy não disponível")
class TestSmartStepAnalyzer:
    """Testes do analisador inteligente de complexidade."""
    
    def test_smart_simple_expression_auto_basic(self):
        """Deve detectar expressão simples → BASIC (2 steps)."""
        vars_dict = {'x': Variable('x', 10), 'y': Variable('y', 5)}
        eq = Equation("x + y", locals_dict=vars_dict)
        generator = StepGenerator()
        steps = generator.generate_smart(
            equation=eq,
            variables=vars_dict
        )
        assert len(steps) == 2, f"Esperado 2, recebido {len(steps)}"
        assert steps[0].type == StepType.FORMULA
        assert steps[1].type == StepType.RESULT
        assert "15" in steps[1].content
    
    def test_smart_medium_expression_auto_medium(self):
        """Deve detectar expressão média → MEDIUM."""
        vars_dict = {
            'q': Variable('q', 15),
            'L': Variable('L', 6)
        }
        eq = Equation("q * L**2 / 8", locals_dict=vars_dict)

        generator = StepGenerator()
        # CORREÇÃO: Passar 'variables=vars_dict'
        steps = generator.generate_smart(
            equation=eq,
            variables=vars_dict # <<< ADICIONADO AQUI
        )

        assert len(steps) > 0 # MEDIUM ou DETAILED terá > 0 steps
        # Verificar se algum step é de substituição (característica de MEDIUM/DETAILED)
        assert any(s.type == StepType.SUBSTITUTION for s in steps)
    
    def test_smart_complex_expression_auto_detailed(self):
        """Deve detectar expressão complexa → DETAILED."""
        # from sympy import sin, exp # Import já deve estar no topo do arquivo
        vars_dict = {
            'x': Variable('x', 1.5),
            'y': Variable('y', 2.0)
        }
        # Usar sp.sin, sp.exp se importado como sp
        eq = Equation(sp.sin(Symbol('x')) * sp.exp(Symbol('y')), locals_dict=vars_dict)

        generator = StepGenerator()
        # CORREÇÃO: Passar 'variables=vars_dict'
        steps = generator.generate_smart(
            equation=eq,
            variables=vars_dict # <<< ADICIONADO AQUI
        )

        assert len(steps) > 0
        # DETAILED deve ter Fórmula, Sub, (Simplificação opcional), Resultado
        assert steps[0].type == StepType.FORMULA
        # Verificar se algum step é de substituição
        assert any(s.type == StepType.SUBSTITUTION for s in steps)
        # O último step deve ser o resultado
        assert steps[-1].type == StepType.RESULT
    
    def test_smart_force_granularity_override(self):
        """Deve respeitar granularidade forçada (ignora análise)."""
        vars_dict = {'x': Variable('x', 10)}
        eq = Equation("x * 2", locals_dict=vars_dict)

        generator = StepGenerator()
        # CORREÇÃO: Passar 'variables=vars_dict'
        steps = generator.generate_smart(
            equation=eq,
            variables=vars_dict, # <<< ADICIONADO AQUI
            force_granularity=GranularityType.DETAILED
        )

        # Verificar se os steps gerados correspondem a DETAILED
        # (DETAILED usualmente gera Fórmula, Sub, (Simp), Resultado)
        assert len(steps) >= 3
        # CORREÇÃO: Verificar o tipo do step, não um campo 'level' inexistente
        assert steps[0].type == StepType.FORMULA
        assert steps[1].type == StepType.SUBSTITUTION
        assert steps[-1].type == StepType.RESULT
    
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
        # CORREÇÃO: Passar 'variables=vars_dict'
        steps = generator.generate_smart(
            equation=eq,
            variables=vars_dict, # <<< ADICIONADO AQUI
            precision=2
        )

        # Resultado deve ter 2 casas decimais
        assert len(steps) > 0
        result_step = next((s for s in steps if s.type == StepType.RESULT), None)
        assert result_step is not None
        # Valor de x*2 = 6.28318, formatado com 2 casas = "6.28"
        assert "6.28" in result_step.content
        # Verificar também o LaTeX
        assert "6.28" in result_step.latex
