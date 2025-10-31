# tests/unit/test_editor/test_step_engine.py

import pytest
from unittest.mock import MagicMock, patch, call # Para mocks
import sympy as sp # Para criar expressões SymPy de teste
import numpy as np # Para resultados numéricos

# Importar as classes e enums necessários do PyMemorial
# (Ajuste os imports conforme sua estrutura exata)
from pymemorial.editor.step_engine import (
    HybridStepEngine,
    CalculationStep,
    StepSequence,
    StepEngineError,
    StepLevel, # Importar StepLevel
)
# Linha ~17 em test_step_engine.py (CORREÇÃO FINAL)
# Linha ~17 em test_step_engine.py (CORRIGIDO FINALMENTE!)
from pymemorial.editor.render_modes import (
    RenderMode,
    NaturalWriterConfig as RenderConfig, # <--- CORRETO: Importa a classe certa e renomeia
    GranularityType,
    StepLevel
)
from pymemorial.core.calculator import CalculationResult, Calculator # Mockaremos Calculator
from pymemorial.recognition.ast_parser import PyMemorialASTParser # Mockaremos ASTParser
from pymemorial.symbols import SymbolRegistry, get_registry # Mockaremos SymbolRegistry
from pymemorial.core.variable import Variable # Para testes de contexto
from pymemorial.core.units import DimensionalError # Para testar inferência

# --- Fixtures Globais ---

@pytest.fixture
def mock_calculator(mocker):
    """Fixture para mockar o core.Calculator."""
    mock = mocker.MagicMock(spec=Calculator)

    # Configurar mock.compute para retornar CalculationResult básico
    def mock_compute(expression, variables=None, mode='auto', **kwargs):
        # Simular avaliação básica
        try:
            expr_sympy = sp.sympify(expression)
            subs_dict = {sp.Symbol(k): v.value if isinstance(v, Variable) else v
                         for k, v in variables.items() if isinstance(v, (int, float, Variable))}
            result_val = float(expr_sympy.subs(subs_dict).evalf())
            return CalculationResult(
                value=result_val,
                expression=expression,
                symbolic=expr_sympy,
                unit=None, # Mock simples, unidade será testada separadamente
                metadata={'mode': mode}
            )
        except Exception as e:
            # Simular erro de avaliação
            raise EvaluationError(f"Mock evaluation error: {e}")

    mock.compute.side_effect = mock_compute
    # Mockar simplify (pode ser necessário para DETAILED)
    mock.simplify.side_effect = lambda expr: sp.simplify(expr)

    return mock

@pytest.fixture
def mock_ast_parser(mocker):
    """Fixture para mockar o recognition.ASTParser."""
    mock = mocker.MagicMock(spec=PyMemorialASTParser)
    # Mockar to_latex para retornar LaTeX básico do SymPy (simplificado)
    mock.to_latex.side_effect = lambda expr_str, context=None: sp.latex(sp.sympify(expr_str))
    # Mockar extract_variables
    mock.extract_variables.side_effect = lambda expr_str: [str(s) for s in sp.sympify(expr_str).free_symbols]
    return mock

# tests/unit/test_editor/test_step_engine.py

@pytest.fixture
def mock_symbol_registry(mocker):
    """Fixture para mockar o symbols.SymbolRegistry."""
    # --- Usar autospec=True ---
    mock = mocker.MagicMock(autospec=True, spec=SymbolRegistry) # Usar autospec=True

    # Mockar get_latex (a lógica aqui está ok)
    def mock_get_latex(name):
        if '_' in name:
            base, sub = name.split('_', 1)
            if base in ['gamma', 'alpha', 'beta', 'sigma', 'rho', 'mu']:
                return f"\\{base}_{{{sub}}}"
            return f"{base}_{{{sub}}}"
        if name in ['gamma', 'alpha', 'beta', 'sigma', 'rho', 'mu']:
            return f"\\{name}"
        return name

    # Configurar o side_effect DEPOIS de criar o mock
    mock.get_latex.side_effect = mock_get_latex
    # Adicionar mocks para outros métodos se forem chamados nos testes
    # mock.get_unicode = MagicMock(...)
    # mock.has = MagicMock(...)

    return mock

@pytest.fixture
def default_config():
    """Fixture para a configuração padrão."""
    return RenderConfig()

@pytest.fixture
def engine(mock_calculator, mock_ast_parser, mock_symbol_registry, default_config):
    """Fixture para criar uma instância do HybridStepEngine com mocks."""
    return HybridStepEngine(
        calculator=mock_calculator,
        ast_parser=mock_ast_parser,
        symbol_registry=mock_symbol_registry,
        config=default_config
    )

# --- Testes por Modo de Renderização ---

class TestMinimalMode:
    def test_algebraic_minimal(self, engine):
        """Modo MINIMAL: Apenas Fórmula + Resultado."""
        expr = "M = q * L**2 / 8"
        context = {'q': Variable('q', 15.0), 'L': Variable('L', 6.0)}
        units = {'M': 'kN.m'}
        sequence = engine.generate_steps(expr, context, units, mode=RenderMode.STEPS_MINIMAL)

        assert len(sequence.steps) == 2
        assert sequence.steps[0].level == StepLevel.SYMBOLIC
        assert sequence.steps[0].expression_symbolic == "M = q * L**2 / 8"
        assert sequence.steps[1].level == StepLevel.RESULT
        assert sequence.steps[1].result_value == pytest.approx(67.5)
        assert sequence.steps[1].result_unit == 'kN.m' # Unidade deve ser passada

    def test_numeric_minimal(self, engine, mock_calculator):
        """Modo MINIMAL para resultado numérico."""
        # Simular resultado do Calculator com metadados
        mock_calc_result = CalculationResult(
            value=2.0, expression="solve(x**2-4, x)", symbolic=None, unit=None,
            metadata={'method_type': 'root_finding', 'method': 'brentq', 'variable': 'x', 'converged': True}
        )
        mock_calculator.compute.return_value = mock_calc_result # Sobrescrever mock para este teste

        expr = "x_root = solve(x**2-4, x)"
        context = {}
        units = {}
        sequence = engine.generate_steps(expr, context, units, mode=RenderMode.STEPS_MINIMAL)

        assert len(sequence.steps) == 2
        assert sequence.steps[0].level == StepLevel.SYMBOLIC # Descrição do problema
        assert "Resolver" in sequence.steps[0].description
        assert sequence.steps[1].level == StepLevel.RESULT
        assert sequence.steps[1].result_value == pytest.approx(2.0)
        assert sequence.steps[1].engine_used == 'scipy'

class TestSmartMode:
    def test_algebraic_smart_standard(self, engine):
        """Modo SMART: Fórmula + Substituição + Resultado."""
        expr = "M = q * L**2 / 8"
        context = {'q': Variable('q', 15.0), 'L': Variable('L', 6.0)}
        units = {'M': 'kN.m', 'q': 'kN/m', 'L': 'm'}
        sequence = engine.generate_steps(expr, context, units, mode=RenderMode.STEPS_SMART)

        assert len(sequence.steps) == 3
        assert sequence.steps[0].level == StepLevel.SYMBOLIC
        assert sequence.steps[1].level == StepLevel.SUBSTITUTION
        # Verificar se a substituição tem os números e unidades (mock simplificado)
        assert '15.0' in sequence.steps[1].expression_latex
        assert '6.0' in sequence.steps[1].expression_latex
        assert r'\mathrm{kN/m}' in sequence.steps[1].expression_latex
        assert sequence.steps[2].level == StepLevel.RESULT
        assert sequence.steps[2].result_value == pytest.approx(67.5)
        assert sequence.steps[2].result_unit == 'kN.m'

    def test_algebraic_smart_skip_trivial_substitution(self, engine):
        """Modo SMART: Deve pular substituição trivial (ex: x = 5)."""
        expr = "x = 5.0"
        context = {}
        units = {'x': 'm'}
        # Forçar score baixo para substituição (depende da implementação do score)
        with patch.object(engine, '_calculate_substitution_complexity', return_value=0.1):
             sequence = engine.generate_steps(expr, context, units, mode=RenderMode.STEPS_SMART)

        # Deve ter apenas Fórmula + Resultado (pulou Substituição)
        assert len(sequence.steps) == 2
        assert sequence.steps[0].level == StepLevel.SYMBOLIC
        assert sequence.steps[1].level == StepLevel.RESULT
        assert sequence.steps[1].result_value == 5.0

    def test_numeric_smart(self, engine, mock_calculator):
        """Modo SMART para numérico: Problema + Método + Resultado."""
        mock_calc_result = CalculationResult(
            value=2.0, expression="solve(x**2-4, x)", symbolic=None, unit=None,
            metadata={'method_type': 'root_finding', 'method': 'brentq', 'variable': 'x', 'converged': True, 'tolerance': 1e-6}
        )
        mock_calculator.compute.return_value = mock_calc_result

        expr = "x_root = solve(x**2-4, x)"
        sequence = engine.generate_steps(expr, {}, {}, mode=RenderMode.STEPS_SMART)

        assert len(sequence.steps) == 3
        assert sequence.steps[0].level == StepLevel.SYMBOLIC # Problema
        assert sequence.steps[1].level == StepLevel.EXPLANATION # Método
        assert "brentq" in sequence.steps[1].expression_latex
        assert "1e-06" in sequence.steps[1].expression_latex # Tolerância
        assert sequence.steps[2].level == StepLevel.RESULT # Resultado

class TestDetailedMode:
    def test_algebraic_detailed_includes_intermediate(self, engine):
        """Modo DETAILED: Deve incluir steps intermediários de simplificação."""
        expr = "M = 15.0 * 6.0**2 / 8.0" # Expressão já com números
        context = {}
        units = {'M': 'kN.m'}
        sequence = engine.generate_steps(expr, context, units, mode=RenderMode.STEPS_DETAILED)

        print("\nDetailed Steps:")
        for s in sequence.steps: print(s)

        assert len(sequence.steps) > 3 # Espera-se Fórmula + Subst + Intermediários + Resultado
        assert any(s.level == StepLevel.INTERMEDIATE for s in sequence.steps)
        # Verificar steps específicos esperados (depende da lógica de _generate_intermediate_simplifications)
        assert any("Calcular potências" in s.description for s in sequence.steps)
        assert any("Multiplicar" in s.description or "Simplificar" in s.description for s in sequence.steps) # Simplificar pode cobrir mult/div
        assert sequence.steps[-1].level == StepLevel.RESULT
        assert sequence.steps[-1].result_value == pytest.approx(67.5)

    # Adicionar teste para numérico DETAILED (não deve ser diferente do SMART, pois não há iterações mostradas)

class TestAllMode:
    @pytest.mark.skip(reason="Modo ALL (_generate_arithmetic_steps) ainda em desenvolvimento/refinamento.")
    def test_algebraic_all_shows_each_operation(self, engine):
        """Modo ALL: Deve mostrar CADA operação aritmética."""
        expr = "M = 15.0 * (6.0 + 2.0)**2 / 8.0"
        context = {}
        units = {'M': 'kN.m'}
        sequence = engine.generate_steps(expr, context, units, mode=RenderMode.STEPS_ALL)

        print("\nALL Steps:")
        for s in sequence.steps: print(s)

        assert len(sequence.steps) > 5 # Espera-se Fórmula + Subst + Várias Operações + Resultado
        # Verificar a presença de steps para operações específicas:
        assert any(s.description == "Somar" and "= 8.0" in s.expression_latex for s in sequence.steps), "Deveria mostrar 6+2=8"
        assert any(s.description == "Calcular potência" and r"8.0^{2.0} = 64.0" in s.expression_latex for s in sequence.steps), "Deveria mostrar 8^2=64"
        assert any(s.description == "Multiplicar" and r"15.0 \cdot 64.0 = 960.0" in s.expression_latex for s in sequence.steps), "Deveria mostrar 15*64=960"
        # A divisão pode estar em _generate_arithmetic_steps, precisa verificar a implementação exata
        # assert any(s.description == "Realizar divisão" and r"\frac{960.0}{8.0} = 120.0" in s.expression_latex for s in sequence.steps), "Deveria mostrar 960/8=120"
        assert sequence.steps[-1].level == StepLevel.RESULT
        assert sequence.steps[-1].result_value == pytest.approx(120.0)

    def test_numeric_all_shows_iterations(self, engine, mock_calculator):
        """Modo ALL para numérico: Deve mostrar iterações (se disponíveis)."""
        mock_calc_result = CalculationResult(
            value=2.09455, expression="solve(x**3 - 2*x - 5, x)", symbolic=None, unit=None,
            metadata={
                'method_type': 'root_finding', 'method': 'newton', 'variable': 'x',
                'converged': True, 'tolerance': 1e-6, 'iterations': 3,
                'convergence_history': [2.0, 2.1, 2.094568] # Histórico simulado
            }
        )
        mock_calculator.compute.return_value = mock_calc_result

        expr = "x_sol = solve(x**3 - 2*x - 5, x)"
        sequence = engine.generate_steps(expr, {}, {}, mode=RenderMode.STEPS_ALL)

        print("\nALL Numeric Steps:")
        for s in sequence.steps: print(s)

        assert len(sequence.steps) > 4 # Problema + Método + Iterações + Resultado
        assert any(s.level == StepLevel.INTERMEDIATE and "Iteração 1" in s.description for s in sequence.steps)
        assert any(s.level == StepLevel.INTERMEDIATE and "Iteração 2" in s.description for s in sequence.steps)
        assert any(s.level == StepLevel.INTERMEDIATE and "Iteração 3" in s.description for s in sequence.steps)
        assert sequence.steps[-1].level == StepLevel.RESULT
        assert sequence.steps[-1].result_value == pytest.approx(2.09455)

# --- Testes de Funcionalidades Específicas ---

class TestUnitInference:
    # Mockar calculate_resultant_unit para simular diferentes cenários
    @patch('pymemorial.core.units.UnitRegistry.calculate_resultant_unit')
    def test_infer_unit_algebraic_success(self, mock_calc_unit, engine):
        """Testa se a unidade inferida é usada no step RESULT."""
        mock_calc_unit.return_value = "kN*m" # Simular inferência bem-sucedida

        expr = "M = F * d"
        context = {'F': Variable('F', 10.0, 'kN'), 'd': Variable('d', 2.0, 'm')}
        units = {'F': 'kN', 'd': 'm'} # Passar units originais
        sequence = engine.generate_steps(expr, context, units, mode=RenderMode.STEPS_SMART)

        assert len(sequence.steps) == 3
        assert sequence.steps[2].level == StepLevel.RESULT
        assert sequence.steps[2].result_unit == "kN*m" # Deve usar a unidade inferida!
        mock_calc_unit.assert_called_once() # Verificar se a inferência foi chamada

    @patch('pymemorial.core.units.UnitRegistry.calculate_resultant_unit')
    def test_infer_unit_algebraic_dimensionless(self, mock_calc_unit, engine):
        """Testa caso adimensional."""
        mock_calc_unit.return_value = None # Simular resultado adimensional

        expr = "ratio = L1 / L2"
        context = {'L1': Variable('L1', 10.0, 'm'), 'L2': Variable('L2', 2.0, 'm')}
        units = {'L1': 'm', 'L2': 'm'}
        sequence = engine.generate_steps(expr, context, units, mode=RenderMode.STEPS_SMART)

        assert sequence.steps[2].result_unit is None # Deve ser None para adimensional

    @patch('pymemorial.core.units.UnitRegistry.calculate_resultant_unit')
    def test_infer_unit_algebraic_error(self, mock_calc_unit, engine):
        """Testa tratamento de erro dimensional na inferência."""
        mock_calc_unit.side_effect = DimensionalError("Incompatible units m+kg") # Simular erro

        expr = "Z = L + M" # Expressão dimensionalmente inconsistente
        context = {'L': Variable('L', 10.0, 'm'), 'M': Variable('M', 5.0, 'kg')}
        units = {'L': 'm', 'M': 'kg'}
        sequence = engine.generate_steps(expr, context, units, mode=RenderMode.STEPS_SMART)

        # O step engine não deve quebrar, mas o resultado deve indicar o erro
        assert sequence.steps[2].level == StepLevel.RESULT
        assert sequence.steps[2].result_unit == "Erro Dimensional" # Unidade indica o erro

    def test_infer_unit_numeric_uses_calc_result(self, engine, mock_calculator):
        """Testa se step numérico usa unidade do CalculationResult."""
        mock_calc_result = CalculationResult(
            value=2.0, expression="solve(...)", unit="m", # Unidade veio do Calculator
            metadata={'method_type': 'root_finding', 'converged': True}
        )
        mock_calculator.compute.return_value = mock_calc_result

        expr = "x_root = solve(x**2-4, x)"
        sequence = engine.generate_steps(expr, {}, {'x_root': 'm'}, mode=RenderMode.STEPS_SMART) # Unit para LHS

        assert sequence.steps[2].level == StepLevel.RESULT
        assert sequence.steps[2].result_unit == "m" # Deve usar a unidade do calc_result

class TestErrorHandling:
    def test_calculator_evaluation_error(self, engine, mock_calculator):
        """Testa se erro do Calculator é propagado ou tratado."""
        mock_calculator.compute.side_effect = EvaluationError("Division by zero")

        expr = "M = 10 / 0"
        sequence = engine.generate_steps(expr, {}, {}, mode=RenderMode.STEPS_SMART)

        # Espera-se um step de erro
        assert len(sequence.steps) > 0
        assert sequence.steps[-1].level == StepLevel.EXPLANATION
        assert "Erro no Cálculo" in sequence.steps[-1].description
        assert "Division by zero" in sequence.steps[-1].expression_latex

    def test_invalid_expression_syntax(self, engine):
        """Testa erro de sintaxe na expressão."""
        expr = "M = q * L** / 8" # Erro de sintaxe
        sequence = engine.generate_steps(expr, {}, {}, mode=RenderMode.STEPS_SMART)

        assert len(sequence.steps) > 0
        assert sequence.steps[-1].level == StepLevel.EXPLANATION
        assert "Erro no Cálculo" in sequence.steps[-1].description # Erro do Calculator
        # Ou um step de erro do próprio StepEngine se o parse falhar antes

    def test_missing_variable_in_context(self, engine, mock_calculator):
        """Testa erro quando variável não está no contexto."""
        # Configurar mock para levantar erro se variável faltar
        def compute_with_check(expression, variables=None, **kwargs):
            expr_sympy = sp.sympify(expression)
            missing = [str(s) for s in expr_sympy.free_symbols if s not in variables]
            if missing:
                raise EvaluationError(f"Missing variables: {missing}")
            # ... (avaliação normal) ...
            result_val = float(expr_sympy.subs({sp.Symbol(k):v for k,v in variables.items()}).evalf())
            return CalculationResult(value=result_val, expression=expression, symbolic=expr_sympy)

        mock_calculator.compute.side_effect = compute_with_check

        expr = "M = q * L**2 / 8"
        context = {'q': 15.0} # 'L' está faltando!
        sequence = engine.generate_steps(expr, context, {}, mode=RenderMode.STEPS_SMART)

        assert len(sequence.steps) > 0
        assert sequence.steps[-1].level == StepLevel.EXPLANATION
        assert "Erro no Cálculo" in sequence.steps[-1].description
        assert "Missing variables: ['L']" in sequence.steps[-1].expression_latex

# Adicionar mais classes/testes para:
# - _generate_intermediate_simplifications
# - _generate_arithmetic_steps (quando implementado)
# - _calculate_complexity_score e _calculate_substitution_complexity
# - Casos com funções (sqrt, sin, etc.)
# - Casos com unidades complexas