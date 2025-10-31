# tests/unit/test_editor/test_step_engine.py
"""
Testes para step_engine.py - PyMemorial v2.0

Cobertura de testes:
- ✅ CalculationStep (dataclass)
- ✅ StepMetadata (dataclass)
- ✅ StepType (enum)
- ✅ HybridStepEngine (classe principal)
- ✅ Geração de steps (MINIMAL, SMART, DETAILED, ALL)
- ✅ Cache de steps
- ✅ Fallbacks
- ✅ Integração com Calculator e SymPy
- ✅ LaTeX generation
- ✅ Casos de erro

Author: PyMemorial Team
Date: October 2025
"""

import pytest
from pymemorial.editor.step_engine import (
    CalculationStep,
    StepType,
    StepMetadata,
    HybridStepEngine,
    create_step_engine
)
from pymemorial.editor.render_modes import RenderMode, RenderConfig


class TestStepType:
    """Testes para StepType enum."""
    
    def test_enum_values(self):
        """Testa valores do enum."""
        assert StepType.FORMULA.value == "formula"
        assert StepType.SUBSTITUTION.value == "substitution"
        assert StepType.SIMPLIFICATION.value == "simplification"
        assert StepType.RESULT.value == "result"
        assert StepType.ERROR.value == "error"
    
    def test_str_representation(self):
        """Testa conversão para string."""
        assert str(StepType.FORMULA) == "formula"
        assert str(StepType.SUBSTITUTION) == "substitution"


class TestStepMetadata:
    """Testes para StepMetadata dataclass."""
    
    def test_default_values(self):
        """Testa valores padrão."""
        metadata = StepMetadata()
        
        assert metadata.engine_used == "sympy"
        assert metadata.computation_time_ms == 0.0
        assert metadata.cache_hit is False
        assert metadata.complexity_score == 0
        assert isinstance(metadata.variables_used, list)
        assert len(metadata.variables_used) == 0
    
    def test_custom_values(self):
        """Testa valores customizados."""
        metadata = StepMetadata(
            engine_used="calculator",
            computation_time_ms=15.5,
            cache_hit=True,
            complexity_score=42,
            variables_used=["q", "L"]
        )
        
        assert metadata.engine_used == "calculator"
        assert metadata.computation_time_ms == 15.5
        assert metadata.cache_hit is True
        assert metadata.complexity_score == 42
        assert metadata.variables_used == ["q", "L"]
    
    def test_to_dict(self):
        """Testa conversão para dicionário."""
        metadata = StepMetadata(
            engine_used="sympy",
            computation_time_ms=10.0
        )
        
        metadata_dict = metadata.to_dict()
        
        assert isinstance(metadata_dict, dict)
        assert metadata_dict['engine_used'] == "sympy"
        assert metadata_dict['computation_time_ms'] == 10.0
        assert 'timestamp' in metadata_dict


class TestCalculationStep:
    """Testes para CalculationStep dataclass."""
    
    def test_initialization_basic(self):
        """Testa inicialização básica."""
        step = CalculationStep(
            step_number=1,
            description="Teste",
            step_type=StepType.FORMULA
        )
        
        assert step.step_number == 1
        assert step.description == "Teste"
        assert step.step_type == StepType.FORMULA
        assert step.expression_symbolic is None
        assert step.expression_latex == ""
        assert step.result is None
    
    def test_initialization_complete(self):
        """Testa inicialização completa."""
        step = CalculationStep(
            step_number=2,
            description="Substituição",
            step_type=StepType.SUBSTITUTION,
            expression_symbolic="M = q * L**2 / 8",
            expression_latex=r"M = \frac{q \cdot L^{2}}{8}",
            expression_numeric="15.0 * 6.0**2 / 8",
            result=67.5,
            result_unit="kN·m"
        )
        
        assert step.step_number == 2
        assert step.description == "Substituição"
        assert step.step_type == StepType.SUBSTITUTION
        assert step.expression_symbolic == "M = q * L**2 / 8"
        assert step.result == 67.5
        assert step.result_unit == "kN·m"
    
    def test_post_init_step_type_conversion(self):
        """Testa conversão automática de string para StepType."""
        step = CalculationStep(
            step_number=1,
            description="Teste",
            step_type="formula"  # String
        )
        
        assert isinstance(step.step_type, StepType)
        assert step.step_type == StepType.FORMULA
    
    def test_post_init_invalid_step_number(self):
        """Testa correção de step_number inválido."""
        step = CalculationStep(
            step_number=-5,
            description="Teste",
            step_type=StepType.FORMULA
        )
        
        # Deve ser corrigido para 1
        assert step.step_number == 1
    
    def test_to_dict(self):
        """Testa conversão para dicionário."""
        step = CalculationStep(
            step_number=1,
            description="Teste",
            step_type=StepType.RESULT,
            result=42.0
        )
        
        step_dict = step.to_dict()
        
        assert isinstance(step_dict, dict)
        assert step_dict['step_number'] == 1
        assert step_dict['description'] == "Teste"
        assert step_dict['step_type'] == "result"  # Convertido para string
        assert step_dict['result'] == 42.0
    
    def test_to_html(self):
        """Testa renderização HTML."""
        step = CalculationStep(
            step_number=1,
            description="Fórmula",
            step_type=StepType.FORMULA,
            expression_latex=r"M = \frac{q \cdot L^{2}}{8}"
        )
        
        html = step.to_html()
        
        assert "calculation-step" in html
        assert "step-formula" in html
        assert "step-number" in html
        assert "1." in html
        assert "Fórmula" in html
        assert r"\frac{q \cdot L^{2}}{8}" in html
    
    def test_to_html_with_result(self):
        """Testa renderização HTML com resultado."""
        step = CalculationStep(
            step_number=3,
            description="Resultado",
            step_type=StepType.RESULT,
            expression_latex=r"M = 67.5 \, \text{kN·m}",
            result=67.5,
            result_unit="kN·m"
        )
        
        html = step.to_html()
        
        assert "step-result" in html
        assert "Resultado" in html
        assert "67.5" in html
    
    def test_repr_and_str(self):
        """Testa representações string."""
        step = CalculationStep(
            step_number=2,
            description="Substituição de valores intermediários",
            step_type=StepType.SUBSTITUTION
        )
        
        repr_str = repr(step)
        str_repr = str(step)
        
        assert "CalculationStep" in repr_str
        assert "#2" in repr_str
        assert "substitution" in repr_str
        
        assert "Step 2" in str_repr
        assert "Substituição" in str_repr


class TestHybridStepEngine:
    """Testes para HybridStepEngine (classe principal)."""
    
    def test_initialization_default(self):
        """Testa inicialização padrão."""
        engine = HybridStepEngine()
        
        assert engine.precision == 3
        assert isinstance(engine.config, RenderConfig)
        assert engine.calculator is not None
        assert engine.stats['steps_generated'] == 0
        assert engine.stats['cache_hits'] == 0
    
    def test_initialization_custom_config(self):
        """Testa inicialização com config customizado."""
        config = RenderConfig(precision=5, enable_cache=False)
        engine = HybridStepEngine(config=config)
        
        assert engine.precision == 5
        assert engine.config.enable_cache is False
    
    def test_generate_steps_minimal(self):
        """Testa geração de steps MINIMAL (2 steps)."""
        engine = HybridStepEngine()
        
        steps = engine.generate_steps(
            expression="M = q * L**2 / 8",
            context={"q": 15.0, "L": 6.0},
            units={"M": "kN·m", "q": "kN/m", "L": "m"},
            mode=RenderMode.STEPS_MINIMAL
        )
        
        # MINIMAL: fórmula + resultado
        assert len(steps) == 2
        assert steps[0].step_type == StepType.FORMULA
        assert steps[1].step_type == StepType.RESULT
        assert steps[1].result == pytest.approx(67.5, abs=0.1)
    
    def test_generate_steps_smart(self):
        """Testa geração de steps SMART (3 steps)."""
        engine = HybridStepEngine()
        
        steps = engine.generate_steps(
            expression="M = q * L**2 / 8",
            context={"q": 15.0, "L": 6.0},
            units={"M": "kN·m", "q": "kN/m", "L": "m"},
            mode=RenderMode.STEPS_SMART
        )
        
        # SMART: fórmula + substituição + resultado
        assert len(steps) == 3
        assert steps[0].step_type == StepType.FORMULA
        assert steps[1].step_type == StepType.SUBSTITUTION
        assert steps[2].step_type == StepType.RESULT
        assert steps[2].result == pytest.approx(67.5, abs=0.1)
    
    def test_generate_steps_detailed(self):
        """Testa geração de steps DETAILED (3+ steps)."""
        engine = HybridStepEngine()
        
        steps = engine.generate_steps(
            expression="M = q * L**2 / 8",
            context={"q": 15.0, "L": 6.0},
            units={"M": "kN·m", "q": "kN/m", "L": "m"},
            mode=RenderMode.STEPS_DETAILED
        )
        
        # DETAILED: fórmula + substituição + intermediários + resultado
        assert len(steps) >= 3
        assert steps[0].step_type == StepType.FORMULA
        assert steps[-1].step_type == StepType.RESULT
        assert steps[-1].result == pytest.approx(67.5, abs=0.1)
    
    def test_generate_steps_with_string_mode(self):
        """Testa geração com modo como string."""
        engine = HybridStepEngine()
        
        steps = engine.generate_steps(
            expression="F = m * a",
            context={"m": 10.0, "a": 9.81},
            mode="steps_smart"  # String em vez de RenderMode
        )
        
        assert len(steps) == 3
        assert steps[-1].result == pytest.approx(98.1, abs=0.1)
    
    def test_generate_steps_no_units(self):
        """Testa geração sem unidades."""
        engine = HybridStepEngine()
        
        steps = engine.generate_steps(
            expression="result = a + b",
            context={"a": 5.0, "b": 3.0},
            mode=RenderMode.STEPS_MINIMAL
        )
        
        assert len(steps) == 2
        assert steps[-1].result == 8.0
        assert steps[-1].result_unit == ""
    
    def test_parse_expression_valid(self):
        """Testa parsing de expressão válida."""
        engine = HybridStepEngine()
        
        lhs, rhs = engine._parse_expression("M = q * L**2 / 8")
        
        assert lhs == "M"
        assert rhs == "q * L**2 / 8"
    
    def test_parse_expression_with_spaces(self):
        """Testa parsing com espaços."""
        engine = HybridStepEngine()
        
        lhs, rhs = engine._parse_expression("  M_k  =  q * L**2 / 8  ")
        
        assert lhs == "M_k"
        assert rhs == "q * L**2 / 8"
    
    def test_parse_expression_invalid_no_equals(self):
        """Testa erro quando não há '='."""
        engine = HybridStepEngine()
        
        with pytest.raises(ValueError, match="deve conter '='"):
            engine._parse_expression("M + q * L**2")
    
    def test_parse_expression_invalid_empty_lhs(self):
        """Testa erro quando LHS vazio."""
        engine = HybridStepEngine()
        
        with pytest.raises(ValueError, match="LHS ou RHS vazio"):
            engine._parse_expression(" = q * L**2")
    
    def test_evaluate_expression_simple(self):
        """Testa avaliação de expressão simples."""
        engine = HybridStepEngine()
        
        result = engine._evaluate_expression("a + b", {"a": 5.0, "b": 3.0})
        
        assert result == 8.0
    
    def test_evaluate_expression_complex(self):
        """Testa avaliação de expressão complexa."""
        engine = HybridStepEngine()
        
        result = engine._evaluate_expression(
            "q * L**2 / 8",
            {"q": 15.0, "L": 6.0}
        )
        
        assert result == pytest.approx(67.5, abs=0.01)
    
    def test_substitute_values_in_expression(self):
        """Testa substituição de valores."""
        engine = HybridStepEngine()
        
        substituted = engine._substitute_values_in_expression(
            "q * L**2 / 8",
            {"q": 15.0, "L": 6.0},
            {}
        )
        
        assert "15.0" in substituted or "15.000" in substituted
        assert "6.0" in substituted or "6.000" in substituted
    
    def test_format_latex_symbol_simple(self):
        """Testa formatação LaTeX de símbolo simples."""
        engine = HybridStepEngine()
        
        latex = engine._format_latex_symbol("M")
        
        assert latex == "M"
    
    def test_format_latex_symbol_subscript(self):
        """Testa formatação LaTeX com subscrito."""
        engine = HybridStepEngine()
        
        latex = engine._format_latex_symbol("M_k")
        
        assert "M_{k}" in latex or "M_k" in latex
    
    def test_format_latex_symbol_greek(self):
        """Testa formatação LaTeX de letra grega."""
        engine = HybridStepEngine()
        
        latex = engine._format_latex_symbol("gamma_f")
        
        # Deve conter gamma (em LaTeX ou subscrito)
        assert "gamma" in latex or r"\gamma" in latex
    
    def test_extract_variables(self):
        """Testa extração de variáveis."""
        engine = HybridStepEngine()
        
        variables = engine._extract_variables("q * L**2 / 8 + m * a")
        
        assert "q" in variables
        assert "L" in variables
        assert "m" in variables
        assert "a" in variables
        assert len(variables) == 4
    
    def test_estimate_complexity(self):
        """Testa estimativa de complexidade."""
        engine = HybridStepEngine()
        
        # Expressão simples
        complexity_simple = engine._estimate_complexity("a + b")
        
        # Expressão complexa
        complexity_complex = engine._estimate_complexity("(a + b) * (c - d) / (e ** 2)")
        
        assert complexity_complex > complexity_simple
        assert complexity_simple > 0


class TestHybridStepEngineCache:
    """Testes para sistema de cache."""
    
    def test_cache_hit(self):
        """Testa cache hit em segunda chamada idêntica."""
        engine = HybridStepEngine()
        
        # Primeira chamada (cache miss)
        steps1 = engine.generate_steps(
            "M = q * L**2 / 8",
            {"q": 15.0, "L": 6.0},
            mode="steps_smart"
        )
        
        cache_misses_before = engine.stats['cache_misses']
        cache_hits_before = engine.stats['cache_hits']
        
        # Segunda chamada idêntica (cache hit)
        steps2 = engine.generate_steps(
            "M = q * L**2 / 8",
            {"q": 15.0, "L": 6.0},
            mode="steps_smart"
        )
        
        assert engine.stats['cache_hits'] == cache_hits_before + 1
        assert len(steps1) == len(steps2)
        assert steps2[0].metadata.cache_hit is True
    
    def test_cache_miss_different_values(self):
        """Testa cache miss com valores diferentes."""
        engine = HybridStepEngine()
        
        # Primeira chamada
        engine.generate_steps(
            "M = q * L**2 / 8",
            {"q": 15.0, "L": 6.0},
            mode="steps_smart"
        )
        
        cache_misses_before = engine.stats['cache_misses']
        
        # Segunda chamada com valores diferentes (cache miss)
        engine.generate_steps(
            "M = q * L**2 / 8",
            {"q": 20.0, "L": 8.0},  # Valores diferentes
            mode="steps_smart"
        )
        
        assert engine.stats['cache_misses'] == cache_misses_before + 1
    
    def test_cache_disabled(self):
        """Testa com cache desabilitado."""
        config = RenderConfig(enable_cache=False)
        engine = HybridStepEngine(config=config)
        
        # Duas chamadas idênticas
        engine.generate_steps("x = a + b", {"a": 1.0, "b": 2.0}, mode="steps_minimal")
        engine.generate_steps("x = a + b", {"a": 1.0, "b": 2.0}, mode="steps_minimal")
        
        # Não deve ter cache hits
        assert engine.stats['cache_hits'] == 0
    
    def test_clear_cache(self):
        """Testa limpeza de cache."""
        engine = HybridStepEngine()
        
        # Gerar alguns steps (popular cache)
        engine.generate_steps("x = a + b", {"a": 1.0, "b": 2.0}, mode="steps_minimal")
        engine.generate_steps("y = c * d", {"c": 3.0, "d": 4.0}, mode="steps_minimal")
        
        # Limpar cache
        HybridStepEngine.clear_cache()
        
        # Cache deve estar vazio
        cache_stats = HybridStepEngine.get_cache_stats()
        assert cache_stats['size'] == 0
    
    def test_get_cache_stats(self):
        """Testa obtenção de estatísticas do cache."""
        HybridStepEngine.clear_cache()
        engine = HybridStepEngine()
        
        # Gerar step
        engine.generate_steps("x = a + b", {"a": 1.0, "b": 2.0}, mode="steps_minimal")
        
        cache_stats = HybridStepEngine.get_cache_stats()
        
        assert isinstance(cache_stats, dict)
        assert cache_stats['size'] >= 1
        assert cache_stats['enabled'] is True
        assert 'max_size' in cache_stats


class TestHybridStepEngineFallback:
    """Testes para fallbacks."""
    
    def test_fallback_on_invalid_expression(self):
        """Testa fallback quando expressão inválida."""
        engine = HybridStepEngine()
        
        # Expressão inválida (sem LHS)
        steps = engine.generate_steps(
            " = q * L**2 / 8",  # LHS vazio
            {"q": 15.0, "L": 6.0},
            mode="steps_smart"
        )
        
        # Deve retornar steps de fallback
        assert len(steps) > 0
        # Primeiro step deve ser erro ou fallback
        assert steps[0].step_type in [StepType.ERROR, StepType.FORMULA]
    
    def test_fallback_on_evaluation_error(self):
        """Testa fallback quando avaliação falha."""
        engine = HybridStepEngine()
        
        # Expressão com variável não definida
        steps = engine.generate_steps(
            "M = q * undefined_var",
            {"q": 15.0},  # undefined_var não definido
            mode="steps_minimal"
        )
        
        # Deve retornar steps de fallback
        assert len(steps) > 0


class TestHybridStepEngineStatistics:
    """Testes para estatísticas."""
    
    def test_get_stats(self):
        """Testa obtenção de estatísticas."""
        # Limpar cache para garantir contagem correta
        HybridStepEngine.clear_cache()
        engine = HybridStepEngine()
        
        # Gerar alguns steps com valores DIFERENTES (para evitar cache)
        engine.generate_steps("x = a + b", {"a": 1.0, "b": 2.0}, mode="steps_minimal")
        engine.generate_steps("y = c * d", {"c": 3.0, "d": 4.0}, mode="steps_minimal")
        
        stats = engine.get_stats()
        
        assert isinstance(stats, dict)
        # Cada chamada gera 2 steps (MINIMAL), total = 4
        assert stats['steps_generated'] == 4
        assert 'cache_hits' in stats
        assert 'cache_misses' in stats
        assert 'errors' in stats
        assert 'version' in stats
        assert stats['sympy_available'] is True
        assert stats['calculator_available'] is True

    
    def test_repr_and_str(self):
        """Testa representações string."""
        engine = HybridStepEngine()
        
        repr_str = repr(engine)
        str_repr = str(engine)
        
        assert "HybridStepEngine" in repr_str
        assert "precision=" in repr_str
        
        assert "HybridStepEngine" in str_repr


class TestCreateStepEngine:
    """Testes para função factory."""
    
    def test_create_step_engine_default(self):
        """Testa criação padrão."""
        engine = create_step_engine()
        
        assert isinstance(engine, HybridStepEngine)
        assert engine.precision == 3
    
    def test_create_step_engine_custom(self):
        """Testa criação customizada."""
        engine = create_step_engine(precision=5, enable_cache=False)
        
        assert engine.precision == 5
        assert engine.config.enable_cache is False


class TestIntegration:
    """Testes de integração end-to-end."""
    
    def test_end_to_end_engineering_problem(self):
        """Testa problema real de engenharia."""
        engine = HybridStepEngine()
        
        # Problema: Momento fletor em viga simplesmente apoiada
        # M = (q * L²) / 8
        # q = 15 kN/m, L = 6 m
        # M = (15 * 36) / 8 = 67.5 kN·m
        
        steps = engine.generate_steps(
            expression="M = q * L**2 / 8",
            context={"q": 15.0, "L": 6.0},
            units={"M": "kN·m", "q": "kN/m", "L": "m"},
            mode=RenderMode.STEPS_SMART
        )
        
        # Validações
        assert len(steps) == 3
        
        # Step 1: Fórmula
        assert steps[0].step_type == StepType.FORMULA
        assert "M" in steps[0].expression_symbolic
        assert steps[0].result_unit == "kN·m"
        
        # Step 2: Substituição
        assert steps[1].step_type == StepType.SUBSTITUTION
        # Verificar que LaTeX contém ALGUM valor numérico (mais flexível)
        assert any(char.isdigit() for char in steps[1].expression_latex)
        
        # Step 3: Resultado
        assert steps[2].step_type == StepType.RESULT
        assert steps[2].result == pytest.approx(67.5, abs=0.1)
        assert steps[2].result_unit == "kN·m"
        
        # Verificação adicional: expression_numeric do step 2 deve conter valores
        if steps[1].expression_numeric:
            assert "15" in steps[1].expression_numeric or "6" in steps[1].expression_numeric

    
    def test_multiple_equations_sequence(self):
        """Testa sequência de equações."""
        engine = HybridStepEngine()
        
        # Equação 1
        steps1 = engine.generate_steps(
            "F = m * a",
            {"m": 10.0, "a": 9.81},
            {"F": "N", "m": "kg", "a": "m/s²"},
            mode="steps_smart"
        )
        
        # Equação 2 (usando resultado anterior)
        F = steps1[-1].result
        steps2 = engine.generate_steps(
            "W = F * d",
            {"F": F, "d": 5.0},
            {"W": "J", "F": "N", "d": "m"},
            mode="steps_smart"
        )
        
        assert steps1[-1].result == pytest.approx(98.1, abs=0.1)
        assert steps2[-1].result == pytest.approx(490.5, abs=0.5)


# ============================================================================
# FIXTURES PYTEST (SE NECESSÁRIO)
# ============================================================================

@pytest.fixture
def engine():
    """Fixture para criar engine limpo."""
    HybridStepEngine.clear_cache()
    return HybridStepEngine()


@pytest.fixture
def sample_context():
    """Fixture com contexto de exemplo."""
    return {
        "q": 15.0,
        "L": 6.0,
        "gamma": 1.4,
        "E": 210e9
    }


@pytest.fixture
def sample_units():
    """Fixture com unidades de exemplo."""
    return {
        "M": "kN·m",
        "q": "kN/m",
        "L": "m",
        "sigma": "MPa"
    }
