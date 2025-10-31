# tests/unit/test_editor/test_step_engine.py
"""
PyMemorial v2.0 - Testes para step_engine.py

Suite completa de testes TDD para o motor de steps automáticos (Calcpad-inspired).

METODOLOGIA TDD:
═══════════════
1. RED:   Escrever teste que FALHA
2. GREEN: Implementar código MÍNIMO para passar
3. REFACTOR: Melhorar código mantendo testes VERDES

COVERAGE TARGET: 95%+

Test Categories:
   - Step: Criação, validação, formatação PT-BR, serialização
   - StepSequence: Criação, filtros, renderização
   - StepEngine: Geração, cache, generators, batch
   - Generators: Arithmetic, Trig, Algebraic
   - Integration: Core integration, helpers

Author: PyMemorial Team
Version: 2.0.0
Date: 2025-10-27
"""

import pytest
import logging
from typing import Dict, Any, List
from unittest.mock import Mock, patch, MagicMock

from pymemorial.editor.step_engine import (
    Step,
    StepSequence,
    StepEngine,
    StepGenerator,
    ArithmeticStepGenerator,
    TrigonometricStepGenerator,
    AlgebraicStepGenerator,
    create_step_engine,
    quick_steps,
    get_step_engine_info,
    SYMPY_AVAILABLE,
)

from pymemorial.editor.render_modes import (
    StepLevel,
    GranularityType,
    NaturalWriterConfig,
)


# =========================================================================
# TEST FIXTURES
# =========================================================================

@pytest.fixture
def config():
    """Fixture: NaturalWriterConfig padrão"""
    return NaturalWriterConfig()


@pytest.fixture
def simple_step():
    """Fixture: Step simples"""
    return Step(
        level=StepLevel.SYMBOLIC,
        expression="M = q × L² / 8",
        complexity_score=1.0
    )


@pytest.fixture
def step_with_value():
    """Fixture: Step com valor numérico"""
    return Step(
        level=StepLevel.RESULT,
        expression="M = 67,5 kN⋅m",
        numeric_value=67.5,
        unit="kN⋅m",
        complexity_score=1.0
    )


@pytest.fixture
def simple_sequence():
    """Fixture: StepSequence simples"""
    steps = [
        Step(StepLevel.SYMBOLIC, "M = q × L² / 8", complexity_score=1.0),
        Step(StepLevel.SUBSTITUTION, "M = 15,0 × 6,0² / 8", complexity_score=0.9),
        Step(StepLevel.RESULT, "M = 67,5", numeric_value=67.5, complexity_score=1.0),
    ]
    return StepSequence(
        variable_name="M",
        steps=steps
    )


@pytest.fixture
def step_engine():
    """Fixture: StepEngine"""
    return StepEngine(enable_cache=False)  # Cache desabilitado para testes


@pytest.fixture
def arithmetic_generator():
    """Fixture: ArithmeticStepGenerator"""
    return ArithmeticStepGenerator()


# =========================================================================
# TESTES: Step - Criação e Validação
# =========================================================================

class TestStepCreation:
    """Testes de criação e validação de Step"""
    
    def test_step_creation_minimal(self):
        """Testa criação de Step com campos mínimos"""
        step = Step(
            level=StepLevel.SYMBOLIC,
            expression="x + y"
        )
        
        assert step.level == StepLevel.SYMBOLIC
        assert step.expression == "x + y"
        assert step.complexity_score == 0.5  # Default
    
    def test_step_creation_full(self):
        """Testa criação de Step com todos os campos"""
        step = Step(
            level=StepLevel.RESULT,
            expression="M = 67,5 kN⋅m",
            latex=r"M = 67{,}5~\text{kN}\cdot\text{m}",
            explanation="Resultado final",
            numeric_value=67.5,
            unit="kN⋅m",
            complexity_score=1.0,
            metadata={'norm': 'NBR 6118'}
        )
        
        assert step.level == StepLevel.RESULT
        assert step.numeric_value == 67.5
        assert step.unit == "kN⋅m"
        assert step.complexity_score == 1.0
        assert step.metadata['norm'] == 'NBR 6118'
    
    def test_step_complexity_score_validation_negative(self):
        """Testa validação de complexity_score negativo"""
        step = Step(
            level=StepLevel.SYMBOLIC,
            expression="x",
            complexity_score=-0.5
        )
        
        # Deve ajustar para 0.5
        assert step.complexity_score == 0.5
    
    def test_step_complexity_score_validation_too_high(self):
        """Testa validação de complexity_score > 1"""
        step = Step(
            level=StepLevel.SYMBOLIC,
            expression="x",
            complexity_score=1.5
        )
        
        # Deve ajustar para 0.5
        assert step.complexity_score == 0.5
    
    def test_step_complexity_score_valid_range(self):
        """Testa complexity_score em range válido"""
        for score in [0.0, 0.3, 0.5, 0.8, 1.0]:
            step = Step(
                level=StepLevel.SYMBOLIC,
                expression="x",
                complexity_score=score
            )
            assert step.complexity_score == score


# =========================================================================
# TESTES: Step - Formatação PT-BR
# =========================================================================

class TestStepFormatting:
    """Testes de formatação de Step em texto natural PT-BR"""
    
    def test_step_format_number_ptbr_simple(self, simple_step):
        """Testa formatação de número simples PT-BR"""
        result = simple_step._format_number_ptbr(1234.56)
        assert result == "1.234,56"
    
    def test_step_format_number_ptbr_small(self, simple_step):
        """Testa formatação de número pequeno"""
        result = simple_step._format_number_ptbr(0.00345, precision=4)
        assert result == "0,0035"
    
    def test_step_format_number_ptbr_large(self, simple_step):
        """Testa formatação de número grande"""
        result = simple_step._format_number_ptbr(1234567.89, precision=2)
        assert result == "1.234.567,89"
    
    def test_step_format_number_ptbr_zero(self, simple_step):
        """Testa formatação de zero"""
        result = simple_step._format_number_ptbr(0.0, precision=2)
        assert result == "0,00"
    
    def test_step_to_natural_text_symbolic(self, config):
        """Testa renderização de step SYMBOLIC"""
        step = Step(
            level=StepLevel.SYMBOLIC,
            expression="M = q × L² / 8",
            explanation="Fórmula do momento",
            complexity_score=1.0
        )
        
        text = step.to_natural_text(config)
        assert "Fórmula do momento:" in text
        assert "M = q × L² / 8" in text
    
    def test_step_to_natural_text_substitution(self, config):
        """Testa renderização de step SUBSTITUTION"""
        step = Step(
            level=StepLevel.SUBSTITUTION,
            expression="M = 15,0 × 6,0² / 8",
            complexity_score=0.9
        )
        
        text = step.to_natural_text(config)
        assert "Substituindo os valores:" in text
        assert "M = 15,0 × 6,0² / 8" in text
    
    def test_step_to_natural_text_result(self, config):
        """Testa renderização de step RESULT"""
        step = Step(
            level=StepLevel.RESULT,
            expression="M",
            numeric_value=67.5,
            unit="kN⋅m",
            complexity_score=1.0
        )
        
        text = step.to_natural_text(config)
        assert "67,50" in text  # Formatado PT-BR
        assert "kN⋅m" in text
    
    def test_step_to_natural_text_intermediate(self, config):
        """Testa renderização de step INTERMEDIATE"""
        step = Step(
            level=StepLevel.INTERMEDIATE,
            expression="M = 540,0 / 8",
            complexity_score=0.7
        )
        
        text = step.to_natural_text(config)
        assert "M = 540,0 / 8" in text


# =========================================================================
# TESTES: Step - Serialização
# =========================================================================

class TestStepSerialization:
    """Testes de serialização/deserialização de Step"""
    
    def test_step_to_dict(self, step_with_value):
        """Testa serialização to_dict()"""
        data = step_with_value.to_dict()
        
        assert isinstance(data, dict)
        assert data['level'] == 'result'
        assert data['expression'] == "M = 67,5 kN⋅m"
        assert data['numeric_value'] == 67.5
        assert data['unit'] == "kN⋅m"
    
    def test_step_from_dict_basic(self):
        """Testa deserialização from_dict() básica"""
        data = {
            'level': 'symbolic',
            'expression': 'M = q × L²',
            'complexity_score': 0.8
        }
        
        step = Step.from_dict(data)
        
        assert step.level == StepLevel.SYMBOLIC
        assert step.expression == 'M = q × L²'
        assert step.complexity_score == 0.8
    
    def test_step_from_dict_invalid_level(self):
        """Testa from_dict() com level inválido (fallback)"""
        data = {
            'level': 'invalid_level',
            'expression': 'x'
        }
        
        step = Step.from_dict(data)
        
        # Deve usar fallback SYMBOLIC
        assert step.level == StepLevel.SYMBOLIC
    
    def test_step_roundtrip_serialization(self, step_with_value):
        """Testa roundtrip: step → dict → step"""
        data = step_with_value.to_dict()
        step2 = Step.from_dict(data)
        
        assert step2.level == step_with_value.level
        assert step2.expression == step_with_value.expression
        assert step2.numeric_value == step_with_value.numeric_value
        assert step2.unit == step_with_value.unit


# =========================================================================
# TESTES: StepSequence - Criação
# =========================================================================

class TestStepSequenceCreation:
    """Testes de criação de StepSequence"""
    
    def test_sequence_creation_minimal(self):
        """Testa criação de StepSequence mínima"""
        seq = StepSequence(variable_name="x")
        
        assert seq.variable_name == "x"
        assert len(seq.steps) == 0
    
    def test_sequence_creation_with_steps(self, simple_sequence):
        """Testa criação com steps"""
        assert simple_sequence.variable_name == "M"
        assert len(simple_sequence.steps) == 3
    
    def test_sequence_len(self, simple_sequence):
        """Testa __len__()"""
        assert len(simple_sequence) == 3
    
    def test_sequence_getitem(self, simple_sequence):
        """Testa __getitem__() (acesso por índice)"""
        first_step = simple_sequence[0]
        assert first_step.level == StepLevel.SYMBOLIC
        
        last_step = simple_sequence[-1]
        assert last_step.level == StepLevel.RESULT
    
    def test_sequence_iter(self, simple_sequence):
        """Testa __iter__() (iteração)"""
        levels = [step.level for step in simple_sequence]
        
        assert len(levels) == 3
        assert levels[0] == StepLevel.SYMBOLIC
        assert levels[1] == StepLevel.SUBSTITUTION
        assert levels[2] == StepLevel.RESULT


# =========================================================================
# TESTES: StepSequence - Filtros de Granularidade
# =========================================================================

class TestStepSequenceFiltering:
    """Testes de filtros de granularidade"""
    
    @pytest.fixture
    def full_sequence(self):
        """Fixture: Sequência completa com todos os níveis"""
        steps = [
            Step(StepLevel.SYMBOLIC, "M = q × L² / 8", complexity_score=1.0),
            Step(StepLevel.SUBSTITUTION, "M = 15,0 × 6,0² / 8", complexity_score=0.9),
            Step(StepLevel.INTERMEDIATE, "M = 15,0 × 36,0 / 8", complexity_score=0.7),
            Step(StepLevel.INTERMEDIATE, "M = 540,0 / 8", complexity_score=0.6),
            Step(StepLevel.RESULT, "M = 67,5", complexity_score=1.0),
        ]
        return StepSequence(variable_name="M", steps=steps)
    
    def test_filter_minimal(self, full_sequence):
        """Testa filtro MINIMAL (apenas resultado)"""
        filtered = full_sequence.filter_by_granularity(GranularityType.MINIMAL)
        
        assert len(filtered) == 1
        assert filtered.steps[0].level == StepLevel.RESULT
    
    def test_filter_basic(self, full_sequence):
        """Testa filtro BASIC (fórmula + resultado)"""
        filtered = full_sequence.filter_by_granularity(GranularityType.BASIC)
        
        assert len(filtered) == 2
        assert filtered.steps[0].level == StepLevel.SYMBOLIC
        assert filtered.steps[1].level == StepLevel.RESULT
    
    def test_filter_medium(self, full_sequence):
        """Testa filtro MEDIUM (fórmula + substituição + resultado)"""
        filtered = full_sequence.filter_by_granularity(GranularityType.MEDIUM)
        
        assert len(filtered) == 3
        levels = [s.level for s in filtered.steps]
        assert StepLevel.SYMBOLIC in levels
        assert StepLevel.SUBSTITUTION in levels
        assert StepLevel.RESULT in levels
    
    def test_filter_detailed(self, full_sequence):
        """Testa filtro DETAILED (todos os steps)"""
        filtered = full_sequence.filter_by_granularity(GranularityType.DETAILED)
        
        assert len(filtered) == 5
        # Todos os steps preservados
        assert filtered.steps == full_sequence.steps
    
    def test_filter_smart(self):
        """Testa filtro SMART (omite triviais)"""
        steps = [
            Step(StepLevel.SYMBOLIC, "x", complexity_score=1.0),
            Step(StepLevel.INTERMEDIATE, "1 × x", complexity_score=0.0),  # Trivial
            Step(StepLevel.INTERMEDIATE, "x + 0", complexity_score=0.05), # Trivial
            Step(StepLevel.INTERMEDIATE, "15 × 36", complexity_score=0.8), # Importante
            Step(StepLevel.RESULT, "540", complexity_score=1.0),
        ]
        
        config = NaturalWriterConfig(smart_min_step_complexity=0.1)
        seq = StepSequence(variable_name="x", steps=steps, config=config)
        
        filtered = seq.filter_by_granularity(GranularityType.SMART)
        
        # Deve manter: SYMBOLIC, step importante, RESULT
        assert len(filtered) >= 3
        
        # Triviais devem ser omitidos
        for step in filtered.steps:
            if step.level == StepLevel.INTERMEDIATE:
                assert step.complexity_score >= 0.1


# =========================================================================
# TESTES: StepSequence - Renderização
# =========================================================================

class TestStepSequenceRendering:
    """Testes de renderização de StepSequence"""
    
    def test_sequence_to_natural_text_basic(self, simple_sequence):
        """Testa renderização básica em texto natural"""
        text = simple_sequence.to_natural_text()
        
        assert "M = q × L² / 8" in text
        assert "M = 15,0 × 6,0² / 8" in text
        assert "M = 67,5" in text
    
    def test_sequence_to_natural_text_with_intro(self):
        """Testa renderização com intro"""
        seq = StepSequence(
            variable_name="M",
            steps=[Step(StepLevel.RESULT, "M = 67,5", complexity_score=1.0)],
            intro_text="Cálculo do momento máximo:"
        )
        
        text = seq.to_natural_text()
        assert "Cálculo do momento máximo:" in text
    
    def test_sequence_to_natural_text_with_conclusion(self):
        """Testa renderização com conclusão"""
        seq = StepSequence(
            variable_name="M",
            steps=[Step(StepLevel.RESULT, "M = 67,5", complexity_score=1.0)],
            conclusion_text="Portanto, M = 67,5 kN⋅m."
        )
        
        text = seq.to_natural_text()
        assert "Portanto, M = 67,5 kN⋅m." in text
    
    def test_sequence_to_natural_text_with_norm_reference(self):
        """Testa renderização com referência normativa"""
        config = NaturalWriterConfig(include_norm_references=True)
        seq = StepSequence(
            variable_name="M",
            steps=[Step(StepLevel.RESULT, "M = 67,5", complexity_score=1.0)],
            config=config,
            norm_reference="NBR 6118:2023 item 11.7.1"
        )
        
        text = seq.to_natural_text()
        assert "NBR 6118:2023 item 11.7.1" in text
    
    def test_sequence_to_markdown(self, simple_sequence):
        """Testa renderização to_markdown()"""
        markdown = simple_sequence.to_markdown()
        
        # to_markdown() é alias de to_natural_text()
        assert isinstance(markdown, str)
        assert "M = q × L² / 8" in markdown


# =========================================================================
# TESTES: StepSequence - Serialização
# =========================================================================

class TestStepSequenceSerialization:
    """Testes de serialização de StepSequence"""
    
    def test_sequence_to_dict(self, simple_sequence):
        """Testa serialização to_dict()"""
        data = simple_sequence.to_dict()
        
        assert data['variable_name'] == 'M'
        assert len(data['steps']) == 3
        assert isinstance(data['steps'][0], dict)
    
    def test_sequence_from_dict(self):
        """Testa deserialização from_dict()"""
        data = {
            'variable_name': 'x',
            'steps': [
                {'level': 'symbolic', 'expression': 'x + y', 'complexity_score': 1.0}
            ],
            'intro_text': 'Cálculo:',
            'conclusion_text': 'Resultado'
        }
        
        seq = StepSequence.from_dict(data)
        
        assert seq.variable_name == 'x'
        assert len(seq.steps) == 1
        assert seq.intro_text == 'Cálculo:'
    
    def test_sequence_roundtrip(self, simple_sequence):
        """Testa roundtrip: sequence → dict → sequence"""
        data = simple_sequence.to_dict()
        seq2 = StepSequence.from_dict(data)
        
        assert seq2.variable_name == simple_sequence.variable_name
        assert len(seq2.steps) == len(simple_sequence.steps)


# =========================================================================
# TESTES: StepEngine - Criação e Inicialização
# =========================================================================

class TestStepEngineCreation:
    """Testes de criação e inicialização de StepEngine"""
    
    def test_engine_creation_default(self):
        """Testa criação com defaults"""
        engine = StepEngine()
        
        assert isinstance(engine.config, NaturalWriterConfig)
        assert engine.enable_cache is True
        assert len(engine._generators) >= 3  # 3 generators padrão
    
    def test_engine_creation_with_config(self):
        """Testa criação com config customizado"""
        config = NaturalWriterConfig(granularity=GranularityType.DETAILED)
        engine = StepEngine(config=config)
        
        assert engine.config.granularity == GranularityType.DETAILED
    
    def test_engine_creation_cache_disabled(self):
        """Testa criação com cache desabilitado"""
        engine = StepEngine(enable_cache=False)
        
        assert engine.enable_cache is False
    
    def test_engine_repr(self):
        """Testa __repr__()"""
        engine = StepEngine()
        repr_str = repr(engine)
        
        assert "StepEngine" in repr_str
        assert "generators=" in repr_str
        assert "cache_size=" in repr_str


# =========================================================================
# TESTES: StepEngine - Geração de Steps
# =========================================================================

class TestStepEngineGeneration:
    """Testes de geração de steps"""
    
    @pytest.mark.skipif(not SYMPY_AVAILABLE, reason="SymPy não disponível")
    def test_generate_steps_simple_expression(self, step_engine):
        """Testa geração para expressão simples"""
        expr = "q * L**2 / 8"
        context = {'q': 15.0, 'L': 6.0}
        
        sequence = step_engine.generate_steps(expr, context)
        
        assert isinstance(sequence, StepSequence)
        assert len(sequence) >= 2  # Pelo menos symbolic + result
    
    @pytest.mark.skipif(not SYMPY_AVAILABLE, reason="SymPy não disponível")
    def test_generate_steps_with_variable_name(self, step_engine):
        """Testa geração com nome de variável"""
        expr = "15.0 * 6.0**2 / 8"
        
        sequence = step_engine.generate_steps(
            expr,
            variable_name="M_max"
        )
        
        assert sequence.variable_name == "M_max"
    
    @pytest.mark.skipif(not SYMPY_AVAILABLE, reason="SymPy não disponível")
    def test_generate_steps_with_unit(self, step_engine):
        """Testa geração com unidade"""
        expr = "15.0 * 6.0"
        
        sequence = step_engine.generate_steps(
            expr,
            unit="kN⋅m"
        )
        
        # Step final deve ter unidade
        if sequence.steps:
            final_step = sequence.steps[-1]
            if final_step.level == StepLevel.RESULT:
                assert final_step.unit == "kN⋅m"
    
    @pytest.mark.skipif(not SYMPY_AVAILABLE, reason="SymPy não disponível")
    def test_generate_steps_with_intro_conclusion(self, step_engine):
        """Testa geração com intro e conclusão"""
        expr = "2 + 2"
        
        sequence = step_engine.generate_steps(
            expr,
            intro_text="Cálculo simples:",
            conclusion_text="Portanto, 2 + 2 = 4."
        )
        
        assert sequence.intro_text == "Cálculo simples:"
        assert sequence.conclusion_text == "Portanto, 2 + 2 = 4."
    
    @pytest.mark.skipif(not SYMPY_AVAILABLE, reason="SymPy não disponível")
    def test_generate_steps_with_norm_reference(self, step_engine):
        """Testa geração com referência normativa"""
        expr = "1.4 * 67.5"
        
        sequence = step_engine.generate_steps(
            expr,
            norm_reference="NBR 6118:2023 item 11.7.1"
        )
        
        assert sequence.norm_reference == "NBR 6118:2023 item 11.7.1"
    
    def test_generate_steps_fallback_without_sympy(self, step_engine):
        """Testa fallback quando SymPy não disponível"""
        # Simular SymPy indisponível através de expressão inválida
        # que força fallback
        
        expr = "invalid expression that cannot be parsed"
        context = {}
        
        # Não deve crashar, deve usar fallback
        sequence = step_engine.generate_steps(expr, context)
        
        assert isinstance(sequence, StepSequence)
        # Fallback retorna steps básicos
        assert len(sequence) >= 1


# =========================================================================
# TESTES: StepEngine - Cache
# =========================================================================

class TestStepEngineCache:
    """Testes de cache de steps"""
    
    @pytest.mark.skipif(not SYMPY_AVAILABLE, reason="SymPy não disponível")
    def test_cache_enabled_stores_results(self):
        """Testa que cache armazena resultados"""
        engine = StepEngine(enable_cache=True)
        
        expr = "2 + 2"
        context = {}
        
        # Primeira chamada: computa
        seq1 = engine.generate_steps(expr, context)
        
        # Cache deve ter 1 entrada
        stats = engine.get_cache_stats()
        assert stats['size'] == 1
        
        # Segunda chamada: recupera do cache
        seq2 = engine.generate_steps(expr, context)
        
        # Deve ser o mesmo objeto (cache hit)
        assert seq1 is seq2
    
    @pytest.mark.skipif(not SYMPY_AVAILABLE, reason="SymPy não disponível")
    def test_cache_disabled_recomputes(self):
        """Testa que com cache desabilitado recomputa sempre"""
        engine = StepEngine(enable_cache=False)
        
        expr = "2 + 2"
        
        seq1 = engine.generate_steps(expr)
        seq2 = engine.generate_steps(expr)
        
        # Deve ser objetos diferentes (não cacheado)
        assert seq1 is not seq2
        
        # Cache deve estar vazio
        stats = engine.get_cache_stats()
        assert stats['size'] == 0
    
    @pytest.mark.skipif(not SYMPY_AVAILABLE, reason="SymPy não disponível")
    def test_cache_clear(self):
        """Testa limpeza de cache"""
        engine = StepEngine(enable_cache=True)
        
        # Gerar alguns steps
        engine.generate_steps("1 + 1")
        engine.generate_steps("2 + 2")
        
        assert engine.get_cache_stats()['size'] == 2
        
        # Limpar cache
        engine.clear_cache()
        
        assert engine.get_cache_stats()['size'] == 0
    
    @pytest.mark.skipif(not SYMPY_AVAILABLE, reason="SymPy não disponível")
    def test_cache_key_considers_context(self):
        """Testa que cache diferencia contextos diferentes"""
        engine = StepEngine(enable_cache=True)
        
        expr = "x + y"
        context1 = {'x': 1, 'y': 2}
        context2 = {'x': 3, 'y': 4}
        
        seq1 = engine.generate_steps(expr, context1)
        seq2 = engine.generate_steps(expr, context2)
        
        # Devem ser objetos diferentes (contextos diferentes)
        assert seq1 is not seq2
        
        # Cache deve ter 2 entradas
        assert engine.get_cache_stats()['size'] == 2
    
    @pytest.mark.skipif(not SYMPY_AVAILABLE, reason="SymPy não disponível")
    def test_cache_key_considers_granularity(self):
        """Testa que cache diferencia granularidades diferentes"""
        engine = StepEngine(enable_cache=True)
        
        expr = "2 + 2"
        
        seq1 = engine.generate_steps(expr, granularity=GranularityType.MINIMAL)
        seq2 = engine.generate_steps(expr, granularity=GranularityType.DETAILED)
        
        # Devem ser objetos diferentes (granularidades diferentes)
        assert seq1 is not seq2
        
        # Cache deve ter 2 entradas
        assert engine.get_cache_stats()['size'] == 2


# =========================================================================
# TESTES: StepEngine - Plugin System
# =========================================================================

class TestStepEnginePluginSystem:
    """Testes do sistema de plugins (generators customizados)"""
    
    def test_register_generator(self, step_engine):
        """Testa registro de generator customizado"""
        initial_count = len(step_engine._generators)
        
        # Criar generator mock
        class MockGenerator(StepGenerator):
            priority = 999
            name = "MockGenerator"
            
            def can_handle(self, expr, context):
                return False
            
            def generate(self, expr, context, config):
                return []
        
        mock_gen = MockGenerator()
        step_engine.register_generator(mock_gen)
        
        # Deve ter mais 1 generator
        assert len(step_engine._generators) == initial_count + 1
        
        # Deve estar no início (maior prioridade)
        assert step_engine._generators[0].name == "MockGenerator"
    
    def test_generator_priority_ordering(self):
        """Testa que generators são ordenados por prioridade"""
        engine = StepEngine(enable_cache=False)
        
        # Generators padrão devem estar ordenados
        priorities = [g.priority for g in engine._generators]
        
        # Verificar ordem decrescente
        assert priorities == sorted(priorities, reverse=True)
    
    def test_default_generators_registered(self, step_engine):
        """Testa que generators padrão são registrados"""
        generator_names = [g.name for g in step_engine._generators]
        
        assert "ArithmeticStepGenerator" in generator_names
        assert "TrigonometricStepGenerator" in generator_names
        assert "AlgebraicStepGenerator" in generator_names


# =========================================================================
# TESTES: StepEngine - Batch Processing
# =========================================================================

class TestStepEngineBatch:
    """Testes de processamento em batch"""
    
    @pytest.mark.skipif(not SYMPY_AVAILABLE, reason="SymPy não disponível")
    def test_batch_generate_basic(self, step_engine):
        """Testa geração em batch"""
        expressions = [
            ("2 + 2", {}),
            ("3 * 3", {}),
            ("4 / 2", {}),
        ]
        
        sequences = step_engine.batch_generate(expressions)
        
        assert len(sequences) == 3
        assert all(isinstance(seq, StepSequence) for seq in sequences)
    
    @pytest.mark.skipif(not SYMPY_AVAILABLE, reason="SymPy não disponível")
    def test_batch_generate_with_granularity(self, step_engine):
        """Testa batch com granularidade"""
        expressions = [
            ("2 + 2", {}),
            ("3 * 3", {}),
        ]
        
        sequences = step_engine.batch_generate(
            expressions,
            granularity=GranularityType.MINIMAL
        )
        
        # Todas devem ter apenas 1 step (MINIMAL)
        for seq in sequences:
            assert len(seq) <= 2  # MINIMAL pode ter 1-2 steps
    
    def test_batch_generate_handles_errors(self, step_engine):
        """Testa que batch continua mesmo com erros"""
        expressions = [
            ("2 + 2", {}),
            ("invalid expr that will fail", {}),
            ("3 * 3", {}),
        ]
        
        # Não deve crashar
        sequences = step_engine.batch_generate(expressions)
        
        # Deve ter ao menos 2 sequences (válidas)
        assert len(sequences) >= 2


# =========================================================================
# TESTES: Generators - ArithmeticStepGenerator
# =========================================================================

class TestArithmeticStepGenerator:
    """Testes do ArithmeticStepGenerator"""
    
    @pytest.mark.skipif(not SYMPY_AVAILABLE, reason="SymPy não disponível")
    def test_arithmetic_can_handle_basic_ops(self, arithmetic_generator):
        """Testa detecção de operações aritméticas"""
        assert arithmetic_generator.can_handle("2 + 2", {}) is True
        assert arithmetic_generator.can_handle("3 * 5", {}) is True
        assert arithmetic_generator.can_handle("10 / 2", {}) is True
        assert arithmetic_generator.can_handle("2 ** 3", {}) is True
    
    @pytest.mark.skipif(not SYMPY_AVAILABLE, reason="SymPy não disponível")
    def test_arithmetic_generate_basic(self, arithmetic_generator, config):
        """Testa geração de steps aritméticos"""
        expr = "15.0 * 6.0"
        context = {}
        
        steps = arithmetic_generator.generate(expr, context, config)
        
        assert len(steps) >= 2  # Pelo menos symbolic + result
        assert steps[0].level == StepLevel.SYMBOLIC
        assert steps[-1].level == StepLevel.RESULT
    
    @pytest.mark.skipif(not SYMPY_AVAILABLE, reason="SymPy não disponível")
    def test_arithmetic_complexity_score_trivial(self, arithmetic_generator):
        """Testa detecção de operações triviais"""
        import sympy as sp
        
        # 1 * x é trivial (score baixo)
        before = sp.sympify("1 * x")
        after = sp.sympify("x")
        
        score = arithmetic_generator._calculate_complexity_score(before, after)
        assert score < 0.5  # Trivial
    
    @pytest.mark.skipif(not SYMPY_AVAILABLE, reason="SymPy não disponível")
    def test_arithmetic_complexity_score_important(self, arithmetic_generator):
        """Testa detecção de operações importantes"""
        import sympy as sp
        
        # 15.0 * 36.0 é importante (score alto)
        before = sp.sympify("15.0 * 36.0")
        after = sp.sympify("540.0")
        
        score = arithmetic_generator._calculate_complexity_score(before, after)
        assert score >= 0.7  # Importante


# =========================================================================
# TESTES: Generators - TrigonometricStepGenerator
# =========================================================================

class TestTrigonometricStepGenerator:
    """Testes do TrigonometricStepGenerator"""
    
    @pytest.mark.skipif(not SYMPY_AVAILABLE, reason="SymPy não disponível")
    def test_trig_can_handle_sin_cos_tan(self):
        """Testa detecção de funções trigonométricas"""
        trig_gen = TrigonometricStepGenerator()
        
        assert trig_gen.can_handle("sin(x)", {}) is True
        assert trig_gen.can_handle("cos(x)", {}) is True
        assert trig_gen.can_handle("tan(x)", {}) is True
    
    @pytest.mark.skipif(not SYMPY_AVAILABLE, reason="SymPy não disponível")
    def test_trig_cannot_handle_non_trig(self):
        """Testa que NÃO detecta expressões não-trigonométricas"""
        trig_gen = TrigonometricStepGenerator()
        
        assert trig_gen.can_handle("x + y", {}) is False
        assert trig_gen.can_handle("2 * 3", {}) is False


# =========================================================================
# TESTES: Generators - AlgebraicStepGenerator
# =========================================================================

class TestAlgebraicStepGenerator:
    """Testes do AlgebraicStepGenerator"""
    
    @pytest.mark.skipif(not SYMPY_AVAILABLE, reason="SymPy não disponível")
    def test_algebraic_can_handle_symbols(self):
        """Testa detecção de expressões algébricas"""
        alg_gen = AlgebraicStepGenerator()
        
        assert alg_gen.can_handle("x + y", {}) is True
        assert alg_gen.can_handle("a**2 + b**2", {}) is True
    
    @pytest.mark.skipif(not SYMPY_AVAILABLE, reason="SymPy não disponível")
    def test_algebraic_cannot_handle_pure_numbers(self):
        """Testa que NÃO detecta números puros"""
        alg_gen = AlgebraicStepGenerator()
        
        # Números puros não têm símbolos
        assert alg_gen.can_handle("2 + 2", {}) is False


# =========================================================================
# TESTES: Helper Functions
# =========================================================================

class TestHelperFunctions:
    """Testes de funções helper"""
    
    def test_create_step_engine(self):
        """Testa factory create_step_engine()"""
        engine = create_step_engine()
        
        assert isinstance(engine, StepEngine)
        assert engine.enable_cache is True
    
    def test_create_step_engine_with_config(self):
        """Testa factory com config"""
        config = NaturalWriterConfig(granularity=GranularityType.MINIMAL)
        engine = create_step_engine(config=config)
        
        assert engine.config.granularity == GranularityType.MINIMAL
    
    @pytest.mark.skipif(not SYMPY_AVAILABLE, reason="SymPy não disponível")
    def test_quick_steps_basic(self):
        """Testa helper quick_steps()"""
        text = quick_steps("2 + 2")
        
        assert isinstance(text, str)
        assert len(text) > 0
    
    @pytest.mark.skipif(not SYMPY_AVAILABLE, reason="SymPy não disponível")
    def test_quick_steps_with_context(self):
        """Testa quick_steps() com contexto"""
        text = quick_steps("x + y", {'x': 1, 'y': 2})
        
        assert isinstance(text, str)
    
    def test_get_step_engine_info(self):
        """Testa get_step_engine_info()"""
        info = get_step_engine_info()
        
        assert 'version' in info
        assert 'sympy_available' in info
        assert 'default_generators' in info
        assert 'features' in info


# =========================================================================
# TESTES: Integration
# =========================================================================

class TestStepEngineIntegration:
    """Testes de integração completa"""
    
    @pytest.mark.skipif(not SYMPY_AVAILABLE, reason="SymPy não disponível")
    def test_full_workflow_realistic(self):
        """Testa workflow completo com exemplo realista"""
        engine = StepEngine()
        
        # Exemplo: Cálculo de momento máximo em viga
        expr = "q * L**2 / 8"
        context = {'q': 15.0, 'L': 6.0}
        
        sequence = engine.generate_steps(
            expr,
            context,
            variable_name="M_max",
            intro_text="Cálculo do momento máximo:",
            conclusion_text="Portanto, M_max = 67,5 kN⋅m.",
            unit="kN⋅m"
        )
        
        # Verificações
        assert sequence.variable_name == "M_max"
        assert sequence.intro_text == "Cálculo do momento máximo:"
        assert len(sequence) >= 2
        
        # Renderizar
        text = sequence.to_natural_text()
        assert "M_max" in text or "M" in text
        assert "67" in text  # Resultado aproximado
    
    @pytest.mark.skipif(not SYMPY_AVAILABLE, reason="SymPy não disponível")
    def test_smart_mode_omits_trivials(self):
        """Testa que modo SMART omite steps triviais"""
        config = NaturalWriterConfig(
            granularity=GranularityType.SMART,
            smart_min_step_complexity=0.3
        )
        engine = StepEngine(config=config)
        
        # Expressão com operação trivial (1 * x)
        expr = "1 * 100 + 0"
        
        sequence = engine.generate_steps(expr)
        
        # SMART deve omitir steps triviais
        # (difícil testar exatamente quantos, mas deve ter menos que DETAILED)
        assert len(sequence) <= 4  # Heurística


# =========================================================================
# TESTES: Edge Cases
# =========================================================================

class TestStepEngineEdgeCases:
    """Testes de edge cases"""
    
    def test_generate_steps_empty_expression(self, step_engine):
        """Testa expressão vazia"""
        sequence = step_engine.generate_steps("")
        
        # Não deve crashar
        assert isinstance(sequence, StepSequence)
    
    def test_generate_steps_none_context(self, step_engine):
        """Testa contexto None"""
        sequence = step_engine.generate_steps("2 + 2", context=None)
        
        assert isinstance(sequence, StepSequence)
    
    @pytest.mark.skipif(not SYMPY_AVAILABLE, reason="SymPy não disponível")
    def test_generate_steps_empty_context(self, step_engine):
        """Testa contexto vazio"""
        sequence = step_engine.generate_steps("x + y", context={})
        
        # Deve funcionar (expressão simbólica)
        assert isinstance(sequence, StepSequence)


# =========================================================================
# COVERAGE SUMMARY
# =========================================================================

def test_coverage_summary():
    """
    Helper test para reportar cobertura
    
    Este teste sempre passa e serve para documentar a cobertura esperada.
    """
    coverage_target = 95.0
    
    # Módulos testados
    tested_components = [
        'Step',
        'StepSequence',
        'StepEngine',
        'StepGenerator (abstract)',
        'ArithmeticStepGenerator',
        'TrigonometricStepGenerator',
        'AlgebraicStepGenerator',
        'create_step_engine',
        'quick_steps',
        'get_step_engine_info',
    ]
    
    assert len(tested_components) == 10
    assert coverage_target >= 95.0
    
    # Este teste sempre passa
    assert True


# =========================================================================
# PYTEST CONFIGURATION
# =========================================================================

# Markers customizados
pytest.mark.unit = pytest.mark.unit
pytest.mark.integration = pytest.mark.integration
pytest.mark.slow = pytest.mark.slow


# =========================================================================
# RUN TESTS
# =========================================================================

if __name__ == "__main__":
    pytest.main([
        __file__,
        "-v",
        "--cov=pymemorial.editor.step_engine",
        "--cov-report=html"
    ])
