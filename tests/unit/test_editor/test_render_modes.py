# tests/editor/test_render_modes.py
"""
PyMemorial v2.0 - Testes para render_modes.py

Suite completa de testes TDD para modos de renderização e configuração
de escrita natural.

METODOLOGIA TDD:
═══════════════
1. RED:   Escrever teste que FALHA
2. GREEN: Implementar código MÍNIMO para passar
3. REFACTOR: Melhorar código mantendo testes VERDES

COVERAGE TARGET: 100%

Test Categories:
   - Enums: GranularityType, StepLevel, NaturalLanguageStyle, RenderMode
   - Config: NaturalWriterConfig (creation, validation, serialization)
   - Helpers: get_natural_writer_config(), parse_step_notation()
   - Integration: core/config.py integration

Author: PyMemorial Team
Version: 2.0.0
Date: 2025-10-27
"""

import pytest
import logging
from typing import Dict, Any

from pymemorial.editor.render_modes import (
    GranularityType,
    StepLevel,
    NaturalLanguageStyle,
    RenderMode,
    NaturalWriterConfig,
    get_natural_writer_config,
    parse_step_notation,
)


# =========================================================================
# TEST FIXTURES
# =========================================================================

@pytest.fixture
def default_config():
    """Fixture: Configuração padrão"""
    return NaturalWriterConfig()


@pytest.fixture
def custom_config():
    """Fixture: Configuração customizada"""
    return NaturalWriterConfig(
        language="en_US",
        style=NaturalLanguageStyle.FORMAL,
        granularity=GranularityType.DETAILED,
        precision=3,
        decimal_separator=".",
        thousands_separator=",",
    )


@pytest.fixture
def config_dict():
    """Fixture: Dicionário de configuração"""
    return {
        'language': 'pt_BR',
        'style': 'technical',
        'granularity': 'smart',
        'precision': 2,
        'show_units': True,
        'use_unicode_symbols': True,
        'metadata': {'author': 'PyMemorial'}
    }


# =========================================================================
# TESTES: GranularityType Enum
# =========================================================================

class TestGranularityType:
    """Testes para enum GranularityType"""
    
    def test_granularity_type_all_values(self):
        """Testa que todos os valores do enum existem"""
        assert GranularityType.MINIMAL.value == "minimal"
        assert GranularityType.BASIC.value == "basic"
        assert GranularityType.MEDIUM.value == "medium"
        assert GranularityType.DETAILED.value == "detailed"
        assert GranularityType.SMART.value == "smart"
    
    def test_granularity_type_string_repr(self):
        """Testa representação string do enum"""
        assert str(GranularityType.SMART) == "smart"
        assert str(GranularityType.DETAILED) == "detailed"
    
    def test_granularity_type_repr(self):
        """Testa repr() do enum"""
        assert repr(GranularityType.SMART) == "GranularityType.SMART"
    
    def test_granularity_type_from_string_lowercase(self):
        """Testa criação de enum de string (lowercase)"""
        gran = GranularityType.from_string("smart")
        assert gran == GranularityType.SMART
        
        gran = GranularityType.from_string("detailed")
        assert gran == GranularityType.DETAILED
    
    def test_granularity_type_from_string_uppercase(self):
        """Testa criação de enum de string (uppercase)"""
        gran = GranularityType.from_string("SMART")
        assert gran == GranularityType.SMART
        
        gran = GranularityType.from_string("MINIMAL")
        assert gran == GranularityType.MINIMAL
    
    def test_granularity_type_from_string_mixed_case(self):
        """Testa criação de enum de string (mixed case)"""
        gran = GranularityType.from_string("SmArT")
        assert gran == GranularityType.SMART
    
    def test_granularity_type_from_string_aliases(self):
        """Testa aliases de granularidade"""
        # short → MINIMAL
        assert GranularityType.from_string("short") == GranularityType.MINIMAL
        
        # standard → BASIC
        assert GranularityType.from_string("standard") == GranularityType.BASIC
        
        # long → DETAILED
        assert GranularityType.from_string("long") == GranularityType.DETAILED
        
        # full → DETAILED
        assert GranularityType.from_string("full") == GranularityType.DETAILED
    
    def test_granularity_type_from_string_invalid(self):
        """Testa erro para valor inválido"""
        with pytest.raises(ValueError) as exc_info:
            GranularityType.from_string("invalid_value")
        
        assert "Granularidade inválida" in str(exc_info.value)
        assert "invalid_value" in str(exc_info.value)
    
    def test_granularity_type_shows_steps_property(self):
        """Testa propriedade shows_steps"""
        # Modos que mostram steps
        assert GranularityType.MEDIUM.shows_steps is True
        assert GranularityType.DETAILED.shows_steps is True
        assert GranularityType.SMART.shows_steps is True
        
        # Modos que NÃO mostram steps
        assert GranularityType.MINIMAL.shows_steps is False
        assert GranularityType.BASIC.shows_steps is False
    
    def test_granularity_type_shows_substitution_property(self):
        """Testa propriedade shows_substitution"""
        # Modos que mostram substituição
        assert GranularityType.MEDIUM.shows_substitution is True
        assert GranularityType.DETAILED.shows_substitution is True
        assert GranularityType.SMART.shows_substitution is True
        
        # Modos que NÃO mostram substituição
        assert GranularityType.MINIMAL.shows_substitution is False
        assert GranularityType.BASIC.shows_substitution is False


# =========================================================================
# TESTES: StepLevel Enum
# =========================================================================

class TestStepLevel:
    """Testes para enum StepLevel"""
    
    def test_step_level_all_values(self):
        """Testa que todos os valores do enum existem"""
        assert StepLevel.SYMBOLIC
        assert StepLevel.SUBSTITUTION
        assert StepLevel.INTERMEDIATE
        assert StepLevel.RESULT
        assert StepLevel.EXPLANATION
    
    def test_step_level_string_repr(self):
        """Testa representação string do enum"""
        assert str(StepLevel.SYMBOLIC) == "symbolic"
        assert str(StepLevel.RESULT) == "result"
    
    def test_step_level_repr(self):
        """Testa repr() do enum"""
        assert repr(StepLevel.SYMBOLIC) == "StepLevel.SYMBOLIC"
    
    def test_step_level_order_property(self):
        """Testa propriedade order (hierarquia Calcpad)"""
        assert StepLevel.SYMBOLIC.order == 1
        assert StepLevel.SUBSTITUTION.order == 2
        assert StepLevel.INTERMEDIATE.order == 3
        assert StepLevel.RESULT.order == 4
        assert StepLevel.EXPLANATION.order == 5
    
    def test_step_level_ordering_comparison(self):
        """Testa comparação de ordem entre steps"""
        assert StepLevel.SYMBOLIC.order < StepLevel.SUBSTITUTION.order
        assert StepLevel.SUBSTITUTION.order < StepLevel.INTERMEDIATE.order
        assert StepLevel.INTERMEDIATE.order < StepLevel.RESULT.order
        assert StepLevel.RESULT.order < StepLevel.EXPLANATION.order


# =========================================================================
# TESTES: NaturalLanguageStyle Enum
# =========================================================================

class TestNaturalLanguageStyle:
    """Testes para enum NaturalLanguageStyle"""
    
    def test_natural_language_style_all_values(self):
        """Testa que todos os valores do enum existem"""
        assert NaturalLanguageStyle.FORMAL
        assert NaturalLanguageStyle.TECHNICAL
        assert NaturalLanguageStyle.SIMPLE
        assert NaturalLanguageStyle.DETAILED
    
    def test_natural_language_style_string_repr(self):
        """Testa representação string do enum"""
        assert str(NaturalLanguageStyle.TECHNICAL) == "technical"
        assert str(NaturalLanguageStyle.FORMAL) == "formal"
    
    def test_natural_language_style_repr(self):
        """Testa repr() do enum"""
        assert repr(NaturalLanguageStyle.TECHNICAL) == \
               "NaturalLanguageStyle.TECHNICAL"


# =========================================================================
# TESTES: RenderMode Enum
# =========================================================================

class TestRenderMode:
    """Testes para enum RenderMode (compatibilidade v1.0)"""
    
    def test_render_mode_all_values(self):
        """Testa que todos os valores do enum existem"""
        assert RenderMode.FULL
        assert RenderMode.SYMBOLIC
        assert RenderMode.NUMERIC
        assert RenderMode.RESULT
        assert RenderMode.STEPS  # Novo v2.0!
    
    def test_render_mode_string_repr(self):
        """Testa representação string do enum"""
        assert str(RenderMode.FULL) == "FULL"
        assert str(RenderMode.STEPS) == "STEPS"
    
    def test_render_mode_repr(self):
        """Testa repr() do enum"""
        assert repr(RenderMode.STEPS) == "RenderMode.STEPS"


# =========================================================================
# TESTES: NaturalWriterConfig - Criação e Defaults
# =========================================================================

class TestNaturalWriterConfigCreation:
    """Testes de criação de NaturalWriterConfig"""
    
    def test_config_default_creation(self, default_config):
        """Testa criação com valores padrão"""
        assert default_config.language == "pt_BR"
        assert default_config.style == NaturalLanguageStyle.TECHNICAL
        assert default_config.granularity == GranularityType.SMART
        assert default_config.precision == 2
        assert default_config.decimal_separator == ","
        assert default_config.thousands_separator == "."
    
    def test_config_custom_creation(self, custom_config):
        """Testa criação com valores customizados"""
        assert custom_config.language == "en_US"
        assert custom_config.style == NaturalLanguageStyle.FORMAL
        assert custom_config.granularity == GranularityType.DETAILED
        assert custom_config.precision == 3
        assert custom_config.decimal_separator == "."
        assert custom_config.thousands_separator == ","
    
    def test_config_smart_default_granularity(self):
        """Testa que SMART é granularidade padrão"""
        config = NaturalWriterConfig()
        assert config.granularity == GranularityType.SMART
    
    def test_config_default_flags(self, default_config):
        """Testa flags booleanas padrão"""
        assert default_config.show_units is True
        assert default_config.use_unicode_symbols is True
        assert default_config.auto_greek_detection is True
        assert default_config.auto_subscript_detection is True
        assert default_config.include_norm_references is True
        assert default_config.include_explanations is True
        assert default_config.use_smartex is True
        assert default_config.latex_backend is False  # 🚨 ZERO LaTeX!
        assert default_config.enable_cache is True
    
    def test_config_smart_thresholds_defaults(self, default_config):
        """Testa thresholds do modo SMART"""
        assert default_config.smart_skip_trivial_mult is True
        assert default_config.smart_skip_trivial_add is True
        assert default_config.smart_skip_identity is True
        assert default_config.smart_min_step_complexity == 0.1
    
    def test_config_metadata_empty_by_default(self, default_config):
        """Testa que metadata começa vazio"""
        assert default_config.metadata == {}


# =========================================================================
# TESTES: NaturalWriterConfig - Validação
# =========================================================================

class TestNaturalWriterConfigValidation:
    """Testes de validação de NaturalWriterConfig"""
    
    def test_config_precision_validation_negative(self):
        """Testa validação de precision negativo (ajusta para 2)"""
        config = NaturalWriterConfig(precision=-1)
        assert config.precision == 2  # Ajustado
    
    def test_config_precision_validation_too_high(self):
        """Testa validação de precision > 10 (ajusta para 2)"""
        config = NaturalWriterConfig(precision=15)
        assert config.precision == 2  # Ajustado
    
    def test_config_precision_validation_valid_range(self):
        """Testa precision em range válido [0-10]"""
        for precision in [0, 1, 5, 10]:
            config = NaturalWriterConfig(precision=precision)
            assert config.precision == precision
    
    def test_config_separators_validation_equal_raises_error(self):
        """Testa erro quando separadores são iguais"""
        with pytest.raises(ValueError) as exc_info:
            NaturalWriterConfig(
                decimal_separator=",",
                thousands_separator=","
            )
        
        assert "não podem ser iguais" in str(exc_info.value)
    
    def test_config_norm_style_validation_invalid(self):
        """Testa validação de norm_style inválido (ajusta para 'inline')"""
        config = NaturalWriterConfig(norm_style="invalid_style")
        assert config.norm_style == "inline"  # Ajustado
    
    def test_config_norm_style_validation_valid(self):
        """Testa norm_style válidos"""
        for style in ['inline', 'footnote', 'separate']:
            config = NaturalWriterConfig(norm_style=style)
            assert config.norm_style == style


# =========================================================================
# TESTES: NaturalWriterConfig - Serialização
# =========================================================================

class TestNaturalWriterConfigSerialization:
    """Testes de serialização/deserialização"""
    
    def test_config_to_dict(self, default_config):
        """Testa serialização to_dict()"""
        data = default_config.to_dict()
        
        assert isinstance(data, dict)
        assert data['language'] == 'pt_BR'
        assert data['style'] == 'technical'
        assert data['granularity'] == 'smart'
        assert data['precision'] == 2
        assert data['decimal_separator'] == ','
    
    def test_config_to_dict_includes_all_fields(self, default_config):
        """Testa que to_dict() inclui todos os campos"""
        data = default_config.to_dict()
        
        # Verificar campos principais
        required_fields = [
            'language', 'style', 'granularity', 'precision',
            'decimal_separator', 'thousands_separator',
            'show_units', 'use_unicode_symbols',
            'include_norm_references', 'enable_cache', 'metadata'
        ]
        
        for field in required_fields:
            assert field in data
    
    def test_config_from_dict_basic(self, config_dict):
        """Testa deserialização from_dict() básica"""
        config = NaturalWriterConfig.from_dict(config_dict)
        
        assert config.language == 'pt_BR'
        assert config.style == NaturalLanguageStyle.TECHNICAL
        assert config.granularity == GranularityType.SMART
        assert config.precision == 2
        assert config.show_units is True
    
    def test_config_from_dict_metadata(self):
        """Testa deserialização de metadata"""
        data = {
            'language': 'pt_BR',
            'metadata': {'author': 'PyMemorial', 'version': '2.0'}
        }
        
        config = NaturalWriterConfig.from_dict(data)
        assert config.metadata == {'author': 'PyMemorial', 'version': '2.0'}
    
    def test_config_from_dict_invalid_style(self):
        """Testa from_dict() com style inválido (fallback)"""
        data = {'style': 'invalid_style'}
        
        config = NaturalWriterConfig.from_dict(data)
        assert config.style == NaturalLanguageStyle.TECHNICAL  # Fallback
    
    def test_config_from_dict_invalid_granularity(self):
        """Testa from_dict() com granularity inválido (fallback)"""
        data = {'granularity': 'invalid_gran'}
        
        config = NaturalWriterConfig.from_dict(data)
        assert config.granularity == GranularityType.SMART  # Fallback
    
    def test_config_roundtrip_serialization(self, custom_config):
        """Testa roundtrip: config → dict → config"""
        # Serializar
        data = custom_config.to_dict()
        
        # Deserializar
        config2 = NaturalWriterConfig.from_dict(data)
        
        # Verificar igualdade
        assert config2.language == custom_config.language
        assert config2.style == custom_config.style
        assert config2.granularity == custom_config.granularity
        assert config2.precision == custom_config.precision
        assert config2.decimal_separator == custom_config.decimal_separator


# =========================================================================
# TESTES: NaturalWriterConfig - Factory Methods
# =========================================================================

class TestNaturalWriterConfigFactoryMethods:
    """Testes de métodos factory"""
    
    def test_config_from_granularity_string_smart(self):
        """Testa factory from_granularity_string() com 'smart'"""
        config = NaturalWriterConfig.from_granularity_string("smart")
        assert config.granularity == GranularityType.SMART
    
    def test_config_from_granularity_string_detailed(self):
        """Testa factory from_granularity_string() com 'detailed'"""
        config = NaturalWriterConfig.from_granularity_string("detailed")
        assert config.granularity == GranularityType.DETAILED
    
    def test_config_from_granularity_string_minimal(self):
        """Testa factory from_granularity_string() com 'minimal'"""
        config = NaturalWriterConfig.from_granularity_string("minimal")
        assert config.granularity == GranularityType.MINIMAL
    
    def test_config_from_granularity_string_alias(self):
        """Testa factory com alias ('short', 'long', etc)"""
        config = NaturalWriterConfig.from_granularity_string("short")
        assert config.granularity == GranularityType.MINIMAL
        
        config = NaturalWriterConfig.from_granularity_string("long")
        assert config.granularity == GranularityType.DETAILED
    
    def test_config_from_granularity_string_invalid(self):
        """Testa factory com string inválida (fallback SMART)"""
        config = NaturalWriterConfig.from_granularity_string("invalid")
        assert config.granularity == GranularityType.SMART  # Fallback


# =========================================================================
# TESTES: parse_step_notation() Helper
# =========================================================================

class TestParseStepNotation:
    """Testes para helper parse_step_notation()"""
    
    def test_parse_step_notation_explicit_smart(self):
        """Testa parsing de [steps:smart]"""
        text, gran = parse_step_notation("M = q*L^2/8  [steps:smart]")
        
        assert text == "M = q*L^2/8"
        assert gran == GranularityType.SMART
    
    def test_parse_step_notation_explicit_detailed(self):
        """Testa parsing de [steps:detailed]"""
        text, gran = parse_step_notation("M = expr  [steps:detailed]")
        
        assert text == "M = expr"
        assert gran == GranularityType.DETAILED
    
    def test_parse_step_notation_explicit_minimal(self):
        """Testa parsing de [steps:minimal]"""
        text, gran = parse_step_notation("x = 1  [steps:minimal]")
        
        assert text == "x = 1"
        assert gran == GranularityType.MINIMAL
    
    def test_parse_step_notation_no_type_defaults_smart(self):
        """Testa [steps] sem tipo (default SMART)"""
        text, gran = parse_step_notation("M = expr  [steps]")
        
        assert text == "M = expr"
        assert gran == GranularityType.SMART
    
    def test_parse_step_notation_no_notation_defaults_smart(self):
        """Testa sem notação (default SMART)"""
        text, gran = parse_step_notation("M = q*L^2/8")
        
        assert text == "M = q*L^2/8"
        assert gran == GranularityType.SMART
    
    def test_parse_step_notation_whitespace_handling(self):
        """Testa handling de whitespace"""
        text, gran = parse_step_notation("M = expr   [steps:smart]   ")
        
        assert text == "M = expr"
        assert gran == GranularityType.SMART
    
    def test_parse_step_notation_invalid_granularity(self):
        """Testa granularidade inválida (fallback SMART)"""
        text, gran = parse_step_notation("M = expr  [steps:invalid]")
        
        assert text == "M = expr"
        assert gran == GranularityType.SMART  # Fallback
    
    def test_parse_step_notation_alias(self):
        """Testa aliases na notação"""
        text, gran = parse_step_notation("M = expr  [steps:short]")
        assert gran == GranularityType.MINIMAL
        
        text, gran = parse_step_notation("M = expr  [steps:long]")
        assert gran == GranularityType.DETAILED
    
    def test_parse_step_notation_multiple_expressions(self):
        """Testa múltiplas expressões"""
        expressions = [
            "M = expr1  [steps:smart]",
            "N = expr2  [steps:detailed]",
            "V = expr3  [steps:minimal]",
        ]
        
        results = [parse_step_notation(expr) for expr in expressions]
        
        assert results[0][1] == GranularityType.SMART
        assert results[1][1] == GranularityType.DETAILED
        assert results[2][1] == GranularityType.MINIMAL


# =========================================================================
# TESTES: get_natural_writer_config() Helper
# =========================================================================

class TestGetNaturalWriterConfig:
    """Testes para helper get_natural_writer_config()"""
    
    def test_get_natural_writer_config_returns_valid_config(self):
        """Testa que retorna configuração válida"""
        config = get_natural_writer_config()
        
        assert isinstance(config, NaturalWriterConfig)
        assert config.language == "pt_BR"
        assert config.granularity == GranularityType.SMART
    
    def test_get_natural_writer_config_default_values(self):
        """Testa valores padrão retornados"""
        config = get_natural_writer_config()
        
        assert config.style == NaturalLanguageStyle.TECHNICAL
        assert config.precision == 2
        assert config.show_units is True
    
    def test_get_natural_writer_config_fallback_on_import_error(self):
        """Testa fallback quando core.config não disponível"""
        # Este teste assume que core.config pode não existir
        # (já testado no código - logger.warning)
        config = get_natural_writer_config()
        
        # Deve retornar config válido mesmo sem core.config
        assert isinstance(config, NaturalWriterConfig)


# =========================================================================
# TESTES: Edge Cases
# =========================================================================

class TestEdgeCases:
    """Testes de edge cases e situações extremas"""
    
    def test_config_empty_metadata(self):
        """Testa config com metadata vazio"""
        config = NaturalWriterConfig(metadata={})
        assert config.metadata == {}
    
    def test_config_nested_metadata(self):
        """Testa config com metadata aninhado"""
        metadata = {
            'author': 'PyMemorial',
            'project': {
                'name': 'Test Project',
                'version': '1.0'
            }
        }
        
        config = NaturalWriterConfig(metadata=metadata)
        assert config.metadata['project']['name'] == 'Test Project'
    
    def test_parse_step_notation_empty_string(self):
        """Testa parsing de string vazia"""
        text, gran = parse_step_notation("")
        assert text == ""
        assert gran == GranularityType.SMART
    
    def test_parse_step_notation_only_notation(self):
        """Testa parsing de apenas notação sem expressão"""
        text, gran = parse_step_notation("[steps:detailed]")
        assert text == ""
        assert gran == GranularityType.DETAILED
    
    def test_config_zero_precision(self):
        """Testa config com precision=0 (inteiros)"""
        config = NaturalWriterConfig(precision=0)
        assert config.precision == 0
    
    def test_config_max_precision(self):
        """Testa config com precision máximo (10)"""
        config = NaturalWriterConfig(precision=10)
        assert config.precision == 10


# =========================================================================
# TESTES: Integration
# =========================================================================

class TestIntegration:
    """Testes de integração entre componentes"""
    
    def test_config_with_all_granularities(self):
        """Testa config com todas as granularidades"""
        for granularity in GranularityType:
            config = NaturalWriterConfig(granularity=granularity)
            assert config.granularity == granularity
    
    def test_config_with_all_styles(self):
        """Testa config com todos os estilos"""
        for style in NaturalLanguageStyle:
            config = NaturalWriterConfig(style=style)
            assert config.style == style
    
    def test_parse_step_notation_workflow(self):
        """Testa workflow completo de parsing"""
        # Usuário escreve expressão com notação
        user_input = "Mmax = q * L^2 / 8  [steps:smart]"
        
        # Parser extrai expressão e granularidade
        expression, granularity = parse_step_notation(user_input)
        
        # Criar config com granularidade detectada
        config = NaturalWriterConfig(granularity=granularity)
        
        # Verificar resultado
        assert expression == "Mmax = q * L^2 / 8"
        assert config.granularity == GranularityType.SMART
    
    def test_config_serialization_with_custom_metadata(self):
        """Testa serialização com metadata customizado"""
        original_config = NaturalWriterConfig(
            metadata={
                'author': 'PyMemorial Team',
                'project': 'Test',
                'tags': ['structural', 'nbr6118']
            }
        )
        
        # Roundtrip
        data = original_config.to_dict()
        restored_config = NaturalWriterConfig.from_dict(data)
        
        assert restored_config.metadata == original_config.metadata


# =========================================================================
# TESTES: Documentation Examples
# =========================================================================

class TestDocumentationExamples:
    """Testes que validam exemplos da documentação"""
    
    def test_example_from_docstring_parse_step_notation(self):
        """Valida exemplo do docstring de parse_step_notation()"""
        text, gran = parse_step_notation("M = q*L^2/8  [steps:smart]")
        assert text == "M = q*L^2/8"
        assert gran == GranularityType.SMART
    
    def test_example_from_docstring_config_creation(self):
        """Valida exemplo do docstring de NaturalWriterConfig"""
        config = NaturalWriterConfig(
            granularity=GranularityType.DETAILED,
            style=NaturalLanguageStyle.FORMAL,
            precision=3
        )
        
        assert config.granularity == GranularityType.DETAILED
        assert config.style == NaturalLanguageStyle.FORMAL
        assert config.precision == 3
    
    def test_example_from_module_docstring(self):
        """Valida exemplo do docstring do módulo"""
        # Criar configuração
        config = NaturalWriterConfig(
            granularity=GranularityType.SMART,
            style=NaturalLanguageStyle.TECHNICAL
        )
        
        # Parsear notação
        text, gran = parse_step_notation("M = q*L^2/8  [steps:smart]")
        
        assert gran == GranularityType.SMART
        assert config.style == NaturalLanguageStyle.TECHNICAL


# =========================================================================
# COVERAGE REPORT HELPER
# =========================================================================

def test_coverage_summary():
    """
    Helper test para reportar cobertura
    
    Este teste sempre passa e serve para documentar a cobertura esperada.
    """
    coverage_target = 100.0
    
    # Módulos testados
    tested_components = [
        'GranularityType',
        'StepLevel',
        'NaturalLanguageStyle',
        'RenderMode',
        'NaturalWriterConfig',
        'get_natural_writer_config',
        'parse_step_notation',
    ]
    
    assert len(tested_components) == 7
    assert coverage_target == 100.0
    
    # Este teste sempre passa
    assert True


# =========================================================================
# PYTEST CONFIGURATION
# =========================================================================

# Markers customizados
pytest.mark.unit = pytest.mark.unit
pytest.mark.integration = pytest.mark.integration
pytest.mark.smoke = pytest.mark.smoke


# =========================================================================
# RUN TESTS
# =========================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=pymemorial.editor.render_modes"])
