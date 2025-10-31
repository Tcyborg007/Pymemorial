# tests/unit/test_editor/test_render_modes.py
"""
Testes para render_modes.py - PyMemorial v2.0
"""

import pytest
from pymemorial.editor.render_modes import (
    RenderMode,
    StepGranularity,
    OutputFormat,
    RenderConfig,
    create_render_config,
    get_default_render_config
)


class TestRenderMode:
    """Testes para RenderMode enum."""
    
    def test_enum_values(self):
        """Testa valores do enum."""
        assert RenderMode.STEPS_SMART.value == "steps_smart"
        assert RenderMode.SYMBOLIC.value == "symbolic"
        assert RenderMode.STEPS_MINIMAL.value == "steps_minimal"
    
    def test_is_steps_mode(self):
        """Testa detecção de modo steps."""
        assert RenderMode.STEPS_SMART.is_steps_mode() is True
        assert RenderMode.STEPS_DETAILED.is_steps_mode() is True
        assert RenderMode.SYMBOLIC.is_steps_mode() is False
        assert RenderMode.NUMERIC.is_steps_mode() is False
    
    def test_is_handcalcs_mode(self):
        """Testa detecção de modo handcalcs."""
        assert RenderMode.SYMBOLIC.is_handcalcs_mode() is True
        assert RenderMode.PARAMS.is_handcalcs_mode() is True
        assert RenderMode.STEPS_SMART.is_handcalcs_mode() is False
    
    def test_get_step_count_estimate(self):
        """Testa estimativa de steps."""
        assert RenderMode.STEPS_MINIMAL.get_step_count_estimate() == 2
        assert RenderMode.STEPS_SMART.get_step_count_estimate() == 3
        assert RenderMode.STEPS_DETAILED.get_step_count_estimate() == 7
        assert RenderMode.STEPS_ALL.get_step_count_estimate() == 20
    
    def test_from_string(self):
        """Testa criação a partir de string."""
        assert RenderMode.from_string('smart') == RenderMode.STEPS_SMART
        assert RenderMode.from_string('SYMBOLIC') == RenderMode.SYMBOLIC
        assert RenderMode.from_string('steps_detailed') == RenderMode.STEPS_DETAILED
    
    def test_from_string_invalid(self):
        """Testa erro com string inválida."""
        with pytest.raises(ValueError, match="Modo de renderização inválido"):
            RenderMode.from_string('invalid_mode')
    
    def test_str_and_repr(self):
        """Testa representações string."""
        mode = RenderMode.STEPS_SMART
        assert str(mode) == "steps_smart"
        assert repr(mode) == "RenderMode.STEPS_SMART"


class TestStepGranularity:
    """Testes para StepGranularity enum."""
    
    def test_from_render_mode(self):
        """Testa conversão de RenderMode."""
        assert StepGranularity.from_render_mode(RenderMode.STEPS_MINIMAL) == StepGranularity.MINIMAL
        assert StepGranularity.from_render_mode(RenderMode.STEPS_SMART) == StepGranularity.SMART
        assert StepGranularity.from_render_mode(RenderMode.STEPS_DETAILED) == StepGranularity.DETAILED
        assert StepGranularity.from_render_mode(RenderMode.STEPS_ALL) == StepGranularity.EXHAUSTIVE


class TestOutputFormat:
    """Testes para OutputFormat enum."""
    
    def test_supports_math(self):
        """Testa suporte a matemática renderizada."""
        assert OutputFormat.HTML.supports_math() is True
        assert OutputFormat.LATEX.supports_math() is True
        assert OutputFormat.MARKDOWN.supports_math() is False
        assert OutputFormat.DOCX.supports_math() is False
    
    def test_is_interactive(self):
        """Testa formato interativo."""
        assert OutputFormat.HTML.is_interactive() is True
        assert OutputFormat.JUPYTER.is_interactive() is True
        assert OutputFormat.PDF.is_interactive() is False


class TestRenderConfig:
    """Testes para RenderConfig dataclass."""
    
    def test_default_values(self):
        """Testa valores padrão."""
        config = RenderConfig()
        
        assert config.mode == RenderMode.STEPS_SMART
        assert config.precision == 3
        assert config.show_units is True
        assert config.show_substitution is True
        assert config.use_unicode is True
        assert config.enable_cache is True
    
    def test_custom_values(self):
        """Testa valores customizados."""
        config = RenderConfig(
            mode=RenderMode.SYMBOLIC,
            precision=5,
            show_units=False,
            theme='dark'
        )
        
        assert config.mode == RenderMode.SYMBOLIC
        assert config.precision == 5
        assert config.show_units is False
        assert config.theme == 'dark'
    
    def test_post_init_validation_precision(self):
        """Testa validação de precisão."""
        # Precisão inválida deve ser corrigida para 3
        config = RenderConfig(precision=20)
        assert config.precision == 3
        
        config2 = RenderConfig(precision=-5)
        assert config2.precision == 3
    
    def test_post_init_validation_max_steps(self):
        """Testa validação de max_steps."""
        config = RenderConfig(max_steps=-10)
        assert config.max_steps == 0
    
    def test_from_config(self):
        """Testa criação a partir do config global."""
        config = RenderConfig.from_config()
        
        assert isinstance(config, RenderConfig)
        assert config.mode == RenderMode.STEPS_SMART
        # Precisão deve vir do config global
        assert isinstance(config.precision, int)
    
    def test_for_abnt(self):
        """Testa preset ABNT."""
        config = RenderConfig.for_abnt()
        
        assert config.theme == 'abnt'
        assert config.precision == 2
        assert config.output_format == OutputFormat.LATEX
        assert config.max_steps == 5
        assert 'abnt-equation' in config.css_classes
    
    def test_for_technical_report(self):
        """Testa preset relatório técnico."""
        config = RenderConfig.for_technical_report()
        
        assert config.theme == 'technical'
        assert config.mode == RenderMode.STEPS_DETAILED
        assert config.output_format == OutputFormat.HTML
    
    def test_for_jupyter(self):
        """Testa preset Jupyter."""
        config = RenderConfig.for_jupyter()
        
        assert config.output_format == OutputFormat.JUPYTER
        assert config.mode == RenderMode.MIXED
        assert config.enable_cache is False
    
    def test_to_dict(self):
        """Testa conversão para dicionário."""
        config = RenderConfig(precision=4, theme='dark')
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict['precision'] == 4
        assert config_dict['theme'] == 'dark'
    
    def test_copy(self):
        """Testa cópia com overrides."""
        config = RenderConfig(precision=3, theme='light')
        config_copy = config.copy(precision=5, show_units=False)
        
        # Original não deve mudar
        assert config.precision == 3
        assert config.show_units is True
        
        # Cópia deve ter overrides
        assert config_copy.precision == 5
        assert config_copy.show_units is False
        assert config_copy.theme == 'light'  # Mantém valor não sobrescrito
    
    def test_str_and_repr(self):
        """Testa representações string."""
        config = RenderConfig(mode=RenderMode.SYMBOLIC, precision=4)
        
        assert "RenderConfig" in repr(config)
        assert "SYMBOLIC" in repr(config)
        assert "RenderConfig[symbolic, 4 decimais]" == str(config)


class TestHelperFunctions:
    """Testes para funções auxiliares."""
    
    def test_create_render_config_basic(self):
        """Testa create_render_config básico."""
        config = create_render_config()
        
        assert isinstance(config, RenderConfig)
        assert config.mode == RenderMode.STEPS_SMART
    
    def test_create_render_config_with_mode(self):
        """Testa create_render_config com modo."""
        config = create_render_config(mode='symbolic')
        
        assert config.mode == RenderMode.SYMBOLIC
    
    def test_create_render_config_with_kwargs(self):
        """Testa create_render_config com kwargs."""
        config = create_render_config(
            mode='detailed',
            precision=5,
            show_units=False,
            theme='dark'
        )
        
        assert config.mode == RenderMode.STEPS_DETAILED
        assert config.precision == 5
        assert config.show_units is False
        assert config.theme == 'dark'
    
    def test_get_default_render_config(self):
        """Testa get_default_render_config."""
        config = get_default_render_config()
        
        assert isinstance(config, RenderConfig)
        assert config.mode == RenderMode.STEPS_SMART
