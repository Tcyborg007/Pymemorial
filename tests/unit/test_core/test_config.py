"""
Testes completos para core/config.py

Seguindo TDD: Estes testes DEVEM FALHAR inicialmente (RED),
depois implementamos config.py para fazê-los passar (GREEN).
"""

import pytest
import os
import json
import tempfile
from pathlib import Path

from pymemorial.core.config import (
    PyMemorialConfig,
    DisplayConfig,
    SymbolsConfig,
    StandardConfig,
    RenderingConfig,
    get_config,
    set_option,
    reset_config,
    ConfigError
)


class TestPyMemorialConfigSingleton:
    """Testes do padrão Singleton."""
    
    def test_get_config_returns_singleton(self):
        """get_config() deve retornar sempre a mesma instância."""
        config1 = get_config()
        config2 = get_config()
        assert config1 is config2
    
    def test_reset_config_creates_new_instance(self):
        """reset_config() deve criar nova instância."""
        config1 = get_config()
        reset_config()
        config2 = get_config()
        assert config1 is not config2


class TestDisplayConfig:
    """Testes de configuração de display."""
    
    def test_default_precision(self):
        """Precisão padrão deve ser 3."""
        config = get_config()
        assert config.display.precision == 3
    
    def test_set_precision_valid(self):
        """Deve aceitar precisão entre 0 e 15."""
        set_option('display_precision', 5)
        config = get_config()
        assert config.display.precision == 5
    
    def test_set_precision_invalid_negative(self):
        """Deve rejeitar precisão negativa."""
        with pytest.raises(ConfigError, match="precision must be between 0 and 15"):
            set_option('display_precision', -1)
    
    def test_set_precision_invalid_too_high(self):
        """Deve rejeitar precisão acima de 15."""
        with pytest.raises(ConfigError, match="precision must be between 0 and 15"):
            set_option('display_precision', 20)
    
    def test_default_latex_style(self):
        """Estilo LaTeX padrão deve ser 'inline'."""
        config = get_config()
        assert config.display.latex_style == 'inline'
    
    def test_set_latex_style_valid(self):
        """Deve aceitar estilos válidos."""
        for style in ['inline', 'block', 'equation']:
            set_option('display_latex_style', style)
            config = get_config()
            assert config.display.latex_style == style
    
    def test_set_latex_style_invalid(self):
        """Deve rejeitar estilo inválido."""
        with pytest.raises(ConfigError, match="Invalid latex_style"):
            set_option('display_latex_style', 'invalid_style')


class TestSymbolsConfig:
    """Testes de configuração de símbolos."""
    
    def test_default_greek_style(self):
        """Estilo grego padrão deve ser 'latex'."""
        config = get_config()
        assert config.symbols.greek_style == 'latex'
    
    def test_set_greek_style(self):
        """Deve alternar entre 'latex' e 'unicode'."""
        set_option('symbols_greek_style', 'unicode')
        config = get_config()
        assert config.symbols.greek_style == 'unicode'
    
    def test_default_subscript_detection(self):
        """Detecção de subscrito deve estar ativa por padrão."""
        config = get_config()
        assert config.symbols.auto_subscript is True
    
    def test_default_custom_registry_path(self):
        """Path do registry customizado deve ser ~/.pymemorial/symbols.json."""
        config = get_config()
        expected_path = Path.home() / '.pymemorial' / 'symbols.json'
        assert config.symbols.custom_registry_path == expected_path


class TestStandardConfig:
    """Testes de configuração de normas técnicas."""
    
    def test_default_standard_none(self):
        """Norma ativa padrão deve ser None."""
        config = get_config()
        assert config.standard.active_standard is None
    
    def test_load_profile_nbr6118(self):
        """Deve carregar perfil NBR 6118."""
        config = get_config()
        config.load_profile('nbr6118')
        
        assert config.standard.active_standard == 'nbr6118'
        assert config.standard.partial_factors is not None
        assert 'gamma_c' in config.standard.partial_factors
        assert config.standard.partial_factors['gamma_c'] == 1.4
        assert config.standard.partial_factors['gamma_s'] == 1.15
    
    def test_load_profile_nbr8800(self):
        """Deve carregar perfil NBR 8800."""
        config = get_config()
        config.load_profile('nbr8800')
        
        assert config.standard.active_standard == 'nbr8800'
        assert config.standard.partial_factors is not None
        assert 'gamma_a1' in config.standard.partial_factors
    
    def test_load_profile_eurocode2(self):
        """Deve carregar perfil Eurocode 2."""
        config = get_config()
        config.load_profile('eurocode2')
        
        assert config.standard.active_standard == 'eurocode2'
        assert config.standard.partial_factors is not None
    
    def test_load_profile_invalid(self):
        """Deve rejeitar perfil inexistente."""
        config = get_config()
        with pytest.raises(ConfigError, match="Unknown profile"):
            config.load_profile('invalid_profile')


class TestRenderingConfig:
    """Testes de configuração de renderização."""
    
    def test_default_render_mode(self):
        """Modo padrão deve ser 'steps:smart'."""
        config = get_config()
        assert config.rendering.default_mode == 'steps:smart'
    
    def test_set_render_mode(self):
        """Deve aceitar modos válidos e normalizar aliases."""
        # Modos exatos (não devem ser alterados)
        exact_modes = [
            'steps:minimal', 'steps:smart', 'steps:detailed', 'steps:all',
            'params', 'long', 'short', 'symbolic'
        ]
        
        for mode in exact_modes:
            set_option('rendering_default_mode', mode)
            config = get_config()
            assert config.rendering.default_mode == mode
        
        # Aliases devem ser normalizados para a forma canônica
        alias_mapping = {
            'minimal': 'steps:minimal',
            'smart': 'steps:smart',
            'detailed': 'steps:detailed',
            'all': 'steps:all'
        }
        
        for alias, canonical in alias_mapping.items():
            set_option('rendering_default_mode', alias)
            config = get_config()
            assert config.rendering.default_mode == canonical
    
    def test_default_show_units(self):
        """Unidades devem ser mostradas por padrão."""
        config = get_config()
        assert config.rendering.show_units is True
    
    def test_default_show_comments(self):
        """Comentários devem ser mostrados por padrão."""
        config = get_config()
        assert config.rendering.show_comments is True


class TestConfigPersistence:
    """Testes de persistência em arquivo."""
    
    @pytest.fixture
    def temp_config_dir(self):
        """Fixture: diretório temporário para config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    def test_save_config(self, temp_config_dir, monkeypatch):
        """Deve salvar configuração em JSON."""
        config_file = temp_config_dir / 'config.json'
        
        config = get_config()
        monkeypatch.setattr(config, 'config_file', config_file)
        
        set_option('display_precision', 5)
        config.save_config()
        
        assert config_file.exists()
        
        with open(config_file) as f:
            data = json.load(f)
        
        assert data['display']['precision'] == 5
    
    def test_load_config(self, temp_config_dir, monkeypatch):
        """Deve carregar configuração de JSON."""
        config_file = temp_config_dir / 'config.json'
        
        config_data = {
            'display': {'precision': 7, 'latex_style': 'block'},
            'symbols': {'greek_style': 'unicode'},
            'standard': {'active_standard': 'nbr6118'},
            'rendering': {'default_mode': 'steps:detailed'}
        }
        
        with open(config_file, 'w') as f:
            json.dump(config_data, f)
        
        reset_config()
        config = get_config()
        monkeypatch.setattr(config, 'config_file', config_file)
        config.load_config()
        
        assert config.display.precision == 7
        assert config.display.latex_style == 'block'
        assert config.symbols.greek_style == 'unicode'
        assert config.standard.active_standard == 'nbr6118'
        assert config.rendering.default_mode == 'steps:detailed'
    
    def test_auto_create_config_dir(self, temp_config_dir, monkeypatch):
        """Deve criar diretório ~/.pymemorial se não existir."""
        config_dir = temp_config_dir / '.pymemorial'
        config_file = config_dir / 'config.json'
        
        config = get_config()
        monkeypatch.setattr(config, 'config_file', config_file)
        
        config.save_config()
        
        assert config_dir.exists()
        assert config_file.exists()


class TestSetOptionAPI:
    """Testes da API set_option."""
    
    def test_set_option_display_precision(self):
        """set_option deve modificar display.precision."""
        set_option('display_precision', 4)
        config = get_config()
        assert config.display.precision == 4
    
    def test_set_option_nested_underscore(self):
        """Deve aceitar notação com underscore (categoria_atributo)."""
        set_option('symbols_greek_style', 'unicode')
        config = get_config()
        assert config.symbols.greek_style == 'unicode'
    
    def test_set_option_nested_dot(self):
        """Deve aceitar notação com ponto (categoria.atributo)."""
        set_option('rendering.show_units', False)
        config = get_config()
        assert config.rendering.show_units is False
    
    def test_set_option_invalid_key(self):
        """Deve rejeitar chave inexistente (sem separador válido)."""
        # Chave sem separador - formato inválido
        with pytest.raises(ConfigError, match="Unknown config key"):
            set_option('invalidkey', 123)
    
    def test_set_option_invalid_category(self):
        """Deve rejeitar categoria inexistente (com separador válido)."""
        # Chave com separador mas categoria inválida
        with pytest.raises(ConfigError, match="Unknown config category"):
            set_option('nonexistent_category.attr', 123)


class TestConfigRepr:
    """Testes de representação string."""
    
    def test_config_repr(self):
        """Config deve ter __repr__ legível."""
        config = get_config()
        repr_str = repr(config)
        
        assert 'PyMemorialConfig' in repr_str
        assert 'display' in repr_str
        assert 'symbols' in repr_str
    
    def test_display_config_str(self):
        """DisplayConfig deve ter __str__ formatado."""
        config = get_config()
        str_repr = str(config.display)
        
        assert 'precision' in str_repr
        assert 'latex_style' in str_repr