"""
Sistema de configuração global do PyMemorial v2.0
"""

from __future__ import annotations

import json
import os
import contextlib
import threading
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Any, Optional, Literal

__all__ = [
    'PyMemorialConfig',
    'DisplayConfig',
    'SymbolsConfig',
    'StandardConfig',
    'RenderingConfig',
    'get_config',
    'set_option',
    'get_option',
    'reset_config',
    'override_options',
    'ConfigError'
]


# =============================================================================
# EXCEÇÕES
# =============================================================================

class ConfigError(Exception):
    """Erro de configuração do PyMemorial."""
    pass


# =============================================================================
# CONFIGURAÇÕES POR CATEGORIA
# =============================================================================

@dataclass
class DisplayConfig:
    """
    Configurações de exibição de resultados.
    
    Attributes:
        precision: Número de casas decimais (0-15)
        latex_style: Estilo de equações LaTeX ('inline', 'block', 'equation')
        number_format: Formato de números ('decimal', 'scientific', 'engineering')
        use_unicode: Usar caracteres Unicode (√, π, etc)
    """
    precision: int = 3
    latex_style: Literal['inline', 'block', 'equation'] = 'inline'
    number_format: Literal['decimal', 'scientific', 'engineering'] = 'decimal'
    use_unicode: bool = True
    
    def __post_init__(self):
        """Validação após inicialização."""
        if not (0 <= self.precision <= 15):
            raise ConfigError(f"precision must be between 0 and 15, got {self.precision}")
        
        valid_latex_styles = ['inline', 'block', 'equation']
        if self.latex_style not in valid_latex_styles:
            raise ConfigError(
                f"Invalid latex_style '{self.latex_style}'. "
                f"Must be one of {valid_latex_styles}"
            )
    
    def __str__(self) -> str:
        return (
            f"DisplayConfig(precision={self.precision}, "
            f"latex_style='{self.latex_style}', "
            f"number_format='{self.number_format}')"
        )


@dataclass
class SymbolsConfig:
    """
    Configurações de símbolos e notação.
    
    Attributes:
        greek_style: Estilo de letras gregas ('latex', 'unicode')
        auto_subscript: Auto-detectar subscritos (gamma_s → γₛ)
        auto_superscript: Auto-detectar superescritos (x_2 → x²)
        custom_registry_path: Path do registry de símbolos customizados
    """
    greek_style: Literal['latex', 'unicode'] = 'latex'
    auto_subscript: bool = True
    auto_superscript: bool = False
    custom_registry_path: Path = field(
        default_factory=lambda: Path.home() / '.pymemorial' / 'symbols.json'
    )
    
    def __str__(self) -> str:
        return (
            f"SymbolsConfig(greek_style='{self.greek_style}', "
            f"auto_subscript={self.auto_subscript})"
        )


@dataclass
class StandardConfig:
    """
    Configurações de normas técnicas.
    
    Attributes:
        active_standard: Norma ativa ('nbr6118', 'nbr8800', 'eurocode2', 'aci318', None)
        partial_factors: Coeficientes de ponderação da norma ativa
        unit_system: Sistema de unidades ('SI', 'imperial')
    """
    active_standard: Optional[str] = None
    partial_factors: Optional[Dict[str, float]] = None
    unit_system: Literal['SI', 'imperial'] = 'SI'
    
    def __str__(self) -> str:
        return (
            f"StandardConfig(active='{self.active_standard}', "
            f"unit_system='{self.unit_system}')"
        )


@dataclass
class RenderingConfig:
    """
    Configurações de renderização de memoriais.
    
    Attributes:
        default_mode: Modo de renderização padrão
        show_units: Mostrar unidades nos resultados
        show_comments: Mostrar comentários inline do código
        show_dimensions: Mostrar dimensões em diagramas
        color_scheme: Esquema de cores ('default', 'dark', 'nbr')
    """
    default_mode: str = 'steps:smart'
    show_units: bool = True
    show_comments: bool = True
    show_dimensions: bool = True
    color_scheme: Literal['default', 'dark', 'nbr'] = 'default'
    
    VALID_MODES = [
        # Modos Handcalcs-compatible
        'params', 'long', 'short', 'symbolic',
        # Modos PyMemorial (steps)
        'minimal', 'smart', 'detailed', 'all',
        'steps:minimal', 'steps:smart', 'steps:detailed', 'steps:all'
    ]
    
    VALID_MODE_ALIASES = {
        'minimal': 'steps:minimal',
        'smart': 'steps:smart',
        'detailed': 'steps:detailed',
        'all': 'steps:all'
    }
    
    def __post_init__(self):
        """Validação e normalização de modo de renderização."""
        # Normalizar aliases
        if self.default_mode in self.VALID_MODE_ALIASES:
            self.default_mode = self.VALID_MODE_ALIASES[self.default_mode]
        
        if self.default_mode not in self.VALID_MODES:
            raise ConfigError(
                f"Invalid render mode '{self.default_mode}'. "
                f"Must be one of {self.VALID_MODES}"
            )
    
    def __str__(self) -> str:
        return (
            f"RenderingConfig(mode='{self.default_mode}', "
            f"show_units={self.show_units})"
        )


# =============================================================================
# UTILITÁRIOS INTERNOS
# =============================================================================

def _split_key(key: str) -> tuple[str, str]:
    """Divide chave no formato 'categoria.atributo' ou 'categoria_atributo'."""
    if '.' in key:
        parts = key.split('.', 1)
    elif '_' in key:
        parts = key.split('_', 1)
    else:
        raise ConfigError(f"Unknown config key '{key}'")
    
    if len(parts) != 2:
        raise ConfigError(f"Unknown config key '{key}'")
    
    return parts[0], parts[1]


# =============================================================================
# CONFIGURAÇÃO PRINCIPAL
# =============================================================================

class PyMemorialConfig:
    """
    Configuração global do PyMemorial (Singleton).
    """
    
    # Versão do schema de configuração
    CONFIG_VERSION = 1
    
    # Perfis de normas técnicas (coeficientes padrão)
    STANDARD_PROFILES = {
        'nbr6118': {
            'gamma_c': 1.4,      # Concreto
            'gamma_s': 1.15,     # Aço
            'gamma_f': 1.4,      # Cargas
            'gamma_f_exc': 1.2   # Cargas excepcionais
        },
        'nbr8800': {
            'gamma_a1': 1.10,    # Aço (combinação normal)
            'gamma_a2': 1.35,    # Aço (combinação última)
            'gamma_f': 1.4
        },
        'eurocode2': {
            'gamma_c': 1.5,      # Concreto (EC2)
            'gamma_s': 1.15,     # Aço
            'gamma_g': 1.35,     # Cargas permanentes
            'gamma_q': 1.5       # Cargas variáveis
        },
        'aci318': {
            'phi_flexure': 0.90,       # Fator redução flexão
            'phi_shear': 0.75,         # Fator redução cisalhamento
            'phi_compression': 0.65    # Fator redução compressão
        }
    }
    
    def __init__(self):
        self.display = DisplayConfig()
        self.symbols = SymbolsConfig()
        self.standard = StandardConfig()
        self.rendering = RenderingConfig()

        # Determinar arquivo de configuração (com suporte a variável de ambiente)
        config_file_env = os.getenv('PYMEMORIAL_CONFIG_FILE')
        if config_file_env:
            self.config_file = Path(config_file_env)
        else:
            config_dir = Path.home() / '.pymemorial'
            self.config_file = config_dir / 'config.json'

        # Só faz autoload se a env var NÃO estiver ativada
        disable = os.getenv('PYMEMORIAL_DISABLE_AUTOLOAD', '').lower() in ('1', 'true', 'yes')
        if not disable and self.config_file.exists():
            try:
                self.load_config()
            except Exception:
                pass
    
    def load_profile(self, profile_name: str) -> None:
        """Carrega perfil de norma técnica."""
        if profile_name not in self.STANDARD_PROFILES:
            raise ConfigError(
                f"Unknown profile '{profile_name}'. "
                f"Available: {list(self.STANDARD_PROFILES.keys())}"
            )
        
        self.standard.active_standard = profile_name
        self.standard.partial_factors = self.STANDARD_PROFILES[profile_name].copy()
    
    def save_config(self) -> None:
        """Salva configuração em arquivo JSON com escrita atômica."""
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = {
            'config_version': self.CONFIG_VERSION,
            'display': asdict(self.display),
            'symbols': {
                'greek_style': self.symbols.greek_style,
                'auto_subscript': self.symbols.auto_subscript,
                'auto_superscript': self.symbols.auto_superscript,
                'custom_registry_path': str(self.symbols.custom_registry_path)
            },
            'standard': {
                'active_standard': self.standard.active_standard,
                'partial_factors': self.standard.partial_factors,
                'unit_system': self.standard.unit_system
            },
            'rendering': asdict(self.rendering)
        }
        
        # Escrita atômica para evitar corrupção
        tmp_file = self.config_file.with_suffix('.json.tmp')
        try:
            with open(tmp_file, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
            os.replace(tmp_file, self.config_file)
        except Exception:
            # Limpar arquivo temporário em caso de erro
            if tmp_file.exists():
                tmp_file.unlink()
            raise
    
    def load_config(self) -> None:
        """Carrega configuração de arquivo JSON com merge seguro."""
        if not self.config_file.exists():
            return
        
        with open(self.config_file, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        # Verificar versão e migrar se necessário (placeholder para futuras versões)
        file_version = config_dict.get('config_version', 0)
        if file_version < self.CONFIG_VERSION:
            # Rotina de migração iria aqui
            pass
        
        # Restaurar cada categoria com merge seguro
        if 'display' in config_dict:
            # Merge com defaults
            display_defaults = asdict(DisplayConfig())
            display_defaults.update(config_dict['display'])
            self.display = DisplayConfig(**display_defaults)
        
        if 'symbols' in config_dict:
            symbols_dict = config_dict['symbols'].copy()
            if 'custom_registry_path' in symbols_dict:
                symbols_dict['custom_registry_path'] = Path(symbols_dict['custom_registry_path'])
            # Merge com defaults
            symbols_defaults = asdict(SymbolsConfig())
            symbols_defaults.update(symbols_dict)
            self.symbols = SymbolsConfig(**symbols_defaults)
        
        if 'standard' in config_dict:
            # Merge com defaults
            standard_defaults = asdict(StandardConfig())
            standard_defaults.update(config_dict['standard'])
            self.standard = StandardConfig(**standard_defaults)
        
        if 'rendering' in config_dict:
            # Merge com defaults
            rendering_defaults = asdict(RenderingConfig())
            rendering_defaults.update(config_dict['rendering'])
            self.rendering = RenderingConfig(**rendering_defaults)
    
    def __repr__(self) -> str:
        return (
            f"PyMemorialConfig(\n"
            f" \tdisplay={self.display},\n"
            f" \tsymbols={self.symbols},\n"
            f" \tstandard={self.standard},\n"
            f" \trendering={self.rendering}\n"
            f")"
        )


# =============================================================================
# SINGLETON GLOBAL COM THREAD-SAFETY
# =============================================================================

_global_config: Optional[PyMemorialConfig] = None
_config_lock = threading.Lock()


def get_config() -> PyMemorialConfig:
    """Obtém instância singleton da configuração global (thread-safe)."""
    global _global_config
    if _global_config is None:
        with _config_lock:
            if _global_config is None:
                _global_config = PyMemorialConfig()
    return _global_config


def reset_config() -> None:
    """Reseta configuração para padrões (cria nova instância)."""
    global _global_config
    with _config_lock:
        _global_config = None


def set_option(key: str, value: Any) -> None:
    """
    API simplificada para alterar opção de configuração.
    
    Suporta formatos:
    - 'categoria.atributo' (ponto)
    - 'categoria_atributo' (underscore)
    """
    config = get_config()
    valid_categories = {'display', 'symbols', 'standard', 'rendering'}

    category, attr = _split_key(key)

    # Validar categoria
    if category not in valid_categories:
        raise ConfigError(
            f"Unknown config category '{category}'. "
            f"Available: {', '.join(valid_categories)}"
        )

    # Validar atributo
    config_obj = getattr(config, category)
    if not hasattr(config_obj, attr):
        raise ConfigError(f"Unknown config key '{attr}' in category '{category}'")

    # Aplicar valor com validação via dataclass
    try:
        current_values = asdict(config_obj)
        current_values[attr] = value
        
        if category == 'display':
            new_config_obj = DisplayConfig(**current_values)
        elif category == 'symbols':
            new_config_obj = SymbolsConfig(**current_values)
        elif category == 'standard':
            new_config_obj = StandardConfig(**current_values)
        elif category == 'rendering':
            new_config_obj = RenderingConfig(**current_values)
        else:
            raise ConfigError(f"Unknown category '{category}'")
        
        setattr(config, category, new_config_obj)
        
    except (ValueError, TypeError) as e:
        raise ConfigError(f"Invalid value for '{key}': {e}")
    except ConfigError:
        raise


def get_option(key: str) -> Any:
    """
    Obtém o valor de uma opção de configuração.
    
    Examples:
        >>> get_option('display.precision')
        3
        >>> get_option('symbols_greek_style')
        'latex'
    """
    config = get_config()
    category, attr = _split_key(key)
    
    valid_categories = {'display', 'symbols', 'standard', 'rendering'}
    if category not in valid_categories:
        raise ConfigError(
            f"Unknown config category '{category}'. "
            f"Available: {', '.join(valid_categories)}"
        )
    
    config_obj = getattr(config, category)
    if not hasattr(config_obj, attr):
        raise ConfigError(f"Unknown config key '{attr}' in category '{category}'")
    
    return getattr(config_obj, attr)


@contextlib.contextmanager
def override_options(**kwargs):
    """
    Context manager para overrides temporários de configuração.
    
    Examples:
        >>> with override_options(display_precision=5, symbols_greek_style='unicode'):
        ...     # configurações temporárias
        ...     print(get_option('display.precision'))  # 5
        >>> # configurações restauradas
    """
    cfg = get_config()
    old_values = {}
    
    try:
        # Salvar valores antigos e aplicar novos
        for key, value in kwargs.items():
            old_values[key] = get_option(key)
            set_option(key, value)
        yield cfg
    finally:
        # Restaurar valores originais
        for key, value in old_values.items():
            set_option(key, value)


# =============================================================================
# AUTO-EXPORTAÇÃO
# =============================================================================

if __name__ == '__main__':
    print("PyMemorial v2.0 - Configuração Global\n")
    
    config = get_config()
    print(f"Config padrão:\n{config}\n")
    
    set_option('display_precision', 5)
    print(f"Após set_option('display_precision', 5):")
    print(f" \tprecision = {config.display.precision}\n")
    
    config.load_profile('nbr6118')
    print(f"Após load_profile('nbr6118'):")
    print(f" \tgamma_c = {config.standard.partial_factors['gamma_c']}")
    print(f" \tgamma_s = {config.standard.partial_factors['gamma_s']}\n")
    
    config.save_config()
    print(f"✅ Config salvo em: {config.config_file}")