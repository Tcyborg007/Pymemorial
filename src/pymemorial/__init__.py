# src/pymemorial/__init__.py

"""
PyMemorial v2.0 - Sistema profissional de memoriais de cálculo estrutural

PyMemorial combina:
- Sintaxe elegante Python (Handcalcs-style)
- Steps automáticos em 4 níveis (Calcpad-style)  
- Métodos numéricos avançados (SciPy)
- Backends de análise estrutural (FEA)
- Bibliotecas de normas técnicas (NBR, EC, ACI)

Examples:
    >>> from pymemorial import get_config, set_option
    >>> 
    >>> # Configurar sistema
    >>> set_option('display_precision', 4)
    >>> config = get_config()
    >>> config.load_profile('nbr6118')
    >>> 
    >>> # Criar memorial (quando editor estiver pronto)
    >>> # from pymemorial.editor import PyMemorialNaturalWriter
    >>> # writer = PyMemorialNaturalWriter()
"""

# =============================================================================
# VERSIONAMENTO
# =============================================================================

try:
    from importlib.metadata import version as _get_version, PackageNotFoundError
    try:
        __version__ = _get_version("pymemorial")
    except PackageNotFoundError:
        # Fallback para desenvolvimento (src layout sem instalação editable)
        from .__version__ import __version__
except Exception:
    # Fallback defensivo final
    from .__version__ import __version__


# =============================================================================
# CONFIGURAÇÃO GLOBAL (core/config.py - JÁ IMPLEMENTADO!)
# =============================================================================

from .core.config import (
    get_config,
    set_option,
    reset_config,
    PyMemorialConfig,
    DisplayConfig,
    SymbolsConfig,
    StandardConfig,
    RenderingConfig,
    ConfigError
)


# =============================================================================
# IMPORTS FUTUROS (Adicionar conforme módulos forem criados)
# =============================================================================

# Fase 1 (Semanas 1-2) - A criar:
# from .recognition import PyMemorialASTParser
# from .symbols import get_registry, define_symbol

# Fase 2 (Semanas 3-4) - A criar:
# from .editor import (
#     PyMemorialNaturalWriter,
#     HybridStepEngine,
#     RenderMode
# )

# Fase 3 (Semanas 5-8) - A criar:
# from .numerical import (
#     solve_equation,
#     optimize,
#     integrate,
#     monte_carlo_analysis
# )

# Fase 4 (Semanas 9-12) - A criar:
# from .libraries.nbr6118 import dimensionar_flexao_simples_nbr6118
# from .libraries.nbr8800 import verificar_flambagem_perfil

# Fase 5 (Semanas 13-16) - A criar:
# from .api import pymemorial, handcalc  # Decorators


# =============================================================================
# EXPORTS PÚBLICOS
# =============================================================================

__all__ = [
    # Versão
    '__version__',
    
    # Configuração (DISPONÍVEL AGORA!)
    'get_config',
    'set_option',
    'reset_config',
    'PyMemorialConfig',
    'DisplayConfig',
    'SymbolsConfig',
    'StandardConfig',
    'RenderingConfig',
    'ConfigError',
    
    # Futuros (descomentar conforme implementado):
    # 'PyMemorialASTParser',
    # 'get_registry',
    # 'define_symbol',
    # 'PyMemorialNaturalWriter',
    # 'HybridStepEngine',
    # 'RenderMode',
    # 'solve_equation',
    # 'optimize',
    # 'integrate',
    # 'monte_carlo_analysis',
    # 'dimensionar_flexao_simples_nbr6118',
    # 'pymemorial',
    # 'handcalc',
]


# =============================================================================
# INFORMAÇÕES DO PACOTE
# =============================================================================

__author__ = "PyMemorial Team"
__license__ = "MIT"
__url__ = "https://github.com/yourusername/pymemorial"  # Ajustar quando houver repo

# Desabilitar warning de imports não usados (para os comentados)
# flake8: noqa: F401
