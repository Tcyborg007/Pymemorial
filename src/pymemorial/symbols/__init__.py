# src/pymemorial/symbols/__init__.py

"""
Módulo de símbolos customizados do PyMemorial v2.0

Sistema inteligente de registry de símbolos com:
- Auto-aprendizado de símbolos do código
- Persistência em JSON
- Conversão LaTeX automática
- Busca fuzzy
"""

from pymemorial.symbols.custom_registry import (
    Symbol,
    SymbolRegistry,
    RegistryError,
    get_global_registry,
    reset_global_registry
)

__all__ = [
    'Symbol',
    'SymbolRegistry',
    'RegistryError',
    'get_global_registry',
    'reset_global_registry'
]
