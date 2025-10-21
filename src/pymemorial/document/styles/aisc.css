# src/pymemorial/__init__.py
"""
PyMemorial 2.0 - Memoriais de cálculo estrutural profissionais
"""

# Estratégia robusta de versionamento
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

# Expor apenas o necessário
__all__ = ["__version__"]