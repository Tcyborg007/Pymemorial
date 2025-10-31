# src/pymemorial/__init__.py
"""
PyMemorial - Biblioteca para Memoriais de Cálculo Estruturais

Versão 3.0.0 - API Unificada com EngMemorial

Author: PyMemorial Team
License: MIT
"""

__version__ = "3.0.0"

# ============================================================================
# NOVA API - EngMemorial (v3.0+)
# ============================================================================

try:
    from .eng_memorial import EngMemorial, MemorialMetadata
    ENGMEMORIAL_AVAILABLE = True
except ImportError as e:
    import warnings
    warnings.warn(f"EngMemorial não disponível: {e}")
    ENGMEMORIAL_AVAILABLE = False
    EngMemorial = None
    MemorialMetadata = None

# ============================================================================
# ENGINE (v3.0+)
# ============================================================================

try:
    from .engine import (
        MemorialContext,
        UnifiedProcessor,
        GranularityLevel,
        get_context,
    )
    ENGINE_AVAILABLE = True
except ImportError as e:
    import warnings
    warnings.warn(f"Engine não disponível: {e}")
    ENGINE_AVAILABLE = False

# ============================================================================
# API FLUENTE (MANTIDA) - (CalculationReport)
# ============================================================================

try:
    from .api import (
        CalculationReport,
        ReportBuilder,
        ReportSection,
    )
    FLUENT_API_AVAILABLE = True
except ImportError as e:
    import warnings
    warnings.warn(f"API Fluente (CalculationReport) não disponível: {e}")
    FLUENT_API_AVAILABLE = False
    CalculationReport = None
    ReportBuilder = None
    ReportSection = None

# ============================================================================
# CORE (v1.0+ - MANTIDO)
# ============================================================================

try:
    from .core import (
        Variable,
        VariableFactory,
        Equation,
        Calculator,
        get_config,
    )
    CORE_AVAILABLE = True
except ImportError:
    CORE_AVAILABLE = False

# ============================================================================
# SECTIONS (v1.0+ - MANTIDO)
# ============================================================================

try:
    from .sections import (
        ConcreteSection,
        SteelSection,
        CompositeSection,
        SectionFactory,
    )
    SECTIONS_AVAILABLE = True
except ImportError:
    SECTIONS_AVAILABLE = False

# ============================================================================
# BACKENDS (v1.0+ - MANTIDO)
# ============================================================================

try:
    from .backends import (
        BackendFactory,
        PyniteBackend,
        OpenSeesBackend,
    )
    BACKENDS_AVAILABLE = True
except ImportError:
    BACKENDS_AVAILABLE = False

# ============================================================================
# RECOGNITION (v1.0+ - MANTIDO)
# ============================================================================

try:
    from .recognition import (
        PyMemorialASTParser,
        SmartTextEngine,
        GreekSymbols,
    )
    RECOGNITION_AVAILABLE = True
except ImportError:
    RECOGNITION_AVAILABLE = False

# ============================================================================
# SYMBOLS (v2.0+ - MANTIDO)
# ============================================================================

try:
    from .symbols import (
        SymbolRegistry,
        get_global_registry,
    )
    SYMBOLS_AVAILABLE = True
except ImportError:
    SYMBOLS_AVAILABLE = False

# ============================================================================
# VISUALIZATION (v1.0+ - MANTIDO)
# ============================================================================

try:
    from .visualization import (
        PlotlyEngine,
        PyVistaEngine,
        export_figure,
    )
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False

# ============================================================================
# LEGACY APIs (v1.0-2.0 - DEPRECATED)
# ============================================================================

import warnings

def _deprecated_import(module_name: str, class_name: str):
    """Helper para imports deprecados."""
    warnings.warn(
        f"{module_name}.{class_name} está deprecado. "
        f"Use EngMemorial ao invés. "
        f"Ver: https://pymemorial.readthedocs.io/migration/v3",
        DeprecationWarning,
        stacklevel=3
    )

# Builder (DEPRECATED - usar EngMemorial)
try:
    from .builder import MemorialBuilder as _MemorialBuilder
    
    class MemorialBuilder(_MemorialBuilder):
        """DEPRECATED: Use EngMemorial ao invés."""
        def __init__(self, *args, **kwargs):
            _deprecated_import("builder", "MemorialBuilder")
            super().__init__(*args, **kwargs)
    
    BUILDER_AVAILABLE = True
except ImportError:
    BUILDER_AVAILABLE = False
    MemorialBuilder = None

# Document (DEPRECATED - usar EngMemorial)
try:
    from .document import Memorial as _DocumentMemorial
    
    class Memorial(_DocumentMemorial):
        """DEPRECATED: Use EngMemorial ao invés."""
        def __init__(self, *args, **kwargs):
            _deprecated_import("document", "Memorial")
            super().__init__(*args, **kwargs)
    
    DOCUMENT_AVAILABLE = True
except ImportError:
    DOCUMENT_AVAILABLE = False
    Memorial = None

# Editor (DEPRECATED - usar EngMemorial)
try:
    from .editor import NaturalWriter as _NaturalWriter
    
    class NaturalWriter(_NaturalWriter):
        """DEPRECATED: Use EngMemorial ao invés."""
        def __init__(self, *args, **kwargs):
            _deprecated_import("editor", "NaturalWriter")
            super().__init__(*args, **kwargs)
    
    EDITOR_AVAILABLE = True
except ImportError:
    EDITOR_AVAILABLE = False
    NaturalWriter = None

# ============================================================================
# PUBLIC API EXPORTS
# ============================================================================

__all__ = [
    # Nova API v3.0 (RECOMENDADO)
    "EngMemorial",
    "MemorialMetadata",
    "MemorialContext",
    "UnifiedProcessor",
    "GranularityLevel",
    "get_context",
    
    # API Fluente (MANTIDA)
    "CalculationReport",
    "ReportBuilder",
    "ReportSection",
    
    # Core (MANTIDO)
    "Variable",
    "VariableFactory",
    "Equation",
    "Calculator",
    "get_config",
    
    # Sections (MANTIDO)
    "ConcreteSection",
    "SteelSection",
    "CompositeSection",
    "SectionFactory",
    
    # Backends (MANTIDO)
    "BackendFactory",
    "PyniteBackend",
    "OpenSeesBackend",
    
    # Recognition (MANTIDO)
    "PyMemorialASTParser",
    "SmartTextEngine",
    "GreekSymbols",
    
    # Symbols (MANTIDO)
    "SymbolRegistry",
    "get_global_registry",
    
    # Visualization (MANTIDO)
    "PlotlyEngine",
    "PyVistaEngine",
    "export_figure",
    
    # Legacy (DEPRECATED)
    "MemorialBuilder",  # Deprecated
    "Memorial",         # Deprecated
    "NaturalWriter",    # Deprecated
]

# ============================================================================
# VERSION CHECK & WELCOME MESSAGE
# ============================================================================

def _check_environment():
    """Verifica ambiente e mostra status."""
    missing = []
    if not ENGMEMORIAL_AVAILABLE:
        missing.append("engine")
    if not CORE_AVAILABLE:
        missing.append("core")
    if not SECTIONS_AVAILABLE:
        missing.append("sections")
    
    if missing:
        warnings.warn(
            f"Módulos faltando: {', '.join(missing)}. "
            f"Instale com: pip install pymemorial[all]",
            ImportWarning
        )

# Executar check na importação