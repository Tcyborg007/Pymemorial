# src/pymemorial/engine/__init__.py
"""
PyMemorial Engine - Motor Unificado

Consolida context + processor em um único pacote.
"""



from .context import (
    MemorialContext,
    VariableScope,
    get_context,
    reset_context,
)

from .processor import (
    UnifiedProcessor,
    ProcessingResult,
    StepData,
    GranularityLevel,
    ProcessingMode,
)

__all__ = [
    # Context
    "MemorialContext",
    "VariableScope",
    "get_context",
    "reset_context",
    
    # Processor
    "UnifiedProcessor",
    "ProcessingResult",
    "StepData",
    "GranularityLevel",
    "ProcessingMode",
]
