# src/pymemorial/document/internal/text_processing/__init__.py
"""
Text Processing Module - Internal

Bridge entre Recognition e Document.
"""

from .integration import (
    TextProcessingBridge,
    DocumentTextProcessor,
    create_text_bridge,
    RECOGNITION_AVAILABLE,
    TEXT_UTILS_AVAILABLE,
)

__all__ = [
    'TextProcessingBridge',
    'DocumentTextProcessor', 
    'create_text_bridge',
    'RECOGNITION_AVAILABLE',
    'TEXT_UTILS_AVAILABLE',
]
