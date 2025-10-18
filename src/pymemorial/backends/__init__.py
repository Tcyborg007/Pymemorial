"""Backends estruturais para análise."""
from .backend_base import StructuralBackend, Node, Member, Load, Support
from .factory import BackendFactory
from .adapter import StructuralAdapter, SimpleFrameAdapter

__all__ = [
    'StructuralBackend',
    'Node',
    'Member',
    'Load',
    'Support',
    'BackendFactory',
    'StructuralAdapter',
    'SimpleFrameAdapter',
]
