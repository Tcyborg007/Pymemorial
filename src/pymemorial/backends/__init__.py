"""Backends estruturais para análise."""
from .backend_base import StructuralBackend, Node, Member, Load, Support
from .factory import BackendFactory
from .adapter import StructuralAdapter, SimpleFrameAdapter

# ✅ ADICIONE ESTAS LINHAS PARA EXPOR OS BACKENDS
from .pynite import PyniteBackend
from .opensees import OpenSeesBackend


__all__ = [
    'StructuralBackend',
    'Node',
    'Member',
    'Load',
    'Support',
    'BackendFactory',
    'StructuralAdapter',
    'SimpleFrameAdapter',
    
    # ✅ ADICIONE OS NOMES AQUI
    'PyniteBackend',
    'OpenSeesBackend',
]