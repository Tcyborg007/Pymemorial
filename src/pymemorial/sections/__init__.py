"""
Módulo de análise de seções transversais.
Suporte para aço, concreto e mistas (composite).
"""
from .section_base import SectionAnalyzer, SectionProperties
from .factory import SectionFactory
from .steel import SteelSection

# Imports opcionais (podem não estar disponíveis)
try:
    from .concrete import ConcreteSection, CONCRETEPROPERTIES_AVAILABLE
except ImportError:
    ConcreteSection = None
    CONCRETEPROPERTIES_AVAILABLE = False

try:
    from .composite import CompositeSection, CompositeType, ShearConnectorType
except ImportError:
    CompositeSection = None
    CompositeType = None
    ShearConnectorType = None

__all__ = [
    'SectionAnalyzer',
    'SectionProperties',
    'SectionFactory',
    'SteelSection',
    'ConcreteSection',
    'CompositeSection',
    'CompositeType',
    'ShearConnectorType',
]
