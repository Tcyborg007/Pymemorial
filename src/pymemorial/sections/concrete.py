"""Seção de concreto (stub temporário)."""
from .base import SectionAnalyzer, SectionProperties


class ConcreteSection(SectionAnalyzer):
    """Stub temporário para ConcreteSection."""
    
    def __init__(self, name: str, **kwargs):
        super().__init__(name=name, section_type='concrete')
    
    def build_geometry(self, **kwargs):
        self.geometry_built = True
    
    def calculate_properties(self) -> SectionProperties:
        return SectionProperties(
            area=0.02, perimeter=0.8,
            ixx=2e-5, iyy=1e-5, ixy=0.0,
            cx=0.0, cy=0.0
        )
