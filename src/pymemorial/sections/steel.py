"""Seção de aço (stub temporário)."""
from .section_base import SectionAnalyzer, SectionProperties


class SteelSection(SectionAnalyzer):
    """Stub temporário para SteelSection."""
    
    def __init__(self, name: str, **kwargs):
        super().__init__(name=name, section_type='steel')
    
    def build_geometry(self, **kwargs):
        self.geometry_built = True
    
    def calculate_properties(self) -> SectionProperties:
        return SectionProperties(
            area=0.01, perimeter=0.4,
            ixx=1e-5, iyy=5e-6, ixy=0.0,
            cx=0.0, cy=0.0
        )
