"""Seção composta (stub temporário)."""
from .base import SectionAnalyzer, SectionProperties


class CompositeSection(SectionAnalyzer):
    """Stub temporário para CompositeSection."""
    
    def __init__(self, name: str, **kwargs):
        super().__init__(name=name, section_type='composite')
    
    def build_geometry(self, **kwargs):
        self.geometry_built = True
    
    def calculate_properties(self) -> SectionProperties:
        return SectionProperties(
            area=0.03, perimeter=1.0,
            ixx=3e-5, iyy=1.5e-5, ixy=0.0,
            cx=0.0, cy=0.0
        )
