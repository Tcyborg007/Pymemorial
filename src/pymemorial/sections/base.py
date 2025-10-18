"""
Interface base para análise de seções transversais.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class SectionProperties:
    """
    Propriedades geométricas de seção transversal.
    
    Todas as unidades em SI (metros, m², m³, m⁴).
    """
    
    # Campos obrigatórios
    area: float
    perimeter: float
    ixx: float
    iyy: float
    ixy: float
    cx: float
    cy: float
    
    # Campos opcionais
    j: Optional[float] = None
    wxx_top: Optional[float] = None
    wxx_bot: Optional[float] = None
    wyy_left: Optional[float] = None
    wyy_right: Optional[float] = None
    zxx: Optional[float] = None
    zyy: Optional[float] = None
    rx: Optional[float] = None
    ry: Optional[float] = None
    i11: Optional[float] = None
    i22: Optional[float] = None
    phi: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário, removendo None."""
        return {k: v for k, v in asdict(self).items() if v is not None}


class SectionAnalyzer(ABC):
    """Interface abstrata para análise de seções transversais."""
    
    def __init__(self, name: str, section_type: str):
        self.name = name
        self.section_type = section_type
        self._properties: Optional[SectionProperties] = None
        self._is_analyzed = False
    
    @abstractmethod
    def build_geometry(self, **kwargs) -> None:
        """Constrói geometria da seção."""
        pass
    
    @abstractmethod
    def calculate_properties(self) -> SectionProperties:
        """Calcula propriedades geométricas da seção."""
        pass
    
    def get_properties(self) -> SectionProperties:
        """Retorna propriedades (calcula se necessário)."""
        if not self._is_analyzed:
            self._properties = self.calculate_properties()
            self._is_analyzed = True
        return self._properties
    
    def export_to_dict(self) -> Dict[str, Any]:
        """Exporta seção e propriedades para dicionário."""
        props = self.get_properties()
        return {
            'name': self.name,
            'type': self.section_type,
            'analyzer': self.__class__.__name__,
            'properties': props.to_dict()
        }
    
    def __repr__(self) -> str:
        status = "analyzed" if self._is_analyzed else "not analyzed"
        return f"{self.__class__.__name__}(name='{self.name}', type='{self.section_type}', {status})"
