"""
Interface base para análise de seções transversais.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class SectionProperties:
    """Propriedades geométricas de seção."""
    
    # Geométricas
    area: float                    # m²
    perimeter: float              # m
    ixx: float                    # m⁴ (momento de inércia eixo x)
    iyy: float                    # m⁴ (momento de inércia eixo y)
    ixy: float                    # m⁴ (produto de inércia)
    j: float                      # m⁴ (constante torsional)
    cx: float                     # m (centroide x)
    cy: float                     # m (centroide y)
    
    # Módulos resistentes
    zxx_plus: Optional[float] = None   # m³ (módulo plástico)
    zxx_minus: Optional[float] = None
    zyy_plus: Optional[float] = None
    zyy_minus: Optional[float] = None
    
    # Raios de giração
    rx: Optional[float] = None    # m
    ry: Optional[float] = None    # m
    
    # Propriedades principais
    i11: Optional[float] = None   # m⁴
    i22: Optional[float] = None   # m⁴
    phi: Optional[float] = None   # rad (ângulo eixos principais)


class SectionAnalyzer(ABC):
    """
    Interface abstrata para análise de seções.
    
    Implementações concretas: SteelSection, ConcreteSection, CompositeSection
    """
    
    def __init__(self, name: str):
        self.name = name
        self._properties: Optional[SectionProperties] = None
        self._is_analyzed = False
    
    @abstractmethod
    def build_geometry(self, **kwargs) -> None:
        """
        Constrói geometria da seção.
        
        Args:
            **kwargs: Parâmetros específicos da seção
        """
        pass
    
    @abstractmethod
    def calculate_properties(self) -> SectionProperties:
        """
        Calcula propriedades geométricas.
        
        Returns:
            SectionProperties com todas as propriedades calculadas
        """
        pass
    
    @abstractmethod
    def plot_geometry(self, filename: Optional[str] = None) -> None:
        """
        Plota geometria da seção.
        
        Args:
            filename: Salvar em arquivo (se None, exibe interativo)
        """
        pass
    
    def get_properties(self) -> SectionProperties:
        """Retorna propriedades (calcula se necessário)."""
        if not self._is_analyzed:
            self._properties = self.calculate_properties()
            self._is_analyzed = True
        return self._properties
    
    def export_to_dict(self) -> Dict[str, Any]:
        """Exporta seção para dicionário."""
        props = self.get_properties()
        return {
            'name': self.name,
            'type': self.__class__.__name__,
            'properties': {
                'area': props.area,
                'ixx': props.ixx,
                'iyy': props.iyy,
                'j': props.j,
                'cx': props.cx,
                'cy': props.cy
            }
        }
