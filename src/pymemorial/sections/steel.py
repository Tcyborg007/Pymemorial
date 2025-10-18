"""
Análise de seções de aço usando sectionproperties.
"""
from typing import Optional, Dict, Any
import matplotlib.pyplot as plt

try:
    from sectionproperties.pre.library import (
        i_section,
        circular_hollow_section,
        rectangular_section,
        channel_section
    )
    from sectionproperties.analysis import Section
    SECTIONPROPERTIES_AVAILABLE = True
except ImportError:
    SECTIONPROPERTIES_AVAILABLE = False

from .section_base import SectionAnalyzer, SectionProperties


class SteelSection(SectionAnalyzer):
    """
    Seção de aço usando sectionproperties 3.x.
    
    Suporta:
    - Perfis I, H, U, C
    - Seções tubulares (circulares e retangulares)
    - Seções retangulares maciças
    - Propriedades elásticas e plásticas
    """
    
    def __init__(
        self,
        name: str,
        E: float = 200e9,
        nu: float = 0.3,
        fy: float = 250e6,
        **kwargs
    ):
        if not SECTIONPROPERTIES_AVAILABLE:
            raise ImportError(
                "sectionproperties não está instalado. "
                "Instale com: pip install 'pymemorial[sections]'"
            )
        
        super().__init__(name=name, section_type='steel')
        
        self.E = E
        self.nu = nu
        self.fy = fy
        
        self.geometry = None
        self.section = None
    
    def build_i_section(self, d: float, b: float, tf: float, tw: float, r: float = 0.0) -> None:
        """Constrói perfil I."""
        self.geometry = i_section(
            d=d*1000, b=b*1000, t_f=tf*1000, 
            t_w=tw*1000, r=r*1000, n_r=8
        )
    
    def build_circular_hollow(self, d: float, t: float) -> None:
        """Constrói seção tubular circular."""
        self.geometry = circular_hollow_section(d=d*1000, t=t*1000, n=64)
    
    def build_rectangular(self, b: float, d: float) -> None:
        """Constrói seção retangular maciça."""
        self.geometry = rectangular_section(b=b*1000, d=d*1000)
    
    def build_channel(self, d: float, b: float, tf: float, tw: float, r: float = 0.0) -> None:
        """Constrói perfil U (channel)."""
        self.geometry = channel_section(
            d=d*1000, b=b*1000, t_f=tf*1000,
            t_w=tw*1000, r=r*1000, n_r=8
        )
    
    def build_geometry(self, section_type: str, **kwargs) -> None:
        """Constrói geometria parametrizada."""
        builders = {
            'i': self.build_i_section,
            'circular_hollow': self.build_circular_hollow,
            'rectangular': self.build_rectangular,
            'channel': self.build_channel
        }
        
        if section_type not in builders:
            raise ValueError(
                f"Tipo '{section_type}' não suportado. "
                f"Tipos válidos: {list(builders.keys())}"
            )
        
        builders[section_type](**kwargs)
    
    def calculate_properties(self) -> SectionProperties:
        """Calcula propriedades geométricas da seção."""
        if self.geometry is None:
            raise RuntimeError(
                "Geometria não foi construída. Use build_geometry() ou build_*() primeiro."
            )
        
        # Criar mesh
        self.geometry.create_mesh(mesh_sizes=0)
        
        # Criar Section e calcular
        self.section = Section(self.geometry)
        self.section.calculate_geometric_properties()
        self.section.calculate_plastic_properties()
        
        # Acessar propriedades
        sp = self.section.section_props
        
        # Helper para converter
        def safe_convert(value, factor):
            return value * factor if value is not None else None
        
        return SectionProperties(
            area=safe_convert(sp.area, 1e-6) or 0,
            perimeter=safe_convert(sp.perimeter, 1e-3) or 0,
            ixx=safe_convert(sp.ixx_c, 1e-12) or 0,
            iyy=safe_convert(sp.iyy_c, 1e-12) or 0,
            ixy=safe_convert(sp.ixy_c, 1e-12) or 0,
            cx=safe_convert(sp.cx, 1e-3) or 0,
            cy=safe_convert(sp.cy, 1e-3) or 0,
            j=safe_convert(sp.j, 1e-12),
            zxx=safe_convert(sp.zxx_plus, 1e-9),
            zyy=safe_convert(sp.zyy_plus, 1e-9),
            rx=safe_convert(sp.rx_c, 1e-3),
            ry=safe_convert(sp.ry_c, 1e-3),
            i11=safe_convert(sp.i11_c, 1e-12),
            i22=safe_convert(sp.i22_c, 1e-12),
            phi=sp.phi if sp.phi is not None else 0
        )
    
    def plot_geometry(self, filename: Optional[str] = None, **kwargs) -> None:
        """Plota geometria da seção."""
        if self.section is None:
            self.calculate_properties()
        
        fig, ax = plt.subplots(figsize=(8, 8))
        self.section.plot_geometry(ax=ax, **kwargs)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(f"{self.name} - Geometria")
        
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def get_stress_at_yield(self) -> Dict[str, float]:
        """Calcula momentos de escoamento."""
        props = self.get_properties()
        
        return {
            'Mx_yield': self.fy * props.zxx if props.zxx else 0,
            'My_yield': self.fy * props.zyy if props.zyy else 0
        }
