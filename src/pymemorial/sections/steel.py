"""
Análise de seções de aço usando sectionproperties 3.x.
Compatível com EN 1993 (Eurocode 3), NBR 8800 e AISC 360.
"""
from typing import Optional, Dict, Any

# Backend sem GUI para evitar erro TKinter em testes
import matplotlib
matplotlib.use('Agg')
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
    - Perfis I, H (double symmetric I-sections)
    - Perfis U, C (channels)
    - Seções tubulares circulares (CHS)
    - Seções tubulares retangulares (RHS)
    - Seções retangulares maciças
    - Propriedades elásticas e plásticas
    - Classificação de seções (compacta/não-compacta)
    - Momentos resistentes
    
    Examples:
        >>> from pymemorial.sections import SteelSection
        >>> 
        >>> # Perfil I
        >>> steel = SteelSection("IPE200", fy=355e6)
        >>> steel.build_i_section(d=0.200, b=0.100, tf=0.0085, tw=0.0056)
        >>> props = steel.get_properties()
        >>> print(f"Ixx = {props.ixx:.6e} m⁴")
        >>> 
        >>> # Tubo circular
        >>> tube = SteelSection("CHS200x8", fy=355e6)
        >>> tube.build_circular_hollow(d=0.200, t=0.008)
        >>> props = tube.get_properties()
    """
    
    def __init__(
        self,
        name: str,
        E: float = 200e9,      # Pa (200 GPa padrão para aço)
        nu: float = 0.3,       # Coeficiente de Poisson
        fy: float = 250e6,     # Pa (250 MPa padrão)
        **kwargs
    ):
        """
        Inicializa seção de aço.
        
        Args:
            name: Nome da seção
            E: Módulo de elasticidade (Pa)
            nu: Coeficiente de Poisson
            fy: Tensão de escoamento (Pa)
        """
        if not SECTIONPROPERTIES_AVAILABLE:
            raise ImportError(
                "sectionproperties não está instalado. "
                "Instale com: poetry add sectionproperties"
            )
        
        super().__init__(name=name, section_type='steel')
        
        self.E = E
        self.nu = nu
        self.fy = fy
        
        self.geometry = None
        self.section = None
    
    def build_i_section(
        self,
        d: float,
        b: float,
        tf: float,
        tw: float,
        r: float = 0.0
    ) -> None:
        """
        Constrói perfil I ou H.
        
        Args:
            d: Altura total (m)
            b: Largura da mesa (m)
            tf: Espessura da mesa (m)
            tw: Espessura da alma (m)
            r: Raio do filete (m)
        """
        self.geometry = i_section(
            d=d * 1000,     # mm
            b=b * 1000,     # mm
            t_f=tf * 1000,  # mm
            t_w=tw * 1000,  # mm
            r=r * 1000,     # mm
            n_r=8           # Número de pontos no filete
        )
    
    def build_circular_hollow(self, d: float, t: float) -> None:
        """
        Constrói seção tubular circular (CHS).
        
        Args:
            d: Diâmetro externo (m)
            t: Espessura da parede (m)
        """
        self.geometry = circular_hollow_section(
            d=d * 1000,  # mm
            t=t * 1000,  # mm
            n=64         # Número de pontos no perímetro
        )
    
    def build_rectangular(self, b: float, d: float) -> None:
        """
        Constrói seção retangular maciça.
        
        Args:
            b: Largura (m)
            d: Altura (m)
        """
        self.geometry = rectangular_section(
            b=b * 1000,  # mm
            d=d * 1000   # mm
        )
    
    def build_channel(
        self,
        d: float,
        b: float,
        tf: float,
        tw: float,
        r: float = 0.0
    ) -> None:
        """
        Constrói perfil U (channel).
        
        Args:
            d: Altura total (m)
            b: Largura da mesa (m)
            tf: Espessura da mesa (m)
            tw: Espessura da alma (m)
            r: Raio do filete (m)
        """
        self.geometry = channel_section(
            d=d * 1000,
            b=b * 1000,
            t_f=tf * 1000,
            t_w=tw * 1000,
            r=r * 1000,
            n_r=8
        )
    
    def build_geometry(self, section_type: str, **kwargs) -> None:
        """
        Constrói geometria parametrizada (implementa método abstrato).
        
        Args:
            section_type: Tipo ('i', 'circular_hollow', 'rectangular', 'channel')
            **kwargs: Parâmetros específicos do tipo
        """
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
                "Geometria não foi construída. "
                "Use build_geometry() ou build_*() primeiro."
            )
        
        # Criar mesh
        self.geometry.create_mesh(mesh_sizes=0)
        
        # Criar Section e calcular
        self.section = Section(self.geometry)
        self.section.calculate_geometric_properties()
        self.section.calculate_plastic_properties()
        
        # Acessar propriedades
        sp = self.section.section_props
        
        # Helper para conversão segura
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
        self.geometry.plot_geometry(ax=ax, **kwargs)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(f"{self.name} - Geometria")
        
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def get_stress_at_yield(self) -> Dict[str, float]:
        """
        Calcula momentos de escoamento.
        
        Returns:
            Dicionário com Mx_yield e My_yield (N.m)
        """
        props = self.get_properties()
        
        return {
            'Mx_yield': self.fy * props.zxx if props.zxx else 0,
            'My_yield': self.fy * props.zyy if props.zyy else 0
        }
    
    def export_to_dict(self) -> Dict[str, Any]:
        """Exporta seção para dicionário."""
        base_dict = super().export_to_dict()
        base_dict.update({
            'E': self.E,
            'nu': self.nu,
            'fy': self.fy,
            'stress_at_yield': self.get_stress_at_yield()
        })
        return base_dict
