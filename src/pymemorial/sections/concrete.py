"""
Análise de seções de concreto armado usando concreteproperties.
Compatível com NBR 6118:2023 e outras normas internacionais.
"""
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np
import matplotlib.pyplot as plt

try:
    from concreteproperties import (
        Concrete,
        Steel,
        RectangularSection,
        ConcreteSection as CPConcreteSection,
        add_bar,
        add_bar_rectangular_array
    )
    from concreteproperties.stress_strain_profile import (
        ConcreteLinear,
        ConcreteServiceEurocode,
        ConcreteUltimateEurocode,
        SteelElasticPlastic,
        SteelHardening
    )
    CONCRETEPROPERTIES_AVAILABLE = True
except ImportError:
    CONCRETEPROPERTIES_AVAILABLE = False

from .section_base import SectionAnalyzer, SectionProperties


class ConcreteGrade(Enum):
    """Classes de resistência do concreto (NBR 6118:2023)."""
    C20 = 20
    C25 = 25
    C30 = 30
    C35 = 35
    C40 = 40
    C45 = 45
    C50 = 50
    C60 = 60
    C70 = 70
    C80 = 80
    C90 = 90


class SteelGrade(Enum):
    """Classes de aço para armadura (NBR 6118:2023)."""
    CA25 = (250, 210000)
    CA50 = (500, 210000)
    CA60 = (600, 210000)


class AnalysisType(Enum):
    """Tipo de análise."""
    GROSS = "gross"
    CRACKED = "cracked"
    ULTIMATE = "ultimate"


class ConcreteSection(SectionAnalyzer):
    """Seção de concreto armado usando concreteproperties."""
    
    def __init__(self, name: str, fc: float = 30.0, fy: float = 500.0, 
                 Ec: Optional[float] = None, Es: float = 210000.0,
                 concrete_grade: Optional[ConcreteGrade] = None,
                 steel_grade: Optional[SteelGrade] = None,
                 code: str = "NBR6118:2023", **kwargs):
        if not CONCRETEPROPERTIES_AVAILABLE:
            raise ImportError("concreteproperties não está instalado.")
        
        super().__init__(name=name, section_type='concrete')
        
        if concrete_grade:
            fc = concrete_grade.value
        if steel_grade:
            fy, Es = steel_grade.value
        
        self.fc = fc
        self.fy = fy
        self.Es = Es
        
        if Ec is None:
            alpha_E = 1.2 if fc <= 50 else 1.0
            self.Ec = alpha_E * 5600 * np.sqrt(fc)
        else:
            self.Ec = Ec
        
        self.code = code
        self.geometry = None
        self.concrete_section = None
        self.rebars: List[Dict] = []
        self._concrete_material = None
        self._steel_material = None
    
    def _create_materials(self, analysis_type: AnalysisType = AnalysisType.ULTIMATE):
        """Cria materiais conforme tipo de análise e norma."""
        if self.code.startswith("NBR"):
            if analysis_type == AnalysisType.ULTIMATE:
                fcd = self.fc / 1.4
                self._concrete_material = Concrete(
                    name=f"C{int(self.fc)}",
                    density=2500,
                    stress_strain_profile=ConcreteUltimateEurocode(
                        compressive_strength=fcd,
                        compressive_strain=0.0035,
                        tensile_strength=0,
                        elastic_modulus=self.Ec,
                        ultimate_strain=0.0035,
                        alpha=0.85,
                        gamma=1.4,
                        n=2
                    )
                )
            else:
                self._concrete_material = Concrete(
                    name=f"C{int(self.fc)}",
                    density=2500,
                    stress_strain_profile=ConcreteLinear(elastic_modulus=self.Ec)
                )
        
        if analysis_type == AnalysisType.ULTIMATE:
            fyd = self.fy / 1.15
            self._steel_material = Steel(
                name=f"CA-{int(self.fy)}",
                density=7850,
                stress_strain_profile=SteelElasticPlastic(
                    yield_strength=fyd,
                    elastic_modulus=self.Es,
                    fracture_strain=0.01
                )
            )
        else:
            self._steel_material = Steel(
                name=f"CA-{int(self.fy)}",
                density=7850,
                stress_strain_profile=SteelElasticPlastic(
                    yield_strength=self.fy,
                    elastic_modulus=self.Es,
                    fracture_strain=0.01
                )
            )
    
    def build_rectangular(self, b: float, h: float) -> None:
        """Constrói seção retangular."""
        self._create_materials()
        self.geometry = RectangularSection(b=b*1000, d=h*1000, material=self._concrete_material)
    
    def add_rebar_bottom(self, diameter: float, n_bars: int, cover: float, 
                        spacing: Optional[float] = None) -> None:
        """Adiciona armadura inferior."""
        if self.geometry is None:
            raise RuntimeError("Geometria não foi construída.")
        
        b = self.geometry.b
        d_linha = cover + diameter / 2
        
        if spacing is None:
            spacing = (b - 2*cover - n_bars*diameter) / (n_bars-1) if n_bars > 1 else 0
        
        for i in range(n_bars):
            x = cover + diameter/2 + i*(spacing + diameter)
            y = d_linha
            self.geometry = add_bar(self.geometry, np.pi*(diameter/2)**2, 
                                   self._steel_material, x, y)
        
        self.rebars.append({'type': 'bottom', 'diameter': diameter, 
                           'n_bars': n_bars, 'cover': cover})
    
    def add_rebar_top(self, diameter: float, n_bars: int, cover: float,
                     spacing: Optional[float] = None) -> None:
        """Adiciona armadura superior."""
        if self.geometry is None:
            raise RuntimeError("Geometria não foi construída.")
        
        b = self.geometry.b
        h = self.geometry.d
        d_linha = h - cover - diameter/2
        
        if spacing is None:
            spacing = (b - 2*cover - n_bars*diameter) / (n_bars-1) if n_bars > 1 else 0
        
        for i in range(n_bars):
            x = cover + diameter/2 + i*(spacing + diameter)
            y = d_linha
            self.geometry = add_bar(self.geometry, np.pi*(diameter/2)**2,
                                   self._steel_material, x, y)
        
        self.rebars.append({'type': 'top', 'diameter': diameter,
                           'n_bars': n_bars, 'cover': cover})
    
    def calculate_properties(self, analysis_type: AnalysisType = AnalysisType.GROSS) -> SectionProperties:
        """Calcula propriedades da seção."""
        if self.geometry is None:
            raise RuntimeError("Geometria não foi construída.")
        
        self.concrete_section = CPConcreteSection(self.geometry)
        
        if analysis_type == AnalysisType.GROSS:
            props = self.concrete_section.get_gross_properties()
        elif analysis_type == AnalysisType.CRACKED:
            props = self.concrete_section.get_transformed_gross_properties()
        else:
            props = self.concrete_section.calculate_ultimate_section_actions()
        
        def safe(val, factor):
            return val * factor if val is not None else None
        
        return SectionProperties(
            area=safe(props.area, 1e-6) or 0,
            perimeter=safe(props.perimeter, 1e-3) or 0,
            ixx=safe(props.ixx_c, 1e-12) or 0,
            iyy=safe(props.iyy_c, 1e-12) or 0,
            ixy=safe(props.ixy_c, 1e-12) or 0,
            cx=safe(props.cx, 1e-3) or 0,
            cy=safe(props.cy, 1e-3) or 0
        )
    
    def moment_curvature_analysis(self, theta: float = 0.0, n: float = 0.0,
                                  kappa_inc: float = 1e-7, kappa_mult: float = 2.0,
                                  kappa_max: float = 5e-4, progress_bar: bool = False) -> Dict[str, np.ndarray]:
        """Análise momento-curvatura."""
        if self.concrete_section is None:
            self.calculate_properties(AnalysisType.ULTIMATE)
        
        mk_results = self.concrete_section.moment_curvature_analysis(
            theta=theta, n=n, kappa_inc=kappa_inc, kappa_mult=kappa_mult,
            kappa_max=kappa_max, progress_bar=progress_bar
        )
        
        return {'kappa': mk_results.kappa, 'moment': mk_results.m_xy, 'results': mk_results}
    
    def moment_interaction_diagram(self, max_comp: Optional[float] = None,
                                   max_tens: Optional[float] = None,
                                   progress_bar: bool = False) -> Dict[str, np.ndarray]:
        """Diagrama de interação P-M."""
        if self.concrete_section is None:
            self.calculate_properties(AnalysisType.ULTIMATE)
        
        mi_results = self.concrete_section.moment_interaction_diagram(
            max_comp=max_comp, max_tens=max_tens, progress_bar=progress_bar
        )
        
        return {'n': mi_results.n, 'moment': mi_results.m_xy, 'results': mi_results}
    
    def plot_geometry(self, filename: Optional[str] = None, **kwargs) -> None:
        """Plota geometria."""
        if self.concrete_section is None:
            self.calculate_properties()
        
        fig, ax = plt.subplots(figsize=(8, 8))
        self.concrete_section.plot_section(ax=ax, **kwargs)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(f"{self.name} - Seção de Concreto")
        
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_stress(self, m: float, n: float = 0.0, filename: Optional[str] = None) -> None:
        """Plota distribuição de tensões."""
        if self.concrete_section is None:
            self.calculate_properties(AnalysisType.ULTIMATE)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        self.concrete_section.plot_stress(m=m, n=n, ax=ax)
        
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
