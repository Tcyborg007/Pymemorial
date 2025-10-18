"""
Análise de seções mistas (aço-concreto) usando sectionproperties.
Compatível com EN 1994 (Eurocode 4) 2025, NBR 8800:2024 e AISC 360.
"""
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np

# Configurar matplotlib para backend sem GUI
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from sectionproperties.pre.library import (
        i_section,
        circular_hollow_section,
        circular_section
    )
    from sectionproperties.pre import Material
    from sectionproperties.pre.geometry import Geometry
    from sectionproperties.analysis import Section
    from shapely.geometry import Polygon
    SECTIONPROPERTIES_AVAILABLE = True
except ImportError:
    SECTIONPROPERTIES_AVAILABLE = False

from .section_base import SectionAnalyzer, SectionProperties


class CompositeType(Enum):
    """Tipos de seção composta."""
    COMPOSITE_BEAM = "composite_beam"
    ENCASED_BEAM = "encased_beam"
    FILLED_COLUMN = "filled_column"
    ENCASED_COLUMN = "encased_column"
    PARTIALLY_ENCASED = "partially_encased"


class ShearConnectorType(Enum):
    """Tipos de conectores de cisalhamento."""
    HEADED_STUD = "headed_stud"
    ANGLE_CONNECTOR = "angle"
    CHANNEL_CONNECTOR = "channel"
    PERFOBOND = "perfobond"


@dataclass
class ShearConnectorProperties:
    """Propriedades de conectores de cisalhamento."""
    type: ShearConnectorType
    diameter: float
    height: float
    fu: float
    spacing: float
    n_per_row: int = 2


class CompositeSection(SectionAnalyzer):
    """
    Seção mista aço-concreto com suporte completo NBR 8800:2024.
    
    Examples:
        >>> viga = CompositeSection("VM-1", code="NBR8800:2024")
        >>> viga.build_composite_beam(steel_profile="W310x52")
        >>> viga.plot_geometry(filename="viga.png")
    """
    
    def __init__(
        self,
        name: str,
        composite_type: CompositeType = CompositeType.COMPOSITE_BEAM,
        steel_fy: float = 355e6,
        concrete_fck: float = 30e6,
        code: str = "EC4:2025",
        **kwargs
    ):
        if not SECTIONPROPERTIES_AVAILABLE:
            raise ImportError("sectionproperties não está instalado.")
        
        super().__init__(name=name, section_type='composite')
        
        self.composite_type = composite_type
        self.steel_fy = steel_fy
        self.concrete_fck = concrete_fck
        self.code = code
        
        self.steel_material = None
        self.concrete_material = None
        self.steel_geometry = None
        self.concrete_geometry = None
        self.compound_geometry = None
        self.geometry = None
        
        self.shear_connectors: List[ShearConnectorProperties] = []
        self.n0 = 0
        self.creep_factor = 0
        self.shrinkage_strain = 0
        
        # NBR 8800:2024
        self._outer_d = None
        self._wall_t = None
        self._section_class = None
        
        # Identificação de perfis
        self._steel_profile_name = None  # ← NOVO: armazena nome do perfil
    
    def build_geometry(self, geometry_type: str = "composite_beam", **kwargs) -> None:
        """Constrói geometria parametrizada."""
        if geometry_type == "composite_beam":
            self.build_composite_beam(**kwargs)
        elif geometry_type == "filled_column":
            self.build_filled_column(**kwargs)
        else:
            raise ValueError(f"Tipo '{geometry_type}' não suportado.")
    
    def _create_materials(self):
        """Cria materiais aço e concreto."""
        Es = 210e9
        
        self.steel_material = Material(
            name=f"Steel_S{int(self.steel_fy/1e6)}",
            elastic_modulus=Es / 1e6,
            poissons_ratio=0.3,
            density=7850e-9,
            yield_strength=self.steel_fy / 1e6,
            color="lightblue"
        )
        
        fcm = self.concrete_fck / 1e6 + 8
        
        if self.code.startswith("NBR"):
            alpha_E = 1.2 if fcm <= 50 else 1.0
            Ec = alpha_E * 5600 * np.sqrt(self.concrete_fck / 1e6)
        else:
            Ec = 22 * (fcm / 10) ** 0.3 * 1000
        
        self.concrete_material = Material(
            name=f"Concrete_C{int(self.concrete_fck/1e6)}",
            elastic_modulus=Ec,
            poissons_ratio=0.2,
            density=2500e-9,
            yield_strength=self.concrete_fck / 1e6,
            color="lightgray"
        )
        
        self.n0 = Es / (Ec * 1e6)
    
    def build_composite_beam(
        self,
        steel_profile: str = "W310x52",
        steel_d: Optional[float] = None,
        steel_b: Optional[float] = None,
        steel_tf: Optional[float] = None,
        steel_tw: Optional[float] = None,
        slab_width: float = 2.5,
        slab_height: float = 0.15,
        concrete_fc: float = 30,
        haunch_height: float = 0.0
    ) -> None:
        """Constrói viga mista."""
        self._create_materials()
        
        # Armazenar nome do perfil para plotagem
        self._steel_profile_name = steel_profile
        
        if steel_d:
            self.steel_geometry = i_section(
                d=steel_d * 1000,
                b=steel_b * 1000,
                t_f=steel_tf * 1000,
                t_w=steel_tw * 1000,
                r=12,
                n_r=8,
                material=self.steel_material
            )
        else:
            self.steel_geometry = i_section(
                d=317.5, b=166.9, t_f=13.2, t_w=7.6, r=12, n_r=8,
                material=self.steel_material
            )
        
        slab_w_mm = slab_width * 1000
        slab_h_mm = slab_height * 1000
        steel_height = self.steel_geometry.calculate_extents()[3]
        
        y_bottom = steel_height + haunch_height * 1000
        y_top = y_bottom + slab_h_mm
        
        slab_coords = [
            (-slab_w_mm / 2, y_bottom),
            (slab_w_mm / 2, y_bottom),
            (slab_w_mm / 2, y_top),
            (-slab_w_mm / 2, y_top)
        ]
        
        self.concrete_geometry = Geometry(
            geom=Polygon(slab_coords),
            material=self.concrete_material
        )
        
        self.compound_geometry = self.steel_geometry + self.concrete_geometry
        self.geometry = self.compound_geometry
    
    def build_filled_column(
        self,
        outer_diameter: float,
        wall_thickness: float,
        concrete_fc: float = 40
    ) -> None:
        """Constrói pilar tubular preenchido."""
        self._create_materials()
        
        D_mm = outer_diameter * 1000
        t_mm = wall_thickness * 1000
        
        self._outer_d = D_mm
        self._wall_t = t_mm
        
        # Armazenar identificação do tubo
        self._steel_profile_name = f"Ø{D_mm:.0f}x{t_mm:.1f}mm"
        
        self.steel_geometry = circular_hollow_section(
            d=D_mm, t=t_mm, n=64, material=self.steel_material
        )
        
        D_inner = D_mm - 2 * t_mm
        self.concrete_geometry = circular_section(
            d=D_inner, n=64, material=self.concrete_material
        )
        
        self.compound_geometry = self.steel_geometry + self.concrete_geometry
        self.geometry = self.compound_geometry
    
    def add_shear_connectors(
        self,
        type: ShearConnectorType = ShearConnectorType.HEADED_STUD,
        diameter: float = 19,
        height: float = 100,
        fu: float = 450,
        spacing: float = 150,
        n_per_row: int = 2
    ) -> None:
        """Adiciona conectores de cisalhamento."""
        connector = ShearConnectorProperties(
            type=type, diameter=diameter, height=height,
            fu=fu, spacing=spacing, n_per_row=n_per_row
        )
        self.shear_connectors.append(connector)
    
    def calculate_shear_connector_capacity(self, connector: ShearConnectorProperties) -> float:
        """Calcula capacidade de conector."""
        if self.code.startswith("NBR"):
            return self.calculate_connector_resistance_nbr8800(connector)
        else:
            if connector.type == ShearConnectorType.HEADED_STUD:
                d = connector.diameter
                fu = connector.fu
                As = np.pi * (d ** 2) / 4
                PRk = 0.8 * fu * As / 1000
                PRd = PRk / 1.25
                return PRd
            else:
                raise NotImplementedError(f"Conector {connector.type} não implementado.")
    
    # ========== NBR 8800:2024 ==========
    
    def classify_section_nbr8800(self) -> str:
        """Classifica seção conforme NBR 8800:2024."""
        if self.composite_type != CompositeType.FILLED_COLUMN:
            raise NotImplementedError("Classificação apenas para pilares preenchidos.")
        
        if self._outer_d is None or self._wall_t is None:
            raise RuntimeError("Geometria não construída.")
        
        D = self._outer_d
        t = self._wall_t
        fy_mpa = self.steel_fy / 1e6
        E = 210000
        
        lambda_compacta = 0.15 * np.sqrt(E / fy_mpa)
        lambda_semicompacta = 0.19 * np.sqrt(E / fy_mpa)
        
        lambda_atual = D / t
        
        if lambda_atual <= lambda_compacta:
            self._section_class = "compacta"
        elif lambda_atual <= lambda_semicompacta:
            self._section_class = "semicompacta"
        else:
            self._section_class = "esbelta"
        
        return self._section_class
    
    def calculate_nbr8800_stiffness_reduction(self) -> float:
        """Redução de rigidez NBR 8800:2024."""
        if self.code.startswith("NBR") and self.composite_type in [
            CompositeType.FILLED_COLUMN,
            CompositeType.ENCASED_COLUMN
        ]:
            return 0.64
        return 1.0
    
    def calculate_connector_resistance_nbr8800(self, connector: ShearConnectorProperties) -> float:
        """Resistência de conector NBR 8800:2024 Anexo Q."""
        if connector.type == ShearConnectorType.HEADED_STUD:
            d = connector.diameter
            fu = connector.fu
            
            Ec_mpa = self.concrete_material.elastic_modulus if self.concrete_material else 30000
            fck = self.concrete_fck / 1e6
            
            As = np.pi * (d ** 2) / 4
            
            QRk1 = 0.8 * fu * As
            QRk2 = 0.45 * (d ** 2) * np.sqrt(fck * Ec_mpa)
            
            QRk = min(QRk1, QRk2) / 1000
            QRd = QRk / 1.25
            
            return QRd
        else:
            raise NotImplementedError(f"Conector {connector.type} não implementado.")
    
    def get_nbr8800_info(self) -> Dict[str, Any]:
        """Informações NBR 8800:2024."""
        info = {
            'code': self.code,
            'stiffness_reduction': self.calculate_nbr8800_stiffness_reduction()
        }
        
        if self.composite_type == CompositeType.FILLED_COLUMN and self._outer_d:
            try:
                info['classification'] = self.classify_section_nbr8800()
                info['D/t_ratio'] = self._outer_d / self._wall_t
                info['D_mm'] = self._outer_d
                info['t_mm'] = self._wall_t
            except:
                pass
        
        if self.shear_connectors and self.code.startswith("NBR"):
            info['connector_capacities_kN'] = [
                self.calculate_connector_resistance_nbr8800(c)
                for c in self.shear_connectors
            ]
        
        return info
    
    # ========== PROPRIEDADES ==========
    
    def calculate_composite_properties(self, include_long_term: bool = True) -> SectionProperties:
        """Calcula propriedades da seção composta."""
        if self.geometry is None:
            raise RuntimeError("Geometria não construída.")
        
        self.geometry.create_mesh(mesh_sizes=0)
        section = Section(self.geometry)
        section.calculate_geometric_properties()
        section.calculate_warping_properties()
        
        props = section.section_props
        reduction = self.calculate_nbr8800_stiffness_reduction()
        
        def safe(val, factor):
            return val * factor if val is not None else None
        
        return SectionProperties(
            area=safe(props.area, 1e-6) or 0,
            perimeter=safe(props.perimeter, 1e-3) or 0,
            ixx=safe(props.ixx_c * reduction, 1e-12) or 0,
            iyy=safe(props.iyy_c * reduction, 1e-12) or 0,
            ixy=safe(props.ixy_c * reduction, 1e-12) or 0,
            cx=safe(props.cx, 1e-3) or 0,
            cy=safe(props.cy, 1e-3) or 0,
            j=safe(props.j * reduction, 1e-12),
            zxx=safe(props.zxx_plus, 1e-9),
            zyy=safe(props.zyy_plus, 1e-9),
            rx=safe(props.rx_c * np.sqrt(reduction), 1e-3),
            ry=safe(props.ry_c * np.sqrt(reduction), 1e-3)
        )
    
    def calculate_properties(self) -> SectionProperties:
        """Alias para calculate_composite_properties()."""
        return self.calculate_composite_properties()
    
    # ========== PLOTAGEM PROFISSIONAL ==========
    
    def plot_geometry(
        self,
        filename: Optional[str] = None,
        show_dimensions: bool = True,
        show_materials: bool = True,
        show_properties: bool = False,
        material_colors: bool = True,
        **kwargs
    ) -> None:
        """Plota geometria com estilo profissional."""
        if self.geometry is None:
            self.calculate_properties()
        
        self.geometry.create_mesh(mesh_sizes=0)
        
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111)
        
        if material_colors:
            self._plot_materials_colored(ax)
        else:
            self.geometry.plot_geometry(ax=ax, **kwargs)
        
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2, linestyle=':', linewidth=0.5, color='gray')
        ax.set_axisbelow(True)
        
        title_parts = [f"{self.name} - Seção Mista ({self.code})"]
        if hasattr(self, '_section_class') and self._section_class:
            title_parts.append(f"Classificação: {self._section_class.upper()}")
        ax.set_title('\n'.join(title_parts), fontsize=15, fontweight='bold', pad=15)
        
        if show_materials:
            self._add_material_labels_v2(ax)
        
        if show_dimensions:
            self._add_dimensions_v2(ax)
        
        if show_properties:
            self._add_properties_table_v2(fig, ax)
        
        ax.set_xlabel('X (mm)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Y (mm)', fontsize=12, fontweight='bold')
        ax.tick_params(axis='both', labelsize=10)
        
        self._add_custom_legend(ax)
        
        plt.tight_layout()
        
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
        else:
            plt.show()
    
    def _plot_materials_colored(self, ax):
        """Plota materiais com cores diferenciadas."""
        steel_coords = self._get_geometry_coords(self.steel_geometry)
        concrete_coords = self._get_geometry_coords(self.concrete_geometry)
        
        from matplotlib.patches import Polygon as MPLPolygon
        
        # Label para legenda (nome do perfil ou tipo genérico)
        steel_label = f'{self._steel_profile_name} (S{int(self.steel_fy/1e6)})' if self._steel_profile_name else f'Aço S{int(self.steel_fy/1e6)}'
        
        steel_patch = MPLPolygon(
            steel_coords,
            facecolor='#B0C4DE',
            edgecolor='#000080',
            linewidth=2.5,
            label=steel_label,
            alpha=0.85,
            hatch='//'
        )
        ax.add_patch(steel_patch)
        
        concrete_patch = MPLPolygon(
            concrete_coords,
            facecolor='#D3D3D3',
            edgecolor='#505050',
            linewidth=2,
            label=f'Concreto C{int(self.concrete_fck/1e6)}',
            alpha=0.75,
            hatch='.'
        )
        ax.add_patch(concrete_patch)
        
        all_coords = np.vstack([steel_coords, concrete_coords])
        margin = 0.1 * (all_coords.max() - all_coords.min())
        ax.set_xlim(all_coords[:, 0].min() - margin, all_coords[:, 0].max() + margin)
        ax.set_ylim(all_coords[:, 1].min() - margin, all_coords[:, 1].max() + margin)
    
    def _get_geometry_coords(self, geometry):
        """Extrai coordenadas de uma geometria."""
        if hasattr(geometry.geom, 'exterior'):
            coords = np.array(geometry.geom.exterior.coords)
        elif hasattr(geometry, 'points'):
            coords = np.array(geometry.points)
        else:
            ext = geometry.calculate_extents()
            coords = np.array([
                [ext[0], ext[2]], [ext[1], ext[2]],
                [ext[1], ext[3]], [ext[0], ext[3]]
            ])
        return coords
    
    def _add_material_labels_v2(self, ax):
        """Anotações de materiais otimizadas."""
        steel_ext = self.steel_geometry.calculate_extents()
        concrete_ext = self.concrete_geometry.calculate_extents()
        
        if self.composite_type == CompositeType.COMPOSITE_BEAM:
            steel_cx = steel_ext[1] + 100
            steel_cy = (steel_ext[2] + steel_ext[3]) / 2
            
            # Usar nome do perfil se disponível
            steel_text = f'Perfil\n{self._steel_profile_name}\n(S{int(self.steel_fy/1e6)})' if self._steel_profile_name else f'Perfil\nAço S{int(self.steel_fy/1e6)}'
            
            ax.text(
                steel_cx, steel_cy, steel_text,
                fontsize=10, fontweight='bold', color='#000080',
                bbox=dict(boxstyle='round,pad=0.6', facecolor='#B0C4DE',
                         edgecolor='#000080', linewidth=2, alpha=0.9),
                ha='left', va='center'
            )
            
            conc_cx = (concrete_ext[0] + concrete_ext[1]) / 2
            conc_cy = concrete_ext[3] - 50
            
            ax.text(
                conc_cx, conc_cy, f'Laje Concreto C{int(self.concrete_fck/1e6)}',
                fontsize=10, fontweight='bold', color='#505050',
                bbox=dict(boxstyle='round,pad=0.6', facecolor='#D3D3D3',
                         edgecolor='#505050', linewidth=2, alpha=0.9),
                ha='center', va='center'
            )
        
        elif self.composite_type == CompositeType.FILLED_COLUMN:
            D = self._outer_d
            
            tubo_text = f'Tubo Aço\nS{int(self.steel_fy/1e6)}\n{self._steel_profile_name}' if self._steel_profile_name else f'Tubo Aço\nS{int(self.steel_fy/1e6)}\nØ{D:.0f}×{self._wall_t:.1f}mm'
            
            ax.text(
                -D/2 - 80, D/2 + 30, tubo_text,
                fontsize=10, fontweight='bold', color='#000080',
                bbox=dict(boxstyle='round,pad=0.6', facecolor='#B0C4DE',
                         edgecolor='#000080', linewidth=2, alpha=0.9),
                ha='right', va='center'
            )
            
            ax.text(
                D/2 + 80, -D/2 - 30, f'Núcleo\nConcreto\nC{int(self.concrete_fck/1e6)}',
                fontsize=10, fontweight='bold', color='#505050',
                bbox=dict(boxstyle='round,pad=0.6', facecolor='#D3D3D3',
                         edgecolor='#505050', linewidth=2, alpha=0.9),
                ha='left', va='center'
            )
    
    def _add_dimensions_v2(self, ax):
        """Cotas otimizadas."""
        if self.composite_type == CompositeType.COMPOSITE_BEAM:
            conc_ext = self.concrete_geometry.calculate_extents()
            steel_ext = self.steel_geometry.calculate_extents()
            
            b_eff = conc_ext[1] - conc_ext[0]
            y_cota = conc_ext[3] + 100
            
            ax.annotate('', xy=(conc_ext[1], y_cota), xytext=(conc_ext[0], y_cota),
                       arrowprops=dict(arrowstyle='<|-|>', color='red', lw=2.5,
                                      mutation_scale=20, shrinkA=0, shrinkB=0))
            
            ax.text((conc_ext[0] + conc_ext[1])/2, y_cota + 30,
                   f'b_eff = {b_eff:.0f} mm',
                   ha='center', va='bottom', fontsize=11, color='red', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                            edgecolor='red', linewidth=1.5))
            
            h_total = conc_ext[3] - steel_ext[2]
            x_cota = max(conc_ext[1], steel_ext[1]) + 150
            
            ax.annotate('', xy=(x_cota, conc_ext[3]), xytext=(x_cota, steel_ext[2]),
                       arrowprops=dict(arrowstyle='<|-|>', color='red', lw=2.5,
                                      mutation_scale=20, shrinkA=0, shrinkB=0))
            
            ax.text(x_cota + 40, (conc_ext[3] + steel_ext[2])/2,
                   f'h = {h_total:.0f} mm',
                   rotation=90, va='center', ha='left', fontsize=11,
                   color='red', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                            edgecolor='red', linewidth=1.5))
        
        elif self.composite_type == CompositeType.FILLED_COLUMN:
            D = self._outer_d
            
            ax.annotate('', xy=(D/2, 0), xytext=(-D/2, 0),
                       arrowprops=dict(arrowstyle='<|-|>', color='red', lw=2.5,
                                      mutation_scale=20, shrinkA=0, shrinkB=0))
            
            ax.text(0, -D/2 - 80, f'D = {D:.0f} mm',
                   ha='center', va='top', fontsize=12, color='red', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                            edgecolor='red', linewidth=2))
    
    def _add_properties_table_v2(self, fig, ax):
        """Tabela profissional."""
        props = self.get_properties()
        
        table_data = [
            ['Propriedade', 'Valor', 'Unidade'],
            ['Área', f'{props.area:.4f}', 'm²'],
            ['Ixx', f'{props.ixx:.3e}', 'm⁴'],
            ['Iyy', f'{props.iyy:.3e}', 'm⁴'],
            ['Zxx', f'{props.zxx:.3e}' if props.zxx else 'N/A', 'm³'],
        ]
        
        if self.code.startswith("NBR"):
            info = self.get_nbr8800_info()
            table_data.extend([
                ['Red. Rigidez', f'{info["stiffness_reduction"]:.0%}', '-'],
                ['n₀ (Es/Ec)', f'{self.n0:.2f}', '-']
            ])
        
        table = ax.table(
            cellText=table_data, cellLoc='center',
            bbox=[0.55, 0.02, 0.43, 0.30],
            colWidths=[0.45, 0.35, 0.20]
        )
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.8)
        
        for i, key in enumerate(table.get_celld().keys()):
            cell = table.get_celld()[key]
            if key[0] == 0:
                cell.set_facecolor('#2E7D32')
                cell.set_text_props(weight='bold', color='white', fontsize=10)
                cell.set_linewidth(2)
            else:
                cell.set_facecolor('#E8F5E9' if key[0] % 2 == 1 else '#F1F8F4')
                cell.set_linewidth(1)
            cell.set_edgecolor('#1B5E20')
    
    def _add_custom_legend(self, ax):
        """Legenda customizada."""
        from matplotlib.patches import Patch
        
        steel_label = f'{self._steel_profile_name} (S{int(self.steel_fy/1e6)})' if self._steel_profile_name else f'Aço S{int(self.steel_fy/1e6)}'
        
        legend_elements = [
            Patch(facecolor='#B0C4DE', edgecolor='#000080', linewidth=2,
                  label=steel_label, hatch='//'),
            Patch(facecolor='#D3D3D3', edgecolor='#505050', linewidth=2,
                  label=f'Concreto C{int(self.concrete_fck/1e6)}', hatch='.')
        ]
        
        ax.legend(handles=legend_elements, loc='upper right',
                 fontsize=11, framealpha=0.95, edgecolor='black',
                 fancybox=True, shadow=True)
    
    def export_to_dict(self) -> Dict[str, Any]:
        """Exporta seção para dicionário."""
        base_dict = super().export_to_dict()
        base_dict.update({
            'composite_type': self.composite_type.value,
            'steel_fy': self.steel_fy,
            'concrete_fck': self.concrete_fck,
            'steel_profile': self._steel_profile_name,
            'n_shear_connectors': len(self.shear_connectors),
            'code': self.code,
            'modular_ratio_n0': self.n0
        })
        
        if self.code.startswith("NBR"):
            base_dict['nbr8800_info'] = self.get_nbr8800_info()
        
        return base_dict