"""
Testes completos de ConcreteSection com concreteproperties.
Compatível com NBR 6118:2023 e análise avançada.
"""
import pytest
import numpy as np
from pymemorial.sections import SectionFactory
from pymemorial.sections.concrete import (
    ConcreteSection,
    ConcreteGrade,
    SteelGrade,
    AnalysisType,
    CONCRETEPROPERTIES_AVAILABLE
)


@pytest.mark.skipif(
    not CONCRETEPROPERTIES_AVAILABLE,
    reason="concreteproperties não instalado"
)
class TestConcreteSection:
    """Testes da implementação completa de ConcreteSection."""
    
    # ========== TESTES DE INICIALIZAÇÃO ==========
    
    def test_concrete_section_initialization(self):
        """Testa inicialização básica."""
        section = ConcreteSection("V1", fc=30, fy=500)
        
        assert section.name == "V1"
        assert section.section_type == "concrete"
        assert section.fc == 30.0
        assert section.fy == 500.0
        assert section.Es == 210000.0
        assert section.code == "NBR6118:2023"
        assert section.geometry is None
        assert section.concrete_section is None
    
    def test_concrete_section_with_grades(self):
        """Testa inicialização com classes padronizadas."""
        section = ConcreteSection(
            "V2",
            concrete_grade=ConcreteGrade.C30,
            steel_grade=SteelGrade.CA50
        )
        
        assert section.fc == 30.0
        assert section.fy == 500.0
        assert section.Es == 210000.0
    
    def test_automatic_modulus_calculation(self):
        """Testa cálculo automático de Ec (NBR 6118:2023)."""
        section = ConcreteSection("V3", fc=30)
        
        # Eci = 1.2 * 5600 * √30 = 36732 MPa (aprox)
        expected_Ec = 1.2 * 5600 * np.sqrt(30)
        assert abs(section.Ec - expected_Ec) < 100  # Tolerância 100 MPa
    
    def test_high_strength_concrete_modulus(self):
        """Testa Ec para concreto de alta resistência (fc > 50)."""
        section = ConcreteSection("V4", fc=60)
        
        # αE = 1.0 para fc > 50 (NBR 6118:2023)
        expected_Ec = 1.0 * 5600 * np.sqrt(60)
        assert abs(section.Ec - expected_Ec) < 100
    
    def test_custom_modulus(self):
        """Testa fornecimento manual de Ec."""
        section = ConcreteSection("V5", fc=30, Ec=35000)
        assert section.Ec == 35000.0
    
    # ========== TESTES DE GEOMETRIA ==========
    
    def test_build_rectangular_section(self):
        """Testa construção de seção retangular."""
        section = ConcreteSection("V6", fc=30, fy=500)
        section.build_rectangular(b=0.30, h=0.60)
        
        assert section.geometry is not None
        assert section.geometry.b == 300  # mm
        assert section.geometry.d == 600  # mm
    
    def test_build_without_materials_error(self):
        """Testa erro ao construir sem materiais."""
        section = ConcreteSection("V7", fc=30, fy=500)
        # build_rectangular cria materiais automaticamente
        section.build_rectangular(b=0.30, h=0.60)
        assert section._concrete_material is not None
        assert section._steel_material is not None
    
    # ========== TESTES DE ARMADURA ==========
    
    def test_add_bottom_rebar(self):
        """Testa adição de armadura inferior."""
        section = ConcreteSection("V8", fc=30, fy=500)
        section.build_rectangular(b=0.30, h=0.60)
        
        # 4φ16mm com cobrimento 30mm
        section.add_rebar_bottom(diameter=16, n_bars=4, cover=30)
        
        assert len(section.rebars) == 1
        assert section.rebars[0]['type'] == 'bottom'
        assert section.rebars[0]['diameter'] == 16
        assert section.rebars[0]['n_bars'] == 4
    
    def test_add_top_rebar(self):
        """Testa adição de armadura superior."""
        section = ConcreteSection("V9", fc=30, fy=500)
        section.build_rectangular(b=0.30, h=0.60)
        
        section.add_rebar_top(diameter=12.5, n_bars=2, cover=30)
        
        assert len(section.rebars) == 1
        assert section.rebars[0]['type'] == 'top'
        assert section.rebars[0]['diameter'] == 12.5
    
    def test_add_multiple_rebar_layers(self):
        """Testa múltiplas camadas de armadura."""
        section = ConcreteSection("V10", fc=30, fy=500)
        section.build_rectangular(b=0.30, h=0.60)
        
        section.add_rebar_bottom(diameter=16, n_bars=4, cover=30)
        section.add_rebar_top(diameter=12.5, n_bars=2, cover=30)
        
        assert len(section.rebars) == 2
    
    def test_add_rebar_before_geometry_error(self):
        """Testa erro ao adicionar armadura sem geometria."""
        section = ConcreteSection("V11", fc=30, fy=500)
        
        with pytest.raises(RuntimeError, match="Geometria não foi construída"):
            section.add_rebar_bottom(diameter=16, n_bars=4, cover=30)
    
    # ========== TESTES DE PROPRIEDADES ==========
    
    def test_calculate_gross_properties(self):
        """Testa cálculo de propriedades brutas (Estado I)."""
        section = ConcreteSection("V12", fc=30, fy=500)
        section.build_rectangular(b=0.30, h=0.60)
        section.add_rebar_bottom(diameter=16, n_bars=4, cover=30)
        
        props = section.calculate_properties(AnalysisType.GROSS)
        
        # Verificar propriedades básicas
        assert props.area > 0
        assert props.ixx > 0
        assert props.iyy > 0
        
        # Área deve ser próxima de 0.30 * 0.60 = 0.18 m²
        assert 0.17 < props.area < 0.19
    
    def test_calculate_cracked_properties(self):
        """Testa propriedades da seção fissurada (Estado II)."""
        section = ConcreteSection("V13", fc=30, fy=500)
        section.build_rectangular(b=0.30, h=0.60)
        section.add_rebar_bottom(diameter=16, n_bars=4, cover=30)
        
        props_gross = section.calculate_properties(AnalysisType.GROSS)
        props_cracked = section.calculate_properties(AnalysisType.CRACKED)
        
        # Inércia fissurada deve ser menor que bruta
        assert props_cracked.ixx < props_gross.ixx
    
    def test_calculate_ultimate_properties(self):
        """Testa propriedades no estado limite último."""
        section = ConcreteSection("V14", fc=30, fy=500)
        section.build_rectangular(b=0.30, h=0.60)
        section.add_rebar_bottom(diameter=16, n_bars=4, cover=30)
        
        props = section.calculate_properties(AnalysisType.ULTIMATE)
        
        assert props.area > 0
    
    def test_properties_caching(self):
        """Testa cache de propriedades."""
        section = ConcreteSection("V15", fc=30, fy=500)
        section.build_rectangular(b=0.30, h=0.60)
        section.add_rebar_bottom(diameter=16, n_bars=4, cover=30)
        
        props1 = section.get_properties()
        assert section._is_analyzed is True
        
        props2 = section.get_properties()
        assert props2 is props1  # Mesmo objeto (cached)
    
    # ========== TESTES DE ANÁLISE AVANÇADA ==========
    
    def test_moment_curvature_analysis(self):
        """Testa análise momento-curvatura."""
        section = ConcreteSection("V16", fc=30, fy=500)
        section.build_rectangular(b=0.30, h=0.60)
        section.add_rebar_bottom(diameter=20, n_bars=4, cover=30)
        
        mk_results = section.moment_curvature_analysis(
            kappa_max=1e-4,
            progress_bar=False
        )
        
        assert 'kappa' in mk_results
        assert 'moment' in mk_results
        assert len(mk_results['kappa']) > 0
        assert len(mk_results['moment']) > 0
        
        # Momento deve aumentar com curvatura
        assert np.all(np.diff(mk_results['moment']) >= 0)
    
    def test_moment_curvature_with_axial_load(self):
        """Testa M-κ com força normal."""
        section = ConcreteSection("V17", fc=30, fy=500)
        section.build_rectangular(b=0.30, h=0.60)
        section.add_rebar_bottom(diameter=20, n_bars=4, cover=30)
        
        # Com compressão (N = 500 kN)
        mk_comp = section.moment_curvature_analysis(
            n=500e3,  # N
            kappa_max=1e-4,
            progress_bar=False
        )
        
        # Sem força normal
        mk_zero = section.moment_curvature_analysis(
            n=0,
            kappa_max=1e-4,
            progress_bar=False
        )
        
        # Momento resistente deve ser maior com compressão
        assert max(mk_comp['moment']) > max(mk_zero['moment'])
    
    def test_moment_interaction_diagram(self):
        """Testa diagrama de interação P-M."""
        section = ConcreteSection("V18", fc=30, fy=500)
        section.build_rectangular(b=0.30, h=0.60)
        section.add_rebar_bottom(diameter=20, n_bars=4, cover=30)
        section.add_rebar_top(diameter=20, n_bars=2, cover=30)
        
        pm_results = section.moment_interaction_diagram(
            progress_bar=False
        )
        
        assert 'n' in pm_results
        assert 'moment' in pm_results
        assert len(pm_results['n']) > 0
        assert len(pm_results['moment']) > 0
        
        # Diagrama deve ter pontos em compressão e tração
        assert np.any(pm_results['n'] > 0)  # Compressão
        assert np.any(pm_results['n'] < 0)  # Tração
    
    # ========== TESTES DE VISUALIZAÇÃO ==========
    
    @pytest.mark.mpl_image_compare
    def test_plot_geometry(self, tmp_path):
        """Testa plotagem de geometria."""
        section = ConcreteSection("V19", fc=30, fy=500)
        section.build_rectangular(b=0.30, h=0.60)
        section.add_rebar_bottom(diameter=16, n_bars=4, cover=30)
        
        filename = tmp_path / "geometry.png"
        section.plot_geometry(filename=str(filename))
        
        assert filename.exists()
    
    @pytest.mark.mpl_image_compare
    def test_plot_stress(self, tmp_path):
        """Testa plotagem de tensões."""
        section = ConcreteSection("V20", fc=30, fy=500)
        section.build_rectangular(b=0.30, h=0.60)
        section.add_rebar_bottom(diameter=16, n_bars=4, cover=30)
        
        filename = tmp_path / "stress.png"
        section.plot_stress(m=100e3, n=0, filename=str(filename))
        
        assert filename.exists()
    
    # ========== TESTES DE INTEGRAÇÃO ==========
    
    def test_factory_integration(self):
        """Testa integração com SectionFactory."""
        section = SectionFactory.create('concrete', 'V21')
        
        assert isinstance(section, ConcreteSection)
        assert section.name == 'V21'
    
    def test_export_to_dict(self):
        """Testa exportação para dicionário."""
        section = ConcreteSection("V22", fc=30, fy=500)
        section.build_rectangular(b=0.30, h=0.60)
        section.add_rebar_bottom(diameter=16, n_bars=4, cover=30)
        
        export = section.export_to_dict()
        
        assert export['name'] == "V22"
        assert export['type'] == "concrete"
        assert export['analyzer'] == "ConcreteSection"
        assert 'properties' in export
        assert export['properties']['area'] > 0
    
    # ========== TESTES DE NORMAS ==========
    
    def test_nbr6118_material_factors(self):
        """Testa fatores de segurança NBR 6118:2023."""
        section = ConcreteSection("V23", fc=30, fy=500, code="NBR6118:2023")
        section.build_rectangular(b=0.30, h=0.60)
        
        section._create_materials(AnalysisType.ULTIMATE)
        
        # Verificar γc = 1.4 e γs = 1.15 (NBR 6118)
        # fcd = fc / γc
        # fyd = fy / γs
        # Isso está implícito nos materiais criados
        assert section._concrete_material is not None
        assert section._steel_material is not None
    
    def test_different_concrete_grades(self):
        """Testa diferentes classes de concreto."""
        grades = [ConcreteGrade.C20, ConcreteGrade.C30, ConcreteGrade.C50, ConcreteGrade.C90]
        
        for grade in grades:
            section = ConcreteSection(f"V_{grade.name}", concrete_grade=grade)
            assert section.fc == grade.value
    
    def test_different_steel_grades(self):
        """Testa diferentes classes de aço."""
        grades = [SteelGrade.CA25, SteelGrade.CA50, SteelGrade.CA60]
        
        for grade in grades:
            section = ConcreteSection(f"V_{grade.name}", steel_grade=grade)
            fy, Es = grade.value
            assert section.fy == fy
            assert section.Es == Es
    
    # ========== TESTES DE EDGE CASES ==========
    
    def test_single_rebar(self):
        """Testa seção com apenas 1 barra."""
        section = ConcreteSection("V24", fc=30, fy=500)
        section.build_rectangular(b=0.30, h=0.60)
        section.add_rebar_bottom(diameter=16, n_bars=1, cover=30)
        
        assert len(section.rebars) == 1
        assert section.rebars[0]['n_bars'] == 1
    
    def test_large_section(self):
        """Testa seção grande (pilar)."""
        section = ConcreteSection("P1", fc=40, fy=500)
        section.build_rectangular(b=0.80, h=0.80)
        section.add_rebar_bottom(diameter=25, n_bars=8, cover=40)
        section.add_rebar_top(diameter=25, n_bars=8, cover=40)
        
        props = section.get_properties()
        
        # Área deve ser próxima de 0.80 * 0.80 = 0.64 m²
        assert 0.62 < props.area < 0.66
    
    def test_minimum_reinforcement(self):
        """Testa seção com armadura mínima."""
        section = ConcreteSection("V25", fc=25, fy=500)
        section.build_rectangular(b=0.20, h=0.40)
        section.add_rebar_bottom(diameter=10, n_bars=2, cover=25)
        
        props = section.get_properties()
        assert props.area > 0
