"""
Testes completos de CompositeSection - Seções Mistas Aço-Concreto.
Compatível com EN 1994 (Eurocode 4) - Edição 2025.
"""
import pytest
import numpy as np
from pymemorial.sections import SectionFactory
from pymemorial.sections.composite import (
    CompositeSection,
    CompositeType,
    ShearConnectorType,
    ShearConnectorProperties,
    SECTIONPROPERTIES_AVAILABLE
)


@pytest.mark.skipif(
    not SECTIONPROPERTIES_AVAILABLE,
    reason="sectionproperties não instalado"
)
class TestCompositeSection:
    """Testes da implementação completa de CompositeSection."""
    
    # ========== TESTES DE INICIALIZAÇÃO ==========
    
    def test_composite_section_initialization(self):
        """Testa inicialização básica."""
        section = CompositeSection("VM-1")
        
        assert section.name == "VM-1"
        assert section.section_type == "composite"
        assert section.composite_type == CompositeType.COMPOSITE_BEAM
        assert section.steel_fy == 355e6  # Pa (S355 padrão)
        assert section.concrete_fck == 30e6  # Pa (C30/37 padrão)
        assert section.code == "EC4:2025"
        assert section.geometry is None
    
    def test_composite_section_custom_materials(self):
        """Testa inicialização com materiais customizados."""
        section = CompositeSection(
            "VM-2",
            steel_fy=450e6,      # S450
            concrete_fck=50e6,   # C50/60
            code="NBR8800"
        )
        
        assert section.steel_fy == 450e6
        assert section.concrete_fck == 50e6
        assert section.code == "NBR8800"
    
    def test_composite_types_available(self):
        """Testa todos os tipos de composição disponíveis."""
        types = [
            CompositeType.COMPOSITE_BEAM,
            CompositeType.ENCASED_BEAM,
            CompositeType.FILLED_COLUMN,
            CompositeType.ENCASED_COLUMN,
            CompositeType.PARTIALLY_ENCASED
        ]
        
        for comp_type in types:
            section = CompositeSection("Test", composite_type=comp_type)
            assert section.composite_type == comp_type
    
    # ========== TESTES DE GEOMETRIA - VIGA MISTA ==========
    
    def test_build_composite_beam_standard(self):
        """Testa construção de viga mista com perfil padrão."""
        section = CompositeSection("VM-1")
        section.build_composite_beam(
            steel_profile="W310x52",
            slab_width=2.5,  # m
            slab_height=0.20,  # m
            concrete_fc=30  # MPa
        )
        
        assert section.geometry is not None
        assert section.steel_geometry is not None
        assert section.concrete_geometry is not None
        assert section.compound_geometry is not None
    
    def test_build_composite_beam_custom_dimensions(self):
        """Testa viga mista com dimensões customizadas."""
        section = CompositeSection("VM-2")
        section.build_composite_beam(
            steel_d=0.40,    # m
            steel_b=0.20,    # m
            steel_tf=0.015,  # m
            steel_tw=0.010,  # m
            slab_width=3.0,  # m
            slab_height=0.15  # m
        )
        
        assert section.geometry is not None
    
    def test_build_composite_beam_with_haunch(self):
        """Testa viga mista com calço/reforço (haunch)."""
        section = CompositeSection("VM-3")
        section.build_composite_beam(
            steel_profile="W310x52",
            slab_width=2.5,
            slab_height=0.20,
            haunch_height=0.05  # 50mm de calço
        )
        
        assert section.geometry is not None
    
    # ========== TESTES DE GEOMETRIA - PILAR PREENCHIDO ==========
    
    def test_build_filled_column_circular(self):
        """Testa pilar tubular circular preenchido."""
        section = CompositeSection(
            "PM-1",
            composite_type=CompositeType.FILLED_COLUMN
        )
        section.build_filled_column(
            outer_diameter=0.40,  # m
            wall_thickness=0.012,  # m
            concrete_fc=40  # MPa
        )
        
        assert section.geometry is not None
        assert section.steel_geometry is not None
        assert section.concrete_geometry is not None
    
    def test_build_filled_column_high_strength(self):
        """Testa pilar preenchido com concreto de alta resistência."""
        section = CompositeSection(
            "PM-2",
            composite_type=CompositeType.FILLED_COLUMN,
            concrete_fck=80e6  # C80/95
        )
        section.build_filled_column(
            outer_diameter=0.60,
            wall_thickness=0.020,
            concrete_fc=80
        )
        
        assert section.geometry is not None
    
    # ========== TESTES DE CONECTORES DE CISALHAMENTO ==========
    
    def test_add_shear_connectors_headed_stud(self):
        """Testa adição de conectores tipo pino com cabeça."""
        section = CompositeSection("VM-4")
        section.build_composite_beam(
            steel_profile="W310x52",
            slab_width=2.5,
            slab_height=0.20
        )
        
        section.add_shear_connectors(
            type=ShearConnectorType.HEADED_STUD,
            diameter=19,  # mm (3/4")
            height=100,   # mm
            fu=450,       # MPa
            spacing=150,  # mm
            n_per_row=2
        )
        
        assert len(section.shear_connectors) == 1
        connector = section.shear_connectors[0]
        assert connector.type == ShearConnectorType.HEADED_STUD
        assert connector.diameter == 19
        assert connector.spacing == 150
    
    def test_add_multiple_shear_connector_groups(self):
        """Testa múltiplos grupos de conectores."""
        section = CompositeSection("VM-5")
        section.build_composite_beam(
            steel_profile="W310x52",
            slab_width=2.5,
            slab_height=0.20
        )
        
        # Grupo 1: Região de momento positivo
        section.add_shear_connectors(
            type=ShearConnectorType.HEADED_STUD,
            diameter=19,
            spacing=150
        )
        
        # Grupo 2: Região de momento negativo (espaçamento menor)
        section.add_shear_connectors(
            type=ShearConnectorType.HEADED_STUD,
            diameter=22,
            spacing=100
        )
        
        assert len(section.shear_connectors) == 2
    
    def test_calculate_shear_connector_capacity(self):
        """Testa cálculo de capacidade de conector (Eurocode 4)."""
        section = CompositeSection("VM-6")
        
        connector = ShearConnectorProperties(
            type=ShearConnectorType.HEADED_STUD,
            diameter=19,    # mm
            height=100,     # mm
            fu=450,         # MPa
            spacing=150,
            n_per_row=2
        )
        
        capacity = section.calculate_shear_connector_capacity(connector)
        
        # Capacidade deve ser > 0 e razoável (tipicamente 50-150 kN)
        assert capacity > 0
        assert 40 < capacity < 200  # kN (range esperado para studs 19mm)
    
    def test_shear_connector_capacity_varies_with_diameter(self):
        """Testa que capacidade aumenta com diâmetro."""
        section = CompositeSection("VM-7")
        
        # Conector pequeno (16mm)
        conn_16 = ShearConnectorProperties(
            type=ShearConnectorType.HEADED_STUD,
            diameter=16, height=100, fu=450, spacing=150, n_per_row=2
        )
        cap_16 = section.calculate_shear_connector_capacity(conn_16)
        
        # Conector grande (22mm)
        conn_22 = ShearConnectorProperties(
            type=ShearConnectorType.HEADED_STUD,
            diameter=22, height=100, fu=450, spacing=150, n_per_row=2
        )
        cap_22 = section.calculate_shear_connector_capacity(conn_22)
        
        # Capacidade deve aumentar com diâmetro
        assert cap_22 > cap_16
    
    # ========== TESTES DE PROPRIEDADES ==========
    
    def test_calculate_composite_properties_beam(self):
        """Testa cálculo de propriedades de viga mista."""
        section = CompositeSection("VM-8")
        section.build_composite_beam(
            steel_profile="W310x52",
            slab_width=2.5,
            slab_height=0.20,
            concrete_fc=30
        )
        
        props = section.calculate_composite_properties()
        
        # Verificar propriedades básicas
        assert props.area > 0
        assert props.ixx > 0
        assert props.iyy > 0
        
        # Área deve incluir aço + concreto (aproximadamente)
        # W310x52: ~6700 mm² + laje 2.5m x 0.2m = 500,000 mm²
        assert 0.4 < props.area < 0.6  # m² (ordem de grandeza)
    
    def test_composite_inertia_greater_than_steel_alone(self):
        """Testa que inércia composta > inércia do aço sozinho."""
        # Seção de aço pura
        from pymemorial.sections.steel import SteelSection
        steel = SteelSection("W310x52")
        steel.build_i_section(d=0.3175, b=0.1669, tf=0.0132, tw=0.0076)
        props_steel = steel.get_properties()
        
        # Seção composta
        composite = CompositeSection("VM-9")
        composite.build_composite_beam(
            steel_profile="W310x52",
            slab_width=2.5,
            slab_height=0.20
        )
        props_composite = composite.get_properties()
        
        # Inércia composta deve ser muito maior devido à laje
        assert props_composite.ixx > props_steel.ixx * 2
    
    def test_calculate_properties_filled_column(self):
        """Testa propriedades de pilar preenchido."""
        section = CompositeSection(
            "PM-3",
            composite_type=CompositeType.FILLED_COLUMN
        )
        section.build_filled_column(
            outer_diameter=0.40,
            wall_thickness=0.012,
            concrete_fc=40
        )
        
        props = section.get_properties()
        
        assert props.area > 0
        assert props.ixx > 0
        assert props.iyy > 0
        
        # Seção circular: Ixx ≈ Iyy
        ratio = props.ixx / props.iyy if props.iyy > 0 else 0
        assert 0.95 < ratio < 1.05  # Tolerância 5%
    
    def test_modular_ratio_calculation(self):
        """Testa cálculo de razão modular (n = Es/Ec)."""
        section = CompositeSection("VM-10", concrete_fck=30e6)
        section.build_composite_beam(
            steel_profile="W310x52",
            slab_width=2.5,
            slab_height=0.20
        )
        
        # n0 deve ser calculado (tipicamente 6-10 para C30)
        assert section.n0 > 0
        assert 5 < section.n0 < 12  # Range típico
    
    # ========== TESTES DE MATERIAIS ==========
    
    def test_material_creation(self):
        """Testa criação de materiais aço e concreto."""
        section = CompositeSection("VM-11")
        section._create_materials()
        
        assert section.steel_material is not None
        assert section.concrete_material is not None
        assert section.steel_material.name.startswith("Steel_")
        assert section.concrete_material.name.startswith("Concrete_")
    
    def test_concrete_modulus_varies_with_strength(self):
        """Testa que Ec aumenta com fck (Eurocode 2)."""
        # C30/37
        section_c30 = CompositeSection("Test1", concrete_fck=30e6)
        section_c30._create_materials()
        Ec_30 = section_c30.concrete_material.elastic_modulus
        
        # C50/60
        section_c50 = CompositeSection("Test2", concrete_fck=50e6)
        section_c50._create_materials()
        Ec_50 = section_c50.concrete_material.elastic_modulus
        
        # Ec aumenta com fck (não linearmente)
        assert Ec_50 > Ec_30
    
    # ========== TESTES DE VISUALIZAÇÃO ==========
    
    @pytest.mark.mpl_image_compare
    def test_plot_composite_beam_geometry(self, tmp_path):
        """Testa plotagem de viga mista."""
        section = CompositeSection("VM-12")
        section.build_composite_beam(
            steel_profile="W310x52",
            slab_width=2.5,
            slab_height=0.20
        )
        
        filename = tmp_path / "composite_beam.png"
        section.plot_geometry(filename=str(filename))
        
        assert filename.exists()
    
    @pytest.mark.mpl_image_compare
    def test_plot_filled_column_geometry(self, tmp_path):
        """Testa plotagem de pilar preenchido."""
        section = CompositeSection(
            "PM-4",
            composite_type=CompositeType.FILLED_COLUMN
        )
        section.build_filled_column(
            outer_diameter=0.40,
            wall_thickness=0.012
        )
        
        filename = tmp_path / "filled_column.png"
        section.plot_geometry(filename=str(filename))
        
        assert filename.exists()
    
    # ========== TESTES DE INTEGRAÇÃO ==========
    
    def test_factory_integration(self):
        """Testa integração com SectionFactory."""
        section = SectionFactory.create('composite', 'VM-13')
        
        assert isinstance(section, CompositeSection)
        assert section.name == 'VM-13'
    
    def test_export_to_dict_composite(self):
        """Testa exportação para dicionário."""
        section = CompositeSection("VM-14")
        section.build_composite_beam(
            steel_profile="W310x52",
            slab_width=2.5,
            slab_height=0.20
        )
        section.add_shear_connectors(diameter=19, spacing=150)
        
        export = section.export_to_dict()
        
        assert export['name'] == "VM-14"
        assert export['type'] == "composite"
        assert export['composite_type'] == CompositeType.COMPOSITE_BEAM.value
        assert export['steel_fy'] == 355e6
        assert export['concrete_fck'] == 30e6
        assert export['n_shear_connectors'] == 1
        assert 'properties' in export
    
    # ========== TESTES DE NORMAS ==========
    
    def test_eurocode4_default(self):
        """Testa padrão Eurocode 4:2025."""
        section = CompositeSection("VM-15")
        assert section.code == "EC4:2025"
    
    def test_different_codes_supported(self):
        """Testa suporte a diferentes normas."""
        codes = ["EC4:2025", "AISC360", "NBR8800"]
        
        for code in codes:
            section = CompositeSection(f"Test_{code}", code=code)
            assert section.code == code
    
    # ========== TESTES DE EDGE CASES ==========
    
    def test_very_wide_slab(self):
        """Testa viga com laje muito larga."""
        section = CompositeSection("VM-16")
        section.build_composite_beam(
            steel_profile="W310x52",
            slab_width=5.0,  # 5m - muito largo
            slab_height=0.20
        )
        
        props = section.get_properties()
        assert props.area > 0
    
    def test_thin_slab(self):
        """Testa viga com laje fina."""
        section = CompositeSection("VM-17")
        section.build_composite_beam(
            steel_profile="W310x52",
            slab_width=2.5,
            slab_height=0.10  # 100mm - fino
        )
        
        props = section.get_properties()
        assert props.area > 0
    
    def test_large_filled_column(self):
        """Testa pilar preenchido grande."""
        section = CompositeSection(
            "PM-5",
            composite_type=CompositeType.FILLED_COLUMN
        )
        section.build_filled_column(
            outer_diameter=1.0,   # 1m - grande
            wall_thickness=0.025
        )
        
        props = section.get_properties()
        assert props.area > 0
    
    def test_high_strength_materials(self):
        """Testa materiais de alta resistência."""
        section = CompositeSection(
            "VM-18",
            steel_fy=690e6,      # S690 (ultra-high strength)
            concrete_fck=90e6    # C90/105
        )
        section.build_composite_beam(
            steel_profile="W310x52",
            slab_width=2.5,
            slab_height=0.20
        )
        
        props = section.get_properties()
        assert props.area > 0
    
    def test_build_geometry_via_type(self):
        """Testa build_geometry() genérico."""
        section = CompositeSection("VM-19")
        section.build_geometry(
            geometry_type="composite_beam",
            steel_profile="W310x52",
            slab_width=2.5,
            slab_height=0.20
        )
        
        assert section.geometry is not None
    
    def test_build_geometry_invalid_type(self):
        """Testa erro com tipo inválido."""
        section = CompositeSection("VM-20")
        
        with pytest.raises(ValueError, match="não suportado"):
            section.build_geometry(geometry_type="invalid_type")
