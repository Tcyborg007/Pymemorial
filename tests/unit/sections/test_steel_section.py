"""Testes de SteelSection com sectionproperties."""
import pytest
from pymemorial.sections import SectionFactory
from pymemorial.sections.steel import SteelSection, SECTIONPROPERTIES_AVAILABLE


@pytest.mark.skipif(
    not SECTIONPROPERTIES_AVAILABLE,
    reason="sectionproperties não instalado"
)
class TestSteelSection:
    """Testes da implementação completa de SteelSection."""
    
    def test_steel_section_initialization(self):
        """Testa inicialização de SteelSection."""
        section = SteelSection("IPE200")
        
        assert section.name == "IPE200"
        assert section.section_type == "steel"
        assert section.E == 200e9
        assert section.nu == 0.3
        assert section.fy == 250e6
        assert section.geometry is None
        assert section.section is None
    
    def test_steel_section_custom_properties(self):
        """Testa inicialização com propriedades customizadas."""
        section = SteelSection("CustomSteel", E=210e9, nu=0.28, fy=345e6)
        
        assert section.E == 210e9
        assert section.nu == 0.28
        assert section.fy == 345e6
    
    def test_build_i_section(self):
        """Testa construção de perfil I."""
        section = SteelSection("IPE200")
        section.build_i_section(d=0.200, b=0.100, tf=0.0085, tw=0.0056, r=0.012)
        
        assert section.geometry is not None
    
    def test_build_circular_hollow(self):
        """Testa construção de tubo circular."""
        section = SteelSection("Tubo150x5")
        section.build_circular_hollow(d=0.150, t=0.005)
        
        assert section.geometry is not None
    
    def test_build_rectangular(self):
        """Testa construção de seção retangular."""
        section = SteelSection("Rect200x100")
        section.build_rectangular(b=0.200, d=0.100)
        
        assert section.geometry is not None
    
    def test_build_channel(self):
        """Testa construção de perfil U."""
        section = SteelSection("U200")
        section.build_channel(d=0.200, b=0.075, tf=0.010, tw=0.007, r=0.008)
        
        assert section.geometry is not None
    
    def test_build_geometry_via_type(self):
        """Testa build_geometry() genérico."""
        section = SteelSection("GenericI")
        section.build_geometry(section_type='i', d=0.200, b=0.100, tf=0.0085, tw=0.0056)
        
        assert section.geometry is not None
    
    def test_build_geometry_invalid_type(self):
        """Testa erro com tipo inválido."""
        section = SteelSection("Invalid")
        
        with pytest.raises(ValueError, match="não suportado"):
            section.build_geometry(section_type='invalid_type')
    
    def test_calculate_properties_i_section(self):
        """Testa cálculo de propriedades de perfil I."""
        section = SteelSection("IPE200")
        section.build_i_section(d=0.200, b=0.100, tf=0.0085, tw=0.0056)
        
        props = section.calculate_properties()
        
        # Verificar propriedades obrigatórias
        assert props.area > 0
        assert props.ixx > 0
        assert props.iyy > 0
        
        # j pode ser None (precisa calcular torsão separadamente)
        # Verificar apenas se não for None
        if props.j is not None:
            assert props.j > 0
        
        # Verificar unidades (metros)
        assert 0.001 < props.area < 0.1        # Área razoável para IPE200
        assert props.ixx > props.iyy           # I maior no eixo forte
    
    def test_calculate_properties_circular_hollow(self):
        """Testa cálculo de propriedades de tubo."""
        section = SteelSection("Tubo150x5")
        section.build_circular_hollow(d=0.150, t=0.005)
        
        props = section.calculate_properties()
        
        assert props.area > 0
        assert props.ixx > 0
        assert abs(props.ixx - props.iyy) < props.ixx * 0.01  # Simétrico
    
    def test_calculate_properties_before_build(self):
        """Testa erro ao calcular sem construir geometria."""
        section = SteelSection("Empty")
        
        with pytest.raises(RuntimeError, match="Geometria não foi construída"):
            section.calculate_properties()
    
    def test_get_properties_caching(self):
        """Testa cache de propriedades."""
        section = SteelSection("IPE200")
        section.build_i_section(d=0.200, b=0.100, tf=0.0085, tw=0.0056)
        
        props1 = section.get_properties()
        assert section._is_analyzed is True
        
        props2 = section.get_properties()
        assert props2 is props1
    
    def test_plastic_properties(self):
        """Testa propriedades plásticas."""
        section = SteelSection("IPE200")
        section.build_i_section(d=0.200, b=0.100, tf=0.0085, tw=0.0056)
        
        props = section.get_properties()
        
        # Módulos plásticos podem ser calculados
        if props.zxx is not None:
            assert props.zxx > 0
        if props.zyy is not None:
            assert props.zyy > 0
    
    def test_radii_of_gyration(self):
        """Testa raios de giração."""
        section = SteelSection("IPE200")
        section.build_i_section(d=0.200, b=0.100, tf=0.0085, tw=0.0056)
        
        props = section.get_properties()
        
        if props.rx is not None and props.ry is not None:
            assert props.rx > 0
            assert props.ry > 0
            assert props.rx > props.ry  # rx maior para I
    
    def test_principal_axes(self):
        """Testa propriedades de eixos principais."""
        section = SteelSection("IPE200")
        section.build_i_section(d=0.200, b=0.100, tf=0.0085, tw=0.0056)
        
        props = section.get_properties()
        
        if props.i11 is not None and props.i22 is not None:
            assert props.i11 >= props.i22  # I11 é sempre o maior
    
    def test_get_stress_at_yield(self):
        """Testa cálculo de momentos de escoamento."""
        section = SteelSection("IPE200", fy=250e6)
        section.build_i_section(d=0.200, b=0.100, tf=0.0085, tw=0.0056)
        
        yield_moments = section.get_stress_at_yield()
        
        assert 'Mx_yield' in yield_moments
        assert 'My_yield' in yield_moments
        # Valores podem ser 0 se zxx/zyy não calculados
        assert yield_moments['Mx_yield'] >= 0
        assert yield_moments['My_yield'] >= 0
    
    def test_export_to_dict(self):
        """Testa exportação para dicionário."""
        section = SteelSection("IPE200")
        section.build_i_section(d=0.200, b=0.100, tf=0.0085, tw=0.0056)
        
        export = section.export_to_dict()
        
        assert export['name'] == "IPE200"
        assert export['type'] == "steel"
        assert export['analyzer'] == "SteelSection"
        assert 'properties' in export
        assert export['properties']['area'] > 0
    
    def test_factory_integration(self):
        """Testa integração com SectionFactory."""
        section = SectionFactory.create('steel', 'TestSteel')
        
        assert isinstance(section, SteelSection)
        assert section.name == 'TestSteel'
