"""Testes da interface base de seções."""
import pytest
from pymemorial.sections.base import SectionProperties, SectionAnalyzer


class DummySection(SectionAnalyzer):
    """Implementação concreta mínima para testes."""
    
    def build_geometry(self, **kwargs):
        self.geometry_built = True
        self.params = kwargs
    
    def calculate_properties(self) -> SectionProperties:
        if not hasattr(self, 'geometry_built'):
            raise RuntimeError("Geometria não construída")
        
        return SectionProperties(
            area=0.01,        # 100 cm²
            perimeter=0.4,    # 40 cm
            ixx=1e-5,         # 10000 cm⁴
            iyy=5e-6,         # 5000 cm⁴
            ixy=0.0,
            j=8e-6,
            cx=0.0,
            cy=0.0
        )


def test_section_properties_creation():
    """Testa criação de SectionProperties."""
    props = SectionProperties(
        area=0.01,
        perimeter=0.4,
        ixx=1e-5,
        iyy=5e-6,
        ixy=0.0,
        cx=0.0,
        cy=0.0
    )
    
    assert props.area == 0.01
    assert props.ixx == 1e-5
    assert props.j is None  # Opcional


def test_section_properties_to_dict():
    """Testa conversão para dicionário."""
    props = SectionProperties(
        area=0.01,
        perimeter=0.4,
        ixx=1e-5,
        iyy=5e-6,
        ixy=0.0,
        j=8e-6,
        cx=0.0,
        cy=0.0
    )
    
    props_dict = props.to_dict()
    
    assert 'area' in props_dict
    assert 'ixx' in props_dict
    assert props_dict['j'] == 8e-6
    # Campos None não devem aparecer
    assert 'zxx' not in props_dict


def test_section_analyzer_initialization():
    """Testa inicialização do analyzer."""
    section = DummySection(name="Test", section_type="dummy")
    
    assert section.name == "Test"
    assert section.section_type == "dummy"
    assert not section._is_analyzed


def test_section_analyzer_build_geometry():
    """Testa construção de geometria."""
    section = DummySection(name="Test", section_type="dummy")
    section.build_geometry(width=0.2, height=0.4)
    
    assert section.geometry_built is True
    assert section.params == {'width': 0.2, 'height': 0.4}


def test_section_analyzer_calculate_before_build():
    """Testa erro ao calcular sem construir geometria."""
    section = DummySection(name="Test", section_type="dummy")
    
    with pytest.raises(RuntimeError, match="Geometria não construída"):
        section.calculate_properties()


def test_section_analyzer_get_properties():
    """Testa obtenção de propriedades (lazy)."""
    section = DummySection(name="Test", section_type="dummy")
    section.build_geometry()
    
    # Primeira chamada: calcula
    props1 = section.get_properties()
    assert section._is_analyzed is True
    assert props1.area == 0.01
    
    # Segunda chamada: retorna cached
    props2 = section.get_properties()
    assert props2 is props1  # Mesmo objeto


def test_section_analyzer_export_to_dict():
    """Testa exportação para dicionário."""
    section = DummySection(name="TestSection", section_type="dummy")
    section.build_geometry()
    
    export = section.export_to_dict()
    
    assert export['name'] == "TestSection"
    assert export['type'] == "dummy"
    assert export['analyzer'] == "DummySection"
    assert 'properties' in export
    assert export['properties']['area'] == 0.01


def test_section_analyzer_repr():
    """Testa representação string."""
    section = DummySection(name="Test", section_type="dummy")
    
    repr_str = repr(section)
    assert "DummySection" in repr_str
    assert "Test" in repr_str
    assert "not analyzed" in repr_str
    
    section.build_geometry()
    section.get_properties()
    
    repr_str = repr(section)
    assert "analyzed" in repr_str


def test_section_properties_optional_fields():
    """Testa campos opcionais de SectionProperties."""
    props = SectionProperties(
        area=0.01,
        perimeter=0.4,
        ixx=1e-5,
        iyy=5e-6,
        ixy=0.0,
        cx=0.0,
        cy=0.0,
        zxx=1.5e-4,  # Módulo plástico
        zyy=8e-5,
        rx=0.05,     # Raio de giração
        ry=0.03
    )
    
    assert props.zxx == 1.5e-4
    assert props.rx == 0.05
    
    props_dict = props.to_dict()
    assert 'zxx' in props_dict
    assert 'rx' in props_dict
