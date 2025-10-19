"""Testes do SectionFactory."""
import pytest
from pymemorial.sections import SectionFactory
from pymemorial.sections.steel import SteelSection
from pymemorial.sections.concrete import ConcreteSection, CONCRETEPROPERTIES_AVAILABLE


def test_available_analyzers():
    """Testa listagem de analyzers disponíveis."""
    available = SectionFactory.available_analyzers()
    
    assert isinstance(available, dict)
    assert 'steel' in available
    assert 'concrete' in available
    assert 'composite' in available
    
    # Valores são booleanos
    assert isinstance(available['steel'], bool)
    assert isinstance(available['concrete'], bool)


def test_create_invalid_type():
    """Testa erro ao criar tipo inválido."""
    with pytest.raises(ValueError, match="não suportado"):
        SectionFactory.create('invalid_type', 'Test')


@pytest.mark.skipif(
    not SectionFactory.available_analyzers()['steel'],
    reason="sectionproperties não instalado"
)
def test_create_steel_section():
    """Testa criação de seção de aço."""
    section = SectionFactory.create('steel', 'TestSteel')
    
    assert section.name == 'TestSteel'
    assert 'Steel' in section.__class__.__name__


@pytest.mark.skipif(
    not SectionFactory.available_analyzers()['steel'],
    reason="sectionproperties não instalado"
)
def test_create_steel_variants():
    """Testa variantes de tipo steel."""
    variants = ['steel', 'steel_i', 'steel_rect', 'steel_tube']
    
    for variant in variants:
        section = SectionFactory.create(variant, f'Test_{variant}')
        assert section.name == f'Test_{variant}'


@pytest.mark.skipif(
    not CONCRETEPROPERTIES_AVAILABLE,
    reason="concreteproperties não instalado"
)
def test_create_concrete_section():
    """Testa criação de seção de concreto."""
    section = SectionFactory.create('concrete', 'TestConcrete')
    
    assert isinstance(section, ConcreteSection)
    assert section.name == 'TestConcrete'


@pytest.mark.skipif(
    not CONCRETEPROPERTIES_AVAILABLE,  # ← CORRIGIDO
    reason="concreteproperties não instalado"
)
def test_create_concrete_variants():
    """Testa variantes de tipo concrete."""
    variants = ['concrete', 'concrete_rect', 'rc']
    
    for variant in variants:
        section = SectionFactory.create(variant, f'Test_{variant}')
        assert section.name == f'Test_{variant}'


def test_create_case_insensitive():
    """Testa que tipo é case-insensitive."""
    available = SectionFactory.available_analyzers()
    
    if available['steel']:
        section1 = SectionFactory.create('STEEL', 'Test1')
        section2 = SectionFactory.create('Steel', 'Test2')
        section3 = SectionFactory.create('steel', 'Test3')
        
        assert all(s.name.startswith('Test') for s in [section1, section2, section3])
