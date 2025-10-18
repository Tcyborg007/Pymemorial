"""Testes do MemorialBuilder."""
import pytest
from pymemorial.builder import MemorialBuilder


def test_memorial_creation():
    """Testa criação básica do memorial."""
    memorial = MemorialBuilder("Teste", author="Engenheiro")
    assert memorial.metadata.title == "Teste"
    assert memorial.metadata.author == "Engenheiro"


def test_add_variable():
    """Testa adição de variável."""
    memorial = MemorialBuilder("Teste")
    memorial.add_variable("fck", value=30, unit="MPa", description="Resistência")
    
    assert "fck" in memorial.variables
    assert memorial.variables["fck"].value.magnitude == 30


def test_add_variable_chaining():
    """Testa encadeamento fluente."""
    memorial = (MemorialBuilder("Teste")
                .add_variable("L", value=5.0, unit="m")
                .add_variable("fck", value=30, unit="MPa"))
    
    assert len(memorial.variables) == 2


def test_add_section():
    """Testa adição de seção."""
    memorial = MemorialBuilder("Teste")
    memorial.add_section("Introdução", level=1)
    
    assert len(memorial.sections) == 1
    assert memorial.sections[0].title == "Introdução"


def test_add_text_without_section():
    """Testa que adicionar texto sem seção gera erro."""
    memorial = MemorialBuilder("Teste")
    
    with pytest.raises(ValueError, match="Nenhuma seção ativa"):
        memorial.add_text("Texto sem seção")


def test_add_text_with_template():
    """Testa texto com placeholders."""
    memorial = (MemorialBuilder("Teste")
                .add_variable("L", value=6.0, unit="m")
                .add_section("Geometria")
                .add_text("O vão é de {{L}} metros."))
    
    # Verificar que o texto foi processado
    content = memorial.sections[0].content[0]
    assert "6.0" in content.content


def test_add_equation():
    """Testa adição de equação."""
    memorial = (MemorialBuilder("Teste")
                .add_variable("a", value=3.0)
                .add_variable("b", value=4.0)
                .add_equation("c = a + b"))
    
    assert len(memorial.equations) == 1


def test_compute():
    """Testa cálculo de equações."""
    memorial = (MemorialBuilder("Teste")
                .add_variable("a", value=3.0)
                .add_variable("b", value=4.0)
                .add_equation("c = a + b"))
    
    results = memorial.compute()
    eq_id = id(memorial.equations[0])
    assert results[eq_id] == 7.0


def test_build():
    """Testa construção do memorial completo."""
    memorial = (MemorialBuilder("Teste", author="Eng")
                .add_variable("L", value=5.0, unit="m")
                .add_section("Dados")
                .add_text("Vão: {{L}}"))
    
    data = memorial.build()
    
    assert data["metadata"]["title"] == "Teste"
    assert "L" in data["variables"]
    assert len(data["sections"]) == 1
def test_add_equation_invalid():
    """Testa equação inválida."""
    memorial = MemorialBuilder("Teste")
    
    with pytest.raises(ValueError, match="Equação inválida"):
        memorial.add_equation("texto sem igual")


def test_add_equation_block():
    """Testa adição de bloco de equação."""
    memorial = (MemorialBuilder("Teste")
                .add_variable("x", value=5.0)
                .add_equation("y = x * 2")
                .add_section("Cálculo")
                .add_equation_block(0))
    
    # Verificar que foi adicionado
    assert len(memorial.sections[0].content) == 1


def test_add_equation_block_invalid_index():
    """Testa índice de equação inválido."""
    memorial = MemorialBuilder("Teste")
    memorial.add_section("Teste")
    
    with pytest.raises(IndexError, match="não existe"):
        memorial.add_equation_block(0)


def test_nested_sections():
    """Testa seções aninhadas."""
    memorial = (MemorialBuilder("Teste")
                .add_section("Pai", level=1)
                .add_section("Filho", level=2))
    
    # Filho deve ser subseção de Pai
    assert len(memorial.sections) == 1
    assert len(memorial.sections[0].subsections) == 1
