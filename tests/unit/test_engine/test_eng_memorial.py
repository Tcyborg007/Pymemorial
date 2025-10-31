# tests/unit/test_engine/test_eng_memorial.py
"""
Testes para eng_memorial (EngMemorial)

Testa:
- Criação de memorial
- API write()
- API var() e calc()
- API section()
- API verify()
- Export para múltiplos formatos
"""

import pytest
from pathlib import Path
from pymemorial import  
from pymemorial.engine.context import reset_context


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture(autouse=True)
def reset_ctx():
    """Reset context antes de cada teste."""
    reset_context()
    yield
    reset_context()


# ============================================================================
# TESTES - CRIAÇÃO
# ============================================================================

def test_create_memorial():
    """Testa criação básica."""
    mem = EngMemorial("Teste")
    assert mem.metadata.title == "Teste"


def test_create_with_metadata():
    """Testa criação com metadados."""
    mem = EngMemorial(
        "Viga V-1",
        author="Eng. João",
        norm="NBR 6118:2023"
    )
    
    assert mem.metadata.author == "Eng. João"
    assert mem.metadata.norm == "NBR 6118:2023"


# ============================================================================
# TESTES - API WRITE
# ============================================================================

def test_write_simple_text():
    """Testa write com texto simples."""
    mem = EngMemorial("Teste")
    mem.write("Este é um teste.")
    
    assert len(mem._content) == 1


def test_write_with_code():
    """Testa write com código inline."""
    mem = EngMemorial("Teste")
    mem.write("""
# Cálculo
q = 15 kN/m
L = 6 m
M = q * L**2 / 8
""")
    
    assert len(mem._content) >= 1


# ============================================================================
# TESTES - API VAR E CALC
# ============================================================================

def test_var_definition():
    """Testa definição de variável."""
    mem = EngMemorial("Teste")
    mem.var("f_ck", 30, "MPa", "Resistência do concreto")
    
    assert len(mem._content) == 1
    
    # Verificar que foi adicionado ao contexto
    from pymemorial.engine.context import get_context
    ctx = get_context()
    assert ctx.has("f_ck")


def test_calc_simple():
    """Testa cálculo simples."""
    mem = EngMemorial("Teste")
    mem.var("a", 10)
    mem.var("b", 5)
    mem.calc("c = a + b")
    
    # CORREÇÃO: Verificar se foi adicionado ao conteúdo
    assert len(mem._content) == 3  # 2 vars + 1 calc
    
    # Verificar no contexto com proteção
    from pymemorial.engine.context import get_context
    ctx = get_context()
    
    # Pode ou não estar no contexto, dependendo da implementação
    c_var = ctx.get("c")
    if c_var:
        assert c_var.value == 15
    else:
        # Verificar nos metadados do memorial
        calc_block = mem._content[2]
        assert calc_block.metadata.get("result") == 15



def test_fluent_api():
    """Testa API fluente (chaining)."""
    mem = EngMemorial("Teste")
    
    result = (mem
              .var("x", 10)
              .var("y", 20)
              .calc("z = x + y"))
    
    assert result is mem  # Deve retornar self


# ============================================================================
# TESTES - API SECTION
# ============================================================================

def test_section_creation():
    """Testa criação de seção."""
    mem = EngMemorial("Teste")
    mem.section("Geometria", level=1)
    
    assert len(mem._content) == 1
    assert mem._current_section == "Geometria"


def test_nested_sections():
    """Testa seções hierárquicas."""
    mem = EngMemorial("Teste")
    mem.section("Capítulo 1", level=1)
    mem.section("Seção 1.1", level=2)
    
    assert len(mem._content) == 2


# ============================================================================
# TESTES - API VERIFY
# ============================================================================

def test_verify_pass():
    """Testa verificação que passa."""
    mem = EngMemorial("Teste")
    mem.var("A", 400, "cm²")
    mem.verify("A >= 360", desc="Seção mínima")
    
    # Verificar conteúdo
    assert len(mem._content) == 2
    verify_block = mem._content[1]
    assert "OK" in verify_block.content or "✅" in verify_block.content


def test_verify_fail():
    """Testa verificação que falha."""
    mem = EngMemorial("Teste")
    mem.var("A", 300, "cm²")
    mem.verify("A >= 360", desc="Seção mínima")
    
    verify_block = mem._content[1]
    assert "NÃO OK" in verify_block.content or "❌" in verify_block.content


# ============================================================================
# TESTES - EXPORT
# ============================================================================

def test_save_markdown(tmp_path):
    """Testa export para Markdown."""
    mem = EngMemorial("Teste")
    mem.write("Conteúdo de teste.")
    
    output = tmp_path / "test.md"
    mem.save(output)
    
    assert output.exists()
    content = output.read_text(encoding="utf-8")
    assert "Teste" in content


def test_save_html(tmp_path):
    """Testa export para HTML."""
    mem = EngMemorial("Teste")
    mem.write("Conteúdo HTML.")
    
    output = tmp_path / "test.html"
    mem.save(output)
    
    assert output.exists()
    content = output.read_text(encoding="utf-8")
    assert "<html" in content.lower()


def test_save_json(tmp_path):
    """Testa export para JSON."""
    mem = EngMemorial("Teste")
    mem.write("Conteúdo JSON.")
    
    output = tmp_path / "test.json"
    mem.save(output)
    
    assert output.exists()
    
    import json
    with open(output) as f:
        data = json.load(f)
    
    assert "metadata" in data
    assert "content" in data


# ============================================================================
# TESTES - INTEGRAÇÃO COMPLETA
# ============================================================================

def test_complete_memorial(tmp_path):
    """Testa memorial completo end-to-end."""
    mem = EngMemorial("Viga Biapoiada", author="Eng. Teste")
    
    mem.section("Dados de Entrada", level=1)
    mem.var("q", 15.0, "kN/m", "Carga distribuída")
    mem.var("L", 6.0, "m", "Vão da viga")
    
    mem.section("Cálculo do Momento", level=1)
    mem.calc("M_max = q * L**2 / 8", unit="kN.m")
    
    mem.section("Verificação", level=1)
    mem.verify("M_max <= 100", desc="Momento admissível")
    
    # Export
    output = tmp_path / "complete.md"
    mem.save(output)
    
    assert output.exists()
    # CORREÇÃO: Ler com UTF-8 explícito
    content = output.read_text(encoding="utf-8")
    assert "Viga Biapoiada" in content
    assert "kN/m" in content

