"""Testes do processador de texto."""
import pytest
from pymemorial.recognition.text_processor import TextProcessor


@pytest.fixture
def processor():
    """Fixture do processador."""
    return TextProcessor()


def test_render_simple(processor):
    """Testa renderização simples."""
    template = "O valor de {{x}} é {{y}}."
    context = {"x": "fck", "y": 30}
    result = processor.render(template, context)
    assert result == "O valor de fck é 30."


def test_render_missing_key(processor):
    """Testa renderização com chave faltando."""
    template = "Valor {{x}} e {{missing}}."
    context = {"x": 10}
    result = processor.render(template, context)
    assert "{{missing}}" in result  # Placeholder não substituído


def test_render_multiple_occurrences(processor):
    """Testa múltiplas ocorrências da mesma variável."""
    template = "{{x}} + {{x}} = {{result}}"
    context = {"x": 5, "result": 10}
    result = processor.render(template, context)
    assert result == "5 + 5 = 10"


def test_to_latex_special_chars(processor):
    """Testa escape de caracteres especiais."""
    text = "valor_1 = 50%"
    result = processor.to_latex(text, greek_to_math=False)
    assert r"\_" in result
    assert r"\%" in result


def test_to_latex_greek_symbols(processor):
    """Testa conversão de símbolos gregos."""
    text = "α = 0.85 e β = 1.2"
    result = processor.to_latex(text, greek_to_math=True)
    assert r"$\alpha$" in result
    assert r"$\beta$" in result


def test_extract_and_replace(processor):
    """Testa extração e substituição."""
    text = "Resistência {{fck}} e tensão {{fy}}"
    replacements = {"fck": "30 MPa", "fy": "500 MPa"}
    result = processor.extract_and_replace(text, replacements)
    assert result == "Resistência 30 MPa e tensão 500 MPa"


def test_extract_and_replace_preserve(processor):
    """Testa preservação de placeholders não encontrados."""
    text = "{{a}} e {{b}}"
    replacements = {"a": "valor_a"}
    result = processor.extract_and_replace(text, replacements, preserve_original=True)
    assert "valor_a" in result
    assert "{{b}}" in result


def test_extract_and_replace_remove(processor):
    """Testa remoção de placeholders não encontrados."""
    text = "{{a}} e {{b}}"
    replacements = {"a": "valor_a"}
    result = processor.extract_and_replace(text, replacements, preserve_original=False)
    assert "valor_a" in result
    assert "{{b}}" not in result


def test_validate_template_valid(processor):
    """Testa validação de template válido."""
    template = "Valores: {{x}}, {{y}}, {{z}}"
    is_valid, vars_required = processor.validate_template(template)
    assert is_valid is True
    assert set(vars_required) == {"x", "y", "z"}


def test_validate_template_malformed(processor):
    """Testa detecção de placeholder malformado."""
    template = "Valores: {{x}, {y}}"  # Falta fechar x, y tem só uma chave
    is_valid, vars_required = processor.validate_template(template)
    # Deve detectar malformados
    assert is_valid is False
def test_validate_template_single_braces(processor):
    """Testa detecção de chaves simples malformadas."""
    template = "Valores: {x} e {{y}}"  # {x} é malformado, {{y}} é válido
    is_valid, vars_required = processor.validate_template(template)
    assert is_valid is False  # Deve detectar {x} como erro
    
def test_validate_template_unbalanced(processor):
    """Testa detecção de chaves desbalanceadas."""
    template = "Valores: {{x} e {y}}"  # Ambos malformados
    is_valid, vars_required = processor.validate_template(template)
    assert is_valid is False
