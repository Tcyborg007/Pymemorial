"""Testes do parser de variáveis."""
import pytest
from pymemorial.recognition.parser import VariableParser, ParsedVariable


@pytest.fixture
def parser():
    """Fixture do parser."""
    return VariableParser()


def test_parse_line_simple(parser):
    """Testa parsing de linha simples."""
    result = parser.parse_line("fck = 30 MPa")
    assert result is not None
    assert result.name == "fck"
    assert result.value == 30.0
    assert result.unit == "MPa"


def test_parse_line_with_description(parser):
    """Testa parsing com descrição."""
    result = parser.parse_line("L = 5.5 m (vão da viga)")
    assert result is not None
    assert result.name == "L"
    assert result.value == 5.5
    assert result.unit == "m"
    assert result.description == "vão da viga"


def test_parse_line_with_greek(parser):
    """Testa parsing com símbolo grego."""
    result = parser.parse_line("α = 0.85 (coeficiente)")
    assert result is not None
    assert result.name == "alpha"  # Convertido
    assert result.value == 0.85
    assert result.has_greek is True


def test_parse_line_scientific_notation(parser):
    """Testa parsing de notação científica."""
    result = parser.parse_line("E = 2.1e5 MPa")
    assert result is not None
    assert result.value == 210000.0


def test_parse_line_no_match(parser):
    """Testa linha sem padrão válido."""
    result = parser.parse_line("Este é um texto qualquer")
    assert result is None


def test_parse_text_multiline(parser):
    """Testa parsing de texto multi-linha."""
    text = """
    fck = 30 MPa (resistência do concreto)
    fy = 500 MPa (tensão de escoamento do aço)
    L = 6.0 m
    """
    results = parser.parse_text(text)
    assert len(results) == 3
    assert results[0].name == "fck"
    assert results[1].name == "fy"
    assert results[2].name == "L"


def test_parse_equation(parser):
    """Testa parsing de equação."""
    name, expr = parser.parse_equation("M = P * L / 4")
    assert name == "M"
    assert expr == "P * L / 4"


def test_parse_equation_complex(parser):
    """Testa equação complexa."""
    name, expr = parser.parse_equation("sigma = M / W")
    assert name == "sigma"
    assert expr == "M / W"


def test_parse_equation_no_equals(parser):
    """Testa texto sem sinal de igual."""
    name, expr = parser.parse_equation("apenas texto")
    assert name is None
    assert expr is None


def test_extract_variable_names(parser):
    """Testa extração de nomes."""
    text = "Calcular M usando fck e fy para dimensionamento"
    names = parser.extract_variable_names(text)
    assert "fck" in names
    assert "fy" in names
    assert "dimensionamento" in names
    # Palavras curtas/comuns devem ser filtradas
    assert "e" not in names
