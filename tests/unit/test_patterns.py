"""Testes de padrões regex."""
import pytest
from pymemorial.recognition.patterns import (
    find_variables,
    find_numbers,
    find_placeholders,
    has_greek_letters,
    VAR_NAME,
    NUMBER,
)


def test_find_variables():
    """Testa extração de variáveis."""
    text = "fck = 30 MPa e fy = 500 MPa"
    vars = find_variables(text)
    assert "fck" in vars
    assert "fy" in vars
    assert "MPa" in vars  # Também captura unidades


def test_find_numbers():
    """Testa extração de números."""
    text = "L = 5.5 m, fck = 30 MPa, taxa = 1.5e-3"
    numbers = find_numbers(text)
    assert 5.5 in numbers
    assert 30.0 in numbers
    assert 0.0015 in numbers


def test_find_placeholders():
    """Testa extração de placeholders."""
    template = "A resistência {{fck}} é de {{valor}} MPa."
    placeholders = find_placeholders(template)
    assert placeholders == ["fck", "valor"]


def test_has_greek_letters():
    """Testa detecção de letras gregas."""
    assert has_greek_letters("α = 0.85")
    assert has_greek_letters("Coef. Δ")
    assert not has_greek_letters("No greek here")


def test_var_name_pattern():
    """Testa padrão de nome de variável."""
    assert VAR_NAME.match("x")
    assert VAR_NAME.match("alpha_c")
    assert VAR_NAME.match("fck_28d")
    assert not VAR_NAME.match("123var")  # Não pode começar com número


def test_number_pattern():
    """Testa padrão de número."""
    assert NUMBER.match("42")
    assert NUMBER.match("3.14")
    assert NUMBER.match("-0.5")
    assert NUMBER.match("1.5e-3")
