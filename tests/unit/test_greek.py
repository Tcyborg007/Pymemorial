"""Testes de conversão de símbolos gregos."""
import pytest
from pymemorial.recognition.greek import (
    GreekSymbols,
    GREEK_LOWER,
    GREEK_UPPER,
    GREEK_TO_ASCII,
)


def test_greek_to_ascii_lowercase():
    """Testa conversão de minúsculas."""
    assert GreekSymbols.to_ascii("α") == "alpha"
    assert GreekSymbols.to_ascii("β") == "beta"
    assert GreekSymbols.to_ascii("γ") == "gamma"


def test_greek_to_ascii_uppercase():
    """Testa conversão de maiúsculas."""
    assert GreekSymbols.to_ascii("Δ") == "Delta"
    assert GreekSymbols.to_ascii("Φ") == "Phi"


def test_greek_to_ascii_mixed_text():
    """Testa conversão em texto misto."""
    text = "Coeficiente α = 0.85 e β = 1.2"
    expected = "Coeficiente alpha = 0.85 e beta = 1.2"
    assert GreekSymbols.to_ascii(text) == expected


def test_ascii_to_unicode():
    """Testa conversão ASCII → Unicode."""
    assert GreekSymbols.to_unicode("alpha") == "α"
    assert GreekSymbols.to_unicode("beta") == "β"


def test_ascii_to_unicode_word_boundary():
    """Testa que não substitui parcialmente."""
    # "alphabet" não deve virar "αbet"
    text = "alphabet"
    result = GreekSymbols.to_unicode(text)
    assert result == "alphabet"  # Sem substituição parcial


def test_get_ascii_name():
    """Testa busca de nome ASCII."""
    assert GreekSymbols.get_ascii_name("θ") == "theta"
    assert GreekSymbols.get_ascii_name("Ω") == "Omega"
    assert GreekSymbols.get_ascii_name("x") is None


def test_get_unicode_symbol():
    """Testa busca de símbolo Unicode."""
    assert GreekSymbols.get_unicode_symbol("sigma") == "σ"
    assert GreekSymbols.get_unicode_symbol("lambda") == "λ"
    assert GreekSymbols.get_unicode_symbol("invalid") is None


def test_greek_dictionaries_populated():
    """Testa que os dicionários estão preenchidos."""
    assert len(GREEK_LOWER) >= 24  # 24 letras no alfabeto grego
    assert len(GREEK_UPPER) >= 24
    assert len(GREEK_TO_ASCII) >= 48
def test_sigma_variants():
    """Testa que ambas as formas de sigma são reconhecidas."""
    # Sigma normal (meio de palavra/variável)
    assert GreekSymbols.to_ascii("σ") == "sigma"
    # Sigma final (menos comum, mas deve funcionar)
    assert GreekSymbols.to_ascii("ς") == "sigma"
    
    # Caso de uso real: variável com sigma no texto técnico
    texto = "A tensão σ_max = 150 MPa"
    assert GreekSymbols.to_ascii(texto) == "A tensão sigma_max = 150 MPa"
