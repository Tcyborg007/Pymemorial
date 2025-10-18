"""Testes de variáveis."""
import pytest
import sympy as sp
from pymemorial.core.variable import Variable, VariableFactory
from pymemorial.core.units import parse_quantity


def test_variable_creation():
    """Testa criação direta de variável."""
    fck = Variable(
        name="fck",
        symbol=sp.Symbol("fck"),
        value=parse_quantity(25, "MPa"),
        description="Resistência do concreto"
    )
    assert fck.name == "fck"
    assert fck.magnitude == 25


def test_variable_factory():
    """Testa factory de variáveis."""
    var = VariableFactory.create("L", value=5.0, unit="m", description="Vão")
    assert var.name == "L"
    assert var.value.magnitude == 5.0
    assert var.value.units.dimensionality == "[length]"


def test_variable_no_value():
    """Testa variável sem valor atribuído."""
    var = VariableFactory.create("x")
    with pytest.raises(ValueError, match="não tem valor atribuído"):
        _ = var.magnitude


def test_variable_invalid_symbol():
    """Testa erro quando symbol não é sympy.Symbol."""
    with pytest.raises(TypeError, match="symbol deve ser sympy.Symbol"):
        Variable(name="x", symbol="not_a_symbol", value=None)


def test_variable_repr_without_value():
    """Testa repr de variável sem valor."""
    var = Variable(name="x", symbol=sp.Symbol("x"), description="Teste")
    repr_str = repr(var)
    assert "x" in repr_str
    assert "Variable" in repr_str


def test_variable_repr_with_value():
    """Testa repr de variável com valor e unidade."""
    var = VariableFactory.create("fck", value=30, unit="MPa")
    repr_str = repr(var)
    assert "fck" in repr_str
    assert "30" in repr_str
    # Pode aparecer como "megapascal" ou "MPa" dependendo da formatação do Pint
    assert ("megapascal" in repr_str.lower() or "MPa" in repr_str or "pascal" in repr_str.lower())


def test_variable_factory_string_value():
    """Testa factory com string contendo unidade."""
    var = VariableFactory.create("L", value="5.5 m")
    assert var.magnitude == 5.5
    assert var.value.dimensionality == "[length]"


def test_variable_factory_no_unit():
    """Testa factory com valor numérico sem unidade (adimensional)."""
    var = VariableFactory.create("alpha", value=0.85)
    assert var.magnitude == 0.85
    assert var.value.dimensionless


def test_variable_factory_without_value():
    """Testa factory sem fornecer valor inicial."""
    var = VariableFactory.create("y", description="Variável simbólica")
    assert var.name == "y"
    assert var.value is None
    assert isinstance(var.symbol, sp.Symbol)


def test_variable_magnitude_with_value():
    """Testa acesso à magnitude quando há valor."""
    var = VariableFactory.create("E", value=200, unit="GPa")
    assert var.magnitude == 200.0


def test_variable_symbol_type():
    """Testa que symbol é sempre sympy.Symbol."""
    var = VariableFactory.create("beta", value=1.2)
    assert isinstance(var.symbol, sp.Symbol)
    assert var.symbol.name == "beta"


def test_variable_with_complex_unit():
    """Testa variável com unidade composta (pressão/tensão)."""
    var = VariableFactory.create("stress", value=150, unit="kN/m**2")
    assert var.magnitude == 150
    
    # Pint decompõe força em dimensões fundamentais: [mass]*[length]/[time]²
    # Pressão = força/área = [mass]*[length]/[time]² / [length]² = [mass]/([length]*[time]²)
    dim = var.value.dimensionality
    assert dim['[mass]'] == 1
    assert dim['[length]'] == -1
    assert dim['[time]'] == -2
