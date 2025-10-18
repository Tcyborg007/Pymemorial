"""Testes do sistema de unidades."""
import pytest
from pymemorial.core.units import (
    parse_quantity, 
    check_dimensional_consistency,
    strip_units,
    ureg
)

def test_parse_quantity_numeric():
    """Testa conversão de numérico com unidade."""
    q = parse_quantity(10, "m")
    assert q.magnitude == 10
    assert q.units == ureg.meter

def test_parse_quantity_string():
    """Testa parsing de string com unidade."""
    q = parse_quantity("25 kN")
    assert q.magnitude == 25
    assert q.units == ureg.kilonewton

def test_dimensional_consistency():
    """Testa verificação dimensional."""
    force = parse_quantity(100, "N")
    mass = parse_quantity(10, "kg")
    force2 = parse_quantity(50, "kN")
    
    assert check_dimensional_consistency(force, force2)
    assert not check_dimensional_consistency(force, mass)

def test_strip_units():
    """Testa extração de magnitude."""
    q = parse_quantity(42.5, "MPa")
    assert strip_units(q) == 42.5

def test_parse_quantity_dimensionless():
    """Testa quantidade adimensional."""
    q = parse_quantity(1.5)  # Sem unidade
    assert q.magnitude == 1.5
    assert q.dimensionless
def test_parse_quantity_already_quantity():
    """Testa parse quando entrada já é Quantity."""
    from pymemorial.core.units import ureg
    q1 = ureg.Quantity(100, "kN")
    q2 = parse_quantity(q1)
    assert q2 is q1  # Deve retornar o mesmo objeto
