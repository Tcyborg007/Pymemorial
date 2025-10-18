"""
Sistema de unidades físicas para PyMemorial.
Integra Pint (SI) e forallpeople (engenharia).
"""
from typing import Union, Any
import pint
from decimal import Decimal

# Registro global Pint (configuração SI)
ureg = pint.UnitRegistry()
ureg.default_format = "~P"  # Formato compacto

# Aliases para tipos
Quantity = Union[pint.Quantity, float, int, Decimal]

def parse_quantity(value: Any, unit: str = None) -> pint.Quantity:
    """
    Converte valor para pint.Quantity.
    
    Args:
        value: valor numérico ou string com unidade
        unit: unidade (opcional se value já contém)
    
    Returns:
        pint.Quantity
    
    Examples:
        >>> parse_quantity(10, "m")
        <Quantity(10, 'meter')>
        >>> parse_quantity("10 kN")
        <Quantity(10, 'kilonewton')>
    """
    if isinstance(value, pint.Quantity):
        return value
    
    if isinstance(value, str):
        return ureg.parse_expression(value)
    
    if unit:
        return ureg.Quantity(value, unit)
    
    # Sem unidade: adimensional
    return ureg.Quantity(value, "dimensionless")

def check_dimensional_consistency(q1: pint.Quantity, q2: pint.Quantity) -> bool:
    """Verifica se duas grandezas têm dimensões compatíveis."""
    return q1.dimensionality == q2.dimensionality

def strip_units(q: pint.Quantity) -> float:
    """Remove unidades e retorna valor numérico (magnitude)."""
    return float(q.magnitude)
