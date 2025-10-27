# tests/unit/test_core/test_units_part2.py

"""
Testes PARTE 2: UnitParser - Aliases PT-BR

OBJETIVO:
- Parsing robusto com aliases em português
- Normalização de unidades (kN.m → kN*m)
- Múltiplos formatos de input
- Cache de conversões (performance)
"""

import pytest

from pymemorial.core.units import (
    get_unit_registry,
    UnitParser,
    PINT_AVAILABLE
)


class TestUnitParserAliasesPTBR:
    """Testes de aliases em português."""
    
    @pytest.mark.skipif(not PINT_AVAILABLE, reason="Pint not installed")
    def test_parse_metros(self):
        """Alias: 'metros' → 'm'"""
        reg = get_unit_registry()
        parser = UnitParser(reg)
        
        quantity = parser.parse_with_alias('10 metros')
        assert quantity is not None
        assert abs(quantity.magnitude - 10.0) < 0.001
    
    @pytest.mark.skipif(not PINT_AVAILABLE, reason="Pint not installed")
    def test_parse_quilonewtons(self):
        """Alias: 'quilonewtons' → 'kN'"""
        reg = get_unit_registry()
        parser = UnitParser(reg)
        
        quantity = parser.parse_with_alias('50 quilonewtons')
        
        # Converter para N para validar
        q_n = reg.convert(quantity, 'N')
        assert abs(q_n.magnitude - 50000) < 1.0
    
    @pytest.mark.skipif(not PINT_AVAILABLE, reason="Pint not installed")
    def test_parse_megapascals(self):
        """Alias: 'megapascals' → 'MPa'"""
        reg = get_unit_registry()
        parser = UnitParser(reg)
        
        quantity = parser.parse_with_alias('25 megapascals')
        
        # Converter para Pa
        q_pa = reg.convert(quantity, 'Pa')
        assert abs(q_pa.magnitude - 25e6) < 1e3


class TestUnitParserNormalization:
    """Testes de normalização de unidades."""
    
    @pytest.mark.skipif(not PINT_AVAILABLE, reason="Pint not installed")
    def test_normalize_dot_to_star(self):
        """Normaliza: 'kN.m' → 'kN*m'"""
        reg = get_unit_registry()
        parser = UnitParser(reg)
        
        # kN.m é ambíguo, deve converter para kN*m
        normalized = parser.normalize('kN.m')
        assert normalized == 'kN*m'
    
    @pytest.mark.skipif(not PINT_AVAILABLE, reason="Pint not installed")
    def test_normalize_slash_to_per(self):
        """Normaliza: 'kN/m2' → 'kN/m**2'"""
        reg = get_unit_registry()
        parser = UnitParser(reg)
        
        normalized = parser.normalize('kN/m2')
        assert 'kN/m**2' in normalized or 'kN/m^2' in normalized
    
    @pytest.mark.skipif(not PINT_AVAILABLE, reason="Pint not installed")
    def test_parse_with_normalization(self):
        """Parse com normalização automática."""
        reg = get_unit_registry()
        parser = UnitParser(reg)
        
        # Usuário escreve kN.m (ambíguo)
        quantity = parser.parse('10 kN.m')
        
        # Sistema interpreta como kN*m (momento)
        assert quantity is not None
        assert abs(quantity.magnitude - 10.0) < 0.001


# Executar: pytest tests/unit/test_core/test_units_part2.py -v
# Esperado: 6 FAILURES (RED) antes da implementação ✅
class TestUnitParserEdgeCases:
    """Testes de casos extremos."""
    
    @pytest.mark.skipif(not PINT_AVAILABLE, reason="Pint not installed")
    def test_parse_nested_alias(self):
        """Alias com sobreposição: 'quilonewtons' contém 'newtons'."""
        reg = get_unit_registry()
        parser = UnitParser(reg)
        
        # Deve processar 'quilonewtons' completo, não 'newtons'
        quantity = parser.parse_with_alias('100 quilonewtons')
        
        # 100 kN = 100000 N
        q_n = reg.convert(quantity, 'N')
        assert abs(q_n.magnitude - 100000) < 10.0
