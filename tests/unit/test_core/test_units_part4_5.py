# tests/unit/test_core/test_units_part4_5.py

"""
Testes PARTES 4+5: UnitFormatter + Integration

PARTE 4: UnitFormatter - Formatação LaTeX de unidades
PARTE 5: Integration - Testes de integração completa
"""

import pytest

from pymemorial.core.units import (
    get_unit_registry,
    UnitParser,
    UnitValidator,
    UnitFormatter,
    PINT_AVAILABLE
)


# =============================================================================
# PARTE 4: UNIT FORMATTER
# =============================================================================

class TestUnitFormatterLaTeX:
    """Testes de formatação LaTeX."""
    
    @pytest.mark.skipif(not PINT_AVAILABLE, reason="Pint not installed")
    def test_format_simple_unit(self):
        """Formatação: unidade simples."""
        reg = get_unit_registry()
        formatter = UnitFormatter(reg)
        
        q = reg.parse('10 m')
        latex = formatter.to_latex(q)
        
        # Deve conter unidade formatada
        assert 'm' in latex or 'meter' in latex
    
    @pytest.mark.skipif(not PINT_AVAILABLE, reason="Pint not installed")
    def test_format_compound_unit(self):
        """Formatação: unidade composta (kN*m)."""
        reg = get_unit_registry()
        formatter = UnitFormatter(reg)
        
        # kN*m (momento)
        q = reg.parse('50 kN*m')
        latex = formatter.to_latex(q)
        
        # Deve conter kN e m
        assert 'kN' in latex or 'kilonewton' in latex
        assert 'm' in latex or 'meter' in latex


class TestUnitFormatterPrecision:
    """Testes de precisão na formatação."""
    
    @pytest.mark.skipif(not PINT_AVAILABLE, reason="Pint not installed")
    def test_format_with_precision(self):
        """Formatação com precisão customizada."""
        reg = get_unit_registry()
        formatter = UnitFormatter(reg)
        
        q = reg.parse('3.14159 m')
        
        # Precisão 2 casas
        latex = formatter.to_latex(q, precision=2)
        assert '3.14' in latex
    
    @pytest.mark.skipif(not PINT_AVAILABLE, reason="Pint not installed")
    def test_format_scientific_notation(self):
        """Formatação em notação científica."""
        reg = get_unit_registry()
        formatter = UnitFormatter(reg)
        
        q = reg.parse('1000000 Pa')
        
        # Notação científica
        latex = formatter.to_latex(q, scientific=True)
        # Deve conter 10^ ou ×10
        assert '10^' in latex or r'\times' in latex


# =============================================================================
# PARTE 5: INTEGRATION TESTS
# =============================================================================

class TestFullIntegration:
    """Testes de integração completa."""
    
    @pytest.mark.skipif(not PINT_AVAILABLE, reason="Pint not installed")
    def test_full_workflow_parse_validate_format(self):
        """Workflow completo: Parse → Validate → Format."""
        reg = get_unit_registry()
        parser = UnitParser(reg)
        validator = UnitValidator(reg)
        formatter = UnitFormatter(reg)
        
        # 1. Parse com alias PT-BR
        q1 = parser.parse_with_alias('10 metros')
        q2 = parser.parse_with_alias('50 centimetros')
        
        # 2. Validar compatibilidade
        assert validator.are_compatible(q1, q2) is True
        
        # 3. Formatar LaTeX
        latex1 = formatter.to_latex(q1)
        assert isinstance(latex1, str)
        assert len(latex1) > 0
    
    @pytest.mark.skipif(not PINT_AVAILABLE, reason="Pint not installed")
    def test_integration_with_normalization(self):
        """Integração: Normalização + Validação."""
        reg = get_unit_registry()
        parser = UnitParser(reg)
        validator = UnitValidator(reg)
        
        # Parse com normalização (kN.m → kN*m)
        q1 = parser.parse('10 kN.m')
        q2 = parser.parse('5 kN*m')
        
        # Ambos devem ser compatíveis (mesma dimensão: força × comprimento)
        assert validator.are_compatible(q1, q2) is True
    
    @pytest.mark.skipif(not PINT_AVAILABLE, reason="Pint not installed")
    def test_integration_error_handling(self):
        """Integração: Error handling consistente."""
        reg = get_unit_registry()
        parser = UnitParser(reg)
        validator = UnitValidator(reg)
        
        # Parse OK
        q1 = parser.parse('10 m')
        q2 = parser.parse('5 kg')
        
        # Validação deve detectar incompatibilidade
        assert validator.are_compatible(q1, q2) is False


# Executar: pytest tests/unit/test_core/test_units_part4_5.py -v
# Esperado: 7 FAILURES (RED) antes da implementação ✅
