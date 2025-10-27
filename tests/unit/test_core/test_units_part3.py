# tests/unit/test_core/test_units_part3.py

"""
Testes PARTE 3: UnitValidator - Validação Dimensional

OBJETIVO:
- Validação de consistência dimensional
- Verificação de operações (força + comprimento = ???)
- Inferência de unidades resultado
- Error messages claros
"""

import pytest

from pymemorial.core.units import (
    get_unit_registry,
    UnitValidator,
    UnitError,
    PINT_AVAILABLE
)


class TestUnitValidatorConsistency:
    """Testes de consistência dimensional."""
    
    @pytest.mark.skipif(not PINT_AVAILABLE, reason="Pint not installed")
    def test_check_compatible_units(self):
        """Unidades compatíveis (mesma dimensão)."""
        reg = get_unit_registry()
        validator = UnitValidator(reg)
        
        # m e cm são compatíveis (comprimento)
        q1 = reg.parse('10 m')
        q2 = reg.parse('50 cm')
        
        assert validator.are_compatible(q1, q2) is True
    
    @pytest.mark.skipif(not PINT_AVAILABLE, reason="Pint not installed")
    def test_check_incompatible_units(self):
        """Unidades incompatíveis (dimensões diferentes)."""
        reg = get_unit_registry()
        validator = UnitValidator(reg)
        
        # m (comprimento) e kg (massa) são incompatíveis
        q1 = reg.parse('10 m')
        q2 = reg.parse('5 kg')
        
        assert validator.are_compatible(q1, q2) is False


class TestUnitValidatorOperations:
    """Testes de validação de operações."""
    
    @pytest.mark.skipif(not PINT_AVAILABLE, reason="Pint not installed")
    def test_validate_addition_compatible(self):
        """Adição: unidades compatíveis (OK)."""
        reg = get_unit_registry()
        validator = UnitValidator(reg)
        
        q1 = reg.parse('10 m')
        q2 = reg.parse('50 cm')
        
        # Deve passar (ambos comprimento)
        validator.validate_operation(q1, q2, 'add')  # Não levanta erro
    
    @pytest.mark.skipif(not PINT_AVAILABLE, reason="Pint not installed")
    def test_validate_addition_incompatible(self):
        """Adição: unidades incompatíveis (ERRO)."""
        reg = get_unit_registry()
        validator = UnitValidator(reg)
        
        q1 = reg.parse('10 m')
        q2 = reg.parse('5 kg')
        
        # Deve falhar (comprimento + massa)
        # CORRIGIDO: Apenas verificar se contém "incompatible"
        with pytest.raises(UnitError, match="incompatible"):
            validator.validate_operation(q1, q2, 'add')

    
    @pytest.mark.skipif(not PINT_AVAILABLE, reason="Pint not installed")
    def test_validate_multiplication(self):
        """Multiplicação: sempre válida."""
        reg = get_unit_registry()
        validator = UnitValidator(reg)
        
        q1 = reg.parse('10 m')
        q2 = reg.parse('5 kg')
        
        # Multiplicação sempre válida (cria nova dimensão)
        validator.validate_operation(q1, q2, 'mul')  # Não levanta erro


# Executar: pytest tests/unit/test_core/test_units_part3.py -v
# Esperado: 5 FAILURES (RED) antes da implementação ✅
