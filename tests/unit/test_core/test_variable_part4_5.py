# tests/unit/test_core/test_variable_part4_5.py

"""
Testes PARTE 4 & 5: Auto-registro + Histórico + Comparações

PARTE 4: Integração com registry (auto_register)
PARTE 5: Histórico de valores e comparações
"""

import pytest

from pymemorial.core.variable import Variable, VariableError
from pymemorial.symbols import get_global_registry, reset_global_registry


# =============================================================================
# PARTE 4: AUTO-REGISTRO NO REGISTRY
# =============================================================================

class TestVariableAutoRegister:
    """Testes de auto-registro no registry."""
    
    def test_auto_register_on_creation(self):
        """Variável com auto_register=True deve registrar no registry."""
        reset_global_registry()
        registry = get_global_registry()
        
        v = Variable(name='f_ck', value=30, auto_register=True)
        
        assert registry.has('f_ck')
        sym = registry.get('f_ck')
        assert sym.name == 'f_ck'
    
    def test_no_register_when_disabled(self):
        """Não registrar quando auto_register=False."""
        reset_global_registry()
        registry = get_global_registry()
        
        v = Variable(name='temp_var', value=10, auto_register=False)
        
        assert not registry.has('temp_var')
    
    def test_default_auto_register_is_false(self):
        """Por padrão, auto_register deve ser False."""
        reset_global_registry()
        registry = get_global_registry()
        
        v = Variable(name='no_register', value=5)
        
        # Não deve registrar automaticamente
        assert not registry.has('no_register')


# =============================================================================
# PARTE 5: HISTÓRICO E COMPARAÇÕES
# =============================================================================

class TestVariableHistory:
    """Testes de histórico de valores."""
    
    def test_history_tracks_changes(self):
        """Histórico deve rastrear mudanças."""
        v = Variable(name='x', value=10)
        v.update_value(20)
        v.update_value(30)
        
        history = v.get_history()
        assert len(history) >= 2
        # Primeiro valor registrado
        assert any(h[1] == 10 for h in history)
    
    def test_rollback_one_step(self):
        """Rollback para valor anterior."""
        v = Variable(name='y', value=5)
        v.update_value(10)
        v.update_value(15)
        
        v.rollback(1)
        assert v.value == 10
    
    def test_rollback_to_original(self):
        """Rollback para valor original."""
        v = Variable(name='z', value=1)
        v.update_value(2)
        v.update_value(3)
        
        v.rollback(2)
        assert v.value == 1


class TestVariableComparison:
    """Testes de comparação."""
    
    def test_equality_by_value(self):
        """Comparação de igualdade por valor."""
        v1 = Variable(name='a', value=10)
        v2 = Variable(name='b', value=10)
        assert v1 == v2  # Compara valores
    
    def test_less_than(self):
        """Menor que."""
        v1 = Variable(name='m', value=3)
        v2 = Variable(name='n', value=5)
        assert v1 < v2
    
    def test_greater_than(self):
        """Maior que."""
        v1 = Variable(name='p', value=100)
        v2 = Variable(name='q', value=50)
        assert v1 > v2
    
    def test_comparison_with_scalar(self):
        """Comparação com escalar."""
        v = Variable(name='x', value=10)
        assert v == 10
        assert v < 20
        assert v > 5


# Executar: pytest tests/unit/test_core/test_variable_part4_5.py -v
# Esperado: 10 FAILURES (RED) antes da implementação ✅
