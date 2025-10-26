# tests/unit/test_core/test_variable_part1.py

"""
Testes PARTE 1: Core básico de Variable

Apenas criação, atribuição e representação.
SEM operadores, SEM LaTeX, SEM registry ainda!
"""

import pytest

from pymemorial.core.variable import Variable, VariableError


class TestVariableCreationBasic:
    """Testes básicos de criação."""
    
    def test_create_simple_scalar(self):
        """Criar variável escalar simples."""
        v = Variable(name='M', value=150.0)
        assert v.name == 'M'
        assert v.value == 150.0
        assert v.unit is None
    
    def test_create_with_description(self):
        """Criar variável com descrição."""
        v = Variable(name='f_ck', value=30, description='Resistência do concreto')
        assert v.description == 'Resistência do concreto'
    
    def test_create_with_unit_string(self):
        """Criar variável com unidade (string simples)."""
        v = Variable(name='L', value=6.0, unit='m')
        assert v.value == 6.0
        assert v.unit == 'm'
    
    def test_str_representation(self):
        """String simples."""
        v = Variable(name='M', value=150.5)
        s = str(v)
        assert 'M' in s
        assert '150' in s
    
    def test_repr_representation(self):
        """Repr deve mostrar informações."""
        v = Variable(name='sigma', value=25, unit='MPa')
        r = repr(v)
        assert 'sigma' in r
        assert '25' in r


# Executar: pytest tests/unit/test_core/test_variable_part1.py -v
# Esperado: 5 FAILURES (RED) ✅
