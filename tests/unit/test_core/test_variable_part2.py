# tests/unit/test_core/test_variable_part2.py

"""
Testes PARTE 2: Operadores matemáticos

Adiciona suporte a operações aritméticas entre variáveis.
"""

import pytest

from pymemorial.core.variable import Variable, VariableError


class TestVariableArithmetic:
    """Testes de operações aritméticas."""
    
    def test_add_two_variables(self):
        """Somar duas variáveis."""
        v1 = Variable(name='a', value=10)
        v2 = Variable(name='b', value=5)
        v3 = v1 + v2
        assert v3.value == 15
        assert v3.name.startswith('result_')  # Nome auto-gerado
    
    def test_subtract_variables(self):
        """Subtrair variáveis."""
        v1 = Variable(name='M_max', value=200)
        v2 = Variable(name='M_min', value=50)
        v3 = v1 - v2
        assert v3.value == 150
    
    def test_multiply_variables(self):
        """Multiplicar variáveis."""
        v1 = Variable(name='A', value=0.05)
        v2 = Variable(name='f_y', value=500)
        v3 = v1 * v2
        assert v3.value == pytest.approx(25.0)
    
    def test_divide_variables(self):
        """Dividir variáveis."""
        v1 = Variable(name='M', value=100)
        v2 = Variable(name='W', value=4)
        v3 = v1 / v2
        assert v3.value == 25.0
    
    def test_power_operation(self):
        """Potenciação."""
        v = Variable(name='L', value=2.0)
        v2 = v ** 2
        assert v2.value == 4.0
    
    def test_add_variable_and_scalar(self):
        """Somar variável com escalar."""
        v = Variable(name='x', value=10)
        v2 = v + 5
        assert v2.value == 15
    
    def test_multiply_scalar_and_variable(self):
        """Multiplicar escalar por variável (ordem reversa)."""
        v = Variable(name='k', value=3)
        v2 = 2 * v  # __rmul__
        assert v2.value == 6
    
    def test_negate_variable(self):
        """Negação unária."""
        v = Variable(name='F', value=100)
        v_neg = -v
        assert v_neg.value == -100
    
    def test_arithmetic_with_none_value_raises_error(self):
        """Operação com valor None deve falhar."""
        v = Variable(name='empty')
        with pytest.raises(VariableError):
            _ = v + 10


# Executar: pytest tests/unit/test_core/test_variable_part2.py -v
# Esperado: 9 FAILURES (RED) antes da implementação ✅
