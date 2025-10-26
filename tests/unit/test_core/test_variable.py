# tests/unit/test_core/test_variable.py

"""
Testes completos para core/variable.py v2.0

FILOSOFIA:
- Variável SEM pré-definições de norma
- Liberdade total de unidades
- Auto-registro no custom_registry
- Integração com config.py

FUNCIONALIDADES TESTADAS:
- Criação simples (escalar, string, com unidade)
- Operadores matemáticos (+, -, *, /, **)
- Conversão LaTeX automática
- Auto-registro no registry
- Histórico de valores
- Comparações e validações
"""

import pytest
import tempfile
from pathlib import Path

# Imports condicionais para dependências opcionais
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    import pint
    PINT_AVAILABLE = True
except ImportError:
    PINT_AVAILABLE = False

from pymemorial.core.variable import Variable, VariableError
from pymemorial.symbols import get_global_registry, reset_global_registry


class TestVariableCreation:
    """Testes de criação de variáveis."""
    
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
    
    @pytest.mark.skipif(not PINT_AVAILABLE, reason="Pint not installed")
    def test_create_with_unit_string(self):
        """Criar variável com unidade (string)."""
        v = Variable(name='L', value=6.0, unit='m')
        assert v.value == 6.0
        assert v.unit == 'm'
    
    @pytest.mark.skipif(not PINT_AVAILABLE, reason="Pint not installed")
    def test_create_with_compound_unit(self):
        """Criar variável com unidade composta."""
        v = Variable(name='sigma', value=25, unit='MPa')
        assert v.value == 25
        assert v.unit == 'MPa'
    
    def test_create_from_string_value(self):
        """Criar variável parseando string."""
        v = Variable(name='x', value='10.5')
        assert v.value == 10.5
    
    @pytest.mark.skipif(not NUMPY_AVAILABLE, reason="NumPy not installed")
    def test_create_vector(self):
        """Criar variável vetorial."""
        v = Variable(name='coords', value=[1.0, 2.0, 3.0])
        assert v.is_vector is True
        assert len(v.value) == 3
    
    def test_create_none_value(self):
        """Criar variável sem valor inicial."""
        v = Variable(name='empty')
        assert v.value is None


class TestVariableArithmetic:
    """Testes de operações aritméticas."""
    
    def test_add_two_variables(self):
        """Somar duas variáveis."""
        v1 = Variable(name='a', value=10)
        v2 = Variable(name='b', value=5)
        v3 = v1 + v2
        assert v3.value == 15
    
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


class TestVariableLatex:
    """Testes de conversão LaTeX."""
    
    def test_to_latex_simple(self):
        """Conversão LaTeX simples."""
        v = Variable(name='M', value=100)
        latex = v.to_latex()
        assert latex == 'M'  # Nome simples
    
    def test_to_latex_with_subscript(self):
        """Conversão LaTeX com subscrito."""
        v = Variable(name='M_d', value=150)
        latex = v.to_latex()
        assert latex == r'M_{d}'
    
    def test_to_latex_greek_symbol(self):
        """Conversão LaTeX com letra grega."""
        v = Variable(name='gamma_c', value=1.4)
        latex = v.to_latex()
        assert r'\gamma' in latex
        assert '{c}' in latex
    
    @pytest.mark.skipif(not PINT_AVAILABLE, reason="Pint not installed")
    def test_to_latex_with_unit(self):
        """Conversão LaTeX com unidade."""
        v = Variable(name='sigma', value=25, unit='MPa')
        latex = v.to_latex(include_unit=True)
        assert 'MPa' in latex or 'Pa' in latex


class TestVariableRegistry:
    """Testes de integração com custom_registry."""
    
    def test_auto_register_on_creation(self):
        """Variável deve auto-registrar no registry."""
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
    
    def test_registry_updates_latex(self):
        """Registry deve fornecer LaTeX customizado."""
        reset_global_registry()
        registry = get_global_registry()
        
        # Define símbolo customizado no registry
        registry.define('My_custom', latex=r'M^{\text{custom}}_{y}')
        
        v = Variable(name='My_custom', value=50)
        latex = v.to_latex()
        assert latex == r'M^{\text{custom}}_{y}'


class TestVariableHistory:
    """Testes de histórico de valores."""
    
    def test_history_tracks_changes(self):
        """Histórico deve rastrear mudanças."""
        v = Variable(name='x', value=10)
        v.update_value(20)
        v.update_value(30)
        
        history = v.get_history()
        assert len(history) == 2
        assert history[0][1] == 10  # Primeiro valor
        assert history[1][1] == 20  # Segundo valor
    
    def test_rollback_one_step(self):
        """Rollback para valor anterior."""
        v = Variable(name='y', value=5)
        v.update_value(10)
        v.update_value(15)
        
        v.rollback(1)
        assert v.value == 10
    
    def test_rollback_multiple_steps(self):
        """Rollback múltiplos passos."""
        v = Variable(name='z', value=1)
        v.update_value(2)
        v.update_value(3)
        v.update_value(4)
        
        v.rollback(2)
        assert v.value == 2
    
    def test_rollback_too_far_raises_error(self):
        """Rollback além do histórico deve falhar."""
        v = Variable(name='w', value=10)
        v.update_value(20)
        
        with pytest.raises(VariableError):
            v.rollback(5)  # Apenas 1 item no histórico


class TestVariableComparison:
    """Testes de comparação."""
    
    def test_equality(self):
        """Comparação de igualdade."""
        v1 = Variable(name='a', value=10)
        v2 = Variable(name='b', value=10)
        assert v1 == v2  # Compara valores
    
    def test_inequality(self):
        """Comparação de desigualdade."""
        v1 = Variable(name='x', value=5)
        v2 = Variable(name='y', value=10)
        assert v1 != v2
    
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


class TestVariableRepresentation:
    """Testes de representação string."""
    
    def test_str_simple(self):
        """String simples."""
        v = Variable(name='M', value=150.5)
        s = str(v)
        assert 'M' in s
        assert '150' in s
    
    def test_str_with_unit(self):
        """String com unidade."""
        v = Variable(name='F', value=100, unit='kN')
        s = str(v)
        assert 'F' in s
        assert '100' in s
        assert 'kN' in s or 'N' in s  # Pode estar em unidade base
    
    def test_repr_shows_all_info(self):
        """Repr deve mostrar informações completas."""
        v = Variable(name='sigma', value=25, unit='MPa', description='Tensão')
        r = repr(v)
        assert 'sigma' in r
        assert '25' in r


# Executar: pytest tests/unit/test_core/test_variable.py -v
# Esperado: TODOS FALHAM (RED) ✅
