# tests/unit/test_core/test_variable_part3.py

"""
Testes PARTE 3: Conversão LaTeX e integração com registry

Integra Variable com ast_parser e custom_registry.
"""

import pytest

from pymemorial.core.variable import Variable
from pymemorial.symbols import get_global_registry, reset_global_registry


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
        assert '{c}' in latex or '_{c}' in latex
    
    def test_to_latex_respects_registry_custom(self):
        """LaTeX deve usar definição customizada do registry."""
        reset_global_registry()
        registry = get_global_registry()
        
        # Definir símbolo customizado
        registry.define('My_custom', latex=r'M^{\text{custom}}_{y}')
        
        v = Variable(name='My_custom', value=50)
        latex = v.to_latex()
        assert latex == r'M^{\text{custom}}_{y}'


# Executar: pytest tests/unit/test_core/test_variable_part3.py -v
# Esperado: 4 FAILURES (RED) antes da implementação ✅
