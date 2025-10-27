# tests/test_smart_text_processor.py
"""
Testes Unitários: SmartTextProcessor v3.0

Testa todas as funcionalidades do processador de texto inteligente:
- {var} value display
- {{var}} formula display
- @eq equation processing
- Integration bridge
- Compatibility v2.0

Author: PyMemorial Team
Date: October 2025
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Imports
try:
    from pymemorial.recognition import (
        SmartTextProcessor,
        VariableRegistry,
        ProcessingOptions,
        DocumentType,
        RenderMode,
        TEXT_PROCESSOR_V3_AVAILABLE,
    )
    from pymemorial.document._internal.text_processing import (
        create_text_bridge,
        TextProcessingBridge,
        RECOGNITION_AVAILABLE,
    )
    TESTS_AVAILABLE = TEXT_PROCESSOR_V3_AVAILABLE and RECOGNITION_AVAILABLE
except ImportError:
    TESTS_AVAILABLE = False


# Skip all tests if v3.0 not available
pytestmark = pytest.mark.skipif(
    not TESTS_AVAILABLE,
    reason="SmartTextProcessor v3.0 not available (requires SymPy)"
)


class TestVariableRegistry:
    """Testa VariableRegistry."""
    
    def test_register_variable(self):
        """Testa registro de variável."""
        registry = VariableRegistry()
        ctx = registry.register('M_k', 112.5, 'kN.m', 'Momento')
        
        assert ctx.name == 'M_k'
        assert ctx.value == 112.5
        assert ctx.unit == 'kN.m'
        assert registry.exists('M_k')
    
    def test_register_equation(self):
        """Testa registro de equação."""
        registry = VariableRegistry()
        registry.register('M_k', 112.5, 'kN.m', 'Momento')
        registry.register('gamma_f', 1.4, '', 'Coeficiente')
        
        eq_ctx = registry.register_equation(
            'M_d', 'gamma_f * M_k', 157.5, 'kN.m', ['M_k', 'gamma_f']
        )
        
        assert eq_ctx.name == 'M_d'
        assert eq_ctx.result_value == 157.5
        assert registry.exists('M_d')
    
    def test_list_variables(self):
        """Testa listagem de variáveis."""
        registry = VariableRegistry()
        registry.register('M_k', 112.5, 'kN.m', 'Momento')
        registry.register('gamma_f', 1.4, '', 'Coeficiente')
        
        vars_list = registry.list_variables()
        assert 'M_k' in vars_list
        assert 'gamma_f' in vars_list
        assert len(vars_list) == 2


class TestSmartTextProcessor:
    """Testa SmartTextProcessor."""
    
    def test_value_display(self):
        """Testa {var} - exibição de valor."""
        processor = SmartTextProcessor()
        processor.define_variables({'M_k': (112.5, 'kN.m', 'Momento')})
        
        text = "O momento é {M_k}"
        result = processor.process(text)
        
        assert '112.5 kN.m' in result
    
    def test_formula_display(self):
        """Testa {{var}} - exibição de fórmula."""
        processor = SmartTextProcessor()
        processor.define_variables({
            'M_k': (112.5, 'kN.m', 'Momento'),
            'gamma_f': (1.4, '', 'Coeficiente'),
        })
        
        # Primeiro calcular M_d
        processor.process("@eq M_d = gamma_f * M_k")
        
        # Agora exibir fórmula
        text = "Calculado por {{M_d}}"
        result = processor.process(text)
        
        assert 'gamma_f' in result or 'M_k' in result
    
    def test_equation_processing(self):
        """Testa @eq - processamento de equação."""
        processor = SmartTextProcessor()
        processor.define_variables({
            'M_k': (112.5, 'kN.m', 'Momento'),
            'gamma_f': (1.4, '', 'Coeficiente'),
        })
        
        text = "@eq M_d = gamma_f * M_k"
        result = processor.process(text)
        
        # Deve conter renderização LaTeX
        assert '$$' in result or '$' in result
        # Deve conter resultado
        assert '157' in result or '158' in result  # ~157.5
    
    def test_multiple_equations(self):
        """Testa múltiplas equações."""
        processor = SmartTextProcessor()
        processor.define_variables({
            'f_ck': (30, 'MPa', 'Resistência'),
            'gamma_c': (1.4, '', 'Coeficiente'),
        })
        
        text = """
        @eq f_cd = f_ck / gamma_c
        @eq f_cd_2 = 2 * f_cd
        """
        result = processor.process(text)
        
        # Deve ter ambas equações
        assert result.count('$$') >= 4  # 2 equações, 2x $$ cada
    
    def test_inline_variables(self):
        """Testa conversão inline de variáveis."""
        processor = SmartTextProcessor()
        processor.define_variables({'M_k': (112.5, 'kN.m', 'Momento')})
        
        text = "O momento M_k é importante"
        result = processor.process(text)
        
        # Deve converter M_k para LaTeX
        assert '$M_{k}$' in result or '$M_k$' in result


class TestTextProcessingBridge:
    """Testa TextProcessingBridge (integração)."""
    
    def test_create_bridge(self):
        """Testa criação do bridge."""
        bridge = create_text_bridge(document_type='memorial')
        assert bridge is not None
        assert isinstance(bridge, TextProcessingBridge)
    
    def test_define_variables(self):
        """Testa definição de variáveis."""
        bridge = create_text_bridge()
        bridge.define_variables({'M_k': (112.5, 'kN.m', 'Momento')})
        
        value = bridge.get_variable_value('M_k')
        assert value == 112.5
    
    def test_process_text(self):
        """Testa processamento de texto."""
        bridge = create_text_bridge()
        bridge.define_variables({'M_k': (112.5, 'kN.m', 'Momento')})
        
        text = "Momento: {M_k}"
        result = bridge.process(text)
        
        assert '112.5' in result
    
    def test_list_variables(self):
        """Testa listagem de variáveis."""
        bridge = create_text_bridge()
        bridge.define_variables({
            'M_k': (112.5, 'kN.m', 'Momento'),
            'gamma_f': (1.4, '', 'Coeficiente'),
        })
        
        vars_dict = bridge.list_variables()
        assert 'M_k' in vars_dict
        assert vars_dict['M_k']['value'] == 112.5
    
    def test_format_value(self):
        """Testa formatação de valores."""
        bridge = create_text_bridge()
        
        formatted = bridge.format_value(112.567, precision=2, unit='kN.m')
        assert '112.57' in formatted
        assert 'kN.m' in formatted


class TestCompatibility:
    """Testa compatibilidade v2.0."""
    
    def test_v2_still_works(self):
        """Testa que v2.0 SmartTextEngine ainda funciona."""
        from pymemorial.recognition import SmartTextEngine
        
        engine = SmartTextEngine()
        text = "M_k = 150 kN"
        result = engine.process_text(text, {'M_k': 150})
        
        assert result is not None


class TestEdgeCases:
    """Testa casos extremos."""
    
    def test_undefined_variable(self):
        """Testa variável não definida."""
        processor = SmartTextProcessor()
        text = "Valor: {undefined_var}"
        result = processor.process(text)
        
        # Deve manter placeholder ou retornar erro gracefully
        assert result is not None
    
    def test_invalid_equation(self):
        """Testa equação inválida."""
        processor = SmartTextProcessor()
        text = "@eq invalid syntax here"
        result = processor.process(text)
        
        # Deve retornar erro ou mensagem
        assert 'ERRO' in result or result != ""
    
    def test_empty_text(self):
        """Testa texto vazio."""
        processor = SmartTextProcessor()
        result = processor.process("")
        assert result == ""


def test_full_integration():
    """Teste de integração completo."""
    bridge = create_text_bridge(document_type='memorial')
    
    # Definir variáveis
    bridge.define_variables({
        'M_k': (112.5, 'kN.m', 'Momento característico'),
        'gamma_f': (1.4, '', 'Coeficiente de majoração'),
    })
    
    # Texto completo
    text = """
    ## Cálculo do Momento
    
    O momento M_k = {M_k} é majorado por gamma_f = {gamma_f}.
    
    Cálculo:
    @eq M_d = gamma_f * M_k
    
    Resultado: M_d = {M_d}
    """
    
    result = bridge.process(text)
    
    # Verificações
    assert '112.5' in result  # Valor M_k
    assert '1.4' in result or '1.40' in result  # Valor gamma_f
    assert '157' in result or '158' in result  # Resultado ~157.5
    assert '$$' in result  # LaTeX math mode


if __name__ == "__main__":
    pytest.main([__file__, '-v', '--tb=short'])
