# tests/unit/test_editor/test_smart_textblock.py
"""
Testes para SmartTextBlock - PyMemorial v2.0
"""

import pytest
from pymemorial.editor.smart_textblock import (
    SmartTextBlock,
    ReferenceStyle,
    TemplateType,
    ValidationResult,
    RenderMetadata
)

class TestSmartTextBlock:
    """Testes para SmartTextBlock."""
    
    def test_init_basic(self):
        """Testa inicialização básica."""
        block = SmartTextBlock(
            text="Valor: {x:.2f} m",
            variables={'x': 3.14159}
        )
        
        assert block.text == "Valor: {x:.2f} m"
        assert block.local_variables == {'x': 3.14159}
        assert len(block.temp_symbols) == 0
    
    def test_init_with_temp_symbols_list(self):
        """Testa inicialização com temp_symbols como lista."""
        block = SmartTextBlock(
            text="Coeficiente k_{md}",
            temp_symbols=['k_{md}', 'gamma_{f}']
        )
        
        assert 'k_md' in block.temp_symbols
        assert 'gamma_f' in block.temp_symbols
    
    def test_init_with_temp_symbols_dict(self):
        """Testa inicialização com temp_symbols como dict."""
        block = SmartTextBlock(
            text="Coeficiente k_{md}",
            temp_symbols={'k_md': 'k_{md}'}
        )
        
        assert block.temp_symbols == {'k_md': 'k_{md}'}
    
    def test_extract_symbol_key(self):
        """Testa extração de chave de símbolo LaTeX."""
        assert SmartTextBlock._extract_symbol_key('k_{md}') == 'k_md'
        assert SmartTextBlock._extract_symbol_key(r'\lambda_{lim}') == 'lambda_lim'
        assert SmartTextBlock._extract_symbol_key('gamma_f') == 'gamma_f'
    
    def test_merge_contexts_local_priority(self):
        """Testa prioridade local > global na mesclagem."""
        block = SmartTextBlock(
            text="Test",
            variables={'x': 10, 'y': 20}
        )
        
        global_context = {'x': 5, 'z': 30}
        merged = block._merge_contexts(global_context)
        
        assert merged['x'] == 10  # Local sobrescreve global
        assert merged['y'] == 20  # Local apenas
        assert merged['z'] == 30  # Global apenas
    
    def test_validate_success(self):
        """Testa validação com todas variáveis presentes."""
        block = SmartTextBlock(
            text="Valores: {x:.2f} e {y:.2f}",
            variables={'x': 1.5, 'y': 2.5}
        )
        
        result = block.validate()
        
        assert result.valid is True
        assert len(result.missing_vars) == 0
        assert result.total_vars == 2
    
    def test_validate_missing_vars(self):
        """Testa validação com variáveis faltantes."""
        block = SmartTextBlock(
            text="Valores: {x:.2f} e {y:.2f}",
            variables={'x': 1.5}
        )
        
        result = block.validate()
        
        assert result.valid is False
        assert 'y' in result.missing_vars
        assert result.total_vars == 2
    
    def test_cache_clear(self):
        """Testa limpeza de cache."""
        SmartTextBlock._render_cache = {'test_key': ('html', None)}
        
        SmartTextBlock.clear_cache()
        
        assert len(SmartTextBlock._render_cache) == 0
    
    def test_cache_stats(self):
        """Testa estatísticas de cache."""
        SmartTextBlock.clear_cache()
        SmartTextBlock._render_cache = {'key1': ('html1', None), 'key2': ('html2', None)}
        
        stats = SmartTextBlock.get_cache_stats()
        
        assert stats['size'] == 2
        assert stats['enabled'] is True
    
    def test_repr(self):
        """Testa representação string."""
        block = SmartTextBlock(
            text="Test block with some text",
            variables={'x': 1}
        )
        
        repr_str = repr(block)
        
        assert 'SmartTextBlock' in repr_str
        assert 'vars=1' in repr_str
    
    def test_fallback_render(self):
        """Testa renderização de fallback."""
        block = SmartTextBlock(text="Test")
        
        html = block._fallback_render("Test error")
        
        assert 'smart-textblock-error' in html
        assert 'Test error' in html
        assert 'Test' in html

class TestReferenceStyle:
    """Testes para ReferenceStyle enum."""
    
    def test_enum_values(self):
        """Testa valores do enum."""
        assert ReferenceStyle.INLINE.value == "inline"
        assert ReferenceStyle.FOOTNOTE.value == "footnote"
        assert ReferenceStyle.SIDEBAR.value == "sidebar"
        assert ReferenceStyle.ABNT.value == "abnt"

class TestValidationResult:
    """Testes para ValidationResult."""
    
    def test_to_dict(self):
        """Testa conversão para dicionário."""
        result = ValidationResult(
            valid=True,
            missing_vars=['x', 'y'],
            total_vars=5
        )
        
        dict_result = result.to_dict()
        
        assert dict_result['valid'] is True
        assert dict_result['missing_vars'] == ['x', 'y']
        assert dict_result['total_vars'] == 5
        assert 'success_rate' in dict_result

class TestRenderMetadata:
    """Testes para RenderMetadata."""
    
    def test_to_dict(self):
        """Testa conversão para dicionário."""
        metadata = RenderMetadata(
            render_time_ms=15.5,
            text_length=100,
            output_length=150,
            vars_resolved=5,
            vars_failed=2,
            cache_hit=False
        )
        
        dict_metadata = metadata.to_dict()
        
        assert dict_metadata['render_time_ms'] == 15.5
        assert dict_metadata['text_length'] == 100
        assert dict_metadata['cache_hit'] is False
