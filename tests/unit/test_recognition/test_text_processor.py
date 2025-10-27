"""
Tests unitários para text_processor.py (MVP: Compat, Safe Eval, Whitelist, Lazy, Natural).
"""

import pytest
import time
import re
from unittest.mock import MagicMock

# Correct import (fix 'your_lib' → 'pymemorial')
from pymemorial.recognition.text_processor import (
    SmartTextEngine, PLACEHOLDER, DetectedVar, EngineeringNLP
)
# Stub if no pint (for context)
try:
    from pint import ureg
except ImportError:
    ureg = MagicMock(return_value=150)  # Stub for test


class TestSmartTextEngine:
    def test_legacy_render(self):
        """Test legacy {{var}} compat."""
        engine = SmartTextEngine(enable_latex=False)
        result = engine.render("Valor {{M_k}} = 150", {'M_k': 150})
        assert result == "Valor 150 = 150", "Legacy {{ }} pass"

    def test_safe_eval(self):
        """Test AST safe math (no injection)."""
        engine = SmartTextEngine()
        assert engine._safe_eval_expr("M_k * gamma_s", {'M_k': 150, 'gamma_s': 1.4}) == 210.0, "Safe math pass"
        with pytest.raises(ValueError):
            engine._safe_eval_expr("__import__('os').system('rm')", {}), "Injection blocked pass"

    def test_whitelist_regex(self):
        """Test whitelist (full vars, ignores 'para')."""
        engine = SmartTextEngine()
        text = "O momento M_k para o pilar é 150 kN, via gamma_s."
        detected = [m.group(1).lower() for m in engine.var_pattern.finditer(text)]
        assert detected == ['m_k', 'gamma_s'], "Whitelist pass (ignores 'para')"

    def test_lazy_nlp_off(self):
        """Test lazy init fast (no NLTK overhead)."""
        start = time.time()
        engine = SmartTextEngine(enable_nlp=False)
        assert time.time() - start < 0.01, "Lazy init fast pass"

    def test_natural_process(self):
        """Test natural text (~expr~ eval + LaTeX stub)."""
        engine = SmartTextEngine(enable_latex=True)
        text = "M_k = 150 kN majorado por gamma_s=1.4 via ~M_d = M_k * gamma_s~."
        context = {'M_k': 150, 'gamma_s': 1.4}
        processed = engine.process_natural_text(text, context)
        assert '210' in processed, "Natural eval/LaTeX pass"


if __name__ == "__main__":
    pytest.main(['-v', __file__])