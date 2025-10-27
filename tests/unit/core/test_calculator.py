# tests/unit/core/test_calculator.py
def test_calculator_basic():
    """Test basic equation evaluation."""
    calc = Calculator()
    M_k = Variable('M_k', value=150.0)
    gamma_s = Variable('gamma_s', value=1.4)
    eq = Equation('M_d = M_k * gamma_s', variables={'M_k': M_k, 'gamma_s': gamma_s})
    calc.add_equation(eq)
    results = calc.evaluate_all()
    assert results[id(eq)] == 210.0

def test_calculator_cache():
    """Test cache statistics."""
    calc = Calculator(max_cache=2)
    # ... add 3 equations ...
    assert calc.cache_stats['evictions'] == 1

def test_calculator_safe_eval():
    """Test AST safe evaluation."""
    calc = Calculator()
    with pytest.raises(ValueError, match="Unsafe operation"):
        calc._safe_ast_eval("__import__('os').system('ls')", {})
