# tests/unit/test_recognition/test_ast_parser.py

import pytest
from pymemorial.recognition.ast_parser import PyMemorialASTParser

def test_parse_simple_assignment():
    p = PyMemorialASTParser()
    a = p.parse_assignment("M = 100")
    assert a.lhs == "M"
    assert a.rhs_symbolic == "100"
    assert a.comment is None

def test_parse_arithmetic_expression():
    p = PyMemorialASTParser()
    a = p.parse_assignment("M = q * L**2 / 8", context={'q': 15, 'L': 6})
    assert a.lhs == "M"
    assert a.rhs_symbolic.replace(" ", "") == "q*L**2/8"

def test_extract_variables():
    p = PyMemorialASTParser()
    vars_ = p.extract_variables("q * L**2 / 8")
    assert set(vars_) == {"q", "L"}

def test_comment_capture_inline():
    p = PyMemorialASTParser()
    a = p.parse_assignment("L = 6.0  # m - Vão")
    assert a.comment == "m - Vão"

def test_to_latex_power_and_frac():
    p = PyMemorialASTParser()
    latex = p.to_latex("q * L**2 / 8")
    assert r"\frac{q \cdot L^{2}}{8}" in latex

def test_to_latex_greek_subscript():
    p = PyMemorialASTParser()
    # gamma_s → \gamma_{s} conforme SymbolsConfig(greek_style='latex')
    latex = p.to_latex("gamma_s")
    assert r"\gamma_{s}" == latex

def test_parse_block_preserves_order_and_lines():
    code = '''
q = 15.0   # kN/m
L = 6.0    # m
M = q * L**2 / 8  # kN.m
'''.strip()
    p = PyMemorialASTParser()
    assigns = p.parse_code_block(code)
    assert [a.lhs for a in assigns] == ["q", "L", "M"]
    assert assigns[2].comment.startswith("kN.m")
