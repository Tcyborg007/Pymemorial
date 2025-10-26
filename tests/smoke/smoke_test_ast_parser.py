# tests/smoke/smoke_test_ast_parser.py
from pymemorial.recognition.ast_parser import PyMemorialASTParser

code = """
gamma_f = 1.4  # fator
M_k = 150      # kN.m
M_d = gamma_f * M_k
""".strip()

p = PyMemorialASTParser()
assigns = p.parse_code_block(code)

for a in assigns:
    print(f"{a.lhs} = {a.rhs_symbolic}  # {a.comment or ''}")
    print("LaTeX:", p.to_latex(a.rhs_symbolic))
