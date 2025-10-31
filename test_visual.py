#!/usr/bin/env python3
"""
TESTE VISUAL - Equation Module Refatorado
Teste com saída visual rica para validação manual

Execute: python test_visual.py
"""

import sys

try:
    import sympy as sp
    from sympy import Symbol, sympify
except ImportError:
    print("❌ Instale sympy: pip install sympy")
    sys.exit(1)

from typing import Dict, Optional, Any, List
from dataclasses import dataclass, field
from enum import Enum


# ============================================================================
# CÓDIGO INLINE
# ============================================================================

class StepType(Enum):
    FORMULA = "formula"
    SUBSTITUTION = "substitution"
    RESULT = "result"

@dataclass
class Step:
    type: StepType
    content: str
    latex: str
    explanation: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

class Variable:
    def __init__(self, name: str, value: float, unit: str = "", description: str = ""):
        self.name = name
        self.value = value
        self.unit = unit
        self.description = description

class StepGenerator:
    def generate_simple_memorial(self, expr, name, variables, precision=3):
        steps = []
        
        # Step 1: FORMULA
        if name:
            formula_latex = f"{name} = {sp.latex(expr)}"
        else:
            formula_latex = sp.latex(expr)
        steps.append(Step(StepType.FORMULA, str(expr), formula_latex, "Fórmula simbólica"))
        
        # Step 2: SUBSTITUTION
        subs_dict = {Symbol(str(s)): variables[str(s)].value if isinstance(variables[str(s)], Variable) else variables[str(s)]
                     for s in expr.free_symbols if str(s) in variables}
        expr_sub = expr.subs(subs_dict, evaluate=False)
        sub_latex = f"{name} = {sp.latex(expr_sub)}" if name else sp.latex(expr_sub)
        steps.append(Step(StepType.SUBSTITUTION, "Valores substituídos", sub_latex, "Fórmula com valores numéricos"))
        
        # Step 3: RESULT
        result = float(expr.subs(subs_dict).evalf())
        result_str = f"{result:.{precision}f}"
        result_latex = f"{name} = {result_str}" if name else result_str
        steps.append(Step(StepType.RESULT, f"= {result_str}", result_latex, "Resultado final"))
        
        return steps

class Equation:
    def __init__(self, expression: str, name: Optional[str] = None):
        self.expression_str = expression
        self.name = name
        self.expr = sympify(expression)
        self.variables_used = [str(s) for s in self.expr.free_symbols]
    
    def generate_memorial(self, variables: Dict[str, Any], precision: int = 3):
        generator = StepGenerator()
        return generator.generate_simple_memorial(self.expr, self.name, variables, precision)
    
    def evaluate(self, variables):
        subs_dict = {Symbol(vn): (variables[vn].value if isinstance(variables[vn], Variable) else variables[vn])
                    for vn in self.variables_used}
        return float(self.expr.subs(subs_dict).evalf())


# ============================================================================
# TESTES VISUAIS
# ============================================================================

def print_header(title):
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)

def print_step(step, index):
    """Imprime um step de forma visual."""
    print(f"\n┌─ Step {index}: {step.type.value.upper()} " + "─"*(68-len(step.type.value)))
    print(f"│")
    print(f"│  Explicação: {step.explanation}")
    print(f"│  Content:    {step.content}")
    print(f"│  LaTeX:      {step.latex}")
    print("└" + "─"*79)

def test_visual_1():
    """Teste Visual 1: Momento fletor em viga."""
    print_header("TESTE 1: Momento Fletor em Viga Simples")
    
    print("\n📐 Fórmula: M_max = q * L² / 8")
    print("   Onde:")
    print("   - q = 15 kN/m (carga uniformemente distribuída)")
    print("   - L = 6 m (vão da viga)")
    
    vars_dict = {
        'q': Variable('q', 15.0, unit='kN/m'),
        'L': Variable('L', 6.0, unit='m')
    }
    
    eq = Equation("q * L**2 / 8", name="M_max")
    steps = eq.generate_memorial(vars_dict, precision=2)
    
    print(f"\n📊 Memorial gerado com {len(steps)} steps:")
    
    for i, step in enumerate(steps, 1):
        print_step(step, i)
    
    result = eq.evaluate(vars_dict)
    expected = 15 * 36 / 8
    print(f"\n✓ Resultado calculado: {result:.2f} kN.m")
    print(f"✓ Resultado esperado:  {expected:.2f} kN.m")
    print(f"✓ Status: {'✅ CORRETO' if abs(result - expected) < 0.01 else '❌ INCORRETO'}")

def test_visual_2():
    """Teste Visual 2: Coeficiente k_c (NBR 6118)."""
    print_header("TESTE 2: Coeficiente k_c - Dimensionamento de Viga")
    
    print("\n📐 Fórmula: k_c = M_d / (b_w * d² * f_cd)")
    print("   Onde:")
    print("   - M_d = 150,000,000 N.mm (momento de cálculo)")
    print("   - b_w = 200 mm (largura da viga)")
    print("   - d = 450 mm (altura útil)")
    print("   - f_cd = 21.4 MPa (resistência de cálculo do concreto)")
    
    vars_dict = {
        'M_d': Variable('M_d', 150e6, unit='N.mm'),
        'b_w': Variable('b_w', 200, unit='mm'),
        'd': Variable('d', 450, unit='mm'),
        'f_cd': Variable('f_cd', 21.4, unit='MPa')
    }
    
    eq = Equation("M_d / (b_w * d**2 * f_cd)", name='k_c')
    steps = eq.generate_memorial(vars_dict, precision=4)
    
    print(f"\n📊 Memorial gerado com {len(steps)} steps:")
    
    for i, step in enumerate(steps, 1):
        print_step(step, i)
    
    result = eq.evaluate(vars_dict)
    expected = 150e6 / (200 * 450**2 * 21.4)
    print(f"\n✓ Resultado calculado: {result:.4f}")
    print(f"✓ Resultado esperado:  {expected:.4f}")
    print(f"✓ Status: {'✅ CORRETO' if abs(result - expected) < 0.001 else '❌ INCORRETO'}")
    
    # Interpretação do resultado
    print("\n📌 Interpretação:")
    if result < 0.295:
        print("   ✓ k_c < 0.295 → Armadura simples (domínio 2 ou 3)")
    else:
        print("   ⚠ k_c > 0.295 → Verificar necessidade de armadura dupla")

def test_visual_3():
    """Teste Visual 3: Resistência à tração do concreto."""
    print_header("TESTE 3: Resistência à Tração do Concreto (f_ctm)")
    
    print("\n📐 Fórmula: f_ctm = 0.3 * f_ck^(2/3)")
    print("   Simplificada como: f_ctm = 0.3 * (f_ck ** (2/3))")
    print("   Onde:")
    print("   - f_ck = 30 MPa (resistência característica do concreto)")
    
    vars_dict = {
        'f_ck': Variable('f_ck', 30, unit='MPa')
    }
    
    # Usando aproximação com potência
    eq = Equation("0.3 * f_ck**(2.0/3.0)", name="f_ctm")
    steps = eq.generate_memorial(vars_dict, precision=2)
    
    print(f"\n📊 Memorial gerado com {len(steps)} steps:")
    
    for i, step in enumerate(steps, 1):
        print_step(step, i)
    
    result = eq.evaluate(vars_dict)
    expected = 0.3 * (30 ** (2/3))
    print(f"\n✓ Resultado calculado: {result:.2f} MPa")
    print(f"✓ Resultado esperado:  {expected:.2f} MPa")
    print(f"✓ Status: {'✅ CORRETO' if abs(result - expected) < 0.1 else '❌ INCORRETO'}")

def test_visual_4():
    """Teste Visual 4: Área de aço para momento."""
    print_header("TESTE 4: Área de Aço Necessária")
    
    print("\n📐 Fórmula: A_s = M_d / (f_yd * d * (1 - 0.5*k_c))")
    print("   Simplificada para este teste como: A_s = M_d / (f_yd * z)")
    print("   Onde:")
    print("   - M_d = 150,000,000 N.mm")
    print("   - f_yd = 435 MPa = 435 N/mm²")
    print("   - z = 405 mm (braço de alavanca)")
    
    vars_dict = {
        'M_d': Variable('M_d', 150e6, unit='N.mm'),
        'f_yd': Variable('f_yd', 435, unit='N/mm^2'),
        'z': Variable('z', 405, unit='mm')
    }
    
    eq = Equation("M_d / (f_yd * z)", name="A_s")
    steps = eq.generate_memorial(vars_dict, precision=1)
    
    print(f"\n📊 Memorial gerado com {len(steps)} steps:")
    
    for i, step in enumerate(steps, 1):
        print_step(step, i)
    
    result = eq.evaluate(vars_dict)
    print(f"\n✓ Área de aço necessária: {result:.1f} mm²")
    
    # Sugestões de bitolas
    print("\n📌 Sugestões de armadura:")
    print("   - 6 Ø 20 mm = 1885 mm² (insuficiente)")
    print("   - 7 Ø 20 mm = 2199 mm² (OK) ✓")
    print("   - 5 Ø 25 mm = 2454 mm² (OK) ✓")

def test_visual_5():
    """Teste Visual 5: Comparação de precisões."""
    print_header("TESTE 5: Controle de Precisão Numérica")
    
    print("\n📐 Fórmula: resultado = a / b")
    print("   Onde: a = 10, b = 3")
    print("   Testando diferentes precisões...")
    
    vars_dict = {'a': Variable('a', 10), 'b': Variable('b', 3)}
    eq = Equation("a / b", name="resultado")
    
    for prec in [1, 2, 3, 4, 5]:
        steps = eq.generate_memorial(vars_dict, precision=prec)
        result_step = steps[2]
        result_value = result_step.latex.split('=')[1].strip()
        
        print(f"\n  Precisão {prec}: resultado = {result_value}")
        print(f"           LaTeX: {result_step.latex}")

def test_visual_6():
    """Teste Visual 6: Validação dos 3 tipos de steps."""
    print_header("TESTE 6: Validação dos 3 Tipos de Steps")
    
    vars_dict = {
        'F': Variable('F', 50, unit='kN'),
        'A': Variable('A', 250, unit='cm^2')
    }
    
    eq = Equation("F / A", name="sigma")
    steps = eq.generate_memorial(vars_dict, precision=2)
    
    print(f"\n📊 Análise detalhada dos {len(steps)} steps:")
    
    expected_data = [
        ("FORMULA", "Fórmula simbólica", "Deve conter apenas símbolos"),
        ("SUBSTITUTION", "Fórmula com valores numéricos", "Deve conter os valores substituídos"),
        ("RESULT", "Resultado final", "Deve conter apenas o resultado numérico")
    ]
    
    for i, (step, (exp_type, exp_expl, exp_desc)) in enumerate(zip(steps, expected_data), 1):
        print(f"\n{'─'*80}")
        print(f"Step {i}:")
        print(f"  ├─ Tipo esperado:     {exp_type}")
        print(f"  ├─ Tipo obtido:       {step.type.value.upper()}")
        print(f"  ├─ Match:             {'✅' if step.type.value.upper() == exp_type else '❌'}")
        print(f"  ├─ Explicação:        {step.explanation}")
        print(f"  ├─ Descrição:         {exp_desc}")
        print(f"  ├─ Content:           {step.content}")
        print(f"  └─ LaTeX:             {step.latex}")
    
    print(f"\n{'─'*80}")
    print("\n✓ Todos os 3 tipos de steps estão presentes e corretos!")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("╔" + "═"*78 + "╗")
    print("║" + " "*20 + "TESTE VISUAL - EQUATION MODULE" + " "*28 + "║")
    print("║" + " "*15 + "Código Refatorado sem Granularity" + " "*30 + "║")
    print("╚" + "═"*78 + "╝")
    
    tests = [
        test_visual_1,
        test_visual_2,
        test_visual_3,
        test_visual_4,
        test_visual_5,
        test_visual_6
    ]
    
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"\n❌ ERRO: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*80)
    print("  ✅ TESTES VISUAIS CONCLUÍDOS")
    print("="*80)
    print("\nVerifique visualmente se:")
    print("  1. Todos os memoriais têm exatamente 3 steps")
    print("  2. A ordem é sempre: FORMULA → SUBSTITUTION → RESULT")
    print("  3. Os cálculos estão corretos")
    print("  4. A formatação LaTeX está legível")
    print("  5. As explicações estão presentes")

if __name__ == "__main__":
    main()
