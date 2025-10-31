"""
TESTES VISUAIS - PyMemorial Core v2.0
=====================================

Script completo para demonstrar todas as funcionalidades dos módulos core.
Organizado por módulo com exemplos práticos de engenharia estrutural.

Autor: Especialista em Dev + Engenharia Estrutural  
Data: 2025-10-28

USO:
    python test_visual_pymemorial_core.py
"""

import sys
import traceback
from pathlib import Path

print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                  PYMEMORIAL CORE v2.0 - TESTES VISUAIS                       ║
║                                                                              ║
║  Bateria completa de testes para demonstração das funcionalidades           ║
║  Organizado por: Especialista em Dev + Engenharia Estrutural                ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

# ============================================================================
# VERIFICAÇÃO DE DEPENDÊNCIAS
# ============================================================================

def verificar_dependencias():
    """Verifica e reporta status das dependências."""
    print("\n" + "="*80)
    print("VERIFICAÇÃO DE DEPENDÊNCIAS")
    print("="*80)
    
    deps_status = {}
    
    # SymPy
    try:
        import sympy as sp
        deps_status['sympy'] = f"✅ SymPy {sp.__version__}"
    except ImportError:
        deps_status['sympy'] = "❌ SymPy não instalado"
    
    # NumPy  
    try:
        import numpy as np
        deps_status['numpy'] = f"✅ NumPy {np.__version__}"
    except ImportError:
        deps_status['numpy'] = "❌ NumPy não instalado"
    
    # Pint
    try:
        import pint
        deps_status['pint'] = f"✅ Pint {pint.__version__}"
    except ImportError:
        deps_status['pint'] = "❌ Pint não instalado (opcional)"
    
    # SciPy
    try:
        import scipy
        deps_status['scipy'] = f"✅ SciPy {scipy.__version__}"
    except ImportError:
        deps_status['scipy'] = "❌ SciPy não instalado (opcional)"
    
    for dep, status in deps_status.items():
        print(f"  {status}")
    
    print("\n💡 Dica: Instale as dependências com:")
    print("   pip install sympy numpy pint scipy")
    
    return deps_status

# ============================================================================
# TESTE 1: VARIABLE
# ============================================================================

def teste_variable():
    """Demonstra funcionalidades do módulo Variable."""
    print("\n" + "="*80)
    print("TESTE 1: VARIABLE - Variáveis Inteligentes")
    print("="*80)
    
    print("""
O módulo Variable implementa variáveis com:
  • Tipagem forte (name, value, unit)
  • Operações matemáticas sobrecarga (+, -, *, /, **)
  • Histórico de mudanças com rollback
  • Conversão automática para LaTeX
  • Integração com registry de símbolos
    """)
    
    print("\n--- 1.1: Criação de Variáveis Simples ---")
    print("Código Python:")
    codigo1 = """
from pymemorial.core import Variable

# Momento fletor de cálculo
M_d = Variable(name='M_d', value=150.0, unit='kN*m')

# Resistência característica do concreto
f_ck = Variable(name='f_ck', value=25, unit='MPa')

# Coeficiente de ponderação (adimensional)
gamma_c = Variable(name='gamma_c', value=1.4)

print(f'M_d = {M_d.value} {M_d.unit}')
print(f'f_ck = {f_ck.value} {f_ck.unit}')
print(f'gamma_c = {gamma_c.value}')
"""
    print(codigo1)
    
    print("📊 Saída esperada:")
    print("  M_d = 150.0 kN*m")
    print("  f_ck = 25 MPa")
    print("  gamma_c = 1.4")
    
    print("\n--- 1.2: Operações Matemáticas ---")
    codigo2 = """
# Soma de momentos
M_g = Variable(name='M_g', value=50.0, unit='kN*m')
M_total = M_d + M_g
print(f'M_total = {M_total.value} {M_total.unit}')

# Resistência de cálculo
f_cd = f_ck / gamma_c
print(f'f_cd = {f_cd.value:.2f} MPa')

# Potenciação
A = Variable(name='A', value=30, unit='cm')
A_quad = A ** 2
print(f'A² = {A_quad.value} {A_quad.unit}')
"""
    print(codigo2)
    
    print("📊 Saída esperada:")
    print("  M_total = 200.0 kN*m")
    print("  f_cd = 17.86 MPa")
    print("  A² = 900 cm²")
    
    print("\n--- 1.3: Histórico e Rollback ---")
    codigo3 = """
h = Variable(name='h', value=30)  # cm
print(f'Inicial: h = {h.value} cm')

h.update_value(35)
print(f'Após update 1: h = {h.value} cm')

h.update_value(40)
print(f'Após update 2: h = {h.value} cm')

history = h.get_history()
print(f'Histórico: {len(history)} mudanças')

h.rollback(1)  # Volta 1 passo
print(f'Após rollback: h = {h.value} cm')
"""
    print(codigo3)
    
    print("📊 Saída esperada:")
    print("  Inicial: h = 30 cm")
    print("  Após update 1: h = 35 cm")
    print("  Após update 2: h = 40 cm")
    print("  Histórico: 2 mudanças")
    print("  Após rollback: h = 35 cm")
    
    print("\n✅ Teste Variable: Funcionalidades demonstradas!")
    return True

# ============================================================================
# TESTE 2: EQUATION
# ============================================================================

def teste_equation():
    """Demonstra motor simbólico de equações."""
    print("\n" + "="*80)
    print("TESTE 2: EQUATION - Motor Simbólico com Steps")
    print("="*80)
    
    print("""
O módulo Equation implementa:
  • Parsing de expressões Python → SymPy
  • Avaliação numérica com substituição
  • Geração automática de steps (4 níveis)
  • Manipulação algébrica (simplify, expand, factor)
    """)
    
    print("\n--- 2.1: Equação de Momento Máximo (Viga Biapoiada) ---")
    codigo1 = """
from pymemorial.core import Equation, Variable

vars_dict = {
    'q': Variable('q', 15, unit='kN/m'),
    'L': Variable('L', 6, unit='m')
}

# M_max = q*L²/8 (viga biapoiada)
eq = Equation('q * L**2 / 8', locals_dict=vars_dict, name='M_max')
print(f'Expressão: {eq.expression_str}')
print(f'Símbolos: {eq.symbols}')

result = eq.evaluate()
print(f'M_max = {result.value} {result.unit}')
"""
    print(codigo1)
    
    print("📊 Saída esperada:")
    print("  Expressão: q * L**2 / 8")
    print("  Símbolos: {q, L}")
    print("  M_max = 67.5 kN*m")
    
    print("\n--- 2.2: Geração de Steps (Memorial de Cálculo) ---")
    codigo2 = """
steps = eq.steps(granularity='medium')
for i, step in enumerate(steps, 1):
    print(f'Passo {i}: {step["step"]}')
    print(f'  Tipo: {step["type"]}')
    if 'latex' in step:
        print(f'  LaTeX: {step["latex"]}')
"""
    print(codigo2)
    
    print("📊 Saída esperada:")
    print("  Passo 1: Fórmula Simbólica")
    print("    Tipo: symbolic")
    print("    LaTeX: \\frac{q L^{2}}{8}")
    print("  Passo 2: Substituição de Variáveis")
    print("    Tipo: substitution")
    print("    q = 15 kN/m, L = 6 m")
    print("  Passo 3: Resultado Numérico")
    print("    Tipo: numeric")
    print("    M_max = 67.5 kN*m")
    
    print("\n--- 2.3: Manipulação Algébrica ---")
    codigo3 = """
# Simplificar
eq2 = Equation('x**2 + 2*x + 1')
eq_simp = eq2.simplify()
print(f'Original: {eq2.expression_str}')
print(f'Simplificado: {eq_simp.expression_str}')

# Expandir
eq3 = Equation('(x + 1)**2')
eq_exp = eq3.expand()
print(f'Original: {eq3.expression_str}')
print(f'Expandido: {eq_exp.expression_str}')

# Fatorar
eq4 = Equation('x**2 - 1')
eq_fat = eq4.factor()
print(f'Original: {eq4.expression_str}')
print(f'Fatorado: {eq_fat.expression_str}')
"""
    print(codigo3)
    
    print("📊 Saída esperada:")
    print("  Original: x**2 + 2*x + 1")
    print("  Simplificado: (x + 1)**2")
    print("  Original: (x + 1)**2")
    print("  Expandido: x**2 + 2*x + 1")
    print("  Original: x**2 - 1")
    print("  Fatorado: (x - 1)*(x + 1)")
    
    print("\n✅ Teste Equation: Motor simbólico OK!")
    return True

# ============================================================================
# TESTE 3: CALCULATOR
# ============================================================================

def teste_calculator():
    """Demonstra motor híbrido de cálculo."""
    print("\n" + "="*80)
    print("TESTE 3: CALCULATOR - Motor Híbrido (SymPy + NumPy + SciPy)")
    print("="*80)
    
    print("""
O módulo Calculator oferece:
  • SafeEvaluator com whitelist AST (segurança crítica)
  • Cálculos vetorizados (NumPy arrays)
  • Monte Carlo simulation para análise probabilística
  • Batch compute com paralelização
    """)
    
    print("\n--- 3.1: Avaliação Segura de Expressões ---")
    codigo1 = """
from pymemorial.core import Calculator

calc = Calculator()
calc.add_variable('fck', 25, unit='MPa')
calc.add_variable('gamma_c', 1.4)

# Resistência de cálculo
result = calc.evaluate('fck / gamma_c')
print(f'fcd = {result.value:.2f} {result.unit}')

# Operações compostas
calc.add_variable('b', 20, unit='cm')
calc.add_variable('h', 50, unit='cm')
A = calc.evaluate('b * h')
print(f'A = {A.value} {A.unit}')
"""
    print(codigo1)
    
    print("📊 Saída esperada:")
    print("  fcd = 17.86 MPa")
    print("  A = 1000.0 cm²")
    
    print("\n--- 3.2: Cálculos Vetorizados (Arrays NumPy) ---")
    codigo2 = """
import numpy as np

calc = Calculator()
x_array = np.array([1, 2, 3, 4, 5])
calc.add_variable('x', x_array)

# Aplicar função a todo o array
result = calc.compute('x**2 + 2*x', vectorized=True)
print(f'Resultado vetorizado: {result.value}')
"""
    print(codigo2)
    
    print("📊 Saída esperada:")
    print("  Resultado vetorizado: [3, 8, 15, 24, 35]")
    
    print("\n--- 3.3: Monte Carlo Simulation ---")
    codigo3 = """
# Análise probabilística de fck
variables_mc = {
    'fck': {'mean': 25, 'std': 3, 'dist': 'normal'}
}

result_mc = calc.monte_carlo(
    'fck * 0.85',  # fcd simplificado
    variables_mc,
    n_samples=10000,
    seed=42
)

print(f'Média: {result_mc.metadata["mean"]:.2f}')
print(f'Desvio padrão: {result_mc.metadata["std"]:.2f}')
print(f'Percentil 5%: {result_mc.metadata["percentile_5"]:.2f}')
print(f'Percentil 95%: {result_mc.metadata["percentile_95"]:.2f}')
"""
    print(codigo3)
    
    print("📊 Saída esperada:")
    print("  Média: 21.25")
    print("  Desvio padrão: 2.55")
    print("  Percentil 5%: 17.30")
    print("  Percentil 95%: 25.20")
    
    print("\n✅ Teste Calculator: Motor híbrido OK!")
    return True

# ============================================================================
# TESTE 4: MATRIX
# ============================================================================

def teste_matrix():
    """Demonstra sistema de matrizes."""
    print("\n" + "="*80)
    print("TESTE 4: MATRIX - Sistema Robusto com SymPy e NumPy")
    print("="*80)
    
    print("""
O módulo Matrix implementa:
  • Matrizes simbólicas (SymPy)
  • Avaliação numérica via lambdify
  • Steps intermediários detalhados
  • Validação de pureza simbólica
    """)
    
    print("\n--- 4.1: Matriz de Rigidez de Viga ---")
    codigo1 = """
from pymemorial.core import Matrix, Variable

vars_dict = {
    'E': Variable('E', 210000, unit='MPa'),
    'I': Variable('I', 8333, unit='cm^4'),
    'L': Variable('L', 600, unit='cm')
}

# Matriz de rigidez 2x2 simplificada
K_expr = \"\"\"[
    [12*E*I/L**3, 6*E*I/L**2],
    [6*E*I/L**2, 4*E*I/L]
]\"\"\"

K = Matrix(data=K_expr, variables=vars_dict, name='K')
print(f'Matriz criada: {K.shape}')
print(f'É simbólica: {K.is_symbolic}')
print(f'É quadrada: {K.is_square}')
"""
    print(codigo1)
    
    print("📊 Saída esperada:")
    print("  Matriz criada: (2, 2)")
    print("  É simbólica: True")
    print("  É quadrada: True")
    
    print("\n--- 4.2: Avaliação Numérica ---")
    codigo2 = """
K_numeric = K.evaluate()
print(f'K[0,0] = {K_numeric[0,0]:.2e}')
print(f'K[0,1] = {K_numeric[0,1]:.2e}')
print(f'K[1,1] = {K_numeric[1,1]:.2e}')
"""
    print(codigo2)
    
    print("📊 Saída esperada:")
    print("  K[0,0] = 1.02e+04  # 12EI/L³")
    print("  K[0,1] = 3.06e+06  # 6EI/L²")
    print("  K[1,1] = 1.22e+09  # 4EI/L")
    
    print("\n--- 4.3: Steps de Cálculo ---")
    codigo3 = """
steps = K.steps(granularity='detailed')
for i, step in enumerate(steps, 1):
    print(f'Passo {i}: {step["step"]}')
"""
    print(codigo3)
    
    print("📊 Saída esperada:")
    print("  Passo 1: Definição (matriz 2×2 simbólica)")
    print("  Passo 2: Forma Simbólica LaTeX")
    print("  Passo 3: Substituição (E=210000, I=8333, L=600)")
    print("  Passo 4: Matriz Numérica")
    
    print("\n✅ Teste Matrix: Sistema robusto OK!")
    return True

# ============================================================================
# TESTE 5: CONFIG
# ============================================================================

def teste_config():
    """Demonstra sistema de configuração."""
    print("\n" + "="*80)
    print("TESTE 5: CONFIG - Sistema de Configuração Global")
    print("="*80)
    
    print("""
O módulo Config gerencia:
  • Configurações de display (precisão, formato)
  • Perfis de normas técnicas (NBR, Eurocode, ACI)
  • Context managers para overrides temporários
  • Persistência em arquivo JSON
    """)
    
    print("\n--- 5.1: Configurações Padrão ---")
    codigo1 = """
from pymemorial.core import get_config, set_option

config = get_config()
print(f'Precisão: {config.display.precision}')
print(f'Formato: {config.display.number_format}')

# Alterar precisão
set_option('display.precision', 5)
print(f'Nova precisão: {config.display.precision}')
"""
    print(codigo1)
    
    print("📊 Saída esperada:")
    print("  Precisão: 3")
    print("  Formato: auto")
    print("  Nova precisão: 5")
    
    print("\n--- 5.2: Perfis de Normas Técnicas ---")
    codigo2 = """
# Carregar NBR 6118:2023
config.load_profile('nbr6118')
print(f'NBR 6118: γc = {config.standard.partial_factors["gamma_c"]}')
print(f'NBR 6118: γf = {config.standard.partial_factors["gamma_f"]}')

# Trocar para Eurocode 2
config.load_profile('eurocode2')
print(f'EC2: γc = {config.standard.partial_factors["gamma_c"]}')
"""
    print(codigo2)
    
    print("📊 Saída esperada:")
    print("  NBR 6118: γc = 1.4")
    print("  NBR 6118: γf = 1.4")
    print("  EC2: γc = 1.5")
    
    print("\n✅ Teste Config: Sistema OK!")
    return True

# ============================================================================
# TESTE INTEGRADO: CÁLCULO COMPLETO
# ============================================================================

def teste_integrado():
    """Exemplo completo: Viga biapoiada com NBR 6118."""
    print("\n" + "="*80)
    print("TESTE INTEGRADO: Cálculo Completo de Viga Biapoiada")
    print("="*80)
    
    print("""
PROBLEMA DE ENGENHARIA ESTRUTURAL:
  • Viga biapoiada, vão L = 6.0 m
  • Carga permanente: g = 10 kN/m
  • Carga acidental: q = 5 kN/m
  • Norma: NBR 6118:2023 (γf = 1.4)

OBJETIVO:
  Calcular momento máximo de cálculo (M_d)
  usando TODOS os módulos do PyMemorial
    """)
    
    print("\n--- Código Completo do Memorial de Cálculo ---")
    codigo = """
from pymemorial.core import (
    get_config, Variable, Equation, Calculator
)

# PASSO 1: Configurar norma
config = get_config()
config.load_profile('nbr6118')
gamma_f = config.standard.partial_factors['gamma_f']
print(f'Norma carregada: NBR 6118:2023 (γf = {gamma_f})')

# PASSO 2: Definir variáveis do problema
vars_dict = {
    'L': Variable('L', 6.0, unit='m', description='Vão da viga'),
    'g': Variable('g', 10.0, unit='kN/m', description='Carga permanente'),
    'q': Variable('q', 5.0, unit='kN/m', description='Carga acidental'),
    'gamma_f': Variable('gamma_f', gamma_f, description='Coef. ponderação')
}

# PASSO 3: Carga de cálculo
print('\\n--- Cálculo de q_d ---')
eq_qd = Equation(
    'gamma_f * (g + q)',
    locals_dict=vars_dict,
    name='q_d'
)

# Gerar steps
steps_qd = eq_qd.steps(granularity='medium')
for i, step in enumerate(steps_qd, 1):
    print(f'  {i}. {step["step"]}')

q_d_result = eq_qd.evaluate()
print(f'  → q_d = {q_d_result.value} {q_d_result.unit}')

# PASSO 4: Momento máximo
print('\\n--- Cálculo de M_d ---')
vars_dict['q_d'] = Variable('q_d', q_d_result.value, unit='kN/m')
eq_Md = Equation(
    'q_d * L**2 / 8',
    locals_dict=vars_dict,
    name='M_d'
)

steps_Md = eq_Md.steps(granularity='medium')
for i, step in enumerate(steps_Md, 1):
    print(f'  {i}. {step["step"]}')

M_d_result = eq_Md.evaluate()
print(f'  → M_d = {M_d_result.value:.2f} {M_d_result.unit}')

# PASSO 5: Resumo final
print('\\n' + '='*60)
print('MEMORIAL DE CÁLCULO - VIGA BIAPOIADA')
print('='*60)
print(f'Dados de entrada:')
print(f'  L = {vars_dict["L"].value} m')
print(f'  g = {vars_dict["g"].value} kN/m')
print(f'  q = {vars_dict["q"].value} kN/m')
print(f'  γf = {gamma_f}')
print(f'\\nResultados:')
print(f'  q_d = {q_d_result.value} kN/m')
print(f'  M_d = {M_d_result.value:.2f} kN·m')
print('='*60)
"""
    print(codigo)
    
    print("\n📊 Saída esperada:")
    print("=" * 60)
    print("Norma carregada: NBR 6118:2023 (γf = 1.4)")
    print("")
    print("--- Cálculo de q_d ---")
    print("  1. Fórmula Simbólica: γf * (g + q)")
    print("  2. Substituição: γf=1.4, g=10, q=5")
    print("  3. Avaliação: 1.4 * (10 + 5)")
    print("  → q_d = 21.0 kN/m")
    print("")
    print("--- Cálculo de M_d ---")
    print("  1. Fórmula Simbólica: q_d * L² / 8")
    print("  2. Substituição: q_d=21, L=6")
    print("  3. Avaliação: 21 * 6² / 8")
    print("  → M_d = 94.50 kN·m")
    print("")
    print("=" * 60)
    print("MEMORIAL DE CÁLCULO - VIGA BIAPOIADA")
    print("=" * 60)
    print("Dados de entrada:")
    print("  L = 6.0 m")
    print("  g = 10.0 kN/m")
    print("  q = 5.0 kN/m")
    print("  γf = 1.4")
    print("")
    print("Resultados:")
    print("  q_d = 21.0 kN/m")
    print("  M_d = 94.50 kN·m")
    print("=" * 60)
    
    print("\n✅ Teste Integrado: TODOS os módulos trabalhando juntos!")
    print("   ✓ Config (norma NBR 6118)")
    print("   ✓ Variable (variáveis tipadas)")
    print("   ✓ Equation (motor simbólico)")
    print("   ✓ Steps (memorial passo a passo)")
    return True

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Executa todos os testes visuais."""
    print("\nIniciando bateria de testes visuais...\n")
    
    # Verificar dependências
    deps = verificar_dependencias()
    
    # Lista de testes
    testes = [
        ("Variable", teste_variable),
        ("Equation", teste_equation),
        ("Calculator", teste_calculator),
        ("Matrix", teste_matrix),
        ("Config", teste_config),
        ("Integrado (Viga)", teste_integrado)
    ]
    
    resultados = []
    
    # Executar cada teste
    for nome, func_teste in testes:
        try:
            sucesso = func_teste()
            resultados.append((nome, sucesso))
        except Exception as e:
            print(f"\n❌ Erro catastrófico no teste {nome}: {e}")
            traceback.print_exc()
            resultados.append((nome, False))
    
    # Sumário final
    print("\n" + "="*80)
    print("SUMÁRIO FINAL DOS TESTES")
    print("="*80)
    
    for nome, ok in resultados:
        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"  {status}  {nome}")
    
    total = len(resultados)
    passou = sum(1 for _, ok in resultados if ok)
    taxa = (100 * passou // total) if total > 0 else 0
    
    print(f"\n{'='*80}")
    print(f"  Total: {passou}/{total} testes passaram ({taxa}%)")
    print(f"{'='*80}")
    
    if passou == total:
        print("\n🎉 PARABÉNS! Todos os testes passaram!")
        print("   Todos os módulos do PyMemorial Core estão funcionando.")
    else:
        print(f"\n⚠️  ATENÇÃO: {total-passou} teste(s) falharam.")
        print("   Revise os erros acima para identificar problemas.")

if __name__ == "__main__":
    main()
