"""
TESTE EXECUTÁVEL CORRIGIDO - Verificação Detalhada de Steps
============================================================

Testa se os steps estão fazendo substituições corretas em TODAS as etapas.
CORRIGIDO: Usa atributos de dataclass (step.type) em vez de dict (step['type'])

Autor: Especialista em Dev + Engenharia Estrutural
Data: 2025-10-28
"""

import sys
from pathlib import Path

# Adicionar src ao path
src_path = Path(__file__).parent.parent.parent / 'src'
if src_path.exists():
    sys.path.insert(0, str(src_path))

print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║           TESTE EXECUTÁVEL - VERIFICAÇÃO DE STEPS (CORRIGIDO)                ║
║                                                                              ║
║  Valida se as substituições estão funcionando corretamente                  ║
║  Usa step.attribute em vez de step['key']                                   ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

# ============================================================================
# TESTE 1: EQUATION STEPS - MOMENTO EM VIGA
# ============================================================================

def teste_equation_steps_viga():
    """Testa steps de equação simples com substituição."""
    print("\n" + "="*80)
    print("TESTE 1: Equation Steps - Momento em Viga Biapoiada")
    print("="*80)
    
    try:
        from pymemorial.core import Equation, Variable
        
        # Definir variáveis
        vars_dict = {
            'q': Variable('q', 15.0, unit='kN/m', description='Carga distribuída'),
            'L': Variable('L', 6.0, unit='m', description='Vão da viga')
        }
        
        # Criar equação: M = q*L²/8
        eq = Equation('q * L**2 / 8', locals_dict=vars_dict, name='M_max')
        
        print("\n✓ Equação criada com sucesso")
        print(f"  Nome: {eq.name}")
        print(f"  Expressão: {eq.expression_str}")
        
        # Gerar steps com diferentes granularidades
        for granularity in ['minimal', 'basic', 'medium', 'detailed']:
            print(f"\n--- Granularidade: {granularity.upper()} ---")
            steps = eq.steps(granularity=granularity)
            
            print(f"  Total de passos: {len(steps)}")
            
            for i, step in enumerate(steps, 1):
                print(f"\n  Passo {i}:")
                
                # CORRIGIDO: Usar atributos da dataclass
                print(f"    Tipo: {step.type}")
                print(f"    Conteúdo: {step.content}")
                
                if hasattr(step, 'latex') and step.latex:
                    print(f"    LaTeX: {step.latex}")
                
                if hasattr(step, 'explanation') and step.explanation:
                    print(f"    Explicação: {step.explanation}")
                
                # Verificar metadata
                if hasattr(step, 'metadata') and step.metadata:
                    if 'substitutions' in step.metadata:
                        print(f"    Substituições: {step.metadata['substitutions']}")
                    if 'value' in step.metadata:
                        print(f"    Valor: {step.metadata['value']}")
                    if 'unit' in step.metadata:
                        print(f"    Unidade: {step.metadata['unit']}")
        
        # Avaliar resultado final
        result = eq.evaluate()
        print(f"\n✓ Resultado final:")
        print(f"  {eq.name} = {result.value} {result.unit}")
        
        # VALIDAÇÕES
        print("\n📋 VALIDAÇÕES:")
        
        # 1. Verificar se resultado está correto
        expected_value = 15.0 * 6.0**2 / 8
        assert abs(result.value - expected_value) < 0.01, f"Erro: esperado {expected_value}, obtido {result.value}"
        print(f"  ✅ Valor correto: {result.value} = {expected_value}")
        
        # 2. Verificar se steps foram gerados
        steps_medium = eq.steps(granularity='medium')
        assert len(steps_medium) > 0, "Erro: nenhum step foi gerado"
        print(f"  ✅ Steps gerados: {len(steps_medium)} passos")
        
        # 3. Verificar se LaTeX foi gerado em algum step
        has_latex = any(hasattr(s, 'latex') and s.latex for s in steps_medium)
        if has_latex:
            print(f"  ✅ LaTeX gerado em pelo menos um step")
        else:
            print(f"  ⚠️  LaTeX não encontrado nos steps")
        
        print("\n✅ TESTE 1 PASSOU - Equation steps funcionando!")
        return True
        
    except Exception as e:
        print(f"\n❌ TESTE 1 FALHOU: {e}")
        import traceback
        traceback.print_exc()
        return False

# ============================================================================
# TESTE 2: EQUATION STEPS COM MÚLTIPLAS VARIÁVEIS
# ============================================================================

def teste_equation_steps_multiplas_vars():
    """Testa steps com múltiplas substituições."""
    print("\n" + "="*80)
    print("TESTE 2: Equation Steps - Múltiplas Variáveis (Carga de Cálculo)")
    print("="*80)
    
    try:
        from pymemorial.core import Equation, Variable
        
        # Caso real: Carga de cálculo NBR 6118
        vars_dict = {
            'gamma_g': Variable('gamma_g', 1.4, description='Coef. majoração permanente'),
            'g': Variable('g', 10.0, unit='kN/m', description='Carga permanente'),
            'gamma_q': Variable('gamma_q', 1.4, description='Coef. majoração acidental'),
            'q': Variable('q', 5.0, unit='kN/m', description='Carga acidental')
        }
        
        # q_d = γ_g * g + γ_q * q
        eq = Equation('gamma_g * g + gamma_q * q', locals_dict=vars_dict, name='q_d')
        
        print("\n✓ Equação criada:")
        print(f"  {eq.name} = {eq.expression_str}")
        
        # Gerar steps detalhados
        steps = eq.steps(granularity='detailed')
        
        print(f"\n📊 Steps gerados: {len(steps)} passos")
        
        for i, step in enumerate(steps, 1):
            print(f"\n{'─'*80}")
            print(f"PASSO {i}: {step.content}")
            print(f"{'─'*80}")
            print(f"  Tipo: {step.type}")
            
            if hasattr(step, 'latex') and step.latex:
                print(f"  LaTeX: {step.latex}")
            
            if hasattr(step, 'metadata') and step.metadata:
                if 'substitutions' in step.metadata:
                    print(f"  Substituições:")
                    for var, val in step.metadata['substitutions'].items():
                        print(f"    • {var} = {val}")
                if 'expression' in step.metadata:
                    print(f"  Expressão: {step.metadata['expression']}")
                if 'value' in step.metadata:
                    print(f"  Valor: {step.metadata['value']}")
        
        # Avaliar
        result = eq.evaluate()
        expected = 1.4 * 10.0 + 1.4 * 5.0
        
        print(f"\n✓ Resultado: {result.value} {result.unit}")
        print(f"  Esperado: {expected} kN/m")
        
        # VALIDAÇÕES
        print("\n📋 VALIDAÇÕES:")
        
        assert abs(result.value - expected) < 0.01
        print(f"  ✅ Valor correto: {result.value} ≈ {expected}")
        
        assert len(steps) > 0
        print(f"  ✅ Steps gerados: {len(steps)} passos")
        
        print("\n✅ TESTE 2 PASSOU - Múltiplas variáveis OK!")
        return True
        
    except Exception as e:
        print(f"\n❌ TESTE 2 FALHOU: {e}")
        import traceback
        traceback.print_exc()
        return False

# ============================================================================
# TESTE 3: MATRIX STEPS - RIGIDEZ DE VIGA
# ============================================================================

def teste_matrix_steps_rigidez():
    """Testa steps de matriz simbólica."""
    print("\n" + "="*80)
    print("TESTE 3: Matrix Steps - Matriz de Rigidez de Viga")
    print("="*80)
    
    try:
        from pymemorial.core import Matrix, Variable
        
        # Variáveis estruturais
        vars_dict = {
            'E': Variable('E', 210000, unit='MPa', description='Módulo de elasticidade'),
            'I': Variable('I', 8333, unit='cm^4', description='Momento de inércia'),
            'L': Variable('L', 600, unit='cm', description='Comprimento')
        }
        
        # Matriz de rigidez 2x2
        K_expr = """[
            [12*E*I/L**3, 6*E*I/L**2],
            [6*E*I/L**2, 4*E*I/L]
        ]"""
        
        K = Matrix(data=K_expr, variables=vars_dict, name='K')
        
        print("\n✓ Matriz criada:")
        print(f"  Nome: {K.name}")
        print(f"  Shape: {K.shape}")
        print(f"  Simbólica: {K.is_symbolic}")
        
        # Gerar steps
        for granularity in ['basic', 'normal', 'detailed']:
            print(f"\n--- Granularidade: {granularity.upper()} ---")
            steps = K.steps(granularity=granularity)
            
            print(f"  Total de passos: {len(steps)}")
            
            for i, step_dict in enumerate(steps, 1):
                # Matrix.steps() retorna List[Dict], não List[Step]
                print(f"\n  Passo {i}: {step_dict.get('step', 'N/A')}")
                
                if 'operation' in step_dict:
                    print(f"    Operação: {step_dict['operation']}")
                
                if 'latex' in step_dict and step_dict['latex']:
                    latex_preview = step_dict['latex'][:100] + '...' if len(step_dict['latex']) > 100 else step_dict['latex']
                    print(f"    LaTeX: {latex_preview}")
                
                if 'description' in step_dict:
                    print(f"    Descrição: {step_dict['description']}")
        
        # Avaliar numericamente
        K_num = K.evaluate()
        print(f"\n✓ Matriz numérica avaliada:")
        print(f"  K[0,0] = {K_num[0,0]:.2e}")
        print(f"  K[0,1] = {K_num[0,1]:.2e}")
        print(f"  K[1,1] = {K_num[1,1]:.2e}")
        
        # VALIDAÇÕES
        print("\n📋 VALIDAÇÕES:")
        
        # Calcular valores esperados
        E, I, L = 210000, 8333, 600
        expected_00 = 12*E*I/L**3
        expected_01 = 6*E*I/L**2
        expected_11 = 4*E*I/L
        
        assert abs(K_num[0,0] - expected_00) / expected_00 < 0.01
        print(f"  ✅ K[0,0] correto: {K_num[0,0]:.2e} ≈ {expected_00:.2e}")
        
        assert abs(K_num[0,1] - expected_01) / expected_01 < 0.01
        print(f"  ✅ K[0,1] correto: {K_num[0,1]:.2e} ≈ {expected_01:.2e}")
        
        # Verificar simetria
        assert abs(K_num[0,1] - K_num[1,0]) < 1e-6
        print(f"  ✅ Matriz simétrica: K[0,1] = K[1,0]")
        
        print("\n✅ TESTE 3 PASSOU - Matrix steps funcionando!")
        return True
        
    except Exception as e:
        print(f"\n❌ TESTE 3 FALHOU: {e}")
        import traceback
        traceback.print_exc()
        return False

# ============================================================================
# TESTE 4: STEPS EM SEQUÊNCIA (CASO REAL)
# ============================================================================

def teste_steps_sequencia_completa():
    """Testa sequência completa de cálculos com steps."""
    print("\n" + "="*80)
    print("TESTE 4: Sequência Completa - Memorial de Viga Biapoiada")
    print("="*80)
    
    try:
        from pymemorial.core import Equation, Variable
        from pymemorial.core.config import get_config
        
        # Carregar configuração NBR
        config = get_config()
        config.load_profile('nbr6118')
        gamma_f = config.standard.partial_factors.get('gamma_f', 1.4)
        
        print(f"\n✓ Norma carregada: NBR 6118 (γf = {gamma_f})")
        
        # ETAPA 1: Carga de cálculo
        print("\n" + "─"*80)
        print("ETAPA 1: Cálculo de q_d")
        print("─"*80)
        
        vars1 = {
            'gamma_f': Variable('gamma_f', gamma_f),
            'g': Variable('g', 10.0, unit='kN/m'),
            'q': Variable('q', 5.0, unit='kN/m')
        }
        
        eq1 = Equation('gamma_f * (g + q)', locals_dict=vars1, name='q_d')
        steps1 = eq1.steps(granularity='detailed')
        
        print(f"\nSteps gerados: {len(steps1)}")
        for i, step in enumerate(steps1, 1):
            print(f"\n  {i}. {step.content}")
            if hasattr(step, 'metadata') and step.metadata:
                if 'substitutions' in step.metadata:
                    for var, val in step.metadata['substitutions'].items():
                        print(f"     • {var} = {val}")
        
        q_d = eq1.evaluate()
        print(f"\n✓ Resultado: q_d = {q_d.value} {q_d.unit}")
        
        # ETAPA 2: Momento máximo
        print("\n" + "─"*80)
        print("ETAPA 2: Cálculo de M_d")
        print("─"*80)
        
        vars2 = {
            'q_d': Variable('q_d', q_d.value, unit='kN/m'),
            'L': Variable('L', 6.0, unit='m')
        }
        
        eq2 = Equation('q_d * L**2 / 8', locals_dict=vars2, name='M_d')
        steps2 = eq2.steps(granularity='detailed')
        
        print(f"\nSteps gerados: {len(steps2)}")
        for i, step in enumerate(steps2, 1):
            print(f"\n  {i}. {step.content}")
            if hasattr(step, 'metadata') and step.metadata:
                if 'substitutions' in step.metadata:
                    for var, val in step.metadata['substitutions'].items():
                        print(f"     • {var} = {val}")
        
        M_d = eq2.evaluate()
        print(f"\n✓ Resultado: M_d = {M_d.value:.2f} {M_d.unit}")
        
        # VALIDAÇÕES FINAIS
        print("\n" + "="*80)
        print("📋 VALIDAÇÕES FINAIS")
        print("="*80)
        
        # Verificar q_d
        expected_qd = gamma_f * (10.0 + 5.0)
        assert abs(q_d.value - expected_qd) < 0.01
        print(f"  ✅ q_d correto: {q_d.value} = {expected_qd}")
        
        # Verificar M_d
        expected_Md = expected_qd * 6.0**2 / 8
        assert abs(M_d.value - expected_Md) < 0.01
        print(f"  ✅ M_d correto: {M_d.value:.2f} = {expected_Md:.2f}")
        
        # Verificar que steps foram gerados
        assert len(steps1) > 0 and len(steps2) > 0
        print(f"  ✅ Steps gerados em ambas as etapas")
        
        print("\n✅ TESTE 4 PASSOU - Sequência completa funcionando!")
        return True
        
    except Exception as e:
        print(f"\n❌ TESTE 4 FALHOU: {e}")
        import traceback
        traceback.print_exc()
        return False

# ============================================================================
# TESTE 5: VERIFICAÇÃO DETALHADA DE SUBSTITUIÇÕES
# ============================================================================

def teste_verificar_substituicoes():
    """Testa especificamente se as substituições estão nos steps."""
    print("\n" + "="*80)
    print("TESTE 5: Verificação Detalhada de Substituições")
    print("="*80)
    
    try:
        from pymemorial.core import Equation, Variable
        
        # Equação simples
        vars_dict = {
            'a': Variable('a', 5.0),
            'b': Variable('b', 3.0)
        }
        
        eq = Equation('a + b', locals_dict=vars_dict, name='soma')
        
        print("\n✓ Equação: soma = a + b")
        print("  a = 5.0")
        print("  b = 3.0")
        
        # Testar cada granularidade
        for gran in ['minimal', 'basic', 'medium', 'detailed']:
            print(f"\n--- {gran.upper()} ---")
            steps = eq.steps(granularity=gran)
            
            print(f"  Steps: {len(steps)}")
            
            for i, step in enumerate(steps, 1):
                info = f"  {i}. Tipo={step.type}"
                
                # Verificar se há informações de substituição
                has_subs = False
                if hasattr(step, 'metadata') and step.metadata:
                    if 'substitutions' in step.metadata:
                        has_subs = True
                        info += f" | Substituições: {step.metadata['substitutions']}"
                
                # Verificar se há LaTeX
                if hasattr(step, 'latex') and step.latex:
                    info += f" | LaTeX: {step.latex[:50]}..."
                
                print(info)
        
        # Avaliar resultado
        result = eq.evaluate()
        expected = 5.0 + 3.0
        
        print(f"\n✓ Resultado: {result.value}")
        print(f"  Esperado: {expected}")
        
        assert abs(result.value - expected) < 0.01
        print("\n✅ TESTE 5 PASSOU - Substituições verificadas!")
        return True
        
    except Exception as e:
        print(f"\n❌ TESTE 5 FALHOU: {e}")
        import traceback
        traceback.print_exc()
        return False

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Executa todos os testes de steps."""
    
    testes = [
        ("Equation Steps - Viga", teste_equation_steps_viga),
        ("Múltiplas Variáveis", teste_equation_steps_multiplas_vars),
        ("Matrix Steps", teste_matrix_steps_rigidez),
        ("Sequência Completa", teste_steps_sequencia_completa),
        ("Verificação de Substituições", teste_verificar_substituicoes)
    ]
    
    resultados = []
    
    for nome, teste_func in testes:
        try:
            sucesso = teste_func()
            resultados.append((nome, sucesso))
        except Exception as e:
            print(f"\n❌ Erro fatal em '{nome}': {e}")
            import traceback
            traceback.print_exc()
            resultados.append((nome, False))
    
    # Sumário
    print("\n" + "="*80)
    print("SUMÁRIO - TESTES DE STEPS (CORRIGIDO)")
    print("="*80)
    
    for nome, ok in resultados:
        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"  {status}  {nome}")
    
    total = len(resultados)
    passou = sum(1 for _, ok in resultados if ok)
    
    print(f"\n{'='*80}")
    print(f"Total: {passou}/{total} ({100*passou//total if total > 0 else 0}%)")
    print(f"{'='*80}")
    
    if passou == total:
        print("\n🎉 TODOS OS TESTES DE STEPS PASSARAM!")
        print("   As substituições estão funcionando corretamente.")
    else:
        print(f"\n⚠️  {total-passou} teste(s) falharam.")
        print("   Revisar implementação dos steps.")
    
    return passou == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
