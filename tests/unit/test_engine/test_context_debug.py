"""
TESTE DE VALIDAÇÃO ROBUSTO (Ponta-a-Ponta)

Valida as correções críticas em:
1. EngMemorial.calc() - Salvando resultados no contexto.
2. MemorialContext.scope() - Restaurando escopo pai.
3. MemorialContext.list_variables() - Prioridade de escopo.
"""

import sys
import logging
from pymemorial import EngMemorial
from pymemorial.engine.context import get_context, reset_context

# Configurar logging para ver o que está acontecendo
# Use logging.DEBUG para ver TODOS os logs que você adicionou
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Flag global de sucesso
all_tests_passed = True

def print_header(title):
    print("\n" + "=" * 70)
    print(f"🔬 {title}")
    print("=" * 70)

def print_check(message, condition):
    """Helper para imprimir SUCESSO ou FALHA."""
    global all_tests_passed
    if condition:
        print(f"  ✅ SUCESSO: {message}")
    else:
        print(f"  ❌ FALHA:   {message}")
        all_tests_passed = False

def run_all_tests():
    global all_tests_passed
    all_tests_passed = True # Reseta a flag para esta execução
    
    try:
        # ------------------------------------------------------------------
        # CENÁRIO 1: CÁLCULO ENCADEADO (O BUG ORIGINAL)
        # ------------------------------------------------------------------
        print_header("CENÁRIO 1: CÁLCULO ENCADEADO (a, b -> c -> d)")
        
        reset_context()
        mem = EngMemorial("Teste 1")
        ctx = get_context()

        print("\n1. Definindo a=10, b=5...")
        mem.var("a", 10)
        mem.var("b", 5)
        
        print("\n2. Calculando c = a + b...")
        mem.calc("c = a + b")
        
        c_var = ctx.get("c")
        print_check("'c' foi salvo no contexto", c_var is not None)
        if c_var:
            print_check(f"Valor de 'c' está correto ({c_var.value})", c_var.value == 15.0)

        print("\n3. Calculando d = c * 2 (Teste de dependência)...")
        mem.calc("d = c * 2")
        
        d_var = ctx.get("d")
        print_check("'d' foi calculado com sucesso", d_var is not None)
        if d_var:
            print_check(f"Valor de 'd' está correto ({d_var.value})", d_var.value == 30.0)
            
        print("\n4. Verificação final do Cenário 1:")
        print(f"  Contexto final: {list(ctx.list_variables().keys())}")
        print_check("Contexto final contém 'a, b, c, d'", 
                    all(k in ctx.list_variables() for k in ['a', 'b', 'c', 'd']))

        # ------------------------------------------------------------------
        # CENÁRIO 2: HIERARQUIA DE ESCOPO (O BUG DO 'scope()')
        # ------------------------------------------------------------------
        print_header("CENÁRIO 2: HIERARQUIA DE ESCOPO (Restauração do 'finally')")
        
        reset_context()
        mem = EngMemorial("Teste 2")
        ctx = get_context()
        
        print("\n1. Definindo 'g_global' = 100 no escopo 'root'...")
        mem.var("g_global", 100)
        print_check(f"Escopo atual é '{ctx.get_scope_path()}'", ctx.get_scope_path() == "root")

        print("\n2. Entrando no escopo 'Secao1' com 'with'...")
        with ctx.scope("Secao1"):
            print_check(f"Escopo mudou para '{ctx.get_scope_path()}'", ctx.get_scope_path() == "root.Secao1")
            print("3. Definindo 'x_local' = 20 dentro de 'Secao1'...")
            mem.var("x_local", 20)
            print("4. Calculando 'y_local = g_global + x_local'...")
            mem.calc("y_local = g_global + x_local") # Deve ser 100 + 20
            
            y_var = ctx.get("y_local")
            print_check("'y_local' (120.0) foi calculado corretamente", y_var and y_var.value == 120.0)

        print("\n5. Saindo do escopo 'with' (Teste do 'finally')...")
        print_check(f"Escopo foi restaurado para '{ctx.get_scope_path()}'", ctx.get_scope_path() == "root")

        print("\n6. Verificação final do Cenário 2:")
        print_check("'y_local' NÃO está mais no escopo 'root'", ctx.get("y_local") is None)
        print_check("'x_local' NÃO está mais no escopo 'root'", ctx.get("x_local") is None)
        print_check("'g_global' (100.0) AINDA está no escopo 'root'", ctx.get("g_global") and ctx.get("g_global").value == 100.0)

        # ------------------------------------------------------------------
        # CENÁRIO 3: PRIORIDADE DE ESCOPO (O BUG DO 'list_variables()')
        # ------------------------------------------------------------------
        print_header("CENÁRIO 3: PRIORIDADE DE ESCOPO (Sobrescrita de Variável)")
        
        reset_context()
        mem = EngMemorial("Teste 3")
        ctx = get_context()

        print("\n1. Definindo 'val' = 1 no escopo 'root'...")
        mem.var("val", 1)

        print("\n2. Entrando no escopo 'Override'...")
        with ctx.scope("Override"):
            print_check(f"Escopo mudou para '{ctx.get_scope_path()}'", ctx.get_scope_path() == "root.Override")
            print("3. Sobrescrevendo 'val' = 99 no escopo 'Override'...")
            mem.var("val", 99)
            
            print("4. Calculando 'res = val + 1'...")
            mem.calc("res = val + 1") # Deve usar val=99
            
            res_var = ctx.get("res")
            print_check("Cálculo usou 'val' local (99+1=100)", res_var and res_var.value == 100.0)

        print("\n5. Saindo do escopo 'Override'...")
        print_check(f"Escopo foi restaurado para '{ctx.get_scope_path()}'", ctx.get_scope_path() == "root")

        print("\n6. Verificação final do Cenário 3:")
        val_var = ctx.get("val")
        print_check("'val' no escopo 'root' ainda é 1", val_var and val_var.value == 1.0)
        print_check("'res' não está no escopo 'root'", ctx.get("res") is None)
        
    except Exception as e:
        print(f"\n\n{'-'*30} ERRO CATASTRÓFICO {'-'*30}")
        print(f"Um erro inesperado interrompeu os testes:")
        logging.exception(e)
        all_tests_passed = False
        
    finally:
        # Resetar o contexto global para não afetar outros testes
        reset_context()

    # Resumo final
    print("\n" + "=" * 70)
    print("RESUMO DA VALIDAÇÃO")
    print("=" * 70)
    if all_tests_passed:
        print("✅ ✅ ✅  TODOS OS CENÁRIOS PASSARAM COM SUCESSO!  ✅ ✅ ✅")
        print("As correções no 'calc' e 'scope' foram validadas.")
    else:
        print("❌ ❌ ❌  FALHA NA VALIDAÇÃO!  ❌ ❌ ❌")
        print("Um ou mais cenários falharam. Verifique os logs '❌ FALHA' acima.")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()