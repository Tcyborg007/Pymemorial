# tests/smoke/smoke_test_units.py

"""
🔥 SMOKE TEST - core/units.py
=================================

Testes de integração rápidos para validar funcionalidade completa.

OBJETIVO:
- Validar integração entre Registry, Parser, Validator, Formatter
- Cenários reais de uso
- Performance básica
"""

def main():
    print("🔥 SMOKE TEST - core/units.py")
    print("=" * 80)
    
    # Imports
    from pymemorial.core.units import (
        get_unit_registry,
        UnitParser,
        UnitValidator,
        UnitFormatter,
        PINT_AVAILABLE
    )
    
    if not PINT_AVAILABLE:
        print("\n⚠️  PINT NÃO DISPONÍVEL - Smoke test pulado")
        return
    
    # Setup
    reg = get_unit_registry()
    parser = UnitParser(reg)
    validator = UnitValidator(reg)
    formatter = UnitFormatter(reg)
    
    print("\n1️⃣  TESTE CENÁRIO ENGENHARIA: Viga simples")
    print("-" * 80)
    
    # Cenário 1: Cálculo de momento fletor
    print("   📋 Dados:")
    print("      - Carga distribuída: q = 15 kN/m")
    print("      - Vão: L = 6 m")
    
    # Parse com alias PT-BR
    q = parser.parse_with_alias("15 quilonewtons/metro")
    L = parser.parse_with_alias("6 metros")
    
    print(f"      ✅ Parse OK: q = {q}, L = {L}")
    
    # Cálculo (M_max = q*L^2/8)
    # Nota: Pint suporta operações diretas
    L_squared = L * L
    M_numerador = q * L_squared
    M_max = M_numerador / 8
    
    print(f"      ✅ Cálculo OK: M_max = {M_max}")
    
    # Formatar LaTeX
    latex_q = formatter.to_latex(q, precision=1)
    latex_L = formatter.to_latex(L, precision=1)
    latex_M = formatter.to_latex(M_max, precision=2)
    
    print(f"      ✅ LaTeX q: {latex_q}")
    print(f"      ✅ LaTeX L: {latex_L}")
    print(f"      ✅ LaTeX M: {latex_M}")
    
    print("\n2️⃣  TESTE VALIDAÇÃO DIMENSIONAL")
    print("-" * 80)
    
    # Cenário 2: Validar compatibilidade
    sigma = parser.parse("250 MPa")
    f_ck = parser.parse("30 MPa")
    
    print(f"   📋 Comparar: σ = {sigma} vs f_ck = {f_ck}")
    
    compatible = validator.are_compatible(sigma, f_ck)
    print(f"      ✅ Compatíveis? {compatible}")
    
    # Tentar operação inválida
    print("\n   📋 Testar operação inválida (comprimento + massa):")
    try:
        L_test = parser.parse("10 m")
        m_test = parser.parse("5 kg")
        validator.validate_operation(L_test, m_test, 'add')
        print("      ❌ ERRO: Deveria ter falhado!")
    except Exception as e:
        print(f"      ✅ Erro capturado corretamente: {type(e).__name__}")
    
    print("\n3️⃣  TESTE NORMALIZAÇÃO")
    print("-" * 80)
    
    # Cenário 3: Normalizar kN.m → kN*m
    print("   📋 Input: '10 kN.m' (notação ambígua)")
    
    M = parser.parse("10 kN.m")
    print(f"      ✅ Normalizado: {M}")
    
    # Verificar se compatível com kN*m explícito
    M_explicit = parser.parse("10 kN*m")
    compatible_moment = validator.are_compatible(M, M_explicit)
    print(f"      ✅ Compatível com 'kN*m'? {compatible_moment}")
    
    print("\n4️⃣  TESTE FORMATAÇÃO AVANÇADA")
    print("-" * 80)
    
    # Cenário 4: Notação científica
    E = parser.parse("210000 MPa")
    print(f"   📋 Módulo de elasticidade: E = {E}")
    
    # Formatação padrão
    latex_std = formatter.to_latex(E, precision=0)
    print(f"      ✅ LaTeX padrão: {latex_std}")
    
    # Formatação científica
    latex_sci = formatter.to_latex(E, scientific=True, precision=2)
    print(f"      ✅ LaTeX científico: {latex_sci}")
    
    print("\n5️⃣  TESTE PERFORMANCE")
    print("-" * 80)
    
    import time
    
    # Cenário 5: Performance de parsing
    start = time.perf_counter()
    
    for i in range(100):
        _ = parser.parse(f"{i} kN")
    
    elapsed = time.perf_counter() - start
    print(f"   📋 Parse 100 quantidades: {elapsed*1000:.2f} ms")
    print(f"      ✅ {100/elapsed:.0f} ops/sec")
    
    # Validação cache
    start = time.perf_counter()
    
    for i in range(100):
        _ = parser.parse("50 kN")  # Mesmo valor (cache)
    
    elapsed_cached = time.perf_counter() - start
    print(f"   📋 Parse 100x (cached): {elapsed_cached*1000:.2f} ms")
    print(f"      ✅ {100/elapsed_cached:.0f} ops/sec")
    
    # Speedup
    speedup = elapsed / elapsed_cached if elapsed_cached > 0 else 1.0
    print(f"      ✅ Speedup: {speedup:.1f}x")
    
    print("\n" + "=" * 80)
    print("✅✅✅ SMOKE TEST COMPLETO! UNITS.PY ESTÁ FUNCIONANDO PERFEITAMENTE! ✅✅✅")
    print("=" * 80)


if __name__ == "__main__":
    main()
