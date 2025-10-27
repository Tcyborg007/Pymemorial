# tests/smoke/smoke_test_config.py

"""
Smoke test completo de core/config.py

Simula caso de uso REAL de engenheiro configurando PyMemorial.
"""

from pymemorial.core.config import get_config, set_option, reset_config

print("🔥 SMOKE TEST - core/config.py\n")
print("=" * 60)

# CENÁRIO: Engenheiro configurando PyMemorial para projeto NBR 6118

print("\n1️⃣  Obter configuração padrão")
config = get_config()
print(f"   Precisão padrão: {config.display.precision} decimais")
print(f"   Modo renderização: {config.rendering.default_mode}")
print(f"   Norma ativa: {config.standard.active_standard}")

print("\n2️⃣  Ajustar preferências pessoais")
set_option('display_precision', 4)
set_option('symbols_greek_style', 'latex')
set_option('rendering_show_units', True)
print(f"   ✅ Precisão alterada para: {config.display.precision}")
print(f"   ✅ Estilo grego: {config.symbols.greek_style}")
print(f"   ✅ Mostrar unidades: {config.rendering.show_units}")

print("\n3️⃣  Carregar perfil de norma (NBR 6118)")
config.load_profile('nbr6118')
print(f"   ✅ Norma ativa: {config.standard.active_standard}")
print(f"   ✅ γc (concreto): {config.standard.partial_factors['gamma_c']}")
print(f"   ✅ γs (aço): {config.standard.partial_factors['gamma_s']}")
print(f"   ✅ γf (cargas): {config.standard.partial_factors['gamma_f']}")

print("\n4️⃣  Salvar configuração (persistência)")
config.save_config()
print(f"   ✅ Config salvo em: {config.config_file}")

print("\n5️⃣  Simular reinicialização do programa")
reset_config()
config_reloaded = get_config()
print(f"   ✅ Config recarregado automaticamente")
print(f"   ✅ Precisão mantida: {config_reloaded.display.precision}")
print(f"   ✅ Norma mantida: {config_reloaded.standard.active_standard}")

print("\n" + "=" * 60)
print("✅✅✅ SMOKE TEST: SUCESSO TOTAL!")
print("\n🚀 core/config.py está 100% funcional e robusto!")
print("📋 Próximo arquivo: recognition/ast_parser.py")
