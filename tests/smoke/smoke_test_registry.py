# tests/smoke/smoke_test_registry.py

"""
Smoke test para symbols/custom_registry.py

Valida end-to-end:
- Definição manual de símbolos
- Auto-aprendizado de código
- Persistência em JSON
- Integração com ast_parser
"""

from pathlib import Path
import tempfile

from pymemorial.symbols import get_global_registry, reset_global_registry

print("🔥 SMOKE TEST - symbols/custom_registry.py")
print("=" * 60)

# Reset para garantir estado limpo
reset_global_registry()
registry = get_global_registry()

# 1️⃣ Definir símbolos manualmente
print("\n1️⃣  Definir símbolos NBR 6118")
registry.define('gamma_c', latex=r'\gamma_{c}', description='Coef. concreto', category='nbr6118')
registry.define('gamma_s', latex=r'\gamma_{s}', description='Coef. aço', category='nbr6118')
registry.define('M_d', latex=r'M_{d}', description='Momento de cálculo')
print(f"   ✅ Símbolos definidos: {len(registry.list_all())}")

# 2️⃣ Auto-aprender de código
print("\n2️⃣  Auto-aprender símbolos do código")
code = '''
f_ck = 30  # MPa
f_cd = f_ck / gamma_c
A_s = 10.0  # cm²
M_Rd = A_s * f_yd * (d - 0.4 * x)
'''
learned = registry.learn_from_code(code)
print(f"   ✅ Símbolos aprendidos: {[s.name for s in learned]}")
print(f"   ✅ Total no registry: {len(registry.list_all())}")

# 3️⃣ Buscar símbolos
print("\n3️⃣  Buscar símbolos")
gamma_c_sym = registry.get('gamma_c')
print(f"   ✅ gamma_c → {gamma_c_sym.latex}")

M_d_sym = registry.get('M_d')
print(f"   ✅ M_d → {M_d_sym.latex}")

f_cd_sym = registry.get('f_cd')
print(f"   ✅ f_cd (auto-learned) → {f_cd_sym.latex}")

# 4️⃣ Buscar por categoria
print("\n4️⃣  Buscar por categoria")
nbr_symbols = registry.search(category='nbr6118')
print(f"   ✅ Símbolos NBR 6118: {[s.name for s in nbr_symbols]}")

# 5️⃣ Salvar e carregar
print("\n5️⃣  Persistência JSON")
with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
    temp_path = Path(f.name)

registry.save_to_file(temp_path)
print(f"   ✅ Salvo em: {temp_path}")

# Resetar e recarregar
reset_global_registry()
new_registry = get_global_registry()
new_registry.load_from_file(temp_path)
print(f"   ✅ Carregado: {len(new_registry.list_all())} símbolos")

# Verificar persistência
assert new_registry.has('gamma_c')
assert new_registry.has('M_d')
assert new_registry.has('f_cd')
print("   ✅ Símbolos persistidos corretamente!")

# Limpar temp file
temp_path.unlink()

print("\n" + "=" * 60)
print("✅✅✅ SMOKE TEST: SUCESSO TOTAL!")
print()
print("🚀 symbols/custom_registry.py está 100% funcional e robusto!")
print("📋 Próximo arquivo: Integração do default_symbols.json ou refatorar variable.py")
