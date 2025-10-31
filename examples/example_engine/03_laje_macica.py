"""
Exemplo 3: Laje Maciça - Análise Elástica
==========================================

Memorial de cálculo para laje maciça retangular com bordas
apoiadas, utilizando tabelas de Marcus.

Norma: NBR 6118:2023
"""

from pymemorial import EngMemorial

mem = EngMemorial(
    title="Laje L1 - Pavimento Tipo",
    author="Eng. Carlos Oliveira",
    norm="NBR 6118:2023"
)

# ============================================================================
# GEOMETRIA E MATERIAIS
# ============================================================================

mem.section("Dados da Laje", level=1)

mem.var("l_x", 4.0, "m", "Menor vão")
mem.var("l_y", 5.5, "m", "Maior vão")
mem.var("h", 0.12, "m", "Espessura da laje")

mem.var("f_ck", 25, "MPa", "Concreto C25")
mem.var("f_yk", 500, "MPa", "Aço CA-50")

# ============================================================================
# CARGAS
# ============================================================================

mem.section("Carregamento", level=1)

mem.section("Peso Próprio", level=2)
mem.var("gamma_c", 25, "kN/m³", "Peso específico do concreto")
mem.calc("g_pp = gamma_c * h", unit="kN/m²", description="Peso próprio")

mem.section("Revestimentos e Sobrecarga", level=2)
mem.var("g_rev", 1.5, "kN/m²", "Revestimento + contrapiso")
mem.calc("g_total = g_pp + g_rev", unit="kN/m²", description="Carga permanente total")

mem.var("q", 2.0, "kN/m²", "Sobrecarga de uso (residencial)")

mem.section("Carga de Projeto", level=2)
mem.var("gamma_g", 1.4, "", "γ_g")
mem.var("gamma_q", 1.4, "", "γ_q")

mem.calc("p_d = gamma_g * g_total + gamma_q * q", unit="kN/m²", description="Carga de projeto")

# ============================================================================
# ANÁLISE DE MOMENTOS (TABELAS DE MARCUS)
# ============================================================================

mem.section("Momentos Fletores", level=1)

mem.section("Relação entre Vãos", level=2)
mem.calc("lambda_laje = l_y / l_x", unit="", description="λ (relação de vãos)")

mem.write("""
Laje trabalhando em **duas direções** (λ < 2.0).
Utilizando coeficientes de Marcus para laje com 4 bordas apoiadas.
""")

mem.section("Momentos Máximos", level=2)
# Coeficientes tabelados (Marcus)
mem.var("mu_x", 0.0825, "", "Coef. momento direção x (tabela)")
mem.var("mu_y", 0.0450, "", "Coef. momento direção y (tabela)")

mem.calc("M_x = mu_x * p_d * l_x**2", unit="kN.m/m", description="Momento em x")
mem.calc("M_y = mu_y * p_d * l_x**2", unit="kN.m/m", description="Momento em y")

# ============================================================================
# DIMENSIONAMENTO DAS ARMADURAS
# ============================================================================

mem.section("Armadura de Flexão", level=1)

mem.section("Parâmetros", level=2)
mem.var("d_x", 0.10, "m", "Altura útil direção x (h - 2cm)")
mem.var("d_y", 0.09, "m", "Altura útil direção y (h - 3cm)")

mem.calc("f_cd = f_ck / 1.4", unit="MPa")
mem.calc("f_yd = f_yk / 1.15", unit="MPa")

mem.section("Direção X (menor vão)", level=2)
mem.calc("K_x = M_x / (1.0 * d_x**2 * f_cd) * 1000", unit="", description="K_x")
mem.calc("A_sx = M_x * 100 / (f_yd * 0.9 * d_x)", unit="cm²/m", description="A_s,x calculada")

mem.var("A_sx_min", 1.5, "cm²/m", "Armadura mínima (NBR 6118)")
mem.calc("A_sx_adot = max(A_sx, A_sx_min)", unit="cm²/m", description="A_s,x adotada")

mem.write("**Armadura adotada (dir. x):** φ8mm c/ 15cm (A_s = 3.35 cm²/m)")

mem.section("Direção Y (maior vão)", level=2)
mem.calc("K_y = M_y / (1.0 * d_y**2 * f_cd) * 1000", unit="")
mem.calc("A_sy = M_y * 100 / (f_yd * 0.9 * d_y)", unit="cm²/m", description="A_s,y calculada")

mem.var("A_sy_min", 1.5, "cm²/m", "Armadura mínima")
mem.calc("A_sy_adot = max(A_sy, A_sy_min)", unit="cm²/m", description="A_s,y adotada")

mem.write("**Armadura adotada (dir. y):** φ6.3mm c/ 15cm (A_s = 2.10 cm²/m)")

# ============================================================================
# VERIFICAÇÕES
# ============================================================================

mem.section("Verificações", level=1)

mem.section("Flecha", level=2)
mem.write("""
Verificação da flecha máxima conforme NBR 6118 (ELS-DEF):
- Relação L/h = 33.3 < 35 (laje com bordas apoiadas) ✅
- Dispensa verificação detalhada de flechas
""")

mem.section("Fissuração", level=2)
mem.var("w_k_lim", 0.3, "mm", "Abertura limite de fissuras (CAA I)")
mem.write("Verificação dispensada para h > 10cm e armadura adequada (NBR 6118).")

# Exportar
mem.save("memorial_laje_l1.pdf")
print("✅ Memorial de laje gerado!")
