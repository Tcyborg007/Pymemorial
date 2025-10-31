"""
Exemplo 5: Escada com Degraus em Balanço
==========================================

Memorial de cálculo para escada com degraus engastados em viga lateral
(escada jacaré/cascata).

Norma: NBR 6118:2023
"""

from pymemorial import EngMemorial
import math

mem = EngMemorial(
    title="Escada E-1 - Acesso ao 2º Pavimento",
    author="Eng. Roberto Lima",
    norm="NBR 6118:2023"
)

# ============================================================================
# GEOMETRIA DA ESCADA
# ============================================================================

mem.section("Geometria", level=1)

mem.var("L_degrau", 1.20, "m", "Comprimento do degrau (balanço)")
mem.var("e", 0.18, "m", "Espelho do degrau")
mem.var("p", 0.28, "m", "Piso do degrau")
mem.var("h_degrau", 0.12, "m", "Espessura do degrau")

mem.var("n_degraus", 15, "", "Número de degraus")

mem.section("Inclinação", level=2)
mem.calc("tg_alpha = e / p", unit="", description="tg(α)")
mem.calc("alpha = math.atan(tg_alpha) * 180 / math.pi", unit="graus", description="Inclinação")

mem.write(f"""
Escada com inclinação de {round(mem._context.get('alpha').value, 1)}°
conforme NBR 9050 (acessibilidade).
""")

# ============================================================================
# CARREGAMENTO
# ============================================================================

mem.section("Carregamento", level=1)

mem.section("Peso Próprio", level=2)
mem.var("gamma_c", 25, "kN/m³", "Concreto armado")
mem.calc("g_degrau = gamma_c * h_degrau", unit="kN/m²", description="P.P. do degrau")

mem.section("Revestimento", level=2)
mem.var("g_rev", 1.0, "kN/m²", "Revestimento cerâmico")

mem.section("Sobrecarga", level=2)
mem.var("q", 3.0, "kN/m²", "Sobrecarga (escada residencial)")

mem.section("Carga Total", level=2)
mem.calc("g_total = g_degrau + g_rev", unit="kN/m²")

mem.var("gamma_g", 1.4, "", "γ_g")
mem.var("gamma_q", 1.4, "", "γ_q")
mem.calc("p_d = gamma_g * g_total + gamma_q * q", unit="kN/m²", description="Carga de projeto")

# ============================================================================
# ANÁLISE ESTRUTURAL (DEGRAU EM BALANÇO)
# ============================================================================

mem.section("Análise Estrutural", level=1)

mem.write("""
Cada degrau trabalha como **viga em balanço** engastada na viga lateral.
""")

mem.section("Carga por Metro Linear", level=2)
mem.var("b_ref", 1.0, "m", "Faixa de referência (1 metro)")
mem.calc("p_linha = p_d * b_ref", unit="kN/m", description="Carga linear")

mem.section("Momento de Engastamento", level=2)
mem.calc("M_eng = p_linha * L_degrau**2 / 2", unit="kN.m/m", description="Momento no engaste")

# ============================================================================
# DIMENSIONAMENTO DA ARMADURA
# ============================================================================

mem.section("Dimensionamento da Armadura", level=1)

mem.section("Materiais", level=2)
mem.var("f_ck", 30, "MPa", "Concreto C30")
mem.var("f_yk", 500, "MPa", "Aço CA-50")

mem.calc("f_cd = f_ck / 1.4", unit="MPa")
mem.calc("f_yd = f_yk / 1.15", unit="MPa")

mem.section("Altura Útil", level=2)
mem.var("d", 0.09, "m", "Altura útil (h - 3cm)")

mem.section("Armadura Principal (Negativa)", level=2)
mem.calc("A_s_neg = M_eng * 100 / (f_yd * 0.9 * d)", unit="cm²/m", description="A_s negativa")

mem.var("A_s_min", 1.5, "cm²/m", "Armadura mínima")
mem.calc("A_s_neg_adot = max(A_s_neg, A_s_min)", unit="cm²/m", description="A_s negativa adotada")

mem.write("""
**Armadura negativa (superior):** φ8mm c/ 10cm (A_s = 5.03 cm²/m)

**Armadura positiva (inferior):** φ6.3mm c/ 15cm (A_s = 2.10 cm²/m)
Armadura de distribuição.

**Armadura de pele (lateral):** φ5mm c/ 20cm
""")

# ============================================================================
# VIGA LATERAL (LONGARINAS)
# ============================================================================

mem.section("Viga Lateral (Longarina)", level=1)

mem.write("""
As vigas laterais recebem as reações dos degraus em balanço.
Dimensionamento simplificado considerando viga contínua.
""")

mem.var("b_v", 0.15, "m", "Largura da viga")
mem.var("h_v", 0.30, "m", "Altura da viga")
mem.var("L_lance", 3.20, "m", "Comprimento do lance")

mem.calc("R_degrau = p_linha * L_degrau", unit="kN", description="Reação de cada degrau")
mem.calc("V_total = R_degrau * n_degraus", unit="kN", description="Carga total no lance")

mem.write("""
A viga lateral deve ser dimensionada como viga contínua recebendo
cargas concentradas (reações dos degraus). Recomenda-se análise por
software de elementos finitos.
""")

# Exportar
mem.save("memorial_escada_e1.pdf")
print("✅ Memorial de escada gerado!")
