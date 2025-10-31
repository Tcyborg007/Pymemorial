"""
Exemplo 4: Sapata Isolada Centrada
===================================

Dimensionamento de fundação superficial (sapata rígida) para pilar.

Norma: NBR 6122:2019 (Fundações)
      NBR 6118:2023 (Concreto)
"""

from pymemorial import EngMemorial

mem = EngMemorial(
    title="Sapata S-5 - Pilar P-5",
    author="Eng. Ana Paula",
    norm="NBR 6122:2019 / NBR 6118:2023"
)

# ============================================================================
# DADOS DO PILAR E SOLO
# ============================================================================

mem.section("Dados de Entrada", level=1)

mem.section("Pilar", level=2)
mem.var("a_p", 0.25, "m", "Largura do pilar")
mem.var("b_p", 0.40, "m", "Comprimento do pilar")
mem.var("N_k", 750, "kN", "Carga característica")

mem.section("Solo", level=2)
mem.var("sigma_adm", 250, "kPa", "Tensão admissível do solo (SPT)")
mem.var("prof", 1.50, "m", "Profundidade de assentamento")

mem.section("Materiais", level=2)
mem.var("f_ck", 25, "MPa", "Concreto C25")
mem.var("f_yk", 500, "MPa", "Aço CA-50")
mem.var("gamma_c", 25, "kN/m³", "Peso esp. concreto")
mem.var("gamma_solo", 18, "kN/m³", "Peso esp. solo")

# ============================================================================
# DIMENSIONAMENTO EM PLANTA
# ============================================================================

mem.section("Dimensionamento da Base", level=1)

mem.section("Área Necessária", level=2)
mem.write("""
A sapata será dimensionada considerando a carga total (pilar + peso próprio
estimado + solo sobre a sapata).
""")

mem.var("h_est", 0.60, "m", "Altura estimada da sapata")
mem.calc("P_pp = N_k * 0.05", unit="kN", description="Peso próprio estimado (5%)")
mem.calc("P_solo = gamma_solo * prof * 2.0", unit="kN", description="Solo sobre sapata (estimado)")

mem.calc("N_total = N_k + P_pp + P_solo", unit="kN", description="Carga total")
mem.calc("A_nec = N_total / sigma_adm", unit="m²", description="Área necessária")

mem.section("Dimensões da Sapata", level=2)
mem.write("Adotando sapata quadrada:")
mem.calc("L_sap = (A_nec ** 0.5)", unit="m", description="Lado necessário")
mem.var("L", 2.00, "m", "Lado adotado (arredondado)")

mem.calc("A_sap = L * L", unit="m²", description="Área adotada")
mem.calc("sigma_solo = N_total / A_sap", unit="kPa", description="Tensão no solo")

mem.verify("sigma_solo <= sigma_adm", norm="NBR 6122", desc="Tensão admissível")

# ============================================================================
# ALTURA DA SAPATA (PUNÇÃO)
# ============================================================================

mem.section("Verificação ao Puncionamento", level=1)

mem.section("Perímetro Crítico", level=2)
mem.var("h", 0.60, "m", "Altura adotada")
mem.var("d", 0.54, "m", "Altura útil (h - 6cm)")

mem.calc("u = 2 * (a_p + b_p + 2 * d)", unit="m", description="Perímetro crítico")

mem.section("Tensão de Puncionamento", level=2)
mem.var("gamma_f", 1.4, "", "Coef. de majoração")
mem.calc("N_d = gamma_f * N_k", unit="kN", description="Carga de projeto")

mem.calc("tau_Sd = N_d / (u * d) * 1000", unit="kPa", description="Tensão solicitante")

mem.calc("f_cd = f_ck / 1.4", unit="MPa")
mem.calc("tau_Rd2 = 0.27 * (1 - f_ck/250) * f_cd * 1000", unit="kPa", description="Tensão resistente")

mem.verify("tau_Sd <= tau_Rd2", norm="NBR 6118", desc="Puncionamento")

# ============================================================================
# ARMADURA DE FLEXÃO
# ============================================================================

mem.section("Armadura de Flexão", level=1)

mem.section("Momento Fletor", level=2)
mem.calc("sigma_d = gamma_f * sigma_solo", unit="kPa", description="Tensão de projeto")
mem.calc("a_bal = (L - a_p) / 2", unit="m", description="Balanço")

mem.calc("M_d = sigma_d * L * a_bal**2 / 2", unit="kN.m", description="Momento no balanço")

mem.section("Armadura Principal", level=2)
mem.calc("f_yd = f_yk / 1.15", unit="MPa")
mem.calc("A_s = M_d * 1000 / (f_yd * 0.9 * d)", unit="cm²", description="A_s calculada")

mem.var("A_s_min", 6.0, "cm²", "Armadura mínima (ρ_min = 0.15%)")
mem.calc("A_s_adot = max(A_s, A_s_min)", unit="cm²", description="A_s adotada")

mem.write("""
**Armadura adotada:**
- Fundo (duas direções): 10φ12.5mm (A_s = 12.27 cm²)
- Espaçamento: ≈ 20cm
""")

# Exportar
mem.save("memorial_sapata_s5.pdf")
print("✅ Memorial de sapata gerado!")
