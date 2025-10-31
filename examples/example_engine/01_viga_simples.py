"""
Exemplo 1: Viga Biapoiada com Carga Distribuída
================================================

Memorial de cálculo para dimensionamento de viga isostática.

Norma: NBR 6118:2023
Autor: Eng. João Silva
Data: 2025-10-28
"""

from pymemorial import EngMemorial

# Criar memorial
mem = EngMemorial(
    title="Viga V-1 - Pavimento Térreo",
    author="Eng. João Silva",
    company="Estrutural Consultoria",
    project="Edifício Residencial Boa Vista",
    norm="NBR 6118:2023"
)

# ============================================================================
# DADOS DE ENTRADA
# ============================================================================

mem.section("Dados de Entrada", level=1)

mem.write("""
A viga V-1 é biapoiada e está submetida a uma carga uniformemente 
distribuída proveniente da laje L1.
""")

mem.section("Geometria", level=2)
mem.var("L", 6.0, "m", "Vão livre da viga")
mem.var("b_w", 0.20, "m", "Largura da alma")
mem.var("h", 0.50, "m", "Altura total da seção")

mem.section("Materiais", level=2)
mem.var("f_ck", 30, "MPa", "Resistência característica do concreto")
mem.var("f_yk", 500, "MPa", "Resistência característica do aço")

mem.section("Carregamento", level=2)
mem.var("g", 10.0, "kN/m", "Carga permanente")
mem.var("q", 5.0, "kN/m", "Carga acidental")

# ============================================================================
# ANÁLISE ESTRUTURAL
# ============================================================================

mem.section("Análise Estrutural", level=1)

mem.section("Carga Total de Projeto", level=2)
mem.var("gamma_g", 1.4, "", "Coeficiente de majoração - perm.")
mem.var("gamma_q", 1.4, "", "Coeficiente de majoração - acid.")

mem.calc("g_d = gamma_g * g", unit="kN/m", description="Carga perm. de projeto")
mem.calc("q_d = gamma_q * q", unit="kN/m", description="Carga acid. de projeto")
mem.calc("p_d = g_d + q_d", unit="kN/m", description="Carga total de projeto")

mem.section("Esforços Solicitantes", level=2)
mem.write("""
Para viga biapoiada com carga uniformemente distribuída, os esforços 
máximos ocorrem no meio do vão.
""")

mem.calc("M_d = p_d * L**2 / 8", unit="kN.m", description="Momento fletor máximo")
mem.calc("V_d = p_d * L / 2", unit="kN", description="Cortante máximo")

# ============================================================================
# DIMENSIONAMENTO À FLEXÃO
# ============================================================================

mem.section("Dimensionamento à Flexão", level=1)

mem.section("Parâmetros de Cálculo", level=2)
mem.var("d", 0.45, "m", "Altura útil (d = h - 5cm)")
mem.calc("f_cd = f_ck / 1.4", unit="MPa", description="Resistência de cálculo do concreto")
mem.calc("f_yd = f_yk / 1.15", unit="MPa", description="Resistência de cálculo do aço")

mem.section("Momento Resistente", level=2)
mem.calc("K_c = M_d / (b_w * d**2 * f_cd) * 1000", unit="", description="Coef. adimensional")

mem.verify("K_c <= 0.295", norm="NBR 6118", desc="Domínio 2/3 (armadura simples)")

mem.section("Área de Aço Necessária", level=2)
mem.calc("A_s = M_d * 1000 / (f_yd * 0.9 * d)", unit="cm²", description="Área de aço calculada")

mem.var("A_s_min", 1.5, "cm²", "Área de aço mínima (tabela NBR 6118)")
mem.verify("A_s >= A_s_min", norm="NBR 6118", desc="Armadura mínima")

mem.write("""
**Armadura adotada:** 4φ12.5mm (A_s = 4.90 cm²)
""")

# ============================================================================
# VERIFICAÇÃO AO CISALHAMENTO
# ============================================================================

mem.section("Verificação ao Cisalhamento", level=1)

mem.calc("tau_wd = V_d / (b_w * d) * 1000", unit="MPa", description="Tensão de cisalhamento")

mem.var("tau_wd1", 2.5, "MPa", "Tensão limite para f_ck=30 MPa (NBR 6118)")
mem.verify("tau_wd <= tau_wd1", norm="NBR 6118", desc="Bielas comprimidas")

mem.write("""
Como a tensão de cisalhamento é baixa, adota-se armadura mínima de 
estribos φ5.0mm c/ 20cm.
""")

# ============================================================================
# EXPORTAR MEMORIAL
# ============================================================================

mem.save("memorial_viga_v1.pdf")
mem.save("memorial_viga_v1.html")
mem.save("memorial_viga_v1.md")

print("✅ Memorial gerado com sucesso!")
print("📄 Arquivos: memorial_viga_v1.pdf, .html, .md")
