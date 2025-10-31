"""
Exemplo 2: Pilar em Flexo-Compressão
=====================================

Dimensionamento de pilar retangular submetido a esforços de compressão
e momento fletor (pequena excentricidade).

Norma: NBR 6118:2023
"""

from pymemorial import EngMemorial
import math

# Criar memorial
mem = EngMemorial(
    title="Pilar P-3 - Térreo ao 1º Pavimento",
    author="Eng. Maria Santos",
    company="Projetos & Cálculos Ltda",
    norm="NBR 6118:2023"
)

# ============================================================================
# DADOS DO PILAR
# ============================================================================

mem.section("Dados de Entrada", level=1)

mem.section("Geometria", level=2)
mem.var("b", 0.25, "m", "Largura da seção")
mem.var("h", 0.60, "m", "Altura da seção")
mem.var("L_0", 3.00, "m", "Comprimento de flambagem")

mem.section("Materiais", level=2)
mem.var("f_ck", 30, "MPa", "Resistência do concreto")
mem.var("f_yk", 500, "MPa", "Resistência do aço CA-50")

mem.section("Esforços Solicitantes", level=2)
mem.var("N_d", 850, "kN", "Força normal de cálculo")
mem.var("M1d_A", 15, "kN.m", "Momento de 1ª ordem - topo")
mem.var("M1d_B", 10, "kN.m", "Momento de 1ª ordem - base")

# ============================================================================
# VERIFICAÇÃO DE ESBELTEZ
# ============================================================================

mem.section("Verificação de Esbeltez", level=1)

mem.section("Índice de Esbeltez", level=2)
mem.calc("i = h / math.sqrt(12)", unit="m", description="Raio de giração")
mem.calc("lambda_ = L_0 / i", unit="", description="Índice de esbeltez")

mem.section("Esbeltez Limite", level=2)
mem.var("alpha_b", 0.60, "", "Coef. para M1d_A e M1d_B")
mem.var("n", 1.0, "", "Esforço normal adimensional (estimado)")

mem.calc("lambda_1 = 25 + 12.5 * (M1d_B / M1d_A)", unit="", description="Lambda 1 (NBR)")

mem.verify("lambda_ <= 90", norm="NBR 6118", desc="Esbeltez máxima")
mem.verify("lambda_ <= lambda_1", norm="NBR 6118", desc="Dispensa 2ª ordem")

# ============================================================================
# EXCENTRICIDADES
# ============================================================================

mem.section("Excentricidades", level=1)

mem.section("Excentricidades Mínimas", level=2)
mem.calc("e_a = max(0.015 + 0.03 * h, 0.02)", unit="m", description="Excent. acidental")
mem.calc("e_i = max(M1d_A / N_d, e_a)", unit="m", description="Excent. inicial")

mem.section("Momento Total de Cálculo", level=2)
mem.calc("M_d_tot = N_d * e_i", unit="kN.m", description="Momento total de projeto")

# ============================================================================
# DIMENSIONAMENTO DA ARMADURA
# ============================================================================

mem.section("Dimensionamento da Armadura", level=1)

mem.section("Parâmetros", level=2)
mem.var("d_linha", 0.04, "m", "Cobrimento + estribo + φ/2")
mem.calc("d = h - d_linha", unit="m", description="Altura útil")

mem.calc("f_cd = f_ck / 1.4", unit="MPa", description="f_cd")
mem.calc("f_yd = f_yk / 1.15", unit="MPa", description="f_yd")

mem.section("Dimensionamento (Ábacos)", level=2)
mem.write("""
Utilizando ábacos de Venturini para flexo-compressão:
- Diagrama retangular simplificado
- Seção retangular com armadura simétrica
""")

mem.calc("nu = N_d / (b * h * f_cd) * 1000", unit="", description="ν (esforço normal adim.)")
mem.calc("mu = M_d_tot / (b * h**2 * f_cd) * 1000", unit="", description="μ (momento adim.)")

mem.var("omega", 0.35, "", "ω (taxa mecânica) - lido do ábaco")
mem.calc("A_s = omega * b * h * f_cd / f_yd * 10000", unit="cm²", description="Área de aço")

mem.section("Armadura Mínima", level=2)
mem.calc("A_s_min = max(0.004 * b * h * 10000, 4 * 1.25)", unit="cm²", description="A_s,mín (NBR)")

mem.verify("A_s >= A_s_min", norm="NBR 6118", desc="Armadura mínima")

mem.write("""
**Armadura longitudinal adotada:** 8φ16mm (A_s = 16.08 cm²)

**Armadura transversal:** φ5.0mm c/ 15cm (região de emenda e nós)
                         φ5.0mm c/ 20cm (restante)
""")

# ============================================================================
# EXPORTAR
# ============================================================================

mem.save("memorial_pilar_p3.pdf")
print("✅ Memorial de pilar gerado!")
