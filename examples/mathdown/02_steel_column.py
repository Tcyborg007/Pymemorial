"""
Example 2: Steel Column Design - NBR 8800:2024

Demonstrates:
- Professional engineering calculations
- Multi-line equations
- Greek letters (λ, χ, σ)
- Engineering units (kN, MPa, cm²)
- Code references
"""

from pymemorial.mathdown import MathDownParser

mathdown_text = """
# Memorial de Cálculo - Pilar Metálico W310x107

**Projeto:** Edifício Comercial São Paulo  
**Norma:** NBR 8800:2024  
**Data:** 2025-10-18

---

## 1. Dados da Seção

Perfil estrutural:

#$ Perfil = W310x107 $#

Propriedades geométricas:

#$ A_g = 137 cm² $#

#$ r_x = 13.6 cm $#

#$ Z_x = 1790 cm³ $#

## 2. Propriedades do Material

Aço ASTM A572 Gr. 50:

#$ f_y = 345 MPa $#

#$ E = 200000 MPa $#

#$ γ_a = 1.1 $#

## 3. Solicitações de Cálculo

Esforços atuantes:

#$ N_d = 1200 kN $#

#$ M_d = 150 kN·m $#

#$ L = 4.0 m $#

## 4. Índice de Esbeltez

#$ sabemos que o índice de esbeltez é definido por 
   => λ = L/r_x $#

Calculando:

#$ λ = 4.0/0.136 $#

#$ λ = 29.4 $#

**Referência:** NBR 8800:2024 - Item 5.3.2

Como #$ λ < 100 $#, o pilar é considerado **robusto**.

## 5. Fator de Redução

Para #$ λ < 100 $#, adota-se:

#$ χ = 0.877 $#

**Referência:** NBR 8800:2024 - Tabela 5.3

## 6. Capacidade Axial

#$ sabemos que a força axial resistente é calculada por 
   => N_Rd = χ*A_g*f_y/γ_a $#

Substituindo os valores:

#$ N_Rd = 0.877*137*345/1.1 $#

#$ N_Rd = 3777 kN $#

**Referência:** NBR 8800:2024 - Item 5.3.3

## 7. Capacidade Flexional

#$ sabemos que o momento resistente é dado por 
   => M_Rd = f_y*Z_x/γ_a $#

Convertendo unidades (#$ Z_x $# de cm³ para m³):

#$ Z_x = 0.001790 m³ $#

Calculando:

#$ M_Rd = 345*0.001790/1.1 $#

#$ M_Rd = 561.8 kN·m $#

**Referência:** NBR 8800:2024 - Item 5.4.2

## 8. Verificação de Utilização

Razão de utilização considerando interação P-M:

#$ sabemos que para interação usa-se => η = ((M_d/M_Rd)**2 + (N_d/N_Rd)**2)**0.5 $#

Calculando cada termo:

#$ M_d/M_Rd = 150/561.8 $#

#$ M_d/M_Rd = 0.267 $#

#$ N_d/N_Rd = 1200/3777 $#

#$ N_d/N_Rd = 0.318 $#

Razão de utilização:

#$ η = (0.267**2 + 0.318**2)**0.5 $#

#$ η = 0.415 $#

#$ η = 41.5% $#

## 9. Conclusão

Verificações:

#$ η < 1.0 $# ✓ **APROVADO**

#$ λ < 100 $# ✓ **ROBUSTO**

O pilar **W310x107** está adequadamente dimensionado conforme NBR 8800:2024, 
com utilização de **41.5%** de sua capacidade resistente.

**Margem de segurança:** #$ 1.0 - η = 0.585 = 58.5% $#
"""

# ============================================================================
# PARSE AND RENDER
# ============================================================================

parser = MathDownParser()
result = parser.parse(mathdown_text)

# Print summary
print("\n" + "=" * 80)
print("STEEL COLUMN ANALYSIS - SUMMARY")
print("=" * 80)
print(result.summary())
print()

# Variables detected
print("=" * 80)
print("VARIABLES DETECTED")
print("=" * 80)
variables = result.get_variables()
for name in sorted(variables.keys()):
    var = variables[name]
    if var.value is not None:
        print(f"  {name:12s} = {var.value:12.4f}  {var.unit or ''}")
print()

# Export
html_output = "outputs/02_steel_column.html"
result.to_html(html_output)

print("=" * 80)
print(f"✓ Professional HTML report generated: {html_output}")
print("=" * 80)
