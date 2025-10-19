"""
Example 1: Basic MathDown Usage - Simply Supported Beam

Demonstrates:
- Natural math notation with #$...$#
- Variable detection
- Equation parsing
- Automatic computation
- HTML export
"""

from pymemorial.mathdown import MathDownParser

# ============================================================================
# MATHDOWN TEXT - Write naturally!
# ============================================================================

mathdown_text = """
# Memorial de Cálculo - Viga Biapoiada

## 1. Dados do Problema

Considere uma viga biapoiada submetida a uma carga uniformemente distribuída.

Os dados de entrada são:

#$ q = 10 kN/m $#

#$ L = 6 m $#

## 2. Cálculo do Momento Máximo

#$ sabemos que o momento fletor máximo (M_max) em uma viga biapoiada 
   é definido pela fórmula => M_max = q*L^2/8 $#


Substituindo os valores fornecidos:

#$ M_max = 10*6**2/8 $#

#$ M_max = 45 kN·m $#

## 3. Cálculo da Tensão

Considere uma seção com módulo de resistência:

#$ W = 0.002 m³ $#

A tensão normal é calculada por:

#$ sabemos que a tensão é dada pela relação momento sobre módulo 
   => σ = M_max/W $#

Substituindo:

#$ σ = 45/0.002 $#

#$ σ = 22500 kPa $#

#$ σ = 22.5 MPa $#

## 4. Verificação

A tensão de escoamento do aço é:

#$ f_y = 250 MPa $#

Comparando com a tensão atuante:

#$ σ < f_y $#

**Conclusão**: Como #$ σ < f_y $#, a viga está **APROVADA**.

A razão de utilização é:

#$ η = σ/f_y $#

#$ η = 22.5/250 $#

#$ η = 0.09 $#

Portanto, a viga está trabalhando a apenas **9%** de sua capacidade.
"""

# ============================================================================
# PARSE AND RENDER
# ============================================================================

# Create parser
parser = MathDownParser()

# Parse the text
result = parser.parse(mathdown_text)

# Print summary
print("=" * 80)
print("MATHDOWN PARSING SUMMARY")
print("=" * 80)
print(result.summary())
print()

# Print detected variables
print("=" * 80)
print("DETECTED VARIABLES")
print("=" * 80)
for name, var in result.get_variables().items():
    print(f"  {name:10s} = {var.value:>10.2f} {var.unit or ''}")
print()

# Print equations
print("=" * 80)
print("DETECTED EQUATIONS")
print("=" * 80)
for i, eq in enumerate(result.get_equations(), 1):
    print(f"{i}. {eq.description}")
    print(f"   LaTeX: {eq.latex}")
    print()

# Export to HTML
from pathlib import Path

output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)  # Create directory if not exists

html_output = output_dir / "01_basic_beam.html"
result.to_html(str(html_output))

print("=" * 80)
print(f"✓ HTML exported to: {html_output}")
print("✓ Open in browser to see beautifully rendered equations!")
print("=" * 80)
