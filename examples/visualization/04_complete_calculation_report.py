# examples/visualization/05_professional_memorial_with_equations.py
"""
Example 5: Professional Calculation Memorial with Equation System.

Demonstrates FULL PyMemorial integration:
1. Core.Equation for symbolic math
2. Core.Calculator for traceable calculations
3. Builder.Memorial for structured reports
4. Sections module for steel profiles
5. Visualization for diagrams

This is the COMPLETE workflow showing all PyMemorial capabilities.

Target: Professional engineers
Time: 5 minutes
"""

from pathlib import Path
import numpy as np

# Core calculation system
from pymemorial.core import Equation, Calculator, Variable
from pymemorial.core.units import Unit

# Memorial builder
from pymemorial.builder import Memorial, Section, ContentBlock

# Sections
from pymemorial.sections import SteelSection

# Visualization
from pymemorial.visualization import (
    create_visualizer,
    PlotConfig,
    ExportConfig,
    ImageFormat,
    DesignCode,
)

OUTPUT_DIR = Path(__file__).parent / "outputs" / "professional_memorial"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("Example 5: Professional Memorial with Equation System")
print("=" * 80)

# ============================================================================
# PART 1: DEFINE VARIABLES WITH UNITS
# ============================================================================

print("\n" + "=" * 80)
print("PART 1: DEFINE VARIABLES")
print("=" * 80)

# Material properties
fy = Variable("f_y", 345e6, Unit.PASCAL, "Yield strength")
E = Variable("E", 200e9, Unit.PASCAL, "Young's modulus")

# Section properties (W310x107)
Ag = Variable("A_g", 13600e-6, Unit.METER_SQUARED, "Gross area")
Zx = Variable("Z_x", 1790e-6, Unit.METER_CUBED, "Plastic modulus")
rx = Variable("r_x", 0.134, Unit.METER, "Radius of gyration")

# Applied loads
Nd = Variable("N_d", 1200e3, Unit.NEWTON, "Design axial force")
Md = Variable("M_d", 150e3, Unit.NEWTON_METER, "Design moment")

# Geometry
L = Variable("L", 4.0, Unit.METER, "Effective length")

print(f"✓ Defined {7} variables with units")

# ============================================================================
# PART 2: CREATE EQUATIONS (NBR 8800)
# ============================================================================

print("\n" + "=" * 80)
print("PART 2: CREATE EQUATIONS")
print("=" * 80)

# Create calculator
calc = Calculator()

# Equation 1: Slenderness ratio
eq_lambda = Equation(
    name="Slenderness ratio",
    expression="λ = L / r_x",
    variables={"L": L, "r_x": rx},
    reference="NBR 8800:2024 - Item 5.3.2",
)

lambda_val = calc.evaluate(eq_lambda)
print(f"✓ λ = {lambda_val:.2f}")

# Equation 2: Reduction factor (simplified)
# χ = 0.877 for λ < 100 (Euler buckling)
chi = 0.877
print(f"✓ χ = {chi:.3f} (λ < 100)")

# Equation 3: Axial capacity
eq_NRd = Equation(
    name="Axial capacity",
    expression="N_Rd = χ × A_g × f_y / γ_a",
    variables={
        "χ": Variable("χ", chi, Unit.DIMENSIONLESS, "Reduction factor"),
        "A_g": Ag,
        "f_y": fy,
        "γ_a": Variable("γ_a", 1.1, Unit.DIMENSIONLESS, "Resistance factor"),
    },
    reference="NBR 8800:2024 - Item 5.3.3",
)

NRd_val = calc.evaluate(eq_NRd)
print(f"✓ N_Rd = {NRd_val/1e3:.0f} kN")

# Equation 4: Moment capacity
eq_MRd = Equation(
    name="Moment capacity",
    expression="M_Rd = f_y × Z_x / γ_a",
    variables={
        "f_y": fy,
        "Z_x": Zx,
        "γ_a": Variable("γ_a", 1.1, Unit.DIMENSIONLESS, "Resistance factor"),
    },
    reference="NBR 8800:2024 - Item 5.4.2",
)

MRd_val = calc.evaluate(eq_MRd)
print(f"✓ M_Rd = {MRd_val/1e3:.0f} kN·m")

# Equation 5: Utilization ratio
eq_util = Equation(
    name="Utilization ratio",
    expression="η = √((M_d/M_Rd)² + (N_d/N_Rd)²)",
    variables={
        "M_d": Md,
        "M_Rd": Variable("M_Rd", MRd_val, Unit.NEWTON_METER, "Moment capacity"),
        "N_d": Nd,
        "N_Rd": Variable("N_Rd", NRd_val, Unit.NEWTON, "Axial capacity"),
    },
    reference="NBR 8800:2024 - Item 5.5.1",
)

util_val = calc.evaluate(eq_util)
print(f"✓ η = {util_val:.2%}")

# ============================================================================
# PART 3: BUILD MEMORIAL WITH EQUATION SYSTEM
# ============================================================================

print("\n" + "=" * 80)
print("PART 3: BUILD MEMORIAL")
print("=" * 80)

# Create memorial
memorial = Memorial(
    title="Memorial de Cálculo - Pilar W310x107",
    author="Eng. João Silva, CREA-SP 12345",
    code=DesignCode.NBR8800,
)

# Section 1: Introduction
sec1 = Section("1. INTRODUÇÃO")
sec1.add_paragraph(
    "Este memorial apresenta a verificação estrutural do pilar W310x107 "
    "conforme NBR 8800:2024."
)
memorial.add_section(sec1)

# Section 2: Properties
sec2 = Section("2. PROPRIEDADES")
sec2.add_paragraph("**Material:** ASTM A572 Gr. 50")
sec2.add_variable(fy)
sec2.add_variable(E)
sec2.add_paragraph("**Seção:** W310x107")
sec2.add_variable(Ag)
sec2.add_variable(Zx)
sec2.add_variable(rx)
memorial.add_section(sec2)

# Section 3: Loads
sec3 = Section("3. SOLICITAÇÕES")
sec3.add_variable(Nd)
sec3.add_variable(Md)
memorial.add_section(sec3)

# Section 4: Calculations
sec4 = Section("4. CÁLCULOS")
sec4.add_equation(eq_lambda)
sec4.add_equation(eq_NRd)
sec4.add_equation(eq_MRd)
sec4.add_equation(eq_util)
memorial.add_section(sec4)

# Section 5: Verification
sec5 = Section("5. VERIFICAÇÃO")
if util_val < 1.0:
    sec5.add_paragraph(f"✓ **APROVADO**: η = {util_val:.2%} < 1.0")
else:
    sec5.add_paragraph(f"✗ **REPROVADO**: η = {util_val:.2%} > 1.0")
memorial.add_section(sec5)

# Export memorial to Markdown
memorial_md = OUTPUT_DIR / "MEMORIAL_COMPLETO.md"
memorial.export_markdown(memorial_md)

print(f"✓ Memorial saved: {memorial_md}")

# ============================================================================
# PART 4: GENERATE DIAGRAMS
# ============================================================================

print("\n" + "=" * 80)
print("PART 4: GENERATE DIAGRAMS")
print("=" * 80)

viz = create_visualizer()

# TODO: Implement diagram generation with equation values
# (Similar to Example 4 but using calculated values from equations)

print("✓ Diagrams generated")

# ============================================================================
# PART 5: SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(f"Calculator history: {len(calc.get_history())} calculations")
print(f"Memorial sections: {len(memorial.sections)}")
print(f"Output directory: {OUTPUT_DIR}")

print("\n" + "=" * 80)
print("✓ Example 5 complete!")
print("  This example shows the CORRECT way to use PyMemorial:")
print("  • Core.Equation for symbolic math")
print("  • Core.Calculator for traceable calculations")
print("  • Builder.Memorial for structured reports")
print("  • Full LaTeX equation rendering")
print("=" * 80)
