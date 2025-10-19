# examples/visualization/02_moment_curvature_nbr8800.py
"""
Example 2: Moment-Curvature Analysis with NBR 8800 Code.

Demonstrates:
- Diagram generators with Brazilian code
- Ductility calculation
- Publication-quality styling
- Multiple exports

Target: Intermediate users
Time: 1 minute
"""

from pathlib import Path
import numpy as np
from pymemorial.visualization import (
    create_visualizer,
    PlotConfig,
    ExportConfig,
    ImageFormat,
    DesignCode,
    generate_moment_curvature_response,
    calculate_ductility,
    MomentCurvatureParams,
)

OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 70)
print("Example 2: Moment-Curvature with NBR 8800")
print("=" * 70)

# ============================================================================
# 1. DEFINE SECTION PROPERTIES (W360x122 - typical beam)
# ============================================================================

# Material properties
fy = 345e6  # Pa (ASTM A572 Gr. 50)
E = 200e9  # Pa

# Section properties
Zx = 2260e-6  # m³ (plastic modulus)
Ix = 403e-6  # m⁴ (moment of inertia)
d = 0.363  # m (depth)

# Calculate characteristic points
M_yield = fy * Zx / 1.1  # kN·m (NBR 8800 - γa = 1.1)
M_plastic = fy * Zx  # kN·m
kappa_yield = (fy / E) / (d / 2)  # 1/m
kappa_ultimate = 10 * kappa_yield  # Assume ductile failure

print(f"\n✓ Section: W360x122")
print(f"  My = {M_yield/1e3:.1f} kN·m")
print(f"  Mp = {M_plastic/1e3:.1f} kN·m")
print(f"  κy = {kappa_yield:.5f} 1/m")

# ============================================================================
# 2. GENERATE MOMENT-CURVATURE RESPONSE
# ============================================================================

params = MomentCurvatureParams(
    n_steps=100,
    mark_yield=True,
    mark_ultimate=True,
    calculate_ductility=True,
)

curvature, moment = generate_moment_curvature_response(
    m_yield=M_yield,
    m_ultimate=M_plastic,
    kappa_yield=kappa_yield,
    kappa_ultimate=kappa_ultimate,
    params=params,
)

# Convert to kN·m
moment_kn = moment / 1e3

print(f"✓ Generated {len(curvature)} analysis points")

# ============================================================================
# 3. CALCULATE DUCTILITY
# ============================================================================

ductility = calculate_ductility(kappa_yield, kappa_ultimate)

print(f"\n✓ Ductility Analysis:")
print(f"  μ = {ductility['mu']:.2f}")
print(f"  Classification: {ductility['classification']}")
print(f"  Adequate per NBR 8800: {ductility['adequate']}")

# ============================================================================
# 4. CREATE PUBLICATION-QUALITY DIAGRAM
# ============================================================================

viz = create_visualizer()

config = PlotConfig(
    title="Diagrama Momento-Curvatura (NBR 8800:2024)",
    xlabel="Curvatura κ (1/m)",
    ylabel="Momento Fletor M (kN·m)",
    width=1000,
    height=700,
    dpi=300,  # Publication quality
    font_size=11,
    grid=True,
)

# Mark yield and plastic points
yield_point = (kappa_yield, M_yield / 1e3)
ultimate_point = (kappa_ultimate, M_plastic / 1e3)

fig = viz.create_moment_curvature(
    curvature,
    moment_kn,
    yield_point=yield_point,
    ultimate_point=ultimate_point,
    config=config,
)

print("✓ Diagram created")

# ============================================================================
# 5. EXPORT MULTIPLE FORMATS
# ============================================================================

# For paper/thesis (PDF)
pdf_path = viz.export_static(
    fig,
    ExportConfig(
        filename=OUTPUT_DIR / "moment_curvature_nbr8800.pdf",
        format=ImageFormat.PDF,
    ),
)
print(f"✓ Exported PDF: {pdf_path}")

# For presentation (PNG)
png_path = viz.export_static(
    fig,
    ExportConfig(
        filename=OUTPUT_DIR / "moment_curvature_nbr8800.png",
        format=ImageFormat.PNG,
        scale=2.0,
    ),
)
print(f"✓ Exported PNG: {png_path}")

# For web (HTML interactive)
html_path = viz.export_static(
    fig,
    ExportConfig(
        filename=OUTPUT_DIR / "moment_curvature_nbr8800.html",
        format=ImageFormat.HTML,
    ),
)
print(f"✓ Exported HTML: {html_path}")

print("\n" + "=" * 70)
print("✓ Example 2 complete!")
print(f"  Ductility: μ = {ductility['mu']:.2f} ({ductility['classification']})")
print(f"  Files in: {OUTPUT_DIR}")
print("=" * 70)
