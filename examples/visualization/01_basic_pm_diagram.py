# examples/visualization/01_basic_pm_diagram.py
"""
Example 1: Basic P-M Interaction Diagram.

This example demonstrates the simplest workflow:
1. Create visualizer
2. Generate data
3. Create diagram
4. Export to file

Target: Beginners
Time: 30 seconds
"""

from pathlib import Path
import sys

# Add src to path for direct execution
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
from pymemorial.visualization import (
    create_visualizer,
    PlotConfig,
    ExportConfig,
    ImageFormat,
)

# ============================================================================
# 1. SETUP OUTPUT DIRECTORY
# ============================================================================

OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 70)
print("Example 1: Basic P-M Interaction Diagram")
print("=" * 70)

try:
    # ============================================================================
    # 2. CREATE VISUALIZER (auto-selects best available)
    # ============================================================================

    viz = create_visualizer()
    print(f"\n✓ Using engine: {viz.name} v{viz.version}")

    # ============================================================================
    # 3. GENERATE P-M INTERACTION DATA
    # ============================================================================

    # Normalized P-M envelope (typical steel column)
    p = np.array([0.0, 0.2, 0.5, 0.8, 1.0, 0.8, 0.5, 0.2, 0.0])
    m = np.array([1.0, 0.95, 0.85, 0.6, 0.0, -0.3, -0.5, -0.7, -0.8])

    # Design point (applied loads)
    design_point = (0.4, 0.6)  # (M/Mn, P/Pn)

    print(
        f"✓ Data: {len(p)} points, design point at M={design_point[0]}, P={design_point[1]}"
    )

    # ============================================================================
    # 4. CREATE DIAGRAM
    # ============================================================================

    config = PlotConfig(
        title="P-M Interaction Diagram - Steel Column",
        xlabel="M/Mn (Normalized Moment)",
        ylabel="P/Pn (Normalized Axial Load)",
        width=800,
        height=600,
    )

    fig = viz.create_pm_diagram(p, m, design_point=design_point, config=config)

    print("✓ Diagram created")

    # ============================================================================
    # 5. EXPORT TO FILE
    # ============================================================================

    # Export PNG
    png_path = viz.export_static(
        fig,
        ExportConfig(
            filename=OUTPUT_DIR / "pm_diagram_basic.png",
            format=ImageFormat.PNG,
            scale=2.0,  # High resolution
        ),
    )

    print(f"✓ Exported PNG: {png_path}")

    # Export HTML (interactive)
    html_path = viz.export_static(
        fig,
        ExportConfig(
            filename=OUTPUT_DIR / "pm_diagram_basic.html",
            format=ImageFormat.HTML,
        ),
    )

    print(f"✓ Exported HTML: {html_path}")

    print("\n" + "=" * 70)
    print("✓ Example 1 complete!")
    print(f"  Check outputs in: {OUTPUT_DIR}")
    print("=" * 70)

    # Exit success
    sys.exit(0)

except Exception as e:
    print(f"\n✗ ERROR: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
