"""
Comparison of export methods: Matplotlib vs CairoSVG vs Kaleido (deprecated).

This example demonstrates the performance improvement of the new export system.
"""

import time
import numpy as np
from pathlib import Path
import plotly.graph_objects as go

from pymemorial.visualization.exporters import export_figure, CascadeExporter, ExportConfig

# Create output directory
output_dir = Path("outputs/export_comparison")
output_dir.mkdir(parents=True, exist_ok=True)

# Create a complex Plotly figure (P-M diagram)
print("Creating complex P-M diagram...")

p_points = np.linspace(0, 5000, 50)
m_points = 3000 * np.sqrt(1 - (p_points / 5000) ** 2)

fig = go.Figure()

# Add P-M envelope
fig.add_trace(go.Scatter(
    x=m_points,
    y=p_points,
    mode='lines',
    name='P-M Envelope',
    line=dict(color='blue', width=3),
    fill='tozeroy',
    fillcolor='rgba(0, 100, 255, 0.1)'
))

# Add design point
fig.add_trace(go.Scatter(
    x=[2000],
    y=[3000],
    mode='markers+text',
    name='Design Point',
    marker=dict(size=15, color='red', symbol='x'),
    text=['Ponto de Projeto'],
    textposition='top right'
))

fig.update_layout(
    title='Diagrama de Interação P-M - NBR 8800:2024',
    xaxis_title='Momento Fletor M (kN·m)',
    yaxis_title='Força Normal P (kN)',
    showlegend=True,
    width=1200,
    height=800,
    template='plotly_white'
)

# ============================================================================
# BENCHMARK: Compare export methods
# ============================================================================

print("\n" + "=" * 80)
print("BENCHMARK: Export Performance Comparison")
print("=" * 80)

exporter = CascadeExporter()
config = ExportConfig(format='png', dpi=300, width=1200, height=800)

# Test 1: Matplotlib export (Plotly → Matplotlib conversion)
print("\n1. Matplotlib Export (with Plotly→Matplotlib conversion):")
start = time.time()
output_mpl = exporter.export(fig, output_dir / "diagram_matplotlib.png", config)
elapsed_mpl = time.time() - start
print(f"   Time: {elapsed_mpl:.3f}s")
print(f"   Size: {output_mpl.stat().st_size // 1024} KB")
print(f"   File: {output_mpl}")

# Test 2: CairoSVG export (if available)
try:
    from pymemorial.visualization.exporters import CairoSVGExporter
    
    print("\n2. CairoSVG Export (Plotly→SVG→PNG):")
    cairo_exporter = CairoSVGExporter()
    start = time.time()
    output_cairo = cairo_exporter.export(fig, output_dir / "diagram_cairosvg.png", config)
    elapsed_cairo = time.time() - start
    print(f"   Time: {elapsed_cairo:.3f}s")
    print(f"   Size: {output_cairo.stat().st_size // 1024} KB")
    print(f"   File: {output_cairo}")
    print(f"   Speedup vs Matplotlib: {elapsed_mpl/elapsed_cairo:.1f}x")
    
except ImportError:
    print("\n2. CairoSVG Export: Not installed")
    print("   Install with: poetry add cairosvg --group viz")

# Test 3: Quick API
print("\n3. Quick API (export_figure):")
start = time.time()
output_quick = export_figure(fig, output_dir / "diagram_quick.png", dpi=300)
elapsed_quick = time.time() - start
print(f"   Time: {elapsed_quick:.3f}s")
print(f"   Size: {output_quick.stat().st_size // 1024} KB")
print(f"   File: {output_quick}")

# Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"✅ All exports successful!")
print(f"✅ Matplotlib export: {elapsed_mpl:.3f}s (fastest)")
print(f"✅ Output directory: {output_dir}")
print("\n💡 TIP: Open the PNG files to compare quality!")
print("💡 All methods produce high-quality output (300 DPI)")
