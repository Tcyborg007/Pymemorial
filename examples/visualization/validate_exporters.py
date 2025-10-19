"""
Quick validation script for simplified exporter system.

Tests:
1. Import chain (no errors)
2. CascadeExporter initialization
3. export_figure() function
4. PlotlyEngine.export() method
5. Quick export test
"""

import sys
from pathlib import Path

print("=" * 80)
print("EXPORTER SYSTEM VALIDATION")
print("=" * 80)

# ============================================================================
# TEST 1: Import Chain
# ============================================================================

print("\n📦 TEST 1: Import Chain")
print("-" * 80)

try:
    print("  Importing base_exporter...", end=" ")
    from pymemorial.visualization.exporters.base_exporter import (
        BaseExporter,
        ExportConfig,
        ExportError,
        ImageFormat
    )
    print("✅")
except Exception as e:
    print(f"❌ {e}")
    sys.exit(1)

try:
    print("  Importing matplotlib_exporter...", end=" ")
    from pymemorial.visualization.exporters.matplotlib_exporter import (
        MatplotlibExporter
    )
    print("✅")
except Exception as e:
    print(f"❌ {e}")
    sys.exit(1)

try:
    print("  Importing cascade_exporter...", end=" ")
    from pymemorial.visualization.exporters.cascade_exporter import (
        CascadeExporter,
        export_figure
    )
    print("✅")
except Exception as e:
    print(f"❌ {e}")
    sys.exit(1)

try:
    print("  Importing exporters/__init__...", end=" ")
    from pymemorial.visualization.exporters import (
        export_figure,
        CascadeExporter,
        MatplotlibExporter,
        ExportConfig,
        MATPLOTLIB_AVAILABLE
    )
    print("✅")
except Exception as e:
    print(f"❌ {e}")
    sys.exit(1)

print("\n✅ All imports successful!")

# ============================================================================
# TEST 2: Check Available Exporters
# ============================================================================

print("\n📊 TEST 2: Available Exporters")
print("-" * 80)

exporter = CascadeExporter()
available = exporter.get_available_exporters()

print(f"  Available: {available}")

if 'matplotlib' not in available:
    print("  ❌ Matplotlib exporter not found!")
    sys.exit(1)

if len(available) > 1:
    print(f"  ⚠️  WARNING: More than 1 exporter found! Expected only 'matplotlib'")
    print(f"     Found: {available}")

print("  ✅ Matplotlib exporter available")

# ============================================================================
# TEST 3: Create Test Figure
# ============================================================================

print("\n🎨 TEST 3: Create Test Figure")
print("-" * 80)

try:
    import matplotlib.pyplot as plt
    import numpy as np
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.linspace(0, 10, 100)
    ax.plot(x, np.sin(x), label='sin(x)', linewidth=2)
    ax.plot(x, np.cos(x), label='cos(x)', linewidth=2)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Test Figure')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    print("  ✅ Matplotlib figure created")
except Exception as e:
    print(f"  ❌ Failed: {e}")
    sys.exit(1)

# ============================================================================
# TEST 4: Export with CascadeExporter
# ============================================================================

print("\n💾 TEST 4: CascadeExporter.export()")
print("-" * 80)

output_dir = Path("outputs/validation")
output_dir.mkdir(parents=True, exist_ok=True)

try:
    config = ExportConfig(format='png', dpi=300)
    output = exporter.export(fig, output_dir / "test_cascade.png", config)
    
    if not output.exists():
        print(f"  ❌ File not created: {output}")
        sys.exit(1)
    
    size_kb = output.stat().st_size // 1024
    print(f"  ✅ Export successful: {output.name} ({size_kb} KB)")
except Exception as e:
    print(f"  ❌ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# TEST 5: export_figure() convenience function
# ============================================================================

print("\n🚀 TEST 5: export_figure() function")
print("-" * 80)

try:
    output = export_figure(
        fig,
        output_dir / "test_convenience.png",
        format="png",
        dpi=300
    )
    
    if not output.exists():
        print(f"  ❌ File not created: {output}")
        sys.exit(1)
    
    size_kb = output.stat().st_size // 1024
    print(f"  ✅ Export successful: {output.name} ({size_kb} KB)")
except Exception as e:
    print(f"  ❌ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# TEST 6: PlotlyEngine.export() method
# ============================================================================

print("\n🎯 TEST 6: PlotlyEngine.export() method")
print("-" * 80)

try:
    print("  Importing PlotlyEngine...", end=" ")
    from pymemorial.visualization.plotly_engine import PlotlyEngine
    print("✅")
except Exception as e:
    print(f"❌ {e}")
    sys.exit(1)

try:
    print("  Creating Plotly figure...", end=" ")
    import plotly.graph_objects as go
    
    fig_plotly = go.Figure()
    fig_plotly.add_trace(go.Scatter(
        x=[1, 2, 3, 4],
        y=[10, 15, 13, 17],
        mode='lines+markers',
        name='Test Data'
    ))
    fig_plotly.update_layout(title='Test Plotly Figure')
    print("✅")
except Exception as e:
    print(f"❌ {e}")
    sys.exit(1)

try:
    print("  Checking PlotlyEngine.export() method...", end=" ")
    engine = PlotlyEngine()
    
    if not hasattr(engine, 'export'):
        print("❌ Method not found!")
        print("     PlotlyEngine needs export() method!")
        sys.exit(1)
    
    print("✅")
except Exception as e:
    print(f"❌ {e}")
    sys.exit(1)

try:
    print("  Exporting via PlotlyEngine.export()...", end=" ")
    output = engine.export(
        fig_plotly,
        output_dir / "test_plotly_engine.png",
        format="png",
        dpi=300
    )
    
    if not output.exists():
        print(f"❌ File not created!")
        sys.exit(1)
    
    size_kb = output.stat().st_size // 1024
    print(f"✅ ({size_kb} KB)")
except Exception as e:
    print(f"❌ {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("✅ ALL TESTS PASSED!")
print("=" * 80)
print(f"\nOutput files saved to: {output_dir}")
print("\n📊 Summary:")
print("  ✅ Import chain working")
print("  ✅ CascadeExporter available (matplotlib only)")
print("  ✅ export_figure() working")
print("  ✅ PlotlyEngine.export() working")
print("\n🎉 Exporter system is fully functional!")

plt.close('all')
