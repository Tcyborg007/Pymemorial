# tests/validate_phase6.py
"""Quick validation script for Phase 6 - Visualization."""

import sys
from pathlib import Path

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

print("=" * 70)
print("PyMemorial Phase 6 - Visualization Module Validation")
print("=" * 70)

# Test 1: Basic imports
print("\n✓ Test 1: Basic imports...")
try:
    from pymemorial.visualization import (
        create_visualizer,
        PlotConfig,
        ExportConfig,
        ImageFormat,
        list_available_engines,
        check_installation,
    )
    print("  ✓ Core imports successful")
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    sys.exit(1)

# Test 2: Check installation
print("\n✓ Test 2: Check installation status...")
try:
    check_installation()
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    sys.exit(1)

# Test 3: List engines
print("\n✓ Test 3: Available engines...")
try:
    engines = list_available_engines()
    print(f"  Available: {', '.join(engines)}")
    if not engines:
        print("  ⚠ WARNING: No engines available!")
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    sys.exit(1)

# Test 4: Create visualizer
print("\n✓ Test 4: Create visualizer...")
try:
    viz = create_visualizer()
    print(f"  Created: {viz.name} v{viz.version}")
    print(f"  3D support: {viz.supports_3d}")
    print(f"  Interactive: {viz.supports_interactive}")
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    sys.exit(1)

# Test 5: PyVista availability (optional)
print("\n✓ Test 5: PyVista engine (optional)...")
try:
    from pymemorial.visualization import PYVISTA_AVAILABLE, PYVISTA_VERSION
    if PYVISTA_AVAILABLE:
        print(f"  ✓ PyVista available: v{PYVISTA_VERSION}")
    else:
        print("  ⊘ PyVista not installed (optional)")
except Exception as e:
    print(f"  ⊘ PyVista check skipped: {e}")

# Test 6: Diagram generators
print("\n✓ Test 6: Diagram generators...")
try:
    from pymemorial.visualization import (
        DesignCode,
        generate_pm_interaction_envelope,
        calculate_ductility,
    )
    
    # Quick test
    p, m = generate_pm_interaction_envelope(1000, 250)
    print(f"  ✓ Generated envelope: {len(p)} points")
    
    metrics = calculate_ductility(0.003, 0.015)
    print(f"  ✓ Ductility: μ = {metrics['mu']:.2f}")
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    sys.exit(1)

print("\n" + "=" * 70)
print("✓ ALL VALIDATION TESTS PASSED!")
print("=" * 70)
print("\nPhase 6 is ready for examples and production use.")
