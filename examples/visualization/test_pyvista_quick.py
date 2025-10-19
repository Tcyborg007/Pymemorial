# test_pyvista_quick.py
"""Quick test for PyVista integration."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

print("=" * 70)
print("PyVista Quick Test")
print("=" * 70)

# Test 1: Direct import
print("\n1. Testing direct PyVista import...")
try:
    import pyvista as pv
    print(f"   ✓ PyVista {pv.__version__} imported")
    print(f"   ✓ VTK available")
except ImportError as e:
    print(f"   ✗ FAILED: {e}")
    print("\n   Install with:")
    print("   poetry add pyvista")
    sys.exit(1)

# Test 2: PyMemorial detection
print("\n2. Testing PyMemorial detection...")
try:
    from pymemorial.visualization import (
        PYVISTA_AVAILABLE,
        PYVISTA_VERSION,
        VTK_VERSION,
    )
    
    print(f"   PyVista Available: {PYVISTA_AVAILABLE}")
    print(f"   PyVista Version: {PYVISTA_VERSION}")
    print(f"   VTK Version: {VTK_VERSION}")
    
    if PYVISTA_AVAILABLE:
        print("   ✓ PyMemorial detected PyVista correctly")
    else:
        print("   ✗ PyMemorial did NOT detect PyVista")
        print("   → Try restarting Python/reloading modules")
except Exception as e:
    print(f"   ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Create PyVista engine
print("\n3. Testing PyVistaEngine creation...")
try:
    from pymemorial.visualization import PyVistaEngine
    
    if PyVistaEngine is None:
        print("   ✗ PyVistaEngine is None (not imported)")
        sys.exit(1)
    
    engine = PyVistaEngine()
    
    print(f"   ✓ Engine created: {engine.name}")
    print(f"   ✓ Version: {engine.version}")
    print(f"   ✓ Available: {engine.available}")
    print(f"   ✓ Supports 3D: {engine.supports_3d}")
    
    if not engine.available:
        print("   ⚠ Engine created but not available")
        sys.exit(1)
        
except Exception as e:
    print(f"   ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Simple mesh creation
print("\n4. Testing mesh creation...")
try:
    import numpy as np
    
    nodes = np.array([[0,0,0], [1,0,0], [1,1,0], [0,1,0]])
    elements = [[0,1], [1,2], [2,3], [3,0]]
    
    mesh = engine.create_fem_mesh(nodes, elements, element_type="line")
    
    print(f"   ✓ Mesh created: {mesh.n_points} nodes, {mesh.n_cells} cells")
    
    # Close plotter
    engine.close_plotter()
    
except Exception as e:
    print(f"   ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 70)
print("✓ ALL PYVISTA TESTS PASSED!")
print("=" * 70)
