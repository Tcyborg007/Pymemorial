import sys
sys.path.insert(0, 'src')
try:
    from pymemorial.visualization.pyvista_engine import PyVistaEngine, PYVISTA_AVAILABLE
    print('✓ Import OK')
    print(f'PYVISTA_AVAILABLE = {PYVISTA_AVAILABLE}')
except Exception as e:
    print(f'✗ Import FAILED: {e}')
    import traceback
    traceback.print_exc()
