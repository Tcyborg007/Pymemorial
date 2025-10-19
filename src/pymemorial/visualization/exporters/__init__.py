"""
PyMemorial Exporters - Figure export with intelligent fallback.

Exports figures to PNG, PDF, SVG with automatic exporter selection.
"""

from .base_exporter import (
    BaseExporter,
    ExportConfig,
    ExportError,
    ImageFormat,
)

from .cascade_exporter import (
    CascadeExporter,
    export_figure,
)

# Try to import optional exporters
try:
    from .matplotlib_exporter import MatplotlibExporter
    MATPLOTLIB_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  MatplotlibExporter not available: {e}")
    MATPLOTLIB_AVAILABLE = False
    MatplotlibExporter = None



__all__ = [
    # Main API
    'export_figure',
    'CascadeExporter',
    
    # Base classes
    'BaseExporter',
    'ExportConfig',
    'ExportError',
    'ImageFormat',
    
    # Optional exporters
    'MatplotlibExporter',
        
    # Availability flags
    'MATPLOTLIB_AVAILABLE',
    ]


def check_available_exporters():
    """Print available exporters."""
    print("Available exporters:")
    if MATPLOTLIB_AVAILABLE:
        print("  ✅ matplotlib (primary, fast)")
    else:
        print("  ❌ matplotlib (pip install matplotlib)")
    

if __name__ == "__main__":
    check_available_exporters()
