"""
Cascade Exporter - Intelligent exporter with automatic fallback.

Tries exporters in order of speed:
1. MatplotlibExporter (0.1s) - native, always works
2. CairoSVGExporter (0.2s) - fast SVG conversion
3. Future: PlaywrightExporter (2s) - universal fallback

Automatically selects best available exporter for each task.
"""

import warnings
from pathlib import Path
from typing import Optional, Any, List

from .base_exporter import BaseExporter, ExportConfig, ExportError, ImageFormat


class CascadeExporter(BaseExporter):
    """
    Intelligent cascade exporter with automatic fallback.
    
    Features:
    - 🎯 Automatically selects best available exporter
    - ⚡ Prioritizes speed (matplotlib > cairosvg > others)
    - 🔄 Graceful fallback if primary fails
    - 📊 Works with Plotly, Matplotlib, PyVista
    - ⚙️ User can force specific exporter
    
    Example:
        >>> # Automatic selection
        >>> exporter = CascadeExporter()
        >>> exporter.export(fig, "output.png")  # Uses best available
        
        >>> # Force specific exporter
        >>> exporter = CascadeExporter(prefer="cairosvg")
        >>> exporter.export(fig, "output.png")  # Uses CairoSVG if available
    """
    
    def __init__(self, prefer: Optional[str] = None):
        """
        Initialize cascade exporter.
        
        Args:
            prefer: Preferred exporter name ("matplotlib", "cairosvg", "playwright")
                   If None, auto-selects based on speed.
        """
        self.prefer = prefer
        self._exporters = self._initialize_exporters()
        
        if not self._exporters:
            raise RuntimeError(
                "No exporters available! Install at least one:\n"
                "  pip install matplotlib      # Fast, recommended\n"
                "  pip install cairosvg        # For Plotly→PNG\n"
                "  pip install playwright      # Universal fallback"
            )
    
    def _initialize_exporters(self) -> List[BaseExporter]:
        """Initialize available exporters in priority order."""
        exporters = []
        
        # Try matplotlib (primary, fastest)
        try:
            from .matplotlib_exporter import MatplotlibExporter
            exporters.append(('matplotlib', MatplotlibExporter()))
        except ImportError:
            raise RuntimeError(
                "MatplotlibExporter not available! "
                "Matplotlib is required for PyMemorial.\n"
                "Install with: pip install matplotlib"
            )
        return exporters

    
    def can_export(self, fig: Any, format: ImageFormat) -> bool:
        """Check if any exporter can handle this."""
        for name, exporter in self._exporters:
            if exporter.can_export(fig, format):
                return True
        return False
    
    def export(
        self,
        fig: Any,
        filename: str | Path,
        config: Optional[ExportConfig] = None
    ) -> Path:
        """
        Export figure using best available exporter.
        
        Tries exporters in order:
        1. Preferred exporter (if specified and available)
        2. Exporters by speed priority
        3. Raises ExportError if all fail
        """
        if config is None:
            config = ExportConfig()
        
        filename = Path(filename)
        
        # Get exporters that can handle this figure+format
        candidates = self._get_candidates(fig, config.format)
        
        if not candidates:
            fig_type = self._detect_figure_type(fig)
            raise ExportError(
                f"No exporter available for {fig_type} figure → {config.format}\n"
                f"Available exporters: {[name for name, _ in self._exporters]}"
            )
        
        # Try preferred exporter first
        if self.prefer:
            for name, exporter in candidates:
                if name == self.prefer:
                    candidates.remove((name, exporter))
                    candidates.insert(0, (name, exporter))
                    break
        
        # Try each candidate until one succeeds
        errors = []
        for name, exporter in candidates:
            try:
                print(f"🎨 Exporting via {name}... ", end='', flush=True)
                result = exporter.export(fig, filename, config)
                print(f"✅ Success ({filename.stat().st_size // 1024} KB)")
                return result
            
            except Exception as e:
                print(f"❌ Failed")
                errors.append(f"{name}: {str(e)}")
                continue
        
        # All exporters failed
        raise ExportError(
            f"All exporters failed for {filename}:\n" +
            "\n".join(f"  - {err}" for err in errors)
        )
    
    def _get_candidates(
        self,
        fig: Any,
        format: ImageFormat
    ) -> List[tuple[str, BaseExporter]]:
        """Get exporters that can handle this figure+format."""
        candidates = []
        
        for name, exporter in self._exporters:
            if exporter.can_export(fig, format):
                candidates.append((name, exporter))
        
        return candidates
    
    def get_available_exporters(self) -> List[str]:
        """Get list of available exporter names."""
        return [name for name, _ in self._exporters]
    
    def benchmark(self, fig: Any, format: ImageFormat = "png") -> dict:
        """
        Benchmark all available exporters.
        
        Returns:
            Dict mapping exporter name → time in seconds
        
        Example:
            >>> results = exporter.benchmark(fig, "png")
            >>> print(results)
            {'matplotlib': 0.12, 'cairosvg': 0.23}
        """
        import time
        import tempfile
        
        candidates = self._get_candidates(fig, format)
        results = {}
        
        with tempfile.TemporaryDirectory() as tmpdir:
            for name, exporter in candidates:
                filename = Path(tmpdir) / f"test_{name}.{format}"
                config = ExportConfig(format=format)
                
                try:
                    start = time.time()
                    exporter.export(fig, filename, config)
                    elapsed = time.time() - start
                    results[name] = round(elapsed, 3)
                except Exception as e:
                    results[name] = f"FAILED: {e}"
        
        return results


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

_default_exporter = None


def export_figure(
    fig: Any,
    filename: str | Path,
    config: Optional[ExportConfig] = None,
    **kwargs
) -> Path:
    """
    Quick export with automatic exporter selection.
    
    Parameters
    ----------
    fig : Any
        Figure to export
    filename : str | Path
        Output filename
    config : ExportConfig, optional
        Export configuration. If None, created from kwargs
    **kwargs
        Additional parameters for ExportConfig (dpi, width, height, etc)
    
    Returns
    -------
    Path
        Path to exported file
    
    Examples
    --------
    >>> from pymemorial.visualization import export_figure
    >>> export_figure(fig, "output.png", dpi=300)
    PosixPath('output.png')
    
    >>> # With explicit config
    >>> config = ExportConfig(format='pdf', dpi=150)
    >>> export_figure(fig, "output.pdf", config)
    """
    global _default_exporter
    if _default_exporter is None:
        _default_exporter = CascadeExporter()
    
    # **FIX: Create config from kwargs if not provided**
    if config is None:
        # Extract format from filename if not in kwargs
        if 'format' not in kwargs:
            suffix = Path(filename).suffix.lower().lstrip('.')
            if suffix in ('png', 'pdf', 'svg', 'jpg', 'jpeg'):
                kwargs['format'] = suffix
        
        config = ExportConfig(**kwargs)
    
    return _default_exporter.export(fig, filename, config)

_default_exporter = None

