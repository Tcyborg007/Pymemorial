"""
Base Exporter - Abstract base class for all exporters.

Defines interface for exporting figures to various formats.
All exporters must implement this interface.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal, Optional, Any
from dataclasses import dataclass

ImageFormat = Literal["png", "pdf", "svg", "jpg", "webp"]


@dataclass
class ExportConfig:
    """Configuration for figure export."""
    
    format: ImageFormat = "png"
    dpi: int = 300
    width: int = 1200
    height: int = 800
    transparent: bool = False
    quality: int = 95  # For JPEG/WebP
    
    # Advanced options
    bbox_inches: str = "tight"
    pad_inches: float = 0.1
    facecolor: str = "white"
    edgecolor: str = "none"
    
    def __post_init__(self):
        """Validate configuration."""
        if self.dpi < 72:
            raise ValueError("DPI must be >= 72")
        if self.width < 100 or self.height < 100:
            raise ValueError("Width and height must be >= 100")
        if not 0 <= self.quality <= 100:
            raise ValueError("Quality must be 0-100")


class BaseExporter(ABC):
    """
    Abstract base class for figure exporters.
    
    All exporters must implement:
    - can_export(): Check if exporter is available
    - export(): Export figure to file
    
    Example:
        >>> class MyExporter(BaseExporter):
        ...     def can_export(self, fig, format):
        ...         return format == "png"
        ...     
        ...     def export(self, fig, filename, config):
        ...         # Implementation
        ...         return Path(filename)
    """
    
    @abstractmethod
    def can_export(
        self,
        fig: Any,
        format: ImageFormat
    ) -> bool:
        """
        Check if exporter can handle this figure type and format.
        
        Args:
            fig: Figure object (Plotly, Matplotlib, etc.)
            format: Desired output format
        
        Returns:
            True if exporter can handle this combination
        """
        pass
    
    @abstractmethod
    def export(
        self,
        fig: Any,
        filename: str | Path,
        config: Optional[ExportConfig] = None
    ) -> Path:
        """
        Export figure to file.
        
        Args:
            fig: Figure object to export
            filename: Output filename (with or without extension)
            config: Export configuration
        
        Returns:
            Path to exported file
        
        Raises:
            ExportError: If export fails
        """
        pass
    
    def _ensure_extension(
        self,
        filename: Path,
        format: ImageFormat
    ) -> Path:
        """Ensure filename has correct extension."""
        if filename.suffix.lower() != f".{format}":
            return filename.with_suffix(f".{format}")
        return filename
    
    def _detect_figure_type(self, fig: Any) -> str:
        """
        Detect figure type.
        
        Returns:
            'plotly', 'matplotlib', 'pyvista', or 'unknown'
        """
        if hasattr(fig, 'to_html') and hasattr(fig, 'to_json'):
            return 'plotly'
        elif hasattr(fig, 'savefig') and hasattr(fig, 'canvas'):
            return 'matplotlib'
        elif hasattr(fig, 'show') and 'pyvista' in str(type(fig)):
            return 'pyvista'
        else:
            return 'unknown'


class ExportError(Exception):
    """Raised when export fails."""
    pass
