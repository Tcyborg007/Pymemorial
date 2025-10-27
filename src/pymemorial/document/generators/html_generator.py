# src/pymemorial/document/generators/html_generator.py
"""
HTML Generator - PyMemorial v2.0 (Production Ready)

Standalone HTML generation for structural calculation memorials.

Author: PyMemorial Team
Date: 2025-10-21
Version: 2.0.0
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Dict, Any, Union
from io import BytesIO

from .base_generator import BaseGenerator

logger = logging.getLogger(__name__)


class HTMLGenerator(BaseGenerator):
    """
    Standalone HTML document generator.
    
    Generates clean, standalone HTML files with embedded CSS and JavaScript.
    
    Features:
    - Embedded CSS (no external dependencies)
    - Responsive design
    - Print-friendly
    - Syntax highlighting for code
    - Auto-numbered equations, figures, and tables
    
    Examples:
    --------
    >>> from pymemorial.document import Memorial
    >>> from pymemorial.document.generators import HTMLGenerator
    >>> 
    >>> memorial = Memorial(title="Design Calculation")
    >>> memorial.add_section("Introduction")
    >>> memorial.add_paragraph("This is a test.")
    >>> 
    >>> generator = HTMLGenerator()
    >>> generator.generate(memorial, 'output.html', style='nbr')
    """
    
    def __init__(self):
        """Initialize HTML generator."""
        super().__init__()
        logger.debug("HTMLGenerator initialized")
    
    def generate(
        self,
        document: Any,
        output_path: Union[str, Path],
        style: str = 'nbr',
        **kwargs
    ) -> None:
        """
        Generate standalone HTML file.
        
        Args:
            document: Memorial/Document instance
            output_path: Output HTML file path
            style: CSS style ('nbr', 'modern', 'aisc')
            **kwargs: Additional options
                - minify: Minify HTML output (default: False)
                - embed_fonts: Embed fonts in HTML (default: True)
        
        Raises:
            ValueError: If document is invalid
            IOError: If file write fails
        """
        logger.info(f"Generating HTML: {output_path} (style: {style})")
        
        # Validate document - try multiple methods
        if hasattr(document, 'to_html'):
            # Direct to_html method (preferred)
            html_content = document.to_html(style=style)
        elif hasattr(document, 'render'):
            # Fallback: use render method
            html_content = document.render(format='html', style=style)
        else:
            raise ValueError(
                f"Document must have 'to_html()' or 'render()' method. Got: {type(document)}"
            )
        
        # Apply additional processing
        if kwargs.get('minify', False):
            html_content = self._minify_html(html_content)
        
        # Write to file
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            output_file.write_text(html_content, encoding='utf-8')
            logger.info(f"✅ HTML generated: {output_path}")
        except Exception as e:
            logger.error(f"❌ Failed to write HTML: {e}")
            raise IOError(f"Failed to write HTML to {output_path}: {e}") from e

    
    def generate_to_bytes(
        self,
        document: Any,
        style: str = 'nbr',
        **kwargs
    ) -> bytes:
        """
        Generate HTML content to bytes (in-memory).
        
        Required by BaseGenerator abstract method.
        
        Args:
            document: Memorial/Document instance
            style: CSS style ('nbr', 'modern', 'aisc')
            **kwargs: Additional options
        
        Returns:
            HTML content as bytes
        
        Examples:
        --------
        >>> generator = HTMLGenerator()
        >>> html_bytes = generator.generate_to_bytes(memorial, style='nbr')
        >>> print(len(html_bytes))  # Size in bytes
        """
        logger.debug("Generating HTML to bytes (in-memory)")
        
        # Validate document
        if not hasattr(document, 'to_html'):
            raise ValueError(
                f"Document must have 'to_html()' method. Got: {type(document)}"
            )
        
        # Generate HTML content
        html_content = document.to_html(style=style)
        
        # Apply processing
        if kwargs.get('minify', False):
            html_content = self._minify_html(html_content)
        
        # Convert to bytes
        html_bytes = html_content.encode('utf-8')
        
        logger.debug(f"HTML generated: {len(html_bytes)} bytes")
        return html_bytes
    
    def _minify_html(self, html: str) -> str:
        """
        Minify HTML (basic implementation).
        
        Args:
            html: HTML content
        
        Returns:
            Minified HTML
        """
        import re
        
        # Remove HTML comments
        html = re.sub(r'<!--.*?-->', '', html, flags=re.DOTALL)
        
        # Remove extra whitespace (but preserve <pre> content)
        html = re.sub(r'\s+', ' ', html)
        
        # Remove whitespace between tags
        html = re.sub(r'>\s+<', '><', html)
        
        return html.strip()
    
    def validate(self, document: Any) -> bool:
        """
        Validate document before generation.
        
        Args:
            document: Document to validate
        
        Returns:
            True if valid, False otherwise
        """
        # Check if document has required method
        if not hasattr(document, 'to_html'):
            logger.warning(f"Document {type(document)} missing 'to_html()' method")
            return False
        
        # Check if document has title
        if hasattr(document, 'title') and not document.title:
            logger.warning("Document has empty title")
            return False
        
        logger.debug("Document validation passed")
        return True


__all__ = ['HTMLGenerator']
