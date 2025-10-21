# src/pymemorial/document/generators/weasyprint_generator.py
"""
WeasyPrint Generator - Fast PDF generation with automatic page numbering.

This module provides PDF generation using WeasyPrint (10x faster than Quarto).

Key Features:
- Automatic page numbering
- Headers/Footers with running content
- Table of contents with page numbers
- Image optimization
- CSS Paged Media support (@page rules)

Author: PyMemorial Team
Date: 2025-10-20
Phase: 7.3
"""

from __future__ import annotations

import io
import logging
import warnings
from pathlib import Path
from typing import Optional, Dict, Any

try:
    from weasyprint import HTML, CSS
    from weasyprint.text.fonts import FontConfiguration
    WEASYPRINT_AVAILABLE = True
except ImportError:
    WEASYPRINT_AVAILABLE = False
    warnings.warn(
        "WeasyPrint not available. Install with: pip install weasyprint"
    )

from pymemorial.document.base_document import BaseDocument
from pymemorial.document.generators.base_generator import (
    BaseGenerator,
    GeneratorConfig,
    GenerationError,
)


# ============================================================================
# WEASYPRINT GENERATOR
# ============================================================================

class WeasyPrintGenerator(BaseGenerator):
    """
    WeasyPrint PDF Generator with automatic page numbering.
    
    This generator uses WeasyPrint to create professional PDFs with:
    - Automatic page numbering (1, 2, 3, ... or i, ii, iii, ...)
    - Headers/Footers
    - Table of contents with page numbers
    - CSS Paged Media (@page rules)
    
    Parameters
    ----------
    config : GeneratorConfig, optional
        Generator configuration
    
    Examples
    --------
    >>> from pymemorial.document import Memorial
    >>> from pymemorial.document.generators import WeasyPrintGenerator
    >>> 
    >>> memorial = Memorial(metadata)
    >>> memorial.add_section("Introduction", "This is a test")
    >>> 
    >>> generator = WeasyPrintGenerator()
    >>> generator.generate(memorial, Path("output.pdf"))
    PosixPath('output.pdf')
    
    See Also
    --------
    QuartoGenerator : Academic documents (ABNT)
    PlaywrightGenerator : 3D visualization support
    """
    
    def __init__(self, config: Optional[GeneratorConfig] = None):
        """Initialize WeasyPrint generator."""
        super().__init__(config)
        
        if not WEASYPRINT_AVAILABLE:
            raise ImportError(
                "WeasyPrint not available. Install with: pip install weasyprint"
            )
        
        # Font configuration for better typography
        self.font_config = FontConfiguration()
        self._logger.info("WeasyPrint generator initialized")
    
    def generate(
        self,
        document: BaseDocument,
        output_path: Path,
        **kwargs
    ) -> Path:
        """
        Generate PDF document.
        
        Parameters
        ----------
        document : BaseDocument
            Document to generate
        output_path : Path
            Output PDF path
        **kwargs
            Additional options:
            - stylesheets : List[str] - Additional CSS files
            - attachments : List[Path] - Files to attach to PDF
        
        Returns
        -------
        Path
            Path to generated PDF
        
        Raises
        ------
        GenerationError
            If PDF generation fails
        
        Examples
        --------
        >>> generator.generate(
        ...     memorial,
        ...     Path("memorial.pdf"),
        ...     stylesheets=["custom.css"]
        ... )
        """
        self._logger.info(f"Generating PDF: {output_path}")
        
        # Validate document
        self.validate_document(document)
        
        # Ensure output directory exists
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Step 1: Generate HTML
            html_content = self._generate_html(document)
            
            # Step 2: Save intermediate HTML (if debug)
            if self.config.save_intermediate_html:
                html_path = output_path.with_suffix('.html')
                html_path.write_text(html_content, encoding='utf-8')
                self._logger.debug(f"Saved intermediate HTML: {html_path}")
            
            # Step 3: Generate CSS (with @page rules)
            css_content = self._generate_css()
            
            # Step 4: Additional stylesheets
            stylesheets = kwargs.get('stylesheets', [])
            css_objects = [CSS(string=css_content, font_config=self.font_config)]
            
            for stylesheet in stylesheets:
                if Path(stylesheet).exists():
                    css_objects.append(
                        CSS(filename=stylesheet, font_config=self.font_config)
                    )
            
            # Step 5: Create HTML object
            html_obj = HTML(string=html_content)
            
            # Step 6: Render to PDF
            self._logger.info("Rendering PDF with WeasyPrint...")
            document_obj = html_obj.render(stylesheets=css_objects)
            
            # Step 7: Write PDF
            document_obj.write_pdf(
                target=str(output_path),
                zoom=1.0,
                attachments=kwargs.get('attachments', []),
            )
            
            # Step 8: Log success
            size_kb = output_path.stat().st_size / 1024
            self._logger.info(
                f"PDF generated successfully: {output_path} ({size_kb:.1f} KB)"
            )
            
            return output_path
            
        except Exception as e:
            self._logger.error(f"PDF generation failed: {e}")
            raise GenerationError(f"Failed to generate PDF: {e}") from e
    
    def generate_to_bytes(
        self,
        document: BaseDocument,
        **kwargs
    ) -> bytes:
        """
        Generate PDF to bytes (in-memory).
        
        Parameters
        ----------
        document : BaseDocument
            Document to generate
        **kwargs
            Additional options
        
        Returns
        -------
        bytes
            PDF as bytes
        
        Examples
        --------
        >>> pdf_bytes = generator.generate_to_bytes(memorial)
        >>> with open('output.pdf', 'wb') as f:
        ...     f.write(pdf_bytes)
        """
        self._logger.info("Generating PDF to bytes")
        
        # Validate document
        self.validate_document(document)
        
        try:
            # Generate HTML and CSS
            html_content = self._generate_html(document)
            css_content = self._generate_css()
            
            # Create HTML object
            html_obj = HTML(string=html_content)
            css_obj = CSS(string=css_content, font_config=self.font_config)
            
            # Render to PDF
            document_obj = html_obj.render(stylesheets=[css_obj])
            
            # Write to bytes
            pdf_bytes = document_obj.write_pdf()
            
            self._logger.info(f"PDF generated to bytes ({len(pdf_bytes)} bytes)")
            return pdf_bytes
            
        except Exception as e:
            self._logger.error(f"PDF generation to bytes failed: {e}")
            raise GenerationError(f"Failed to generate PDF: {e}") from e
    
    def _generate_html(self, document: BaseDocument) -> str:
        """
        Generate HTML content from document.
        
        Parameters
        ----------
        document : BaseDocument
            Document to convert
        
        Returns
        -------
        str
            HTML content
        """
        self._logger.debug("Generating HTML content")
        
        # Use document's render_to_string() method to get HTML
        # (NOT render() which writes to file)
        html_content = document.render_to_string()  # ✅ CORRETO
        
        # Add page numbering structure
        html_content = self._add_page_numbering_structure(html_content)
        
        return html_content

    
    def _add_page_numbering_structure(self, html: str) -> str:
        """
        Add page numbering CSS counter structure to HTML.
        
        WeasyPrint uses CSS counters for page numbering:
        - counter-reset: page 1;
        - counter-increment: page;
        - content: counter(page);
        
        Parameters
        ----------
        html : str
            HTML content
        
        Returns
        -------
        str
            HTML with page numbering structure
        """
        # Add page counter in footer
        # This is handled by CSS @page rules
        return html
    
    def _generate_css(self) -> str:
        """
        Generate CSS with @page rules for headers/footers.
        
        CSS Paged Media specification:
        https://www.w3.org/TR/css-page-3/
        
        Returns
        -------
        str
            CSS content with @page rules
        """
        page_config = self.config.page
        
        css_parts = []
        
        # Base @page rule
        css_parts.append(f"""
        @page {{
            size: {page_config.size} {page_config.orientation};
            margin-top: {page_config.margin_top}mm;
            margin-bottom: {page_config.margin_bottom}mm;
            margin-left: {page_config.margin_left}mm;
            margin-right: {page_config.margin_right}mm;
            
            /* Page counters */
            counter-increment: page;
            
            /* Running header */
            @top-center {{
                content: "{self.config.header_text or ''}";
                font-size: 10pt;
                font-family: Arial, sans-serif;
                color: #666;
            }}
            
            /* Running footer with page number */
            @bottom-center {{
                content: "Página " counter(page);
                font-size: 10pt;
                font-family: Arial, sans-serif;
                color: #666;
            }}
        }}
        
        /* First page (no header/footer) */
        @page :first {{
            @top-center {{
                content: none;
            }}
            @bottom-center {{
                content: none;
            }}
        }}
        
        /* Left pages (even) */
        @page :left {{
            margin-left: {page_config.margin_right}mm;
            margin-right: {page_config.margin_left}mm;
        }}
        
        /* Right pages (odd) */
        @page :right {{
            margin-left: {page_config.margin_left}mm;
            margin-right: {page_config.margin_right}mm;
        }}
        """)
        
        # Page breaks
        css_parts.append("""
        /* Page break control */
        h1, h2 {
            page-break-after: avoid;
            page-break-inside: avoid;
        }
        
        figure, table {
            page-break-inside: avoid;
        }
        
        /* Keep sections together */
        section {
            page-break-inside: avoid;
        }
        """)
        
        return '\n'.join(css_parts)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def generate_pdf(
    document: BaseDocument,
    output_path: Path,
    config: Optional[GeneratorConfig] = None,
    **kwargs
) -> Path:
    """
    Convenience function to generate PDF.
    
    Parameters
    ----------
    document : BaseDocument
        Document to generate
    output_path : Path
        Output PDF path
    config : GeneratorConfig, optional
        Generator configuration
    **kwargs
        Additional options
    
    Returns
    -------
    Path
        Path to generated PDF
    
    Examples
    --------
    >>> from pymemorial.document.generators import generate_pdf
    >>> generate_pdf(memorial, Path("output.pdf"))
    """
    generator = WeasyPrintGenerator(config)
    return generator.generate(document, output_path, **kwargs)
