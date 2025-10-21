# src/pymemorial/document/generators/base_generator.py
"""
Base Generator Module - Abstract interface for document generators.

This module provides the abstract base class for all document generators
(PDF, HTML, LaTeX, etc).

Author: PyMemorial Team
Date: 2025-10-20
Phase: 7.3
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Literal

from pymemorial.document.base_document import BaseDocument


# ============================================================================
# CONFIGURATION CLASSES
# ============================================================================

@dataclass
class PageConfig:
    """Page configuration for PDF generation."""
    
    size: Literal['A4', 'Letter', 'Legal'] = 'A4'
    orientation: Literal['portrait', 'landscape'] = 'portrait'
    
    # Margins (in mm)
    margin_top: float = 30.0
    margin_bottom: float = 20.0
    margin_left: float = 30.0
    margin_right: float = 20.0
    
    # Header/Footer
    header_height: float = 15.0
    footer_height: float = 15.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'size': self.size,
            'orientation': self.orientation,
            'margin_top': f'{self.margin_top}mm',
            'margin_bottom': f'{self.margin_bottom}mm',
            'margin_left': f'{self.margin_left}mm',
            'margin_right': f'{self.margin_right}mm',
            'header_height': f'{self.header_height}mm',
            'footer_height': f'{self.footer_height}mm',
        }


@dataclass
class PDFMetadata:
    """PDF metadata (Dublin Core)."""
    
    title: Optional[str] = None
    author: Optional[str] = None
    subject: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    creator: str = 'PyMemorial v2.0'
    producer: str = 'WeasyPrint + PyMemorial'
    creation_date: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to WeasyPrint metadata format."""
        metadata = {
            'creator': self.creator,
            'producer': self.producer,
        }
        
        if self.title:
            metadata['title'] = self.title
        if self.author:
            metadata['author'] = self.author
        if self.subject:
            metadata['subject'] = self.subject
        if self.keywords:
            metadata['keywords'] = ', '.join(self.keywords)
        if self.creation_date:
            metadata['created'] = self.creation_date.isoformat()
        
        return metadata


@dataclass
class GeneratorConfig:
    """Configuration for document generator."""
    
    page: PageConfig = field(default_factory=PageConfig)
    metadata: PDFMetadata = field(default_factory=PDFMetadata)
    
    # Page numbering
    page_numbering: bool = True
    page_numbering_start: int = 1
    page_numbering_format: str = 'decimal'  # decimal, roman, alpha
    
    # Headers/Footers
    show_header: bool = True
    show_footer: bool = True
    header_text: Optional[str] = None
    footer_text: Optional[str] = None
    
    # Table of contents
    generate_toc: bool = True
    toc_depth: int = 3
    
    # Performance
    optimize_images: bool = True
    compress_pdf: bool = True
    
    # Debug
    debug: bool = False
    save_intermediate_html: bool = False


# ============================================================================
# BASE GENERATOR
# ============================================================================

class BaseGenerator(ABC):
    """
    Abstract base class for document generators.
    
    All generators (WeasyPrint, Quarto, Playwright, LaTeX) inherit from this.
    
    Parameters
    ----------
    config : GeneratorConfig, optional
        Generator configuration
    
    Examples
    --------
    >>> from pymemorial.document.generators import WeasyPrintGenerator
    >>> generator = WeasyPrintGenerator()
    >>> generator.generate(memorial, output_path='memorial.pdf')
    """
    
    def __init__(self, config: Optional[GeneratorConfig] = None):
        """Initialize generator."""
        self.config = config or GeneratorConfig()
        self._logger = logging.getLogger(self.__class__.__name__)
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging for generator."""
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self._logger.addHandler(handler)
        
        if self.config.debug:
            self._logger.setLevel(logging.DEBUG)
        else:
            self._logger.setLevel(logging.INFO)
    
    @abstractmethod
    def generate(
        self,
        document: BaseDocument,
        output_path: Path,
        **kwargs
    ) -> Path:
        """
        Generate document to file.
        
        Parameters
        ----------
        document : BaseDocument
            Document to generate
        output_path : Path
            Output file path
        **kwargs
            Additional generator-specific options
        
        Returns
        -------
        Path
            Path to generated file
        
        Raises
        ------
        GenerationError
            If generation fails
        """
        pass
    
    @abstractmethod
    def generate_to_bytes(
        self,
        document: BaseDocument,
        **kwargs
    ) -> bytes:
        """
        Generate document to bytes (in-memory).
        
        Parameters
        ----------
        document : BaseDocument
            Document to generate
        **kwargs
            Additional generator-specific options
        
        Returns
        -------
        bytes
            Generated document as bytes
        
        Raises
        ------
        GenerationError
            If generation fails
        """
        pass
    
    def validate_document(self, document: BaseDocument) -> bool:
        """
        Validate document before generation.
        
        Parameters
        ----------
        document : BaseDocument
            Document to validate
        
        Returns
        -------
        bool
            True if valid
        
        Raises
        ------
        ValidationError
            If document invalid
        """
        self._logger.debug(f"Validating document: {document.metadata.title}")
        
        # Basic validation
        if not document.metadata:
            raise ValueError("Document must have metadata")
        
        if not document.metadata.title:
            raise ValueError("Document must have title")
        
        return True


# ============================================================================
# EXCEPTIONS
# ============================================================================

class GenerationError(Exception):
    """Raised when document generation fails."""
    pass


class ValidationError(Exception):
    """Raised when document validation fails."""
    pass
