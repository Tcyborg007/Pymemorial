# src/pymemorial/document/memorial.py
"""
Memorial Class - Concrete implementation for calculation memorials (MVP v2.0 Compatible).

This module provides the Memorial document type, specialized for structural
engineering calculation reports following NBR, AISC, and Eurocode standards.

Author: PyMemorial Team
Date: 2025-10-21
Version: 2.0.0 (MVP Compatible)
"""

from __future__ import annotations

import base64

import html
import json
import logging
import tempfile
import warnings
import uuid
import os
from html import escape
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Literal

import html
import base64
try:
    # Tenta importar o markdown2 para renderização robusta de texto
    import markdown2
    MARKDOWN_AVAILABLE = True
except ImportError:
    MARKDOWN_AVAILABLE = False
    
from .base_document import (
    BaseDocument,
    DocumentMetadata,
    NormCode,
    ValidationError,
    DocumentValidationError,
    RenderError,
    CrossReferenceType,
    Section,
    Figure,
    Table,
    EquationDoc,
    TextBlock,  # <--- Certifique-se de importar TextBlock
    TableStyle, # <--- Certifique-se de importar TableStyle
    Verification,
    DocumentLanguage,
)






class HtmlRenderer:
    """
    Converte uma lista de itens do BaseDocument em uma string HTML.

    Esta classe é responsável por "desenhar" cada item (Seção, Parágrafo,
    Equação, etc.) em HTML, aplicando classes de CSS inspiradas no
    visual do efficalc para um relatório limpo e profissional.
   
    """

    def __init__(self):
        # Escapar HTML é crucial para segurança e renderização correta
        self._esc = html.escape

    def render_body(self, content_items: list) -> str:
        """
        Renderiza a lista principal de conteúdo do documento.
        """
        body_parts = [self.render_content_item(item) for item in content_items]
        return "\n".join(body_parts)

    def render_content_item(self, item: Any) -> str:
        """
        Dispatcher principal: chama o método de renderização correto
        para cada tipo de item do memorial.
       
        """
        if isinstance(item, Section):
            return self._render_section(item)
        if isinstance(item, TextBlock):
            return self._render_paragraph(item)
        if isinstance(item, EquationDoc):
            return self._render_equation(item)
        if isinstance(item, Figure):
            return self._render_figure(item)
        if isinstance(item, Table):
            return self._render_table(item)
        if isinstance(item, str):
            # Fallback para strings simples
            return f'<p class="paragraph">{self._esc(item)}</p>'
        
        return f''

    def _render_section(self, section: Section) -> str:
        """
        Renderiza um item de Seção (título + conteúdo).
        Este método é recursivo, renderizando todo o conteúdo *dentro* da seção.
        """
        # Números (ex: "1.1") vêm do método _prepare_render do BaseDocument
        title = self._esc(f"{section.number} {section.title}")
        title_html = f'<h{section.level} class="section-title" id="{section.id}">{title}</h{section.level}>'
        
        # Chama recursivamente render_content_item para todo o conteúdo da seção
        content_html = "\n".join([self.render_content_item(item) for item in section.content])
        
        return f'<div class="section-container">\n{title_html}\n{content_html}\n</div>'

    def _render_paragraph(self, paragraph: TextBlock) -> str:
        """
        Renderiza um TextBlock.
        Assume que o SmartTextProcessor já processou
        o texto, convertendo variáveis inline (ex: f_ck) em LaTeX ($f_{ck}$).
       
        """
        # Se markdown2 estiver instalado, usa para renderizar listas, negrito, etc.
        if MARKDOWN_AVAILABLE:
            # O SmartTextProcessor já protegeu o LaTeX inline ($...$),
            # então o markdown não irá quebrá-lo.
            # Extras úteis para documentos técnicos.
            html_text = markdown2.markdown(
                paragraph.text, 
                extras=["fenced-code-blocks", "tables", "cuddled-lists", "target-blank-links"]
            )
            # Adiciona a classe 'paragraph' ao primeiro <p> gerado
            html_text = html_text.replace('<p>', '<p class="paragraph">', 1)
            return html_text
        else:
            # Fallback se markdown2 não estiver instalado:
            # Trata o texto como pré-formatado, escapa e substitui quebras de linha.
            escaped_text = self._esc(paragraph.text).replace('\n', '<br>\n')
            return f'<p class="paragraph">{escaped_text}</p>'

    def _render_equation(self, equation: EquationDoc) -> str:
        """
        Renderiza um EquationDoc.
        O LaTeX já vem formatado como um bloco "align*"
        pelo StepRegistry.
        """
        description = self._esc(equation.description)
        desc_html = f'<p class="equation-description">{description}</p>' if description else ""
        
        # O MathJax irá renderizar o bloco LaTeX
        # O add_calculation já formatou o latex com \begin{align*}...
        latex_html = f'\\[ {equation.latex} \\]'
        
        # O número da equação (ex: "1.1") vem do BaseDocument
        num_html = f'<span class="equation-number">({self._esc(equation.number)})</span>'
        
        # Usa a classe "calc-item" do efficalc para o estilo
        #
        return f"""
        <div class="calc-item" id="{equation.id}">
            {desc_html}
            <div class="equation-container">
                <div class="equation-latex">{latex_html}</div>
                {num_html}
            </div>
        </div>
        """

    def _render_figure(self, fig: Figure) -> str:
        """
        Renderiza uma Figura, convertendo o caminho para base64
        para um HTML autocontido.
        """
        try:
            with open(fig.path, "rb") as f:
                b64_data = base64.b64encode(f.read()).decode('utf-8')
            
            # Determina o mime type pela extensão do arquivo
            ext = str(fig.path).split('.')[-1].lower()
            if ext in ['jpg', 'jpeg']:
                mime = 'image/jpeg'
            elif ext == 'svg':
                mime = 'image/svg+xml'
            elif ext == 'png':
                mime = 'image/png'
            else:
                mime = 'application/octet-stream' # Fallback
                
            src = f'data:{mime};base64,{b64_data}'
            
            alt_text = self._esc(fig.caption)
            # O número da figura vem do BaseDocument
            caption = self._esc(f"Figura {fig.number} - {fig.caption}")
            
            return f"""
            <figure class="figure-container" id="{fig.id}">
                <img src="{src}" alt="{alt_text}" style="max-width:100%; height:auto;" />
                <figcaption>{caption}</figcaption>
            </figure>
            """
        except Exception as e:
            error_msg = self._esc(f"Erro ao carregar figura: {fig.path} - {e}")
            return f'<div class="figure-container error-box">{error_msg}</div>'

    def _render_table(self, table: Table) -> str:
        """
        Renderiza uma Tabela.
        Tenta detectar se os dados são um DataFrame pandas ou uma lista de listas.
        """
        try:
            # O número da tabela vem do BaseDocument
            caption = self._esc(f"Tabela {table.number} - {table.caption}")
            caption_html = f'<caption>{caption}</caption>'
            
            headers = []
            data_rows = []
            
            # Detecção de DataFrame (sem importar pandas)
            if "DataFrame" in str(type(table.data)):
                headers = table.data.columns.tolist()
                data_rows = table.data.values.tolist()
            # Detecção de lista de listas (assume que a primeira linha é header)
            elif isinstance(table.data, list) and len(table.data) > 0 and isinstance(table.data[0], list):
                headers = table.data[0]
                data_rows = table.data[1:]
            else:
                raise ValueError("Formato de dados da tabela não suportado. Use DataFrame ou lista de listas.")

            # Monta o cabeçalho
            header_html = "<thead><tr>"
            for h in headers:
                header_html += f"<th>{self._esc(str(h))}</th>"
            header_html += "</tr></thead>"
            
            # Monta o corpo
            body_html = "<tbody>"
            for row in data_rows:
                body_html += "<tr>"
                for cell in row:
                    body_html += f"<td>{self._esc(str(cell))}</td>"
                body_html += "</tr>"
            body_html += "</tbody>"
            
            # Usa o estilo "striped" (zebrado) do efficalc
            #
            style_class = "striped"
            if table.style == TableStyle.MINIMAL:
                style_class = "minimal"
            
            return f"""
            <div class="table-container" id="{table.id}">
                <table class="{style_class}">
                    {caption_html}
                    {header_html}
                    {body_html}
                </table>
            </div>
            """
        except Exception as e:
            error_msg = self._esc(f"Erro ao renderizar tabela: {table.caption} - {e}")
            return f'<div class="table-container error-box">{error_msg}</div>'



# ============================================================================
# MEMORIAL CLASS
# ============================================================================

class Memorial(BaseDocument):
    """
    Memorial de Cálculo - Specialized document for structural calculations.
    
    Extends BaseDocument with:
    - Template selection based on norm (NBR, AISC, Eurocode)
    - Automatic title page and TOC generation
    - PDF/HTML/DOCX rendering
    - Validation with norm-specific rules
    """
    
    def __init__(
        self,
        metadata: DocumentMetadata,
        template: str = 'default',
        auto_toc: bool = True,
        auto_title_page: bool = True,
        auto_lists: bool = False,
        strict_validation: bool = False,
        **kwargs
    ):
        """
        Initialize Memorial.
        
        Args:
            metadata: Document metadata
            template: Template name (or auto-selected from norm)
            auto_toc: Generate table of contents automatically
            auto_title_page: Generate title page automatically
            auto_lists: Generate lists of figures/tables/equations
            strict_validation: Raise error on validation failures
        """
        super().__init__(metadata)
        
        self.template_name = template or self._select_template(metadata.norm_code)
        self.strict_validation = strict_validation
        self.auto_toc = auto_toc
        self.auto_title_page = auto_title_page
        
        self._logger.info(f"Memorial initialized with template: {self.template_name}")
        self._logger.debug(f"Auto-TOC: {auto_toc}, Auto-Title: {auto_title_page}")
        
        if auto_lists:
            self._logger.debug("Generating automatic lists...")
            self._generate_list_of_figures()
            self._generate_list_of_tables()
            self._generate_list_of_equations()



    # ========================================================================
    # ENHANCED CONTENT METHODS (SmartText Integration)
    # ========================================================================
    
    def add_paragraph(
        self,
        text: str,
        processing_mode: Literal['smart', 'normal', 'detailed'] = 'normal',
        **kwargs
    ) -> None:
        """
        Add paragraph with SmartText processing.
        
        Args:
            text: Paragraph text (can contain variables like M_k, γ_f)
            processing_mode: 
                - 'normal': Direct text (no processing)
                - 'smart': Auto-detect variables and format
                - 'detailed': Generate step-by-step explanation
            **kwargs: Additional options
        
        Examples:
            >>> memorial.add_paragraph("Base: b = 20 cm", processing_mode='normal')
            >>> memorial.add_paragraph("M_d = M_k * gamma_f", processing_mode='smart')
            >>> memorial.add_paragraph("Calcular M_d", processing_mode='detailed')
        """
        if processing_mode == 'normal':
            # Direct text (current behavior)
            super().add_paragraph(text, **kwargs)
        
        elif processing_mode == 'smart':
            # SmartText: detect variables and format
            processed_text = self._process_smart_text(text)
            super().add_paragraph(processed_text, **kwargs)
        
        elif processing_mode == 'detailed':
            # Generate detailed steps
            steps = self._generate_detailed_steps(text)
            for step in steps:
                super().add_paragraph(step, **kwargs)
        
        else:
            raise ValueError(f"Invalid processing_mode: {processing_mode}")
    
    def _process_smart_text(self, text: str) -> str:
        """
        Process text with SmartText: detect variables and format.
        
        Examples:
            Input:  "M_d = M_k * gamma_f"
            Output: "**M_d** = **M_k** × **γ_f**"
        """
        try:
            from ..smarttext import TextProcessor
            processor = TextProcessor()
            
            # Auto-detect variables
            processed = processor.process(text)
            
            # Format with bold
            import re
            # Bold variables (e.g., M_d, f_ck)
            processed = re.sub(r'\b([A-Za-z]_[a-z]+)\b', r'**\1**', processed)
            
            # Convert greek letter names to symbols
            greek_map = {
                'gamma': 'γ',
                'alpha': 'α',
                'beta': 'β',
                'delta': 'δ',
                'epsilon': 'ε',
                'phi': 'φ',
                'lambda': 'λ',
                'mu': 'μ',
                'rho': 'ρ',
                'sigma': 'σ',
                'tau': 'τ',
            }
            
            for name, symbol in greek_map.items():
                processed = processed.replace(name, symbol)
            
            # Convert * to ×
            processed = processed.replace(' * ', ' × ')
            
            return processed
            
        except ImportError:
            self._logger.warning("SmartText not available, using plain text")
            return text
        except Exception as e:
            self._logger.warning(f"SmartText processing failed: {e}")
            return text
    
    def _generate_detailed_steps(self, text: str) -> List[str]:
        """
        Generate detailed calculation steps.
        
        Examples:
            Input:  "Calcular M_d = M_k * gamma_f com M_k=112.5, gamma_f=1.4"
            Output: [
                "1. **Dados de entrada:**",
                "   • M_k = 112.5 kN.m",
                "   • γ_f = 1.4",
                "2. **Cálculo:**",
                "   M_d = M_k × γ_f",
                "   M_d = 112.5 × 1.4",
                "   M_d = 157.5 kN.m",
                "3. **Resultado:**",
                "   **M_d = 157.5 kN.m**"
            ]
        """
        # MVP: Simple step generation
        # TODO: Implement full step-by-step generator
        
        steps = [
            "**Passo 1:** Identificar dados de entrada",
            "**Passo 2:** Aplicar fórmula",
            "**Passo 3:** Calcular resultado",
        ]
        
        self._logger.debug(f"Generated {len(steps)} detailed steps")
        
        return steps



    # ========================================================================
    # TEMPLATE SELECTION
    # ========================================================================
    
    def _select_template(self, norm_code: NormCode) -> str:
        """Select template based on norm code."""
        template_map = {
            NormCode.NBR6118_2023: 'nbr6118',
            NormCode.NBR8800_2024: 'nbr8800',
            NormCode.NBR9050_2020: 'nbr_modern',
            NormCode.AISC360_22: 'aisc360',
            NormCode.AISC341_22: 'aisc341',
            NormCode.EN1992_2004: 'eurocode',
            NormCode.EN1993_2005: 'eurocode',
            NormCode.EN1994_2004: 'eurocode',
            NormCode.ACI318_19: 'aci318',
            NormCode.CSA_A23_3_19: 'csa',
        }
        
        template = template_map.get(norm_code, 'modern')
        self._logger.debug(f"Template selected: {template} (norm: {norm_code})")
        
        return template
    
    # ========================================================================
    # ABSTRACT METHOD IMPLEMENTATIONS
    # ========================================================================
    
    def render(
        self,
        output_path: Union[str, Path],
        format: Literal['pdf', 'html', 'docx', 'latex'] = 'pdf',
        **kwargs
    ) -> Path:
        """
        Render memorial to file.
        
        Args:
            output_path: Output file path
            format: Output format ('pdf', 'html', 'docx', 'latex')
            **kwargs: Additional rendering options
        
        Returns:
            Path to rendered file
        
        Raises:
            DocumentValidationError: If validation fails (strict mode)
            RenderError: If rendering fails
        """
        output_path = Path(output_path)
        self._logger.info(f"Rendering memorial to: {output_path} (format: {format})")
        
        # Validate document
        try:
            validation_errors = self.validate()
            if validation_errors:
                self._logger.warning(f"Validation failed with {len(validation_errors)} errors")
                if self.strict_validation:
                    raise DocumentValidationError(validation_errors)
        except DocumentValidationError:
            if self.strict_validation:
                raise
        
        # Generate auto-content
        if self.auto_title_page:
            self._generate_title_page()
        
        if self.auto_toc:
            self._generate_toc()
        
        # Render to format
        try:
            if format == 'pdf':
                result_path = self._render_pdf(output_path, **kwargs)
            elif format == 'html':
                result_path = self._render_html(output_path, **kwargs)
            elif format == 'docx':
                result_path = self._render_docx(output_path, **kwargs)
            elif format == 'latex':
                result_path = self._render_latex(output_path, **kwargs)
            else:
                raise ValueError(f"Unknown format: {format}. Use 'pdf', 'html', 'docx', or 'latex'")
        except Exception as e:
            self._logger.error(f"Render failed: {e}")
            raise RenderError(f"Failed to render memorial: {e}") from e
        
        self._frozen = True
        
        if kwargs.get('open_on_save', False):
            self._open_file(result_path)
        
        self._logger.info(f"Memorial rendered successfully: {result_path}")
        
        return result_path

    def render_to_string(self) -> str:
        """
        Render memorial to HTML string (for preview/testing).
        
        Returns:
            HTML content as string
        """
        fd, tmp_path_str = tempfile.mkstemp(suffix='.html', text=True)
        tmp_path = Path(tmp_path_str)
        
        try:
            os.close(fd)
            
            self._render_html(tmp_path)
            
            try:
                html_content = tmp_path.read_text(encoding='utf-8')
            except UnicodeDecodeError as e:
                self._logger.error(f"UTF-8 decode failed: {e}")
                html_bytes = tmp_path.read_bytes()
                html_content = html_bytes.decode('utf-8', errors='replace')
                self._logger.warning("Used 'replace' mode to decode HTML")
                
            self._logger.debug(f"HTML rendered to string ({len(html_content)} chars)")
            return html_content
            
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def to_pdf(
        self,
        output_path: Union[str, Path],
        generator: str = 'weasyprint',
        **kwargs
    ) -> Path:
        """
        Render to PDF (alias for render with format='pdf').
        
        Args:
            output_path: Output PDF path
            generator: PDF generator backend (default: 'weasyprint')
            **kwargs: Additional options
        
        Returns:
            Path to generated PDF
        """
        if generator.lower() != 'weasyprint':
            raise NotImplementedError(f"Generator '{generator}' not yet implemented. Use 'weasyprint'.")
        
        return self.render(output_path, format='pdf', **kwargs)

    def validate(self) -> List[ValidationError]:
        """
        Validate memorial document (MVP: simplified).
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors: List[ValidationError] = []
        
        self._logger.info("Validating memorial document...")
        
        # Basic metadata validation
        errors.extend(self._validate_metadata())
        
        # Structure validation
        errors.extend(self._validate_structure())
        
        # Cross-reference validation
        errors.extend(self._validate_cross_references())
        
        # Figure validation (warnings only)
        warnings_list = self._validate_figures()
        for warn in warnings_list:
            self._logger.warning(warn.message)
        
        # Verification validation
        errors.extend(self._validate_verifications())
        
        self._logger.info(f"Validation complete: {len(errors)} errors, {len(warnings_list)} warnings")
        
        return errors
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert memorial to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            'type': 'Memorial',
            'template': self.template_name,
            'metadata': {
                'title': self.metadata.title,
                'author': self.metadata.author,
                'company': self.metadata.company,
                'norm_code': str(self.metadata.norm_code),
                'revision': self.metadata.revision,
                'date': self.metadata.date,
                'language': self.metadata.language.value,
                'keywords': self.metadata.keywords,
            },
            'sections': [
                {
                    'id': sec.id,
                    'number': sec.number,
                    'title': sec.title,
                    'level': sec.level,
                }
                for sec in self.sections
            ],
            'figures': [
                {
                    'id': fig.id,
                    'number': fig.number,
                    'caption': fig.caption,
                    'path': str(fig.path)
                }
                for fig in self.figures
            ],
            'tables': [
                {
                    'id': tbl.id,
                    'number': tbl.number,
                    'caption': tbl.caption
                }
                for tbl in self.tables
            ],
            'equations': [
                {
                    'id': eq.id,
                    'number': eq.number,
                    'latex': eq.latex[:50] + '...' if len(eq.latex) > 50 else eq.latex
                }
                for eq in self.equations
            ],
            'verifications': [
                {
                    'id': ver.id,
                    'name': ver.name,
                    'passed': ver.passed,
                    'description': ver.description,
                }
                for ver in self.verifications
            ],
            'toc': self.get_toc(),
            'statistics': {
                'sections': len(self.sections),
                'figures': len(self.figures),
                'tables': len(self.tables),
                'equations': len(self.equations),
                'verifications': len(self.verifications),
                'passed_verifications': sum(1 for v in self.verifications if v.passed),
                'failed_verifications': sum(1 for v in self.verifications if not v.passed)
            }
        }
    
    # ========================================================================
    # PRIVATE METHODS - Validation
    # ========================================================================
    
    def _validate_metadata(self) -> List[ValidationError]:
        """Validate document metadata."""
        errors = []
        
        if not self.metadata.title:
            errors.append(ValidationError(
                element_id='metadata',
                message='Document title is required',
                severity='error'
            ))
        
        if not self.metadata.author:
            errors.append(ValidationError(
                element_id='metadata',
                message='Document author is required',
                severity='error'
            ))
        
        self._logger.debug(f"Metadata validation: {len(errors)} errors")
        return errors
    
    def _validate_structure(self) -> List[ValidationError]:
        """Validate document structure."""
        errors = []
        
        if len(self.sections) == 0:
            errors.append(ValidationError(
                element_id='structure',
                message='Document has no sections',
                severity='error'
            ))
        
        for section in self.sections:
            if section.level < 1 or section.level > 6:
                errors.append(ValidationError(
                    element_id=section.id,
                    message=f'Section "{section.title}" has invalid level {section.level}',
                    severity='error'
                ))
        
        self._logger.debug(f"Structure validation: {len(errors)} errors")
        return errors
    
    def _validate_cross_references(self) -> List[ValidationError]:
        """Validate cross-references."""
        errors = []
        
        for from_id, refs in self.cross_refs.items():
            if from_id not in self._element_registry:
                errors.append(ValidationError(
                    element_id=from_id,
                    message=f'Cross-reference source not found: {from_id}',
                    severity='error'
                ))
            
            for ref in refs:
                if ref.to_id not in self._element_registry:
                    errors.append(ValidationError(
                        element_id=ref.to_id,
                        message=f'Cross-reference target not found: {ref.to_id}',
                        severity='error'
                    ))
        
        self._logger.debug(f"Cross-reference validation: {len(errors)} errors")
        return errors
    
    def _validate_figures(self) -> List[ValidationError]:
        """Validate figures (warnings only)."""
        warnings_list = []
        
        for figure in self.figures:
            if not figure.path.exists():
                warnings_list.append(ValidationError(
                    element_id=figure.id,
                    message=f'Figure file not found: {figure.path}',
                    severity='warning'
                ))
            elif figure.path.stat().st_size == 0:
                warnings_list.append(ValidationError(
                    element_id=figure.id,
                    message=f'Figure file is empty: {figure.path}',
                    severity='warning'
                ))
        
        self._logger.debug(f"Figure validation: {len(warnings_list)} warnings")
        return warnings_list
    
    def _validate_verifications(self) -> List[ValidationError]:
        """Validate verifications."""
        errors = []
        
        for verify in self.verifications:
            if not verify.name:
                errors.append(ValidationError(
                    element_id=verify.id,
                    message=f'Verification missing name',
                    severity='error'
                ))
        
        self._logger.debug(f"Verification validation: {len(errors)} errors")
        return errors
    
    # ========================================================================
    # PRIVATE METHODS - Content Generation
    # ========================================================================
    
    def _generate_title_page(self) -> None:
        """Generate title page (if not exists)."""
        # Check if already exists
        for s in self.sections:
            if hasattr(s, 'metadata') and isinstance(s.metadata, dict) and s.metadata.get('type') == 'title_page':
                self._logger.debug("Title page already exists, skipping")
                return
        
        self._logger.info("Generating title page...")
        
        # MVP: Simple metadata display
        project_info = f"**Projeto:** N/A  \n" if not hasattr(self.metadata, 'project_number') else \
                       f"**Projeto:** {self.metadata.project_number}  \n"
        
        content = f"""
# {self.metadata.title}

**{self.metadata.company}**

---

{project_info}**Autor:** {self.metadata.author}  
**Data:** {self.metadata.date}  
**Revisão:** {self.metadata.revision}  
**Norma:** {self.metadata.norm_code}

---
"""
        
        section_id = str(uuid.uuid4())
        
        title_section = Section(
            id=section_id,
            title='',
            number='0',
            level=1,
            numbered=False
        )
        # Add content via parent method (if available)
        if hasattr(title_section, 'add_content'):
            from .base_document import ContentBlock, ContentType
            title_section.add_content(ContentBlock(type=ContentType.TEXT, content=content))
        
        self.sections.insert(0, title_section)
        self._element_registry[section_id] = title_section
        
        self._logger.info("Title page generated")
    
    def _generate_toc(self) -> None:
        """Generate table of contents (if not exists)."""
        # Check if already exists
        for s in self.sections:
            if hasattr(s, 'metadata') and isinstance(s.metadata, dict) and s.metadata.get('type') == 'toc':
                self._logger.debug("TOC already exists, skipping")
                return
        
        self._logger.info("Generating table of contents...")
        
        toc = self.get_toc()
        
        if not toc:
            self._logger.warning("No sections for TOC")
            return
        
        content = "# Sumário\n\n" if self.metadata.language == DocumentLanguage.PT_BR else "# Table of Contents\n\n"
        
        for entry in toc:
            indent = "  " * (entry['level'] - 1)
            content += f"{indent}- {entry['number']} {entry['title']}\n"
        
        section_id = str(uuid.uuid4())
        
        toc_section = Section(
            id=section_id,
            title='',
            number='0.1',
            level=1,
            numbered=False
        )
        
        insert_pos = 1 if self.sections and hasattr(self.sections[0], 'metadata') and \
                     isinstance(self.sections[0].metadata, dict) and \
                     self.sections[0].metadata.get('type') == 'title_page' else 0
        self.sections.insert(insert_pos, toc_section)
        self._element_registry[section_id] = toc_section
        
        self._logger.info("TOC generated")

    def _generate_list_of_figures(self) -> None:
        """Generate list of figures (stub for MVP)."""
        self._logger.debug("List of figures generation skipped (MVP)")
        pass
    
    def _generate_list_of_tables(self) -> None:
        """Generate list of tables (stub for MVP)."""
        self._logger.debug("List of tables generation skipped (MVP)")
        pass
    
    def _generate_list_of_equations(self) -> None:
        """Generate list of equations (stub for MVP)."""
        self._logger.debug("List of equations generation skipped (MVP)")
        pass
    
    # ========================================================================
    # PRIVATE METHODS - Rendering
    # ========================================================================
    
    def _render_pdf(self, output_path: Path, **kwargs) -> Path:
        """Render to PDF via WeasyPrint."""
        self._logger.info("Rendering to PDF via WeasyPrint...")
        
        try:
            import weasyprint
        except ImportError:
            raise ImportError(
                "WeasyPrint is required for PDF rendering. "
                "Install with: pip install weasyprint"
            )
        
        fd, tmp_path_str = tempfile.mkstemp(suffix='.html', text=True)
        tmp_path = Path(tmp_path_str)
        
        try:
            os.close(fd)
            
            html_content = self._generate_html_content(**kwargs)
            
            with open(tmp_path, 'w', encoding='utf-8-sig') as f:
                f.write(html_content)
            
            self._logger.debug(f"Temporary HTML created: {tmp_path}")
            
            try:
                html_obj = weasyprint.HTML(string=html_content, base_url=str(tmp_path.parent))
                html_obj.write_pdf(output_path)
                
                self._logger.info(f"PDF generated: {output_path}")
                
            except Exception as e:
                self._logger.error(f"WeasyPrint rendering failed: {e}")
                
                debug_path = output_path.with_suffix('.debug.html')
                tmp_path.replace(debug_path)
                self._logger.info(f"Debug HTML saved: {debug_path}")
                
                raise
                
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
        
        return output_path.resolve()
    
    def render_html(self, filepath: str, open_on_save: bool = True):
        """
        Renderiza o documento completo para um arquivo HTML com estilo profissional.
        Inspirado na arquitetura de efficalc.ReportBuilder.save_report
        [cite: youandvern/efficalc/efficalc-5920cf5ae2a4920c85adc6d2880faf0e51898ecd/efficalc/report_builder.py]
        """
        self._prepare_render()  # Chama o método do BaseDocument para numerar tudo
        
        renderer = HtmlRenderer()
        # Usa o renderer para gerar o corpo do HTML
        body_content = renderer.render_body(self.content)

        # CSS inspirado diretamente no efficalc para um visual limpo e profissional
        # [cite: youandvern/efficalc/efficalc-5920cf5ae2a4920c85adc6d2880faf0e51898ecd/efficalc/report_builder.py]
        css_styles = """
            body {
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
                line-height: 1.6;
                color: #333;
                background-color: #eeeeee;
                margin: 0;
                padding: 0;
            }
            .page-container {
                max-width: 850px;
                margin: 2rem auto;
                padding: 2.5rem;
                background-color: #ffffff;
                box-shadow: rgba(0, 0, 0, 0.24) 0px 3px 8px;
            }
            h1, h2, h3, h4, h5 {
                color: #2c3e50;
                margin-top: 1.5em;
                margin-bottom: 0.5em;
            }
            .section-title {
                border-bottom: 2px solid #e0e0e0;
                padding-bottom: 8px;
            }
            h1.section-title {
                font-size: 2.2em;
                border-bottom-width: 3px;
                border-color: #2c3e50;
            }
            h2.section-title {
                font-size: 1.8em;
            }
            h3.section-title {
                font-size: 1.5em;
                border-bottom-style: dashed;
            }
            .calc-item {
                margin-block: 2rem;
                padding: 1rem 1.5rem;
                border-left: 4px solid #3498db;
                background-color: #f8f9fa;
                border-radius: 4px;
            }
            .equation-description {
                font-style: italic;
                color: #555;
                margin-top: 0;
            }
            .equation-container {
                display: flex;
                justify-content: space-between;
                align-items: center;
                overflow-x: auto;
            }
            .equation-latex {
                flex-grow: 1;
            }
            .equation-number {
                color: #888;
                font-size: 0.9em;
                padding-left: 1.5rem;
                align-self: flex-end;
            }
            .paragraph {
                text-align: justify;
                font-size: 1.1em;
            }
            /* Estilos de Tabela e Figura */
            .figure-container, .table-container {
                margin: 2rem auto;
                display: flex;
                flex-direction: column;
                align-items: center;
            }
            figure figcaption, table caption {
                caption-side: bottom;
                font-size: 0.95em;
                color: #555;
                margin-top: 0.5rem;
                font-style: italic;
            }
            table {
              border-collapse: collapse;
              width: 90%;
            }
            td, th {
              border: 1px solid #bdbdbd;
              text-align: center;
              padding: 10px;
            }
            th {
              background-color: #f2f2f2;
              border-bottom: 2px solid #424242;
            }
            table.striped tr:nth-child(even) {
              background-color: #f9f9f9;
            }
            .error-box {
                border: 2px solid #e74c3c;
                background: #fbeae8;
                padding: 1rem;
                border-radius: 4px;
            }

            /* Media query para impressão, como no efficalc */
            @media print {
                body {
                    background-color: #ffffff;
                }
                .page-container {
                    box-shadow: none;
                    margin: 0;
                    padding: 0;
                    max-width: none;
                }
                .calc-item {
                    background-color: #ffffff;
                    border: 1px solid #ccc;
                }
                h1, h2, h3 {
                    page-break-after: avoid;
                }
            }
        """

        # Monta a página HTML final, incluindo o script do MathJax
        # [cite: youandvern/efficalc/efficalc-5920cf5ae2a4920c85adc6d2880faf0e51898ecd/efficalc/report_builder.py]
        html_template = f"""
        <!DOCTYPE html>
        <html lang="pt-BR">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{self._esc(self.metadata.title)}</title>
            <script type="text/javascript" async
              src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/4.0.0-beta.7/tex-mml-chtml.min.js"></script>
            <style>{css_styles}</style>
        </head>
        <body>
            <div class="page-container">
                <header>
                    <h1>{self._esc(self.metadata.title)}</h1>
                    <p><strong>Autor:</strong> {self._esc(self.metadata.author)}</p>
                    <p><strong>Data:</strong> {datetime.now().strftime('%d/%m/%Y')}</p>
                </header>
                <hr>
                <main>
                    {body_content}
                </main>
            </div>
        </body>
        </html>
        """

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(html_template)

            if open_on_save:
                webbrowser.open("file://" + os.path.realpath(filepath))

            self.logger.info(f"Memorial renderizado com sucesso em: {filepath}")
        
        except IOError as e:
            self.logger.error(f"Falha ao salvar o arquivo HTML: {e}")
            raise RenderError(f"Não foi possível salvar o arquivo em {filepath}") from e
        except Exception as e:
            self.logger.error(f"Erro inesperado durante a renderização: {e}")
            raise RenderError(f"Erro desconhecido: {e}") from e

    
    def _render_docx(self, output_path: Path, **kwargs) -> Path:
        """Render to DOCX via pandoc."""
        self._logger.info("Rendering to DOCX via pandoc...")
        
        html_temp = Path(tempfile.mktemp(suffix='.html'))
        self._render_html(html_temp, **kwargs)
        
        import subprocess
        
        try:
            result = subprocess.run(
                ['pandoc', str(html_temp), '-o', str(output_path)],
                capture_output=True,
                text=True,
                check=True
            )
            self._logger.info(f"DOCX generated: {output_path}")
        except FileNotFoundError:
            raise ImportError("Pandoc is required for DOCX export. Install from: https://pandoc.org")
        except subprocess.CalledProcessError as e:
            raise RenderError(f"Pandoc conversion failed: {e.stderr}")
        finally:
            html_temp.unlink(missing_ok=True)
        
        return output_path.resolve()
    
    def _render_latex(self, output_path: Path, **kwargs) -> Path:
        """Render to LaTeX (stub)."""
        self._logger.info("Rendering to LaTeX...")
        
        latex_content = f"""\\documentclass{{article}}
\\usepackage{{amsmath}}
\\usepackage{{graphicx}}
\\title{{{self.metadata.title}}}
\\author{{{self.metadata.author}}}
\\date{{{self.metadata.date}}}
\\begin{{document}}
\\maketitle
% TODO: Implement full content
\\end{{document}}
"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(latex_content)
        
        self._logger.info(f"LaTeX generated: {output_path}")
        
        return output_path.resolve()
    
    def _generate_html_content(self, **kwargs) -> str:
        """Generate HTML content with proper formatting."""
        context = self.get_render_context()
        
        html_parts = []
        
        # Doctype and HTML opening
        html_parts.append('<!DOCTYPE html>')
        html_parts.append(f'<html xmlns="http://www.w3.org/1999/xhtml" lang="{self.metadata.language.value}">')
        
        # Head
        html_parts.append('<head>')
        html_parts.append('<meta charset="UTF-8"/>')
        html_parts.append('<meta name="viewport" content="width=device-width, initial-scale=1.0"/>')
        html_parts.append(f'<title>{escape(self.metadata.title)}</title>')
        
        # Load CSS from template
        css_dir = Path(__file__).parent.parent / 'styles'
        try:
            base_css_path = css_dir / 'base.css'
            template_css_path = css_dir / f'{self.template_name}.css'
            
            css_content = ''
            if base_css_path.exists():
                css_content += base_css_path.read_text(encoding='utf-8')
            if template_css_path.exists():
                css_content += '\n' + template_css_path.read_text(encoding='utf-8')
            
            if css_content:
                html_parts.append('<style>')
                html_parts.append(css_content)
                html_parts.append('</style>')
            else:
                self._logger.warning("No CSS files found, using default")
                html_parts.append(self._get_default_css())
        except Exception as e:
            self._logger.warning(f"Failed to load CSS: {e}")
            html_parts.append(self._get_default_css())
        
        # MathJax for LaTeX
        html_parts.append('<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>')
        html_parts.append('<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>')
        
        html_parts.append('</head>')
        
        # Body
        html_parts.append('<body>')
        html_parts.append('<div class="memorial-container">')
        
        # Render sections with content
        for section in self.sections:
            html_parts.append(self._render_section_to_html(section))
        
        # Render figures
        for figure in self.figures:
            html_parts.append(self._render_figure_to_html(figure))
        
        # Render tables
        for table in self.tables:
            html_parts.append(self._render_table_to_html(table))
        
        html_parts.append('</div>')  # memorial-container
        html_parts.append('</body>')
        html_parts.append('</html>')
        
        return '\n'.join(html_parts)
    
    def _render_section_to_html(self, section: Section) -> str:
        """Render section to HTML with all content."""
        html_parts = []
        
        # Section opening
        level = min(section.level, 6)
        html_parts.append(f'<section class="level-{level}" id="{section.id}">')
        
        # Section title
        if section.title:
            number_part = f"{section.number} " if section.numbered else ""
            title_escaped = escape(section.title)
            html_parts.append(f'<h{level} class="section-title">{number_part}{title_escaped}</h{level}>')
        
        # Section content
        if hasattr(section, 'contents') and section.contents:
            for block in section.contents:
                html_parts.append(self._render_content_block_to_html(block))
        
        html_parts.append('</section>')
        
        return '\n'.join(html_parts)
    
    def _render_content_block_to_html(self, block) -> str:
        """Render content block to HTML."""
        if not hasattr(block, 'type') or not hasattr(block, 'content'):
            return ''
        
        from .base_document import ContentType
        
        content_type = block.type
        content = block.content
        
        if content_type == ContentType.TEXT:
            # Process markdown-like formatting
            text = str(content)
            # Bold
            import re
            text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
            # Italic
            text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)
            return f'<p class="paragraph">{text}</p>'
        
        elif content_type == ContentType.EQUATION:
            # Render LaTeX equation
            if hasattr(content, 'latex'):
                latex = content.latex
                return f'<div class="equation">\\[{latex}\\]</div>'
            else:
                return f'<div class="equation">{escape(str(content))}</div>'
        
        elif content_type == ContentType.CODE:
            code_escaped = escape(str(content))
            return f'<pre class="code"><code>{code_escaped}</code></pre>'
        
        elif content_type == ContentType.LIST:
            items = content if isinstance(content, list) else [content]
            list_html = '<ul class="list">\n'
            for item in items:
                list_html += f'<li>{escape(str(item))}</li>\n'
            list_html += '</ul>'
            return list_html
        
        else:
            return f'<p>{escape(str(content))}</p>'
    
    def _render_figure_to_html(self, figure: Figure) -> str:
        """Render figure to HTML."""
        html_parts = []
        html_parts.append(f'<figure id="{figure.id}" class="figure">')
        
        if figure.path.exists():
            # Embed image as base64 or reference
            html_parts.append(f'<img src="{figure.path}" alt="{escape(figure.caption)}"/>')
        else:
            html_parts.append(f'<p class="missing-figure">[Figure not found: {figure.path}]</p>')
        
        html_parts.append(f'<figcaption>Figura {figure.number}: {escape(figure.caption)}</figcaption>')
        html_parts.append('</figure>')
        
        return '\n'.join(html_parts)
    
    def _render_table_to_html(self, table: Table) -> str:
        """Render table to HTML."""
        html_parts = []
        html_parts.append(f'<div id="{table.id}" class="table-container">')
        html_parts.append(f'<table class="table">')
        
        # Render table data (simplified)
        if hasattr(table, 'data') and table.data:
            # Header
            if hasattr(table, 'headers') and table.headers:
                html_parts.append('<thead><tr>')
                for header in table.headers:
                    html_parts.append(f'<th>{escape(str(header))}</th>')
                html_parts.append('</tr></thead>')
            
            # Body
            html_parts.append('<tbody>')
            for row in table.data:
                html_parts.append('<tr>')
                for cell in row:
                    html_parts.append(f'<td>{escape(str(cell))}</td>')
                html_parts.append('</tr>')
            html_parts.append('</tbody>')
        
        html_parts.append('</table>')
        html_parts.append(f'<p class="table-caption">Tabela {table.number}: {escape(table.caption)}</p>')
        html_parts.append('</div>')
        
        return '\n'.join(html_parts)
    
    def _get_default_css(self) -> str:
        """Get default CSS as fallback."""
        return '''
    <style>
        @page { size: A4; margin: 2.5cm; }
        body { 
            font-family: 'Times New Roman', serif; 
            font-size: 12pt; 
            line-height: 1.6; 
            color: #333; 
        }
        .memorial-container {
            max-width: 210mm;
            margin: 0 auto;
            padding: 20mm;
            background: white;
        }
        .section-title {
            color: #003366;
            border-bottom: 2px solid #003366;
            padding-bottom: 5pt;
            margin-top: 20pt;
            margin-bottom: 15pt;
        }
        h1.section-title { font-size: 18pt; }
        h2.section-title { font-size: 16pt; border-bottom: 1px solid #003366; }
        h3.section-title { font-size: 14pt; border-bottom: 1px solid #ccc; }
        .paragraph {
            text-align: justify;
            margin: 10pt 0;
        }
        .equation {
            display: block;
            text-align: center;
            margin: 20pt 0;
            padding: 15pt;
            background: #f5f5f5;
            border-left: 4px solid #003366;
        }
        .list {
            margin: 10pt 0 10pt 20pt;
        }
        .figure {
            text-align: center;
            margin: 20pt 0;
        }
        .table-container {
            margin: 20pt 0;
        }
        table.table {
            width: 100%;
            border-collapse: collapse;
        }
        table.table th,
        table.table td {
            border: 1px solid #ccc;
            padding: 8pt;
            text-align: left;
        }
        table.table th {
            background: #f0f0f0;
            font-weight: bold;
        }
        strong { font-weight: bold; color: #000; }
    </style>
    '''

