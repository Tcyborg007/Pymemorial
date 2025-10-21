# src/pymemorial/document/memorial.py

from __future__ import annotations

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

from .base_document import (
    BaseDocument,
    DocumentMetadata,
    NormCode,
    ValidationResult,
    ValidationError,
    DocumentValidationError,
    RenderError,
    CrossReferenceType,
    Section,
    Figure,
    Table,
    EquationDoc,
    Verification,
    DocumentLanguage,
)


# ============================================================================
# MEMORIAL CLASS
# ============================================================================

class Memorial(BaseDocument):
    
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
        super().__init__(metadata)
        
        self.template_name = template or self._select_template(metadata.code)
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
    # TEMPLATE SELECTION
    # ========================================================================
    
    def _select_template(self, norm_code: NormCode) -> str:
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
        output_path = Path(output_path)
        self._logger.info(f"Rendering memorial to: {output_path} (format: {format})")
        
        validation = self.validate()
        if not validation.valid:
            self._logger.warning(f"Validation failed with {len(validation.errors)} errors")
            if self.strict_validation:
                raise DocumentValidationError(validation.errors)
        
        if self.auto_title_page:
            self._generate_title_page()
        
        if self.auto_toc:
            self._generate_toc()
        
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
        config: Optional['GeneratorConfig'] = None,
        **kwargs
    ) -> Path:
        from pymemorial.document.generators import (
            WeasyPrintGenerator,
            GeneratorConfig as GConfig
        )
        
        output_path = Path(output_path)
        
        if config is None:
            from pymemorial.document.generators import PDFMetadata
            
            pdf_metadata = PDFMetadata(
                title=self.metadata.title,
                author=self.metadata.author,
                subject=f"Memorial de Cálculo - {self.metadata.project_number}",
                keywords=[
                    str(self.metadata.code),
                    'memorial de cálculo',
                    'estrutural',
                ],
                creation_date=datetime.now(),
            )
            
            config = GConfig(metadata=pdf_metadata)
        
        if generator.lower() == 'weasyprint':
            gen = WeasyPrintGenerator(config)
        elif generator.lower() == 'quarto':
            raise NotImplementedError("Quarto generator not yet implemented")
        elif generator.lower() == 'playwright':
            raise NotImplementedError("Playwright generator not yet implemented")
        elif generator.lower() == 'latex':
            raise NotImplementedError("LaTeX generator not yet implemented")
        else:
            raise ValueError(
                f"Unknown generator: {generator}. "
                "Available: 'weasyprint', 'quarto', 'playwright', 'latex'"
            )
        
        self._logger.info(f"Generating PDF with {generator} generator")
        pdf_path = gen.generate(self, output_path, **kwargs)
        
        return pdf_path


    def validate(self) -> ValidationResult:
        errors: List[ValidationError] = []
        warnings: List[ValidationError] = []
        
        self._logger.info("Validating memorial document...")
        
        errors.extend(self._validate_metadata())
        errors.extend(self._validate_structure())
        errors.extend(self._validate_cross_references())
        warnings.extend(self._validate_figures())
        warnings.extend(self._validate_equations())
        errors.extend(self._validate_verifications())
        
        valid = len(errors) == 0
        result = ValidationResult(valid=valid, errors=errors, warnings=warnings)
        
        self._logger.info(f"Validation complete: {len(errors)} errors, {len(warnings)} warnings")
        
        return result
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': 'Memorial',
            'template': self.template_name,
            'metadata': {
                'title': self.metadata.title,
                'author': self.metadata.author,
                'company': self.metadata.company,
                'code': str(self.metadata.code),
                'project_number': self.metadata.project_number,
                'revision': {
                    'number': self.metadata.revision.number,
                    'date': self.metadata.revision.date.isoformat(),
                    'description': self.metadata.revision.description,
                    'author': self.metadata.revision.author,
                    'approved': self.metadata.revision.approved
                },
                'date': self.metadata.date.isoformat(),
                'language': self.metadata.language.value,
                'keywords': self.metadata.keywords,
                'abstract': self.metadata.abstract
            },
            'sections': [
                {
                    'id': sec.id,
                    'number': sec.number,
                    'title': sec.title,
                    'level': sec.level,
                    'content_preview': sec.content[:100] + '...' if len(sec.content) > 100 else sec.content
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
                    'latex_preview': eq.latex[:50] + '...' if len(eq.latex) > 50 else eq.latex
                }
                for eq in self.equations
            ],
            'verifications': [
                {
                    'id': ver.id,
                    'expression': ver.expression,
                    'passed': ver.passed,
                    'description': ver.description,
                    'norm': str(ver.norm_reference.code),
                    'item': ver.norm_reference.item
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
        errors = []
        
        if not self.metadata.title:
            errors.append(ValidationError(
                element_id='metadata',
                error_type='missing_title',
                message='Document title is required',
                severity='error'
            ))
        
        if not self.metadata.author:
            errors.append(ValidationError(
                element_id='metadata',
                error_type='missing_author',
                message='Document author is required',
                severity='error'
            ))
        
        if not self.metadata.company:
            errors.append(ValidationError(
                element_id='metadata',
                error_type='missing_company',
                message='Company/institution is required',
                severity='error'
            ))
        
        self._logger.debug(f"Metadata validation: {len(errors)} errors")
        return errors
    
    def _validate_structure(self) -> List[ValidationError]:
        errors = []
        
        if len(self.sections) == 0:
            errors.append(ValidationError(
                element_id='structure',
                error_type='empty_document',
                message='Document has no sections',
                severity='error'
            ))
        
        for i, section in enumerate(self.sections):
            if section.level < 1 or section.level > 6:
                errors.append(ValidationError(
                    element_id=section.id,
                    error_type='invalid_level',
                    message=f'Section "{section.title}" has invalid level {section.level}',
                    severity='error'
                ))
        
        self._logger.debug(f"Structure validation: {len(errors)} errors")
        return errors
    
    def _validate_cross_references(self) -> List[ValidationError]:
        errors = []
        
        for from_id, refs in self.cross_refs.items():
            if from_id not in self._element_registry:
                errors.append(ValidationError(
                    element_id=from_id,
                    error_type='invalid_reference',
                    message=f'Cross-reference source not found: {from_id}',
                    severity='error'
                ))
            
            for ref in refs:
                if ref.to_id not in self._element_registry:
                    errors.append(ValidationError(
                        element_id=ref.to_id,
                        error_type='invalid_reference',
                        message=f'Cross-reference target not found: {ref.to_id}',
                        severity='error'
                    ))
        
        self._logger.debug(f"Cross-reference validation: {len(errors)} errors")
        return errors
    
    def _validate_figures(self) -> List[ValidationError]:
        warnings = []
        
        for figure in self.figures:
            if not figure.path.exists():
                warnings.append(ValidationError(
                    element_id=figure.id,
                    error_type='missing_file',
                    message=f'Figure file not found: {figure.path}',
                    severity='warning'
                ))
            elif figure.path.stat().st_size == 0:
                warnings.append(ValidationError(
                    element_id=figure.id,
                    error_type='empty_file',
                    message=f'Figure file is empty: {figure.path}',
                    severity='warning'
                ))
        
        self._logger.debug(f"Figure validation: {len(warnings)} warnings")
        return warnings
    
    def _validate_equations(self) -> List[ValidationError]:
        warnings = []
        
        for equation in self.equations:
            if equation.latex.count('{') != equation.latex.count('}'):
                warnings.append(ValidationError(
                    element_id=equation.id,
                    error_type='invalid_latex',
                    message=f'Equation has unbalanced braces: {equation.number}',
                    severity='warning'
                ))
            
            if '\\frac' in equation.latex and not ('{' in equation.latex and '}' in equation.latex):
                warnings.append(ValidationError(
                    element_id=equation.id,
                    error_type='invalid_latex',
                    message=f'Equation has incomplete \\frac: {equation.number}',
                    severity='warning'
                ))
        
        self._logger.debug(f"Equation validation: {len(warnings)} warnings")
        return warnings
    
    def _validate_verifications(self) -> List[ValidationError]:
        errors = []
        
        for verify in self.verifications:
            if not verify.norm_reference:
                errors.append(ValidationError(
                    element_id=verify.id,
                    error_type='missing_norm',
                    message=f'Verification missing norm reference: {verify.description}',
                    severity='error'
                ))
            
            if not verify.expression:
                errors.append(ValidationError(
                    element_id=verify.id,
                    error_type='missing_expression',
                    message=f'Verification missing expression: {verify.description}',
                    severity='error'
                ))
        
        self._logger.debug(f"Verification validation: {len(errors)} errors")
        return errors
    
    # ========================================================================
    # PRIVATE METHODS - Content Generation
    # ========================================================================
    
    def _generate_title_page(self) -> None:
        if any(s.metadata.get('type') == 'title_page' for s in self.sections):
            self._logger.debug("Title page already exists, skipping generation")
            return
        
        self._logger.info("Generating title page...")
        
        content = f"""
# {self.metadata.title}

**{self.metadata.company}**

---

**Projeto:** {self.metadata.project_number or 'N/A'}  
**Autor:** {self.metadata.author}  
**Data:** {self.metadata.date.strftime('%d/%m/%Y')}  
**Revisão:** {self.metadata.revision.number} - {self.metadata.revision.description}  
**Norma:** {self.metadata.code}

---

{self.metadata.abstract or ''}
"""
        
        section_id = str(uuid.uuid4())
        
        title_section = Section(
            id=section_id,
            title='',
            content=content,
            level=1,
            number='0',
            parent_id=None,
            metadata={'type': 'title_page'}
        )
        
        self.sections.insert(0, title_section)
        self._element_registry[section_id] = title_section
        
        self._logger.info("Title page generated")

    
    def _generate_toc(self) -> None:
        if any(s.metadata.get('type') == 'toc' for s in self.sections):
            self._logger.debug("TOC already exists, skipping generation")
            return
        
        self._logger.info("Generating table of contents...")
        
        toc = self.get_toc()
        
        if not toc:
            self._logger.warning("No sections for TOC")
            return
        
        content = "# Sumário\n\n" if self.metadata.language == DocumentLanguage.PT_BR else "# Table of Contents\n\n"
        
        for entry in toc:
            indent = "  " * (entry['level'] - 1)
            content += f"{indent}- {entry['number']} {entry['title']}\n"
        
        section_id = str(uuid.uuid4())
        
        toc_section = Section(
            id=section_id,
            title='',
            content=content,
            level=1,
            number='0.1',
            parent_id=None,
            metadata={'type': 'toc'}
        )
        
        insert_pos = 1 if self.sections and self.sections[0].metadata.get('type') == 'title_page' else 0
        self.sections.insert(insert_pos, toc_section)
        self._element_registry[section_id] = toc_section
        
        self._logger.info("TOC generated")


    def _generate_list_of_figures(self) -> None:
        figures = self.get_list_of_figures()
        
        if not figures:
            self._logger.debug("No figures to list, skipping list of figures")
            return
        
        self._logger.info(f"Generating list of figures ({len(figures)} items)")
        
        content = []
        content.append("# Lista de Figuras\n\n")
        
        for fig in figures:
            line = f"{fig['number']} – {fig['caption']}"
            
            if fig.get('source'):
                line += f" (Fonte: {fig['source']})"
            
            content.append(f"{line}\n")
        
        import uuid
        section_id = str(uuid.uuid4())
        
        from pymemorial.document.base_document import Section
        list_section = Section(
            id=section_id,
            title='Lista de Figuras',
            content=''.join(content),
            level=1,
            number='0',
            parent_id=None,
            metadata={'type': 'list_of_figures', 'pre_textual': True}
        )
        
        insert_pos = self._find_section_position('toc')
        if insert_pos >= 0:
            self.sections.insert(insert_pos + 1, list_section)
        else:
            self.sections.insert(0, list_section)
        
        self._element_registry[section_id] = list_section
        self._logger.debug(f"List of figures added at position {insert_pos + 1}")
    
    def _generate_list_of_tables(self) -> None:
        tables = self.get_list_of_tables()
        
        if not tables:
            self._logger.debug("No tables to list, skipping list of tables")
            return
        
        self._logger.info(f"Generating list of tables ({len(tables)} items)")
        
        content = []
        content.append("# Lista de Tabelas\n\n")
        
        for tbl in tables:
            line = f"{tbl['number']} – {tbl['caption']}"
            
            rows = tbl.get('rows', '?')
            cols = tbl.get('cols', '?')
            line += f" ({rows}×{cols})"
            
            if tbl.get('source'):
                line += f" — Fonte: {tbl['source']}"
            
            content.append(f"{line}\n")
        
        import uuid
        section_id = str(uuid.uuid4())
        
        from pymemorial.document.base_document import Section
        list_section = Section(
            id=section_id,
            title='Lista de Tabelas',
            content=''.join(content),
            level=1,
            number='0',
            parent_id=None,
            metadata={'type': 'list_of_tables', 'pre_textual': True}
        )
        
        insert_pos = self._find_section_position('list_of_figures')
        if insert_pos < 0:
            insert_pos = self._find_section_position('toc')
        
        if insert_pos >= 0:
            self.sections.insert(insert_pos + 1, list_section)
        else:
            self.sections.insert(0, list_section)
        
        self._element_registry[section_id] = list_section
        self._logger.debug(f"List of tables added at position {insert_pos + 1}")
    
    def _generate_list_of_equations(self) -> None:
        equations = self.get_list_of_equations()
        
        if not equations:
            self._logger.debug("No equations to list, skipping list of equations")
            return
        
        self._logger.info(f"Generating list of equations ({len(equations)} items)")
        
        content = []
        content.append("# Lista de Equações\n\n")
        
        for eq in equations:
            line = f"{eq['number']} – {eq['description']}"
            
            if eq.get('reference'):
                line += f" ({eq['reference']})"
            
            content.append(f"{line}\n")
        
        import uuid
        section_id = str(uuid.uuid4())
        
        from pymemorial.document.base_document import Section
        list_section = Section(
            id=section_id,
            title='Lista de Equações',
            content=''.join(content),
            level=1,
            number='0',
            parent_id=None,
            metadata={'type': 'list_of_equations', 'pre_textual': True}
        )
        
        insert_pos = self._find_section_position('list_of_tables')
        if insert_pos < 0:
            insert_pos = self._find_section_position('list_of_figures')
        if insert_pos < 0:
            insert_pos = self._find_section_position('toc')
        
        if insert_pos >= 0:
            self.sections.insert(insert_pos + 1, list_section)
        else:
            self.sections.insert(0, list_section)
        
        self._element_registry[section_id] = list_section
        self._logger.debug(f"List of equations added at position {insert_pos + 1}")
    
    def _find_section_position(self, section_type: str) -> int:
        for i, section in enumerate(self.sections):
            if section.metadata.get('type') == section_type:
                return i
        return -1


    # ========================================================================
    # PRIVATE METHODS - Rendering
    # ========================================================================
    
    def _render_pdf(self, output_path: Path, **kwargs) -> Path:
        self._logger.info("Rendering to PDF via WeasyPrint...")
        
        try:
            import weasyprint
        except ImportError:
            raise ImportError(
                "WeasyPrint is required for PDF rendering. "
                "Install with: pip install weasyprint"
            )
        
        import tempfile
        import os
        
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
    
    def _render_html(self, output_path: Path, **kwargs) -> Path:
        self._logger.info("Rendering to HTML5...")
        
        html_content = self._generate_html_content(**kwargs)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        self._logger.info(f"HTML generated: {output_path}")
        
        return output_path.resolve()
    
    def _render_docx(self, output_path: Path, **kwargs) -> Path:
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
        self._logger.info("Rendering to LaTeX...")
        
        latex_content = self._generate_latex_content(**kwargs)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(latex_content)
        
        self._logger.info(f"LaTeX generated: {output_path}")
        
        return output_path.resolve()
    
    def _generate_html_content(self, **kwargs) -> str:
        context = self.get_render_context()
        
        html_parts = []
        
        html_parts.append('<!DOCTYPE html>')
        html_parts.append(f'<html xmlns="http://www.w3.org/1999/xhtml" lang="{self.metadata.language.value}">')
        
        html_parts.append('<head>')
        html_parts.append('<meta charset="UTF-8"/>')
        html_parts.append('<meta name="viewport" content="width=device-width, initial-scale=1.0"/>')
        html_parts.append(f'<title>{escape(self.metadata.title)}</title>')
        
        html_parts.append('''
    <style>
        @page {
            size: A4;
            margin: 2.5cm;
        }
        body {
            font-family: 'DejaVu Sans', Arial, sans-serif;
            font-size: 11pt;
            line-height: 1.6;
            color: #333;
        }
        h1 {
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
            page-break-after: avoid;
        }
        h2 {
            color: #34495e;
            margin-top: 30px;
            page-break-after: avoid;
        }
        h3 {
            color: #7f8c8d;
            page-break-after: avoid;
        }
        p {
            margin: 1em 0;
            text-align: justify;
        }
        .figure {
            text-align: center;
            margin: 20px 0;
            page-break-inside: avoid;
        }
        .figure img {
            max-width: 100%;
            height: auto;
        }
        .figure-caption {
            font-style: italic;
            color: #7f8c8d;
            margin-top: 10px;
            font-size: 0.9em;
        }
        .verification {
            padding: 10px;
            margin: 10px 0;
            border-radius: 5px;
            page-break-inside: avoid;
        }
        .verification.pass {
            background-color: #d4edda;
            border-left: 4px solid #28a745;
        }
        .verification.fail {
            background-color: #f8d7da;
            border-left: 4px solid #dc3545;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 1em 0;
            page-break-inside: avoid;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        th {
            background-color: #f2f2f2;
        }
    </style>
    ''')
        
        html_parts.append('</head>')
        
        html_parts.append('<body>')
        
        for section in self.sections:
            section_type = section.metadata.get('type', '')
            
            if section.title and section_type not in ('title_page', 'toc'):
                level = min(section.level, 6)
                title_escaped = escape(section.title)
                number_escaped = escape(section.number)
                html_parts.append(f'<h{level}>{number_escaped} {title_escaped}</h{level}>')
            
            if section.content:
                content = section.content
                
                paragraphs = content.split('\n\n')
                for para in paragraphs:
                    if para.strip():
                        para_escaped = escape(para.strip())
                        para_html = para_escaped.replace('\n', '<br/>')
                        html_parts.append(f'<p>{para_html}</p>')
        
        for figure in self.figures:
            from pathlib import Path
            
            fig_path = Path(figure.path).resolve()
            
            if fig_path.exists():
                fig_uri = fig_path.as_uri()
                
                alt_text = escape(figure.alt_text or figure.caption)
                caption = escape(f'{figure.number}: {figure.caption}')
                
                html_parts.append(f'''
<div class="figure">
    <img src="{fig_uri}" alt="{alt_text}"/>
    <div class="figure-caption">{caption}</div>
</div>
                ''')
            else:
                self._logger.warning(f"Figure not found: {fig_path}")
        
        for table in self.tables:
            html_parts.append('<table>')
            
            if table.headers:
                html_parts.append('<thead><tr>')
                for header in table.headers:
                    html_parts.append(f'<th>{escape(str(header))}</th>')
                html_parts.append('</tr></thead>')
            
            html_parts.append('<tbody>')
            for row in table.data:
                html_parts.append('<tr>')
                for cell in row:
                    html_parts.append(f'<td>{escape(str(cell))}</td>')
                html_parts.append('</tr>')
            html_parts.append('</tbody>')
            
            html_parts.append('</table>')
            
            if table.caption:
                caption = escape(f'{table.number}: {table.caption}')
                html_parts.append(f'<p class="figure-caption">{caption}</p>')
        
        html_parts.append('</body>')
        html_parts.append('</html>')
        
        html = '\n'.join(html_parts)
        
        return html
    
    def _generate_latex_content(self, **kwargs) -> str:
        return f"""\\documentclass{{article}}
\\usepackage{{amsmath}}
\\usepackage{{graphicx}}
\\title{{{self.metadata.title}}}
\\author{{{self.metadata.author}}}
\\date{{{self.metadata.date.strftime('%Y-%m-%d')}}}
\\begin{{document}}
\\maketitle
% TODO: Implement full content
\\end{{document}}
"""
    
    def _open_file(self, path: Path) -> None:
        import platform
        import subprocess
        
        system = platform.system()
        
        try:
            if system == 'Windows':
                import os
                os.startfile(path)
            elif system == 'Darwin':
                subprocess.run(['open', path])
            else:
                subprocess.run(['xdg-open', path])
            
            self._logger.info(f"Opened file: {path}")
        except Exception as e:
            self._logger.warning(f"Failed to open file: {e}")
    
    def _validate_html(self, html_content: str) -> bool:
        try:
            from lxml import etree
            
            parser = etree.HTMLParser()
            tree = etree.fromstring(html_content.encode('utf-8'), parser)
            
            for elem in tree.iter():
                if elem.tag is None or not isinstance(elem.tag, str):
                    self._logger.error(f"Invalid tag found: {elem}")
                    return False
                
                if elem.tag.startswith('{'):
                    if '}' not in elem.tag:
                        self._logger.error(f"Malformed namespace in tag: {elem.tag}")
                        return False
            
            self._logger.debug("HTML validation passed")
            return True
            
        except Exception as e:
            self._logger.error(f"HTML validation failed: {e}")
            return False

    # ========================================================================
    # STRING REPRESENTATIONS
    # ========================================================================
    
    def __repr__(self) -> str:
        return (
            f"Memorial("
            f"title='{self.metadata.title}', "
            f"template='{self.template_name}', "
            f"sections={len(self.sections)}, "
            f"figures={len(self.figures)}, "
            f"verifications={len(self.verifications)}"
            f")"
        )
    
    def __str__(self) -> str:
        passed = sum(1 for v in self.verifications if v.passed)
        total = len(self.verifications)
        return (
            f"Memorial: {self.metadata.title}\n"
            f"  Template: {self.template_name}\n"
            f"  Sections: {len(self.sections)}\n"
            f"  Figures: {len(self.figures)}\n"
            f"  Verifications: {passed}/{total} passed"
        )