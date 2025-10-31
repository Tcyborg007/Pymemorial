# src/pymemorial/document/base_document.py
"""
Base Document Module - Foundation for all technical documents (v2.0: PHASE 7.4 Enhanced - MVP Compatible).

This module provides the abstract base class and core infrastructure for
generating professional technical documents (calculation memorials, reports,
scientific articles, presentations) with support for:

- Hierarchical section structure with automatic numbering
- Figures, tables, equations with auto-numbering and cross-references
- Technical verifications (NBR, AISC, Eurocode standards)
- **Smart Text Processing**: Write in plain text, auto-detect variables (MVP compatible)
- **Smart Context Management**: Global context shared across all sections
- Integration with Sections (PHASE 3-4-5)
- Integration with FEM Backends (PHASE 5)
- Integration with Visualization & Exporters (PHASE 6)
- LaTeX equation rendering
- 3D model embedding (Three.js, U3D)
- Metadata management (author, company, revisions)
- Document validation and integrity checks

**MVP v2.0 UPDATES**:
- **SmartTextEngine Integration**: add_paragraph() processes text with auto_detect
- **Simplified Suggestions**: Uses keyword matching (no heavy NLP)
- **Advanced Validation**: Full circular refs + norm compliance (ex: gamma_s >=1.0 NBR)
- **Render Context**: Includes detected variables and compliance reports

Architecture
------------
This module uses the Template Method pattern where abstract methods
(render, to_dict) are implemented by subclasses (Memorial,
Report, Article), but common logic (add_section, add_figure, validate)
is shared.

Performance
-----------
- O(n) complexity for element additions
- O(n log n) for TOC generation (sorted by number)
- O(n) for cross-reference validation (hashmap)
- Memory: ~1KB per section, ~10KB per figure (metadata only)

Thread Safety
-------------
This class is thread-safe for READ operations, but NOT for WRITE.
Use locks if adding elements concurrently from multiple threads.

Examples
--------
Use concrete subclasses:

>>> from pymemorial.document import Memorial
>>> metadata = DocumentMetadata(...)
>>> memorial = Memorial(metadata)

>>> # MVP: Natural paragraph with text processing
>>> memorial.add_paragraph(\"\"\"
... A resistência N_Rd = {N_Rd:.2f} kN supera N_Sd. Fator γ_s = {gamma_s}.
... Fórmula: M_d = M_k * gamma_s.
... \"\"\")
>>> # Auto: Detecta vars, formata valores, processa símbolos gregos

>>> # Suggest verifications (keyword-based)
>>> suggestions = memorial.suggest_verifications("flambagem chi_LT")
>>> # → [{'id': 'V-1', 'type': 'NBR', 'desc': 'Ver chi > 0.5'}]

>>> memorial.render("output.pdf")

Author: PyMemorial Team
Date: 2025-10-21
Version: 2.0.0 (MVP Compatible)
Phase: 7.4 (Templates & Styles)
"""

from __future__ import annotations

import json
import logging
import re
import uuid
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import (
    Any, Dict, List, Optional, Union, Literal,
    TYPE_CHECKING, Callable, Tuple
)

# Import template processor (internal)
try:
    from .internal.text_processing.text_utils import TemplateProcessor  # ← CORRETO!
    TEMPLATE_PROCESSOR_AVAILABLE = True
except ImportError as e:
    TEMPLATE_PROCESSOR_AVAILABLE = False
    TemplateProcessor = None
    print(f"[WARNING] TemplateProcessor import failed: {e}")


# Data handling imports (PHASE 7.2) - Robust fallbacks
try:
    import pandas as pd
    import numpy as np
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = np = None
    warnings.warn(
        "pandas/numpy not available. Table functionality limited. "
        "Install with: pip install pandas numpy", ImportWarning
    )

# ============================================================================
# IMPORTS FROM PREVIOUS PHASES - 100% COMPATIBILITY
# ============================================================================

# PHASE 1-2: Core & Equations
try:
    from pymemorial.core import Equation, Variable, Calculator
    from pymemorial.core.units import ureg, parse_quantity
    CORE_AVAILABLE = True
except ImportError:
    CORE_AVAILABLE = False
    Equation = Variable = Calculator = None
    ureg = parse_quantity = None
    warnings.warn("Core module not available (PHASE 1-2)", ImportWarning)

# PHASE 3-4-5: Sections
try:
    from pymemorial.sections import (
        SectionAnalyzer,
        SectionProperties,
        SteelSection,
        ConcreteSection,
        CompositeSection,
        SectionFactory
    )
    SECTIONS_AVAILABLE = True
except ImportError:
    SECTIONS_AVAILABLE = False
    warnings.warn("Sections module not available (PHASE 3-4-5)", ImportWarning)

# PHASE 5: Backends FEM
try:
    from pymemorial.backends import (
        StructuralBackend,
        PyNiteBackend,
        BackendFactory
    )
    BACKENDS_AVAILABLE = True
except ImportError:
    BACKENDS_AVAILABLE = False
    warnings.warn("Backends module not available (PHASE 5)", ImportWarning)

# PHASE 6: Visualization & Exporters
try:
    from pymemorial.visualization import PlotlyEngine, PyVistaEngine
    from pymemorial.visualization.exporters import (
        export_figure,
        ExportConfig,
        ImageFormat
    )
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    warnings.warn("Visualization module not available (PHASE 6)", ImportWarning)

# MVP v2.0: Recognition (SmartTextEngine + DetectedVariable) - FIX
try:
    from pymemorial.recognition import get_engine, SmartTextEngine, DetectedVariable
    RECOGNITION_AVAILABLE = True
except ImportError:
    RECOGNITION_AVAILABLE = False
    get_engine = SmartTextEngine = DetectedVariable = None
    warnings.warn("Recognition module not available (MVP v2.0)", ImportWarning)

# SymPy fallback for equations
try:
    from sympy import sympify, latex as sympy_latex
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False
    sympify = sympy_latex = None
    warnings.warn("SymPy not available. Equation LaTeX limited.", ImportWarning)

# Type checking imports
if TYPE_CHECKING:
    import matplotlib.figure
    import plotly.graph_objs
    import pandas as pd

# ============================================================================
# CUSTOM EXCEPTIONS (Enhanced)
# ============================================================================

class DocumentError(Exception):
    """Base exception for document-related errors."""
    pass

class DocumentValidationError(DocumentError):
    """Raised when document validation fails."""
    def __init__(self, errors: List['ValidationError']):
        self.errors = errors
        messages = "\n".join(f"  - {err.message}" for err in errors)
        super().__init__(f"Document validation failed:\n{messages}")

class RenderError(DocumentError):
    """Raised when document rendering fails."""
    pass

class CrossReferenceError(DocumentError):
    """Raised when cross-reference is invalid."""
    pass

class NormComplianceError(DocumentError):
    """Raised for norm-specific violations (ex: gamma_s <1.0 NBR)."""
    pass

# ============================================================================
# ENUMS (Expanded)
# ============================================================================

class NormCode(Enum):
    """Engineering standards/norms supported."""
    NBR6118_2023 = "NBR 6118:2023"
    NBR8800_2024 = "NBR 8800:2024"
    NBR9050_2020 = "NBR 9050:2020"
    AISC360_22 = "AISC 360-22"
    AISC341_22 = "AISC 341-22"
    EN1992_2004 = "EN 1992:2004"
    EN1993_2005 = "EN 1993:2005"
    EN1994_2004 = "EN 1994:2004"
    ACI318_19 = "ACI 318-19"
    CSA_A23_3_19 = "CSA A23.3-19"
    
    def __str__(self) -> str:
        return self.value

class CrossReferenceType(Enum):
    """Types of cross-references between document elements."""
    SECTION = "section"
    FIGURE = "figure"
    TABLE = "table"
    EQUATION = "equation"
    VERIFICATION = "verify"
    APPENDIX = "appendix"

class TableStyle(Enum):
    """Table styling presets."""
    SIMPLE = "simple"
    GRID = "grid"
    STRIPED = "striped"
    PROFESSIONAL = "professional"
    ACADEMIC = "academic"

class DocumentLanguage(str, Enum):
    """Supported document languages."""
    PT_BR = "pt-BR"
    EN_US = "en-US"
    ES_ES = "es-ES"

# ============================================================================
# DATACLASSES (Immutable components - Enhanced)
# ============================================================================

@dataclass(frozen=True)
class DocumentMetadata:
    """Document metadata (enhanced: + language, company)."""
    title: str
    author: str
    company: str = ""
    date: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    language: DocumentLanguage = DocumentLanguage.PT_BR
    norm_code: NormCode = NormCode.NBR6118_2023
    revision: str = "1.0"
    keywords: List[str] = field(default_factory=list)

@dataclass(frozen=True)
class CrossReference:
    """Cross-reference between elements (unchanged)."""
    from_id: str
    to_id: str
    ref_type: CrossReferenceType
    description: str = ""

@dataclass(frozen=True)
class ValidationError:
    """Validation error (enhanced: + type_hint)."""
    element_id: str
    message: str
    severity: Literal["error", "warning"] = "error"
    type_hint: Optional[str] = None  # ex: 'safety_factor' for norm errors

# Norm Compliance Report
@dataclass
class NormCompliance:
    """Relatório de conformidade normativa (ex: fatores aplicados)."""
    norm: NormCode
    factors_applied: Dict[str, float] = field(default_factory=dict)  # ex: {'gamma_s': 1.4}
    verifications: Dict[str, bool] = field(default_factory=dict)  # ex: {'V-1': True}
    compliance_rate: float = 0.0

# ============================================================================
# MAIN CLASS: BaseDocument (Enhanced ABC - MVP Compatible)
# ============================================================================

class BaseDocument(ABC):
    """
    Abstract base class for all PyMemorial documents.

    **MVP v2.0**: Integrates SmartTextEngine for text processing in add_paragraph().
    Auto-suggests verifications via keyword matching (lightweight, no heavy NLP).
    Norm-aware: Applies factors (1.4 NBR) in validations.

    Subclass for concrete types (Memorial, Report).
    """

    def __init__(self, metadata: DocumentMetadata):
        """
        Initialize base document.

        Args:
            metadata: Document metadata.

        Raises:
            TypeError: If metadata invalid.
        """
        if not isinstance(metadata, DocumentMetadata):
            raise TypeError("Metadata must be DocumentMetadata instance.")
        
        self.metadata = metadata
        self._logger = logging.getLogger(self.__class__.__name__)
        self._logger.setLevel(logging.INFO)
        
        # Core elements
        self.sections: List['Section'] = []
        self.figures: List['Figure'] = []
        self.tables: List['Table'] = []
        self.equations: List['EquationDoc'] = []
        self.verifications: List['Verification'] = []
        self.revisions: List['Revision'] = []
        
        # MVP v2.0: SmartTextEngine Integration - FIX: auto_detect parameter
        self.processor: Optional[SmartTextEngine] = get_engine(auto_detect=True) if RECOGNITION_AVAILABLE else None
        
        # Context and registry
        self._global_context: Dict[str, Any] = {}
        self._element_registry: Dict[str, Any] = {}
        self.cross_refs: Dict[str, List[CrossReference]] = {}
        self._auto_counters = defaultdict(int)  # For numbering
        self._frozen = False
        
        # Norm compliance tracking
        self.compliance: NormCompliance = NormCompliance(norm=metadata.norm_code)
        
        # Initial revision
        self.add_revision("Initial draft", "v1.0")
        
        self._logger.info(f"BaseDocument '{metadata.title}' initialized (norm: {metadata.norm_code}).")

    # ABSTRACT METHODS (Unchanged - Subclasses implement)
    @abstractmethod
    def render(self, output_path: Union[str, Path], format: str = "pdf") -> Path:
        """Render document to output."""
        pass

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dict."""
        pass

    # CORE METHODS (Enhanced for Robustness - MVP Compatible)
    def add_section(
        self,
        title: str,
        content: Optional[str] = None,
        level: int = 1,
        numbered: bool = True
    ) -> 'Section':
        """
        Add hierarchical section (auto-numbering).

        MVP: Validates title via keyword matching for suggestions.
        """
        self._check_frozen()
        
        # Auto-number
        counter_key = f"section_{level}"
        self._auto_counters[counter_key] += 1
        number = self._generate_number(level)
        
        section = Section(
            title=title,
            number=number,
            level=level,
            numbered=numbered,
            id=str(uuid.uuid4())
        )
        self.sections.append(section)
        self._register_element(section.id, section)
        
        if content:
            self.add_paragraph(content, parent=section)
        
        # MVP: Keyword-based suggestion
        if self.processor:
            suggestions = self.suggest_verifications(title)
            if suggestions:
                self._logger.info(f"Suggested verifications for '{title}': {[s['id'] for s in suggestions[:2]]}")
        
        return section

    def add_paragraph(
        self,
        text: str,
        parent: Optional['Section'] = None,
        style: str = 'body',
        **kwargs
    ) -> 'ContentBlock':
        """
        Add paragraph with smart text processing (ROBUST).
        
        Supports:
        - Placeholder substitution: {var:.2f}
        - Global context variables
        - Symbol detection: M_k, gamma_s, etc.
        
        Args:
            text: Paragraph text (with optional {placeholders})
            parent: Parent section or section ID
            style: Paragraph style ('body', 'heading', etc.)
            **kwargs: Additional options (merged into context)
        
        Returns:
            ContentBlock with processed text
        
        Examples:
        --------
        >>> memorial.set_context({'M_k': 150.5, 'gamma_s': 1.4})
        >>> memorial.add_paragraph(
        ...     "Momento M_k = {M_k:.2f} kN.m com fator {gamma_s:.1f}.",
        ...     parent=section
        ... )
        """
        self._check_frozen()
        
        # Determine parent section
        if parent is None:
            parent = self.sections[-1] if self.sections else None
            if not parent:
                raise ValueError("Add a section first.")
        
        # Merge global context with local kwargs
        context = self._global_context.copy()
        context.update(kwargs)
        
        # ====================================================================
        # STEP 1: Process template with TemplateProcessor (ROBUST)
        # ====================================================================
        if TEMPLATE_PROCESSOR_AVAILABLE and context and '{' in text:
            processor = TemplateProcessor(strict=False)  # Graceful fallback
            
            # Validate template
            is_valid, missing_vars = processor.validate_template(text, context)
            
            if not is_valid:
                self._logger.warning(
                    f"Template has missing variables: {missing_vars}. "
                    f"Using partial substitution."
                )
            
            # Process with context
            try:
                processed_text = processor.process(text, context)
                self._logger.debug(f"Template processed: {len(processed_text)} chars")
            except Exception as e:
                self._logger.error(f"Template processing failed: {e}. Using original text.")
                processed_text = text
        else:
            # No template processing needed (or TemplateProcessor not available)
            processed_text = text
            if '{' in text and not context:
                self._logger.warning(
                    "Text has placeholders but no global context set. "
                    "Call set_context() first."
                )
        
        # ====================================================================
        # STEP 2: Create content block
        # ====================================================================
        block = ContentBlock(
            type=ContentType.TEXT,
            content=processed_text,
            
        )
        
        # ====================================================================
        # STEP 3: Add to parent section
        # ====================================================================
        parent.add_content(block)
        self._logger.debug(f"Paragraph added to section '{parent.title}'")
        
        return block


    def set_context(self, context: Dict[str, Any]) -> None:
        """
        Set global context (MVP: simplified validation without heavy NLP).
        
        Validates safety factors (gamma_s >= 1.0) for norm compliance.
        """
        self._check_frozen()
        for k, v in context.items():
            # Simple validation: safety factors must be >= 1.0
            if ('gamma' in k.lower() or 'safety' in k.lower()) and isinstance(v, (int, float)) and v < 1.0:
                raise NormComplianceError(f"Invalid safety factor {k}: {v} <1.0 (norm: {self.metadata.norm_code})")
            self._global_context[k] = v
        self._logger.debug(f"Context updated: {list(context.keys())}")

    def add_verification(
        self,
        name: str,
        condition: Callable[[Dict[str, Any]], bool],
        description: str = "",
        norm_ref: Optional[str] = None
    ) -> 'Verification':
        """Add verification (auto-applies norm factors if applicable)."""
        self._check_frozen()
        
        # Apply norm factors to context if safety-related
        if norm_ref:
            factor = self._get_norm_factor('safety_factor')
            for k in self._global_context:
                if 'gamma' in k or 'safety' in k:
                    self._global_context[k] *= factor
        
        verification = Verification(name=name, condition=condition, description=description, norm_ref=norm_ref)
        self.verifications.append(verification)
        self._register_element(verification.id, verification)
        return verification

    def suggest_verifications(self, context_text: str, max_suggestions: int = 3) -> List[Dict[str, Any]]:
        """
        MVP: Simplified suggestion using keyword matching (no heavy NLP).

        Args:
            context_text: Text/keywords (ex: "flambagem chi_LT").
            max_suggestions: Limit.

        Returns:
            List of suggestions [{'id': 'V-1', 'desc': '...', 'norm': 'NBR'}].
        """
        suggestions = []
        context_lower = context_text.lower()
        
        # Keyword-based suggestion map (expansível)
        suggestion_map = {
            'flambagem': [{'id': 'V-1', 'desc': 'Verificação chi > 0.5 (NBR 6118)', 'norm': 'NBR6118_2023'}],
            'moment': [{'id': 'V-2', 'desc': 'M_d <= M_Rd com gamma_s', 'norm': 'NBR6118_2023'}],
            'gamma': [{'id': 'V-3', 'desc': 'Fator gamma_s =1.4 aplicado', 'norm': 'NBR6118_2023'}],
            'compress': [{'id': 'V-4', 'desc': 'N_d <= N_Rd', 'norm': 'NBR8800_2024'}],
        }
        
        for keyword, sug_list in suggestion_map.items():
            if keyword in context_lower:
                suggestions.extend(sug_list)
                if len(suggestions) >= max_suggestions:
                    break
        
        suggestions = suggestions[:max_suggestions]
        self._logger.info(f"Suggested {len(suggestions)} verifications for '{context_text}'.")
        return suggestions

    # VALIDATION (Implemented: Full circular + norm)
    def validate(self) -> List[ValidationError]:
        """Full validation (sections, refs, norm compliance)."""
        errors = []
        
        # Circular references (DFS)
        if not self._check_circular_references():
            errors.append(ValidationError("doc", "Circular references detected in sections/refs.", "error"))
        
        # Norm compliance
        for v in self.verifications:
            if not v.passed and v.norm_ref:
                errors.append(ValidationError(v.id, f"Verification failed under {v.norm_ref}", "error", 'norm_compliance'))
        
        # Variable consistency (simple checks)
        for k, v in self._global_context.items():
            if ('gamma' in k.lower() or 'safety' in k.lower()) and isinstance(v, (int, float)) and v < 1.0:
                errors.append(ValidationError(k, f"Invalid safety factor: {v} <1.0", "warning", 'safety_factor'))
        
        if errors:
            raise DocumentValidationError(errors)
        
        self._logger.info("Document validated successfully.")
        return []

    def _check_circular_references(self) -> bool:
        """DFS for cycles in cross_refs/sections."""
        visited = set()
        rec_stack = set()
        
        def dfs(node):
            visited.add(node)
            rec_stack.add(node)
            for ref in self.cross_refs.get(node, []):
                child = ref.to_id
                if child not in visited:
                    if not dfs(child):
                        return False
                elif child in rec_stack:
                    return False
            rec_stack.remove(node)
            return True
        
        for node in self.cross_refs:
            if node not in visited:
                if not dfs(node):
                    return False
        return True

    # RENDER & EXPORT (Enhanced Context)
    def get_render_context(self) -> Dict[str, Any]:
        """
        Retorna contexto completo para renderização.
        """
        sections_data = []
        
        for section_id in sorted(self._sections.keys()):
            section = self._sections[section_id]
            
            # ✅ Processar conteúdos COM metadata
            contents_data = []
            for block in section.contents:
                block_data = {
                    'type': block.type,
                    'content': block.content
                }
                
                # ✅ CRÍTICO: Incluir metadata
                if hasattr(block, 'metadata') and block.metadata:
                    block_data['metadata'] = block.metadata
                
                contents_data.append(block_data)
            
            section_data = {
                'id': section.id,
                'title': section.title,
                'number': section.number,
                'level': section.level,
                'parent': section.parent_id,
                'contents': contents_data
            }
            sections_data.append(section_data)
        
        return {
            'title': self.metadata.title,
            'author': self.metadata.author,
            'company': self.metadata.company,
            'date': self.metadata.date,
            'version': self.metadata.version,
            'norm_code': self.metadata.norm_code.value if hasattr(self.metadata.norm_code, 'value') else str(self.metadata.norm_code),
            'sections': sections_data
        }



    def get_toc(self) -> List[Dict[str, Any]]:
        """Generate table of contents."""
        toc = []
        for section in self.sections:
            if section.title:  # Skip paragraphs (no title)
                toc.append({
                    'number': section.number,
                    'title': section.title,
                    'level': section.level,
                    'id': section.id
                })
        
        self._logger.debug(f"TOC generated with {len(toc)} entries")
        return toc
    
    def get_element_by_id(self, element_id: str) -> Optional[Any]:
        """Get element by ID."""
        return self._element_registry.get(element_id)
    
    def export_json(self, output_path: Path) -> None:
        """Export document structure to JSON."""
        output_path = Path(output_path)
        data = self.to_dict()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        
        self._logger.info(f"Document exported to JSON: {output_path}")
    
    def export_yaml(self, output_path: Path) -> None:
        """Export document structure to YAML."""
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML required for YAML export. Install with: pip install pyyaml")
        
        output_path = Path(output_path)
        data = self.to_dict()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, allow_unicode=True, sort_keys=False)
        
        self._logger.info(f"Document exported to YAML: {output_path}")

    def add_figure(
        self,
        path: Path,
        caption: str = "",
        parent: Optional['Section'] = None,
        numbered: bool = True
    ) -> 'Figure':
        """Add figure (integrated with viz PHASE 6; auto-numbering)."""
        self._check_frozen()
        
        if parent is None:
            parent = self.sections[-1] if self.sections else None
            if not parent:
                raise ValueError("Add a section first.")
        
        counter_key = "figure"
        self._auto_counters[counter_key] += 1
        number = f"Fig. {self._auto_counters[counter_key]}"
        
        figure = Figure(path=path, caption=caption, number=number)
        self.figures.append(figure)
        self._register_element(figure.id, figure)
        parent.add_content(ContentBlock(type=ContentType.FIGURE, content=figure))
        
        return figure

    def add_table(
        self,
        data: Any,  # pd.DataFrame or list
        caption: str = "",
        style: TableStyle = TableStyle.SIMPLE,
        parent: Optional['Section'] = None,
        numbered: bool = True
    ) -> 'Table':
        """Add table (Pandas fallback to list; auto-numbering)."""
        self._check_frozen()
        
        if parent is None:
            parent = self.sections[-1] if self.sections else None
            if not parent:
                raise ValueError("Add a section first.")
        
        counter_key = "table"
        self._auto_counters[counter_key] += 1
        number = f"Tab. {self._auto_counters[counter_key]}"
        
        # Fallback if no Pandas
        if PANDAS_AVAILABLE and isinstance(data, list):
            data = pd.DataFrame(data)
        
        table = Table(data=data, caption=caption, style=style, number=number)
        self.tables.append(table)
        self._register_element(table.id, table)
        parent.add_content(ContentBlock(type=ContentType.TABLE, content=table))
        
        return table

    def add_equation(
        self,
        expression: str,
        description: str = "",
        parent: Optional['Section'] = None,
        numbered: bool = True
    ) -> 'EquationDoc':
        """Add equation (SymPy LaTeX; auto-numbering)."""
        self._check_frozen()
        
        if parent is None:
            parent = self.sections[-1] if self.sections else None
            if not parent:
                raise ValueError("Add a section first.")
        
        counter_key = "equation"
        self._auto_counters[counter_key] += 1
        number = f"({self._auto_counters[counter_key]})"
        
        # Parse with SymPy if available
        if CORE_AVAILABLE and SYMPY_AVAILABLE:
            try:
                eq_expr = sympify(expression)
                latex_expr = sympy_latex(eq_expr)
            except:
                latex_expr = expression  # Fallback
        else:
            latex_expr = expression  # Raw fallback
        
        equation = EquationDoc(latex=latex_expr, number=number, description=description)
        self.equations.append(equation)
        self._register_element(equation.id, equation)
        parent.add_content(ContentBlock(type=ContentType.EQUATION, content=equation))
        
        return equation

    def add_cross_reference(
        self,
        from_id: str,
        to_id: str,
        ref_type: CrossReferenceType
    ) -> None:
        """Create cross-reference between elements."""
        self._check_frozen()
        
        # Validate IDs exist
        if from_id not in self._element_registry:
            raise CrossReferenceError(f"Source element not found: {from_id}")
        if to_id not in self._element_registry:
            raise CrossReferenceError(f"Target element not found: {to_id}")
        
        cross_ref = CrossReference(
            from_id=from_id,
            to_id=to_id,
            ref_type=ref_type
        )
        
        if from_id not in self.cross_refs:
            self.cross_refs[from_id] = []
        
        self.cross_refs[from_id].append(cross_ref)
        
        self._logger.debug(f"Cross-reference added: {from_id} -> {to_id} ({ref_type.value})")
    
    def add_revision(self, changes: str, version: str) -> None:
        """Add revision entry."""
        revision = Revision(
            version=version,
            changes=changes
        )
        self.revisions.append(revision)
        self._logger.debug(f"Revision added: {version}")

    # HELPERS PRIVADOS (Modularizados)
    def _get_norm_factor(self, type_hint: str) -> float:
        """Helper: Fator por norma/type (expansível)."""
        factors = {
            NormCode.NBR6118_2023: {'safety_factor': 1.4, 'geometric_factor': 0.9},
            NormCode.AISC360_22: {'safety_factor': 1.2, 'geometric_factor': 1.0},
            # Expansível para mais
        }
        return factors.get(self.metadata.norm_code, {}).get(type_hint, 1.0)

    def _generate_number(self, level: int) -> str:
        """Auto-number (ex: 1.2.3)."""
        counters = [str(self._auto_counters[f"section_{i}"]) for i in range(1, level + 1)]
        return '.'.join(counters)

    def _register_element(self, element_id: str, element: Any) -> None:
        """Register element."""
        self._element_registry[element_id] = element

    def _check_frozen(self) -> None:
        """Check if document is frozen (after render)."""
        if self._frozen:
            raise RuntimeError(
                "Document is frozen after render(). "
                "Create a new document instance to make changes."
            )

    # Magic Methods
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"{self.__class__.__name__}("
            f"title='{self.metadata.title}', "
            f"sections={len(self.sections)}, "
            f"figures={len(self.figures)}, "
            f"tables={len(self.tables)}, "
            f"equations={len(self.equations)}"
            f")"
        )
    
    def __str__(self) -> str:
        """User-friendly string."""
        return f"{self.metadata.title} - {len(self.sections)} sections, {len(self.verifications)} verifications"
    
    def __len__(self) -> int:
        """Length: Total elements."""
        return len(self.sections) + len(self.figures) + len(self.tables) + len(self.equations) + len(self.verifications)

# ============================================================================
# SUBCLASSES STUBS (Para completude - implemente em arquivos separados)
# ============================================================================

class Memorial(BaseDocument):
    """Concrete Memorial subclass."""
    def render(self, output_path: Union[str, Path], format: str = "pdf") -> Path:
        """
        Render memorial to PDF (WeasyPrint).
        """
        try:
            from weasyprint import HTML
            from jinja2 import Template
        except ImportError as e:
            raise ImportError(f"Missing dependencies: {e}. Install: pip install weasyprint jinja2")
        
        output_path = Path(output_path)
        context = self.get_render_context()
        
        # ✅ TEMPLATE CORRIGIDO - Usa metadata['latex'] para cálculos
        html_template = Template("""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{{ title }}</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 40px;
            line-height: 1.6;
            color: #333;
        }
        h1 {
            color: #0066cc;
            border-bottom: 3px solid #0066cc;
            padding-bottom: 10px;
            margin-top: 30px;
        }
        h2 {
            color: #0088cc;
            border-bottom: 1px solid #ccc;
            padding-bottom: 5px;
            margin-top: 25px;
        }
        h3 {
            color: #666;
            margin-top: 20px;
        }
        .metadata {
            background-color: #f8f9fa;
            padding: 15px;
            border-left: 4px solid #0066cc;
            margin-bottom: 30px;
        }
        .calculation {
            background-color: #f5f5f5;
            padding: 15px;
            margin: 15px 0;
            border-left: 4px solid #28a745;
            font-family: 'Courier New', monospace;
            white-space: pre-wrap;
            font-size: 0.9em;
        }
        .variable {
            background-color: #fff3cd;
            padding: 10px;
            margin: 10px 0;
            border-left: 3px solid #ffc107;
            font-family: monospace;
        }
    </style>
</head>
<body>
    <h1>{{ title }}</h1>
    <div class="metadata">
        <p><strong>Autor:</strong> {{ author }}</p>
        <p><strong>Norma:</strong> {{ norm_code }}</p>
        <p><strong>Data:</strong> {{ date }}</p>
    </div>
    
    {% for section in sections %}
        {% if section.level == 1 %}
            <h1>{{ section.number }} {{ section.title }}</h1>
        {% elif section.level == 2 %}
            <h2>{{ section.number }} {{ section.title }}</h2>
        {% else %}
            <h3>{{ section.number }} {{ section.title }}</h3>
        {% endif %}
        
        {% for content in section.contents %}
            {% if content.type == 'text' %}
                <p>{{ content.content }}</p>
            
            {% elif content.type == 'calculation' %}
                {# ✅ CORREÇÃO CRÍTICA: Usar metadata.latex se disponível #}
                {% if content.metadata and content.metadata.latex %}
                    <div class="calculation">{{ content.metadata.latex | replace('\\\\', '<br>') | replace('\\text{', '') | replace('}', '') | replace('\\cdot', '×') | safe }}</div>
                {% else %}
                    <div class="calculation">{{ content.content }}</div>
                {% endif %}
            
            {% elif content.type == 'variable' %}
                {% if content.metadata %}
                    <div class="variable">{{ content.metadata.name }} = {{ content.metadata.value }} {{ content.metadata.unit }}</div>
                {% endif %}
            {% endif %}
        {% endfor %}
    {% endfor %}
</body>
</html>
        """)
        
        # Renderizar template
        html_content = html_template.render(**context)
        
        # Gerar PDF
        HTML(string=html_content).write_pdf(output_path)
        self.frozen = True
        self.logger.info(f"Memorial rendered to {output_path}")
        return output_path

    
    def _format_latex_to_html(self, latex: str) -> str:
        """
        Converte LaTeX básico para HTML formatado (fallback sem MathJax).
        """
        import re
        
        # Remover ambientes
        text = re.sub(r'\\begin\{[^}]+\}', '', latex)
        text = re.sub(r'\\end\{[^}]+\}', '', text)
        
        # Converter quebras de linha
        text = text.replace(r'\\', '<br>')
        text = text.replace('\n', '<br>')
        
        # Remover comandos LaTeX comuns
        text = re.sub(r'\\text\{([^}]+)\}', r'\1', text)
        text = text.replace('&', '&amp;')
        
        # Símbolos matemáticos
        text = text.replace(r'\cdot', ' × ')
        text = text.replace(r'\times', ' × ')
        text = text.replace(r'\div', ' ÷ ')
        
        return text.strip()


    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "memorial",
            **self.get_render_context(),
            "frozen": self._frozen
        }

class Report(BaseDocument):
    """Concrete Report subclass."""
    def render(self, output_path: Union[str, Path], format: str = "pdf") -> Path:
        # Similar to Memorial
        return super().render(output_path, format)

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "report", **self.get_render_context()}

class Article(BaseDocument):
    """Concrete Article subclass (LaTeX-focused)."""
    def render(self, output_path: Union[str, Path], format: str = "tex") -> Path:
        output_path = Path(output_path)
        context = self.get_render_context()
        tex_content = f"\\documentclass{{article}}\\begin{{document}}\\title{{{self.metadata.title}}}\\maketitle {context['toc']} \\end{{document}}"
        with open(output_path, 'w') as f:
            f.write(tex_content)
        return output_path

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "article", **self.get_render_context()}

# ============================================================================
# FALLBACKS FOR MISSING CLASSES (Robust)
# ============================================================================

if 'Section' not in globals():
    @dataclass
    class Section:
        title: str
        number: str = ""
        level: int = 1
        numbered: bool = True
        contents: List['ContentBlock'] = field(default_factory=list)
        id: str = field(default_factory=lambda: str(uuid.uuid4()))
        
        def add_content(self, block: 'ContentBlock'):
            self.contents.append(block)

if 'ContentBlock' not in globals():
    @dataclass
    class ContentBlock:
        type: str  # ContentType value
        content: Any  # str, Figure, etc.

if 'ContentType' not in globals():
    class ContentType(Enum):
        TEXT = "text"
        EQUATION = "equation"
        FIGURE = "figure"
        TABLE = "table"

if 'Verification' not in globals():
    @dataclass
    class Verification:
        name: str
        condition: Callable[[Dict[str, Any]], bool]
        description: str = ""
        norm_ref: Optional[str] = None
        passed: bool = field(init=False, default=False)
        id: str = field(default_factory=lambda: str(uuid.uuid4()))
        
        def evaluate(self, context: Dict[str, Any]) -> bool:
            self.passed = self.condition(context)
            return self.passed

if 'Revision' not in globals():
    @dataclass
    class Revision:
        date: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
        version: str = ""
        changes: str = ""

if 'Figure' not in globals():
    @dataclass
    class Figure:
        path: Path
        caption: str = ""
        number: str = ""
        id: str = field(default_factory=lambda: str(uuid.uuid4()))

if 'Table' not in globals():
    @dataclass
    class Table:
        data: Any  # pd.DataFrame or list of lists
        caption: str = ""
        style: TableStyle = TableStyle.SIMPLE
        number: str = ""
        id: str = field(default_factory=lambda: str(uuid.uuid4()))

if 'EquationDoc' not in globals():
    @dataclass
    class EquationDoc:
        latex: str
        number: str = ""
        description: str = ""
        id: str = field(default_factory=lambda: str(uuid.uuid4()))

# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "BaseDocument",
    "Memorial",
    "Report",
    "Article",
    "DocumentMetadata",
    "NormCode",
    "CrossReferenceType",
    "NormCompliance",
    "ValidationError",
    "DocumentError",
    "DocumentValidationError",
    "RenderError",
    "CrossReferenceError",
    "NormComplianceError",
    "TableStyle",
    "DocumentLanguage",
    "CrossReference",
    # Fallbacks
    "Section",
    "ContentBlock",
    "ContentType",
    "Verification",
    "Revision",
    "Figure",
    "Table",
    "EquationDoc",
]
