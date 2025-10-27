# src/pymemorial/document/_internal/text_processing/integration.py
"""
Integration Bridge: Recognition ↔ Document

Conecta o módulo `recognition/` (SmartTextProcessor v3.0) com o módulo
`document/` (BaseDocument, Memorial, Article, etc.).

Permite usar escrita natural em documentos:
- Processa {var}, {{var}}, @eq automaticamente
- Integra com text_utils.py existente
- Middleware entre user text → processed text → rendered HTML/PDF

Architecture:
    User Text (natural language)
         ↓
    SmartTextProcessor (@eq, {var}, {{var}})
         ↓
    TextProcessingBridge (this file)
         ↓
    text_utils.py (clean, escape, format)
         ↓
    BaseDocument (add_text, render)

Example:
    >>> from pymemorial.document._internal.text_processing import TextProcessingBridge
    >>> bridge = TextProcessingBridge()
    >>> bridge.define_variables({'M_k': (112.5, 'kN.m', 'Momento')})
    >>> processed = bridge.process("O momento {M_k} resulta em @eq M_d = 1.4 * M_k")

Author: PyMemorial Team
Date: October 2025
Version: 1.0.0
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple, Optional, Any
from pathlib import Path

# Import recognition module (v3.0)
try:
    from pymemorial.recognition import (
        SmartTextProcessor,
        VariableRegistry,
        ProcessingOptions,
        DocumentType,
        RenderMode,
        CitationStyle,
        TEXT_PROCESSOR_V3_AVAILABLE,
    )
    RECOGNITION_AVAILABLE = TEXT_PROCESSOR_V3_AVAILABLE
except ImportError:
    RECOGNITION_AVAILABLE = False
    SmartTextProcessor = None
    VariableRegistry = None
    _logger = logging.getLogger(__name__)
    _logger.warning("Recognition module v3.0 not available. Bridge limited to text_utils only.")

# Import local text_utils (existente)
try:
    from .text_utils import (
        clean_text,
        escape_latex,
        format_number,
        wrap_text,
        sanitize_filename,
    )
    TEXT_UTILS_AVAILABLE = True
except ImportError:
    TEXT_UTILS_AVAILABLE = False
    _logger = logging.getLogger(__name__)
    _logger.warning("text_utils.py not found. Using fallbacks.")
    # Fallbacks simples
    clean_text = lambda t: t.strip()
    escape_latex = lambda t: t
    format_number = lambda n, p=2: f"{n:.{p}f}"
    wrap_text = lambda t, w=80: t
    sanitize_filename = lambda f: f.replace(' ', '_')


# ============================================================================
# TEXT PROCESSING BRIDGE (MAIN CLASS)
# ============================================================================

class TextProcessingBridge:
    """
    Bridge entre Recognition e Document.
    
    Processa texto natural com variáveis, equações, e formatação,
    integrando SmartTextProcessor v3.0 com text_utils.py.
    
    Features:
    - Processa @eq, {var}, {{var}}
    - Limpa e escapa LaTeX
    - Formata números
    - Wrapper de linha
    - Contexto de variáveis persistente
    
    Example:
        >>> bridge = TextProcessingBridge(document_type='memorial')
        >>> bridge.define_variables({
        ...     'M_k': (112.5, 'kN.m', 'Momento característico'),
        ...     'gamma_f': (1.4, '', 'Coeficiente de majoração'),
        ... })
        >>> text = "O momento {M_k} majorado resulta em @eq M_d = gamma_f * M_k"
        >>> processed = bridge.process(text)
    """
    
    def __init__(
        self,
        document_type: str = 'memorial',
        render_mode: str = 'full',
        citation_style: str = 'abnt',
        enable_latex_escape: bool = True,
        enable_text_cleaning: bool = True,
    ):
        """
        Inicializa bridge.
        
        Args:
            document_type: 'memorial', 'article', 'tcc', 'report'
            render_mode: 'full', 'symbolic', 'numeric', 'result'
            citation_style: 'abnt', 'ieee', 'apa'
            enable_latex_escape: Se True, escapa caracteres LaTeX
            enable_text_cleaning: Se True, limpa texto
        """
        self.logger = logging.getLogger(__name__)
        self.enable_latex_escape = enable_latex_escape
        self.enable_text_cleaning = enable_text_cleaning
        
        # Inicializar SmartTextProcessor v3.0 (se disponível)
        if RECOGNITION_AVAILABLE and SmartTextProcessor:
            options = self._create_processing_options(
                document_type, render_mode, citation_style
            )
            self.processor = SmartTextProcessor(options=options)
            self.registry = self.processor.get_registry()
            self.logger.info(f"TextProcessingBridge initialized with v3.0 (document_type={document_type})")
        else:
            self.processor = None
            self.registry = None
            self.logger.warning("SmartTextProcessor v3.0 not available. Using text_utils only.")
    
    # ========================================================================
    # PUBLIC API
    # ========================================================================
    
    def define_variables(self, variables: Dict[str, Tuple[float, str, str]]):
        """
        Define variáveis no contexto.
        
        Args:
            variables: Dict {name: (value, unit, description)}
            
        Example:
            >>> bridge.define_variables({
            ...     'M_k': (112.5, 'kN.m', 'Momento característico'),
            ...     'gamma_f': (1.4, '', 'Coeficiente de majoração'),
            ...     'M_d': (157.5, 'kN.m', 'Momento de cálculo'),
            ... })
        """
        if self.processor:
            self.processor.define_variables(variables)
            self.logger.debug(f"Defined {len(variables)} variables")
        else:
            self.logger.warning("Processor not available. Variables not registered.")
    
    def process(
        self,
        text: str,
        clean: bool = True,
        escape: bool = True,
        wrap_width: Optional[int] = None,
    ) -> str:
        """
        Processa texto completo (pipeline completo).
        
        Pipeline:
        1. Clean text (se enabled)
        2. Process @eq, {var}, {{var}} (v3.0)
        3. Escape LaTeX (se enabled)
        4. Wrap text (se wrap_width fornecido)
        
        Args:
            text: Texto em linguagem natural
            clean: Se True, limpa texto
            escape: Se True, escapa caracteres LaTeX especiais
            wrap_width: Largura para quebra de linha (None = sem quebra)
        
        Returns:
            Texto processado
            
        Example:
            >>> text = '''
            ... O momento característico M_k = {M_k} é majorado por gamma_f = {gamma_f}.
            ... 
            ... O momento de cálculo é obtido por:
            ... @eq M_d = gamma_f * M_k
            ... 
            ... Resultando em M_d = {M_d}.
            ... '''
            >>> processed = bridge.process(text)
        """
        result = text
        
        # 1. Clean
        if clean and self.enable_text_cleaning and TEXT_UTILS_AVAILABLE:
            result = clean_text(result)
        
        # 2. Process v3.0 (@eq, {var}, {{var}})
        if self.processor:
            result = self.processor.process(result)
        
        # 3. Escape LaTeX (cuidado: não escapar dentro de $ $)
        if escape and self.enable_latex_escape and TEXT_UTILS_AVAILABLE:
            result = self._smart_escape_latex(result)
        
        # 4. Wrap
        if wrap_width and TEXT_UTILS_AVAILABLE:
            result = wrap_text(result, width=wrap_width)
        
        return result
    
    def process_inline(self, text: str) -> str:
        """
        Processa texto inline (sem quebras de linha, sem escape).
        
        Útil para títulos, legendas, labels.
        
        Args:
            text: Texto inline
        
        Returns:
            Texto processado
        """
        if self.processor:
            return self.processor.process(text)
        return text
    
    def get_variable_value(self, name: str) -> Optional[float]:
        """
        Obtém valor de variável registrada.
        
        Args:
            name: Nome da variável
        
        Returns:
            Valor ou None se não encontrada
        """
        if self.registry:
            ctx = self.registry.get(name)
            return ctx.value if ctx else None
        return None
    
    def list_variables(self) -> Dict[str, Dict[str, Any]]:
        """
        Lista todas as variáveis registradas.
        
        Returns:
            Dict {name: {'value': float, 'unit': str, 'description': str}}
        """
        if not self.registry:
            return {}
        
        result = {}
        for name in self.registry.list_variables():
            ctx = self.registry.get(name)
            if ctx:
                result[name] = {
                    'value': ctx.value,
                    'unit': ctx.unit,
                    'description': ctx.description,
                }
        return result
    
    def format_value(
        self,
        value: float,
        precision: int = 2,
        unit: str = '',
    ) -> str:
        """
        Formata valor numérico com unidade.
        
        Args:
            value: Valor numérico
            precision: Casas decimais
            unit: Unidade
        
        Returns:
            String formatada
            
        Example:
            >>> bridge.format_value(112.5, precision=1, unit='kN.m')
            '112.5 kN.m'
        """
        if TEXT_UTILS_AVAILABLE:
            formatted = format_number(value, precision=precision)
        else:
            formatted = f"{value:.{precision}f}"
        
        return f"{formatted} {unit}".strip() if unit else formatted
    
    # ========================================================================
    # PRIVATE HELPERS
    # ========================================================================
    
    def _create_processing_options(
        self,
        document_type: str,
        render_mode: str,
        citation_style: str,
    ) -> 'ProcessingOptions':
        """Cria ProcessingOptions do enum string."""
        try:
            return ProcessingOptions(
                document_type=DocumentType[document_type.upper()],
                render_mode=RenderMode[render_mode.upper()],
                citation_style=CitationStyle[citation_style.upper()],
            )
        except (KeyError, AttributeError) as e:
            self.logger.warning(f"Invalid option: {e}. Using defaults.")
            return ProcessingOptions()
    
    def _smart_escape_latex(self, text: str) -> str:
        """
        Escapa LaTeX inteligentemente (preserva math mode $ $).
        
        Não escapa dentro de:
        - $ ... $ (inline math)
        - $$ ... $$ (display math)
        """
        if not TEXT_UTILS_AVAILABLE:
            return text
        
        # Split por math mode
        import re
        # Pattern: $ ... $ ou $$ ... $$
        math_pattern = re.compile(r'(\$\$?.*?\$\$?)', re.DOTALL)
        
        parts = math_pattern.split(text)
        
        # Escapa apenas partes não-math
        result_parts = []
        for i, part in enumerate(parts):
            if part.startswith('$'):
                # É math mode, não escapar
                result_parts.append(part)
            else:
                # É texto normal, escapar
                result_parts.append(escape_latex(part))
        
        return ''.join(result_parts)


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_text_bridge(
    document_type: str = 'memorial',
    render_mode: str = 'full',
    **kwargs
) -> TextProcessingBridge:
    """
    Factory para criar TextProcessingBridge.
    
    Args:
        document_type: Tipo de documento
        render_mode: Modo de renderização
        **kwargs: Argumentos adicionais para TextProcessingBridge
    
    Returns:
        TextProcessingBridge configurado
        
    Example:
        >>> bridge = create_text_bridge(document_type='tcc', render_mode='symbolic')
    """
    return TextProcessingBridge(
        document_type=document_type,
        render_mode=render_mode,
        **kwargs
    )


# ============================================================================
# INTEGRATION HELPERS (para BaseDocument)
# ============================================================================

class DocumentTextProcessor:
    """
    Helper class para integração com BaseDocument.
    
    Mantém contexto persistente entre chamadas add_text().
    """
    
    def __init__(self, bridge: TextProcessingBridge):
        self.bridge = bridge
        self.logger = logging.getLogger(__name__)
    
    def process_text_block(self, text: str) -> str:
        """Processa bloco de texto (para add_text)."""
        return self.bridge.process(text, clean=True, escape=False)
    
    def process_inline_text(self, text: str) -> str:
        """Processa texto inline (para títulos, legendas)."""
        return self.bridge.process_inline(text)
    
    def define_calculation_context(self, variables: Dict[str, Tuple[float, str, str]]):
        """Define contexto de variáveis para cálculos."""
        self.bridge.define_variables(variables)


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'TextProcessingBridge',
    'DocumentTextProcessor',
    'create_text_bridge',
    'RECOGNITION_AVAILABLE',
    'TEXT_UTILS_AVAILABLE',
]
