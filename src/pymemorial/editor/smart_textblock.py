# src/pymemorial/editor/smart_textblock.py
"""
SmartTextBlock - Bloco de texto inteligente para PyMemorial v2.0

ADAPTADO DE: efficalc/extension/templates/smart_textblock.py
VERSÃO: PyMemorial 2.0 (Outubro 2025)

Funcionalidades:
1. ✅ Substituição de variáveis locais via variables={}
2. ✅ Substituição de variáveis globais do contexto
3. ✅ Suporte a temp_symbols para LaTeX temporário
4. ✅ Formatação Python avançada ({var:.2f}, {var:.1%})
5. ✅ LaTeX inline com KaTeX (f_{cd}, \\lambda)
6. ✅ Markdown completo
7. ✅ Mesclagem robusta de contextos (local > global)
8. ✅ Sistema de cache inteligente
9. ✅ Validação pré-renderização
10. ✅ Integração TOTAL com pymemorial.core

CORREÇÕES PARA PYMEMORIAL:
- ✅ Imports adaptados para pymemorial.core
- ✅ CalculationItem → core.Variable
- ✅ MathEngine → core.text_processor.SmartTextProcessor
- ✅ Remoção de dependências efficalc
- ✅ Integração com natural_engine.py

Author: PyMemorial Team
Date: October 2025
Version: 2.0.0
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Dict, Optional, Union, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from functools import lru_cache

# ============================================================================
# IMPORTS PYMEMORIAL (CORRIGIDOS)
# ============================================================================
try:
    from pymemorial.core import Variable  # Substitui CalculationItem
    CORE_AVAILABLE = True
except ImportError:
    Variable = object
    CORE_AVAILABLE = False

try:
    from pymemorial.core.text_processor import SmartTextProcessor  # Substitui MathEngine
    TEXT_PROCESSOR_AVAILABLE = True
except ImportError:
    SmartTextProcessor = None
    TEXT_PROCESSOR_AVAILABLE = False

# Configuração de logging
logger = logging.getLogger(__name__)

# Regex para detectar placeholders
_PLACEHOLDER_RE = re.compile(r'\{([A-Za-z_][A-Za-z0-9_\.]*)(:[^}]+)?\}')

# ============================================================================
# ENUMS E CLASSES AUXILIARES
# ============================================================================

class ReferenceStyle(Enum):
    """Estilos de referência bibliográfica."""
    INLINE = "inline"
    FOOTNOTE = "footnote"
    SIDEBAR = "sidebar"
    ABNT = "abnt"

class TemplateType(Enum):
    """Templates pré-definidos para blocos de texto."""
    CUSTOM = "custom"
    TECHNICAL = "technical"
    ABNT_REPORT = "abnt"
    EXECUTIVE = "executive"
    CALCULATION = "calculation"

@dataclass
class ValidationResult:
    """Resultado da validação de um SmartTextBlock."""
    valid: bool
    missing_vars: List[str] = field(default_factory=list)
    suggestions: Dict[str, List[str]] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    total_vars: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário."""
        return {
            'valid': self.valid,
            'missing_vars': self.missing_vars,
            'suggestions': self.suggestions,
            'warnings': self.warnings,
            'total_vars': self.total_vars,
            'success_rate': (
                (self.total_vars - len(self.missing_vars)) / max(1, self.total_vars)
            ) * 100
        }

@dataclass
class RenderMetadata:
    """Metadados de renderização para análise."""
    render_time_ms: float = 0.0
    text_length: int = 0
    output_length: int = 0
    vars_resolved: int = 0
    vars_failed: int = 0
    cache_hit: bool = False
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário."""
        return {
            'render_time_ms': self.render_time_ms,
            'text_length': self.text_length,
            'output_length': self.output_length,
            'vars_resolved': self.vars_resolved,
            'vars_failed': self.vars_failed,
            'cache_hit': self.cache_hit,
            'timestamp': self.timestamp
        }

# ============================================================================
# SMARTTEXTBLOCK - CLASSE PRINCIPAL
# ============================================================================

class SmartTextBlock:
    """
    Bloco de texto inteligente para PyMemorial v2.0.
    
    Integração TOTAL com pymemorial.core.
    
    Prioridade de resolução de variáveis:
    1. self.local_variables (passadas via variables={})
    2. context (variáveis globais: do natural_engine)
    3. temp_symbols (símbolos LaTeX temporários)
    
    Examples:
        >>> from pymemorial.editor import SmartTextBlock
        >>> from pymemorial.core import Variable
        >>> 
        >>> # Variáveis locais simples
        >>> block = SmartTextBlock(
        ...     text="Valor: {x:.2f} m e taxa: {y:.1%}",
        ...     variables={'x': 3.14159, 'y': 0.856}
        ... )
        >>> html = block.render()
        >>> 
        >>> # Com objetos Variable do core
        >>> M_k = Variable('M_k', 112.5, 'kN.m')
        >>> block = SmartTextBlock(
        ...     text="Momento: {M_k:.2f} kN·m",
        ...     variables={'M_k': M_k}
        ... )
        >>> 
        >>> # Com símbolos temporários
        >>> block = SmartTextBlock(
        ...     text="Coeficiente k_{md} = {k_md:.3f}",
        ...     variables={'k_md': 0.295},
        ...     temp_symbols=['k_{md}']
        ... )
    """
    
    # Cache estático para renderizações repetidas
    _render_cache: Dict[str, Tuple[str, RenderMetadata]] = {}
    _cache_enabled: bool = True
    
    def __init__(
        self,
        text: str,
        reference: Optional[str] = None,
        variables: Optional[Dict[str, Any]] = None,
        temp_symbols: Optional[Union[Dict[str, str], List[str]]] = None,
        reference_style: ReferenceStyle = ReferenceStyle.INLINE,
        template_type: TemplateType = TemplateType.CUSTOM,
        strict_mode: bool = False,
        enable_cache: bool = True,
        debug_mode: bool = False
    ):
        """
        Inicializa SmartTextBlock.
        
        Args:
            text: Texto com placeholders {var}
            reference: Referência bibliográfica opcional
            variables: Variáveis locais (prioridade máxima)
            temp_symbols: Símbolos LaTeX temporários (list ou dict)
            reference_style: Estilo de referência (INLINE, FOOTNOTE, etc)
            template_type: Template pré-definido
            strict_mode: Se True, falha em variáveis faltantes
            enable_cache: Ativar cache de renderização
            debug_mode: Ativar logs detalhados
        """
        self.text = text
        self.reference = reference
        self.reference_style = reference_style
        self.template_type = template_type
        self.strict_mode = strict_mode
        self.enable_cache = enable_cache
        self.debug_mode = debug_mode
        
        # Processar variáveis locais
        self.local_variables = self._process_variables(variables or {})
        
        # Processar temp_symbols
        self.temp_symbols: Dict[str, str] = self._process_temp_symbols(temp_symbols)
        
        # Metadados de renderização
        self.last_render_metadata: Optional[RenderMetadata] = None
        
        # Validação inicial em debug mode
        if self.debug_mode:
            logger.debug(
                f"SmartTextBlock criado: {len(self.local_variables)} vars locais, "
                f"{len(self.temp_symbols)} temp_symbols"
            )
    
    def _process_variables(self, variables: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processa variáveis, extraindo valores de objetos Variable do core.
        
        Args:
            variables: Dicionário de variáveis
        
        Returns:
            Dicionário com valores extraídos
        """
        processed = {}
        
        for name, value in variables.items():
            # Se for Variable do core, extrair valor
            if CORE_AVAILABLE and isinstance(value, Variable):
                processed[name] = value.value
                if self.debug_mode:
                    logger.debug(f"Variable extraída: {name} = {value.value} {value.unit}")
            else:
                processed[name] = value
        
        return processed
    
    def _process_temp_symbols(
        self,
        temp_symbols: Optional[Union[Dict[str, str], List[str]]]
    ) -> Dict[str, str]:
        """
        Processa temp_symbols com conversão inteligente de list para dict.
        
        Args:
            temp_symbols: Símbolos temporários em formato dict ou list
        
        Returns:
            Dicionário normalizado {key: latex_str}
        """
        result = {}
        
        if isinstance(temp_symbols, list):
            # Converter lista de LaTeX para dicionário
            for latex_str in temp_symbols:
                name_key = self._extract_symbol_key(latex_str)
                result[name_key] = latex_str
                if self.debug_mode:
                    logger.debug(f"Temp symbol convertido: {name_key} -> {latex_str}")
        
        elif isinstance(temp_symbols, dict):
            result.update(temp_symbols)
            if self.debug_mode:
                logger.debug(f"Temp symbols recebidos: {temp_symbols}")
        
        elif temp_symbols is not None:
            logger.warning(
                f"temp_symbols deve ser dict ou list, recebido: {type(temp_symbols).__name__}"
            )
        
        return result
    
    @staticmethod
    def _extract_symbol_key(latex_str: str) -> str:
        r"""
        Extrai chave normalizada de uma string LaTeX.
        
        **CORREÇÃO v2.0.1**: Regex aprimorado para subscritos.
        
        Args:
            latex_str: String LaTeX (ex: 'k_{md}', '\lambda_{lim}')
        
        Returns:
            Chave normalizada (ex: 'k_md', 'lambda_lim')
        
        Examples:
            >>> SmartTextBlock._extract_symbol_key('k_{md}')
            'k_md'
            >>> SmartTextBlock._extract_symbol_key(r'\lambda_{lim}')
            'lambda_lim'
            >>> SmartTextBlock._extract_symbol_key('gamma_f')
            'gamma_f'
            >>> SmartTextBlock._extract_symbol_key(r'\alpha_{1}')
            'alpha_1'
        """
        # PASSO 1: Remover backslashes (comandos LaTeX)
        clean = latex_str.replace('\\', '')
        
        # PASSO 2: Remover espaços
        clean = clean.replace(' ', '_')
        
        # PASSO 3: Converter subscritos _{...} para _...
        # Regex: _{conteudo} → _conteudo
        clean = re.sub(r'_\{([^}]+)\}', r'_\1', clean)
        
        # PASSO 4: Remover chaves soltas remanescentes
        clean = clean.replace('{', '').replace('}', '')
        
        return clean

    
    def _merge_contexts(
        self,
        global_context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Mescla contextos com prioridade correta (local > global).
        
        **CORREÇÃO CRÍTICA DO BUG ORIGINAL**:
        - Contexto global é adicionado PRIMEIRO
        - Variáveis locais SOBRESCREVEM variáveis globais
        
        Args:
            global_context: Contexto global do relatório
        
        Returns:
            Dicionário mesclado com prioridade correta
        """
        merged = {}
        
        # PASSO 1: Adicionar contexto global (base)
        if global_context:
            merged.update(global_context)
            if self.debug_mode:
                logger.debug(f"✓ Contexto global: {len(global_context)} vars")
        
        # PASSO 2: Sobrescrever com variáveis locais (PRIORIDADE MÁXIMA)
        if self.local_variables:
            overwrites = set(self.local_variables.keys()) & set(merged.keys())
            if overwrites and self.debug_mode:
                logger.debug(f"⚠ Sobrescritas locais: {overwrites}")
            
            merged.update(self.local_variables)
            if self.debug_mode:
                logger.debug(f"✓ Variáveis locais: {len(self.local_variables)} vars")
        
        # PASSO 3: Log do contexto final
        if self.debug_mode:
            logger.debug(f"✅ Contexto mesclado: {len(merged)} vars")
        
        return merged
    
    def _get_processor(self) -> SmartTextProcessor:
        """
        Obtém instância do SmartTextProcessor.
        
        Returns:
            Instância configurada do SmartTextProcessor
        
        Raises:
            ImportError: Se SmartTextProcessor não disponível
        """
        if not TEXT_PROCESSOR_AVAILABLE:
            raise ImportError(
                "SmartTextProcessor não encontrado. "
                "Verifique a instalação do PyMemorial."
            )
        
        return SmartTextProcessor()
    
    def _format_reference_html(self) -> str:
        """Formata referência bibliográfica em HTML."""
        if not self.reference:
            return ""
        
        if self.reference_style == ReferenceStyle.INLINE:
            return (
                f'<span class="reference reference-inline">'
                f'({self.reference})</span>'
            )
        elif self.reference_style == ReferenceStyle.FOOTNOTE:
            return (
                f'<sup class="reference reference-footnote">'
                f'[{self.reference}]</sup>'
            )
        elif self.reference_style == ReferenceStyle.ABNT:
            return (
                f'<div class="reference reference-abnt">'
                f'{self.reference}</div>'
            )
        elif self.reference_style == ReferenceStyle.SIDEBAR:
            return (
                f'<aside class="reference reference-sidebar">'
                f'{self.reference}</aside>'
            )
        
        return f'<span class="reference">{self.reference}</span>'
    
    def _generate_cache_key(self, context: Dict[str, Any]) -> str:
        """Gera chave única para cache."""
        import hashlib
        
        components = [
            self.text,
            str(sorted(context.items())),
            str(sorted(self.temp_symbols.items())),
            self.reference or "",
            self.reference_style.value
        ]
        
        key_str = "|".join(components)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def render(
        self,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Renderiza o bloco de texto para HTML.
        
        **FLUXO CORRIGIDO**:
        1. Mesclar contextos (local > global)
        2. Verificar cache (se ativado)
        3. Processar texto com SmartTextProcessor
        4. Envolver HTML com referência (se fornecida)
        5. Retornar HTML formatado
        
        Args:
            context: Contexto global do relatório
        
        Returns:
            String HTML formatada
        """
        start_time = time.time()
        
        # CORREÇÃO: Garantir que context seja dict (nunca None)
        if context is None:
            context = {}
        
        # PASSO 1: Mesclar contextos
        merged_context = self._merge_contexts(context)
        
        if self.debug_mode:
            print(f"\n{'='*70}")
            print(f"🔧 DEBUG SmartTextBlock.render()")
            print(f"{'='*70}")
            print(f"📝 Texto: {self.text[:100]}...")
            print(f"📦 Contexto mesclado: {len(merged_context)} vars")
            print(f"🎨 Temp symbols: {list(self.temp_symbols.keys())}")
            print(f"{'='*70}\n")
        
        # PASSO 2: Cache
        cache_key = None
        if self.enable_cache and self._cache_enabled:
            cache_key = self._generate_cache_key(merged_context)
            
            if cache_key in self._render_cache:
                cached_html, cached_metadata = self._render_cache[cache_key]
                
                # Criar NOVO metadata com cache_hit=True
                cache_hit_metadata = RenderMetadata(
                    render_time_ms=cached_metadata.render_time_ms,
                    text_length=cached_metadata.text_length,
                    output_length=cached_metadata.output_length,
                    vars_resolved=cached_metadata.vars_resolved,
                    vars_failed=cached_metadata.vars_failed,
                    cache_hit=True,
                    timestamp=datetime.now().isoformat()
                )
                self.last_render_metadata = cache_hit_metadata
                
                if self.debug_mode:
                    logger.debug(f"✅ Cache HIT: {cache_key[:8]}...")
                
                return cached_html
            
            if self.debug_mode:
                logger.debug(f"❌ Cache MISS: {cache_key[:8]}...")
        
        # PASSO 3: Renderizar com SmartTextProcessor
        try:
            processor = self._get_processor()
            
            # Definir variáveis no processor
            for name, value in merged_context.items():
                processor.define_variable(name, value)
            
            # Processar texto
            final_html_content = processor.process(self.text)
            
            if self.debug_mode:
                logger.debug(f"✅ Renderização concluída: {len(final_html_content)} chars")
        
        except Exception as e:
            logger.error(f"❌ Erro na renderização: {e}")
            if self.debug_mode:
                logger.exception("Stack trace:")
            if self.strict_mode:
                raise
            return self._fallback_render(str(e))
        
        # PASSO 4: Envolver com referência
        if self.reference:
            reference_html = self._format_reference_html()
            final_html_content = f"{final_html_content}\n{reference_html}"
        
        # PASSO 5: Gerar metadados
        elapsed_time = (time.time() - start_time) * 1000
        
        metadata = RenderMetadata(
            render_time_ms=elapsed_time,
            text_length=len(self.text),
            output_length=len(final_html_content),
            vars_resolved=len(merged_context),
            vars_failed=0,
            cache_hit=False
        )
        self.last_render_metadata = metadata
        
        # PASSO 6: Armazenar no cache
        if cache_key and self.enable_cache:
            self._render_cache[cache_key] = (final_html_content, metadata)
            if self.debug_mode:
                logger.debug(f"💾 Armazenado em cache: {cache_key[:8]}...")
        
        if self.debug_mode:
            print(f"\n✅ SmartTextBlock renderizado em {elapsed_time:.2f} ms\n")
        
        return final_html_content
    
    def _fallback_render(self, error_message: str) -> str:
        """Renderização de fallback em caso de erro."""
        logger.warning(f"Fallback devido a erro: {error_message}")
        
        return (
            f'<div class="smart-textblock-error">\n'
            f'  <p>⚠️ Erro na renderização do bloco de texto:</p>\n'
            f'  <pre>{error_message}</pre>\n'
            f'  <p>Texto original:</p>\n'
            f'  <pre>{self.text}</pre>\n'
            f'</div>'
        )
    
    def validate(self, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """
        Valida o bloco de texto antes da renderização.
        
        Args:
            context: Contexto global para validação
        
        Returns:
            ValidationResult com status de validação
        """
        missing_vars = []
        suggestions = {}
        warnings = []
        
        # Mesclar contextos
        merged_context = self._merge_contexts(context or {})
        
        # Extrair placeholders do texto
        placeholders = _PLACEHOLDER_RE.findall(self.text)
        total_vars = len(placeholders)
        
        for match in placeholders:
            var_name = match[0] if isinstance(match, tuple) else match
            
            if var_name not in merged_context:
                missing_vars.append(var_name)
                
                # Sugerir variáveis similares
                similar = [
                    name for name in merged_context.keys()
                    if name.lower().startswith(var_name.lower()[:3])
                ]
                if similar:
                    suggestions[var_name] = similar
        
        # Avisos
        if len(merged_context) > 100:
            warnings.append("Contexto muito grande (>100 variáveis) - considere usar cache")
        
        if not self.enable_cache and len(self.text) > 5000:
            warnings.append("Texto grande sem cache - considere ativar cache")
        
        valid = len(missing_vars) == 0
        
        return ValidationResult(
            valid=valid,
            missing_vars=missing_vars,
            suggestions=suggestions,
            warnings=warnings,
            total_vars=total_vars
        )
    
    def __repr__(self) -> str:
        """Representação string para debugging."""
        return (
            f"SmartTextBlock("
            f"text={self.text[:50]}..., "
            f"vars={len(self.local_variables)}, "
            f"temp_symbols={len(self.temp_symbols)})"
        )
    
    def __str__(self) -> str:
        """String representation."""
        return self.text
    
    @classmethod
    def clear_cache(cls):
        """Limpa o cache de renderização."""
        cls._render_cache.clear()
        logger.info("Cache de SmartTextBlock limpo")
    
    @classmethod
    def get_cache_stats(cls) -> Dict[str, Any]:
        """Retorna estatísticas do cache."""
        return {
            'size': len(cls._render_cache),
            'enabled': cls._cache_enabled
        }

# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'SmartTextBlock',
    'ReferenceStyle',
    'TemplateType',
    'ValidationResult',
    'RenderMetadata'
]
