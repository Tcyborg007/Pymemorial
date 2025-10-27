# src/pymemorial/document/internal/text_processing/text_utils.py
"""
Text Utilities - Robust template processing for PyMemorial documents.

Provides:
- Placeholder substitution {var:.2f}
- Symbol recognition (M_k, gamma_s, etc.)
- Safe fallback for missing variables
- Greek letter conversion (optional)
- Text cleaning and escaping

Author: PyMemorial Team
Date: 2025-10-21
Version: 1.0.0 (Production Ready)
"""

from __future__ import annotations
import re
import logging
from typing import Dict, Any, Optional
from string import Formatter

# ============================================================================
# LOGGER
# ============================================================================

logger = logging.getLogger(__name__)

# ============================================================================
# TEXT CLEANING FUNCTIONS (FALTAVAM!)
# ============================================================================

def clean_text(text: str) -> str:
    """
    Limpa texto removendo espaços extras e normalizando quebras de linha.
    
    Args:
        text: Texto a limpar
    
    Returns:
        Texto limpo
    """
    # Remove espaços extras
    text = re.sub(r'[ \t]+', ' ', text)
    # Normaliza quebras de linha (max 2 consecutivas)
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Remove espaços no início/fim de linhas
    text = '\n'.join(line.strip() for line in text.split('\n'))
    return text.strip()


def escape_latex(text: str) -> str:
    """
    Escapa caracteres especiais LaTeX.
    
    Args:
        text: Texto a escapar
    
    Returns:
        Texto escapado
    """
    special_chars = {
        '\\': r'\textbackslash{}',
        '&': r'\&',
        '%': r'\%',
        '$': r'\$',
        '#': r'\#',
        '_': r'\_',
        '{': r'\{',
        '}': r'\}',
        '~': r'\textasciitilde{}',
        '^': r'\^{}',
    }
    
    result = text
    for char, escaped in special_chars.items():
        result = result.replace(char, escaped)
    
    return result


def format_number(value: float, precision: int = 2) -> str:
    """
    Formata número com precisão específica.
    
    Args:
        value: Valor numérico
        precision: Casas decimais
    
    Returns:
        String formatada
    """
    return f"{value:.{precision}f}"


def wrap_text(text: str, width: int = 80) -> str:
    """
    Quebra texto em linhas com largura máxima.
    
    Args:
        text: Texto a quebrar
        width: Largura máxima
    
    Returns:
        Texto quebrado
    """
    import textwrap
    return '\n'.join(textwrap.wrap(text, width=width))


def sanitize_filename(filename: str) -> str:
    """
    Sanitiza nome de arquivo removendo caracteres inválidos.
    
    Args:
        filename: Nome original
    
    Returns:
        Nome sanitizado
    """
    # Remove caracteres inválidos
    filename = re.sub(r'[<>:"/\\|?*]', '', filename)
    # Substitui espaços por underscore
    filename = filename.replace(' ', '_')
    # Remove múltiplos underscores
    filename = re.sub(r'_+', '_', filename)
    return filename.strip('_')


# ============================================================================
# TEMPLATE PROCESSOR (EXISTENTE - MANTIDO)
# ============================================================================

class TemplateProcessor:
    """
    Robust template processor for PyMemorial text with placeholders.
    
    Features:
    - Uses Python's string.Formatter (built-in, safe)
    - Supports format specs: {var:.2f}, {value:>10}, etc.
    - Graceful fallback for missing variables
    - Symbol detection (M_k, N_Rd, gamma_s)
    
    Examples:
    --------
    >>> processor = TemplateProcessor()
    >>> context = {'M_k': 150.5, 'gamma_s': 1.4}
    >>> text = "Momento M_k = {M_k:.2f} kN.m com fator {gamma_s:.1f}."
    >>> result = processor.process(text, context)
    >>> print(result)
    'Momento M_k = 150.50 kN.m com fator 1.4.'
    """
    
    def __init__(self, strict: bool = False):
        """
        Initialize processor.
        
        Args:
            strict: If True, raise KeyError for missing variables.
                   If False, leave placeholder unchanged (safe).
        """
        self.strict = strict
        self.formatter = Formatter()
        # Pattern to detect placeholders {var} or {var:.2f}
        self.placeholder_pattern = re.compile(r'\{([^}:]+)(?::([^}]+))?\}')
    
    def process(self, text: str, context: Dict[str, Any]) -> str:
        """
        Process text with placeholders using context.
        
        Args:
            text: Text with placeholders like "Value = {var:.2f}"
            context: Dictionary with variable values
        
        Returns:
            Processed text with substitutions
        
        Raises:
            KeyError: If strict=True and variable not in context
        """
        if not text or '{' not in text:
            return text
        
        try:
            # Use string.Formatter for robust parsing
            result = self.formatter.vformat(text, (), context)
            logger.debug(f"Template processed: {len(result)} chars")
            return result
        except (KeyError, ValueError) as e:
            if self.strict:
                raise KeyError(f"Missing variable in template: {e}") from e
            
            # Fallback: Partial substitution (graceful)
            logger.warning(f"Template processing failed: {e}. Using partial substitution.")
            return self._partial_substitute(text, context)
    
    def _partial_substitute(self, text: str, context: Dict[str, Any]) -> str:
        """
        Partial substitution: Replace available vars, leave others unchanged.
        
        Args:
            text: Template text
            context: Available variables
        
        Returns:
            Partially processed text
        """
        def replace_match(match):
            var_name = match.group(1)
            format_spec = match.group(2) or ''
            
            if var_name in context:
                try:
                    value = context[var_name]
                    if format_spec:
                        return f"{value:{format_spec}}"
                    else:
                        return str(value)
                except Exception as e:
                    logger.warning(f"Failed to format {var_name}: {e}")
                    return match.group(0)  # Return original
            else:
                # Variable not in context, leave placeholder
                return match.group(0)
        
        result = self.placeholder_pattern.sub(replace_match, text)
        return result
    
    def extract_variables(self, text: str) -> list:
        """
        Extract all variable names from template.
        
        Args:
            text: Template text
        
        Returns:
            List of variable names (without format specs)
        """
        return [match.group(1) for match in self.placeholder_pattern.finditer(text)]
    
    def validate_template(self, text: str, context: Dict[str, Any]) -> tuple:
        """
        Validate template against context.
        
        Args:
            text: Template text
            context: Available variables
        
        Returns:
            Tuple: (is_valid, missing_vars)
        """
        required_vars = self.extract_variables(text)
        missing_vars = [var for var in required_vars if var not in context]
        is_valid = len(missing_vars) == 0
        return is_valid, missing_vars


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def process_template(text: str, context: Dict[str, Any], strict: bool = False) -> str:
    """
    Convenience function to process template with context.
    
    Args:
        text: Template text
        context: Variable values
        strict: Raise error if variable missing
    
    Returns:
        Processed text
    """
    processor = TemplateProcessor(strict=strict)
    return processor.process(text, context)


def extract_variables(text: str) -> list:
    """
    Convenience function to extract variables from template.
    
    Args:
        text: Template text
    
    Returns:
        List of variable names
    """
    processor = TemplateProcessor()
    return processor.extract_variables(text)


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Text cleaning
    'clean_text',
    'escape_latex',
    'format_number',
    'wrap_text',
    'sanitize_filename',
    # Template processing
    'TemplateProcessor',
    'process_template',
    'extract_variables',
]
