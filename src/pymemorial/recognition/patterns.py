# src/pymemorial/recognition/patterns.py
"""
Regex Patterns for Text Recognition v3.0

Expanded patterns for SmartTextEngine with support for:
- Variable inline recognition: M_d, gamma_f
- Value display: {M_d} → "157.5 kN.m"
- Formula display: {{M_d}} → formula expandida
- Equation blocks: @eq M_d = gamma_f * M_k
- Headings: # ## ###
- ABNT citations: [1], (SILVA, 2023)
"""
import re
from typing import Dict, Pattern, List

# ============================================================================
# LEGACY PATTERNS (Manter compatibilidade)
# ============================================================================

# Pattern para variáveis gregas
GREEK_PATTERN = re.compile(
    r'\b('
    r'alpha|beta|gamma|delta|epsilon|zeta|eta|theta|'
    r'iota|kappa|lambda|mu|nu|xi|omicron|pi|rho|'
    r'sigma|tau|upsilon|phi|chi|psi|omega'
    r')(?:_([a-zA-Z0-9_]+))?\b'
)

# Pattern para variáveis com subscript: f_ck, M_d
SUBSCRIPT_PATTERN = re.compile(
    r'([a-zA-Z]+)_([a-zA-Z0-9]+)'
)

# Pattern para expoentes: x^2, E^5
SUPERSCRIPT_PATTERN = re.compile(
    r'([a-zA-Z0-9_]+)\^([0-9]+)'
)

# ============================================================================
# NEW SMART PATTERNS (v3.0)
# ============================================================================

# Pattern para {variável} - mostra valor numérico
VALUE_DISPLAY_PATTERN = re.compile(
    r'\{([A-Za-z_][A-Za-z0-9_]*)\}'
)

# Pattern para {{variável}} - mostra fórmula expandida
FORMULA_DISPLAY_PATTERN = re.compile(
    r'\{\{([A-Za-z_][A-Za-z0-9_]*)\}\}'
)

# Pattern para @eq - equações completas
# Suporta: @eq, @eq[mode], @eq[symbolic]
EQUATION_BLOCK_PATTERN = re.compile(
    r'@eq(?:\[(\w+)\])?\s+(.+)',
    re.MULTILINE
)

# Pattern para variáveis inline (palavras)
VARIABLE_INLINE_PATTERN = re.compile(
    r'\b([A-Za-z_][A-Za-z0-9_]*)\b'
)

# Pattern para headings Markdown
HEADING_PATTERN = re.compile(
    r'^(#{1,6})\s+(.+)$',
    re.MULTILINE
)

# Pattern para listas numeradas
NUMBERED_LIST_PATTERN = re.compile(
    r'^\d+\.\s+(.+)$',
    re.MULTILINE
)

# Pattern para listas com marcadores
BULLET_LIST_PATTERN = re.compile(
    r'^[*\-+]\s+(.+)$',
    re.MULTILINE
)

# Pattern para citações ABNT: (SILVA, 2023) ou [1]
CITATION_ABNT_PATTERN = re.compile(
    r'\(([A-Z][A-ZÀ-Ú\s]+),\s*(\d{4})\)'  # (SILVA, 2023)
)

CITATION_NUMERIC_PATTERN = re.compile(
    r'\[(\d+(?:,\s*\d+)*)\]'  # [1], [1, 2, 3]
)

# Pattern para tabelas Markdown
TABLE_PATTERN = re.compile(
    r'^\|.+\|$',
    re.MULTILINE
)

# Pattern para código inline: `code`
CODE_INLINE_PATTERN = re.compile(
    r'`([^`]+)`'
)

# Pattern para blocos de código: ```
CODE_BLOCK_PATTERN = re.compile(
    r'```',
    re.DOTALL
)

# Pattern para imagens Markdown: ![alt](url)
IMAGE_PATTERN = re.compile(
    r'!\[([^\]]*)\]\(([^)]+)\)'
)

# Pattern para links: [text](url)
LINK_PATTERN = re.compile(
    r'\[([^\]]+)\]\(([^)]+)\)'
)

# ============================================================================
# ALIASES para compatibilidade v3.0 (text_processor.py espera esses nomes!)
# ============================================================================

PLACEHOLDER = VALUE_DISPLAY_PATTERN  # {var}
VARNAME = VARIABLE_INLINE_PATTERN    # palavra
GREEKLETTER = GREEK_PATTERN          # alpha, beta, etc
NUMBER = re.compile(r'-?\d+\.?\d*([eE][-+]?\d+)?')  # números

# v3.0 names
VALUEDISPLAYPATTERN = VALUE_DISPLAY_PATTERN
FORMULADISPLAYPATTERN = FORMULA_DISPLAY_PATTERN
EQUATIONBLOCKPATTERN = EQUATION_BLOCK_PATTERN

# ============================================================================
# PATTERN REGISTRY
# ============================================================================

PATTERNS: Dict[str, Pattern] = {
    # Legacy
    'greek': GREEK_PATTERN,
    'subscript': SUBSCRIPT_PATTERN,
    'superscript': SUPERSCRIPT_PATTERN,
    # Smart v3.0
    'value_display': VALUE_DISPLAY_PATTERN,
    'formula_display': FORMULA_DISPLAY_PATTERN,
    'equation_block': EQUATION_BLOCK_PATTERN,
    'variable_inline': VARIABLE_INLINE_PATTERN,
    'heading': HEADING_PATTERN,
    # Lists & Tables
    'numbered_list': NUMBERED_LIST_PATTERN,
    'bullet_list': BULLET_LIST_PATTERN,
    'table': TABLE_PATTERN,
    # Citations
    'citation_abnt': CITATION_ABNT_PATTERN,
    'citation_numeric': CITATION_NUMERIC_PATTERN,
    # Code
    'code_inline': CODE_INLINE_PATTERN,
    'code_block': CODE_BLOCK_PATTERN,
    # Media
    'image': IMAGE_PATTERN,
    'link': LINK_PATTERN,
}

# ============================================================================
# UTILITY FUNCTIONS (v3.0 - FALTAVAM!)
# ============================================================================

def find_variables(text: str) -> List[str]:
    """Find all variable names in text."""
    return VARNAME.findall(text)


def find_numbers(text: str) -> List[float]:
    """Find all numbers in text."""
    matches = NUMBER.findall(text)
    # NUMBER retorna tuplas por causa do grupo de captura
    return [float(m[0] if isinstance(m, tuple) else m) for m in matches]


def find_placeholders(text: str) -> List[str]:
    """Find all {var} placeholders."""
    return PLACEHOLDER.findall(text)


def has_greek_letters(text: str) -> bool:
    """Check if text contains Greek letters."""
    return bool(GREEKLETTER.search(text))


def get_pattern(name: str) -> Pattern:
    """Obtém pattern compilado pelo nome."""
    return PATTERNS.get(name)


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Patterns v2.0 (compatibilidade)
    'PLACEHOLDER',
    'VARNAME',
    'GREEKLETTER',
    'NUMBER',
    # Patterns v3.0
    'VALUEDISPLAYPATTERN',
    'FORMULADISPLAYPATTERN',
    'EQUATIONBLOCKPATTERN',
    # Legacy patterns
    'GREEK_PATTERN',
    'SUBSCRIPT_PATTERN',
    'SUPERSCRIPT_PATTERN',
    'VALUE_DISPLAY_PATTERN',
    'FORMULA_DISPLAY_PATTERN',
    'EQUATION_BLOCK_PATTERN',
    'VARIABLE_INLINE_PATTERN',
    # Functions
    'find_variables',
    'find_numbers',
    'find_placeholders',
    'has_greek_letters',
    'get_pattern',
    # Registry
    'PATTERNS',
]
