"""
Padrões regex compilados para parsing (v2.0: Whitelist Engineering + Utils Expand).

Base pra text_processor MVP: ENGINEERING_VAR (M_, gamma_ zero false positives).
Novo: find_equations, has_units. Compat 100%.

Exemplo:
    find_equations("M_d = M_k * gamma_s") → [('M_d', 'M_k * gamma_s')]
"""

import re
from typing import Pattern, List, Tuple, Optional

# Legacy
VAR_NAME: Pattern = re.compile(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b')
NUMBER: Pattern = re.compile(r'[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?')
UNIT: Pattern = re.compile(r'[a-zA-Z]+(?:/[a-zA-Z]+)?(?:\*\*\d+)?')
PLACEHOLDER: Pattern = re.compile(r'\{\{([a-zA-Z_][a-zA-Z0-9_]*)\}\}')
GREEK_LETTER: Pattern = re.compile(r'[α-ωΑ-Ω]')
EQUATION_PATTERN: Pattern = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(.+)')

# NEW: Whitelist Engineering (MVP tie-in)
ENGINEERING_VAR: Pattern = re.compile(r'\b([MNVQPFEA]|gamma|sigma|chi|phi|mu|alpha|beta|delta|tau)_[a-z]{1,3}\b', re.I)


def find_variables(text: str, engineering_only: bool = False) -> List[str]:
    """Encontra variáveis (legacy or whitelist)."""
    pattern = ENGINEERING_VAR if engineering_only else VAR_NAME
    return list(set(pattern.findall(text)))

def find_numbers(text: str) -> List[float]:
    """Encontra números."""
    return [float(m) for m in NUMBER.findall(text)]

def find_placeholders(text: str) -> List[str]:
    """Encontra {{var}}."""
    return PLACEHOLDER.findall(text)

def has_greek_letters(text: str) -> bool:
    """Verifica gregos."""
    return GREEK_LETTER.search(text) is not None

# NEW: Utils Expand
def find_equations(text: str) -> List[Tuple[str, str]]:
    """Encontra equações (var = expr)."""
    return [(m.group(1), m.group(2).strip()) for m in EQUATION_PATTERN.finditer(text)]

def has_units(text: str) -> bool:
    """Verifica unidades após números."""
    return bool(re.search(r'\d\s*' + UNIT.pattern, text))


__all__ = [
    'find_variables', 'find_numbers', 'find_placeholders', 'has_greek_letters',
    'find_equations', 'has_units',  # NEW
    'ENGINEERING_VAR',  # NEW whitelist
]