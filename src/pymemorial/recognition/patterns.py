"""
Padrões regex compilados para parsing de variáveis e expressões.
"""
import re
from typing import Pattern

# Padrão para identificadores de variáveis
# Aceita: x, x1, fck, alpha_c, sigma_max
VAR_NAME: Pattern = re.compile(
    r'\b[a-zA-Z_][a-zA-Z0-9_]*\b'
)

# Padrão para números (int, float, científico)
# Aceita: 25, 3.14, 1.5e-3, -42.0
NUMBER: Pattern = re.compile(
    r'[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?'
)

# Padrão para unidades (após número)
# Aceita: kN, MPa, m², kg/m³
UNIT: Pattern = re.compile(
    r'[a-zA-Z]+(?:/[a-zA-Z]+)?(?:\*\*\d+)?'
)

# Padrão para placeholder de template {{var}}
PLACEHOLDER: Pattern = re.compile(
    r'\{\{([a-zA-Z_][a-zA-Z0-9_]*)\}\}'
)

# Padrão para detectar símbolo grego
GREEK_LETTER: Pattern = re.compile(
    r'[α-ωΑ-Ω]'
)

# Padrão para equação completa (var = expr)
EQUATION_PATTERN: Pattern = re.compile(
    r'([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(.+)'
)


def find_variables(text: str) -> list[str]:
    """
    Encontra todos os identificadores de variáveis no texto.
    
    Args:
        text: texto para analisar
    
    Returns:
        Lista de nomes de variáveis únicos
    """
    return list(set(VAR_NAME.findall(text)))


def find_numbers(text: str) -> list[float]:
    """Encontra todos os números no texto."""
    matches = NUMBER.findall(text)
    return [float(m) for m in matches]


def find_placeholders(text: str) -> list[str]:
    """Encontra placeholders {{var}} no template."""
    return PLACEHOLDER.findall(text)


def has_greek_letters(text: str) -> bool:
    """Verifica se o texto contém letras gregas."""
    return GREEK_LETTER.search(text) is not None
