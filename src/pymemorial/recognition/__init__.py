"""API pública do módulo de reconhecimento."""
from .greek import GreekSymbols, GREEK_TO_ASCII, ASCII_TO_GREEK
from .parser import VariableParser, ParsedVariable
from .text_processor import TextProcessor
from .patterns import (
    find_variables,
    find_numbers,
    find_placeholders,
    has_greek_letters,
)

__all__ = [
    # Classes principais
    "GreekSymbols",
    "VariableParser",
    "ParsedVariable",
    "TextProcessor",
    # Dicionários de símbolos
    "GREEK_TO_ASCII",
    "ASCII_TO_GREEK",
    # Funções utilitárias
    "find_variables",
    "find_numbers",
    "find_placeholders",
    "has_greek_letters",
]
