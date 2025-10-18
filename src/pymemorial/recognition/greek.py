"""
Mapeamento de símbolos gregos para uso em memoriais de cálculo.
Permite conversão bidirecional entre Unicode e ASCII.
"""
from typing import Dict, Optional

# Mapeamento completo de letras gregas (minúsculas e maiúsculas)
GREEK_LOWER: Dict[str, str] = {
    "α": "alpha",
    "β": "beta", 
    "γ": "gamma",
    "δ": "delta",
    "ε": "epsilon",
    "ζ": "zeta",
    "η": "eta",
    "θ": "theta",
    "ι": "iota",
    "κ": "kappa",
    "λ": "lambda",
    "μ": "mu",
    "ν": "nu",
    "ξ": "xi",
    "ο": "omicron",
    "π": "pi",
    "ρ": "rho",
    "σ": "sigma",
    "ς": "sigma",  # sigma final (mesmo nome)
    "τ": "tau",
    "υ": "upsilon",
    "φ": "phi",
    "χ": "chi",
    "ψ": "psi",
    "ω": "omega",
}

GREEK_UPPER: Dict[str, str] = {
    "Α": "Alpha",
    "Β": "Beta",
    "Γ": "Gamma",
    "Δ": "Delta",
    "Ε": "Epsilon",
    "Ζ": "Zeta",
    "Η": "Eta",
    "Θ": "Theta",
    "Ι": "Iota",
    "Κ": "Kappa",
    "Λ": "Lambda",
    "Μ": "Mu",
    "Ν": "Nu",
    "Ξ": "Xi",
    "Ο": "Omicron",
    "Π": "Pi",
    "Ρ": "Rho",
    "Σ": "Sigma",
    "Τ": "Tau",
    "Υ": "Upsilon",
    "Φ": "Phi",
    "Χ": "Chi",
    "Ψ": "Psi",
    "Ω": "Omega",
}

# Mapeamento unificado
GREEK_TO_ASCII = {**GREEK_LOWER, **GREEK_UPPER}

# Mapeamento reverso (ASCII → Unicode)
# IMPORTANTE: usar sigma normal (σ), não final (ς)
ASCII_TO_GREEK = {
    "alpha": "α",
    "beta": "β",
    "gamma": "γ",
    "delta": "δ",
    "epsilon": "ε",
    "zeta": "ζ",
    "eta": "η",
    "theta": "θ",
    "iota": "ι",
    "kappa": "κ",
    "lambda": "λ",
    "mu": "μ",
    "nu": "ν",
    "xi": "ξ",
    "omicron": "ο",
    "pi": "π",
    "rho": "ρ",
    "sigma": "σ",  # Sempre usar sigma normal
    "tau": "τ",
    "upsilon": "υ",
    "phi": "φ",
    "chi": "χ",
    "psi": "ψ",
    "omega": "ω",
}


class GreekSymbols:
    """Utilitários para conversão de símbolos gregos."""
    
    @staticmethod
    def to_ascii(text: str) -> str:
        """
        Converte símbolos gregos Unicode para ASCII.
        
        Args:
            text: texto com símbolos gregos
        
        Returns:
            Texto com nomes ASCII
        
        Examples:
            >>> GreekSymbols.to_ascii("Coeficiente α = 0.85")
            "Coeficiente alpha = 0.85"
        """
        result = text
        for greek, ascii_name in GREEK_TO_ASCII.items():
            result = result.replace(greek, ascii_name)
        return result
    
    @staticmethod
    def to_unicode(text: str) -> str:
        """
        Converte nomes ASCII para símbolos gregos Unicode.
        
        Args:
            text: texto com nomes ASCII
        
        Returns:
            Texto com símbolos Unicode
        
        Examples:
            >>> GreekSymbols.to_unicode("alpha = 0.85")
            "α = 0.85"
        """
        result = text
        for ascii_name, greek in ASCII_TO_GREEK.items():
            # Usar word boundary para evitar substituições parciais
            import re
            pattern = r'\b' + re.escape(ascii_name) + r'\b'
            result = re.sub(pattern, greek, result)
        return result
    
    @staticmethod
    def get_ascii_name(symbol: str) -> Optional[str]:
        """Retorna o nome ASCII de um símbolo grego."""
        return GREEK_TO_ASCII.get(symbol)
    
    @staticmethod
    def get_unicode_symbol(name: str) -> Optional[str]:
        """Retorna o símbolo Unicode de um nome ASCII."""
        return ASCII_TO_GREEK.get(name)
