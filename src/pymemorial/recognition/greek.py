# src/pymemorial/recognition/greek.py
r"""
Conversores de símbolos gregos (Unicode ↔ ASCII ↔ LaTeX).

Mapeia símbolos gregos para uso em memoriais de cálculo:
- Unicode: α, β, γ, σ
- ASCII: alpha, beta, gamma, sigma
- LaTeX: \alpha, \beta, \gamma, \sigma

Example:
    >>> GreekSymbols.to_unicode("gamma_s = 1.4")
    'γ_s = 1.4'
    
    >>> GreekSymbols.to_latex("γ_s")
    '$\\gamma_{s}$'
"""

# Mapa completo Unicode ↔ ASCII
ASCII_TO_GREEK = {
    'alpha': 'α',
    'beta': 'β',
    'gamma': 'γ',
    'delta': 'δ',
    'epsilon': 'ε',
    'zeta': 'ζ',
    'eta': 'η',
    'theta': 'θ',
    'iota': 'ι',
    'kappa': 'κ',
    'lambda': 'λ',
    'mu': 'μ',
    'nu': 'ν',
    'xi': 'ξ',
    'omicron': 'ο',
    'pi': 'π',
    'rho': 'ρ',
    'sigma': 'σ',
    'tau': 'τ',
    'upsilon': 'υ',
    'phi': 'φ',
    'chi': 'χ',
    'psi': 'ψ',
    'omega': 'ω',
}

# Mapa reverso (Unicode → ASCII)
GREEK_TO_ASCII = {v: k for k, v in ASCII_TO_GREEK.items()}

# Mapa para LaTeX
GREEK_TO_LATEX = {
    symbol: f"\\{name}" for name, symbol in ASCII_TO_GREEK.items()
}


class GreekSymbols:
    r"""
    Utilitário para conversão de símbolos gregos.
    
    Suporta:
    - Unicode ↔ ASCII
    - Unicode → LaTeX
    - Subscritos (gamma_s → γ_s ou $\gamma_{s}$)
    """
    
    @staticmethod
    def to_unicode(text: str) -> str:
        """
        Converte nomes ASCII para Unicode.
        
        Args:
            text: Texto com nomes ASCII (ex: "gamma_s = 1.4")
        
        Returns:
            Texto com símbolos Unicode (ex: "γ_s = 1.4")
        
        Example:
            >>> GreekSymbols.to_unicode("gamma_s * alpha_m")
            'γ_s * α_m'
        """
        result = text
        for ascii_name, unicode_symbol in ASCII_TO_GREEK.items():
            # Word boundary para evitar substituição parcial
            import re
            pattern = r'\b' + ascii_name + r'\b'
            result = re.sub(pattern, unicode_symbol, result, flags=re.IGNORECASE)
        
        return result
    
    @staticmethod
    def to_latex(text: str, with_subscripts: bool = True) -> str:
        r"""
        Converte símbolos para LaTeX.
        
        Args:
            text: Texto com símbolos Unicode ou ASCII
            with_subscripts: Se True, formata subscritos (gamma_s → $\gamma_{s}$)
        
        Returns:
            Texto em LaTeX
        
        Example:
            >>> GreekSymbols.to_latex("γ_s")
            '$\\gamma_{s}$'
        """
        result = text
        
        # 1. Unicode → LaTeX
        for unicode_symbol, latex_cmd in GREEK_TO_LATEX.items():
            if unicode_symbol in result:
                # Detecta subscrito
                if with_subscripts:
                    import re
                    pattern = re.escape(unicode_symbol) + r'_([a-zA-Z0-9]+)'
                    replacement = f"${latex_cmd}_{{\\1}}$"
                    result = re.sub(pattern, replacement, result)
                    
                    # Sem subscrito
                    if unicode_symbol in result:
                        result = result.replace(unicode_symbol, f"${latex_cmd}$")
                else:
                    result = result.replace(unicode_symbol, f"${latex_cmd}$")
        
        # 2. ASCII → LaTeX
        for ascii_name, unicode_symbol in ASCII_TO_GREEK.items():
            if ascii_name in result.lower():
                import re
                if with_subscripts:
                    pattern = r'\b' + ascii_name + r'_([a-zA-Z0-9]+)\b'
                    replacement = f"$\\{ascii_name}_{{\\1}}$"
                    result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
                
                # Sem subscrito
                pattern_no_sub = r'\b' + ascii_name + r'\b'
                replacement_no_sub = f"$\\{ascii_name}$"
                result = re.sub(pattern_no_sub, replacement_no_sub, result, flags=re.IGNORECASE)
        
        return result
    
    @staticmethod
    def get_unicode(ascii_name: str) -> str:
        """Retorna símbolo Unicode para nome ASCII."""
        return ASCII_TO_GREEK.get(ascii_name.lower(), ascii_name)
    
    @staticmethod
    def get_latex(unicode_symbol: str) -> str:
        """Retorna comando LaTeX para símbolo Unicode."""
        return GREEK_TO_LATEX.get(unicode_symbol, unicode_symbol)


__all__ = [
    'GreekSymbols',
    'ASCII_TO_GREEK',
    'GREEK_TO_ASCII',
    'GREEK_TO_LATEX',
]
