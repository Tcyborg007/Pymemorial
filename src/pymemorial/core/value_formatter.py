"""
Formatador de valores robusto - evita zeros falsos e arredondamentos destrutivos
"""

import numpy as np
import re
from typing import Union


class RobustValueFormatter:
    """Formatador de valores à prova de falhas."""
    
    @staticmethod
    def format_value(value: Union[int, float], 
                     precision: int = 2,
                     avoid_zero: bool = True) -> str:
        """
        Formata valor de forma robusta e inteligente.
        
        Args:
            value: Valor a formatar
            precision: Casas decimais
            avoid_zero: NUNCA transformar valor não-zero em "0"
        
        Returns:
            String formatada (LaTeX-compatible)
        """
        # ✅ PROTEÇÃO CRÍTICA #1: Detecção de pseudo-zeros
        if avoid_zero and isinstance(value, float):
            if value != 0 and abs(value) < 0.01:
                # NUNCA arredondar para zero!
                return RobustValueFormatter._format_scientific(value, precision)
        
        # ✅ PROTEÇÃO CRÍTICA #2: Checagem de valor real zero
        if value == 0 or abs(value) < 1e-15:
            return "0"
        
        # ✅ PROTEÇÃO CRÍTICA #3: Magnitude analysis
        magnitude = np.floor(np.log10(abs(value)))
        
        # Notação científica para valores pequenos
        if abs(value) < 0.01:
            return RobustValueFormatter._format_scientific(value, precision)
        
        # Notação com separador para valores grandes
        elif abs(value) >= 1000:
            return RobustValueFormatter._format_large(value, precision)
        
        # Notação decimal normal
        else:
            if magnitude >= 0:
                decimals = precision
            else:
                decimals = precision + int(abs(magnitude))
            
            formatted = f"{value:.{decimals}f}"
            
            # ✅ PROTEÇÃO CRÍTICA #4: Detectar "0.0" espúrio
            if formatted == "0.0" or formatted.startswith("0.00"):
                if value != 0:
                    return RobustValueFormatter._format_scientific(value, precision)
            
            return formatted
    
    
    @staticmethod
    def _format_scientific(value: float, precision: int = 2) -> str:
        """Formata em notação científica com proteção."""
        if value == 0:
            return "0"
        
        # Calcular expoente
        exp = int(np.floor(np.log10(abs(value))))
        mantissa = value / (10 ** exp)
        
        # ✅ PROTEÇÃO: Nunca deixar mantissa como 0.0
        if abs(mantissa) < 0.1:
            exp -= 1
            mantissa = value / (10 ** exp)
        
        # Formatar
        mantissa_str = f"{mantissa:.{precision}f}"
        
        return f"{mantissa_str} \\times 10^{{{exp}}}"
    
    
    @staticmethod
    def _format_large(value: float, precision: int = 2) -> str:
        """Formata valores grandes com separador."""
        return f"{value:,.{precision}f}".replace(",", "\\,")
    
    
    @staticmethod
    def validate_expression(expr_latex: str) -> str:
        """
        Valida e corrige expressão LaTeX com problemas de zero.
        
        Args:
            expr_latex: Expressão LaTeX
        
        Returns:
            Expressão corrigida
        """
        # ✅ Detectar e alertar sobre "0.0 ×"
        if re.search(r'0\.0+\s*\\times', expr_latex):
            raise ValueError(
                "❌ ERRO CRÍTICO: Detectado zero espúrio na expressão! "
                "Há um problema com arredondamento destrutivo."
            )
        
        return expr_latex


# ============================================================================
# FUNÇÕES HELPER
# ============================================================================

def safe_format(value: Union[int, float], precision: int = 2) -> str:
    """Helper simples para uso rápido."""
    return RobustValueFormatter.format_value(value, precision)


def validate_latex_expr(expr: str) -> bool:
    """Valida se expressão tem problemas."""
    return not bool(re.search(r'0\.0+\s*\\times', expr))
