"""
Processador de texto para templates de memoriais de cálculo.
"""
from typing import Dict, Any, Optional
import re
from .patterns import PLACEHOLDER
from .greek import GreekSymbols


class TextProcessor:
    """
    Processa templates com substituição de variáveis.
    
    Examples:
        >>> processor = TextProcessor()
        >>> template = "A resistência {{fck}} é de {{valor}} MPa."
        >>> context = {"fck": "característica", "valor": 30}
        >>> processor.render(template, context)
        'A resistência característica é de 30 MPa.'
    """
    
    def __init__(self, enable_latex: bool = True):
        """
        Args:
            enable_latex: se True, converte símbolos para LaTeX
        """
        self.enable_latex = enable_latex
    
    def render(self, template: str, context: Dict[str, Any]) -> str:
        """
        Renderiza template substituindo placeholders.
        
        Args:
            template: texto com {{var}}
            context: dicionário de valores
        
        Returns:
            Texto renderizado
        """
        result = template
        
        for match in PLACEHOLDER.finditer(template):
            placeholder = match.group(0)  # {{var}}
            var_name = match.group(1)     # var
            
            if var_name in context:
                value = context[var_name]
                result = result.replace(placeholder, str(value))
        
        return result
    
    def to_latex(self, text: str, greek_to_math: bool = True) -> str:
        """
        Converte texto para formato LaTeX.
        
        Args:
            text: texto original
            greek_to_math: se True, coloca símbolos gregos em math mode
        
        Returns:
            Texto em LaTeX
        """
        result = text
        
        # Escapar caracteres especiais LaTeX
        special_chars = {'_': r'\_', '%': r'\%', '&': r'\&', '#': r'\#'}
        for char, escaped in special_chars.items():
            result = result.replace(char, escaped)
        
        # Converter símbolos gregos para comandos LaTeX
        if greek_to_math:
            greek_map = {
                'α': r'$\alpha$',
                'β': r'$\beta$',
                'γ': r'$\gamma$',
                'δ': r'$\delta$',
                'σ': r'$\sigma$',
                'τ': r'$\tau$',
                'φ': r'$\phi$',
                'ω': r'$\omega$',
            }
            for greek, latex in greek_map.items():
                result = result.replace(greek, latex)
        
        return result
    
    def extract_and_replace(
        self,
        text: str,
        replacements: Dict[str, str],
        preserve_original: bool = False
    ) -> str:
        """
        Extrai placeholders e substitui por valores do dicionário.
        
        Args:
            text: texto com {{var}}
            replacements: mapa var → valor
            preserve_original: se True, mantém placeholder quando não encontrar valor
        
        Returns:
            Texto processado
        """
        def replace_fn(match):
            var_name = match.group(1)
            if var_name in replacements:
                return str(replacements[var_name])
            elif preserve_original:
                return match.group(0)  # Mantém {{var}}
            else:
                return ""  # Remove placeholder
        
        return PLACEHOLDER.sub(replace_fn, text)
    
    def validate_template(self, template: str) -> tuple[bool, list[str]]:
        """
        Valida template e retorna variáveis necessárias.
    
        Args:
            template: texto para validar
    
        Returns:
            (é_válido, lista_de_variáveis_requeridas)
        """
        placeholders = PLACEHOLDER.findall(template)
    
        # Verifica placeholders malformados:
        # - Chaves únicas isoladas: {var} (deveria ser {{var}})
        # - Chaves desbalanceadas: {{var} ou {var}}
        # Regex: procura { ou } que não fazem parte de {{ ou }}
        malformed = re.findall(r'(?<!\{)\{(?!\{)|(?<!\})\}(?!\})', template)
    
        is_valid = len(malformed) == 0
        return is_valid, list(set(placeholders))

