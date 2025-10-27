# src/pymemorial/editor/smart_parser.py
"""
Smart Symbol Recognition Engine v2.1 (Two-Pass Parsing)

Intelligent parser for engineering symbols, supporting both direct numeric
assignments and simple expression assignments without requiring @calc.
"""

from __future__ import annotations
import re
import logging
from typing import Dict, List, Tuple, Set, Optional, Any
from dataclasses import dataclass

# SymPy para a Passada 2
try:
    import sympy as sp
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False
    sp = None

logger = logging.getLogger(__name__)

# ... (GREEK_LETTERS, KNOWN_UNITS, SmartSymbol - permanecem iguais) ...
# Greek letters database
GREEK_LETTERS = {
    'alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta',
    'eta', 'theta', 'iota', 'kappa', 'lambda', 'mu',
    'nu', 'xi', 'omicron', 'pi', 'rho', 'sigma',
    'tau', 'upsilon', 'phi', 'chi', 'psi', 'omega'
}

# Base de unidades robusta
KNOWN_UNITS = {
    # Força
    'N', 'kN', 'MN', 'GN', 'tf', 'kgf', 'lbf',
    # Pressão/Tensão
    'Pa', 'kPa', 'MPa', 'GPa', 'psi', 'ksi',
    # Comprimento
    'm', 'cm', 'mm', 'km', 'in', 'ft', 'yd',
    # Área
    'm²', 'm2', 'cm²', 'cm2', 'mm²', 'mm2', 'in²', 'in2',
    # Volume
    'm³', 'm3', 'cm³', 'cm3', 'L', 'mL',
    # Massa
    'kg', 'g', 'mg', 't', 'ton', 'lb',
    # Momento
    'kN.m', 'kNm', 'N.m', 'Nm', 'tf.m', 'tfm',
    # Ângulo
    'rad', 'deg', '°', 'grad',
    # Tempo
    's', 'min', 'h', 'day', 'year',
    # Frequência
    'Hz', 'kHz', 'MHz', 'GHz',
    # Outros
    '%', 'ratio', 'W', 'kW', 'MW', 'J', 'kJ',
}


@dataclass
class SmartSymbol:
    """Parsed engineering symbol"""
    full_name: str
    base: str
    subscripts: List[str]
    is_greek: bool = False

# ============================================================================
# PARSER PRINCIPAL (COM DUAS PASSADAS)
# ============================================================================

class SmartVariableParser:
    """
    Intelligent parser with two-pass detection:
    1. Detects direct numeric assignments (NAME = NUMBER UNIT) using Regex.
    2. Detects simple expression assignments (NAME = EXPRESSION) using SymPy.
    """
    
    def __init__(self):
        self.known_symbols: Dict[str, SmartSymbol] = {}
        self._setup_patterns()
    
    def _setup_patterns(self):
        """Setup regex patterns"""
        # Padrão para encontrar símbolos DENTRO de expressões (usado em find_symbols...)
        self.var_pattern = re.compile(r'\b([a-zA-Z][a-zA-Z0-9_]*)\b')
        
        # Padrão da Passada 1: NOME = NÚMERO UNIDADE # Comentário
        self.numeric_assignment_pattern = re.compile(
            r'^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([0-9.eE+-]+)(?:\s+([^\s#\n]+))?(?:\s*#.*)?$',
            re.MULTILINE
        )
        
        # Padrão da Passada 2: NOME = EXPRESSÃO # Comentário
        # Captura qualquer coisa após o '=' que não seja apenas um número
        self.expression_assignment_pattern = re.compile(
            r'^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+?)(?:\s*#.*)?$',
            re.MULTILINE
        )

    def parse_symbol(self, symbol_name: str) -> SmartSymbol:
        # ... (lógica inalterada) ...
        if symbol_name in self.known_symbols:
            return self.known_symbols[symbol_name]
        parts = symbol_name.split('_')
        base = parts[0]
        subscripts = parts[1:] if len(parts) > 1 else []
        is_greek = base.lower() in GREEK_LETTERS
        symbol = SmartSymbol(
            full_name=symbol_name, base=base,
            subscripts=subscripts, is_greek=is_greek
        )
        self.known_symbols[symbol_name] = symbol
        logger.debug(f"Parsed symbol: {symbol_name} -> {symbol}")
        return symbol
    
    def detect_all_variables(self, text: str) -> Dict[str, Tuple[float, str]]:
        """
        Detect ALL variable declarations (numeric and expression) in two passes.

        Returns:
            Dictionary {name: (value, unit)}
        """
        variables: Dict[str, Tuple[float, str]] = {}
        processed_text = text # Cópia para modificar

        # --- PASSADA 1: Detectar atribuições numéricas diretas ---
        logger.debug("Starting Pass 1: Numeric Assignments")
        lines_processed_pass1 = set()
        
        # Usamos finditer para obter os spans e marcar as linhas processadas
        matches_pass1 = list(self.numeric_assignment_pattern.finditer(processed_text))
        
        for i, match in enumerate(matches_pass1):
            name = match.group(1)
            value_str = match.group(2)
            unit_str = (match.group(3) or '').strip()
            
            # Marca a linha inteira como processada para evitar reprocessamento na Passada 2
            start_line = processed_text.rfind('\n', 0, match.start()) + 1
            end_line = processed_text.find('\n', match.end())
            if end_line == -1: end_line = len(processed_text)
            lines_processed_pass1.add(processed_text[start_line:end_line].strip())

            try:
                value = float(value_str)
            except ValueError:
                logger.warning(f"[Pass 1] Invalid numeric value '{value_str}' for variable '{name}'. Skipping.")
                continue
            
            unit = self._extract_unit_intelligent(unit_str)
            
            if name in variables:
                 logger.warning(f"[Pass 1] Variable '{name}' redefined. Overwriting.")
            
            variables[name] = (value, unit)
            self.parse_symbol(name) # Registra o símbolo
            logger.debug(f"[Pass 1] Detected: {name} = {value} {unit}")
            
        logger.info(f"Pass 1 completed. Detected {len(variables)} numeric variables.")

        # --- PASSADA 2: Detectar atribuições de expressão simples ---
        logger.debug("Starting Pass 2: Simple Expression Assignments")
        if not SYMPY_AVAILABLE:
            logger.warning("SymPy not available. Skipping Pass 2 (expression assignments).")
            return variables

        # Itera sobre todas as linhas que parecem atribuições
        for match in self.expression_assignment_pattern.finditer(text): # Usa o texto original
            line_content = match.group(0).strip()
            
            # Pula se a linha já foi processada na Passada 1
            if line_content in lines_processed_pass1:
                continue
                
            name = match.group(1)
            expr_str = match.group(2).strip()

            # Pula se já definimos essa variável (Passada 1 tem prioridade)
            if name in variables:
                continue

            # Pula se o lado direito for apenas um número (já deveria ter sido pego na Passada 1)
            try:
                 float(expr_str)
                 # Se chegou aqui, é um número puro sem unidade que falhou na Passada 1 por algum motivo
                 logger.warning(f"[Pass 2] Line '{line_content}' looks numeric but failed Pass 1. Treating as number.")
                 variables[name] = (float(expr_str), '')
                 self.parse_symbol(name)
                 logger.debug(f"[Pass 2] Fallback Detected: {name} = {float(expr_str)}")
                 continue
            except ValueError:
                 pass # É uma expressão, continuar

            # Tenta avaliar a expressão usando SymPy e as variáveis da Passada 1
            local_dict_sympy = {n: sp.Symbol(n, real=True) for n in variables.keys()}
            subs_dict_values = {sp.Symbol(n, real=True): v[0] for n, v in variables.items()} # {symbol: value}

            try:
                # Usa evaluate=True para tentar obter um resultado numérico
                sympy_expr = sp.sympify(expr_str, locals=local_dict_sympy, evaluate=True)
                
                # Substitui os símbolos pelos valores numéricos conhecidos
                numeric_result = sympy_expr.subs(subs_dict_values)

                # Verifica se o resultado é puramente numérico
                if numeric_result.is_Number:
                    value = float(numeric_result)
                    # Passada 2 NÃO tenta adivinhar unidades
                    unit = '' 
                    
                    if name in variables:
                         logger.warning(f"[Pass 2] Variable '{name}' redefined. Overwriting.")
                    
                    variables[name] = (value, unit)
                    self.parse_symbol(name)
                    logger.debug(f"[Pass 2] Detected: {name} = {expr_str} -> {value}")
                else:
                    # A expressão não pôde ser totalmente avaliada (ex: 'a = b + c' e 'c' não definido)
                    logger.debug(f"[Pass 2] Expression for '{name}' ('{expr_str}') could not be fully evaluated to a number. Skipping. Result: {numeric_result}")

            except (SyntaxError, TypeError, NameError, Exception) as e:
                # Falha ao parsear ou avaliar com SymPy (pode ser complexa demais ou inválida)
                logger.debug(f"[Pass 2] Failed to parse/evaluate expression for '{name}' ('{expr_str}'): {e}. Skipping.")
        
        logger.info(f"Pass 2 completed. Total variables detected: {len(variables)}")
        return variables

    def _extract_unit_intelligent(self, text: str) -> str:
        # ... (lógica inalterada) ...
        if not text: return ''
        text = text.strip()
        if text in KNOWN_UNITS: return text
        unit_chars = {'.', '/', '²', '³', '⁴', '⁵', '⁻', '⁰', '°'}
        if any(c in text for c in unit_chars): return text
        if len(text) <= 3 and any(c.isupper() for c in text): return text
        #logger.debug(f"Rejecting '{text}' as unit (looks like variable name)") # Log muito verboso
        return ''
    
    def find_symbols_in_expression(self, expression: str) -> Set[str]:
         # ... (lógica inalterada) ...
        symbols = set()
        # Adiciona palavras reservadas do SymPy que podem aparecer
        KEYWORDS = {
            'if', 'in', 'or', 'and', 'not', 
            'sin', 'cos', 'tan', 'exp', 'log', 'sqrt', 'pi',
            'integrate', 'diff', 'limit', 'Symbol', 'Eq', 'solve' 
        }
        
        for match in self.var_pattern.finditer(expression):
            name = match.group(1)
            
            # Exclui keywords e números puros
            if name not in KEYWORDS and not name.isdigit():
                symbols.add(name)
                self.parse_symbol(name) # Garante que o símbolo seja conhecido
        
        return symbols

__all__ = ['SmartVariableParser', 'SmartSymbol', 'GREEK_LETTERS', 'KNOWN_UNITS']