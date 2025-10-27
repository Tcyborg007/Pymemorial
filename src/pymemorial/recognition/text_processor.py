# src/pymemorial/recognition/text_processor.py
"""
Processador Inteligente de Texto - VERSÃO 3.0 EXTENDED

MANTÉM 100% COMPATIBILIDADE COM v2.0:
- SmartTextEngine (v2.0) ✅
- TextProcessor (wrapper) ✅
- DetectedVariable ✅
- Todos os métodos existentes ✅

ADICIONA v3.0 (Smart Natural Language):
- SmartTextProcessor (escrita natural + matemática)
- VariableRegistry (contexto global de variáveis)
- EquationParser (parsing com SymPy)
- Support para {var}, {{var}}, [eq:...]
- ABNT/TCC/Article formatting

Migration Path:
    # v2.0 code (FUNCIONA SEM MUDANÇAS):
    engine = SmartTextEngine()
    result = engine.process_text("M_k = 150 kN", {'M_k': 150})
    
    # v3.0 new features:
    processor = SmartTextProcessor()
    processor.define_variables({'M_k': (112.5, 'kN.m', 'Momento')})
    result = processor.process("[eq:M_d = gamma_f * M_k]")

Author: PyMemorial Team
Date: 2025-10-21
Version: 3.0.0 (Extended from 2.0.0)
Phase: PHASE 1-2 (Recognition) + PHASE 7 (Document Integration)
"""

from __future__ import annotations

import ast
import logging
import re
import warnings
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum

# SymPy (opcional para v3.0)
try:
    import sympy as sp
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False
    sp = None

# ============================================================================
# IMPORTS CORRIGIDOS v3.0
# ============================================================================

# Imports patterns v3.0 - CORRIGIDO!
try:
    from .patterns import (
        PLACEHOLDER,
        VARNAME,
        GREEKLETTER,
        NUMBER,
        VALUEDISPLAYPATTERN,
        FORMULADISPLAYPATTERN,
        EQUATIONBLOCKPATTERN,
    )
except ImportError:
    # Fallback patterns se patterns.py não existir
    PLACEHOLDER = re.compile(r'\{([a-zA-Z_][a-zA-Z0-9_]*)\}')
    VARNAME = re.compile(r'[a-zA-Z_][a-zA-Z0-9_]*')
    GREEKLETTER = re.compile(r'(alpha|beta|gamma|delta|epsilon)', re.IGNORECASE)
    NUMBER = re.compile(r'-?\d+\.?\d*')
    VALUEDISPLAYPATTERN = re.compile(r'\{\{([A-Za-z_][A-Za-z0-9_]*)\}\}')
    FORMULADISPLAYPATTERN = re.compile(r'\{\{\{([A-Za-z_][A-Za-z0-9_]*)\}\}\}')
    EQUATIONBLOCKPATTERN = re.compile(r'\[eq(?:\((\w+?)\))?:(.*?)\]', re.MULTILINE)

# Greek symbols
try:
    from .greek import GreekSymbols, ASCII_TO_GREEK
except ImportError:
    class GreekSymbols:
        @staticmethod
        def to_unicode(text: str) -> str:
            return text
        
        @staticmethod
        def to_latex(text: str) -> str:
            return text
    
    ASCII_TO_GREEK = {}

# Pint (opcional)
try:
    from pint import Quantity as PintQuantity
    PINT_AVAILABLE = True
except ImportError:
    PINT_AVAILABLE = False
    PintQuantity = None

# PyMemorial core (para v3.0)
try:
    from pymemorial.core import Variable, Equation
    from pymemorial.core.variable import VariableFactory
    PYMEMORIAL_CORE_AVAILABLE = True
except ImportError:
    PYMEMORIAL_CORE_AVAILABLE = False
    Variable = None
    VariableFactory = None
    Equation = None

# Logger
_logger = logging.getLogger(__name__)


# ============================================================================
# v2.0 CODE (MANTIDO 100%) - NÃO MEXER!
# ============================================================================

@dataclass
class DetectedVariable:
    """v2.0 - Variável detectada automaticamente."""
    name: str
    base: str
    subscript: str = ""
    is_greek: bool = False
    value: Optional[Any] = None
    latex: str = ""
    
    def __post_init__(self):
        if not self.latex:
            self.latex = self._generate_latex()
    
    def _generate_latex(self) -> str:
        if self.is_greek and self.base.lower() in ASCII_TO_GREEK:
            base_latex = f"\\{self.base.lower()}"
        else:
            base_latex = self.base
        
        if self.subscript:
            return f"${base_latex}_{{{self.subscript}}}$"
        else:
            return f"${base_latex}$"


class SmartTextEngine:
    """v2.0 - Engine de processamento inteligente (ORIGINAL)."""
    
    ENGINEERING_VAR_PATTERN = re.compile(
        r'\b([MNVQPFEA]_[a-zA-Z]{1,4}|'
        r'f_[a-z]{1,3}|'
        r'(?:gamma|sigma|tau|chi|phi|mu|alpha|beta|delta|epsilon|omega)_[a-z]{1,3})\b',
        re.IGNORECASE
    )
    
    GREEK_TO_LATEX = {
        'alpha': r'\alpha', 'beta': r'\beta', 'gamma': r'\gamma',
        'delta': r'\delta', 'epsilon': r'\epsilon', 'zeta': r'\zeta',
        'eta': r'\eta', 'theta': r'\theta', 'iota': r'\iota',
        'kappa': r'\kappa', 'lambda': r'\lambda', 'mu': r'\mu',
        'nu': r'\nu', 'xi': r'\xi', 'pi': r'\pi',
        'rho': r'\rho', 'sigma': r'\sigma', 'tau': r'\tau',
        'upsilon': r'\upsilon', 'phi': r'\phi', 'chi': r'\chi',
        'psi': r'\psi', 'omega': r'\omega',
    }
    
    def __init__(self, enable_latex: bool = True, enable_auto_detect: bool = True):
        self.enable_latex = enable_latex
        self.enable_auto_detect = enable_auto_detect
        self._logger = _logger
    
    def process_text(self, text: str, context: Optional[Dict[str, Any]] = None,
                     auto_format: bool = True) -> str:
        """v2.0 - Processa texto (ORIGINAL)."""
        context = context or {}
        result = text
        result = self._process_placeholders(result, context, auto_format)
        if self.enable_auto_detect:
            result = self._process_auto_variables(result, context)
        result = self._convert_greek_names(result)
        return result
    
    def render(self, template: str, context: Dict[str, Any]) -> str:
        return self._process_placeholders(template, context, auto_format=False)
    
    def to_latex(self, text: str, escape_special: bool = True) -> str:
        result = text
        if escape_special:
            result = self._escape_latex(result)
        result = self._greek_to_latex_commands(result)
        return result
    
    def extract_and_replace(self, text: str, replacements: Dict[str, str],
                           preserve_original: bool = False) -> str:
        def replace_fn(match):
            var_name = match.group(1)
            if var_name in replacements:
                return str(replacements[var_name])
            elif preserve_original:
                return match.group(0)
            else:
                return ""
        return PLACEHOLDER.sub(replace_fn, text)
    
    def validate_template(self, template: str) -> Tuple[bool, List[str]]:
        valid_placeholders = PLACEHOLDER.findall(template)
        required_vars = list(set(valid_placeholders))
        malformed_pattern = r'(?<!\{)\{(?!\{)|(?<!\})\}(?!\})'
        malformed = re.findall(malformed_pattern, template)
        open_count = template.count('{{')
        close_count = template.count('}}')
        is_valid = (len(malformed) == 0) and (open_count == close_count)
        return (is_valid, required_vars)
    
    def _process_placeholders(self, text: str, context: Dict[str, Any],
                              auto_format: bool) -> str:
        result = text
        for match in PLACEHOLDER.finditer(text):
            placeholder = match.group(0)
            var_name = match.group(1)
            if var_name in context:
                value = context[var_name]
                formatted_value = self._format_value(value) if auto_format else str(value)
                result = result.replace(placeholder, formatted_value)
        return result
    
    def _process_auto_variables(self, text: str, context: Dict[str, Any]) -> str:
        detected = self._detect_engineering_variables(text)
        result = text
        for var in detected:
            if var.name in context:
                pattern = r'\b' + re.escape(var.name) + r'(?!\})'
                if self.enable_latex:
                    replacement = var.latex
                    result = re.sub(pattern, lambda m: replacement, result)
        return result
    
    def _detect_engineering_variables(self, text: str) -> List[DetectedVariable]:
        detected = []
        for match in self.ENGINEERING_VAR_PATTERN.finditer(text):
            var_name = match.group(1)
            if '_' in var_name:
                base, subscript = var_name.split('_', 1)
            else:
                base, subscript = var_name, ""
            is_greek = base.lower() in self.GREEK_TO_LATEX
            var = DetectedVariable(name=var_name, base=base, subscript=subscript,
                                   is_greek=is_greek)
            detected.append(var)
        unique_vars = list({v.name: v for v in detected}.values())
        return unique_vars
    
    def _convert_greek_names(self, text: str) -> str:
        return GreekSymbols.to_unicode(text)
    
    def _format_value(self, value: Any) -> str:
        if PINT_AVAILABLE and isinstance(value, PintQuantity):
            magnitude = value.magnitude
            unit_str = str(value.units)
            unit_map = {'kilonewton': 'kN', 'megapascal': 'MPa',
                       'meter': 'm', 'millimeter': 'mm'}
            for long_unit, short_unit in unit_map.items():
                if long_unit in unit_str:
                    unit_str = unit_str.replace(long_unit, short_unit)
            return f"{magnitude:.2f} {unit_str}"
        elif isinstance(value, (int, float)):
            if isinstance(value, float):
                return f"{value:.2f}"
            else:
                return str(value)
        else:
            return str(value)
    
    def _escape_latex(self, text: str) -> str:
        special_chars = {
            '\\': r'\textbackslash{}', '_': r'\_', '%': r'\%',
            '&': r'\&', '#': r'\#', '{': r'\{', '}': r'\}',
            '$': r'\$', '^': r'\^{}', '~': r'\textasciitilde{}',
        }
        result = text
        for char, escaped in special_chars.items():
            result = result.replace(char, escaped)
        return result
    
    def _greek_to_latex_commands(self, text: str) -> str:
        result = text
        for ascii_name, unicode_symbol in ASCII_TO_GREEK.items():
            if unicode_symbol in result:
                latex_cmd = self.GREEK_TO_LATEX.get(ascii_name, ascii_name)
                result = result.replace(unicode_symbol, f"${latex_cmd}$")
        return result
    
    def safe_eval_expression(self, expr_str: str,
                            context: Dict[str, Union[int, float]]) -> Optional[float]:
        try:
            tree = ast.parse(expr_str, mode='eval')
            allowed_nodes = (ast.Expression, ast.BinOp, ast.UnaryOp,
                           ast.Num, ast.Constant, ast.Name, ast.Load,
                           ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow,
                           ast.USub, ast.UAdd)
            for node in ast.walk(tree):
                if not isinstance(node, allowed_nodes):
                    raise ValueError(f"Operação não permitida: {node.__class__.__name__}")
            code = compile(tree, '<string>', 'eval')
            result = eval(code, {"__builtins__": {}}, context)
            return float(result)
        except Exception as e:
            self._logger.warning(f"Expressão inválida '{expr_str}': {e}")
            return None


class TextProcessor(SmartTextEngine):
    """v2.0 - Wrapper compatibilidade (ORIGINAL)."""
    def __init__(self, enable_latex: bool = True):
        super().__init__(enable_latex=enable_latex, enable_auto_detect=False)


_engine_instance: Optional[SmartTextEngine] = None


def get_engine(auto_detect: bool = True) -> SmartTextEngine:
    """v2.0 - Singleton (ORIGINAL)."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = SmartTextEngine(enable_latex=True,
                                          enable_auto_detect=auto_detect)
    return _engine_instance


# ============================================================================
# v3.0 NEW CODE (EXTENDED FEATURES)
# ============================================================================

# Enums
class DocumentType(Enum):
    MEMORIAL = "memorial"
    ARTICLE = "article"
    TCC = "tcc"
    REPORT = "report"


class RenderMode(Enum):
    FULL = "full"
    SYMBOLIC = "symbolic"
    NUMERIC = "numeric"
    RESULT = "result"


class CitationStyle(Enum):
    ABNT = "abnt"
    IEEE = "ieee"
    APA = "apa"


# Dataclasses v3.0
@dataclass
class VariableContext:
    """v3.0 - Contexto de variável."""
    name: str
    value: float
    unit: str
    description: str
    symbol: Optional[sp.Symbol] = None
    variable: Optional[Variable] = None
    
    def __post_init__(self):
        if SYMPY_AVAILABLE and sp is not None and self.symbol is None:
            self.symbol = sp.Symbol(self.name, real=True)


@dataclass
class EquationContext:
    """v3.0 - Contexto de equação."""
    name: str
    expression: str
    symbolic_expr: Optional[sp.Expr] = None
    result_value: Optional[float] = None
    result_unit: str = ""
    variables_used: List[str] = field(default_factory=list)


@dataclass
class ProcessingOptions:
    """v3.0 - Opções de processamento."""
    document_type: DocumentType = DocumentType.MEMORIAL
    render_mode: RenderMode = RenderMode.FULL
    citation_style: CitationStyle = CitationStyle.ABNT
    latex_inline_vars: bool = True
    show_equation_steps: bool = True


# Variable Registry v3.0
class VariableRegistry:
    """v3.0 - Registro global de variáveis."""
    
    def __init__(self):
        self.variables: Dict[str, VariableContext] = {}
        self.equations: Dict[str, EquationContext] = {}
        self.logger = logging.getLogger(__name__)
    
    def register(self, name: str, value: float, unit: str = '',
                 description: str = '') -> VariableContext:
        try:
            var = VariableFactory.create(name, value=value, unit=unit,
                                         description=description) if PYMEMORIAL_CORE_AVAILABLE else None
        except Exception as e:
            self.logger.warning(f"Erro ao criar Variable: {e}")
            var = None
        
        ctx = VariableContext(name=name, value=value, unit=unit,
                             description=description, variable=var)
        self.variables[name] = ctx
        return ctx
    
    def register_equation(self, name: str, expression: str, result_value: float,
                         result_unit: str = '', variables_used: List[str] = None) -> EquationContext:
        ctx = EquationContext(name=name, expression=expression,
                             result_value=result_value, result_unit=result_unit,
                             variables_used=variables_used or [])
        self.equations[name] = ctx
        self.register(name, result_value, result_unit, f"Calculado: {name}")
        return ctx
    
    def get(self, name: str) -> Optional[VariableContext]:
        return self.variables.get(name)
    
    def exists(self, name: str) -> bool:
        return name in self.variables
    
    def list_variables(self) -> List[str]:
        return list(self.variables.keys())


# LaTeX Renderer v3.0
class LaTeXRenderer:
    """v3.0 - Renderizador LaTeX."""
    
    @classmethod
    def variable_to_latex(cls, var_name: str) -> str:
        latex = GreekSymbols.to_latex(var_name) if hasattr(GreekSymbols, 'to_latex') else var_name
        if '_' in latex and '{' not in latex:
            parts = latex.split('_', 1)
            if len(parts) == 2 and len(parts[1]) > 1:
                latex = f"{parts[0]}_{{{parts[1]}}}"
        return latex
    
    @classmethod
    def expression_to_latex(cls, expr: str, registry: VariableRegistry = None) -> str:
        result = expr.replace('**', '^').replace('*', r' \times ')
        if registry:
            var_names = sorted(registry.list_variables(), key=len, reverse=True)
            for var_name in var_names:
                var_latex = cls.variable_to_latex(var_name)
                pattern = rf'\b{re.escape(var_name)}\b(?![}}$)])'
                result = re.sub(pattern, var_latex, result)
        return result
    
    @classmethod
    def sympy_to_latex(cls, expr: sp.Expr) -> str:
        if not SYMPY_AVAILABLE or sp is None:
            return str(expr)
        return sp.latex(expr)


# Equation Parser v3.0
class EquationParser:
    """v3.0 - Parser de equações com SymPy."""
    
    def __init__(self, registry: VariableRegistry):
        self.registry = registry
        self.latex_renderer = LaTeXRenderer()
        self.logger = logging.getLogger(__name__)
    
    def parse(self, equation_str: str, mode: RenderMode = RenderMode.FULL) -> str:
        try:
            if '=' not in equation_str:
                raise ValueError(f"Equação deve conter '=': {equation_str}")
            
            left, right = equation_str.split('=', 1)
            result_name = left.strip()
            expression = right.strip()
            
            if not SYMPY_AVAILABLE or sp is None:
                return self._render_simple(result_name, expression)
            
            sympy_expr = self._create_sympy_expr(expression)
            
            if mode == RenderMode.SYMBOLIC:
                return self._render_symbolic(result_name, sympy_expr)
            elif mode == RenderMode.NUMERIC:
                return self._render_numeric(result_name, sympy_expr, expression)
            else:  # FULL
                return self._render_full(result_name, sympy_expr, expression)
                
        except Exception as e:
            self.logger.error(f"Erro ao parsear equação: {e}")
            return f"**[ERRO: {equation_str}]**"
    
    def _create_sympy_expr(self, expression: str) -> sp.Expr:
        sympy_vars = {name: ctx.symbol for name, ctx in self.registry.variables.items()
                     if ctx.symbol is not None}
        return sp.sympify(expression, locals=sympy_vars)
    
    def _render_simple(self, result_name: str, expression: str) -> str:
        result_latex = self.latex_renderer.variable_to_latex(result_name)
        expr_latex = self.latex_renderer.expression_to_latex(expression, self.registry)
        return f"\n$${result_latex} = {expr_latex}$$\n"
    
    def _render_symbolic(self, result_name: str, expr: sp.Expr) -> str:
        result_latex = self.latex_renderer.variable_to_latex(result_name)
        expr_latex = self.latex_renderer.sympy_to_latex(expr)
        return f"\n$${result_latex} = {expr_latex}$$\n"
    
    def _render_numeric(self, result_name: str, expr: sp.Expr, expression: str) -> str:
        subs_dict = {ctx.symbol: ctx.value for ctx in self.registry.variables.values()
                    if ctx.symbol is not None}
        numeric_expr = expr.subs(subs_dict)
        result_value = float(numeric_expr)
        result_latex = self.latex_renderer.variable_to_latex(result_name)
        numeric_latex = sp.latex(numeric_expr)
        return f"\n$${result_latex} = {numeric_latex} = {result_value:.4g}$$\n"
    
    def _render_full(self, result_name: str, expr: sp.Expr, expression: str) -> str:
        result_latex = self.latex_renderer.variable_to_latex(result_name)
        expr_latex = self.latex_renderer.sympy_to_latex(expr)
        
        subs_dict = {ctx.symbol: ctx.value for ctx in self.registry.variables.values()
                    if ctx.symbol is not None}
        numeric_expr = expr.subs(subs_dict)
        result_value = float(numeric_expr)
        numeric_latex = sp.latex(numeric_expr)
        
        output = "\n**Cálculo:**\n\n"
        output += f"$${result_latex} = {expr_latex}$$\n\n"
        output += f"$${result_latex} = {numeric_latex} = {result_value:.4g}$$\n\n"
        output += f"<div class='result-box'>✓ **{result_latex} = {result_value:.4g}**</div>\n\n"
        
        self.registry.register_equation(result_name, expression, result_value)
        return output


# Smart Text Processor v3.0
class SmartTextProcessor:
    """v3.0 - Processor avançado com escrita natural."""
    
    # Patterns v3.0
    VALUE_DISPLAY_PATTERN = VALUEDISPLAYPATTERN
    FORMULA_DISPLAY_PATTERN = FORMULADISPLAYPATTERN
    EQUATION_BLOCK_PATTERN = EQUATIONBLOCKPATTERN
    
    def __init__(self, options: Optional[ProcessingOptions] = None):
        self.options = options or ProcessingOptions()
        self.registry = VariableRegistry()
        self.equation_parser = EquationParser(self.registry)
        self.latex_renderer = LaTeXRenderer()
        self.logger = logging.getLogger(__name__)
    
    def define_variables(self, variables: Dict[str, Tuple[float, str, str]]):
        for name, (value, unit, description) in variables.items():
            self.registry.register(name, value, unit, description)
    
    def process(self, text: str) -> str:
        text = self._process_equations(text)
        text = self._process_formula_displays(text)
        text = self._process_value_displays(text)
        if self.options.latex_inline_vars:
            text = self._process_inline_variables(text)
        return text
    
    def _process_equations(self, text: str) -> str:
        def replace_eq(match):
            mode_str = match.group(1) or 'full'
            equation_str = match.group(2).strip()
            try:
                mode = RenderMode[mode_str.upper()]
            except KeyError:
                mode = RenderMode.FULL
            return self.equation_parser.parse(equation_str, mode)
        return self.EQUATION_BLOCK_PATTERN.sub(replace_eq, text)
    
    def _process_formula_displays(self, text: str) -> str:
        def replace_formula(match):
            var_name = match.group(1)
            eq_ctx = self.registry.equations.get(var_name)
            if eq_ctx is None:
                return match.group(0)
            expr_latex = self.latex_renderer.expression_to_latex(eq_ctx.expression, self.registry)
            var_latex = self.latex_renderer.variable_to_latex(var_name)
            return f"${var_latex} = {expr_latex}$"
        return self.FORMULA_DISPLAY_PATTERN.sub(replace_formula, text)
    
    def _process_value_displays(self, text: str) -> str:
        def replace_value(match):
            var_name = match.group(1)
            ctx = self.registry.get(var_name)
            if ctx is None:
                return match.group(0)
            return f"**{ctx.value:.4g} {ctx.unit}**".strip()
        return self.VALUE_DISPLAY_PATTERN.sub(replace_value, text)
    
    def _process_inline_variables(self, text: str) -> str:
        """Converte variáveis inline para LaTeX (mantém dentro de classes de contexto)."""
        var_names = sorted(self.registry.list_variables(), key=len, reverse=True)
        
        for var_name in var_names:
            # FIX CRÍTICO: Escapar } como }} em f-string raw
            pattern = rf'\b{re.escape(var_name)}\b(?![}}$)])'
            var_latex = self.latex_renderer.variable_to_latex(var_name)
            
            # Função auxiliar para evitar substituir em contextos LaTeX
            def safe_replace(match):
                start = max(0, match.start() - 1)
                end = min(len(text), match.end() + 1)
                context = text[start:end]
                
                # Não substituir se estiver em math mode ou dentro de chaves
                if '$' in context or '{' in context or '}' in context:
                    return match.group(0)
                
                return f"${var_latex}$"
            
            text = re.sub(pattern, safe_replace, text)
        
        return text
    
    def get_registry(self) -> VariableRegistry:
        return self.registry


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # v2.0 (compatibilidade)
    'SmartTextEngine',
    'TextProcessor',
    'DetectedVariable',
    'get_engine',
    
    # v3.0 (novo)
    'SmartTextProcessor',
    'VariableRegistry',
    'EquationParser',
    'LaTeXRenderer',
    'ProcessingOptions',
    'DocumentType',
    'RenderMode',
    'CitationStyle',
    'VariableContext',
    'EquationContext',
    
    # Re-exports
    'PLACEHOLDER',
    'VALUEDISPLAYPATTERN',
    'FORMULADISPLAYPATTERN',
    'EQUATIONBLOCKPATTERN',
]
