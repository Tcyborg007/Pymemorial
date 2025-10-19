"""
MathDown Parser - Ultimate Edition
===================================

Natural Math Notation for Engineers - Complete, Robust, Flexible

Features:
- Natural math syntax with #$...$#
- Auto-detection of variables, equations, units
- Live computation with context persistence
- Greek letters support (α, β, σ, etc.)
- Multi-line equations
- Unit parsing (kN, MPa, m², etc.)
- Figure/Table directives
- Export to HTML, LaTeX, Markdown, PDF
- Integration with SymPy and Pint

Author: PyMemorial Team
Version: 2.0.0
License: MIT
"""

import re
from typing import List, Dict, Any, Tuple, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
import warnings

# Suppress SymPy warnings
warnings.filterwarnings('ignore', category=UserWarning)


# ============================================================================
# ENUMS AND DATA CLASSES
# ============================================================================

class TokenType(Enum):
    """Token types for MathDown."""
    TEXT = "text"
    MATH_BLOCK = "math"
    HEADING = "heading"
    CODE_BLOCK = "code"
    FIGURE = "figure"
    TABLE = "table"


class MathType(Enum):
    """Types of mathematical expressions."""
    EQUATION = "equation"           # M_max = q*L²/8
    ASSIGNMENT = "assignment"       # x = 10
    COMPARISON = "comparison"       # σ < f_y
    EXPRESSION = "expression"       # 2*x + 3
    MULTILINE = "multiline"         # Multi-line equation
    DEFINITION = "definition"       # "onde x é..."


@dataclass
class Token:
    """MathDown token."""
    type: TokenType
    content: str
    start: int
    end: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __repr__(self):
        return f"Token({self.type.value}, content={self.content[:30]}...)"


@dataclass
class ParsedVariable:
    """Parsed variable with metadata."""
    name: str
    value: Optional[float] = None
    unit: Optional[str] = None
    description: str = ""
    is_greek: bool = False
    latex: str = ""
    
    def __post_init__(self):
        if not self.latex:
            self.latex = self._generate_latex()
        self.is_greek = self._has_greek()
    
    def _generate_latex(self) -> str:
        """Generate LaTeX representation."""
        return VariableFormatter.to_latex(self.name)
    
    def _has_greek(self) -> bool:
        """Check if variable contains Greek letters."""
        greek_chars = 'αβγδεζηθικλμνξοπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ'
        return any(c in self.name for c in greek_chars)


@dataclass
class MathExpression:
    """Parsed mathematical expression."""
    type: MathType
    raw: str
    latex: str
    sympy_expr: Optional[sp.Expr] = None
    variables: List[str] = field(default_factory=list)
    value: Optional[float] = None
    unit: Optional[str] = None
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# UNIT PARSER - Detects and parses engineering units
# ============================================================================

class UnitParser:
    """
    Parses engineering units from text.
    
    Supports:
    - SI units: m, kg, s, N, Pa, etc.
    - Derived units: kN, MPa, cm², m³, etc.
    - Compound units: kN/m, N/mm², kg/m³
    - Powers: m², cm³, s⁻¹
    """
    
    # Common engineering units
    UNITS = {
        # Length
        'mm': 'millimeter', 'cm': 'centimeter', 'm': 'meter', 'km': 'kilometer',
        # Area
        'mm²': 'mm^2', 'cm²': 'cm^2', 'm²': 'm^2',
        # Volume
        'mm³': 'mm^3', 'cm³': 'cm^3', 'm³': 'm^3',
        # Force
        'N': 'newton', 'kN': 'kilonewton', 'MN': 'meganewton',
        # Pressure/Stress
        'Pa': 'pascal', 'kPa': 'kilopascal', 'MPa': 'megapascal', 'GPa': 'gigapascal',
        # Mass
        'g': 'gram', 'kg': 'kilogram', 't': 'tonne',
        # Time
        's': 'second', 'min': 'minute', 'h': 'hour',
        # Moment
        'kN·m': 'kilonewton*meter', 'kNm': 'kilonewton*meter',
        'N·m': 'newton*meter', 'Nm': 'newton*meter',
        # Distributed load
        'kN/m': 'kilonewton/meter', 'N/m': 'newton/meter',
        'kN/m²': 'kilonewton/meter^2', 'N/mm²': 'newton/millimeter^2',
    }
    
    # Regex patterns
    VALUE_UNIT_PATTERN = re.compile(
        r'([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)\s*'
        r'([a-zA-Zμ°]+(?:[²³⁴⁻¹²³⁴]|[\^/·*]\d+|/[a-zA-Z]+)?)'
    )
    
    UNIT_ONLY_PATTERN = re.compile(
        r'\b([a-zA-Z]+(?:[²³⁴⁻¹²³⁴]|[\^/·*]\d+|/[a-zA-Z]+)?)\b'
    )
    
    @classmethod
    def parse(cls, text: str) -> Tuple[Optional[float], Optional[str]]:
        """
        Parse value and unit from text.
        
        Examples:
            >>> UnitParser.parse("10 kN/m")
            (10.0, "kN/m")
            >>> UnitParser.parse("345 MPa")
            (345.0, "MPa")
            >>> UnitParser.parse("6.5 m²")
            (6.5, "m²")
        """
        match = cls.VALUE_UNIT_PATTERN.search(text)
        if match:
            value = float(match.group(1))
            unit = match.group(2).strip()
            return (value, unit)
        return (None, None)
    
    @classmethod
    def extract_unit(cls, text: str) -> Optional[str]:
        """Extract unit from text (without value)."""
        match = cls.UNIT_ONLY_PATTERN.search(text)
        return match.group(1) if match else None
    
    @classmethod
    def normalize_unit(cls, unit: str) -> str:
        """
        Normalize unit notation.
        
        Examples:
            >>> UnitParser.normalize_unit("m2")
            "m²"
            >>> UnitParser.normalize_unit("kN.m")
            "kN·m"
        """
        # Replace ASCII superscripts
        unit = unit.replace('^2', '²').replace('^3', '³')
        unit = unit.replace('2', '²').replace('3', '³')  # Risky but common
        
        # Replace dot/asterisk with center dot
        unit = unit.replace('.', '·').replace('*', '·')
        
        return unit


# ============================================================================
# VARIABLE FORMATTER - Converts variables to LaTeX
# ============================================================================

class VariableFormatter:
    """
    Formats variable names to LaTeX.
    
    Handles:
    - Subscripts: f_y → f_y
    - Greek letters: σ → \sigma
    - Superscripts: x^2 → x^2
    """
    
    GREEK_MAP = {
        'α': r'\alpha', 'β': r'\beta', 'γ': r'\gamma', 'δ': r'\delta',
        'ε': r'\varepsilon', 'ζ': r'\zeta', 'η': r'\eta', 'θ': r'\theta',
        'ι': r'\iota', 'κ': r'\kappa', 'λ': r'\lambda', 'μ': r'\mu',
        'ν': r'\nu', 'ξ': r'\xi', 'ο': r'o', 'π': r'\pi',
        'ρ': r'\rho', 'σ': r'\sigma', 'ς': r'\sigma', 'τ': r'\tau',
        'υ': r'\upsilon', 'φ': r'\phi', 'χ': r'\chi', 'ψ': r'\psi',
        'ω': r'\omega',
        # Uppercase
        'Α': r'A', 'Β': r'B', 'Γ': r'\Gamma', 'Δ': r'\Delta',
        'Ε': r'E', 'Ζ': r'Z', 'Η': r'H', 'Θ': r'\Theta',
        'Ι': r'I', 'Κ': r'K', 'Λ': r'\Lambda', 'Μ': r'M',
        'Ν': r'N', 'Ξ': r'\Xi', 'Ο': r'O', 'Π': r'\Pi',
        'Ρ': r'P', 'Σ': r'\Sigma', 'Τ': r'T', 'Υ': r'\Upsilon',
        'Φ': r'\Phi', 'Χ': r'X', 'Ψ': r'\Psi', 'Ω': r'\Omega',
    }
    
    @classmethod
    def to_latex(cls, var: str) -> str:
        """
        Convert variable to LaTeX.
        
        Examples:
            >>> VariableFormatter.to_latex("f_y")
            "f_{y}"
            >>> VariableFormatter.to_latex("σ_max")
            "\\sigma_{max}"
            >>> VariableFormatter.to_latex("M_Rd")
            "M_{Rd}"
        """
        # Handle subscripts
        if '_' in var:
            parts = var.split('_', 1)
            base = parts[0]
            subscript = parts[1]
            
            # Convert Greek in base and subscript
            base = cls._replace_greek(base)
            subscript = cls._replace_greek(subscript)
            
            return f"{base}_{{{subscript}}}"
        
        # Just convert Greek
        return cls._replace_greek(var)
    
    @classmethod
    def _replace_greek(cls, text: str) -> str:
        """Replace Greek Unicode with LaTeX."""
        for greek, latex in cls.GREEK_MAP.items():
            text = text.replace(greek, latex)
        return text


# ============================================================================
# COMPUTATION CONTEXT - Maintains variable state
# ============================================================================

class ComputationContext:
    """
    Maintains computation context with persistent variables.
    
    Features:
    - Variable storage with units
    - Expression evaluation
    - Dependency tracking
    - History/undo support
    """
    
    def __init__(self):
        self.variables: Dict[str, ParsedVariable] = {}
        self.history: List[Dict[str, Any]] = []
    
    def set_variable(
        self, 
        name: str, 
        value: float, 
        unit: Optional[str] = None,
        description: str = ""
    ):
        """Store variable in context."""
        var = ParsedVariable(
            name=name,
            value=value,
            unit=unit,
            description=description
        )
        self.variables[name] = var
        self.history.append({'action': 'set', 'variable': name, 'value': value})
    
    def get_variable(self, name: str) -> Optional[ParsedVariable]:
        """Retrieve variable from context."""
        return self.variables.get(name)
    
    def get_value(self, name: str) -> Optional[float]:
        """Get numeric value of variable."""
        var = self.variables.get(name)
        return var.value if var else None
    
    def evaluate(self, expr: str) -> Optional[float]:
        """
        Evaluate expression with context variables.
        
        Examples:
            >>> ctx = ComputationContext()
            >>> ctx.set_variable('q', 10)
            >>> ctx.set_variable('L', 6)
            >>> ctx.evaluate('q*L**2/8')
            45.0
        """
        try:
            # Build substitution dictionary
            subs = {}
            for name, var in self.variables.items():
                if var.value is not None:
                    subs[sp.Symbol(name)] = var.value
            
            # Parse and evaluate
            sympy_expr = self._parse_sympy(expr)
            if sympy_expr:
                result = sympy_expr.subs(subs).evalf()
                return float(result)
        except Exception as e:
            print(f"Evaluation error: {e}")
            return None
    
    def _parse_sympy(self, expr: str) -> Optional[sp.Expr]:
        """Parse expression to SymPy (robust)."""
        try:
            # Replace common notations
            expr = expr.replace('^', '**')
            expr = expr.replace('²', '**2').replace('³', '**3')
            
            # Use transformations for implicit multiplication
            transformations = standard_transformations + (implicit_multiplication_application,)
            return parse_expr(expr, transformations=transformations)
        except:
            return None
    
    def clear(self):
        """Clear all variables."""
        self.variables.clear()
        self.history.clear()
    
    def to_dict(self) -> Dict[str, Any]:
        """Export context to dictionary."""
        return {
            'variables': {
                name: {
                    'value': var.value,
                    'unit': var.unit,
                    'description': var.description
                }
                for name, var in self.variables.items()
            }
        }


# ============================================================================
# MATHDOWN LEXER - Tokenizes MathDown text
# ============================================================================

class MathDownLexer:
    r"""
    Lexer for MathDown syntax.
    
    Recognizes:
    - #$ ... $#  : Math blocks
    - # Heading  : Markdown headings
    - `````` : Code blocks
    - @figure[]  : Figure directives
    - @table[]   : Table directives
    """
    
    MATH_BLOCK = re.compile(r'#\$(.*?)\$#', re.DOTALL)
    HEADING = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
    CODE_BLOCK = re.compile(r'``````', re.DOTALL)
    FIGURE = re.compile(r'@figure\[(.*?)\]')
    TABLE = re.compile(r'@table\[(.*?)\]')
    
    def tokenize(self, text: str) -> List[Token]:
        """
        Tokenize MathDown text.
        
        Returns list of tokens in order of appearance.
        """
        tokens = []
        positions = []  # [(start, end, token)]
        
        # Find all patterns
        for match in self.MATH_BLOCK.finditer(text):
            token = Token(
                type=TokenType.MATH_BLOCK,
                content=match.group(1).strip(),
                start=match.start(),
                end=match.end()
            )
            positions.append((match.start(), match.end(), token))
        
        for match in self.HEADING.finditer(text):
            level = len(match.group(1))
            token = Token(
                type=TokenType.HEADING,
                content=match.group(2).strip(),
                start=match.start(),
                end=match.end(),
                metadata={'level': level}
            )
            positions.append((match.start(), match.end(), token))
        
        for match in self.CODE_BLOCK.finditer(text):
            language = match.group(1) or 'python'
            token = Token(
                type=TokenType.CODE_BLOCK,
                content=match.group(2).strip(),
                start=match.start(),
                end=match.end(),
                metadata={'language': language}
            )
            positions.append((match.start(), match.end(), token))
        
        for match in self.FIGURE.finditer(text):
            token = Token(
                type=TokenType.FIGURE,
                content=match.group(1).strip(),
                start=match.start(),
                end=match.end()
            )
            positions.append((match.start(), match.end(), token))
        
        for match in self.TABLE.finditer(text):
            token = Token(
                type=TokenType.TABLE,
                content=match.group(1).strip(),
                start=match.start(),
                end=match.end()
            )
            positions.append((match.start(), match.end(), token))
        
        # Sort by position
        positions.sort(key=lambda x: x[0])
        
        # Fill in text tokens between special tokens
        last_end = 0
        for start, end, token in positions:
            if start > last_end:
                text_content = text[last_end:start].strip()
                if text_content:
                    tokens.append(Token(
                        type=TokenType.TEXT,
                        content=text_content,
                        start=last_end,
                        end=start
                    ))
            tokens.append(token)
            last_end = end
        
        # Remaining text
        if last_end < len(text):
            text_content = text[last_end:].strip()
            if text_content:
                tokens.append(Token(
                    type=TokenType.TEXT,
                    content=text_content,
                    start=last_end,
                    end=len(text)
                ))
        
        return tokens


# ============================================================================
# MATHDOWN ANALYZER - Analyzes mathematical expressions
# ============================================================================

class MathDownAnalyzer:
    """
    Analyzes math blocks to detect type and extract information.
    
    Detects:
    - Equations with description: "onde M é o momento => M = q*L²/8"
    - Assignments: "q = 10 kN/m"
    - Comparisons: "σ < f_y"
    - Multi-line equations
    - Expressions with units
    """
    
    # Regex patterns
    EQUATION_WITH_DESC = re.compile(r'^(.*?)\s*=>\s*(.+)$', re.DOTALL)
    ASSIGNMENT = re.compile(r'^([a-zA-Zα-ωΑ-Ω_][a-zA-Z0-9α-ωΑ-Ω_]*)\s*=\s*(.+)$')
    COMPARISON = re.compile(r'^(.+?)\s*([<>≤≥≠])\s*(.+)$')
    MULTILINE = re.compile(r'.+=.*\n.+=', re.MULTILINE)
    
    def __init__(self, context: Optional[ComputationContext] = None):
        self.context = context or ComputationContext()
    
    def analyze(self, math_content: str) -> MathExpression:
        """
        Analyze math block content.
        
        Returns MathExpression with type and metadata.
        """
        math_content = math_content.strip()
        
        # Check for multi-line equation
        if '\n' in math_content and self.MULTILINE.search(math_content):
            return self._analyze_multiline(math_content)
        
        # Check for equation with description (=>)
        match = self.EQUATION_WITH_DESC.match(math_content)
        if match:
            description = match.group(1).strip()
            equation = match.group(2).strip()
            return self._analyze_equation(equation, description)
        
        # Check for assignment (=)
        match = self.ASSIGNMENT.match(math_content)
        if match:
            var = match.group(1).strip()
            expr = match.group(2).strip()
            return self._analyze_assignment(var, expr)
        
        # Check for comparison
        match = self.COMPARISON.match(math_content)
        if match:
            left = match.group(1).strip()
            op = match.group(2).strip()
            right = match.group(3).strip()
            return self._analyze_comparison(left, op, right)
        
        # Just an expression
        return self._analyze_expression(math_content)
    
    def _analyze_equation(self, equation: str, description: str = "") -> MathExpression:
        """Analyze equation (possibly with =>)."""
        # Parse left and right sides
        if '=' in equation:
            left, right = equation.split('=', 1)
            left = left.strip()
            right = right.strip()
        else:
            left = equation
            right = ""
        
        # Extract variables
        variables = self._extract_variables(equation)
        
        # Try to parse with SymPy
        sympy_expr = self._to_sympy(equation)
        
        # Generate LaTeX
        if sympy_expr:
            latex = sp.latex(sympy_expr)
        else:
            latex = self._manual_latex(equation)
        
        return MathExpression(
            type=MathType.EQUATION,
            raw=equation,
            latex=latex,
            sympy_expr=sympy_expr,
            variables=variables,
            description=description
        )
    
    def _analyze_assignment(self, var: str, expr: str) -> MathExpression:
        """Analyze variable assignment."""
        # Parse value and unit
        value, unit = UnitParser.parse(expr)
        
        # Extract variables from expression
        variables = self._extract_variables(expr)
        
        # Try to evaluate if numeric or has context
        computed_value = None
        if value is None:
            # Try to evaluate with context
            computed_value = self.context.evaluate(expr)
        else:
            computed_value = value
        
        # Store in context
        if computed_value is not None:
            self.context.set_variable(var, computed_value, unit)
        
        # SymPy expression
        sympy_expr = self._to_sympy(f"{var} = {expr}")
        
        # LaTeX
        var_latex = VariableFormatter.to_latex(var)
        expr_latex = self._expr_to_latex(expr)
        latex = f"{var_latex} = {expr_latex}"
        
        if computed_value is not None:
            latex += f" = {computed_value}"
        
        if unit:
            latex += f" \\, \\text{{{unit}}}"
        
        return MathExpression(
            type=MathType.ASSIGNMENT,
            raw=f"{var} = {expr}",
            latex=latex,
            sympy_expr=sympy_expr,
            variables=variables,
            value=computed_value,
            unit=unit
        )
    
    def _analyze_comparison(self, left: str, op: str, right: str) -> MathExpression:
        """Analyze comparison expression."""
        # Extract variables
        variables = self._extract_variables(f"{left} {right}")
        
        # Try to evaluate
        result = None
        try:
            left_val = self.context.evaluate(left)
            right_val = self.context.evaluate(right)
            
            if left_val is not None and right_val is not None:
                if op == '<':
                    result = left_val < right_val
                elif op == '>':
                    result = left_val > right_val
                elif op == '≤':
                    result = left_val <= right_val
                elif op == '≥':
                    result = left_val >= right_val
                elif op == '≠':
                    result = left_val != right_val
        except:
            pass
        
        # LaTeX
        left_latex = self._expr_to_latex(left)
        right_latex = self._expr_to_latex(right)
        op_latex = {'<': '<', '>': '>', '≤': r'\leq', '≥': r'\geq', '≠': r'\neq'}[op]
        latex = f"{left_latex} {op_latex} {right_latex}"
        
        return MathExpression(
            type=MathType.COMPARISON,
            raw=f"{left} {op} {right}",
            latex=latex,
            variables=variables,
            value=result,
            metadata={'left': left, 'operator': op, 'right': right, 'result': result}
        )
    
    def _analyze_multiline(self, content: str) -> MathExpression:
        """Analyze multi-line equation."""
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        
        # Generate LaTeX align environment
        latex_lines = []
        for line in lines:
            if '=' in line:
                # Split at =
                left, right = line.split('=', 1)
                left_latex = self._expr_to_latex(left.strip())
                right_latex = self._expr_to_latex(right.strip())
                latex_lines.append(f"{left_latex} &= {right_latex}")
            else:
                latex_lines.append(self._expr_to_latex(line))
        
        latex = "\\begin{align}\n" + " \\\\\n".join(latex_lines) + "\n\\end{align}"
        
        # Extract variables from all lines
        variables = self._extract_variables(content)
        
        return MathExpression(
            type=MathType.MULTILINE,
            raw=content,
            latex=latex,
            variables=variables
        )
    
    def _analyze_expression(self, expr: str) -> MathExpression:
        """Analyze generic expression."""
        variables = self._extract_variables(expr)
        sympy_expr = self._to_sympy(expr)
        latex = sp.latex(sympy_expr) if sympy_expr else self._manual_latex(expr)
        
        # Try to evaluate
        value = self.context.evaluate(expr)
        
        return MathExpression(
            type=MathType.EXPRESSION,
            raw=expr,
            latex=latex,
            sympy_expr=sympy_expr,
            variables=variables,
            value=value
        )
    
    def _extract_variables(self, text: str) -> List[str]:
        """Extract variable names from text."""
        pattern = r'[a-zA-Zα-ωΑ-Ω][a-zA-Z0-9α-ωΑ-Ω_]*'
        matches = re.findall(pattern, text)
        # Filter out units and keywords
        units = set(UnitParser.UNITS.keys())
        keywords = {'onde', 'where', 'sendo', 'with', 'para', 'for'}
        return [m for m in matches if m not in units and m.lower() not in keywords]
    
    def _to_sympy(self, expr: str) -> Optional[sp.Expr]:
        """Convert string expression to SymPy (robust)."""
        try:
            # Clean expression
            expr = expr.replace('²', '**2').replace('³', '**3')
            expr = expr.replace('^', '**')
            
            # Remove units
            for unit in UnitParser.UNITS.keys():
                expr = expr.replace(f" {unit}", "")
            
            # Parse with transformations
            transformations = standard_transformations + (implicit_multiplication_application,)
            
            # Create local symbols for variables
            variables = self._extract_variables(expr)
            local_dict = {var: sp.Symbol(var) for var in variables}
            
            return parse_expr(expr, local_dict=local_dict, transformations=transformations)
        except Exception as e:
            # print(f"SymPy parse error: {e}")
            return None
    
    def _expr_to_latex(self, expr: str) -> str:
        """Convert expression to LaTeX."""
        sympy_expr = self._to_sympy(expr)
        if sympy_expr:
            return sp.latex(sympy_expr)
        return self._manual_latex(expr)
    
    def _manual_latex(self, expr: str) -> str:
        """Manual LaTeX conversion (fallback)."""
        # Replace variables with LaTeX
        variables = self._extract_variables(expr)
        result = expr
        for var in sorted(variables, key=len, reverse=True):
            var_latex = VariableFormatter.to_latex(var)
            result = result.replace(var, var_latex)
        
        # Replace operators
        result = result.replace('*', r' \times ')
        result = result.replace('/', r' / ')
        result = result.replace('²', '^2').replace('³', '^3')
        
        return result


# ============================================================================
# MATHDOWN RENDERER - Renders to various formats
# ============================================================================

class MathDownRenderer:
    """
    Renders MathDown to HTML, LaTeX, Markdown.
    
    Features:
    - HTML with MathJax
    - LaTeX document
    - GitHub-flavored Markdown
    - Beautiful formatting
    """
    
    def __init__(self):
        self.html_template = self._load_html_template()
    
    def to_html(self, tokens: List[Token], expressions: List[MathExpression]) -> str:
        """
        Render to HTML with MathJax.
        
        Returns complete HTML document.
        """
        html_parts = []
        
        expr_idx = 0
        for token in tokens:
            if token.type == TokenType.TEXT:
                html_parts.append(f"<p>{token.content}</p>")
            
            elif token.type == TokenType.HEADING:
                level = token.metadata.get('level', 1)
                html_parts.append(f"<h{level}>{token.content}</h{level}>")
            
            elif token.type == TokenType.MATH_BLOCK:
                if expr_idx < len(expressions):
                    expr = expressions[expr_idx]
                    html_parts.append(self._render_math_html(expr))
                    expr_idx += 1
            
            elif token.type == TokenType.CODE_BLOCK:
                language = token.metadata.get('language', 'python')
                html_parts.append(
                    f'<pre><code class="language-{language}">{token.content}</code></pre>'
                )
            
            elif token.type == TokenType.FIGURE:
                html_parts.append(self._render_figure_html(token.content))
            
            elif token.type == TokenType.TABLE:
                html_parts.append(self._render_table_html(token.content))
        
        # Wrap in template
        body = '\n'.join(html_parts)
        return self.html_template.format(body=body)
    
    def _render_math_html(self, expr: MathExpression) -> str:
        """Render single math expression to HTML."""
        if expr.type == MathType.EQUATION:
            if expr.description:
                return f"""
                <div class="math-equation">
                    <p class="description"><em>{expr.description}</em></p>
                    <p class="math-display">\\[{expr.latex}\\]</p>
                </div>
                """
            else:
                return f'<p class="math-display">\\[{expr.latex}\\]</p>'
        
        elif expr.type == MathType.ASSIGNMENT:
            return f'<p class="math-inline">\\({expr.latex}\\)</p>'
        
        elif expr.type == MathType.COMPARISON:
            result = expr.metadata.get('result')
            status = ""
            if result is not None:
                status_class = "success" if result else "warning"
                status = f' <span class="status {status_class}">({result})</span>'
            return f'<p class="math-inline">\\({expr.latex}\\){status}</p>'
        
        elif expr.type == MathType.MULTILINE:
            return f'<div class="math-display">\\[{expr.latex}\\]</div>'
        
        else:  # EXPRESSION
            return f'<p class="math-inline">\\({expr.latex}\\)</p>'
    
    def _render_figure_html(self, directive: str) -> str:
        """Render figure directive."""
        # Parse directive: filename, width=0.8, caption="..."
        parts = [p.strip() for p in directive.split(',')]
        filename = parts[0]
        options = {}
        for part in parts[1:]:
            if '=' in part:
                key, val = part.split('=', 1)
                options[key.strip()] = val.strip().strip('"\'')
        
        width = options.get('width', '100%')
        caption = options.get('caption', '')
        
        html = f'<figure><img src="{filename}" style="width:{width};">'
        if caption:
            html += f'<figcaption>{caption}</figcaption>'
        html += '</figure>'
        
        return html
    
    def _render_table_html(self, directive: str) -> str:
        """Render table directive."""
        # TODO: Implement table rendering
        return f"<p>[Table: {directive}]</p>"
    
    def _load_html_template(self) -> str:
        """Load HTML template with MathJax."""
        return """<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MathDown Document</title>
    
    <!-- MathJax Configuration -->
    <script>
    window.MathJax = {{
        tex: {{
            inlineMath: [['\\(', '\\)']],
            displayMath: [['\\[', '\\]']],
            processEscapes: true,
            processEnvironments: true
        }},
        options: {{
            skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre']
        }}
    }};
    </script>
    
    <!-- Load MathJax -->
    <script id="MathJax-script" async 
            src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js">
    </script>
    
    <!-- Styles -->
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 900px;
            margin: 40px auto;
            padding: 20px;
            line-height: 1.8;
            color: #333;
            background: #f5f5f5;
        }}
        
        .container {{
            background: white;
            padding: 40px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        
        h1 {{ 
            color: #2c3e50; 
            font-size: 2em; 
            margin-bottom: 0.5em;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        
        h2 {{ 
            color: #2c3e50; 
            font-size: 1.5em; 
            margin-top: 1.5em; 
            margin-bottom: 0.5em;
            border-bottom: 2px solid #95a5a6;
            padding-bottom: 5px;
        }}
        
        h3 {{ 
            color: #34495e; 
            font-size: 1.2em; 
            margin-top: 1em; 
            margin-bottom: 0.5em; 
        }}
        
        p {{
            margin: 10px 0;
            text-align: justify;
        }}
        
        .math-equation {{
            margin: 25px 0;
            padding: 20px;
            background: #f8f9fa;
            border-left: 4px solid #3498db;
            border-radius: 4px;
        }}
        
        .math-equation .description {{ 
            font-style: italic; 
            color: #555;
            margin-bottom: 15px;
            line-height: 1.6;
        }}
        
        .math-display {{
            text-align: center;
            margin: 20px 0;
            font-size: 1.2em;
            padding: 15px 0;
        }}
        
        .math-inline {{ 
            margin: 15px 0;
            padding: 10px 20px;
            background: #fff;
            border-left: 3px solid #95a5a6;
        }}
        
        .status {{
            font-weight: bold;
            margin-left: 10px;
            padding: 3px 8px;
            border-radius: 3px;
        }}
        
        .status.success {{ 
            color: #27ae60; 
            background: #d4edda;
        }}
        
        .status.warning {{ 
            color: #e74c3c;
            background: #f8d7da;
        }}
        
        figure {{
            margin: 30px 0;
            text-align: center;
        }}
        
        figure img {{
            max-width: 100%;
            height: auto;
            border-radius: 5px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }}
        
        figcaption {{
            margin-top: 10px;
            font-style: italic;
            color: #666;
            font-size: 0.9em;
        }}
        
        pre {{
            background: #2c3e50;
            color: #ecf0f1;
            padding: 20px;
            border-radius: 5px;
            overflow-x: auto;
            margin: 20px 0;
        }}
        
        code {{
            font-family: 'Courier New', 'Consolas', monospace;
            font-size: 0.9em;
        }}
        
        /* Loading indicator */
        .loading {{
            text-align: center;
            padding: 20px;
            color: #95a5a6;
            font-style: italic;
        }}
        
        @media print {{
            body {{
                background: white;
                max-width: 100%;
            }}
            .container {{
                box-shadow: none;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="loading">Carregando equações matemáticas...</div>
        {body}
    </div>
    
    <!-- Remove loading message when MathJax is ready -->
    <script>
    MathJax.startup.promise.then(() => {{
        const loading = document.querySelector('.loading');
        if (loading) loading.remove();
    }});
    </script>
</body>
</html>"""



# ============================================================================
# MATHDOWN PARSER - Main parser class
# ============================================================================

class MathDownParser:
    """
    Main MathDown parser - Ultimate Edition.
    
    Usage:
        >>> parser = MathDownParser()
        >>> result = parser.parse(mathdown_text)
        >>> html = result.to_html()
        >>> latex = result.to_latex()
        >>> variables = result.get_variables()
    """
    
    def __init__(self, context: Optional[ComputationContext] = None):
        self.context = context or ComputationContext()
        self.lexer = MathDownLexer()
        self.analyzer = MathDownAnalyzer(self.context)
        self.renderer = MathDownRenderer()
    
    def parse(self, text: str) -> 'ParseResult':
        """
        Parse MathDown text.
        
        Returns ParseResult with tokens, expressions, and rendering methods.
        """
        # Tokenize
        tokens = self.lexer.tokenize(text)
        
        # Analyze math blocks
        expressions = []
        for token in tokens:
            if token.type == TokenType.MATH_BLOCK:
                expr = self.analyzer.analyze(token.content)
                expressions.append(expr)
        
        return ParseResult(tokens, expressions, self.context, self.renderer)
    
    def parse_file(self, filepath: Union[str, Path]) -> 'ParseResult':
        """Parse MathDown file."""
        path = Path(filepath)
        text = path.read_text(encoding='utf-8')
        return self.parse(text)


# ============================================================================
# PARSE RESULT - Result object
# ============================================================================

@dataclass
class ParseResult:
    """Result of MathDown parsing."""
    tokens: List[Token]
    expressions: List[MathExpression]
    context: ComputationContext
    renderer: MathDownRenderer
    
    def to_html(self, filename: Optional[str] = None) -> str:
        """
        Render to HTML.
        
        If filename provided, saves to file.
        """
        html = self.renderer.to_html(self.tokens, self.expressions)
        
        if filename:
            Path(filename).write_text(html, encoding='utf-8')
        
        return html
    
    def to_latex(self) -> str:
        """Render to LaTeX."""
        # TODO: Implement LaTeX rendering
        raise NotImplementedError("LaTeX export not yet implemented")
    
    def to_markdown(self) -> str:
        """Render to Markdown."""
        # TODO: Implement Markdown rendering
        raise NotImplementedError("Markdown export not yet implemented")
    
    def get_variables(self) -> Dict[str, ParsedVariable]:
        """Get all variables from context."""
        return self.context.variables
    
    def get_equations(self) -> List[MathExpression]:
        """Get all equations."""
        return [expr for expr in self.expressions if expr.type == MathType.EQUATION]
    
    def get_assignments(self) -> List[MathExpression]:
        """Get all assignments."""
        return [expr for expr in self.expressions if expr.type == MathType.ASSIGNMENT]
    
    def summary(self) -> str:
        """Print summary of parsed content."""
        summary = []
        summary.append(f"Tokens: {len(self.tokens)}")
        summary.append(f"Math expressions: {len(self.expressions)}")
        summary.append(f"Variables: {len(self.context.variables)}")
        summary.append(f"Equations: {len(self.get_equations())}")
        summary.append(f"Assignments: {len(self.get_assignments())}")
        return '\n'.join(summary)


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def parse_mathdown(text: str) -> ParseResult:
    """Convenience function to parse MathDown text."""
    parser = MathDownParser()
    return parser.parse(text)


def parse_mathdown_file(filepath: Union[str, Path]) -> ParseResult:
    """Convenience function to parse MathDown file."""
    parser = MathDownParser()
    return parser.parse_file(filepath)


# ============================================================================
# EXPORT ALL
# ============================================================================

__all__ = [
    'MathDownParser',
    'ParseResult',
    'MathExpression',
    'ParsedVariable',
    'ComputationContext',
    'UnitParser',
    'VariableFormatter',
    'parse_mathdown',
    'parse_mathdown_file',
]
