"""
PyMemorial MathDown - Natural Math Notation for Engineers
==========================================================

Revolutionary markup language for engineering calculations.

Features:
- Natural text with embedded math: #$ ... $#
- Auto-detection of variables, units, Greek letters
- Live computation with context persistence
- Export to HTML, LaTeX, Markdown, PDF
- Beautiful rendering with MathJax

Quick Start:
    >>> from pymemorial.mathdown import MathDownParser
    >>> parser = MathDownParser()
    >>> result = parser.parse(mathdown_text)
    >>> result.to_html("output.html")

Examples:
    Simple equation:
        #$ M_max = q*L²/8 $#
    
    With description:
        #$ sabemos que o momento é => M = q*L²/8 $#
    
    Assignment with unit:
        #$ f_y = 345 MPa $#
    
    Comparison:
        #$ σ < f_y $#

Author: PyMemorial Team
Version: 2.0.0
License: MIT
"""

from .math_parser import (
    # Main parser
    MathDownParser,
    ParseResult,
    
    # Data structures
    MathExpression,
    ParsedVariable,
    Token,
    TokenType,
    MathType,
    
    # Components
    ComputationContext,
    MathDownLexer,
    MathDownAnalyzer,
    MathDownRenderer,
    
    # Utilities
    UnitParser,
    VariableFormatter,
    
    # Convenience functions
    parse_mathdown,
    parse_mathdown_file,
)

__version__ = "2.0.0"

__all__ = [
    # Main API
    "MathDownParser",
    "ParseResult",
    "parse_mathdown",
    "parse_mathdown_file",
    
    # Data structures
    "MathExpression",
    "ParsedVariable",
    "Token",
    "TokenType",
    "MathType",
    
    # Components (advanced usage)
    "ComputationContext",
    "MathDownLexer",
    "MathDownAnalyzer",
    "MathDownRenderer",
    
    # Utilities
    "UnitParser",
    "VariableFormatter",
]


# Quick start guide
QUICK_START = """
PyMemorial MathDown - Quick Start
==================================

1. Install:
   pip install pymemorial

2. Create a .md file with MathDown:
   
   # My Calculation
   
   #$ f_y = 345 MPa $#
   #$ A = 100 cm² $#
   #$ N = f_y * A $#

3. Parse and render:
   
   from pymemorial.mathdown import parse_mathdown
   
   result = parse_mathdown(text)
   result.to_html("output.html")

4. Open output.html in browser!

For more examples:
   python -m pymemorial.mathdown.examples
"""


def print_quick_start():
    """Print quick start guide."""
    print(QUICK_START)


# Module-level convenience
def parse(text: str) -> ParseResult:
    """
    Quick parse function.
    
    Args:
        text: MathDown text to parse
    
    Returns:
        ParseResult object
    
    Example:
        >>> from pymemorial import mathdown
        >>> result = mathdown.parse("#$ x = 10 $#")
        >>> print(result.context.variables['x'].value)
        10.0
    """
    return parse_mathdown(text)


# Version check
def check_version():
    """Check if all dependencies are installed."""
    try:
        import sympy
        import re
        from pathlib import Path
        print(f"✓ PyMemorial MathDown v{__version__} ready!")
        print(f"  SymPy: {sympy.__version__}")
        return True
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("  Install with: pip install pymemorial[mathdown]")
        return False


# Auto-check on import (only in interactive mode)
import sys
if hasattr(sys, 'ps1'):  # Interactive mode
    check_version()
