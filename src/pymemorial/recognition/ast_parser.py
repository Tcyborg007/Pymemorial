# src/pymemorial/recognition/ast_parser.py

"""
Parser AST para código Python natural com conversão LaTeX.

Inspirado no Handcalcs mas com capacidades expandidas:
- Parsing de atribuições e expressões
- Extração automática de variáveis
- Conversão LaTeX inteligente (frações, potências, símbolos gregos)
- Captura de comentários inline
- Preservação de linha/coluna para debugging

Examples:
    >>> parser = PyMemorialASTParser()
    >>> 
    >>> # Parse simples
    >>> assign = parser.parse_assignment("M = q * L**2 / 8")
    >>> print(assign.lhs, assign.rhs_symbolic)
    M q * L**2 / 8
    >>> 
    >>> # Conversão LaTeX
    >>> latex = parser.to_latex("q * L**2 / 8")
    >>> print(latex)
    \\frac{q \\cdot L^{2}}{8}
    >>> 
    >>> # Parse de bloco com comentários
    >>> code = '''
    ... gamma_f = 1.4  # Coef. majoração
    ... M_k = 150      # kN.m
    ... M_d = gamma_f * M_k
    ... '''
    >>> assigns = parser.parse_code_block(code)
    >>> for a in assigns:
    ...     print(f"{a.lhs} = {a.rhs_symbolic}  # {a.comment or ''}")
    gamma_f = 1.4  # Coef. majoração
    M_k = 150  # kN.m
    M_d = gamma_f * M_k  # 
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from pymemorial.core.config import get_config
from pymemorial.recognition.greek import ASCII_TO_GREEK  # ✅ Usar existente correto!

__all__ = [
    'ParsedAssignment',
    'PyMemorialASTParser',
    'ASTParserError'
]


# =============================================================================
# EXCEÇÕES
# =============================================================================

class ASTParserError(Exception):
    """Erro de parsing AST."""
    pass


# =============================================================================
# ESTRUTURAS DE DADOS
# =============================================================================

@dataclass
class ParsedAssignment:
    """
    Representa uma atribuição parseada.
    
    Attributes:
        lhs: Lado esquerdo (nome da variável)
        rhs_symbolic: Expressão simbólica do lado direito
        comment: Comentário inline (se existir)
        lineno: Número da linha no código fonte
        col_offset: Offset da coluna
        context: Contexto de variáveis conhecidas
    
    Examples:
        >>> a = ParsedAssignment(
        ...     lhs="M",
        ...     rhs_symbolic="q * L**2 / 8",
        ...     comment="kN.m",
        ...     lineno=3,
        ...     col_offset=0,
        ...     context={'q': 15, 'L': 6}
        ... )
    """
    lhs: str
    rhs_symbolic: str
    comment: Optional[str] = None
    lineno: int = 0
    col_offset: int = 0
    context: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# PARSER PRINCIPAL
# =============================================================================

class PyMemorialASTParser:
    """
    Parser AST para código Python com conversão LaTeX.
    
    DIFERENCIAL vs Handcalcs:
    - Suporte completo a símbolos gregos (gamma_s → γₛ ou \\gamma_{s})
    - Detecção automática de subscritos (M_d → M_{d})
    - Conversão LaTeX inteligente (frações, potências)
    - Integração com config.py (precisão, estilo)
    
    Attributes:
        config: Configuração global do PyMemorial
    
    Examples:
        >>> parser = PyMemorialASTParser()
        >>> result = parser.parse_assignment("M = 100")
        >>> print(result.lhs)
        M
    """
    
    def __init__(self):
        """Inicializa parser com config global."""
        self.config = get_config()
    
    def parse_assignment(
        self, 
        line: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> ParsedAssignment:
        """
        Parse uma única linha de atribuição.
        
        Args:
            line: Linha de código Python (ex: "M = q * L**2 / 8  # kN.m")
            context: Contexto de variáveis conhecidas (opcional)
        
        Returns:
            ParsedAssignment com informações extraídas
        
        Raises:
            ASTParserError: Se linha não for atribuição válida
        
        Examples:
            >>> parser = PyMemorialASTParser()
            >>> a = parser.parse_assignment("M = 100")
            >>> a.lhs
            'M'
            >>> a.rhs_symbolic
            '100'
            
            >>> # Com comentário
            >>> a = parser.parse_assignment("L = 6.0  # m")
            >>> a.comment
            'm'
        """
        if context is None:
            context = {}
        
        # Separar código de comentário
        code_part, comment_part = self._split_comment(line)
        
        # Parse com AST
        try:
            tree = ast.parse(code_part, mode='exec')
        except SyntaxError as e:
            raise ASTParserError(f"Syntax error in line: {line}") from e
        
        # Verificar se é atribuição
        if not tree.body:
            raise ASTParserError(f"Empty line: {line}")
        
        node = tree.body[0]
        
        if not isinstance(node, ast.Assign):
            raise ASTParserError(
                f"Line is not an assignment: {line}\n"
                f"Node type: {type(node).__name__}"
            )
        
        # Extrair LHS (lado esquerdo)
        if len(node.targets) != 1:
            raise ASTParserError(
                f"Multiple assignment not supported: {line}"
            )
        
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            raise ASTParserError(
                f"Complex assignment not supported: {line}"
            )
        
        lhs = target.id
        
        # Extrair RHS (lado direito) como string
        rhs_symbolic = ast.get_source_segment(code_part, node.value)
        
        if rhs_symbolic is None:
            # Fallback: reconstruct from AST
            rhs_symbolic = self._unparse_expr(node.value)
        
        return ParsedAssignment(
            lhs=lhs,
            rhs_symbolic=rhs_symbolic,
            comment=comment_part,
            lineno=node.lineno,
            col_offset=node.col_offset,
            context=context.copy()
        )
    
    def parse_code_block(
        self, 
        code: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> List[ParsedAssignment]:
        """
        Parse bloco de código com múltiplas atribuições.
        
        Args:
            code: Bloco de código Python
            context: Contexto inicial de variáveis (opcional)
        
        Returns:
            Lista de ParsedAssignment em ordem
        
        Examples:
            >>> parser = PyMemorialASTParser()
            >>> code = '''
            ... q = 15.0   # kN/m
            ... L = 6.0    # m
            ... M = q * L**2 / 8
            ... '''
            >>> assigns = parser.parse_code_block(code)
            >>> [a.lhs for a in assigns]
            ['q', 'L', 'M']
        """
        if context is None:
            context = {}
        
        assignments = []
        
        # Split em linhas, preservando vazias para numeração correta
        lines = code.split('\n')
        
        for lineno, line in enumerate(lines, start=1):
            stripped = line.strip()
            
            # Ignorar linhas vazias e comentários puros
            if not stripped or stripped.startswith('#'):
                continue
            
            try:
                assignment = self.parse_assignment(line, context)
                assignment.lineno = lineno  # Sobrescrever com linha real
                assignments.append(assignment)
            except ASTParserError:
                # Ignorar linhas que não são atribuições
                # (podem ser imports, def, etc)
                continue
        
        return assignments
    
    def extract_variables(self, expr_src: str) -> List[str]:
        """
        Extrai nomes de variáveis de uma expressão.
        
        Args:
            expr_src: Expressão Python como string
        
        Returns:
            Lista de nomes de variáveis (ordem de aparição)
        
        Examples:
            >>> parser = PyMemorialASTParser()
            >>> parser.extract_variables("q * L**2 / 8")
            ['q', 'L']
            >>> parser.extract_variables("gamma_f * M_k")
            ['gamma_f', 'M_k']
        """
        try:
            tree = ast.parse(expr_src, mode='eval')
        except SyntaxError:
            return []
        
        variables = []
        seen = set()
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                if node.id not in seen:
                    variables.append(node.id)
                    seen.add(node.id)
        
        return variables
    
    def to_latex(
        self, 
        expr_src: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Converte expressão Python para LaTeX.
        
        CAPACIDADES:
        - Divisões → \\frac{num}{den}
        - Potências → base^{exp}
        - Multiplicação → \\cdot
        - Símbolos gregos → \\gamma, \\alpha, etc
        - Subscritos → x_y → x_{y}
        
        Args:
            expr_src: Expressão Python
            context: Contexto de variáveis (opcional)
        
        Returns:
            String LaTeX
        
        Examples:
            >>> parser = PyMemorialASTParser()
            >>> parser.to_latex("q * L**2 / 8")
            '\\\\frac{q \\\\cdot L^{2}}{8}'
            >>> parser.to_latex("gamma_s")
            '\\\\gamma_{s}'
        """
        if context is None:
            context = {}
        
        try:
            tree = ast.parse(expr_src, mode='eval')
        except SyntaxError:
            # Se falhar parsing, retornar original
            return expr_src
        
        return self._to_latex_node(tree.body)
    
    def format_number(self, x: float) -> str:
        """
        Formata número com precisão do config.
        
        Args:
            x: Número a formatar
        
        Returns:
            String formatada
        
        Examples:
            >>> parser = PyMemorialASTParser()
            >>> parser.format_number(1.23456789)
            '1.235'  # com precision=3
        """
        precision = self.config.display.precision
        return f"{x:.{precision}f}"
    
    # =========================================================================
    # MÉTODOS AUXILIARES PRIVADOS
    # =========================================================================
    
    def _split_comment(self, line: str) -> tuple[str, Optional[str]]:
        """
        Separa código de comentário inline.
        
        Respeita strings que contêm # (não split dentro de aspas).
        
        Returns:
            (código, comentário ou None)
        """
        # Regex para encontrar # fora de strings
        # Simplified: assume # não dentro de strings por ora
        parts = line.split('#', 1)
        
        if len(parts) == 1:
            return line.strip(), None
        
        code_part = parts[0].strip()
        comment_part = parts[1].strip()
        
        return code_part, comment_part if comment_part else None
    
    def _unparse_expr(self, node: ast.expr) -> str:
        """
        Reconstrói expressão de nó AST como string.
        
        Fallback caso ast.get_source_segment falhe.
        """
        if isinstance(node, ast.Constant):
            return str(node.value)
        elif isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.BinOp):
            left = self._unparse_expr(node.left)
            right = self._unparse_expr(node.right)
            op = self._op_to_str(node.op)
            return f"{left} {op} {right}"
        elif isinstance(node, ast.UnaryOp):
            operand = self._unparse_expr(node.operand)
            op = self._unaryop_to_str(node.op)
            return f"{op}{operand}"
        else:
            return "..."
    
    def _op_to_str(self, op: ast.operator) -> str:
        """Converte operador AST para string."""
        mapping = {
            ast.Add: '+',
            ast.Sub: '-',
            ast.Mult: '*',
            ast.Div: '/',
            ast.FloorDiv: '//',
            ast.Mod: '%',
            ast.Pow: '**',
        }
        return mapping.get(type(op), '?')
    
    def _unaryop_to_str(self, op: ast.unaryop) -> str:
        """Converte operador unário para string."""
        mapping = {
            ast.UAdd: '+',
            ast.USub: '-',
            ast.Not: 'not ',
        }
        return mapping.get(type(op), '?')
    
    def _to_latex_node(self, node: ast.expr) -> str:
        """
        Converte nó AST para LaTeX recursivamente.
        
        REGRAS:
        - BinOp(Div) → \\frac{left}{right}
        - BinOp(Pow) → base^{exp}
        - BinOp(Mult) → left \\cdot right
        - Name → converter símbolo grego/subscrito
        """
        if isinstance(node, ast.Constant):
            return str(node.value)
        
        elif isinstance(node, ast.Name):
            return self._to_latex_symbol(node.id)
        
        elif isinstance(node, ast.BinOp):
            # Divisão → fração
            if isinstance(node.op, ast.Div):
                left_latex = self._to_latex_node(node.left)
                right_latex = self._to_latex_node(node.right)
                return rf"\frac{{{left_latex}}}{{{right_latex}}}"
            
            # Potência → superscript
            elif isinstance(node.op, ast.Pow):
                base_latex = self._to_latex_node(node.left)
                exp_latex = self._to_latex_node(node.right)
                return f"{base_latex}^{{{exp_latex}}}"
            
            # Multiplicação → \cdot
            elif isinstance(node.op, ast.Mult):
                left_latex = self._to_latex_node(node.left)
                right_latex = self._to_latex_node(node.right)
                return rf"{left_latex} \cdot {right_latex}"
            
            # Outros operadores
            else:
                left_latex = self._to_latex_node(node.left)
                right_latex = self._to_latex_node(node.right)
                op_latex = self._op_to_latex(node.op)
                return f"{left_latex} {op_latex} {right_latex}"
        
        elif isinstance(node, ast.UnaryOp):
            operand_latex = self._to_latex_node(node.operand)
            if isinstance(node.op, ast.USub):
                return f"-{operand_latex}"
            elif isinstance(node.op, ast.UAdd):
                return f"+{operand_latex}"
            else:
                return operand_latex
        
        else:
            # Fallback
            return "..."
    
    def _op_to_latex(self, op: ast.operator) -> str:
        """Converte operador para símbolo LaTeX."""
        mapping = {
            ast.Add: '+',
            ast.Sub: '-',
            ast.Mult: r'\cdot',
            ast.Div: '/',  # Já tratado como fração
            ast.FloorDiv: r'\lfloor / \rfloor',
            ast.Mod: r'\bmod',
        }
        return mapping.get(type(op), '?')
    
    def _to_latex_symbol(self, symbol: str) -> str:
        """
        Converte símbolo Python para LaTeX.
        
        REGRAS:
        1. Detectar letra grega (gamma, alpha, etc) → \\gamma, \\alpha
        2. Detectar subscrito (gamma_s) → \\gamma_{s}
        3. Detectar superscrito (x_2) → x^{2} (futuro)
        
        Examples:
            gamma_s → \\gamma_{s}
            M_d → M_{d}
            alpha → \\alpha
        """
        # Regra 1: Símbolo com subscrito (underscore)
        if '_' in symbol:
            parts = symbol.split('_', 1)
            base = parts[0]
            subscript = parts[1]
            
            # Base pode ser grega
            base_latex = self._greek_to_latex(base) if base in ASCII_TO_GREEK else base
            
            return rf"{base_latex}_{{{subscript}}}"
        
        # Regra 2: Símbolo grego puro
        elif symbol in ASCII_TO_GREEK:
            return self._greek_to_latex(symbol)
        
        # Regra 3: Símbolo comum
        else:
            return symbol
    
    def _greek_to_latex(self, name: str) -> str:
        """
        Converte nome grego para comando LaTeX.
        
        Respeita config.symbols.greek_style.
        
        Examples:
            gamma → \\gamma (se greek_style='latex')
            gamma → γ (se greek_style='unicode')
        """
        if self.config.symbols.greek_style == 'unicode':
            # Retornar símbolo Unicode
            return ASCII_TO_GREEK.get(name, name)
        else:
            # Retornar comando LaTeX
            return rf"\{name}"


# =============================================================================
# FUNÇÃO DE CONVENIÊNCIA
# =============================================================================

def parse_assignment(line: str) -> ParsedAssignment:
    """
    Função de conveniência para parse rápido.
    
    Examples:
        >>> from pymemorial.recognition import parse_assignment
        >>> a = parse_assignment("M = 100")
        >>> a.lhs
        'M'
    """
    parser = PyMemorialASTParser()
    return parser.parse_assignment(line)
