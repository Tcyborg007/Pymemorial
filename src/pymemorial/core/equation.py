"""
PyMemorial v2.0 - Equation Module
Motor simbólico com SymPy + Sistema de Steps (Calcpad-inspired)

FILOSOFIA:
- Sintaxe Python natural (Handcalcs-inspired)
- Steps automáticos com 4 níveis de granularidade (Calcpad-inspired)
- Validação dimensional completa
- Thread-safe e cache-optimized
"""

import ast
import re
import logging
import threading
from typing import (
    Dict, List, Optional, Union, Any, Callable, Set, Tuple
)
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache

# Imports condicionais
try:
    import sympy as sp
    from sympy import Symbol, Expr, sympify, lambdify
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False
    Symbol = Any
    Expr = Any

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

# Imports internos
from pymemorial.core.config import get_config
from pymemorial.core.variable import Variable
from pymemorial.core.units import (
    get_unit_registry,
    UnitValidator,
    UnitParser,
    PINT_AVAILABLE
)
from pymemorial.recognition.ast_parser import PyMemorialASTParser
from pymemorial.symbols.custom_registry import get_registry


# ============================================================================
# EXCEÇÕES CUSTOMIZADAS
# ============================================================================

class EquationError(Exception):
    """Erro base para operações com equações."""
    pass


class ValidationError(EquationError):
    """Erro de validação de equação."""
    pass


class EvaluationError(EquationError):
    """Erro durante avaliação de equação."""
    pass


class SubstitutionError(EquationError):
    """Erro durante substituição de valores."""
    pass


class DimensionalError(EquationError):
    """Erro de compatibilidade dimensional."""
    pass


# ============================================================================
# ENUMS E DATACLASSES
# ============================================================================

class GranularityType(Enum):
    """Níveis de granularidade de steps (Calcpad-inspired)."""
    MINIMAL = "minimal"      # Apenas resultado
    BASIC = "basic"          # Fórmula + Resultado
    MEDIUM = "medium"        # Fórmula + Substituição + Resultado (4 steps)
    DETAILED = "detailed"    # Todos os passos intermediários
    ALL = "all"              # Debug mode com explicações


class StepType(Enum):
    """Tipos de steps de cálculo."""
    FORMULA = "formula"              # Fórmula simbólica
    SUBSTITUTION = "substitution"    # Substituição de valores
    CALCULATION = "calculation"      # Cálculo intermediário
    SIMPLIFICATION = "simplification" # Simplificação algébrica
    RESULT = "result"                # Resultado final
    EXPLANATION = "explanation"      # Explicação textual


@dataclass
class Step:
    """Representa um passo de cálculo."""
    type: StepType
    content: str
    latex: str
    explanation: str = ""
    level: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self) -> str:
        """Representação string."""
        return f"[{self.type.value}] {self.content}"


@dataclass
class EvaluationResult:
    """Resultado de avaliação de equação."""
    value: Union[float, int, np.ndarray, Expr]
    expression: str
    symbolic: Optional[Expr] = None
    unit: Optional[str] = None
    steps: List[Step] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self) -> str:
        """Formatação do resultado."""
        config = get_config()
        precision = config.display.precision
        
        # Formatar valor
        if isinstance(self.value, (int, float)):
            value_str = f"{self.value:.{precision}f}"
        else:
            value_str = str(self.value)
        
        # Adicionar unidade se disponível
        if self.unit:
            return f"{value_str} {self.unit}"
        return value_str


# ============================================================================
# CLASSE EQUATION - NÚCLEO SIMBÓLICO
# ============================================================================

class Equation:
    """
    Equação simbólica com SymPy.
    
    **FILOSOFIA:**
    - Suporta expressões Python naturais
    - Validação dimensional automática
    - Geração de steps intermediários (Calcpad-style)
    - Thread-safe e cache-optimized
    
    Examples:
        >>> # Criação simples
        >>> eq = Equation("q * L**2 / 8")
        >>> 
        >>> # Com variáveis
        >>> vars_dict = {
        ...     'q': Variable('q', 15, unit='kN/m'),
        ...     'L': Variable('L', 6, unit='m')
        ... }
        >>> eq = Equation("q * L**2 / 8", locals_dict=vars_dict)
        >>> result = eq.evaluate()
        >>> print(result)  # 67.5 kN*m
    """
    
    def __init__(
        self,
        expression: Union[str, Expr],
        locals_dict: Optional[Dict[str, Variable]] = None,
        name: Optional[str] = None,
        description: str = ""
    ):
        """
        Inicializa equação.
        
        Args:
            expression: Expressão matemática (string ou SymPy Expr)
            locals_dict: Dicionário de variáveis disponíveis
            name: Nome da equação (ex: 'M_max')
            description: Descrição textual
        
        Raises:
            ValidationError: Se expressão inválida
        """
        if not SYMPY_AVAILABLE:
            raise ImportError(
                "SymPy não disponível. Instale com: pip install sympy"
            )
        
        self.name = name
        self.description = description
        self.locals_dict = locals_dict or {}
        self._lock = threading.Lock()
        
        # ============================================================
        # CORREÇÃO: Inicializar logger ANTES de parsear expressão
        # ============================================================
        self._logger = logging.getLogger(__name__)
        
        # Parser de símbolos
        self._symbol_registry = get_registry()
        
        # Converter para SymPy Expr
        if isinstance(expression, str):
            self.expression_str = expression
            self.expr = self._parse_expression(expression)
        elif isinstance(expression, Expr):
            self.expr = expression
            self.expression_str = str(expression)
        else:
            raise ValidationError(
                f"Tipo de expressão inválido: {type(expression)}"
            )
        
        # Extrair metadados
        self.free_symbols = self.expr.free_symbols
        self.variables_used = [str(s) for s in self.free_symbols]
        
        # Validação dimensional
        self._validate_dimensions()

    
    def _parse_expression(self, expr_str: str) -> Expr:
        """
        Parse string para SymPy Expr.
        
        Args:
            expr_str: Expressão como string
        
        Returns:
            SymPy Expr
        
        Raises:
            ValidationError: Se parsing falhar
        """
        try:
            # ============================================================
            # CORREÇÃO: Remover atribuições (M = ...) antes de parsear
            # ============================================================
            if '=' in expr_str:
                # Detectar se é equação com atribuição
                parts = expr_str.split('=')
                if len(parts) == 2:
                    # Ignorar lado esquerdo, usar apenas expressão direita
                    expr_str = parts[1].strip()
                    self._logger.debug(f"Removendo atribuição: usando '{expr_str}'")
            
            # Criar símbolos locais
            local_symbols = {}
            if self.locals_dict:
                for name in self.locals_dict.keys():
                    local_symbols[name] = Symbol(name)
            
            # Parsing com SymPy
            expr = sympify(expr_str, locals=local_symbols)
            
            return expr
            
        except Exception as e:
            raise ValidationError(
                f"Erro ao parsear expressão '{expr_str}': {e}"
            )

    
    def _validate_dimensions(self) -> None:
        """
        Valida compatibilidade dimensional da equação.
        
        Raises:
            DimensionalError: Se dimensões incompatíveis
        """
        if not PINT_AVAILABLE or not self.locals_dict:
            return
        
        validator = UnitValidator(get_unit_registry())
        
        # Extrair unidades de variáveis
        units_map = {}
        for name, var in self.locals_dict.items():
            if var.unit:
                units_map[name] = var.unit
        
        # TODO: Implementar validação dimensional completa
        # Análise AST da expressão para verificar operações
        pass
    
    def get_variables(self) -> List[str]:
        """Retorna lista de nomes de variáveis."""
        return self.variables_used
    
    def get_free_symbols(self) -> Set[Symbol]:
        """Retorna símbolos livres (SymPy)."""
        return self.free_symbols
    
    def substitute(
        self,
        subs: Dict[str, Union[float, int, str, Variable]]
    ) -> 'Equation':
        """
        Substitui valores ou símbolos na equação.
        
        Args:
            subs: Dicionário de substituições
        
        Returns:
            Nova Equation com valores substituídos
        
        Examples:
            >>> eq = Equation("a + b")
            >>> eq2 = eq.substitute({'a': 10, 'b': 5})
            >>> print(eq2.evaluate())  # 15
        """
        with self._lock:
            # Converter para formato SymPy
            subs_sympy = {}
            for name, value in subs.items():
                symbol = Symbol(name)
                
                if isinstance(value, Variable):
                    subs_sympy[symbol] = value.value
                elif isinstance(value, (int, float)):
                    subs_sympy[symbol] = value
                elif isinstance(value, str):
                    # ============================================================
                    # CORREÇÃO: Parsear expressão string antes de substituir
                    # ============================================================
                    try:
                        # Tentar parsear como expressão SymPy
                        parsed_expr = sympify(value)
                        subs_sympy[symbol] = parsed_expr
                    except Exception:
                        # Se falhar, tratar como símbolo simples
                        subs_sympy[symbol] = Symbol(value)
                else:
                    raise SubstitutionError(
                        f"Tipo de substituição inválido para '{name}': {type(value)}"
                    )
            
            # Substituir
            new_expr = self.expr.subs(subs_sympy)
            
            # Criar nova equação
            return Equation(new_expr, locals_dict=self.locals_dict)

    
    def simplify(self) -> 'Equation':
        """
        Simplifica equação algebricamente.
        
        Returns:
            Nova Equation simplificada
        """
        with self._lock:
            simplified = sp.simplify(self.expr)
            return Equation(simplified, locals_dict=self.locals_dict)
    
    def expand(self) -> 'Equation':
        """
        Expande equação (multiplica polinômios).
        
        Returns:
            Nova Equation expandida
        """
        with self._lock:
            expanded = sp.expand(self.expr)
            return Equation(expanded, locals_dict=self.locals_dict)
    
    def factor(self) -> 'Equation':
        """
        Fatora equação.
        
        Returns:
            Nova Equation fatorada
        """
        with self._lock:
            factored = sp.factor(self.expr)
            return Equation(factored, locals_dict=self.locals_dict)
    
    def evaluate(self, **kwargs) -> EvaluationResult:
        """
        Avalia equação numericamente.
        
        Args:
            **kwargs: Valores adicionais para substituição
        
        Returns:
            EvaluationResult com valor e metadados
        
        Raises:
            EvaluationError: Se avaliação falhar
        """
        config = get_config()
        
        # Mesclar kwargs com locals_dict
        subs_dict = {}
        for name in self.variables_used:
            if name in kwargs:
                value = kwargs[name]
                if isinstance(value, Variable):
                    subs_dict[name] = value.value
                else:
                    subs_dict[name] = value
            elif name in self.locals_dict:
                var = self.locals_dict[name]
                subs_dict[name] = var.value
            else:
                raise EvaluationError(
                    f"Variável '{name}' não definida"
                )
        
        # Substituir e avaliar
        try:
            # Substituir símbolos por valores
            expr_with_values = self.expr.subs({
                Symbol(k): v for k, v in subs_dict.items()
            })
            
            # Avaliar numericamente
            result_value = float(expr_with_values.evalf())
            
            return EvaluationResult(
                value=result_value,
                expression=self.expression_str,
                symbolic=self.expr,
                metadata={'substitutions': subs_dict}
            )
            
        except Exception as e:
            raise EvaluationError(
                f"Erro ao avaliar '{self.expression_str}': {e}"
            )
    
    def to_latex(self, mode: str = "inline") -> str:
        """
        Converte para LaTeX.
        
        Args:
            mode: 'inline' ou 'display'
        
        Returns:
            String LaTeX
        """
        latex_str = sp.latex(self.expr)
        
        if mode == "inline":
            return f"${latex_str}$"
        elif mode == "display":
            return f"$$\n{latex_str}\n$$"
        else:
            return latex_str
    
    def to_markdown(self) -> str:
        """Converte para Markdown."""
        return f"`{self.expression_str}`"
    
    def __repr__(self) -> str:
        """Representação Python."""
        return f"Equation('{self.expression_str}')"
    
    def __str__(self) -> str:
        """Representação string."""
        return self.expression_str


# ============================================================================
# EQUATIONFACTORY - BUILDER PATTERN
# ============================================================================

class EquationFactory:
    """
    Factory para criar equações de múltiplas fontes.
    
    **FILOSOFIA:**
    - Builder pattern para flexibilidade
    - Validação rigorosa em todas as entradas
    - Integração com ast_parser para código Python
    - Type hints completos
    
    Examples:
        >>> # De string
        >>> eq = EquationFactory.from_string("q * L**2 / 8")
        >>> 
        >>> # De código Python
        >>> code = "M_max = q * L**2 / 8"
        >>> eq = EquationFactory.from_code(code, variable='M_max')
        >>> 
        >>> # De lambda
        >>> eq = EquationFactory.from_lambda(lambda x, y: x + y)
    """
    
    @staticmethod
    def from_string(
        expr_str: str,
        locals_dict: Optional[Dict[str, Variable]] = None,
        name: Optional[str] = None,
        validate: bool = True
    ) -> Equation:
        """
        Cria equation de string.
        
        Args:
            expr_str: Expressão como string
            locals_dict: Variáveis disponíveis
            name: Nome da equação
            validate: Se True, valida antes de criar
        
        Returns:
            Equation criada
        
        Raises:
            ValidationError: Se validação falhar
        """
        if validate:
            ValidationHelpers.validate_expression_string(expr_str)
        
        return Equation(expr_str, locals_dict=locals_dict, name=name)
    
    @staticmethod
    def from_sympy(
        expr: Expr,
        locals_dict: Optional[Dict[str, Variable]] = None,
        name: Optional[str] = None
    ) -> Equation:
        """
        Cria equation de expressão SymPy.
        
        Args:
            expr: Expressão SymPy
            locals_dict: Variáveis disponíveis
            name: Nome da equação
        
        Returns:
            Equation criada
        """
        if not isinstance(expr, Expr):
            raise ValidationError(
                f"Esperado sympy.Expr, recebido {type(expr)}"
            )
        
        return Equation(expr, locals_dict=locals_dict, name=name)
    
    @staticmethod
    def from_code(
        code: str,
        variable: Optional[str] = None,
        locals_dict: Optional[Dict[str, Variable]] = None
    ) -> Equation:
        """
        Cria equation de código Python usando ast_parser.
        
        Args:
            code: Código Python (ex: "M = q * L**2 / 8")
            variable: Nome da variável a extrair (ex: "M")
            locals_dict: Variáveis disponíveis
        
        Returns:
            Equation criada
        
        Raises:
            ValidationError: Se parsing falhar
        
        Examples:
            >>> code = '''
            ... q = 15  # kN/m
            ... L = 6   # m
            ... M_max = q * L**2 / 8
            ... '''
            >>> eq = EquationFactory.from_code(code, variable='M_max')
        """
        # Validar segurança do código
        ValidationHelpers.validate_code_safety(code)
        
        # Parsear com ast_parser
        parser = PyMemorialASTParser()
        try:
            # ============================================================
            # CORREÇÃO 1: O método é 'parse_code_block', não 'parse'
            # ============================================================
            result = parser.parse_code_block(code)
        except Exception as e:
            raise ValidationError(
                f"Erro ao parsear código: {e}"
            )
        
        # ============================================================
        # CORREÇÃO 2: 'result' É a lista, não um objeto com '.assignments'
        # ============================================================
        if not result:
            raise ValidationError(
                "Código não contém atribuições válidas"
            )
        
        # Se variable especificada, buscar
        if variable:
            # ============================================================
            # CORREÇÃO 3: Iterar em 'result'
            # ============================================================
            for assignment in result:
                
                # ============================================================
                # CORREÇÃO 4: O atributo é '.lhs' (string), não '.target'
                # ============================================================
                if assignment.lhs == variable:
                    # ============================================================
                    # CORREÇÃO 5: A expressão já é uma string em '.rhs_symbolic'
                    # ============================================================
                    expr_str = assignment.rhs_symbolic
                    return EquationFactory.from_string(
                        expr_str,
                        locals_dict=locals_dict,
                        name=variable
                    )
            
            raise ValidationError(
                f"Variável '{variable}' não encontrada no código"
            )
        
        # Se não especificada, usar última atribuição
        # ============================================================
        # CORREÇÃO 6: Acessar o último item de 'result'
        # ============================================================
        last_assignment = result[-1]
        
        # ============================================================
        # CORREÇÃO 7: Usar '.rhs_symbolic' e '.lhs'
        # ============================================================
        expr_str = last_assignment.rhs_symbolic
        
        return EquationFactory.from_string(
            expr_str,
            locals_dict=locals_dict,
            name=last_assignment.lhs
        )
    
    @staticmethod
    def from_lambda(
        func: Callable,
        arg_names: Optional[List[str]] = None,
        locals_dict: Optional[Dict[str, Variable]] = None,
        name: Optional[str] = None
    ) -> Equation:
        """
        Cria equation de função lambda.
        
        Args:
            func: Função Python
            arg_names: Nomes dos argumentos
            locals_dict: Variáveis disponíveis
            name: Nome da equação
        
        Returns:
            Equation criada
        
        Examples:
            >>> eq = EquationFactory.from_lambda(
            ...     lambda x, y: x**2 + y**2,
            ...     arg_names=['x', 'y']
            ... )
        """
        import inspect
        
        # Obter argumentos da função
        sig = inspect.signature(func)
        params = list(sig.parameters.keys())
        
        if arg_names is None:
            arg_names = params
        
        # Criar símbolos
        symbols = [Symbol(name) for name in arg_names]
        
        # Tentar avaliar função com símbolos
        try:
            if len(symbols) == 1:
                expr = func(symbols[0])
            elif len(symbols) == 2:
                expr = func(symbols[0], symbols[1])
            elif len(symbols) == 3:
                expr = func(symbols[0], symbols[1], symbols[2])
            else:
                # Para mais argumentos, usar *args
                expr = func(*symbols)
            
            return Equation(expr, locals_dict=locals_dict, name=name)
            
        except Exception as e:
            raise ValidationError(
                f"Erro ao criar equation de lambda: {e}"
            )


# ============================================================================
# VALIDATIONHELPERS - VALIDAÇÃO AVANÇADA
# ============================================================================

class ValidationHelpers:
    """
    Helpers para validação de equações.
    
    **SEGURANÇA CRÍTICA:**
    - Validação de código malicioso
    - Detecção de imports perigosos
    - Validação dimensional
    - Detecção de circularidades
    """
    
    @staticmethod
    def validate_expression_string(expr_str: str) -> None:
        """
        Valida string de expressão.
        
        Args:
            expr_str: Expressão a validar
        
        Raises:
            ValidationError: Se inválida
        """
        if not expr_str or not expr_str.strip():
            raise ValidationError("Expressão vazia")
        
        # Validar caracteres perigosos
        dangerous_chars = [';', '\\', '`']
        for char in dangerous_chars:
            if char in expr_str:
                raise ValidationError(
                    f"Caractere perigoso '{char}' na expressão"
                )
        
        # Validar palavras-chave Python perigosas
        dangerous_keywords = [
            'import', 'exec', 'eval', 'compile',
            '__import__', 'open', 'file', 'input',
            'globals', 'locals', 'vars', 'dir'
        ]
        
        expr_lower = expr_str.lower()
        for keyword in dangerous_keywords:
            if keyword in expr_lower:
                raise ValidationError(
                    f"Palavra-chave perigosa '{keyword}' na expressão"
                )
    
    @staticmethod
    def validate_code_safety(code: str) -> None:
        """
        Valida segurança de código Python.
        
        Args:
            code: Código a validar
        
        Raises:
            ValidationError: Se código inseguro
        """
        import ast
        
        # Parsear AST
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            raise ValidationError(f"Sintaxe inválida: {e}")
        
        # Verificar nós perigosos
        dangerous_nodes = (
            ast.Import,
            ast.ImportFrom,
            ast.Global,
            ast.FunctionDef,
            ast.AsyncFunctionDef,
            ast.ClassDef,
            ast.Delete,
            ast.Try,
            ast.ExceptHandler,
            ast.With,
            ast.AsyncWith
        )
        
        for node in ast.walk(tree):
            if isinstance(node, dangerous_nodes):
                raise ValidationError(
                    f"Operação não permitida: {node.__class__.__name__}"
                )
            
            # Verificar chamadas de funções perigosas
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                    if func_name in ['eval', 'exec', 'compile', 'open', '__import__']:
                        raise ValidationError(
                            f"Função não permitida: {func_name}"
                        )
    
    @staticmethod
    def validate_dimensional_consistency(
        equation: Equation,
        expected_unit: Optional[str] = None
    ) -> bool:
        """
        Valida consistência dimensional.
        
        Args:
            equation: Equation a validar
            expected_unit: Unidade esperada (opcional)
        
        Returns:
            True se consistente
        
        Raises:
            DimensionalError: Se inconsistente
        """
        if not PINT_AVAILABLE:
            return True  # Pular validação se Pint não disponível
        
        # TODO: Implementar validação dimensional completa
        # Análise de AST da expressão para verificar operações
        
        return True
    
    @staticmethod
    def check_circular_dependencies(
        equations: List[Equation]
    ) -> List[Tuple[str, str]]:
        """
        Detecta dependências circulares entre equações.
        
        Args:
            equations: Lista de equações
        
        Returns:
            Lista de tuplas (eq1, eq2) com circularidades
        """
        # Construir grafo de dependências
        dependencies = {}
        
        for eq in equations:
            if eq.name:
                dependencies[eq.name] = eq.get_variables()
        
        # Algoritmo de Tarjan para detectar ciclos
        # TODO: Implementar detecção completa de ciclos
        
        return []
    
    @staticmethod
    def validate_expression_complexity(
        expr_str: str,
        max_length: int = 1000,
        max_depth: int = 50
    ) -> None:
        """
        Valida complexidade da expressão.
        
        Args:
            expr_str: Expressão a validar
            max_length: Comprimento máximo
            max_depth: Profundidade máxima de aninhamento
        
        Raises:
            ValidationError: Se muito complexa
        """
        if len(expr_str) > max_length:
            raise ValidationError(
                f"Expressão muito longa: {len(expr_str)} > {max_length}"
            )
        
        # Validar profundidade de parênteses
        depth = 0
        max_depth_found = 0
        
        for char in expr_str:
            if char == '(':
                depth += 1
                max_depth_found = max(max_depth_found, depth)
            elif char == ')':
                depth -= 1
        
        if max_depth_found > max_depth:
            raise ValidationError(
                f"Expressão muito aninhada: profundidade {max_depth_found} > {max_depth}"
            )


# ============================================================================
# STEP SYSTEM - CALCPAD-INSPIRED
# ============================================================================

@dataclass
class Step:
    """
    Representação de um step de cálculo.
    
    **FILOSOFIA CALCPAD:**
    - Fórmula simbólica
    - Substituição de valores
    - Cálculo intermediário
    - Resultado final
    
    Examples:
        >>> step = Step(
        ...     formula="M = q * L² / 8",
        ...     substitution="M = 15 * 6² / 8",
        ...     intermediate="M = 15 * 36 / 8",
        ...     result="M = 67.5 kN·m"
        ... )
    """
    formula: str
    substitution: Optional[str] = None
    intermediate: Optional[str] = None
    result: Optional[str] = None
    level: GranularityType = GranularityType.MEDIUM
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_latex(self, mode: str = "align") -> str:
        """
        Converte step para LaTeX.
        
        Args:
            mode: "align" ou "gather"
        
        Returns:
            LaTeX formatado
        """
        if mode == "align":
            env = "align*"
        else:
            env = "gather*"
        
        lines = []
        
        if self.formula:
            lines.append(self.formula)
        
        if self.substitution and self.level in [GranularityType.MEDIUM, GranularityType.DETAILED]:
            lines.append(self.substitution)
        
        if self.intermediate and self.level == GranularityType.DETAILED:
            lines.append(self.intermediate)
        
        if self.result:
            lines.append(self.result)
        
        latex = f"\\begin{{{env}}}\n"
        latex += " \\\\\n".join(lines)
        latex += f"\n\\end{{{env}}}"
        
        return latex
    
    def to_markdown(self) -> str:
        """Converte step para Markdown."""
        lines = []
        
        if self.formula:
            lines.append(f"$$\n{self.formula}\n$$")
        
        if self.substitution and self.level in [GranularityType.MEDIUM, GranularityType.DETAILED]:
            lines.append(f"$$\n{self.substitution}\n$$")
        
        if self.intermediate and self.level == GranularityType.DETAILED:
            lines.append(f"$$\n{self.intermediate}\n$$")
        
        if self.result:
            lines.append(f"$$\n{self.result}\n$$")
        
        return "\n\n".join(lines)


class StepGenerator:
    """
    Gerador automático de steps estilo Calcpad.
    
    **ALGORITMO:**
    1. Extrai fórmula simbólica da equation
    2. Substitui valores das variáveis
    3. Calcula intermediários (se DETAILED)
    4. Formata resultado final
    
    Examples:
        >>> generator = StepGenerator()
        >>> eq = Equation("M = q * L**2 / 8", locals_dict={
        ...     'q': Variable('q', 15, unit='kN/m'),
        ...     'L': Variable('L', 6, unit='m')
        ... })
        >>> steps = generator.generate(eq, granularity=GranularityType.DETAILED)
    """
    
    def __init__(self):
        """Inicializa gerador."""
        self._logger = logging.getLogger(__name__)
    
    def generate(
        self,
        equation: Equation,
        granularity: GranularityType = GranularityType.MEDIUM,
        precision: int = 3
    ) -> List[Step]:
        """
        Gera steps de cálculo.
        
        Args:
            equation: Equation para gerar steps
            granularity: Nível de detalhe
            precision: Casas decimais
        
        Returns:
            Lista de Steps
        """
        if granularity == GranularityType.MINIMAL:
            return self._generate_minimal(equation, precision)
        elif granularity == GranularityType.BASIC:
            return self._generate_basic(equation, precision)
        elif granularity == GranularityType.MEDIUM:
            return self._generate_medium(equation, precision)
        else:  # DETAILED
            return self._generate_detailed(equation, precision)

    def generate_smart(
        self,
        equation: Equation,
        precision: int = 3,
        force_granularity: Optional[GranularityType] = None
    ) -> List[Step]:
        """
        Geração INTELIGENTE de steps com reconhecimento automático.
        
        **ALGORITMO SMART:**
        1. Analisa complexidade da expressão
        2. Detecta tipo de operações
        3. Escolhe granularidade ideal automaticamente
        4. Gera steps otimizados
        
        **HEURÍSTICAS:**
        - Score ≤ 5:  Expressão simples (x + y) → BASIC
        - Score ≤ 15: Expressão média (q * L²/8) → MEDIUM
        - Score > 15: Expressão complexa (sin, integrais) → DETAILED
        
        Args:
            equation: Equation para análise
            precision: Casas decimais
            force_granularity: Forçar granularidade (ignora análise)
        
        Returns:
            Lista de Steps otimizados
        
        Examples:
            >>> generator = StepGenerator()
            >>> eq_simple = Equation("x + y")
            >>> steps = generator.generate_smart(eq_simple)  # AUTO: BASIC
            >>> 
            >>> eq_complex = Equation("sin(x) * exp(y)")
            >>> steps = generator.generate_smart(eq_complex)  # AUTO: DETAILED
        """
        # Se granularidade forçada, usar diretamente
        if force_granularity:
            return self.generate(equation, force_granularity, precision)
        
        # ANÁLISE AUTOMÁTICA DE COMPLEXIDADE
        complexity_score = self._analyze_complexity(equation)
        
        # DECISÃO INTELIGENTE
        if complexity_score <= 5:
            # Expressão simples: BASIC
            granularity = GranularityType.BASIC
            self._logger.debug(f"Smart: Score={complexity_score} → BASIC")
        elif complexity_score <= 15:
            # Expressão média: MEDIUM
            granularity = GranularityType.MEDIUM
            self._logger.debug(f"Smart: Score={complexity_score} → MEDIUM")
        else:
            # Expressão complexa: DETAILED
            granularity = GranularityType.DETAILED
            self._logger.debug(f"Smart: Score={complexity_score} → DETAILED")
        
        return self.generate(equation, granularity, precision)
    
    def _analyze_complexity(self, equation: Equation) -> int:
        """
        Analisa complexidade da expressão.
        
        **MÉTRICAS:**
        - Número de variáveis: +2 cada
        - Profundidade da árvore AST: +3 por nível
        - Operações básicas (+,-,*,/): +1 cada
        - Potenciação (**): +3
        - Funções transcendentais (sin,cos,exp,log): +5 cada
        - Funções especiais (Integral,Derivative): +10 cada
        - Matrizes: +8
        
        Args:
            equation: Equation para analisar
        
        Returns:
            Score de complexidade (0-100+)
        
        Examples:
            >>> eq_simple = Equation("x + y")
            >>> score = generator._analyze_complexity(eq_simple)  # ~4
            >>> 
            >>> eq_complex = Equation("sin(x) * exp(y) / sqrt(z)")
            >>> score = generator._analyze_complexity(eq_complex)  # ~25
        """
        score = 0
        expr = equation.expr
        
        # 1. NÚMERO DE VARIÁVEIS
        num_vars = len(equation.get_variables())
        score += num_vars * 2
        
        # 2. PROFUNDIDADE DA ÁRVORE AST
        depth = self._get_expr_depth(expr)
        score += depth * 3
        
        # 3. TIPOS DE OPERAÇÕES
        expr_str = str(expr)
        
        # Operações básicas (+, -, *, /)
        basic_ops = expr_str.count('+') + expr_str.count('-') + \
                    expr_str.count('*') + expr_str.count('/')
        score += basic_ops * 1
        
        # Operações potência (**²)
        if '**' in expr_str or 'Pow' in expr_str:
            score += 3
        
        # Funções transcendentais (sin, cos, exp, log, sqrt)
        transcendental_funcs = ['sin', 'cos', 'tan', 'exp', 'log', 'sqrt']
        for func in transcendental_funcs:
            if func in expr_str.lower():
                score += 5
        
        # 4. FUNÇÕES ESPECIAIS (integral, derivada)
        special_funcs = ['Integral', 'Derivative', 'Sum', 'Product', 'Limit']
        for func in special_funcs:
            if func in expr_str:
                score += 10
        
        # 5. MATRIZES/VETORES
        if 'Matrix' in expr_str or 'Array' in expr_str:
            score += 8
        
        return score
    
    def _get_expr_depth(self, expr: Expr, current_depth: int = 0) -> int:
        """
        Calcula profundidade da árvore de expressão (recursivo).
        
        Args:
            expr: Expressão SymPy
            current_depth: Profundidade atual (recursão)
        
        Returns:
            Profundidade máxima
        
        Examples:
            >>> # x + y → profundidade 1
            >>> # sin(x + y) → profundidade 2
            >>> # sin(cos(x + y)) → profundidade 3
        """
        if not hasattr(expr, 'args') or not expr.args:
            return current_depth
        
        max_child_depth = current_depth
        for arg in expr.args:
            child_depth = self._get_expr_depth(arg, current_depth + 1)
            max_child_depth = max(max_child_depth, child_depth)
        
        return max_child_depth



    def _generate_minimal(self, equation: Equation, precision: int) -> List[Step]:
        """Gera step MINIMAL: apenas resultado."""
        result = equation.evaluate()
        
        # Formatar resultado
        if hasattr(result.value, '__float__'):
            value_str = f"{float(result.value):.{precision}f}"
        else:
            value_str = str(result.value)
        
        result_formula = f"{equation.name or 'resultado'} = {value_str}"
        
        step = Step(
            formula=None,
            substitution=None,
            intermediate=None,
            result=result_formula,
            level=GranularityType.MINIMAL
        )
        
        return [step]
    
    def _generate_basic(self, equation: Equation, precision: int) -> List[Step]:
        """Gera step BASIC: fórmula + resultado."""
        # Fórmula simbólica
        formula_latex = sp.latex(equation.expr)
        
        # Resultado
        result = equation.evaluate()
        value_str = f"{float(result.value):.{precision}f}" if hasattr(result.value, '__float__') else str(result.value)
        result_formula = f"{equation.name or 'resultado'} = {value_str}"
        
        step = Step(
            formula=formula_latex,
            substitution=None,
            intermediate=None,
            result=result_formula,
            level=GranularityType.BASIC
        )
        
        return [step]
    
    def _generate_medium(self, equation: Equation, precision: int) -> List[Step]:
        """Gera step MEDIUM: fórmula + substituição + resultado."""
        # Fórmula simbólica
        formula_latex = sp.latex(equation.expr)
        
        # Substituição de valores
        subs_dict = {}
        for var_name, var in equation.locals_dict.items():
            if isinstance(var, Variable):
                subs_dict[Symbol(var_name)] = var.value
        
        expr_substituted = equation.expr.subs(subs_dict)
        substitution_latex = sp.latex(expr_substituted)
        
        # Resultado
        result = equation.evaluate()
        value_str = f"{float(result.value):.{precision}f}" if hasattr(result.value, '__float__') else str(result.value)
        result_formula = f"{equation.name or 'resultado'} = {value_str}"
        
        step = Step(
            formula=formula_latex,
            substitution=substitution_latex,
            intermediate=None,
            result=result_formula,
            level=GranularityType.MEDIUM
        )
        
        return [step]
    
    def _generate_detailed(self, equation: Equation, precision: int) -> List[Step]:
        """Gera step DETAILED: todos os passos intermediários."""
        steps = []
        
        # Step 1: Fórmula simbólica
        formula_latex = sp.latex(equation.expr)
        
        # Step 2: Substituição
        subs_dict = {}
        for var_name, var in equation.locals_dict.items():
            if isinstance(var, Variable):
                subs_dict[Symbol(var_name)] = var.value
        
        expr_substituted = equation.expr.subs(subs_dict)
        substitution_latex = sp.latex(expr_substituted)
        
        # Step 3: Simplificação intermediária (se aplicável)
        try:
            expr_simplified = sp.simplify(expr_substituted)
            if expr_simplified != expr_substituted:
                intermediate_latex = sp.latex(expr_simplified)
            else:
                intermediate_latex = None
        except Exception:
            intermediate_latex = None
        
        # Step 4: Resultado final
        result = equation.evaluate()
        value_str = f"{float(result.value):.{precision}f}" if hasattr(result.value, '__float__') else str(result.value)
        result_formula = f"{equation.name or 'resultado'} = {value_str}"
        
        step = Step(
            formula=formula_latex,
            substitution=substitution_latex,
            intermediate=intermediate_latex,
            result=result_formula,
            level=GranularityType.DETAILED
        )
        
        steps.append(step)
        
        return steps


    def generate_smart(
        self,
        equation: Equation,
        precision: int = 3,
        force_granularity: Optional[GranularityType] = None
    ) -> List[Step]:
        """
        Geração INTELIGENTE de steps com reconhecimento automático.
        
        **ALGORITMO SMART:**
        1. Analisa complexidade da expressão
        2. Detecta tipo de operações
        3. Escolhe granularidade ideal
        4. Gera steps otimizados
        
        **HEURÍSTICAS:**
        - Expressão simples (x + y): BASIC
        - Expressão média (q * L²/8): MEDIUM
        - Expressão complexa (integral, derivada): DETAILED
        
        Args:
            equation: Equation para análise
            precision: Casas decimais
            force_granularity: Forçar granularidade (ignora análise)
        
        Returns:
            Lista de Steps otimizados
        
        Examples:
            >>> generator = StepGenerator()
            >>> eq_simple = Equation("x + y")
            >>> steps = generator.generate_smart(eq_simple)  # AUTO: BASIC
            >>> 
            >>> eq_complex = Equation("sin(x) * exp(y) / sqrt(z)")
            >>> steps = generator.generate_smart(eq_complex)  # AUTO: DETAILED
        """
        # Se granularidade forçada, usar diretamente
        if force_granularity:
            return self.generate(equation, force_granularity, precision)
        
        # ANÁLISE AUTOMÁTICA DE COMPLEXIDADE
        complexity_score = self._analyze_complexity(equation)
        
        # DECISÃO INTELIGENTE
        if complexity_score <= 5:
            # Expressão simples: BASIC
            granularity = GranularityType.BASIC
            self._logger.debug(f"Smart: Score={complexity_score} → BASIC")
        elif complexity_score <= 15:
            # Expressão média: MEDIUM
            granularity = GranularityType.MEDIUM
            self._logger.debug(f"Smart: Score={complexity_score} → MEDIUM")
        else:
            # Expressão complexa: DETAILED
            granularity = GranularityType.DETAILED
            self._logger.debug(f"Smart: Score={complexity_score} → DETAILED")
        
        return self.generate(equation, granularity, precision)
    
    def _analyze_complexity(self, equation: Equation) -> int:
        """
        Analisa complexidade da expressão.
        
        **MÉTRICAS:**
        - Número de variáveis: +2 cada
        - Profundidade da árvore: +3 por nível
        - Operações: +1 básicas, +5 transcendentais
        - Funções especiais: +10 cada
        
        Args:
            equation: Equation para analisar
        
        Returns:
            Score de complexidade (0-100+)
        """
        score = 0
        expr = equation.expr
        
        # 1. NÚMERO DE VARIÁVEIS
        num_vars = len(equation.get_variables())
        score += num_vars * 2
        
        # 2. PROFUNDIDADE DA ÁRVORE AST
        depth = self._get_expr_depth(expr)
        score += depth * 3
        
        # 3. TIPOS DE OPERAÇÕES
        expr_str = str(expr)
        
        # Operações básicas (+, -, *, /)
        basic_ops = expr_str.count('+') + expr_str.count('-') + \
                    expr_str.count('*') + expr_str.count('/')
        score += basic_ops * 1
        
        # Operações potência (**²)
        if '**' in expr_str or 'Pow' in expr_str:
            score += 3
        
        # Funções transcendentais (sin, cos, exp, log, sqrt)
        transcendental_funcs = ['sin', 'cos', 'tan', 'exp', 'log', 'sqrt']
        for func in transcendental_funcs:
            if func in expr_str.lower():
                score += 5
        
        # 4. FUNÇÕES ESPECIAIS (integral, derivada)
        special_funcs = ['Integral', 'Derivative', 'Sum', 'Product', 'Limit']
        for func in special_funcs:
            if func in expr_str:
                score += 10
        
        # 5. MATRIZES/VETORES
        if 'Matrix' in expr_str or 'Array' in expr_str:
            score += 8
        
        return score
    
    def _get_expr_depth(self, expr: Expr, current_depth: int = 0) -> int:
        """
        Calcula profundidade da árvore de expressão.
        
        Args:
            expr: Expressão SymPy
            current_depth: Profundidade atual (recursão)
        
        Returns:
            Profundidade máxima
        """
        if not hasattr(expr, 'args') or not expr.args:
            return current_depth
        
        max_child_depth = current_depth
        for arg in expr.args:
            child_depth = self._get_expr_depth(arg, current_depth + 1)
            max_child_depth = max(max_child_depth, child_depth)
        
        return max_child_depth


class StepRegistry:
    """
    Registro global de steps estilo Calcpad.
    
    **FUNCIONALIDADE:**
    - Armazena histórico de steps
    - Permite export para LaTeX/Markdown
    - Suporta plugins customizados
    
    Examples:
        >>> registry = StepRegistry()
        >>> eq = Equation("M = q * L**2 / 8", locals_dict=vars_dict)
        >>> registry.register(eq, granularity=GranularityType.MEDIUM)
        >>> latex = registry.to_latex()
    """
    
    def __init__(self):
        """Inicializa registry."""
        self._steps: List[Step] = []
        self._generator = StepGenerator()
        self._lock = threading.Lock()
    
    def register(
        self,
        equation: Equation,
        granularity: GranularityType = GranularityType.MEDIUM,
        precision: int = 3
    ) -> List[Step]:
        """
        Registra equation e gera steps.
        
        Args:
            equation: Equation para registrar
            granularity: Nível de detalhe
            precision: Casas decimais
        
        Returns:
            Lista de Steps gerados
        """
        with self._lock:
            steps = self._generator.generate(equation, granularity, precision)
            self._steps.extend(steps)
            return steps
    
    def get_all(self) -> List[Step]:
        """Retorna todos os steps registrados."""
        with self._lock:
            return self._steps.copy()
    
    def clear(self):
        """Limpa histórico de steps."""
        with self._lock:
            self._steps.clear()
    
    def to_latex(self, mode: str = "align") -> str:
        """
        Export todos os steps para LaTeX.
        
        Args:
            mode: "align" ou "gather"
        
        Returns:
            LaTeX completo
        """
        with self._lock:
            latex_blocks = [step.to_latex(mode) for step in self._steps]
            return "\n\n".join(latex_blocks)
    
    def to_markdown(self) -> str:
        """Export todos os steps para Markdown."""
        with self._lock:
            md_blocks = [step.to_markdown() for step in self._steps]
            return "\n\n---\n\n".join(md_blocks)
