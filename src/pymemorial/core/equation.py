# src/pymemorial/core/equation.py
"""
Symbolic and Numerical Equation System - PyMemorial v2.0 (Production Ready).

Features:
- ✅ Safe expression parsing (SymPy + AST fallback)
- ✅ Automatic norm factor application (via standards module)
- ✅ Detailed step-by-step solutions (PT-BR)
- ✅ LaTeX output with Greek symbol support
- ✅ Integration with text processor and recognition
- ✅ Smart optimization suggestions
- ✅ Immutable design (functional updates)
- ✅ Comprehensive validation
- ✅ Cache-aware evaluation
- ✅ HYBRID substitution (xreplace for speed, subs for algebra)
- ✅ Smart step-by-step with granularity control

Author: PyMemorial Team
Date: 2025-10-21
Version: 2.1.0 (OPTIMIZED - xreplace + granularity support)
"""

from __future__ import annotations

import ast
import logging
from typing import Dict, Optional, Union, Any, List, TYPE_CHECKING, Literal
from dataclasses import dataclass, field
from copy import deepcopy

# Core dependencies
try:
    import sympy as sp
    import numpy as np
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False
    sp = None
    np = None
from . import NUMPY_AVAILABLE # Importa a flag do __init__.py do core
# Type checking imports
if TYPE_CHECKING:
    from .variable import Variable
    from .cache import ResultCache
from .units import Quantity, PINT_OK, QuantityTypeHint, ureg, strip_units, latex_unit # Adicionado ureg, strip_units, latex_unit# Optional imports
try:
    from ..recognition import get_engine, DetectedVariable
    RECOGNITION_AVAILABLE = True
except ImportError:
    RECOGNITION_AVAILABLE = False
    get_engine = EngineeringNLP = DetectedVar = None

try:
    from ..standards import get_norm_factor, NormCode
    STANDARDS_AVAILABLE = True
except ImportError:
    STANDARDS_AVAILABLE = False
    # Fallback norm factors
    DEFAULT_NORM_FACTORS = {
        'NBR6118_2023': {'safety_factor': 1.4},
        'AISC360_22': {'safety_factor': 1.5},
        'EC2_2004': {'safety_factor': 1.5},
    }

# ============================================================================
# LOGGER
# ============================================================================

logger = logging.getLogger(__name__)

# ============================================================================
# TYPE DEFINITIONS
# ============================================================================

GranularityType = Literal['minimal', 'basic', 'normal', 'detailed', 'all', 'smart']




# ADICIONAR ao equation.py (após linha 100)

from typing import Protocol, Callable, List, Dict, Any
from abc import ABC, abstractmethod

class StepPlugin(Protocol):
    """
    Protocol para plugins de steps personalizados.
    
    Permite ao usuário injetar processamento customizado em qualquer
    ponto da geração de steps.
    
    Attributes:
        priority: Prioridade de execução (0 = primeiro)
        name: Nome identificador do plugin
    
    Examples:
    --------
    >>> class NBR6118ValidationPlugin:
    ...     priority = 0
    ...     name = 'NBR6118_Validator'
    ...     
    ...     def process(self, expression, variables, context):
    ...         # Validação customizada NBR 6118
    ...         return {
    ...             'step': 'Validação NBR 6118',
    ...             'operation': 'validation',
    ...             'expr': 'OK',
    ...             'numeric': None,
    ...             'description': 'Verificação de limites normativos'
    ...         }
    >>> 
    >>> # Registrar plugin globalmente
    >>> StepRegistry.register(NBR6118ValidationPlugin())
    >>> 
    >>> # Ou usar diretamente em uma equação
    >>> eq.steps(plugins=[NBR6118ValidationPlugin()])
    """
    
    def process(
        self,
        expression: sp.Expr,
        variables: Dict[str, 'Variable'],
        context: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Processa um passo customizado.
        
        Args:
            expression: Expressão SymPy atual
            variables: Variáveis disponíveis
            context: Contexto com histórico de passos e resultados
        
        Returns:
            Dict com estrutura:
            {
                'step': str,          # Descrição do passo
                'operation': str,     # Tipo de operação
                'expr': str,          # LaTeX da expressão
                'numeric': float,     # Valor numérico (opcional)
                'description': str    # Descrição detalhada
            }
            ou None para pular este passo
        """
        ...
    
    @property
    def priority(self) -> int:
        """Prioridade de execução (menor = executa primeiro)."""
        ...
    
    @property
    def name(self) -> str:
        """Nome identificador do plugin."""
        ...


class StepRegistry:
    """
    Registro global de plugins de steps.
    
    Permite registrar plugins que serão automaticamente aplicados
    em todas as chamadas de `steps()`, a menos que sobrescritos.
    
    Examples:
    --------
    >>> # Registrar plugin globalmente
    >>> StepRegistry.register(MyCustomPlugin())
    >>> 
    >>> # Listar plugins ativos
    >>> plugins = StepRegistry.get_plugins()
    >>> 
    >>> # Remover plugin específico
    >>> StepRegistry.unregister('MyCustomPlugin')
    >>> 
    >>> # Limpar todos
    >>> StepRegistry.clear()
    """
    
    _plugins: List[StepPlugin] = []
    
    @classmethod
    def register(cls, plugin: StepPlugin) -> None:
        """
        Registra um novo plugin.
        
        Plugins são ordenados automaticamente por prioridade.
        
        Args:
            plugin: Instância do plugin
        """
        cls._plugins.append(plugin)
        cls._plugins.sort(key=lambda p: p.priority)
        logger.info(f"Plugin registrado: {plugin.name} (priority={plugin.priority})")
    
    @classmethod
    def unregister(cls, plugin_name: str) -> bool:
        """
        Remove um plugin pelo nome.
        
        Args:
            plugin_name: Nome do plugin
        
        Returns:
            True se removido, False se não encontrado
        """
        initial_count = len(cls._plugins)
        cls._plugins = [p for p in cls._plugins if p.name != plugin_name]
        removed = len(cls._plugins) < initial_count
        
        if removed:
            logger.info(f"Plugin removido: {plugin_name}")
        else:
            logger.warning(f"Plugin não encontrado: {plugin_name}")
        
        return removed
    
    @classmethod
    def get_plugins(cls) -> List[StepPlugin]:
        """Retorna cópia da lista de plugins registrados."""
        return cls._plugins.copy()
    
    @classmethod
    def clear(cls) -> None:
        """Remove todos os plugins registrados."""
        count = len(cls._plugins)
        cls._plugins.clear()
        logger.info(f"Registro de plugins limpo ({count} plugins removidos)")


# ============================================================================
# EQUATION CLASS
# ============================================================================

@dataclass
class Equation:
    """
    Symbolic equation with safe parsing, norm optimization, and detailed steps.
    
    Features:
    - Safe expression parsing (SymPy or AST fallback)
    - Automatic norm factor application
    - Step-by-step solution generation (PT-BR) with granularity control
    - LaTeX output with Greek symbols
    - Integration with recognition and text processor
    - Immutable design (functional updates)
    - Cache-aware evaluation
    - Hybrid substitution (xreplace for speed, subs for algebra)
    
    Attributes:
        expression: SymPy expression or string
        variables: Dictionary of Variable instances
        result: Cached numerical result
        description: Human-readable description
        norm_factors: Applied norm factors (metadata)
        _cache: Optional ResultCache instance
    
    Examples:
    --------
    >>> from pymemorial.core import Equation, Variable
    >>> 
    >>> # Create variables
    >>> M_k = Variable('M_k', value=150.0, unit='kN.m')
    >>> gamma_s = Variable('gamma_s', value=1.4, unit='')
    >>> 
    >>> # Create equation
    >>> eq = Equation(
    ...     expression='M_d = M_k * gamma_s',
    ...     variables={'M_k': M_k, 'gamma_s': gamma_s},
    ...     description='Design moment calculation'
    ... )
    >>> 
    >>> # Evaluate
    >>> result = eq.evaluate()
    >>> print(result)  # 210.0
    >>> 
    >>> # Get LaTeX
    >>> latex = eq.latex()
    >>> print(latex)  # "$M_d = M_k \\cdot \\gamma_s$"
    >>> 
    >>> # Get solution steps with granularity
    >>> steps = eq.steps(granularity='detailed')
    >>> for step in steps:
    ...     print(step)
    """
    
    expression: Union[sp.Expr, str, Any]
    variables: Dict[str, 'Variable'] = field(default_factory=dict)
    result: Optional[float] = None
    description: str = ""
    norm_factors: Dict[str, float] = field(default_factory=dict)
    _cache: Optional['ResultCache'] = field(default=None, repr=False)
    
    def __post_init__(self):
        """
        Initialize equation and parse expression.
        
        Raises:
            ValueError: If expression is invalid or variables are missing
            ImportError: If SymPy is not available (required)
        """
        if not SYMPY_AVAILABLE:
            raise ImportError(
                "SymPy is required for Equation. Install via: pip install sympy"
            )
        
        # Validate variables
        self._validate_variables()
        
        # Parse expression
        if isinstance(self.expression, str):
            self.expression = self._parse_expression(self.expression)
        
        logger.debug(
            f"Equation initialized: {len(self.variables)} variables, "
            f"description='{self.description[:50]}...'"
        )
    
    def _validate_variables(self) -> None:
        """
        Validate variables dictionary.
        
        Raises:
            ValueError: If variables is not a dict or contains invalid entries
        """
        if not isinstance(self.variables, dict):
            raise ValueError(f"variables must be dict, got {type(self.variables)}")
        
        for name, var in self.variables.items():
            if not hasattr(var, 'symbol'):
                raise ValueError(
                    f"Variable '{name}' must have 'symbol' attribute "
                    f"(got {type(var)})"
                )
    
    def _parse_expression(self, expr_str: str) -> sp.Expr:
        """
        Parse expression string to SymPy expression.
        
        Supports:
        - Standard format: "expr"
        - Equation format: "lhs = rhs" (extracts rhs)
        
        Args:
            expr_str: Expression string
        
        Returns:
            SymPy expression
        
        Raises:
            ValueError: If parsing fails (PT-BR message)
        """
        # Extract RHS if equation format
        if '=' in expr_str:
            parts = expr_str.split('=', 1)
            if len(parts) == 2:
                expr_str = parts[1].strip()
                logger.debug(f"Extracted RHS from equation: '{expr_str}'")
        
        # Build local dictionary with symbols
        local_dict = {name: var.symbol for name, var in self.variables.items()}
        
        try:
            # Primary method: SymPy sympify
            expr = sp.sympify(expr_str, locals=local_dict, evaluate=False)
            logger.debug(f"Parsed expression using sympify: {expr}")
            return expr
        
        except Exception as e:
            logger.warning(f"SymPy parse failed: {e}. Trying AST fallback.")
            
            try:
                # Fallback: Safe AST parse
                expr = self._safe_ast_parse(expr_str, local_dict)
                logger.debug(f"Parsed expression using AST fallback: {expr}")
                return expr
            
            except Exception as e2:
                # ✅ MENSAGEM EM PORTUGUÊS
                raise ValueError(
                    f"Erro ao converter expressão '{expr_str}': "
                    f"Erro SymPy: {e}, Erro AST: {e2}"
                ) from e2
    
    def _safe_ast_parse(self, expr_str: str, local_dict: Dict) -> Any:
        """
        Safe expression parsing using AST whitelist.
        
        Args:
            expr_str: Expression string
            local_dict: Local symbol dictionary
        
        Returns:
            Parsed expression (SymPy-like)
        
        Raises:
            ValueError: If expression contains unsafe operations
        """
        # Whitelist of allowed AST nodes
        ALLOWED_NODES = (
            ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Name,
            ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd,
            ast.Mod, ast.FloorDiv, ast.Load
        )
        
        # Parse AST
        tree = ast.parse(expr_str, mode='eval')
        
        # Validate all nodes are safe
        for node in ast.walk(tree):
            if not isinstance(node, ALLOWED_NODES):
                raise ValueError(
                    f"Operação não segura na expressão: {node.__class__.__name__}"
                )
        
        # Compile and evaluate
        code = compile(tree, '<string>', 'eval')
        result = eval(code, {"__builtins__": {}}, local_dict)
        
        return result
    
    # ========================================================================
    # EXPRESSION MANIPULATION (HYBRID APPROACH)
    # ========================================================================
    
    def subs(self, *args, **kwargs) -> sp.Expr:
        """
        SymPy-style substitution (algebraic/mathematical).
        
        ⚡ Uses `subs()` for ALGEBRAIC substitution (slower but smarter).
        For pure speed, use `xreplace()` instead.
        
        Supports multiple call styles:
        1. subs({symbol: value})         - dict
        2. subs(symbol, value)           - positional args
        3. subs(x=1, y=2)                - kwargs
        
        Args:
            *args: Either dict or (symbol, value) pairs
            **kwargs: Alternative syntax (x=1, y=2)
        
        Returns:
            SymPy expression with substitutions applied
        
        Raises:
            KeyError: If variable not found
            ValueError: If substitution fails
        
        Examples:
        --------
        >>> # Algebraic substitution (simplifies)
        >>> result = eq.subs(x=2)  # x**2 becomes 4
        
        >>> # For speed, use xreplace() instead
        >>> result = eq.xreplace({x_sym: 2})  # x**2 stays x**2 with x=2
        """
        # Parse arguments
        if args and len(args) == 1 and isinstance(args[0], dict):
            # Style 1: dict
            substitutions = args[0]
        elif args and len(args) == 2:
            # Style 2: positional (symbol, value)
            substitutions = {args[0]: args[1]}
        elif kwargs:
            # Style 3: kwargs - ✅ VALIDATE BEFORE try-except
            subs_dict = {}
            for name, value in kwargs.items():
                if name in self.variables:
                    subs_dict[self.variables[name].symbol] = value
                else:
                    # Try to find in free_symbols
                    symbols_map = {str(s): s for s in self.expression.free_symbols}
                    if name in symbols_map:
                        subs_dict[symbols_map[name]] = value
                    else:
                        # ✅ RAISE KeyError IMMEDIATELY
                        raise KeyError(
                            f"Variável '{name}' não encontrada. "
                            f"Disponíveis: {list(self.variables.keys())}"
                        )
            substitutions = subs_dict
        elif args and len(args) > 2:
            # Multiple pairs: subs(x, 1, y, 2)
            if len(args) % 2 != 0:
                raise ValueError("subs() requer pares de (símbolo, valor)")
            substitutions = {args[i]: args[i+1] for i in range(0, len(args), 2)}
        else:
            raise ValueError("Chamada inválida de subs(). Use subs(dict), subs(symbol, value) ou subs(x=1, y=2)")
        
        # Apply substitution (now can only raise ValueError for other reasons)
        try:
            return self.expression.subs(substitutions)
        except Exception as e:
            raise ValueError(f"Substituição falhou: {e}") from e
    
    def xreplace(self, substitutions: Dict) -> sp.Expr:
        """
        Literal (syntactic) substitution - FASTER and MORE ROBUST.
        
        ⚡ PERFORMANCE: ~10x faster than subs() for large expressions!
        
        Unlike `subs()`, this performs **direct symbol replacement**
        without algebraic simplifications. Perfect for:
        - Step-by-step calculations
        - Numerical evaluation
        - Large expression trees
        
        Args:
            substitutions: Dict {symbol: value}
        
        Returns:
            SymPy expression with literal substitutions
        
        Examples:
        --------
        >>> # Fast literal substitution (no simplification)
        >>> eq = Equation("x**2 + 2*x", {"x": Variable("x", 3)})
        >>> result = eq.xreplace({eq.variables['x'].symbol: 3})
        >>> print(result)  # 3**2 + 2*3 (not simplified to 15)
        
        >>> # Use evaluate() to get final number
        >>> print(result.evalf())  # 15.0
        """
        try:
            return self.expression.xreplace(substitutions)
        except Exception as e:
            raise ValueError(f"xreplace falhou: {e}") from e
    
    def substitute(self, **kwargs) -> 'Equation':
        """
        Substitute variable values in expression (functional update).
        
        ⚡ Uses `xreplace()` for SPEED (literal substitution).
        
        Returns a NEW Equation instance (immutable design).
        
        Args:
            **kwargs: Variable name -> value mapping
        
        Returns:
            New Equation with substituted values
        
        Raises:
            KeyError: If variable not found
        
        Examples:
        --------
        >>> eq2 = eq.substitute(M_k=200.0)
        >>> print(eq2.expression)  # M_d = 200.0 * gamma_s
        """
        # ✅ VALIDATE variables exist BEFORE substitution
        for name in kwargs.keys():
            if name not in self.variables:
                raise KeyError(
                    f"Variável '{name}' não encontrada. "
                    f"Disponíveis: {list(self.variables.keys())}"
                )
        
        # Build substitution dictionary
        subs_dict = {}
        for name, var in self.variables.items():
            if name in kwargs:
                subs_dict[var.symbol] = kwargs[name]
            else:
                value = var.value.magnitude if hasattr(var.value, 'magnitude') else var.value
                if value is not None:
                    subs_dict[var.symbol] = value
        
        # ⚡ Use xreplace for speed (literal substitution)
        new_expr = self.expression.xreplace(subs_dict)
        
        # Create new Equation instance
        return Equation(
            expression=new_expr,
            variables=deepcopy(self.variables),
            result=None,  # Invalidate cached result
            description=self.description,
            norm_factors=deepcopy(self.norm_factors),
            _cache=self._cache
        )
    
    def simplify(self) -> Union[sp.Expr, int, float]:
        """
        Simplify expression algebraically.
        
        Returns:
            Simplified SymPy expression or numeric value
        
        Examples:
        --------
        >>> simplified = eq.simplify()
        >>> print(simplified)  # Simplified form (sp.Expr or number)
        """
        simplified_expr = sp.simplify(self.expression)
        
        # Return direct expression, not wrapped in Equation
        return simplified_expr
    
    # ========================================================================
    # EVALUATION
    # ========================================================================
    
# src/pymemorial/core/equation.py

    # ... (código anterior) ...

    # ========================================================================
    # EVALUATION
    # ========================================================================
    
    def _needs_integer_conversion(self) -> bool:
        """
        Detect if expression requires integers (factorial, binomial, etc.).
        ✅ EXCLUDES floor/ceiling - they MUST receive floats!
        Returns:
            True if expression contains functions requiring integers
        """
        # Verifica se a expressão é um objeto SymPy antes de converter para string
        if isinstance(self.expression, sp.Expr):
            expr_str = str(self.expression).lower()
            integer_functions = ['factorial', 'binomial']  # ✅ ONLY these need int
            return any(func in expr_str for func in integer_functions)
        return False # Não aplicável se não for expressão SymPy

    def evaluate(self, use_cache: bool = True, auto_convert_units: bool = True) -> float:
        """
        Evaluate expression numerically using SymPy's robust evalf(subs=...).
        Assumes Variable.value already holds numerical value in SI base units.

        Args:
            use_cache: Use cached result if available
            auto_convert_units: (Mainly handled by Variable, kept for signature consistency)

        Returns:
            Numerical result as a standard Python float.

        Raises:
            ValueError: If variables are missing values or evaluation fails.
        """
        # Ensure SymPy is available
        if not SYMPY_AVAILABLE:
            raise ImportError("SymPy is required for evaluation.")

        # Ensure expression is a SymPy expression
        if not isinstance(self.expression, sp.Expr):
             try:
                 self.expression = self._parse_expression(str(self.expression))
             except Exception as parse_error:
                 raise ValueError(f"Expression is not a valid SymPy object and parsing failed: {parse_error}") from parse_error

        # Check cache first
        cache_key = str(self.expression)
        if use_cache:
             if self.result is not None:
                  logger.debug(f"Using instance cached result: {self.result}")
                  return self.result
             if self._cache:
                  cached = self._cache.get(cache_key)
                  if cached is not None:
                       logger.debug(f"Using external cache result: {cached}")
                       self.result = cached
                       return cached

        # Prepare substitution dictionary with numerical values from var.value
        used_symbols = self.expression.free_symbols
        missing_vars = []
        subs_dict = {}

        for var_name, var in self.variables.items():
            if var.symbol in used_symbols:
                # --- ✅ SIMPLIFICAÇÃO AQUI ---
                # Assume var.value é None ou um número (float/int/ndarray) em unidade base
                value = var.value
                # --- FIM DA SIMPLIFICAÇÃO ---

                if value is None:
                    missing_vars.append(var_name)
                else:
                    try:
                       # Garante que é float ou int (ou deixa ndarray passar)
                       if NUMPY_AVAILABLE and isinstance(value, np.ndarray):
                            numeric_value = value # Mantém ndarray
                       else:
                            numeric_value = float(value) # Converte para float

                       # Convert to int ONLY for specific functions
                       if self._needs_integer_conversion():
                           # Verifica se o float pode ser convertido para int sem perda
                           if isinstance(numeric_value, float) and numeric_value.is_integer():
                                numeric_value = int(numeric_value)
                           # Não faz nada se já for int ou se não for inteiro

                       subs_dict[var.symbol] = numeric_value

                    except (TypeError, ValueError) as e:
                       logger.error(f"Could not use value for '{var_name}' ({value}): {e}")
                       missing_vars.append(f"{var_name}(valor inválido: {value})")
                       continue

        # Check for symbols in expression not covered by provided variables
        for sym in used_symbols:
            # Verifica se o símbolo está faltando no dicionário de substituição E
            # não é uma variável conhecida (com valor None, já tratada) E
            # não é um símbolo especial (x,y,z) E
            # não é uma função/constante SymPy conhecida
            is_known_sympy = hasattr(sp, str(sym))
            is_integration_var = str(sym) in ['x', 'y', 'z']
            is_missing_var = sym not in subs_dict and str(sym) not in self.variables
            
            if is_missing_var and not is_integration_var and not is_known_sympy:
                missing_vars.append(f"{str(sym)}(não definida)")

        # Raise error if any required variables are missing values
        if missing_vars:
            raise ValueError(f"Variáveis sem valor ou não definidas na expressão: {', '.join(sorted(list(set(missing_vars))))}")

        # Perform evaluation using SymPy's evalf with substitutions
        try:
            logger.debug(f"Evaluating expression: {self.expression}")
            logger.debug(f"With substitutions: {subs_dict}")

            result_sympy = self.expression.evalf(subs=subs_dict)

            if not result_sympy.is_number:
                # Se ainda não for número, tenta simplificar e avaliar de novo (pode ajudar em alguns casos)
                simplified_result = sp.simplify(result_sympy)
                if simplified_result.is_number:
                     result_sympy = simplified_result
                else:
                     raise ValueError(f"A avaliação resultou em uma expressão simbólica não resolvida: {result_sympy}")

            result = float(result_sympy)

            # Cache result
            self.result = result
            if self._cache:
                 self._cache.set(cache_key, result)

            logger.debug(f"Evaluation successful: result={result}")
            return result

        except Exception as e:
            logger.error(f"SymPy evaluation failed for expression '{self.expression}' with subs {subs_dict}: {e}")
            logger.debug(traceback.format_exc())
            raise ValueError(f"Erro ao avaliar expressão '{self.expression}': {e}") from e



    
    # ========================================================================
    # NORM OPTIMIZATION
    # ========================================================================
    
    def optimize_norm(
        self,
        norm_code: str = 'NBR6118_2023',
        auto_detect: bool = True
    ) -> 'Equation':
        """
        Apply norm factors to equation.
        
        Returns a NEW Equation instance (immutable design).
        
        Args:
            norm_code: Norm code (e.g., 'NBR6118_2023')
            auto_detect: Auto-detect safety factors via recognition
        
        Returns:
            New Equation with norm factors applied
        
        Examples:
        --------
        >>> eq2 = eq.optimize_norm('NBR6118_2023')
        >>> print(eq2.norm_factors)  # {'gamma_s': 1.4}
        """
        if not RECOGNITION_AVAILABLE or not auto_detect:
            logger.debug("Norm optimization skipped (recognition unavailable or disabled)")
            return self
        
        try:
            nlp = EngineeringNLP()
            applied_factors = {}
            
            # Check each variable for safety factor
            for var_name, var in self.variables.items():
                if 'gamma' in var_name.lower() or 'factor' in var_name.lower():
                    # Infer type using recognition
                    detected = DetectedVar(name=var_name, base=var_name, subscript='')
                    inferred_type = nlp.infer_type(detected, self.description)
                    
                    if 'safety' in inferred_type or 'factor' in inferred_type:
                        # Get norm factor
                        factor = self._get_norm_factor(norm_code, 'safety_factor')
                        applied_factors[var_name] = factor
                        
                        logger.info(
                            f"Norm factor {factor} applied to '{var_name}' "
                            f"(norm: {norm_code})"
                        )
            
            # Create new Equation with updated norm_factors
            return Equation(
                expression=self.expression,
                variables=deepcopy(self.variables),
                result=None,
                description=f"{self.description} (norm: {norm_code})",
                norm_factors={**self.norm_factors, **applied_factors},
                _cache=self._cache
            )
        
        except Exception as e:
            logger.warning(f"Norm optimization failed: {e}")
            return self
    
    def _get_norm_factor(self, norm_code: str, factor_type: str) -> float:
        """
        Get norm factor from standards module or fallback.
        
        Args:
            norm_code: Norm code
            factor_type: Factor type
        
        Returns:
            Norm factor value
        """
        if STANDARDS_AVAILABLE:
            return get_norm_factor(norm_code, factor_type)
        else:
            return DEFAULT_NORM_FACTORS.get(norm_code, {}).get(factor_type, 1.0)
    
    # ========================================================================
    # SOLUTION STEPS (WITH GRANULARITY CONTROL)
    # ========================================================================
    
    def _operation_description(self, operation: str) -> str:
        """
        Get Portuguese description for operation type.
        
        Args:
            operation: Operation type
        
        Returns:
            Portuguese description
        """
        descriptions = {
            'symbolic': 'Expressão simbólica',
            'substitution': 'Substituição de valores',
            'simplification': 'Simplificação algébrica',
            'evaluation': 'Avaliação numérica',
            'result': 'Resultado final',
            'intermediate': 'Passo intermediário',
        }
        return descriptions.get(operation, 'Passo de cálculo')
    
    def steps(
        self,
        detail: str = 'basic',
        granularity: Optional[GranularityType] = None,
        show_units: bool = True,
        max_steps: Optional[int] = None,
        plugins: Optional[List[StepPlugin]] = None,
        custom_formatters: Optional[Dict[str, Callable]] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate detailed solution steps (PT-BR) with granularity control.

        🎯 GRANULARITY LEVELS (MAPEADOS AUTOMATICAMENTE):
        - 'minimal'/'numeric'/'result': Simbólico + resultado (2 steps)
        - 'basic': Simbólico + substituição + resultado (3 steps)
        - 'normal': Basic + avaliação + simplificação (5+ steps)
        - 'detailed': Normal + operações detalhadas (7+ steps)
        - 'all'/'full': Máximo detalhe com análise (10+ steps)
        - 'smart': Auto-detecta complexidade (padrão)

        📝 ALIASES ACEITOS:
        - 'numeric', 'result' → 'minimal'
        - 'full' → 'all'
        - Qualquer outro valor não mapeado → 'smart'
        """

        # ✅ MAPEAMENTO ROBUSTO DE ALIASES
        granularity_aliases = {
            # Aliases para minimal (resultado direto)
            'numeric': 'minimal',
            'result': 'minimal',
            'minimal': 'minimal',

            # Aliases para basic
            'basic': 'basic',
            'simple': 'basic',

            # Aliases para normal
            'normal': 'normal',
            'medium': 'normal',

            # Aliases para detailed
            'detailed': 'detailed',
            'verbose': 'detailed',

            # Aliases para all
            'all': 'all',
            'full': 'all',
            'maximum': 'all',

            # Smart (auto-detect)
            'smart': 'smart',
            'auto': 'smart',
        }

        # Handle legacy 'detail' parameter
        if granularity is None:
            # Map old 'detail' values to new granularity
            detail_to_granularity = {
                'basic': 'basic',
                'detailed': 'detailed',
                'full': 'all',
                'simple': 'basic',
                'normal': 'normal'
            }
            granularity = detail_to_granularity.get(detail, 'smart')

        # ✅ APLICAR MAPEAMENTO DE ALIASES (case-insensitive)
        granularity_lower = str(granularity).lower().strip()
        granularity = granularity_aliases.get(granularity_lower, 'smart')

        # Validate granularity (agora aceita apenas valores válidos pós-mapeamento)
        valid_levels = {'minimal', 'basic', 'normal', 'detailed', 'all', 'smart'}
        if granularity not in valid_levels:
            logger.warning(
                f"Granularidade '{granularity_lower}' não reconhecida após mapeamento. "
                f"Usando 'smart' como fallback."
            )
            granularity = 'smart'

        # Auto-detect granularity if 'smart'
        if granularity == 'smart':
            try:
                complexity = len(self.expression.free_symbols) + len(self.expression.args)
                if complexity <= 2:
                    granularity = 'basic'
                elif complexity <= 4:
                    granularity = 'normal'
                elif complexity <= 8:
                    granularity = 'detailed'
                else:
                    granularity = 'all'
                logger.debug(f"Smart granularity selected: {granularity} (complexity={complexity})")
            except Exception as e:
                logger.warning(f"Failed to auto-detect complexity: {e}. Using 'normal'.")
                granularity = 'normal'

        steps = []
        expr = self.expression

        try:
            # Step 1: SYMBOLIC (always included)
            steps.append({
                'step': 'Expressão simbólica',
                'operation': 'symbolic',
                'expr': sp.latex(expr),
                'numeric': None,
                'description': self.description or 'Forma simbólica da equação'
            })

            # Calculate result if not cached
            result_value = self.result
            if result_value is None:
                try:
                    result_value = self.evaluate(use_cache=False)
                except Exception as e:
                    logger.warning(f"Could not evaluate result for steps: {e}")
                    result_value = None

            # For 'minimal', skip to result immediately
            if granularity == 'minimal':
                if result_value is not None:
                    steps.append({
                        'step': 'Resultado final',
                        'operation': 'result',
                        'expr': f"{sp.latex(expr)} = {result_value:.6g}",
                        'numeric': result_value,
                        'description': 'Resultado final'
                    })
                steps = self._apply_plugins_and_formatters(
                    steps, result_value, plugins, custom_formatters
                )
                return steps[:max_steps] if max_steps else steps

            # Step 2: SUBSTITUTION (basic+)
            if granularity in ('basic', 'normal', 'detailed', 'all'):
                subs_dict = {}
                for var_name, var in self.variables.items():
                    if var.value is not None:
                        value = var.value.magnitude if hasattr(var.value, 'magnitude') else var.value
                        subs_dict[var.symbol] = value

                if subs_dict:
                    substituted = self.expression.xreplace(subs_dict)
                    if isinstance(substituted, (int, float)):
                        substituted = sp.sympify(substituted)

                    steps.append({
                        'step': 'Substituição de valores',
                        'operation': 'substitution',
                        'expr': sp.latex(substituted),
                        'numeric': None,
                        'description': 'Valores numéricos substituídos'
                    })
                    expr = substituted

            # Step 3: EVALUATION (normal+)
            if granularity in ('normal', 'detailed', 'all'):
                try:
                    if hasattr(expr, 'evalf'):
                        evaluated = expr.evalf()
                        if evaluated != expr and evaluated.is_number:
                            steps.append({
                                'step': 'Avaliação numérica',
                                'operation': 'evaluation',
                                'expr': sp.latex(evaluated),
                                'numeric': float(evaluated),
                                'description': 'Cálculo numérico das operações'
                            })
                            expr = evaluated
                except:
                    pass

            # Step 4: SIMPLIFICATION (normal+)
            if granularity in ('normal', 'detailed', 'all'):
                try:
                    simplified = sp.simplify(expr)
                    if simplified != expr:
                        steps.append({
                            'step': 'Simplificação algébrica',
                            'operation': 'simplification',
                            'expr': sp.latex(simplified),
                            'numeric': None,
                            'description': 'Expressão simplificada'
                        })
                        expr = simplified
                except:
                    pass

            # Step 5: INTERMEDIATE OPERATIONS (detailed+)
            if granularity in ('detailed', 'all'):
                if expr.is_Add:
                    steps.append({
                        'step': 'Operação identificada',
                        'operation': 'intermediate',
                        'expr': sp.latex(expr),
                        'numeric': None,
                        'description': 'Operação: Adição/Subtração'
                    })
                elif expr.is_Mul:
                    steps.append({
                        'step': 'Operação identificada',
                        'operation': 'intermediate',
                        'expr': sp.latex(expr),
                        'numeric': None,
                        'description': 'Operação: Multiplicação/Divisão'
                    })
                elif expr.is_Pow:
                    steps.append({
                        'step': 'Operação identificada',
                        'operation': 'intermediate',
                        'expr': sp.latex(expr),
                        'numeric': None,
                        'description': 'Operação: Potenciação'
                    })

            # Step 6: COMPLEXITY ANALYSIS (all only)
            if granularity == 'all':
                n_symbols = len(expr.free_symbols)
                n_ops = len(expr.args) if hasattr(expr, 'args') else 1
                complexity = n_symbols + n_ops

                steps.append({
                    'step': 'Análise de complexidade',
                    'operation': 'intermediate',
                    'expr': sp.latex(expr),
                    'numeric': complexity,
                    'description': f'Complexidade: {complexity} (símbolos: {n_symbols}, operações: {n_ops})'
                })

            # Step FINAL: RESULT (ALWAYS added if we have a result)
            if result_value is not None:
                steps.append({
                    'step': 'Resultado final',
                    'operation': 'result',
                    'expr': f"{sp.latex(self.expression)} = {result_value:.6g}",
                    'numeric': result_value,
                    'description': 'Resultado final'
                })

            logger.debug(f"Generated {len(steps)} solution steps (granularity={granularity})")

        except Exception as e:
            logger.warning(f"Failed to generate steps: {e}")
            steps.append({
                'step': 'Erro',
                'operation': 'error',
                'expr': '',
                'numeric': None,
                'description': f"Não foi possível gerar passos: {e}"
            })

        # Apply plugins and formatters
        steps = self._apply_plugins_and_formatters(
            steps, result_value, plugins, custom_formatters
        )

        # Apply max_steps limit
        return steps[:max_steps] if max_steps else steps

    
    def _apply_plugins_and_formatters(
        self,
        steps: List[Dict[str, Any]],
        result_value: Optional[float],
        plugins: Optional[List[StepPlugin]],
        custom_formatters: Optional[Dict[str, Callable]]
    ) -> List[Dict[str, Any]]:
        """
        Apply custom plugins and formatters to steps.
        
        Args:
            steps: List of step dictionaries
            result_value: Calculated result value
            plugins: Custom step plugins
            custom_formatters: Custom formatters
        
        Returns:
            Modified steps list
        """
        # Apply custom plugins
        active_plugins = plugins if plugins is not None else StepRegistry.get_plugins()
        
        if active_plugins:
            context = {
                'original_expression': self.expression,
                'variables': self.variables,
                'result': result_value,
                'existing_steps': len(steps)
            }
            
            for plugin in active_plugins:
                try:
                    plugin_step = plugin.process(self.expression, self.variables, context)
                    if plugin_step:
                        steps.append(plugin_step)
                        logger.debug(f"Plugin applied: {plugin.name}")
                except Exception as e:
                    logger.warning(f"Plugin {plugin.name} failed: {e}")
        
        # Apply custom formatters
        if custom_formatters:
            for step in steps:
                operation = step.get('operation')
                if operation in custom_formatters:
                    try:
                        step['formatted'] = custom_formatters[operation](step)
                    except Exception as e:
                        logger.warning(f"Formatter for '{operation}' failed: {e}")
        
        return steps

    
    # ========================================================================
    # LATEX OUTPUT
    # ========================================================================
    
    def latex(self) -> str:
        """
        Convert expression to LaTeX format.
        
        Integrates with text processor for Greek symbol conversion.
        
        Returns:
            LaTeX string
        
        Examples:
        --------
        >>> latex = eq.latex()
        >>> print(latex)  # "$M_d = M_k \\cdot \\gamma_s$"
        """
        try:
            # Primary method: SymPy latex
            latex_str = sp.latex(self.expression)
            
            # Enhancement: Use text processor for Greek symbols
            if RECOGNITION_AVAILABLE:
                try:
                    engine = get_engine()
                    latex_str = engine.to_latex(str(self.expression))
                except Exception as e:
                    logger.debug(f"Text processor enhancement failed: {e}")
            
            return f"${latex_str}$"
        
        except Exception as e:
            logger.warning(f"LaTeX generation failed: {e}")
            return str(self.expression)
    
    # ========================================================================
    # REPRESENTATION
    # ========================================================================
    
    def __repr__(self) -> str:
        """String representation."""
        vars_info = f"{len(self.variables)} vars" if self.variables else "no vars"
        result_info = f", result={self.result:.6g}" if self.result is not None else ""
        return f"Equation({self.expression}, {vars_info}{result_info})"
    
    def __str__(self) -> str:
        """Human-readable string."""
        return f"{self.expression} = {self.result:.6g}" if self.result else str(self.expression)


# ============================================================================
# FACTORY
# ============================================================================

class EquationFactory:
    """
    Factory for creating Equation instances.
    
    Examples:
    --------
    >>> factory = EquationFactory()
    >>> eq = factory.create('M_d = M_k * gamma_s', variables={...})
    """
    
    @staticmethod
    def create(
        expression: str,
        variables: Optional[Dict[str, 'Variable']] = None,
        description: str = "",
        cache: Optional['ResultCache'] = None
    ) -> Equation:
        """
        Create Equation instance.
        
        Args:
            expression: Expression string
            variables: Variable dictionary
            description: Human-readable description
            cache: Optional ResultCache instance
        
        Returns:
            Equation instance
        """
        return Equation(
            expression=expression,
            variables=variables or {},
            description=description,
            _cache=cache
        )


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'Equation',
    'EquationFactory',
    'StepPlugin',     # Protocolo
    'StepRegistry',   # Registro Global - ESSENCIAL ESTAR AQUI
    'GranularityType' # Exporta o tipo Literal também
]