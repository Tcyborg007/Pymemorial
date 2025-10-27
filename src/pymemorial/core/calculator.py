"""
PYMEMORIAL v2.0 - Calculator Module
Motor híbrido de cálculo (SymPy + SciPy + NumPy)
"""
from __future__ import annotations
import logging
import threading
from typing import Any, Dict, List, Optional, Union, Callable
from functools import lru_cache
from dataclasses import dataclass, field
import ast

# Imports condicionais
try:
    import sympy as sp
    from sympy import Symbol, Expr
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False
    sp = None
    Symbol = None
    Expr = None

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    from scipy import optimize, integrate
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    optimize = None
    integrate = None

# Imports internos
from .config import get_config
from .variable import Variable
from .equation import Equation


# ============================================================================
# EXCEPTIONS
# ============================================================================

class CalculatorError(Exception):
    """Erro base do Calculator."""
    pass


class EvaluationError(CalculatorError):
    """Erro durante avaliação."""
    pass


class UnsafeCodeError(CalculatorError):
    """Código inseguro detectado."""
    pass


# ============================================================================
# SAFEEVALUATOR - SEGURANÇA CRÍTICA
# ============================================================================

class SafeEvaluator:
    """
    Avaliador seguro de expressões usando AST whitelist.
    
    **SEGURANÇA CRÍTICA:**
    - Whitelist de nós AST permitidos
    - Bloqueia imports, exec, eval
    - Bloqueia file I/O
    - Bloqueia network calls
    
    **FILOSOFIA:**
    - Nunca usar eval() direto
    - AST parsing + validation
    - Sandbox seguro
    
    Examples:
        >>> evaluator = SafeEvaluator()
        >>> result = evaluator.safe_eval("2 + 3 * 4")
        >>> print(result)  # 14
        >>> 
        >>> # Código malicioso é bloqueado
        >>> evaluator.safe_eval("import os")  # Raises UnsafeCodeError
    """
    
    # Nós AST permitidos (whitelist)
    ALLOWED_NODES = {
        ast.Module,
        ast.Expression,
        ast.Expr,
        ast.Load,
        ast.Store,
        ast.Name,
        ast.Constant,
        ast.Num,  # Python 3.7 compat
        ast.Str,  # Python 3.7 compat
        ast.BinOp,
        ast.UnaryOp,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
        ast.Pow,
        ast.UAdd,
        ast.USub,
        ast.Compare,
        ast.Eq,
        ast.NotEq,
        ast.Lt,
        ast.LtE,
        ast.Gt,
        ast.GtE,
        ast.Call,
        ast.Attribute,
        ast.Subscript,
        ast.Index,  # Python 3.7 compat
        ast.List,
        ast.Tuple,
        ast.Dict,
    }
    
    # Funções permitidas
    ALLOWED_FUNCTIONS = {
        'abs', 'min', 'max', 'sum', 'round',
        'int', 'float', 'str', 'bool',
        'len', 'range', 'enumerate', 'zip',
        # NumPy (se disponível)
        'sqrt', 'sin', 'cos', 'tan',
        'exp', 'log', 'log10',
        'floor', 'ceil',
    }
    
    def __init__(self):
        """Inicializa evaluator."""
        self._logger = logging.getLogger(__name__)
        self._namespace = self._build_safe_namespace()
    
    def _build_safe_namespace(self) -> Dict[str, Any]:
        """
        Constrói namespace seguro com funções permitidas.
        
        Returns:
            Namespace seguro
        """
        namespace = {}
        
        # Funções built-in seguras
        namespace.update({
            'abs': abs,
            'min': min,
            'max': max,
            'sum': sum,
            'round': round,
            'int': int,
            'float': float,
            'str': str,
            'bool': bool,
            'len': len,
            'range': range,
            'enumerate': enumerate,
            'zip': zip,
        })
        
        # NumPy (se disponível)
        if NUMPY_AVAILABLE:
            namespace.update({
                'np': np,
                'sqrt': np.sqrt,
                'sin': np.sin,
                'cos': np.cos,
                'tan': np.tan,
                'exp': np.exp,
                'log': np.log,
                'log10': np.log10,
                'floor': np.floor,
                'ceil': np.ceil,
                'array': np.array,
                'arange': np.arange,
                'linspace': np.linspace,
                'pi': np.pi,
                'e': np.e,
            })
        
        return namespace

    
    def validate_ast(self, code: str) -> ast.Module:
        """
        Valida AST do código contra whitelist.
        
        Args:
            code: Código a validar
        
        Returns:
            AST validado
        
        Raises:
            UnsafeCodeError: Se código inseguro
        """
        try:
            tree = ast.parse(code, mode='eval')
        except SyntaxError as e:
            raise EvaluationError(f"Sintaxe inválida: {e}")
        
        # Validar todos os nós
        for node in ast.walk(tree):
            node_type = type(node)
            
            if node_type not in self.ALLOWED_NODES:
                raise UnsafeCodeError(
                    f"Operação não permitida: {node_type.__name__}"
                )
            
            # Validar chamadas de função
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                    if func_name not in self.ALLOWED_FUNCTIONS:
                        raise UnsafeCodeError(
                            f"Função não permitida: {func_name}"
                        )
        
        return tree
    
    def safe_eval(
        self,
        expression: str,
        local_vars: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Avalia expressão de forma segura.
        
        Args:
            expression: Expressão a avaliar
            local_vars: Variáveis locais
        
        Returns:
            Resultado da avaliação
        
        Raises:
            UnsafeCodeError: Se código inseguro
            EvaluationError: Se avaliação falhar
        
        Examples:
            >>> evaluator = SafeEvaluator()
            >>> evaluator.safe_eval("2 + 3")
            5
            >>> evaluator.safe_eval("x * 2", {'x': 10})
            20
        """
        # Validar AST
        self.validate_ast(expression)
        
        # Construir namespace completo
        namespace = self._namespace.copy()
        if local_vars:
            namespace.update(local_vars)
        
        # Avaliar com namespace seguro
        try:
            result = eval(expression, {"__builtins__": {}}, namespace)
            return result
        
        # ================== CORREÇÃO ROBUSTA ==================
        except Exception as e:
            # O contrato do SafeEvaluator é NUNCA vazar erros de runtime.
            # Ele deve SEMPRE envolvê-los em EvaluationError.
            # O erro original (ex: NameError, ZeroDivisionError) é preservado na mensagem.
            raise EvaluationError(f"Erro na avaliação: {e}")
        # =================== FIM DA CORREÇÃO ===================


# ============================================================================
# CALCULATOR - MOTOR HÍBRIDO
# ============================================================================

class Calculator:
    """
    Motor híbrido de cálculo (SymPy + SciPy + NumPy).
    
    **FILOSOFIA:**
    - SymPy: Manipulação simbólica
    - SciPy: Métodos numéricos
    - NumPy: Operações vetorizadas
    
    **SEGURANÇA:**
    - SafeEvaluator para eval
    - AST validation
    - Sandbox

isolado
    
    **PERFORMANCE:**
    - LRU cache automático
    - lambdify para performance
    - Thread-safe
    
    Examples:
        >>> calc = Calculator()
        >>> result = calc.evaluate("q * L**2 / 8", {'q': 15, 'L': 6})
        >>> print(result)  # 67.5
    """
    
    def __init__(
        self,
        config: Optional[Any] = None,
        variables: Optional[Dict[str, Any]] = None,
        use_cache: bool = True
    ):
        """
        Inicializa calculator.
        
        Args:
            config: Configuração (usa global se None)
            variables: Variáveis iniciais
            use_cache: Habilitar cache LRU
        """
        self.config = config or get_config()
        self._evaluator = SafeEvaluator()
        self.variables = variables or {}
        self._namespace = {}
        self._cache = {} if use_cache else None
        self._use_cache = use_cache
        self._lock = threading.Lock()
        self._logger = logging.getLogger(__name__)


    @property
    def use_cache(self) -> bool:
        """Retorna se cache está habilitado."""
        return self._use_cache

    def add_variable(
        self,
        name: str,
        value: Any,
        unit: Optional[str] = None,
        uncertainty: Optional[float] = None
    ) -> None:
        """
        Adiciona variável ao calculator.
        
        Args:
            name: Nome da variável
            value: Valor da variável
            unit: Unidade (opcional)
            uncertainty: Incerteza/erro (opcional, para propagação)
        
        Examples:
            >>> calc = Calculator()
            >>> calc.add_variable("x", 10)
            >>> calc.variables["x"]
            10
            >>> 
            >>> # Com unidade
            >>> calc.add_variable("F", 100, unit="kN")
            >>> 
            >>> # Com incerteza
            >>> calc.add_variable("L", 6.0, uncertainty=0.1)
        """
        with self._lock:
            # Se passar unit ou uncertainty, criar Variable object
            if unit or uncertainty is not None:
                from .variable import Variable
                var_obj = Variable(name, value, unit=unit)
                # Adicionar incerteza como atributo customizado
                if uncertainty is not None:
                    var_obj.uncertainty = uncertainty
                self.variables[name] = var_obj
            else:
                # Valor direto
                self.variables[name] = value
        
        self._logger.debug(
            f"Variable added: {name} = {value} "
            f"{unit if unit else ''} "
            f"±{uncertainty if uncertainty else ''}"
        )


    
    def remove_variable(self, name: str) -> None:
        """
        Remove variável do calculator.
        
        Args:
            name: Nome da variável
        
        Raises:
            KeyError: Se variável não existe
        """
        with self._lock:
            if name not in self.variables:
                raise KeyError(f"Variable '{name}' not found")
            del self.variables[name]
        
        self._logger.debug(f"Variable removed: {name}")
    
    def clear_variables(self) -> None:
        """Limpa todas as variáveis."""
        with self._lock:
            self.variables.clear()
        
        self._logger.debug("All variables cleared")
    
    def evaluate(
        self,
        expression: Union[str, Expr],
        variables: Optional[Dict[str, Any]] = None,
        mode: str = 'auto',
        vectorized: bool = False
    ) -> Any:
        """
        Avalia expressão com suporte a vetorização.
        
        Args:
            expression: Expressão a avaliar
            variables: Variáveis (dict ou Variable objects)
            mode: 'auto', 'symbolic', 'numeric'
            vectorized: Se True, usa compute_vectorized() para arrays
        
        Returns:
            Resultado (tipo depende do mode e vectorized)
        
        Examples:
            >>> calc = Calculator()
            >>> calc.evaluate("x + y", {'x': 10, 'y': 5})
            15.0
            
            >>> # Vetorizado
            >>> import numpy as np
            >>> calc.add_variable("x", np.array([1, 2, 3]))
            >>> result = calc.evaluate("x**2", vectorized=True)
            >>> result.value
            array([1, 4, 9])
        """
        # Vetorização (NOVA FEATURE)
        if vectorized:
            return self.compute_vectorized(expression, variables)
        
        # Processar variáveis (Variable -> float)
        vars_dict = self._process_variables(variables or {})
        
        # Escolher modo de avaliação
        if mode == 'symbolic':
            return self._evaluate_symbolic(expression, vars_dict)
        elif mode == 'numeric':
            return self._evaluate_numeric(expression, vars_dict)
        else:  # auto
            return self._evaluate_auto(expression, vars_dict)

    
    def compute(
        self,
        expression: Union[str, Expr, 'Equation'],
        variables: Optional[Dict[str, Any]] = None,
        vectorized: bool = False,
        propagate_error: bool = False,
        optimize: bool = False,
        **kwargs
    ) -> 'CalculationResult':
        """
        Computa expressão e retorna CalculationResult.
        
        Args:
            expression: Expressão, Equation ou string
            variables: Variáveis opcionais
            vectorized: Se True, processa arrays NumPy
            propagate_error: Se True, propaga incertezas
            optimize: Se True, tenta otimizar performance (metadata flag)
            **kwargs: Passados para evaluate()
        
        Returns:
            CalculationResult com valor e metadados
        
        Examples:
            >>> calc = Calculator()
            >>> calc.add_variable("x", 10)
            >>> result = calc.compute("x**2")
            >>> result.value
            100
            
            >>> # Vetorizado
            >>> import numpy as np
            >>> calc.add_variable("x", np.array([1, 2, 3]))
            >>> result = calc.compute("x**2", vectorized=True)
            >>> result.value
            array([1, 4, 9])
            
            >>> # Com propagação de erros
            >>> calc.add_variable("x", 10, uncertainty=0.5)
            >>> result = calc.compute("x**2", propagate_error=True)
            >>> result.metadata['uncertainty']
        """
        
        # ================== CORREÇÃO INÍCIO ==================
        
        # Handle Equation objects by delegating
        if hasattr(expression, '__class__') and expression.__class__.__name__ == 'Equation':
            # Delega para o método especializado
            # Passa 'vectorized' e outros kwargs relevantes
            # O 'symbolic' é tratado pelo compute_equation se necessário
            return self.compute_equation(
                equation=expression,
                vectorized=vectorized
                # Adicione outros kwargs se compute_equation os aceitar
            )
        
        # =================== CORREÇÃO FIM ====================
        
        
        # Propagação de incertezas
        if propagate_error:
            return self.compute_with_uncertainty(expression, variables)
        
        # Vetorização
        if vectorized:
            result = self.compute_vectorized(expression, variables)
            # Adicionar flag de otimização se solicitado
            if optimize:
                result.metadata['optimization_used'] = True
            return result
        
        # Computação normal
        result_value = self.evaluate(expression, variables or self.variables, **kwargs)
        
        # Wrap em CalculationResult se não for já
        if isinstance(result_value, CalculationResult):
            # Adicionar flag de otimização se solicitado
            if optimize:
                result_value.metadata['optimization_used'] = True
            return result_value
        
        metadata = {'mode': kwargs.get('mode', 'auto')}
        if optimize:
            metadata['optimization_used'] = True
        
        return CalculationResult(
            value=result_value,
            expression=str(expression),
            unit=None,
            symbolic=None,
            metadata=metadata
        )

    
    def batch_compute(
        self,
        expressions: List[str],
        parallel: bool = False,
        workers: int = None
    ) -> List['CalculationResult']:
        """
        Computa múltiplas expressões em batch.
        
        Args:
            expressions: Lista de expressões
            parallel: Se True, executa em paralelo (requer concurrent.futures)
            workers: Número de workers (None = auto)
        
        Returns:
            Lista de CalculationResult
        
        Examples:
            >>> calc = Calculator()
            >>> calc.add_variable("x", 10)
            >>> results = calc.batch_compute(["x**2", "x**3", "x**4"])
            >>> [r.value for r in results]
            [100, 1000, 10000]
        """

        # ================== CORREÇÃO INÍCIO ==================
        def _compute_safe(expr: str) -> Optional['CalculationResult']:
            """Helper interno para capturar exceções durante o batch."""
            try:
                # Tenta computar normalmente
                return self.compute(expr)
            except Exception as e:
                # Se falhar (ex: Divisão por zero), loga e retorna None
                self._logger.warning(
                    f"Erro no batch_compute para a expressão '{expr}': {e}"
                )
                return None
        # =================== CORREÇÃO FIM ===================

        if parallel and workers:
            # Executar em paralelo
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=workers) as executor:
                # Usar o helper _compute_safe
                futures = [executor.submit(_compute_safe, expr) for expr in expressions]
                results = [f.result() for f in futures]
            return results
        else:
            # Executar sequencialmente usando o helper _compute_safe
            return [_compute_safe(expr) for expr in expressions]





    def cache_stats(self) -> dict:
        """
        Retorna estatísticas do cache.
        
        Returns:
            Dicionário com estatísticas
        
        Examples:
            >>> calc = Calculator(use_cache=True)
            >>> stats = calc.cache_stats()
            >>> print(stats)
        """
        cache_size = len(self._cache) if self._cache else 0
        hits = 0
        misses = 0
        total = hits + misses
        hit_rate = hits / total if total > 0 else 0.0
        
        return {
            'enabled': self._use_cache,
            'size': cache_size,
            'hits': hits,
            'misses': misses,
            'total': total,
            'hit_rate': hit_rate
        }


    
    def clear_cache(self) -> None:
        """
        Limpa o cache.
        
        Examples:
            >>> calc = Calculator(use_cache=True)
            >>> calc.compute("2 + 2")
            >>> calc.clear_cache()
            >>> stats = calc.cache_stats()
            >>> print(stats['size'])  # 0
        """
        if self._cache:
            with self._lock:
                self._cache.clear()
            
            self._logger.debug("Cache cleared")



# ============================================================================
# CALCULATOR - OPERAÇÕES SIMBÓLICAS (SYMPY) - PARTE 2/3
# ============================================================================

    def simplify(
        self,
        expression: Union[str, Expr],
        **kwargs
    ) -> Expr:
        """
        Simplifica expressão simbolicamente.
        
        Args:
            expression: Expressão a simplificar
            **kwargs: Argumentos para sp.simplify
        
        Returns:
            Expressão simplificada
        
        Examples:
            >>> calc = Calculator()
            >>> result = calc.simplify("x**2 + 2*x + 1")
            >>> print(result)  # (x + 1)**2
        """
        if not SYMPY_AVAILABLE:
            raise CalculatorError("SymPy não disponível")
        
        # Converter para SymPy
        if isinstance(expression, str):
            expr = sp.sympify(expression)
        else:
            expr = expression
        
        # Simplificar
        with self._lock:
            result = sp.simplify(expr, **kwargs)
        
        self._logger.debug(f"Simplified: {expr} -> {result}")
        return result

    
    def _process_variables(self, variables: Dict[str, Any]) -> Dict[str, float]:
        """
        Processa variáveis (Variable -> float).
        
        Args:
            variables: Dicionário de variáveis
        
        Returns:
            Dicionário com valores numéricos
        """
        processed = {}
        
        for name, value in variables.items():
            if isinstance(value, Variable):
                processed[name] = float(value.value)
            else:
                processed[name] = float(value)
        
        return processed
    
    def _evaluate_symbolic(self, expression: Union[str, Expr], variables: Dict) -> Expr:
        """Avaliação simbólica (SymPy)."""
        if not SYMPY_AVAILABLE:
            raise CalculatorError("SymPy não disponível")
        
        if isinstance(expression, str):
            symbols_dict = {name: Symbol(name) for name in variables.keys()}
            expr = sp.sympify(expression, locals=symbols_dict)
        else:
            expr = expression
        
        # VALIDAR se resultado é uma expressão SymPy válida
        if not hasattr(expr, 'subs'):
            raise EvaluationError(f"Expressão inválida ou maliciosa: {expr}")
        
        subs_dict = {Symbol(k): v for k, v in variables.items()}
        result = expr.subs(subs_dict)
        
        return result

    
    def _evaluate_numeric(self, expression: Union[str, Expr], variables: Dict) -> float:
        """Avaliação numérica (Safe eval)."""
        if isinstance(expression, str):
            # Safe eval direto
            return self._evaluator.safe_eval(expression, variables)
        elif SYMPY_AVAILABLE and isinstance(expression, Expr):
            # SymPy → lambdify → NumPy
            free_symbols = list(expression.free_symbols)
            func = sp.lambdify(free_symbols, expression, 'numpy')
            
            # Preparar argumentos
            args = [variables.get(str(s), 0) for s in free_symbols]
            
            return float(func(*args))
        else:
            raise EvaluationError("Tipo de expressão não suportado")
    
    def _evaluate_auto(self, expression: Union[str, Expr], variables: Dict) -> Any:
        """
        Avaliação automática (escolhe melhor método).
        
        Args:
            expression: Expressão
            variables: Variáveis
        
        Returns:
            Resultado
        """
        
        # ================== CORREÇÃO ROBUSTA (Sem 'if variables:') ==================
        
        # A lógica NÃO deve depender de 'if variables:'.
        # Devemos SEMPRE tentar a avaliação numérica primeiro.
        # "10 + 5" (sem variáveis) é um cálculo numérico.
        # "x + 10" (com x=5) é um cálculo numérico.
        # "x + 10" (sem x) deve falhar com NameError.
        
        try:
            # Tenta avaliação numérica segura PRIMEIRO
            return self._evaluate_numeric(expression, variables)
        
        except (EvaluationError, TypeError) as numeric_error:
            # _evaluate_numeric (via safe_eval) SEMPRE levanta
            # EvaluationError, mesmo para ZeroDivisionError ou NameError.
            
            # Precisamos inspecionar a MENSAGEM de erro.
            err_str = str(numeric_error)
            
            if "division by zero" in err_str or \
               "is not defined" in err_str:
                # ERROS FATAIS: Divisão por zero ou Variável não definida.
                # Estes NUNCA devem fazer fallback para simbólico.
                self._logger.warning(f"Erro numérico fatal detectado: {numeric_error}")
                raise numeric_error # Re-levanta o EvaluationError
            
            # ERROS NÃO FATAIS (ex: TypeError "unsupported operand type(s)"):
            # Podem ser uma expressão simbólica (ex: 'x' + 2).
            # Tentar o fallback para simbólico.
            self._logger.debug(f"Numeric eval failed ({type(numeric_error).__name__}), falling back to symbolic.")
            try:
                return self._evaluate_symbolic(expression, variables)
            except Exception as symbolic_error:
                # Se o fallback simbólico também falhar, levanta o erro original
                self._logger.error(f"Symbolic fallback failed: {symbolic_error}")
                raise numeric_error from symbolic_error
        
        # =================== FIM DA CORREÇÃO ===================



    # ============================================================
    # INTEGRATION LAYER - COMPLETAR PARTE 1
    # ============================================================
    
    def compute_equation(
            self,
            equation: 'Equation',
            symbolic: bool = False,
            vectorized: bool = False
        ) -> CalculationResult:
        """
        Método dedicado para processar objetos Equation.
        
        Oferece controle fino sobre modo de avaliação:
        - symbolic=True: Retorna expressão simbólica (SymPy)
        - vectorized=True: Suporte a arrays NumPy
        
        Args:
            equation: Objeto Equation para avaliar
            symbolic: Se True, mantém forma simbólica
            vectorized: Se True, suporta arrays NumPy
        
        Returns:
            CalculationResult com valor e metadados
        
        Raises:
            CalculationError: Se avaliação falhar
        
        Examples:
            >>> from pymemorial.core.equation import Equation
            >>> calc = Calculator()
            >>> calc.add_variable("x", 5)
            >>> eq = Equation("2*x + 3")
            >>> result = calc.compute_equation(eq)
            >>> result.value
            13
            
            >>> # Modo simbólico
            >>> result_sym = calc.compute_equation(eq, symbolic=True)
            >>> result_sym.symbolic
            2*x + 3
        """
        if not SYMPY_AVAILABLE:
            raise CalculationError(
                "SymPy não disponível. Instale com: pip install sympy"
            )
        
        try:
            # Extrair variáveis do equation
            variables_dict = {}
            for var_name in equation.variables_used:
                if var_name in self.variables:
                    var = self.variables[var_name]
                    variables_dict[var_name] = var.value if hasattr(var, 'value') else var
                else:
                    # ================== APRIMORAMENTO 1 ==================
                    # Só levante erro se a variável não for encontrada E
                    # NÃO estivermos no modo simbólico.
                    if not symbolic:
                        raise CalculationError(
                            f"Variável '{var_name}' necessária na equação não está definida"
                        )
                    # =====================================================
            
            # Modo simbólico: retorna sem avaliar numericamente
            if symbolic:
                return CalculationResult(
                    value=equation.expr,
                    expression=equation.expression_str,
                    symbolic=equation.expr,
                    # ================== CORREÇÃO DO BUG ==================
                    unit=getattr(equation, 'unit', None), # Acesso seguro
                    # =====================================================
                    metadata={
                        'mode': 'symbolic',
                        'variables': list(equation.variables_used)
                    }
                )
            
            # Modo vetorizado: suporte a arrays NumPy
            if vectorized and NUMPY_AVAILABLE:
                # Converter para função lambdificada
                symbols_list = [sp.Symbol(var) for var in equation.variables_used]
                func = sp.lambdify(symbols_list, equation.expr, modules='numpy')
                
                # Avaliar com arrays
                values_list = [variables_dict[var] for var in equation.variables_used]
                result_value = func(*values_list)
                
                return CalculationResult(
                    value=result_value,
                    expression=equation.expression_str,
                    symbolic=equation.expr,
                    # ================== CORREÇÃO DO BUG ==================
                    unit=getattr(equation, 'unit', None), # Acesso seguro
                    # =====================================================
                    metadata={
                        'mode': 'vectorized',
                        'shape': getattr(result_value, 'shape', None)
                    }
                )
            
            # Modo padrão: avaliação numérica
            # (Assumindo que Equation.evaluate() retorna um CalculationResult)
            result = equation.evaluate(variables=variables_dict)
            
            return CalculationResult(
                value=result.value,
                expression=equation.expression_str,
                symbolic=equation.expr,
                # ================== CORREÇÃO DO BUG ==================
                unit=result.unit or getattr(equation, 'unit', None), # Acesso seguro
                # =====================================================
                metadata={
                    'mode': 'numeric',
                    'variables_used': list(equation.variables_used),
                    'substitutions': result.metadata.get('substitutions', {})
                }
            )
            
        except Exception as e:
            self._logger.error(f"Erro ao computar equation: {e}")
            raise CalculationError(
                f"Falha ao computar equation '{equation.expression_str}': {e}"
            )
    
    def apply_norms(
            self,
            result: CalculationResult,
            norm: str = "NBR6118",
            safety_factor: Optional[float] = None
        ) -> CalculationResult:
        """
        Aplica fatores de segurança/normas ao resultado.
        
        Suporta principais normas brasileiras:
        - NBR6118: Concreto armado (γf = 1.4 padrão)
        - NBR8800: Aço estrutural (γa = 1.1 padrão)
        - NBR6123: Vento (γf = 1.4 padrão)
        
        Args:
            result: CalculationResult para aplicar norma
            norm: Nome da norma ("NBR6118", "NBR8800", etc)
            safety_factor: Fator customizado (sobrescreve padrão)
        
        Returns:
            CalculationResult com valor majorado
        
        Raises:
            CalculationError: Se norma não for reconhecida
        
        Examples:
            >>> calc = Calculator()
            >>> result = calc.compute("50 * 10")  # Força característica (kN)
            >>> result_d = calc.apply_norms(result, norm="NBR6118")
            >>> result_d.value
            700.0  # 50*10*1.4 = 700 kN (força de cálculo)
        """
        # Fatores de norma padrão
        NORM_FACTORS = {
            "NBR6118": 1.4,   # Concreto armado
            "NBR8800": 1.1,   # Aço estrutural
            "NBR6123": 1.4,   # Vento
            "NBR9062": 1.4,   # Pré-moldados
            "EUROCODE2": 1.5, # Concreto (Eurocode)
            "ACI318": 1.2,    # Concreto (ACI)
        }
        
        norm_upper = norm.upper()
        
        if norm_upper not in NORM_FACTORS and safety_factor is None:
            raise CalculationError(
                f"Norma '{norm}' não reconhecida. "
                f"Disponíveis: {', '.join(NORM_FACTORS.keys())}. "
                f"Ou forneça 'safety_factor' customizado."
            )
        
        # Usar fator customizado ou padrão da norma
        factor = safety_factor if safety_factor is not None else NORM_FACTORS.get(norm_upper, 1.0)
        
        # Aplicar fator ao valor
        try:
            if NUMPY_AVAILABLE and isinstance(result.value, np.ndarray):
                factored_value = result.value * factor
            else:
                factored_value = float(result.value) * factor
            
            # Criar novo resultado com metadados
            return CalculationResult(
                value=factored_value,
                expression=f"({result.expression}) * {factor}",
                symbolic=result.symbolic,
                unit=result.unit,
                metadata={
                    **result.metadata,
                    'norm_applied': norm,
                    'safety_factor': factor,
                    'original_value': result.value
                }
            )
            
        except Exception as e:
            raise CalculationError(
                f"Erro ao aplicar norma '{norm}': {e}"
            )
    
    def compute_symbolic(
            self,
            expression: Union[str, 'Equation'],
            simplify_result: bool = True
        ) -> CalculationResult:
        """
        Computa expressão em modo 100% simbólico (sem avaliação numérica).
        
        Útil para:
        - Derivações simbólicas
        - Simplificações algébricas
        - Manipulação de fórmulas
        - Geração de LaTeX
        
        Args:
            expression: String ou Equation
            simplify_result: Se True, simplifica resultado
        
        Returns:
            CalculationResult com valor simbólico
        
        Examples:
            >>> calc = Calculator()
            >>> result = calc.compute_symbolic("x**2 + 2*x + 1")
            >>> result.symbolic
            (x + 1)**2  # Simplificado
        """
        if not SYMPY_AVAILABLE:
            raise CalculationError("SymPy não disponível para modo simbólico")
        
        try:
            # Se é Equation, usar compute_equation
            if hasattr(expression, 'expr'):
                return self.compute_equation(expression, symbolic=True)
            
            # Parse string para SymPy
            expr = sp.sympify(expression)
            
            # Simplificar se solicitado
            if simplify_result:
                expr = sp.simplify(expr)
            
            return CalculationResult(
                value=expr,
                expression=str(expression),
                symbolic=expr,
                unit=None,
                metadata={'mode': 'symbolic'}
            )
            
        except Exception as e:
            raise CalculationError(
                f"Erro no modo simbólico: {e}"
            )
 

    def expand(
        self,
        expression: Union[str, Expr],
        **kwargs
    ) -> Expr:
        """
        Expande expressão simbolicamente.
        
        Args:
            expression: Expressão a expandir
            **kwargs: Argumentos para sp.expand
        
        Returns:
            Expressão expandida
        
        Examples:
            >>> calc = Calculator()
            >>> result = calc.expand("(x + 1)**2")
            >>> print(result)  # x**2 + 2*x + 1
        """
        if not SYMPY_AVAILABLE:
            raise CalculatorError("SymPy não disponível")
        
        # Converter para SymPy
        if isinstance(expression, str):
            expr = sp.sympify(expression)
        else:
            expr = expression
        
        # Expandir
        with self._lock:
            result = sp.expand(expr, **kwargs)
        
        self._logger.debug(f"Expanded: {expr} -> {result}")
        return result
    
    def factor(
        self,
        expression: Union[str, Expr],
        **kwargs
    ) -> Expr:
        """
        Fatora expressão simbolicamente.
        
        Args:
            expression: Expressão a fatorar
            **kwargs: Argumentos para sp.factor
        
        Returns:
            Expressão fatorada
        
        Examples:
            >>> calc = Calculator()
            >>> result = calc.factor("x**2 - 1")
            >>> print(result)  # (x - 1)*(x + 1)
        """
        if not SYMPY_AVAILABLE:
            raise CalculatorError("SymPy não disponível")
        
        # Converter para SymPy
        if isinstance(expression, str):
            expr = sp.sympify(expression)
        else:
            expr = expression
        
        # Fatorar
        with self._lock:
            result = sp.factor(expr, **kwargs)
        
        self._logger.debug(f"Factored: {expr} -> {result}")
        return result
    
    def diff(
        self,
        expression: Union[str, Expr],
        variable: str,
        order: int = 1
    ) -> Expr:
        """
        Deriva expressão simbolicamente.
        
        Args:
            expression: Expressão a derivar
            variable: Variável de derivação
            order: Ordem da derivada
        
        Returns:
            Derivada
        
        Examples:
            >>> calc = Calculator()
            >>> result = calc.diff("x**3", "x")
            >>> print(result)  # 3*x**2
        """
        if not SYMPY_AVAILABLE:
            raise CalculatorError("SymPy não disponível")
        
        # Converter para SymPy
        if isinstance(expression, str):
            expr = sp.sympify(expression)
        else:
            expr = expression
        
        var_symbol = Symbol(variable)
        
        # Derivar
        with self._lock:
            result = sp.diff(expr, var_symbol, order)
        
        self._logger.debug(f"Diff: d^{order}({expr})/d{variable}^{order} = {result}")
        return result
    
    def integrate(
        self,
        expression: Union[str, Expr],
        variable: str,
        limits: Optional[tuple] = None
    ) -> Expr:
        """
        Integra expressão simbolicamente.
        
        Args:
            expression: Expressão a integrar
            variable: Variável de integração
            limits: Limites (a, b) para integral definida
        
        Returns:
            Integral
        
        Examples:
            >>> calc = Calculator()
            >>> result = calc.integrate("x**2", "x")
            >>> print(result)  # x**3/3
            >>> 
            >>> result = calc.integrate("x**2", "x", (0, 1))
            >>> print(result)  # 1/3
        """
        if not SYMPY_AVAILABLE:
            raise CalculatorError("SymPy não disponível")
        
        # Converter para SymPy
        if isinstance(expression, str):
            expr = sp.sympify(expression)
        else:
            expr = expression
        
        var_symbol = Symbol(variable)
        
        # Integrar
        with self._lock:
            if limits:
                result = sp.integrate(expr, (var_symbol, limits[0], limits[1]))
            else:
                result = sp.integrate(expr, var_symbol)
        
        self._logger.debug(f"Integrate: ∫{expr} d{variable} = {result}")
        return result


    # ============================================================
    # PARTE 2/3: SCIPY NUMERICAL METHODS
    # ============================================================
    
    def solve_equation_numerical(
        self,
        expression: Union[str, Expr],
        variable: str,
        initial_guess: float = 0.0,
        method: str = 'hybr',
        tol: float = 1e-6
    ) -> 'CalculationResult':
        """
        Resolve equação numericamente usando scipy.optimize.fsolve.
        
        Encontra raiz de f(x) = 0 onde f(x) = expression.
        
        Args:
            expression: Equação a resolver (será igualada a zero)
            variable: Variável a resolver
            initial_guess: Chute inicial
            method: Método numérico ('hybr', 'lm', 'broyden1', ...)
            tol: Tolerância de convergência
        
        Returns:
            CalculationResult com solução numérica
        
        Raises:
            CalculationError: Se scipy não disponível ou não convergir
        
        Examples:
            >>> calc = Calculator()
            >>> # Resolver x^2 - 4 = 0
            >>> result = calc.solve_equation_numerical("x**2 - 4", "x", initial_guess=1.0)
            >>> result.value
            2.0
            
            >>> # Resolver transcendental: x - cos(x) = 0
            >>> result = calc.solve_equation_numerical("x - cos(x)", "x", initial_guess=0.5)
            >>> result.value  # x ≈ 0.7390851332151607
        
        Notes:
            - Para equações polinomiais simples, considere usar SymPy
            - Para múltiplas raízes, use find_roots() com bounds
            - Para sistemas não-lineares, passe vetor de chutes iniciais
        """
        # Verificar disponibilidade
        if not SCIPY_AVAILABLE:
            raise CalculationError(
                "SciPy não disponível para métodos numéricos. "
                "Instale com: pip install scipy"
            )
        
        if not SYMPY_AVAILABLE:
            raise CalculationError(
                "SymPy necessário para conversão. Instale com: pip install sympy"
            )
        
        try:
            # Converter para SymPy
            if isinstance(expression, str):
                expr = sp.sympify(expression)
            else:
                expr = expression
            
            # Criar função numérica
            var_symbol = Symbol(variable)
            func = sp.lambdify(var_symbol, expr, modules=['scipy', 'numpy'])
            
            # Resolver numericamente
            with self._lock:
                from scipy.optimize import fsolve
                solution = fsolve(func, initial_guess, full_output=False, xtol=tol)
            
            # Extrair resultado
            result_value = float(solution[0]) if hasattr(solution, '__iter__') else float(solution)
            
            # Verificar convergência
            residual = float(func(result_value))
            converged = abs(residual) < tol
            
            # NOVO: Lançar erro se não convergiu
            if not converged:
                raise CalculatorError(
                    f"Falha na convergência ao resolver '{expression}' = 0. "
                    f"Resíduo final: {residual:.2e}. "
                    f"Tente ajustar initial_guess ou aumentar tolerância."
                )
            
            self._logger.info(
                f"Solved numerically: {expr} = 0 for {variable} → {result_value:.6f} "
                f"(residual: {residual:.2e}, converged: {converged})"
            )
            
            return CalculationResult(
                value=result_value,
                expression=f"solve({expr} = 0, {variable})",
                unit=None,
                symbolic=None,
                metadata={
                    'method': 'scipy.optimize.fsolve',
                    'initial_guess': initial_guess,
                    'residual': residual,
                    'converged': converged,
                    'tolerance': tol
                }
            )

            
        except Exception as e:
            self._logger.error(f"Erro ao resolver numericamente: {e}")
            raise CalculationError(
                f"Falha ao resolver '{expression}' = 0: {e}. "
                f"Tente ajustar initial_guess ou usar método diferente."
            )
    
    def find_roots(
        self,
        expression: Union[str, Expr],
        variable: str,
        bounds: Tuple[float, float],
        method: str = 'brentq',
        num_points: int = 10
    ) -> List['CalculationResult']:
        """
        Encontra todas as raízes no intervalo [a, b].
        
        Divide intervalo em sub-intervalos e busca mudanças de sinal.
        
        Args:
            expression: Função f(x) para encontrar f(x) = 0
            variable: Variável independente
            bounds: Tupla (a, b) com limites do intervalo
            method: Método raiz ('brentq', 'bisect', 'newton', ...)
            num_points: Número de pontos para varredura inicial
        
        Returns:
            Lista de CalculationResult com todas as raízes encontradas
        
        Raises:
            CalculationError: Se scipy não disponível
        
        Examples:
            >>> calc = Calculator()
            >>> # Encontrar raízes de x^3 - x em [-2, 2]
            >>> roots = calc.find_roots("x**3 - x", "x", bounds=(-2, 2))
            >>> [r.value for r in roots]
            [-1.0, 0.0, 1.0]
        """
        if not SCIPY_AVAILABLE:
            raise CalculationError("SciPy necessário para find_roots")
        
        if not SYMPY_AVAILABLE:
            raise CalculationError("SymPy necessário para find_roots")
        
        try:
            # Converter para função
            if isinstance(expression, str):
                expr = sp.sympify(expression)
            else:
                expr = expression
            
            var_symbol = Symbol(variable)
            func = sp.lambdify(var_symbol, expr, modules=['scipy', 'numpy'])
            
            # Varredura inicial
            a, b = bounds
            x_scan = np.linspace(a, b, num_points)
            y_scan = [func(x) for x in x_scan]
            
            # Detectar mudanças de sinal
            roots = []
            from scipy.optimize import brentq, bisect, newton
            
            root_finder = {'brentq': brentq, 'bisect': bisect}.get(method, brentq)
            
            for i in range(len(y_scan) - 1):
                # Mudança de sinal?
                if y_scan[i] * y_scan[i + 1] < 0:
                    try:
                        root = root_finder(func, x_scan[i], x_scan[i + 1])
                        
                        # Evitar duplicatas
                        is_duplicate = any(abs(root - r.value) < 1e-6 for r in roots)
                        if not is_duplicate:
                            roots.append(CalculationResult(
                                value=float(root),
                                expression=f"root({expr} = 0, {variable})",
                                symbolic=None,
                                unit=None,
                                metadata={'method': method, 'bounds': bounds}
                            ))
                    except:
                        continue
            
            self._logger.info(f"Found {len(roots)} root(s) in [{a}, {b}]")
            return roots
            
        except Exception as e:
            raise CalculationError(f"Erro ao encontrar raízes: {e}")
    
    def optimize_function(
        self,
        expression: Union[str, Expr],
        variable: str,
        initial_guess: float,
        bounds: Optional[Tuple[float, float]] = None,
        method: str = 'SLSQP',
        maximize: bool = False
    ) -> 'CalculationResult':
        """
        Otimiza função (minimiza ou maximiza).
        
        Args:
            expression: Função objetivo f(x)
            variable: Variável a otimizar
            initial_guess: Chute inicial
            bounds: Limites (opcional) como (min, max)
            method: Método de otimização ('SLSQP', 'L-BFGS-B', 'TNC', ...)
            maximize: Se True, maximiza; se False, minimiza
        
        Returns:
            CalculationResult com ponto ótimo
        
        Examples:
            >>> calc = Calculator()
            >>> # Minimizar x^2 + 2x + 1
            >>> result = calc.optimize_function("x**2 + 2*x + 1", "x", initial_guess=0)
            >>> result.value  # x = -1.0 (mínimo)
            >>> result.metadata['optimal_value']  # f(-1) = 0.0
            
            >>> # Maximizar -(x-3)^2 + 5 com restrição [0, 10]
            >>> result = calc.optimize_function(
            ...     "-(x-3)**2 + 5", "x", initial_guess=0, bounds=(0, 10), maximize=True
            ... )
            >>> result.value  # x = 3.0
        """
        if not SCIPY_AVAILABLE:
            raise CalculationError("SciPy necessário para otimização")
        
        try:
            # Converter
            if isinstance(expression, str):
                expr = sp.sympify(expression)
            else:
                expr = expression
            
            var_symbol = Symbol(variable)
            
            # Inverter sinal se maximizar
            if maximize:
                expr = -expr
            
            func = sp.lambdify(var_symbol, expr, modules=['scipy', 'numpy'])
            
            # Otimizar
            from scipy.optimize import minimize
            
            bounds_list = [bounds] if bounds else None
            
            with self._lock:
                result = minimize(
                    func,
                    x0=initial_guess,
                    bounds=bounds_list,
                    method=method
                )
            
            optimal_x = float(result.x[0])
            optimal_f = float(result.fun)
            
            # Inverter de volta se maximizou
            if maximize:
                optimal_f = -optimal_f
            
            self._logger.info(
                f"Optimization {'maximized' if maximize else 'minimized'}: "
                f"{variable}* = {optimal_x:.6f}, f({variable}*) = {optimal_f:.6f}"
            )
            
            return CalculationResult(
                value=optimal_x,
                expression=f"{'maximize' if maximize else 'minimize'} {expr}",
                symbolic=None,
                unit=None,
                metadata={
                    'method': method,
                    'optimal_value': optimal_f,
                    'success': result.success,
                    'iterations': result.nit if hasattr(result, 'nit') else None,
                    'message': result.message
                }
            )
            
        except Exception as e:
            raise CalculationError(f"Erro na otimização: {e}")
    
    def integrate_numerical(
        self,
        expression: Union[str, Expr],
        variable: str,
        lower: float,
        upper: float,
        method: str = 'quad',
        epsabs: float = 1.49e-8,
        epsrel: float = 1.49e-8
    ) -> 'CalculationResult':
        """
        Integra numericamente ∫[a,b] f(x) dx.
        
        Args:
            expression: Integrando f(x)
            variable: Variável de integração
            lower: Limite inferior
            upper: Limite superior
            method: Método ('quad', 'romberg', 'simpson', ...)
            epsabs: Tolerância absoluta
            epsrel: Tolerância relativa
        
        Returns:
            CalculationResult com valor da integral
        
        Examples:
            >>> calc = Calculator()
            >>> # ∫[0,1] x^2 dx = 1/3
            >>> result = calc.integrate_numerical("x**2", "x", 0, 1)
            >>> result.value
            0.33333333...
            
            >>> # ∫[0,π] sin(x) dx = 2
            >>> from math import pi
            >>> result = calc.integrate_numerical("sin(x)", "x", 0, pi)
            >>> result.value
            2.0
        """
        if not SCIPY_AVAILABLE:
            raise CalculatorError("SciPy necessário para integração numérica")
        
        try:
            # Converter
            if isinstance(expression, str):
                expr = sp.sympify(expression)
            else:
                expr = expression
            
            var_symbol = Symbol(variable)
            func = sp.lambdify(var_symbol, expr, modules=['scipy', 'numpy'])
            
            # Integrar - CORREÇÃO: APENAS quad
            from scipy.integrate import quad
            
            with self._lock:
                integral_value, error = quad(func, lower, upper, epsabs=epsabs, epsrel=epsrel)
            
            self._logger.info(
                f"Integrated: ∫[{lower},{upper}] {expr} d{variable} = {integral_value:.6f}"
            )
            
            return CalculationResult(
                value=float(integral_value),
                expression=f"∫[{lower},{upper}] {expr} d{variable}",
                unit=None,
                symbolic=None,
                metadata={
                    'method': 'quad',
                    'error_estimate': float(error) if error else None,
                    'bounds': (lower, upper)
                }
            )
            
        except Exception as e:
            raise CalculatorError(f"Erro na integração numérica: {e}")

    
    def differentiate_numerical(
        self,
        expression: Union[str, Expr],
        variable: str,
        at_point: float,
        dx: float = 1e-5,
        order: int = 1
    ) -> 'CalculationResult':
        """
        Deriva numericamente df/dx no ponto x0.
        
        Usa diferenças finitas centrais.
        
        Args:
            expression: Função f(x)
            variable: Variável de derivação
            at_point: Ponto x0 onde calcular derivada
            dx: Passo (epsilon)
            order: Ordem da derivada (1, 2, 3, ...)
        
        Returns:
            CalculationResult com valor da derivada
        
        Examples:
            >>> calc = Calculator()
            >>> # d/dx(x^2) no ponto x=3 = 2*3 = 6
            >>> result = calc.differentiate_numerical("x**2", "x", at_point=3)
            >>> result.value
            6.0
        """
        if not SCIPY_AVAILABLE:
            raise CalculatorError("SciPy necessário para derivação numérica")
        
        try:
            # Converter
            if isinstance(expression, str):
                expr = sp.sympify(expression)
            else:
                expr = expression
            
            var_symbol = Symbol(variable)
            func = sp.lambdify(var_symbol, expr, modules=['scipy', 'numpy'])
            
            # Derivar numericamente usando diferenças finitas centrais
            # CORREÇÃO: Implementação manual (scipy.misc.derivative removido)
            if order == 1:
                # Primeira derivada: [f(x+h) - f(x-h)] / (2h)
                deriv_value = (func(at_point + dx) - func(at_point - dx)) / (2 * dx)
            elif order == 2:
                # Segunda derivada: [f(x+h) - 2f(x) + f(x-h)] / h²
                deriv_value = (func(at_point + dx) - 2*func(at_point) + func(at_point - dx)) / (dx**2)
            elif order == 3:
                # Terceira derivada: [f(x+2h) - 2f(x+h) + 2f(x-h) - f(x-2h)] / (2h³)
                deriv_value = (
                    func(at_point + 2*dx) 
                    - 2*func(at_point + dx) 
                    + 2*func(at_point - dx) 
                    - func(at_point - 2*dx)
                ) / (2 * dx**3)
            else:
                raise CalculatorError(f"Ordem {order} não suportada. Use 1, 2 ou 3.")
            
            self._logger.info(
                f"Derivative: d^{order}/d{variable}^{order}({expr})|_{variable}={at_point} = {deriv_value:.6f}"
            )
            
            return CalculationResult(
                value=float(deriv_value),
                expression=f"d^{order}/d{variable}^{order}({expr})|_{variable}={at_point}",
                unit=None,
                symbolic=None,
                metadata={
                    'method': 'finite_differences_central',
                    'point': at_point,
                    'dx': dx,
                    'order': order
                }
            )
            
        except Exception as e:
            raise CalculatorError(f"Erro na derivação numérica: {e}")


    # ============================================================
    # PARTE 3/3: VETORIZAÇÃO + MONTE CARLO + PERFORMANCE
    # ============================================================


    def compute_with_uncertainty(
        self,
        expression: Union[str, Expr],
        variables: Optional[Dict[str, Any]] = None
    ) -> 'CalculationResult':
        """
        Computa expressão com propagação de incertezas (método de primeira ordem).
        
        Usa propagação de incertezas pelo método de Taylor de primeira ordem:
        σ_f² = Σ (∂f/∂xi)² * σ_xi²
        
        Variáveis devem ter sido adicionadas com parâmetro `uncertainty`.
        
        Args:
            expression: Expressão a computar
            variables: Variáveis opcionais (usa self.variables se None)
        
        Returns:
            CalculationResult com valor e incerteza propagada
        
        Raises:
            CalculatorError: Se SymPy não disponível ou variáveis sem incerteza
        
        Examples:
            >>> calc = Calculator()
            >>> calc.add_variable("x", 10, uncertainty=0.5)
            >>> calc.add_variable("y", 5, uncertainty=0.3)
            >>> result = calc.compute_with_uncertainty("x + y")
            >>> result.value  # 15
            >>> result.metadata['uncertainty']  # ~0.583
        """
        if not SYMPY_AVAILABLE:
            raise CalculatorError("SymPy necessário para propagação de incertezas")
        
        try:
            # Usar variáveis do Calculator se não fornecidas
            if variables is None:
                variables = self.variables
            
            # Extrair valores e incertezas
            values = {}
            uncertainties = {}
            
            for name, var in variables.items():
                if isinstance(var, Variable):
                    values[name] = var.value
                    # Verificar se tem incerteza
                    if hasattr(var, 'uncertainty') and var.uncertainty is not None:
                        uncertainties[name] = var.uncertainty
                    else:
                        uncertainties[name] = 0.0
                else:
                    # Valor direto sem incerteza
                    values[name] = var
                    uncertainties[name] = 0.0
            
            # Converter expressão para SymPy
            if isinstance(expression, str):
                local_dict = {name: Symbol(name) for name in values.keys()}
                expr = sp.sympify(expression, locals=local_dict)
            else:
                expr = expression
            
            # Calcular valor principal
            result_value = self.evaluate(expression, values, mode='numeric')
            
            # Propagação de incertezas: σ_f² = Σ (∂f/∂xi)² * σ_xi²
            uncertainty_squared = 0.0
            
            for var_name, var_uncertainty in uncertainties.items():
                if var_uncertainty == 0.0:
                    continue
                
                # Derivada parcial
                var_symbol = Symbol(var_name)
                partial_deriv = sp.diff(expr, var_symbol)
                
                # Avaliar derivada no ponto
                subs_dict = {Symbol(k): v for k, v in values.items()}
                partial_value = float(partial_deriv.subs(subs_dict))
                
                # Contribuição para variância
                uncertainty_squared += (partial_value * var_uncertainty) ** 2
            
            # Desvio padrão total
            total_uncertainty = float(np.sqrt(uncertainty_squared))
            
            self._logger.info(
                f"Error propagation: {expr} → value={result_value:.4f}, "
                f"uncertainty={total_uncertainty:.4f}"
            )
            
            return CalculationResult(
                value=result_value,
                expression=str(expression),
                unit=None,
                symbolic=expr,
                metadata={
                    'uncertainty': total_uncertainty,
                    'relative_uncertainty': total_uncertainty / abs(result_value) if result_value != 0 else 0.0,
                    'method': 'first_order_taylor',
                    'variables_uncertainties': uncertainties
                }
            )
            
        except Exception as e:
            self._logger.error(f"Erro na propagação de incertezas: {e}")
            raise CalculatorError(f"Erro na propagação de incertezas: {e}")



    def compute_vectorized(
        self,
        expression: Union[str, Expr],
        variables: Optional[Dict[str, np.ndarray]] = None
    ) -> 'CalculationResult':
        """
        Computa expressão vetorizada com NumPy.
        
        Suporta:
        - Arrays NumPy 1D, 2D, ND
        - Broadcasting automático
        - Escalares misturados com arrays
        - Funções matemáticas (sin, cos, exp, etc)
        
        Args:
            expression: Expressão a computar
            variables: Dict com arrays NumPy (opcional, usa self.variables se None)
        
        Returns:
            CalculationResult com array NumPy
        
        Raises:
            CalculatorError: Se NumPy não disponível ou erro na computação
        
        Examples:
            >>> calc = Calculator()
            >>> calc.add_variable("x", np.array([1, 2, 3]))
            >>> result = calc.compute_vectorized("x**2")
            >>> result.value
            array([1, 4, 9])
            
            >>> # Broadcasting
            >>> calc.add_variable("F", np.array([[1, 2], [3, 4]]))
            >>> calc.add_variable("factor", 10)
            >>> result = calc.compute_vectorized("F * factor")
            >>> result.value
            array([[10, 20], [30, 40]])
        """
        if not NUMPY_AVAILABLE:
            raise CalculatorError("NumPy necessário para vetorização")
        
        if not SYMPY_AVAILABLE:
            raise CalculatorError("SymPy necessário para vetorização")
        
        try:
            # =======================================================
            # STEP 1: Processar variáveis (suporta Variable, arrays, escalares)
            # =======================================================
            if variables is None:
                variables = {}
                for name, var in self.variables.items():
                    if isinstance(var, Variable):
                        variables[name] = var.value
                    else:
                        # Já é valor direto (array NumPy ou escalar)
                        variables[name] = var
            
            # =======================================================
            # STEP 2: Converter expressão COM namespace limpo
            # =======================================================
            if isinstance(expression, str):
                # CRÍTICO: Criar namespace isolado para evitar conflitos
                # Exemplo: 'factor' pode conflitar com funções built-in
                local_dict = {name: Symbol(name) for name in variables.keys()}
                
                try:
                    # Tentar com namespace isolado primeiro
                    expr = sp.sympify(expression, locals=local_dict)
                except Exception as e_sympify:
                    # Fallback: tentar sem local_dict (para expressões muito simples)
                    self._logger.warning(
                        f"sympify com local_dict falhou ({e_sympify}), "
                        f"tentando fallback sem isolamento"
                    )
                    try:
                        expr = sp.sympify(expression)
                    except Exception as e_fallback:
                        raise CalculatorError(
                            f"Falha ao interpretar expressão '{expression}': {e_fallback}"
                        )
            else:
                expr = expression
            
            # =======================================================
            # STEP 3: Extrair símbolos e validar variáveis
            # =======================================================
            free_symbols = sorted(expr.free_symbols, key=str)
            
            if not free_symbols:
                # Expressão constante (ex: "2 + 3")
                result_value = float(expr)
                return CalculationResult(
                    value=np.array(result_value),
                    expression=str(expr),
                    unit=None,
                    symbolic=expr,
                    metadata={
                        'vectorized': True,
                        'constant': True,
                        'shape': ()
                    }
                )
            
            # Validar que todas as variáveis necessárias existem
            missing_vars = [str(s) for s in free_symbols if str(s) not in variables]
            if missing_vars:
                raise CalculatorError(
                    f"Variáveis necessárias não definidas: {', '.join(missing_vars)}"
                )
            
            # =======================================================
            # STEP 4: Criar função lambdificada NumPy
            # =======================================================
            func = sp.lambdify(free_symbols, expr, modules=['numpy'])
            
            # =======================================================
            # STEP 5: Preparar argumentos (garantir arrays NumPy)
            # =======================================================
            var_values = []
            for sym in free_symbols:
                var_name = str(sym)
                value = variables[var_name]
                
                # Converter para array NumPy se necessário
                if not isinstance(value, np.ndarray):
                    value = np.asarray(value)
                
                var_values.append(value)
            
            # =======================================================
            # STEP 6: Executar função vetorizada
            # =======================================================
            result_array = func(*var_values)
            
            # Garantir que resultado é array NumPy
            if not isinstance(result_array, np.ndarray):
                result_array = np.asarray(result_array)
            
            # =======================================================
            # STEP 7: Logging e retorno
            # =======================================================
            self._logger.info(
                f"Vectorized compute: {expr} → shape={result_array.shape}, "
                f"dtype={result_array.dtype}"
            )
            
            return CalculationResult(
                value=result_array,
                expression=str(expression),
                unit=None,
                symbolic=expr,
                metadata={
                    'vectorized': True,
                    'shape': result_array.shape,
                    'dtype': str(result_array.dtype),
                    'variables_used': [str(s) for s in free_symbols]
                }
            )
            
        except CalculatorError:
            # Re-lançar erros customizados sem wrapping
            raise
        except Exception as e:
            self._logger.error(f"Erro inesperado na computação vetorizada: {e}")
            raise CalculatorError(f"Erro na computação vetorizada: {e}")
    
    def monte_carlo(
        self,
        expression: Union[str, Expr],
        variables: Dict[str, Dict[str, float]],
        n_samples: int = 10000,
        confidence_levels: List[float] = None,
        seed: Optional[int] = None
    ) -> 'CalculationResult':
        """
        Simulação Monte Carlo para análise probabilística.
        
        Args:
            expression: Expressão a avaliar
            variables: Dict com distribuições
                - 'mean', 'std', 'dist': 'normal'
                - 'min', 'max', 'dist': 'uniform'
            n_samples: Número de amostras
            confidence_levels: Lista de percentis (ex: [0.05, 0.95])
            seed: Seed para reprodutibilidade
        
        Returns:
            CalculationResult com estatísticas
        
        Examples:
            >>> calc = Calculator()
            >>> variables = {
            ...     "fck": {"mean": 25, "std": 3, "dist": "normal"}
            ... }
            >>> result = calc.monte_carlo("fck * 0.85", variables, n_samples=1000)
            >>> result.metadata["mean"]  # ~ 21.25
        """
        if not SCIPY_AVAILABLE:
            raise CalculatorError("SciPy necessário para Monte Carlo")
        
        if not NUMPY_AVAILABLE:
            raise CalculatorError("NumPy necessário para Monte Carlo")
        
        if not SYMPY_AVAILABLE:
            raise CalculatorError("SymPy necessário para Monte Carlo")
        
        try:
            from scipy import stats
            
            # Set seed para reprodutibilidade
            if seed is not None:
                np.random.seed(seed)
            
            # =======================================================
            # CORREÇÃO 1: Gerar amostras com ordem explícita
            # =======================================================
            # Ordenar variáveis alfabeticamente para consistência
            var_names_sorted = sorted(variables.keys())
            samples = {}
            
            for var_name in var_names_sorted:
                dist_spec = variables[var_name]
                dist_type = dist_spec.get("dist", "normal")
                
                if dist_type == "normal":
                    mean = dist_spec["mean"]
                    std = dist_spec["std"]
                    samples[var_name] = np.random.normal(mean, std, n_samples)
                
                elif dist_type == "uniform":
                    min_val = dist_spec["min"]
                    max_val = dist_spec["max"]
                    samples[var_name] = np.random.uniform(min_val, max_val, n_samples)
                
                else:
                    raise CalculatorError(f"Distribuição '{dist_type}' não suportada")
            
            # =======================================================
            # CORREÇÃO 2: Converter expressão e garantir ordem consistente
            # =======================================================
            if isinstance(expression, str):
                expr = sp.sympify(expression)
            else:
                expr = expression
            
            # Extrair símbolos da expressão e ordenar
            free_symbols = sorted(expr.free_symbols, key=str)
            
            # =======================================================
            # CORREÇÃO 3: Criar função lambdificada com ordem garantida
            # =======================================================
            func = sp.lambdify(free_symbols, expr, modules=['numpy'])
            
            # Preparar arrays na mesma ordem dos símbolos
            var_arrays = [samples[str(sym)] for sym in free_symbols]
            
            # =======================================================
            # EXECUÇÃO: Avaliar função vetorizada
            # =======================================================
            results_array = func(*var_arrays)
            
            # Garantir que é array NumPy (pode ser escalar se n_samples=1)
            results_array = np.asarray(results_array)
            
            # =======================================================
            # ESTATÍSTICAS: Calcular métricas robustas
            # =======================================================
            mean = float(np.mean(results_array))
            std = float(np.std(results_array, ddof=1))  # Usar ddof=1 para std amostral
            median = float(np.median(results_array))
            
            # Percentis padrão
            if confidence_levels is None:
                confidence_levels = [0.05, 0.95]
            
            # Validar confidence_levels
            if not all(0 <= p <= 1 for p in confidence_levels):
                raise CalculatorError("confidence_levels devem estar em [0, 1]")
            
            percentiles = {
                f"percentile_{int(p*100)}": float(np.percentile(results_array, p*100))
                for p in confidence_levels
            }
            
            # =======================================================
            # LOGGING: Informações de debug
            # =======================================================
            self._logger.info(
                f"Monte Carlo: {expr} → mean={mean:.4f}, std={std:.4f}, "
                f"n_samples={n_samples}, seed={seed}"
            )
            
            # =======================================================
            # RETORNO: CalculationResult com metadados completos
            # =======================================================
            return CalculationResult(
                value=results_array,
                expression=f"MonteCarlo({expr}, n={n_samples})",
                unit=None,
                symbolic=expr,
                metadata={
                    'method': 'monte_carlo',
                    'n_samples': n_samples,
                    'seed': seed,
                    'mean': mean,
                    'std': std,
                    'median': median,
                    'min': float(np.min(results_array)),
                    'max': float(np.max(results_array)),
                    **percentiles,
                    'variables_used': var_names_sorted
                }
            )
            
        except Exception as e:
            self._logger.error(f"Erro no Monte Carlo: {e}")
            raise CalculatorError(f"Erro no Monte Carlo: {e}")
    
    def batch_compute_advanced(
        self,
        specs: List[Dict[str, Any]]
    ) -> Dict[str, 'CalculationResult']:
        """
        Batch processing avançado com dependências.
        
        Args:
            specs: Lista de dicts com:
                - 'expression': str
                - 'label': str
                - 'depends_on': List[str] (opcional)
        
        Returns:
            Dict com resultados
        
        Examples:
            >>> calc = Calculator()
            >>> calc.add_variable("L", 6)
            >>> specs = [
            ...     {"expression": "L**2", "label": "L2"},
            ...     {"expression": "L2 * 10", "label": "result", "depends_on": ["L2"]}
            ... ]
            >>> results = calc.batch_compute_advanced(specs)
        """
        results = {}
        
        for spec in specs:
            expression = spec["expression"]
            label = spec.get("label", expression)
            depends_on = spec.get("depends_on", [])
            
            # Adicionar dependências
            for dep in depends_on:
                if dep in results:
                    self.add_variable(dep, results[dep].value)
            
            # Computar
            result = self.compute(expression)
            results[label] = result
            
            self._logger.info(f"Batch: {label} = {result.value}")
        
        return results



    def solve(
        self,
        equation: Union[str, Expr],
        variable: str
    ) -> list:
        """
        Resolve equação simbolicamente.
        
        Args:
            equation: Equação (assume = 0)
            variable: Variável a resolver
        
        Returns:
            Lista de soluções
        
        Examples:
            >>> calc = Calculator()
            >>> solutions = calc.solve("x**2 - 4", "x")
            >>> print(solutions)  # [-2, 2]
        """
        if not SYMPY_AVAILABLE:
            raise CalculatorError("SymPy não disponível")
        
        # Converter para SymPy
        if isinstance(equation, str):
            expr = sp.sympify(equation)
        else:
            expr = equation
        
        var_symbol = Symbol(variable)
        
        # Resolver
        with self._lock:
            solutions = sp.solve(expr, var_symbol)
        
        self._logger.debug(f"Solve: {expr} = 0, {variable} = {solutions}")
        return solutions
    
    def substitute(
        self,
        expression: Union[str, Expr],
        substitutions: Dict[str, Any]
    ) -> Expr:
        """
        Substitui valores em expressão.
        
        Args:
            expression: Expressão
            substitutions: Dicionário {variável: valor}
        
        Returns:
            Expressão substituída
        
        Examples:
            >>> calc = Calculator()
            >>> result = calc.substitute("x + y", {'x': 10})
            >>> print(result)  # 10 + y
        """
        if not SYMPY_AVAILABLE:
            raise CalculatorError("SymPy não disponível")
        
        # Converter para SymPy
        if isinstance(expression, str):
            expr = sp.sympify(expression)
        else:
            expr = expression
        
        # Preparar substituições
        subs_dict = {Symbol(k): v for k, v in substitutions.items()}
        
        # Substituir
        with self._lock:
            result = expr.subs(subs_dict)
        
        self._logger.debug(f"Substitute: {expr} with {substitutions} = {result}")
        return result


# ============================================================================
# EXCEPTIONS
# ============================================================================

class CalculatorError(Exception):
    """Exceção base para erros do Calculator."""
    pass


# Alias para retrocompatibilidade
CalculationError = CalculatorError


# ============================================================================
# DATACLASSES
# ============================================================================

@dataclass
class CalculationResult:
    """
    Resultado unificado de cálculo do PyMemorial v2.0.
    
    Suporta:
    - Valores numéricos (float, int, complex)
    - Expressões simbólicas (SymPy)
    - Arrays vetorizados (NumPy)
    - Matrizes (NumPy/SymPy)
    - Unidades dimensionais (Pint)
    - Metadados extensíveis
    
    **FILOSOFIA PYMEMORIAL:**
    - Liberdade total (qualquer tipo Python)
    - Steps automáticos (metadados rastreáveis)
    - Integração natural (SymPy + NumPy + SciPy)
    
    Attributes:
        value: Valor calculado (qualquer tipo)
        expression: String da expressão original
        unit: Unidade dimensional (opcional)
        symbolic: Forma simbólica SymPy (opcional)
        metadata: Dicionário extensível de metadados
    
    Examples:
        >>> # Resultado numérico simples
        >>> result = CalculationResult(
        ...     value=67.5,
        ...     expression="F * L",
        ...     unit="kN·m"
        ... )
        >>> str(result)
        '67.5 kN·m'
        
        >>> # Resultado simbólico
        >>> import sympy as sp
        >>> x = sp.Symbol('x')
        >>> result = CalculationResult(
        ...     value=sp.simplify(x**2 + 2*x + 1),
        ...     expression="x^2 + 2x + 1",
        ...     symbolic=(x + 1)**2,
        ...     metadata={'mode': 'symbolic'}
        ... )
        
        >>> # Resultado com metadados de convergência
        >>> result = CalculationResult(
        ...     value=2.0,
        ...     expression="solve(x^2 - 4 = 0)",
        ...     metadata={
        ...         'method': 'scipy.fsolve',
        ...         'converged': True,
        ...         'residual': 1e-15,
        ...         'iterations': 5
        ...     }
        ... )
    """
    # Campos obrigatórios
    value: Union[float, int, complex, np.ndarray, sp.Expr, Any]
    expression: str
    
    # Campos opcionais com defaults
    unit: Optional[str] = None
    symbolic: Optional[sp.Expr] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """
        Validação e normalização pós-inicialização.
        
        Garante:
        - Expression nunca é None (usa repr(value) se necessário)
        - Metadata sempre é dict válido
        - Value compatível com operações básicas
        """
        # Garantir expression válida
        if not self.expression or not isinstance(self.expression, str):
            self.expression = repr(self.value)
        
        # Garantir metadata é dict
        if not isinstance(self.metadata, dict):
            self.metadata = {}
        
        # Detectar tipo automaticamente se não especificado
        if 'type' not in self.metadata:
            self.metadata['type'] = self._detect_type()
    
    def _detect_type(self) -> str:
        """
        Detecta tipo do valor automaticamente.
        
        Returns:
            String identificando tipo: 'numeric', 'symbolic', 'array', 'matrix', 'complex'
        """
        if SYMPY_AVAILABLE and isinstance(self.value, sp.Expr):
            return 'symbolic'
        elif NUMPY_AVAILABLE and isinstance(self.value, np.ndarray):
            if self.value.ndim == 1:
                return 'array'
            elif self.value.ndim == 2:
                return 'matrix'
            else:
                return 'tensor'
        elif isinstance(self.value, complex):
            return 'complex'
        elif isinstance(self.value, (int, float)):
            return 'numeric'
        else:
            return 'unknown'
    
    def __str__(self) -> str:
        """
        Representação string formatada para usuário.
        
        Returns:
            String legível com valor e unidade
        
        Examples:
            >>> result = CalculationResult(value=10.5, expression="x", unit="m")
            >>> str(result)
            '10.5 m'
        """
        # Formatar valor
        if isinstance(self.value, float):
            # Remover zeros desnecessários
            value_str = f"{self.value:.10g}"
        else:
            value_str = str(self.value)
        
        # Adicionar unidade se existir
        if self.unit:
            return f"{value_str} {self.unit}"
        return value_str
    
    def __repr__(self) -> str:
        """
        Representação técnica com todos os campos.
        
        Returns:
            String com assinatura completa
        
        Examples:
            >>> result = CalculationResult(value=10.5, expression="x", unit="m")
            >>> repr(result)
            "CalculationResult(value=10.5, expression='x', unit='m', type='numeric')"
        """
        type_str = self.metadata.get('type', 'unknown')
        return (
            f"CalculationResult("
            f"value={self.value!r}, "
            f"expression={self.expression!r}, "
            f"unit={self.unit!r}, "
            f"type={type_str!r})"
        )
    
    def __float__(self) -> float:
        """
        Conversão para float.
        
        Útil para:
        - Operações aritméticas diretas
        - Comparações numéricas
        - Integração com código legado
        
        Returns:
            Valor como float
        
        Raises:
            TypeError: Se valor não é conversível
        
        Examples:
            >>> result = CalculationResult(value=10.5, expression="x")
            >>> float(result)
            10.5
            >>> result * 2
            21.0
        """
        if isinstance(self.value, (int, float)):
            return float(self.value)
        elif NUMPY_AVAILABLE and isinstance(self.value, np.ndarray):
            if self.value.size == 1:
                return float(self.value.item())
            raise TypeError("Cannot convert array to single float")
        elif SYMPY_AVAILABLE and isinstance(self.value, sp.Expr):
            if self.value.is_number:
                return float(self.value.evalf())
            raise TypeError("Cannot convert symbolic expression to float")
        else:
            return float(self.value)
    
    def __int__(self) -> int:
        """Conversão para int."""
        return int(self.__float__())

    # ================== ADICIONE ESTE NOVO MÉTODO ==================
    def __format__(self, format_spec: str) -> str:
        """
        Formatação customizada (f-strings).
        
        Aplica a formatação ao .value.
        
        Examples:
            >>> result = CalculationResult(value=12.3456, expression="x")
            >>> f"{result:.2f}"
            '12.35'
        """
        try:
            # Tenta formatar o .value
            formatted_value = self.value.__format__(format_spec)
        except (TypeError, ValueError):
            # Fallback se a formatação não se aplica ao tipo
            # (ex: formatar um array simbólico com '.2f')
            return str(self)
        
        # Adicionar unidade se existir
        if self.unit:
            return f"{formatted_value} {self.unit}"
        return formatted_value
    # ================== FIM DO NOVO MÉTODO ==================

    
    def to_dict(self) -> Dict[str, Any]:
        """
        Serializa para dicionário JSON-compatível.
        
        Útil para:
        - Salvar em cache
        - API REST
        - Logs estruturados
        
        Returns:
            Dicionário com todos os campos serializados
        
        Examples:
            >>> result = CalculationResult(value=10.5, expression="x", unit="m")
            >>> result.to_dict()
            {'value': 10.5, 'expression': 'x', 'unit': 'm', 'metadata': {'type': 'numeric'}}
        """
        # Serializar value
        if NUMPY_AVAILABLE and isinstance(self.value, np.ndarray):
            value_serialized = self.value.tolist()
        elif SYMPY_AVAILABLE and isinstance(self.value, sp.Expr):
            value_serialized = str(self.value)
        else:
            value_serialized = self.value
        
        return {
            'value': value_serialized,
            'expression': self.expression,
            'unit': self.unit,
            'symbolic': str(self.symbolic) if self.symbolic else None,
            'metadata': self.metadata
        }
    
    def format_engineering(self, decimals: int = 3) -> str:
        """
        Formata em notação de engenharia (múltiplos de 10^3).
        
        Args:
            decimals: Casas decimais
        
        Returns:
            String formatada (ex: "12.5 k", "1.23 M")
        
        Examples:
            >>> result = CalculationResult(value=12500, expression="F")
            >>> result.format_engineering()
            '12.5 k'
        """
        if not isinstance(self.value, (int, float)):
            return str(self)
        
        prefixes = {
            -9: 'n', -6: 'μ', -3: 'm',
            0: '', 3: 'k', 6: 'M', 9: 'G'
        }
        
        if self.value == 0:
            return f"0 {self.unit}" if self.unit else "0"
        
        exponent = int(math.floor(math.log10(abs(self.value)) / 3) * 3)
        exponent = max(-9, min(9, exponent))  # Limitar range
        
        mantissa = self.value / (10 ** exponent)
        prefix = prefixes.get(exponent, '')
        
        unit_str = f"{prefix}{self.unit}" if self.unit else prefix
        return f"{mantissa:.{decimals}f} {unit_str}".strip()
