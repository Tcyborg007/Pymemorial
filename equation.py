"""
PyMemorial v2.0 - Equation Module
Motor simbólico com SymPy + Sistema de Steps (Calcpad-inspired)

FILOSOFIA:
- Sintaxe Python natural (Handcalcs-inspired)
- Steps automáticos com 4 níveis de granularidade (Calcpad-inspired)
- Validação dimensional completa
- Thread-safe e cache-optimized

============================================================================
REVISÃO (Refatoração):
- Removida a classe 'GranularityType' e toda a lógica de granularidade.
- A geração de steps foi padronizada para o formato:
  [Fórmula Simbólica] -> [Fórmula Numérica] -> [Resultado]
- Simplificados os métodos Equation.steps(), Equation.generate_memorial(),
  Equation.generate_memorial_detailed() e StepRegistry.register()
  para usar apenas o gerador padrão, removendo o parâmetro 'granularity'.
- Simplificado StepType para manter apenas FORMULA, SUBSTITUTION, RESULT.
- Simplificado StepGenerator para ter apenas um método de geração.
- Simplificado StepRegistry.to_markdown() para o novo formato.
============================================================================
"""

import ast
import re
import logging
import threading
from decimal import Decimal
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

# ----------------------------------------------------------------------------
# REVISÃO: GranularityType removido.
# ----------------------------------------------------------------------------
# class GranularityType(Enum): ... (REMOVIDO)


class StepType(Enum):
    """
    Tipos de steps de cálculo.
    REVISÃO: Simplificado para o formato padrão.
    """
    FORMULA = "formula"              # Fórmula simbólica
    SUBSTITUTION = "substitution"    # Substituição de valores
    RESULT = "result"                # Resultado final
    # --- REMOVIDOS ---
    # CALCULATION = "calculation"
    # SIMPLIFICATION = "simplification"
    # EXPLANATION = "explanation"


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
        Parse string para SymPy Expr com validação rigorosa e tratamento de funções.

        Args:
            expr_str: Expressão como string (APENAS o lado direito, ex: '5600 * sqrt(fck)')

        Returns:
            SymPy Expr

        Raises:
            ValidationError: Se parsing ou sintaxe falhar
        """
        # --- DEFINIÇÃO DE FUNÇÕES CONHECIDAS (CLASSE-LEVEL) ---
        KNOWN_FUNCTIONS = {
            'sqrt', 'exp', 'log', 'ln', 'log10', 'log2',
            'sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'atan2',
            'sinh', 'cosh', 'tanh', 'asinh', 'acosh', 'atanh',
            'abs', 'sign', 'floor', 'ceiling', 'factorial',
            'min', 'max', 'sum', 'prod',
            'pi'  # ✅ REMOVER 'e' DESTA LISTA - e será tratado como variável
        }
        
        # ✅ ADICIONAR ESTA SEÇÃO NOVA: PROTEÇÃO PARA CONSTANTES QUE PODEM SER VARIÁVEIS
        # Constantes SymPy que podem conflitar com nomes de variáveis:
        # - 'e' (número de Euler)
        # - 'I' (unidade imaginária)
        # - 'E' (número de Euler - maiúsculo)
        AMBIGUOUS_CONSTANTS = {'e', 'E', 'I', 'N', 'S', 'O'}
        
        # Se alguma dessas constantes está definida como variável no locals_dict,
        # deve ser tratada como variável, não como constante
        protected_constants = set()
        if self.locals_dict:
            for name in AMBIGUOUS_CONSTANTS:
                if name in self.locals_dict:
                    protected_constants.add(name)
                    self._logger.debug(f"Constante '{name}' protegida - será tratada como variável")
        
        try:
            # --- VALIDAÇÕES DE SINTAXE ---
            if re.search(r'[\+\-]{3,}', expr_str):
                raise ValidationError(f"Sintaxe inválida: Operadores consecutivos '+++' ou '---': '{expr_str}'")
            
            try:
                ast.parse(expr_str, mode='eval')
            except SyntaxError as syn_e:
                raise ValidationError(f"Sintaxe inválida na expressão '{expr_str}': {syn_e}")

            # --- CRIAÇÃO DO DICIONÁRIO LOCAL PARA SYMPY ---
            local_symbols_for_sympy = {}

            # 1. Encontrar todas as "palavras" (prováveis variáveis ou funções)
            potential_names = set(re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', expr_str))

            # 2. Iterar sobre os nomes potenciais
            for name in potential_names:
                # IGNORAR nomes de funções conhecidas
                if name.lower() in KNOWN_FUNCTIONS:
                    continue  # Deixa SymPy resolver a função

                # ✅ FORÇAR TRATAMENTO COMO SÍMBOLO PARA CONSTANTES PROTEGIDAS
                if name in protected_constants:
                    local_symbols_for_sympy[name] = Symbol(name)
                    continue
                
                # Se o nome existe no self.locals_dict, cria um Símbolo para ele
                elif self.locals_dict and name in self.locals_dict:
                    local_symbols_for_sympy[name] = Symbol(name)
                # Se não está nas variáveis locais E não é função conhecida, assume que é um Símbolo
                else:
                    local_symbols_for_sympy[name] = Symbol(name)

            # 3. Parsing com SymPy, usando os símbolos locais filtrados.
            self._logger.debug(f"Parsing '{expr_str}' with local symbols: {list(local_symbols_for_sympy.keys())}")
            expr = sympify(expr_str, locals=local_symbols_for_sympy)

            # Validação extra: Verifica se o resultado é uma expressão válida
            if not isinstance(expr, Expr):
                if isinstance(expr, (int, float)):
                    return sp.Number(expr)
                raise ValidationError(f"Parsing result is not a valid SymPy expression: {type(expr)}")

            return expr

        except ValidationError:
            raise  # Propagar nosso erro de sintaxe
        except TypeError as te:
            if "'Symbol' object is not callable" in str(te):
                callable_symbol_match = re.search(r"\'(\w+)\'\s*object is not callable", str(te))
                culprit = callable_symbol_match.group(1) if callable_symbol_match else "desconhecido"
                raise ValidationError(
                    f"Erro ao parsear expressão '{expr_str}': "
                    f"O símbolo '{culprit}' foi usado como função, mas é uma variável. "
                    f"Verifique se '{culprit}' está definido corretamente ou se há conflito de nome."
                ) from te
            else:
                raise ValidationError(f"Erro de tipo ao parsear expressão '{expr_str}': {te}") from te
        except Exception as e:
            self._logger.error(f"Unexpected error parsing '{expr_str}': {e}", exc_info=True)
            raise ValidationError(
                f"Erro inesperado ao parsear expressão '{expr_str}': {e}"
            )


    # ----------------------------------------------------------------------------
    # REVISÃO: 'granularity' e 'show_intermediate' removidos da assinatura.
    # O método agora delega para self.steps() que usa o formato padrão.
    # ----------------------------------------------------------------------------
    def generate_memorial(
        self,
        variables: Optional[Dict[str, Any]] = None,
        # granularity: str = 'medium', # <-- REMOVIDO
        precision: int = 3,
        show_units: bool = True,
        show_descriptions: bool = True,
        # show_intermediate: bool = False, # <-- REMOVIDO
        **kwargs
    ) -> List[Step]:
        """
        Método de conveniência para gerar memorial declarativo padrão.
        
        Workflow: Fórmula Simbólica → Substituição → Resultado
        
        Args:
            variables: Dicionário de variáveis (opcional)
            precision: Casas decimais
            show_units: Incluir unidades
            show_descriptions: Incluir descrições das variáveis
            **kwargs: Variáveis adicionais via keywords
            
        Returns:
            Lista de Steps formatados
        """
        # Unificar contexto (igual ao evaluate e steps)
        eval_context = {}
        if self.locals_dict:
            eval_context.update(self.locals_dict)
        if variables:
            eval_context.update(variables)
        if kwargs:
            eval_context.update(kwargs)
        
        # ✅ REDIRECIONAR PARA O MÉTODO steps() QUE TEM A LÓGICA PADRÃO
        return self.steps(
            variables=eval_context,
            precision=precision,
            show_units=show_units,
            show_descriptions=show_descriptions
        )

    # ----------------------------------------------------------------------------
    # REVISÃO: Método mantido por compatibilidade, mas refatorado
    # para usar o novo gerador padrão (self.steps).
    # A lógica interna complexa foi removida.
    # ----------------------------------------------------------------------------
    def generate_memorial_detailed(
        self,
        precision: int = 3,
        show_units: bool = True,
        show_descriptions: bool = True
        # show_intermediate: bool = True, # <-- REMOVIDO
        # break_down_operations: bool = True # <-- REMOVIDO
    ) -> List[Step]:
        """
        Gera memorial no formato padrão:
        Fórmula Simbólica -> Fórmula Numérica -> Resultado
        (Mantido por compatibilidade, agora usa o gerador padrão)
        """
        if not SYMPY_AVAILABLE:
            raise ImportError("SymPy não disponível para memorial detalhado.")
        
        if not self.expr:
            raise ValueError("Equação não possui expressão válida.")
        
        # Chama self.steps (que agora só tem 1 formato)
        # passando o contexto local da equação.
        return self.steps(
            variables=self.locals_dict,
            precision=precision,
            show_units=show_units,
            show_descriptions=show_descriptions
        )

    
    # ----------------------------------------------------------------------------
    # REVISÃO: _extract_sub_expressions removido
    # Não é mais usado por generate_memorial_detailed.
    # ----------------------------------------------------------------------------
    # def _extract_sub_expressions(self, expr) -> List[Tuple[str, Any]]: ... (REMOVIDO)


    # Dentro da classe Equation
    def _validate_dimensions(self) -> Optional[str]:
        """
        Valida compatibilidade dimensional da equação usando Pint.

        Percorre a árvore de expressão SymPy e verifica as unidades
        das operações (soma, subtração, multiplicação, etc.).

        Returns:
            String da unidade resultante se consistente, None se
            não for possível determinar ou se Pint não estiver disponível.

        Raises:
            DimensionalError: Se dimensões forem incompatíveis.
        """
        if not PINT_AVAILABLE or not self.locals_dict:
            self._logger.debug("Validação dimensional pulada: Pint indisponível ou locals_dict vazio.")
            return None # Não é possível validar

        ureg = get_unit_registry().ureg # Obter o registro do Pint
        if not ureg:
            self._logger.warning("Validação dimensional pulada: Falha ao obter ureg do Pint.")
            return None

        # Mapear símbolos SymPy para unidades Pint (ou adimensional)
        unit_map = {}
        default_unit = None  # ← NOVO: Detectar unidade dominante
        
        for sym in self.free_symbols:
            var_name = str(sym)
            if var_name in self.locals_dict and self.locals_dict[var_name].unit:
                try:
                    unit_map[sym] = ureg(self.locals_dict[var_name].unit)
                    # Captura a primeira unidade não-adimensional como padrão
                    if default_unit is None and not unit_map[sym].dimensionless:
                        default_unit = unit_map[sym]
                except Exception as e:
                    self._logger.warning(f"Falha ao parsear unidade '{self.locals_dict[var_name].unit}' para '{var_name}': {e}")
                    unit_map[sym] = ureg.dimensionless
            else:
                # Variável sem unidade definida ou não no locals_dict -> Adimensional
                unit_map[sym] = ureg.dimensionless

        try:
            # Função recursiva para percorrer a árvore e verificar unidades
            def get_expression_unit(expr):
                if expr.is_Symbol:
                    return unit_map.get(expr, ureg.dimensionless)
                elif expr.is_Number:
                    # ← MELHORIA: Números literais assumem unidade do contexto
                    # Se houver operação de soma/subtração com unidades, o número herda a unidade
                    return ureg.dimensionless  # Será tratado especialmente em Add
                elif expr.is_Add:
                    # Soma/Subtração: Todas as unidades devem ser compatíveis
                    units = [get_expression_unit(arg) for arg in expr.args]
                    
                    # ← NOVA LÓGICA: Filtrar unidades não-adimensionais
                    non_dimensionless_units = [u for u in units if not u.dimensionless]
                    
                    if not non_dimensionless_units:
                        # Todos os termos são adimensionais
                        return ureg.dimensionless
                    
                    # Usar a primeira unidade não-adimensional como referência
                    first_unit = non_dimensionless_units[0]
                    
                    # Verificar compatibilidade apenas entre unidades não-adimensionais
                    for other_unit in non_dimensionless_units[1:]:
                        if not first_unit.is_compatible_with(other_unit):
                            raise DimensionalError(
                                f"Incompatibilidade dimensional em soma/subtração: "
                                f"'{first_unit:~P}' vs '{other_unit:~P}' na expressão '{expr}'"
                            )
                    
                    return first_unit  # A unidade resultante é a dos termos não-adimensionais
                    
                elif expr.is_Mul:
                    # Multiplicação: Unidades se multiplicam
                    result_unit = ureg.dimensionless
                    for arg in expr.args:
                        result_unit *= get_expression_unit(arg)
                    return result_unit
                elif expr.is_Pow:
                    # Potenciação: Unidade da base elevada ao expoente (deve ser adimensional)
                    base_unit = get_expression_unit(expr.base)
                    exp_unit = get_expression_unit(expr.exp)
                    if not exp_unit.dimensionless:
                        raise DimensionalError(
                           f"Expoente deve ser adimensional, mas '{expr.exp}' tem unidade '{exp_unit:~P}'"
                        )
                    # O expoente precisa ser um valor numérico para o Pint
                    try:
                        exponent_value = float(expr.exp)
                        return base_unit ** exponent_value
                    except Exception:
                        self._logger.warning(f"Não foi possível validar dimensão de potenciação com expoente não numérico: {expr}")
                        return ureg.dimensionless

                elif expr.is_Function:
                    # Funções (sin, cos, log, etc.): Argumento geralmente deve ser adimensional
                    arg_units = [get_expression_unit(arg) for arg in expr.args]
                    func_name = expr.func.__name__.lower()
                    
                    # ← NOVO: Funções que preservam unidades
                    if func_name in ('max', 'min', 'abs', 'sign'):
                        # Essas funções retornam a mesma unidade dos argumentos
                        # Verifica compatibilidade e retorna a primeira unidade não-adimensional
                        non_dimensionless = [u for u in arg_units if not u.dimensionless]
                        if non_dimensionless:
                            # Verifica compatibilidade entre argumentos
                            first_unit = non_dimensionless[0]
                            for other_unit in non_dimensionless[1:]:
                                if not first_unit.is_compatible_with(other_unit):
                                    raise DimensionalError(
                                        f"Argumentos de '{func_name}' têm unidades incompatíveis: "
                                        f"'{first_unit:~P}' vs '{other_unit:~P}'"
                                    )
                            return first_unit
                        return ureg.dimensionless
                    
                    # Funções trigonométricas exigem argumento adimensional
                    if func_name in ('sin', 'cos', 'tan', 'asin', 'acos', 'atan'):
                        for i, arg_unit in enumerate(arg_units):
                            if not arg_unit.dimensionless:
                                raise DimensionalError(
                                    f"Argumento de '{func_name}' deve ser adimensional, mas '{expr.args[i]}' tem unidade '{arg_unit:~P}'"
                                )
                    
                    # sqrt preserva a dimensão (√[L] = L^0.5)
                    if func_name == 'sqrt' and len(arg_units) == 1:
                        return arg_units[0] ** 0.5
                    
                    # Resultado da maioria dessas funções é adimensional
                    return ureg.dimensionless
                else:
                    # Tipo de expressão não tratado
                    self._logger.warning(f"Validação dimensional não implementada para tipo: {type(expr)}")
                    return ureg.dimensionless

            # Calcular a unidade resultante da expressão inteira
            result_unit = get_expression_unit(self.expr)
            result_unit_str = f"{result_unit:~P}" # Formato bonito do Pint
            self._logger.info(f"Validação dimensional para '{self.expression_str}': Unidade resultante = {result_unit_str}")
            return result_unit_str

        except DimensionalError as de:
            self._logger.error(f"Erro dimensional em '{self.expression_str}': {de}")
            raise de # Propaga o erro dimensional
        except Exception as e:
            # Captura outros erros inesperados durante a validação
            self._logger.error(f"Erro inesperado durante validação dimensional de '{self.expression_str}': {e}")
            raise EquationError(f"Falha na validação dimensional: {e}") from e
    
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
    
    def evaluate(
        self,
        variables: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> EvaluationResult:
        """
        Avalia equação numericamente.
        
        Permite variáveis via dicionário ou kwargs.
        
        Args:
            variables: Dicionário de variáveis (opcional).
            **kwargs: Valores de variáveis como keyword arguments.
                     (Têm prioridade sobre 'variables' e 'locals_dict')
        
        Returns:
            EvaluationResult com valor e metadados
        
        Raises:
            EvaluationError: Se avaliação falhar
        """
        config = get_config()
        
        # ============================================================
        # CORREÇÃO: Criar um contexto de avaliação unificado
        # Prioridade: kwargs > variables > self.locals_dict
        # ============================================================
        eval_context = {}
        if self.locals_dict:
            eval_context.update(self.locals_dict)
        if variables:
            eval_context.update(variables)
        if kwargs:
            eval_context.update(kwargs)
        
        # Mesclar contexto unificado com variáveis da equação
        subs_dict = {}
        for name in self.variables_used:
            if name in eval_context:
                value = eval_context[name]
                
                # Extrair valor se for um objeto Variable
                if isinstance(value, Variable):
                    subs_dict[name] = value.value
                else:
                    subs_dict[name] = value
            else:
                # Variável necessária não foi encontrada em lugar nenhum
                raise EvaluationError(
                    f"Variável '{name}' não definida no contexto de avaliação"
                )
        
        # Substituir e avaliar
        try:
            # Substituir símbolos por valores
            expr_with_values = self.expr.subs({
                Symbol(k): v for k, v in subs_dict.items()
            })
            
            # Avaliar numericamente
            evaluated_expr = expr_with_values.evalf()
            
            # ============================================================
            # CORREÇÃO: Lidar com resultados complexos (ex: 0.5 + 0.0j)
            # ============================================================
            
            # Tentar extrair partes real e imaginária
            try:
                val_real = sp.re(evaluated_expr).evalf(chop=True)
                val_imag = sp.im(evaluated_expr).evalf(chop=True)
            except Exception:
                # Pode falhar se for um tipo não-numérico
                val_real = evaluated_expr
                val_imag = 0.0

            # Se a parte imaginária for zero (ou muito pequena), usar a real
            if abs(float(val_imag)) < 1e-15:
                result_value = float(val_real)
            else:
                # O número é genuinamente complexo
                raise EvaluationError(
                     f"Erro ao avaliar '{self.expression_str}': "
                     f"Resultado é um número complexo não-real: {evaluated_expr}"
                )
            
            return EvaluationResult(
                value=result_value,
                expression=self.expression_str,
                symbolic=self.expr,
                metadata={'substitutions': subs_dict}
            )
            
        except Exception as e:
            # Se a conversão acima falhar, ou outro erro ocorrer
            if "Resultado é um número complexo" in str(e):
                 raise e # Propagar nosso erro customizado
                 
            raise EvaluationError(
                f"Erro ao avaliar '{self.expression_str}': {e}"
            )

    # ----------------------------------------------------------------------------
    # REVISÃO: Lógica de 'granularity' e 'show_intermediate' removida.
    # O método agora sempre chama o gerador padrão.
    # ----------------------------------------------------------------------------
    def steps(
        self,
        variables: Optional[Dict[str, Any]] = None,
        # granularity: str = 'medium', # <-- REMOVIDO
        precision: int = 3,
        show_units: bool = True,
        show_descriptions: bool = True,
        # show_intermediate: bool = True, # <-- REMOVIDO
        **kwargs
    ) -> List[Step]:
        """
        Gera lista de Steps no formato padrão:
        Fórmula Simbólica -> Fórmula Numérica -> Resultado
        
        Args:
            variables: Dicionário de variáveis (opcional, usa locals_dict se None)
            precision: Casas decimais
            show_units: Incluir unidades
            show_descriptions: Incluir descrições das variáveis
            **kwargs: Argumentos adicionais (passados como variáveis)
            
        Returns:
            Lista de Steps
        """
        # Unificar contexto
        eval_context = {}
        if self.locals_dict:
            eval_context.update(self.locals_dict)
        if variables:
            eval_context.update(variables)
        if kwargs:
            # Adicionar kwargs ao contexto de avaliação
            eval_context.update(kwargs)

        generator = StepGenerator()

        # Chama o único gerador disponível
        # Passa show_units etc. como kwargs para o generator.generate
        return generator.generate(
            equation=self,
            variables=eval_context,
            precision=precision,
            show_units=show_units,
            show_descriptions=show_descriptions
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
    
    # Dentro da classe ValidationHelpers
    @staticmethod
    def check_circular_dependencies(
        equations: Dict[str, Equation] # Usar Dict[nome, eq] é mais eficiente
    ) -> List[List[str]]: # Retorna lista de ciclos encontrados
        """
        Detecta dependências circulares entre um conjunto de equações nomeadas.

        Usa busca em profundidade (DFS) para encontrar ciclos no grafo de dependências.

        Args:
            equations: Dicionário mapeando nome da equação para o objeto Equation.

        Returns:
            Lista de ciclos, onde cada ciclo é uma lista de nomes de equações
            formando a dependência circular (ex: [['A', 'B', 'A'], ['C', 'C']]).
            Retorna lista vazia se não houver ciclos.
        """
        adj = {name: eq.get_variables() for name, eq in equations.items() if eq.name}
        all_nodes = set(adj.keys())

        visited = set() # Nós já completamente explorados
        recursion_stack = set() # Nós atualmente na pilha de chamada DFS
        cycles = [] # Lista para armazenar os ciclos encontrados

        # Armazena o caminho atual para reconstruir o ciclo
        path = []

        def dfs(node):
            if node not in all_nodes: # Variável externa, não faz parte do ciclo interno
                 return False

            visited.add(node)
            recursion_stack.add(node)
            path.append(node)

            if node in adj: # Verifica se o nó (equação) tem dependências
                for neighbor in adj[node]:
                    if neighbor not in visited:
                        if dfs(neighbor):
                            return True # Ciclo encontrado em chamada recursiva
                    elif neighbor in recursion_stack:
                        # CICLO DETECTADO!
                        try:
                            # Encontra onde o ciclo começa no caminho atual
                            cycle_start_index = path.index(neighbor)
                            cycle = path[cycle_start_index:] + [neighbor] # Adiciona o nó vizinho para fechar o ciclo visualmente
                            # Evita adicionar o mesmo ciclo múltiplas vezes
                            if sorted(cycle) not in [sorted(c) for c in cycles]:
                                cycles.append(cycle)
                        except ValueError:
                             # Deveria estar no path se está na recursion_stack, mas por segurança...
                             pass # Ignora se não encontrar
                        return True # Indica que um ciclo foi encontrado

            # Backtrack: remove o nó da pilha de recursão e do caminho
            recursion_stack.remove(node)
            path.pop()
            return False # Nenhum ciclo encontrado a partir deste nó

        # Executa DFS para cada nó não visitado
        for node in list(all_nodes): # Usa list() para poder modificar 'visited' durante a iteração
            if node not in visited:
                dfs(node) # Ignora o retorno aqui, pois 'cycles' armazena o resultado

        return cycles
    
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
# STEPGENERATOR - GERADOR DE STEPS (VERSÃO REVISADA)
# ============================================================================
class StepGenerator:
    """
    Gerador automático de steps estilo Calcpad.
    REVISÃO: Simplificado para gerar apenas o formato padrão:
    Fórmula Simbólica -> Fórmula Numérica -> Resultado
    """

    def __init__(self):
        """Inicializa gerador."""
        self._logger = logging.getLogger(__name__)

    # --- MÉTODO PÚBLICO PRINCIPAL ---

    # ----------------------------------------------------------------------------
    # REVISÃO: Assinatura simplificada. Remove 'granularity'.
    # Delega para o único método gerador: _generate_standard_memorial
    # ----------------------------------------------------------------------------
    def generate(
        self,
        equation: Equation,
        variables: Dict[str, Any],
        precision: int = 3,
        **kwargs # Captura show_units, show_descriptions
    ) -> List[Step]:
        """
        Gera steps de cálculo no formato padrão:
        Fórmula Simbólica -> Fórmula Numérica -> Resultado

        Args:
            equation: Equation para gerar steps
            variables: Dicionário de variáveis para avaliação
            precision: Casas decimais
            **kwargs: Aceita 'show_units' e 'show_descriptions'

        Returns:
            Lista de Steps
        """
        show_units = kwargs.get('show_units', True)
        # show_descriptions não é mais usado para 'Dados de Entrada',
        # mas mantemos para consistência.
        show_descriptions = kwargs.get('show_descriptions', True)

        # Chama o único gerador padrão
        return self._generate_standard_memorial(
            equation,
            variables,
            precision,
            show_units,
            show_descriptions
        )

    # --- MÉTODO GERADOR PADRÃO ---

    # ----------------------------------------------------------------------------
    # REVISÃO: Este é o antigo 'generate_declarative_memorial',
    # renomeado e simplificado para ser o único gerador.
    # Removemos os steps de EXPLANATION e DADOS DE ENTRADA.
    # ----------------------------------------------------------------------------
    def _generate_standard_memorial(
        self,
        equation: Equation,
        variables: Dict[str, Any],
        precision: int = 3,
        show_units: bool = True,
        show_descriptions: bool = True # Mantido por compatibilidade
    ) -> List[Step]:
        """
        Gera memorial no formato: Simbólica → Numérica → Resultado
        
        Workflow:
        1. Mostra fórmula simbólica COM separadores *
        2. Substitui valores na fórmula COM separadores numéricos e *
        3. Apresenta resultado final
        
        Args:
            equation: Equação a avaliar
            variables: Dicionário com Variable objects ou valores
            precision: Casas decimais (padrão: 3)
            show_units: Incluir unidades (padrão: True)
            show_descriptions: (Mantido por compatibilidade)
            
        Returns:
            Lista de Steps formatados
            
        Raises:
            ImportError: Se SymPy não estiver disponível
            ValueError: Se variáveis necessárias não estiverem definidas
        """
        # ========== VALIDAÇÕES INICIAIS ==========
        if not SYMPY_AVAILABLE:
            raise ImportError(
                "SymPy não disponível para memorial declarativo. "
                "Instale com: pip install sympy"
            )
        
        if not equation.expr:
            raise ValueError("Equação não possui expressão válida.")
        
        # Validar que todas as variáveis necessárias estão presentes
        missing_vars = set(equation.variables_used) - set(variables.keys())
        if missing_vars:
            raise ValueError(
                f"Variáveis necessárias não definidas: {', '.join(sorted(missing_vars))}. "
                f"A equação requer: {', '.join(sorted(equation.variables_used))}. "
                f"Fornecido: {', '.join(sorted(variables.keys()))}."
            )
        
        steps = []
        
        # ========== PASSO 1: DADOS DE ENTRADA (REMOVIDO) ==========
        # steps.append(Step(type=StepType.EXPLANATION, ...))
        # for varname in sorted(equation.variables_used): ...

        # ========== PASSO 2: FÓRMULA SIMBÓLICA (MANTER) ==========
        # steps.append(Step(type=StepType.EXPLANATION, ...)) # <-- REMOVIDO
        
        try:
            if equation.name:
                lhs = sp.latex(Symbol(equation.name))
                rhs = sp.latex(equation.expr)
                
                # ✅ ADICIONAR OPERADORES * NA FÓRMULA SIMBÓLICA
                rhs = self._add_symbolic_operators(rhs)
                
                formula_latex = f"{lhs} = {rhs}"
            else:
                formula_latex = sp.latex(equation.expr)
                formula_latex = self._add_symbolic_operators(formula_latex)
        except Exception as e:
            self._logger.warning(f"Erro ao gerar LaTeX da fórmula: {e}. Usando string.")
            formula_latex = str(equation.expr)
        
        steps.append(Step(
            type=StepType.FORMULA,
            content=str(equation.expr),
            latex=formula_latex,
            explanation="Expressão simbólica"
        ))

        
        # ========== PASSO 3: SUBSTITUIÇÃO LITERAL (MANTER) ==========
        # steps.append(Step(type=StepType.EXPLANATION, ...)) # <-- REMOVIDO
        
        try:
            # Obter LaTeX da expressão original
            original_latex = sp.latex(equation.expr)
            
            # Criar dicionário de substituição para LaTeX
            latex_subs = {}
            for varname in equation.variables_used:
                if varname in variables:
                    var = variables[varname]
                    value = var.value if isinstance(var, Variable) else var
                    
                    if value is None:
                        raise ValueError(f"Variável '{varname}' possui valor None.")
                    
                    symbol_latex = sp.latex(Symbol(varname))
                    
                    # ✅ FORMATAR COM SEPARADORES NUMÉRICOS
                    value_latex = self._format_number_latex(value, precision)
                    
                    latex_subs[symbol_latex] = value_latex
            
            # Substituir no LaTeX
            substituted_latex = original_latex
            import re
            for symbol_latex in sorted(latex_subs.keys(), key=len, reverse=True):
                value_latex = latex_subs[symbol_latex]
                pattern = re.escape(symbol_latex) + r'(?![a-zA-Z_0-9])'
                substituted_latex = re.sub(pattern, value_latex, substituted_latex)
            
            # ✅ ADICIONAR SEPARADORES * ENTRE OPERADORES
            substituted_latex = self._add_operator_spacing(substituted_latex)
            
            if equation.name:
                lhs = sp.latex(Symbol(equation.name))
                substitution_latex = f"{lhs} = {substituted_latex}"
            else:
                substitution_latex = substituted_latex
            
        except Exception as e:
            self._logger.warning(f"Erro ao gerar substituição literal: {e}. Usando fallback.")
            subs_dict = {Symbol(vn): (variables[vn].value if isinstance(variables[vn], Variable) else variables[vn]) 
                         for vn in equation.variables_used if vn in variables}
            expr_substituted = equation.expr.subs(subs_dict, evaluate=False)
            substitution_latex = sp.latex(expr_substituted)
            if equation.name:
                substitution_latex = f"{sp.latex(Symbol(equation.name))} = {substitution_latex}"
        
        steps.append(Step(
            type=StepType.SUBSTITUTION,
            content="Valores substituídos",
            latex=substitution_latex,
            explanation="Valores substituídos na fórmula"
        ))
        
        # ========== PASSO 4: RESULTADO FINAL (MANTER) ==========
        # steps.append(Step(type=StepType.EXPLANATION, ...)) # <-- REMOVIDO
        
        # Calcular resultado numérico
        subs_dict_numeric = {}
        for varname in equation.variables_used:
            if varname in variables:
                var = variables[varname]
                value = var.value if isinstance(var, Variable) else var
                subs_dict_numeric[Symbol(varname)] = value
        
        try:
            expr_numeric = equation.expr.xreplace(subs_dict_numeric)
            expr_simplified = sp.simplify(expr_numeric)
        except Exception as e:
            self._logger.warning(f"Não foi possível simplificar: {e}")
            expr_simplified = expr_numeric
        
        # Avaliar numericamente
        try:
            result_value = float(expr_simplified.evalf())
        except (TypeError, ValueError, AttributeError) as e:
            free_symbols = expr_simplified.free_symbols
            if free_symbols:
                missing_vars = [str(s) for s in free_symbols]
                raise ValueError(
                    f"Variáveis não substituídas: {', '.join(missing_vars)}."
                ) from e
            else:
                try:
                    complex_result = complex(expr_simplified.evalf())
                    if complex_result.imag != 0:
                        raise ValueError(f"Resultado com parte imaginária: {complex_result}.") from e
                    result_value = complex_result.real
                except Exception:
                    raise ValueError(f"Não foi possível converter: {expr_simplified}.") from e
        
        # Validar resultado
        if not isinstance(result_value, (int, float)) or not (-1e308 < result_value < 1e308):
            raise ValueError(f"Resultado inválido: {result_value}.")
        
        # Formatar resultado
        try:
            result_str = f"{result_value:.{precision}f}"
        except (ValueError, OverflowError) as e:
            self._logger.warning(f"Erro ao formatar resultado: {e}")
            result_str = str(result_value)
        
        # LaTeX do resultado
        try:
            if equation.name:
                lhs = sp.latex(Symbol(equation.name))
                result_latex = f"{lhs} = {result_str}"
                
                # ✅ FORMATAR UNIDADE CORRETAMENTE COM CONVERSÃO ** → ^
                if show_units:
                    result_unit = self._infer_result_unit_pint(equation, variables)
                    if result_unit:
                        import re
                        # Remover coeficiente numérico do início (ex: "1.0 mm" → "mm")
                        result_unit = re.sub(r'^\d+\.?\d*\s+', '', result_unit)
                        
                        # ✅ CONVERTER ** PARA ^ COM CHAVES (ex: "mm**3" → "mm^{3}")
                        result_unit = self._format_unit_latex(result_unit)
                        
                        # Adicionar unidade ao LaTeX se não for adimensional
                        if result_unit.lower() not in ['dimensionless', '', 'none']:
                            result_latex += f"\\, \\text{{{result_unit}}}"
            else:
                result_latex = result_str
                
                # ✅ MESMO TRATAMENTO PARA EQUAÇÕES SEM NOME
                if show_units:
                    result_unit = self._infer_result_unit_pint(equation, variables)
                    if result_unit:
                        import re
                        result_unit = re.sub(r'^\d+\.?\d*\s+', '', result_unit)
                        result_unit = self._format_unit_latex(result_unit)
                        if result_unit.lower() not in ['dimensionless', '', 'none']:
                            result_latex += f"\\, \\text{{{result_unit}}}"
        
        except Exception as e:
            self._logger.warning(f"Erro ao formatar LaTeX do resultado: {e}")
            result_latex = result_str
        
        steps.append(Step(
            type=StepType.RESULT,
            content=f"= {result_str}",
            latex=result_latex,
            explanation="Valor final calculado"
        ))
        
        return steps

    # --- MÉTODOS HELPER (MANTIDOS) ---

    def _infer_result_unit_pint(
        self,
        equation: Equation,
        variables: Dict[str, Any]
    ) -> Optional[str]:
        """
        Infere unidade do resultado usando Pint via UnitRegistry.calculate_resultant_unit.
        """
        try:
            ureg = get_unit_registry()
            
            if not ureg or not ureg.ureg or not PINT_AVAILABLE:
                self._logger.debug("Pint não disponível para cálculo de unidade")
                return None
            
            unit_context = {}
            for varname in equation.variables_used:
                if varname in variables:
                    var = variables[varname]
                    if isinstance(var, Variable) and var.unit:
                        unit_context[varname] = var.unit
            
            result_unit = ureg.calculate_resultant_unit(equation.expr, unit_context)
            
            if result_unit is None or result_unit.strip() == "":
                return None
            
            import re
            result_unit = re.sub(r'^\d+\.?\d*\s+', '', result_unit)
            
            if result_unit.lower() in ['dimensionless', '']:
                return None
            
            self._logger.debug(f"Unidade calculada para {equation.name or 'expr'}: {result_unit}")
            return result_unit
            
        except Exception as e:
            self._logger.debug(f"Erro ao inferir unidade via Pint: {e}")
            return None


    def _format_unit_latex(self, unit_str: str) -> str:
        """
        Formata unidades para LaTeX, convertendo ** para ^.
        """
        import re
        
        if not unit_str:
            return unit_str
        
        # Converter ** para ^ com chaves: mm**3 → mm^{3}
        unit_str = re.sub(r'([a-zA-Z]+)\*\*(\d+)', r'\1^{\2}', unit_str)
        
        # Casos especiais: também converter potências negativas
        # mm**-1 → mm^{-1}
        unit_str = re.sub(r'([a-zA-Z]+)\*\*\((-?\d+)\)', r'\1^{\2}', unit_str)
        
        return unit_str


    def _format_number_latex(self, value: Union[int, float], precision: int = 3) -> str:
        """Formata número para LaTeX com separadores numéricos."""
        if isinstance(value, int) or (isinstance(value, float) and value.is_integer()):
            value_int = int(value)
            if abs(value_int) >= 1000:
                value_str = f"{value_int:,}".replace(',', '\\,')
            else:
                value_str = str(value_int)
            return value_str
        else:
            value_str = f"{value:.{precision}f}".rstrip('0').rstrip('.')
            return value_str
    

    def _add_symbolic_operators(self, latex_str: str) -> str:
        """
        Adiciona asteriscos (*) entre termos na fórmula SIMBÓLICA.
        """
        import re
        
        # Entre letra/} e letra (variáveis adjacentes)
        latex_str = re.sub(r'([a-zA-Z_}])\s+([a-zA-Z_])', r'\1 * \2', latex_str)
        
        # Entre } e letra (após subscript/superscript)
        latex_str = re.sub(r'}([a-zA-Z_])', r'} * \1', latex_str)
        
        # Entre letra e (
        latex_str = re.sub(r'([a-zA-Z_])\(', r'\1 * (', latex_str)
        
        # Entre ) e letra
        latex_str = re.sub(r'\)([a-zA-Z_])', r') * \1', latex_str)
        
        # Entre ) e (
        latex_str = re.sub(r'\)\(', r') * (', latex_str)
        
        return latex_str
    
    
    def _add_operator_spacing(self, latex_str: str) -> str:
        """
        Adiciona asteriscos (*) e formata números com separadores.
        VERSÃO ROBUSTA que processa múltiplas vezes até estabilizar.
        """
        import re
        
        # ===== PASSO 1: FORMATAR NÚMEROS GRANDES =====
        def format_number_in_latex(match):
            number_str = match.group(0)
            try:
                # Tentar como int primeiro
                if '.' not in number_str:
                    value_int = int(number_str)
                    if abs(value_int) >= 1000:
                        formatted = f"{value_int:,}".replace(',', '\\,')
                        return formatted
                    return number_str
                
                # Tentar como float
                value = float(number_str)
                if value.is_integer() and abs(value) >= 1000:
                    value_int = int(value)
                    formatted = f"{value_int:,}".replace(',', '\\,')
                    return formatted
                else:
                    return number_str # Manter floats como estão
            except (ValueError, OverflowError):
                return number_str
        
        # Aplicar formatação apenas a números inteiros com 4+ dígitos
        latex_str = re.sub(r'\b\d{4,}\b(?!\.)', format_number_in_latex, latex_str)
        
        # ===== PASSO 2: ADICIONAR ASTERISCOS (MÚLTIPLAS PASSAGENS) =====
        # Processar até 5 vezes para garantir que todos os casos sejam capturados
        
        for _ in range(5):
            original = latex_str
            
            # Regra 1: Entre } e dígito (COM OU SEM ESPAÇO)
            # Exemplos: }21 → } * 21  ou  } 21 → } * 21
            latex_str = re.sub(r'(?<!\\left)(?<!\\right)}\s*(\d)', r'} * \1', latex_str)
            
            # Regra 2: Entre } e letra
            latex_str = re.sub(r'(?<!\\left)(?<!\\right)}\s*([a-zA-Z_])', r'} * \1', latex_str)
            
            # Regra 3: Entre dígito e letra
            latex_str = re.sub(r'(\d)\s*([a-zA-Z_])', r'\1 * \2', latex_str)
            
            # Regra 4: Entre letra e dígito
            latex_str = re.sub(r'([a-zA-Z_])\s*(\d)', r'\1 * \2', latex_str)
            
            # Regra 5: Entre ) e dígito/letra
            latex_str = re.sub(r'(?<!\\right)\)\s*(\d)', r') * \1', latex_str)
            latex_str = re.sub(r'(?<!\\right)\)\s*([a-zA-Z_])', r') * \1', latex_str)
            
            # Regra 6: Entre dígito/letra e (
            latex_str = re.sub(r'(\d)\s*\(', r'\1 * (', latex_str)
            latex_str = re.sub(r'([a-zA-Z_])\s*\(', r'\1 * (', latex_str)
            
            # Regra 7: Entre ) e (
            latex_str = re.sub(r'\)\s*\(', r') * (', latex_str)
            
            # Regra 8: Entre números separados por espaço (mas não \,)
            latex_str = re.sub(r'(\d)\s+(?!\\,|\*)(\d)', r'\1 * \2', latex_str)
            
            # Regra 9: Entre letras separadas por espaço (mas não já com *)
            latex_str = re.sub(r'([a-zA-Z_}])\s+(?!\*)([a-zA-Z_])', r'\1 * \2', latex_str)
            
            # Se não mudou nada nesta passagem, parar
            if latex_str == original:
                break
        
        # ===== PASSO 3: LIMPAR MÚLTIPLOS * CONSECUTIVOS =====
        # Remover casos de * * → *
        latex_str = re.sub(r'\*\s*\*', r'*', latex_str)
        
        # ===== PASSO 4: NORMALIZAR ESPAÇOS =====
        # Garantir que sempre há espaço ao redor de *
        latex_str = re.sub(r'\*', r' * ', latex_str)
        # Mas limpar múltiplos espaços
        latex_str = re.sub(r'\s+', r' ', latex_str)
        # Remover espaços antes de pontuação
        latex_str = re.sub(r'\s+([,\.\)])', r'\1', latex_str)
        # Remover espaços após (
        latex_str = re.sub(r'\(\s+', r'(', latex_str)
        
        return latex_str

    # ----------------------------------------------------------------------------
    # REVISÃO: Métodos de granularidade removidos.
    # ----------------------------------------------------------------------------
    # def generate_smart(...) -> ... (REMOVIDO)
    # def _generate_minimal(...) -> ... (REMOVIDO)
    # def _generate_basic(...) -> ... (REMOVIDO)
    # def _generate_medium(...) -> ... (REMOVIDO)
    # def _generate_detailed(...) -> ... (REMOVIDO)
    # def _analyze_complexity(...) -> ... (REMOVIDO)
    # def _get_expr_depth(...) -> ... (REMOVIDO)


# ============================================================================
# STEPREGISTRY - REGISTRO DE STEPS
# ============================================================================

class StepRegistry:
    """
    Registro global de steps estilo Calcpad.
    
    Examples:
        >>> registry = StepRegistry()
        >>> eq = Equation("M = q * L**2 / 8", locals_dict=vars_dict)
        >>> registry.register(eq)
        >>> latex = registry.to_latex()
    """
    
    def __init__(self):
        """Inicializa registry."""
        self._steps: List[Step] = []
        self._generator = StepGenerator()
        self._lock = threading.Lock()
    
    # ----------------------------------------------------------------------------
    # REVISÃO: 'granularity' removido da assinatura.
    # ----------------------------------------------------------------------------
    def register(
        self,
        equation: Equation,
        variables: Optional[Dict[str, Any]] = None,
        # granularity: GranularityType = GranularityType.MEDIUM, # <-- REMOVIDO
        precision: int = 3,
        **kwargs
    ) -> List[Step]:
        """
        Registra equation e gera steps no formato padrão.
        
        Args:
            equation: Equation para registrar
            variables: Dicionário de variáveis (opcional).
            precision: Casas decimais
            **kwargs: Variáveis adicionais e 'show_units', 'show_descriptions'.
        
        Returns:
            Lista de Steps gerados
        """
        # Unificar contexto
        eval_context = {}
        if equation.locals_dict:
            eval_context.update(equation.locals_dict)
        if variables:
            eval_context.update(variables)
        if kwargs:
            eval_context.update(kwargs)
            
        with self._lock:
            # Preparar kwargs para o gerador
            generator_kwargs = {
                'precision': precision,
                'show_units': kwargs.get('show_units', True),
                'show_descriptions': kwargs.get('show_descriptions', True)
            }
            
            # Passar o eval_context e os kwargs
            steps = self._generator.generate(
                equation,
                eval_context,  # <-- Contexto unificado
                **generator_kwargs
            )
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
            mode: "align" (padrão) ou "gather"

        Returns:
            LaTeX completo
        """
        # CORREÇÃO: Formatar usando step.latex, não step.to_latex()
        with self._lock:
            if not self._steps:
                return ""

            # Usar align* para não numerar as equações
            env = "align*" if mode == "align" else "gather*"
            latex_lines = []
            
            # REVISÃO: Agrupar por cálculo (3 steps por cálculo)
            current_calculation = []
            for step in self._steps:
                
                # Adiciona '&' antes do '=' para alinhamento se houver
                if mode == "align" and '=' in step.latex:
                     parts = step.latex.split('=', 1)
                     current_calculation.append(f"{parts[0].strip()} &= {parts[1].strip()}")
                else:
                     current_calculation.append(step.latex)
                
                # Se for o resultado, fechamos o grupo
                if step.type == StepType.RESULT:
                    latex_lines.append(r" \\ ".join(current_calculation))
                    current_calculation = []

            # Junta os cálculos com \\\\ (nova linha com mais espaço)
            joined_lines = r" \\\\ ".join(latex_lines)

            return f"\\begin{{{env}}}\n{joined_lines}\n\\end{{{env}}}"

    # ----------------------------------------------------------------------------
    # REVISÃO: Simplificado para o novo formato de 3 steps,
    # adicionando um separador '---' após cada cálculo (Resultado).
    # ----------------------------------------------------------------------------
    def to_markdown(self) -> str:
        """
        Exporta steps para Markdown com formatação LaTeX.
        Formato: Formula -> Substituição -> Resultado
        
        Returns:
            String Markdown formatada
        """
        output = []
        
        for i, step in enumerate(self._steps):
            
            if step.type == StepType.FORMULA:
                # Iniciar um novo cálculo
                if i > 0:
                    output.append('\n\n---\n\n') # Separador
                output.append(f"$${step.latex}$$")
            
            elif step.type == StepType.SUBSTITUTION:
                output.append(f"\n\n$${step.latex}$$")
            
            elif step.type == StepType.RESULT:
                output.append(f"\n\n$${step.latex}$$")
        
        return ''.join(output)
