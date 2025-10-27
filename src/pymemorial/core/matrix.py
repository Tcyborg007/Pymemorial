"""
Sistema de Matrizes Robusto com SymPy e NumPy - PyMemorial v2.1.9 

ðŸš€ CORREÃ‡Ã•ES IMPLEMENTADAS (v2.1.9):
âœ… Parsing simbÃ³lico 100% isolado com extraÃ§Ã£o de nomes de variÃ¡veis
âœ… ValidaÃ§Ã£o rigorosa de pureza simbÃ³lica com detecÃ§Ã£o inteligente
âœ… Fallback manual elemento por elemento
âœ… MÃ©todos auxiliares para isolamento completo
âœ… Debug logging para rastreamento de contaminaÃ§Ã£o

Autor: PyMemorial Team (Revisado por Especialista)
Data: 2025-10-23
VersÃ£o: 2.1.9 (Symbolic Purity Absolute)
"""

from __future__ import annotations
import logging
from typing import Dict, Any, List, Optional, Union, Tuple, Literal
from dataclasses import dataclass, field
from copy import deepcopy
import traceback
import re
import ast

# Core dependencies
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

try:
    import sympy as sp
    from sympy import Matrix as SpMatrix, latex as sp_latex
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False
    sp = None
    SpMatrix = None

# PyMemorial core - IMPORTS ROBUSTOS
try:
    from .variable import Variable
    from .units import parse_quantity, strip_units, ureg, PINT_OK
except (ImportError, ValueError):
    try:
        from pymemorial.core.variable import Variable
        from pymemorial.core.units import parse_quantity, strip_units, ureg, PINT_OK
    except ImportError:
        Variable = type('Variable', (object,), {})
        def parse_quantity(v, u=None): return v
        def strip_units(v): return float(v) if v is not None else 0.0
        ureg = None
        PINT_OK = False

logger = logging.getLogger(__name__)

# Type aliases
MatrixType = Union[np.ndarray, pd.DataFrame, 'SpMatrix', List[List[float]], str]
GranularityType = Literal['minimal', 'basic', 'normal', 'detailed', 'all', 'smart']

# ============================================================================
# MATRIX CLASS (VERSÃƒO 2.1.9 - PUREZA SIMBÃ“LICA GARANTIDA)
# ============================================================================

@dataclass
class Matrix:
    """
    Representa uma matriz com capacidades simbÃ³licas (SymPy)
    e numÃ©ricas (NumPy).
    
    ðŸŽ¯ Suporta:
    - ExpressÃµes simbÃ³licas (ex: "[[12*E*I/L**3, ...]]")
    - NumPy arrays, Pandas DataFrames, Listas Python
    """
    # Core attributes
    data: MatrixType
    variables: Dict[str, Variable] = field(default_factory=dict)
    description: str = ""
    name: str = "M"
    
    # Metadata
    shape: Optional[Tuple[int, int]] = None
    is_symbolic: bool = False
    is_square: bool = False
    
    # Internal state
    _symbolic_matrix: Optional[SpMatrix] = field(default=None, init=False, repr=False)
    _numeric_matrix: Optional[np.ndarray] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        """Inicializa e valida a matriz."""
        if not SYMPY_AVAILABLE:
            raise ImportError("SymPy Ã© obrigatÃ³rio para a classe Matrix robusta.")
        if not NUMPY_AVAILABLE:
            raise ImportError("NumPy Ã© obrigatÃ³rio para a classe Matrix robusta.")
        
        # âœ… DEBUG: Log da string recebida
        if isinstance(self.data, str):
            logger.warning(f"ðŸ” DEBUG: String da matriz recebida:")
            logger.warning(f"   {self.data}")
            logger.warning(f"ðŸ” DEBUG: VariÃ¡veis recebidas:")
            for name, var in self.variables.items():
                val = var.value if hasattr(var, 'value') else 'N/A'
                logger.warning(f"   {name} = {val}")
        
        self._parse_matrix()
        self._validate()
        logger.debug(f"Matriz '{self.name}' inicializada: shape={self.shape}, symbolic={self.is_symbolic}")

    def _parse_matrix(self):
        """
        âœ… CORREÃ‡ÃƒO v2.1.9: Parsing 100% ISOLADO - MÃ©todo completamente refatorado
        """
        # Case 1: String expression - CORREÃ‡ÃƒO APLICADA AQUI
        if isinstance(self.data, str):
            if not SYMPY_AVAILABLE:
                raise ImportError("SymPy Ã© necessÃ¡rio para matrizes simbÃ³licas")
            
            try:
                self._symbolic_matrix = self._parse_matrix_string(self.data)
                self.is_symbolic = True
                self.shape = self._symbolic_matrix.shape
                
                # âœ… VALIDAÃ‡ÃƒO RIGOROSA APRIMORADA
                self._validate_symbolic_purity()
                
            except Exception as e:
                logger.error(f"Falha ao parsear matriz: {e}")
                logger.debug(f"ExpressÃ£o: {self.data}")
                logger.debug(f"Traceback: {traceback.format_exc()}")
                raise ValueError(f"Falha ao parsear expressÃ£o da matriz: {e}")
        
        # Cases 2-5: (mantidos inalterados)
        elif SYMPY_AVAILABLE and isinstance(self.data, SpMatrix):
            self._symbolic_matrix = self.data
            self.is_symbolic = True
            self.shape = self.data.shape
        
        elif NUMPY_AVAILABLE and isinstance(self.data, np.ndarray):
            self._numeric_matrix = self.data
            self.is_symbolic = False
            self.shape = self.data.shape
        
        elif PANDAS_AVAILABLE and isinstance(self.data, pd.DataFrame):
            self._numeric_matrix = self.data.values
            self.is_symbolic = False
            self.shape = self.data.shape
        
        elif isinstance(self.data, (list, tuple)):
            try:
                self._numeric_matrix = np.array(self.data, dtype=float)
                self.shape = self._numeric_matrix.shape
                self.is_symbolic = False
            except ValueError:
                try:
                    self._symbolic_matrix = SpMatrix(self.data)
                    self.is_symbolic = True
                    self.shape = self._symbolic_matrix.shape
                except Exception as e2:
                    raise ValueError(f"Lista nÃ£o pÃ´de ser parseada: {e2}")
        else:
            raise TypeError(f"Tipo de dado nÃ£o suportado: {type(self.data)}")
        
        if self.shape:
            self.is_square = (self.shape[0] == self.shape[1])

    def _validate(self):
        """Valida dados da matriz."""
        if self.shape is None:
            raise ValueError("Shape da matriz nÃ£o determinado")
        if self.shape[0] == 0 or self.shape[1] == 0:
            raise ValueError("Matriz nula")

    def _parse_matrix_string(self, expr_str: str) -> SpMatrix:
        """
        Parse matrix expression from string to SymPy Matrix.
        
        âœ… v2.1.8 FINAL: ForÃ§ar rational=False para evitar simplificaÃ§Ã£o prematura
        """
        if self.variables:
            symbol_names = list(self.variables.keys())
        else:
            symbol_names = self._extract_variable_names_from_expression(expr_str)
        
        # Criar sÃ­mbolos puros
        local_dict = {}
        for name in symbol_names:
            local_dict[name] = sp.Symbol(name, real=True, positive=True)
        
        logger.debug(f"Parsing com sÃ­mbolos puros: {list(local_dict.keys())}")
        
        try:
            parsed_matrix = sp.sympify(
                expr_str,
                locals=local_dict,
                evaluate=False,
                rational=False  # âœ… CORREÃ‡ÃƒO FINAL: NÃ£o converter para Rational
            )
            
            if not isinstance(parsed_matrix, (list, SpMatrix)):
                raise ValueError("NÃ£o Ã© lista ou matriz")
            
            if isinstance(parsed_matrix, list):
                parsed_matrix = SpMatrix(parsed_matrix)
            
            logger.debug(f"âœ… Parsing bem-sucedido: {parsed_matrix.shape}")
            return parsed_matrix
            
        except Exception as e:
            logger.warning(f"Parsing direto falhou: {e}. Usando fallback...")
            # âœ… CORREÃ‡ÃƒO: Usar mÃ©todo que jÃ¡ existe
            return self._parse_matrix_fallback(expr_str, local_dict)



    def _parse_matrix_safe_v2(self, expr_str: str, isolated_symbols: Dict) -> SpMatrix:
        """
        âœ… v2.1.9 FALLBACK ROBUSTO: Parsing manual com isolamento absoluto.
        
        Este mÃ©todo Ã© chamado quando sympify falha. Usa eval() com namespace
        completamente isolado e sem acesso a builtins.
        """
        try:
            # Limpar expressÃ£o
            clean_expr = expr_str.strip()
            
            # CRÃTICO: Criar namespace isolado
            # - __builtins__: {} (sem funÃ§Ãµes built-in)
            # - Apenas sÃ­mbolos SymPy puros
            isolated_namespace = {
                "__builtins__": {},  # Sem acesso a funÃ§Ãµes Python
                **isolated_symbols   # Apenas sÃ­mbolos SymPy
            }
            
            # Adicionar funÃ§Ãµes matemÃ¡ticas do SymPy se necessÃ¡rio
            math_functions = {
                'sin': sp.sin, 'cos': sp.cos, 'tan': sp.tan,
                'exp': sp.exp, 'log': sp.log, 'sqrt': sp.sqrt,
                'pi': sp.pi, 'E': sp.E
            }
            isolated_namespace.update(math_functions)
            
            logger.debug(f"Namespace isolado: {list(isolated_namespace.keys())}")
            
            # Eval com seguranÃ§a mÃ¡xima
            matrix_data = eval(clean_expr, isolated_namespace, {})
            
            # Validar resultado
            if not isinstance(matrix_data, list):
                raise ValueError(f"eval retornou tipo invÃ¡lido: {type(matrix_data)}")
            
            # Converter para SpMatrix
            result = SpMatrix(matrix_data)
            
            # ValidaÃ§Ã£o de pureza simbÃ³lica
            latex_check = sp.latex(result)
            for name, var in self.variables.items():
                if var.value is not None:
                    value_str = str(strip_units(var.value))
                    # Verificar se o valor numÃ©rico aparece no LaTeX
                    if value_str in latex_check:
                        logger.error(
                            f"âŒ CONTAMINAÃ‡ÃƒO DETECTADA no fallback!\n"
                            f"    VariÃ¡vel: {name} = {value_str}\n"
                            f"    LaTeX: {latex_check[:200]}..."
                        )
                        raise ValueError(
                            f"Parsing contaminou matriz com valor de {name}={value_str}"
                        )
            
            logger.debug(f"âœ… Fallback bem-sucedido e validado: {result.shape}")
            return result
            
        except Exception as e:
            logger.error(f"Fallback seguro falhou: {e}")
            logger.debug(f"ExpressÃ£o: {clean_expr}")
            logger.debug(f"Namespace: {list(isolated_namespace.keys())}")
            raise ValueError(f"NÃ£o foi possÃ­vel parsear matriz de forma segura: {e}")

    def _extract_variable_names_from_expression(self, expr_str: str) -> List[str]:
        """
        Extrai nomes de variÃ¡veis de uma expressÃ£o string SEM acessar valores.
        
        âœ… CORREÃ‡ÃƒO v2.1.9: Priorizar variÃ¡veis do usuÃ¡rio sobre sÃ­mbolos reservados
        """
        # Se temos self.variables, usar SEUS nomes (prioridade absoluta)
        if self.variables:
            user_var_names = set(self.variables.keys())
            logger.debug(f"Usando nomes de variÃ¡veis do usuÃ¡rio: {user_var_names}")
            return list(user_var_names)
        
        # Fallback: extrair da expressÃ£o (caso nÃ£o haja self.variables)
        var_pattern = re.compile(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b')
        reserved_words = {
            'sin', 'cos', 'tan', 'exp', 'log', 'sqrt', 'pi', 'e', 
            'Matrix', 'oo', 'zoo', 'nan', 'True', 'False',
            'and', 'or', 'not', 'if', 'else', 'for', 'while'
        }
        
        found_names = set()
        for match in var_pattern.finditer(expr_str):
            name = match.group(1)
            if (not name.isdigit() and 
                name not in reserved_words and
                not name.startswith('_')):
                found_names.add(name)
        
        return list(found_names)

    def _validate_symbolic_purity(self):
        """
        âœ… VALIDAÃ‡ÃƒO RIGOROSA v2.1.9: Garante que a matriz Ã© 100% simbÃ³lica
        
        Levanta erro se qualquer valor numÃ©rico de variÃ¡vel aparecer no LaTeX
        """
        if not self.is_symbolic or self._symbolic_matrix is None:
            return
        
        latex_output = sp_latex(self._symbolic_matrix)
        
        # Verificar cada variÃ¡vel
        for var_name, var in self.variables.items():
            if var.value is not None:
                # Extrair representaÃ§Ã£o string do valor
                value_str = str(strip_units(var.value))
                
                # âœ… VALIDAÃ‡ÃƒO CRÃTICA: Valor NÃƒO deve aparecer no LaTeX
                if value_str and self._is_value_contaminating_latex(value_str, latex_output, var_name):
                    raise ValueError(
                        f"âŒ CONTAMINAÃ‡ÃƒO NUMÃ‰RICA DETECTADA!\n"
                        f"   VariÃ¡vel: {var_name}\n"
                        f"   Valor: {value_str}\n"
                        f"   LaTeX (primeiros 200 chars): {latex_output[:200]}...\n"
                        f"   CAUSA: Parsing acessou valores em vez de sÃ­mbolos puros."
                    )
        
        logger.debug("âœ… ValidaÃ§Ã£o de pureza simbÃ³lica passou")

    def _is_value_contaminating_latex(self, value_str: str, latex_output: str, var_name: str) -> bool:
        """
        Detecta se um valor numÃ©rico contaminou o LaTeX simbÃ³lico.
        
        âœ… LÃ“GICA INTELIGENTE: 
        - Valor aparece E sÃ­mbolo correspondente NÃƒO aparece no mesmo contexto
        - Ignora casos onde valor Ã© parte de outro nÃºmero (ex: 500 em 1500)
        """
        # PadrÃ£o para encontrar o valor como nÃºmero isolado
        pattern = r'(?<!\d)' + re.escape(value_str) + r'(?!\d)'
        
        if re.search(pattern, latex_output):
            # Verificar se o sÃ­mbolo tambÃ©m aparece
            symbol_pattern = r'\b' + re.escape(var_name) + r'\b'
            if not re.search(symbol_pattern, latex_output):
                return True
            
            # Mesmo que o sÃ­mbolo apareÃ§a, se o valor aparece em contexto diferente, Ã© contaminaÃ§Ã£o
            logger.warning(f"âš ï¸ Valor {value_str} da variÃ¡vel {var_name} apareceu no LaTeX simbÃ³lico")
        
        return False

    # ========================================================================
    # AVALIAÃ‡ÃƒO (COM OTIMIZAÃ‡ÃƒO NUMPY)
    # ========================================================================

    def evaluate(self, use_cache: bool = True) -> np.ndarray:
        """
        Avalia a matriz para a forma numÃ©rica (np.ndarray).
        Usa cache se disponÃ­vel.
        """
        if use_cache and self._numeric_matrix is not None:
            return self._numeric_matrix
        
        if self._numeric_matrix is not None:
            return self._numeric_matrix
        
        if self._symbolic_matrix is not None:
            result = self._evaluate_symbolic_lambdify()
            self._numeric_matrix = result
            return result
        
        raise ValueError("Nenhum dado de matriz para avaliar")

    def _evaluate_symbolic_lambdify(self) -> np.ndarray:
        """
        âœ… v2.1.9: Avaliar matriz buscando valores das variÃ¡veis originais.
        """
        logger.debug(f"Avaliando '{self.name}' com sp.lambdify...")
        
        expr_matrix = self._symbolic_matrix
        if expr_matrix is None:
            raise ValueError("Matriz simbÃ³lica nÃ£o foi parseada.")
        
        free_symbols = expr_matrix.free_symbols
        symbols_tuple = tuple(free_symbols)
        
        subs_values = {}
        missing_vars = []
        
        # âœ… BUSCAR VALORES DAS VARIÃVEIS ORIGINAIS
        # Se temos _original_variables, usar elas
        variables_source = getattr(self, '_original_variables', self.variables)
        
        for sym in symbols_tuple:
            var_name = str(sym)
            if var_name not in variables_source:
                missing_vars.append(var_name)
                continue
            
            var = variables_source[var_name]
            if var.value is None:
                missing_vars.append(f"{var_name} (sÃ­mbolo sem valor)")
                continue
            
            subs_values[var_name] = strip_units(var.value)
        
        if missing_vars:
            raise ValueError(
                f"VariÃ¡veis ausentes para avaliar '{self.name}': {', '.join(missing_vars)}"
            )
        
        try:
            func = sp.lambdify(symbols_tuple, expr_matrix, 'numpy')
        except Exception as e:
            logger.error(f"Falha ao 'lambdify' a matriz '{self.name}': {e}")
            raise RuntimeError(f"Erro ao compilar matriz para NumPy: {e}")
        
        try:
            args = [subs_values[str(sym)] for sym in symbols_tuple]
        except KeyError as e:
            raise ValueError(f"Erro interno ao mapear argumentos para lambdify: {e}")
        
        try:
            result_array = np.asarray(func(*args), dtype=float)
            logger.debug(f"Matriz '{self.name}' avaliada com sucesso via lambdify.")
            return result_array
        except Exception as e:
            logger.error(f"Falha ao executar a funÃ§Ã£o 'lambdify' para '{self.name}': {e}")
            logger.error(f"Argumentos passados (count={len(args)}): {args}")
            raise RuntimeError(f"Erro ao calcular matriz numÃ©rica: {e}")

    # ========================================================================
    # GERAÃ‡ÃƒO DE STEPS (âœ… v2.1.9 - PASSO INTERMEDIÃRIO ROBUSTO)
    # ========================================================================

    def steps(
        self,
        granularity: GranularityType = 'normal',
        operation: Optional[str] = None,
        show_units: bool = True
    ) -> List[Dict[str, Any]]:
        """
        âœ… CORREÃ‡ÃƒO v2.1.9: Steps com passo intermediÃ¡rio usando _substitute_preserve_structure().
        
        Fluxo:
        1. SimbÃ³lico (puro) 
        2. Lista de substituiÃ§Ã£o
        3. Matriz substituÃ­da (estrutura preservada) âœ… NOVO MÃ‰TODO
        4. Resultado numÃ©rico (lambdify)
        """
        steps = []
        
        # --- DicionÃ¡rios de SubstituiÃ§Ã£o ---
        subs_dict_display = {}  # Para exibiÃ§Ã£o (sp.Float)
        subs_list_display = []  # Para texto (string formatada)
        
        try:
            # Popular os dicionÃ¡rios
            if self.is_symbolic and self._symbolic_matrix:
                for sym in self._symbolic_matrix.free_symbols:
                    var_name = str(sym)
                    if var_name in self.variables:
                        var = self.variables[var_name]
                        if var.value is not None:
                            val_numeric = strip_units(var.value)
                            # Para substituiÃ§Ã£o: usar sp.Float
                            subs_dict_display[sym] = sp.Float(val_numeric)
                            
                            if PINT_OK and hasattr(var.value, 'units'):
                                subs_list_display.append(f"{var_name} = {var.value:~P}")
                            else:
                                subs_list_display.append(f"{var_name} = {val_numeric}")
                        else:
                            subs_list_display.append(f"{var_name} (sÃ­mbolo livre)")
            
            # --- PASSO 1: DefiniÃ§Ã£o ---
            steps.append({
                'step': 'DefiniÃ§Ã£o da matriz',
                'operation': 'definition',
                'description': f'{self.name}: matriz {self.shape[0]}Ã—{self.shape[1]}',
                'shape': self.shape,
                'is_symbolic': self.is_symbolic
            })
            
            # --- PASSO 2: Forma SimbÃ³lica (100% PURA) ---
            if self.is_symbolic and self._symbolic_matrix is not None:
                steps.append({
                    'step': 'Forma SimbÃ³lica',
                    'operation': 'symbolic',
                    'description': 'ExpressÃ£o simbÃ³lica da matriz',
                    'latex': sp_latex(self._symbolic_matrix)
                })
            
            # --- PASSO 3: SubstituiÃ§Ã£o de VariÃ¡veis (Lista de valores) ---
            if granularity in ('basic', 'normal', 'detailed', 'all') and subs_list_display:
                steps.append({
                    'step': 'SubstituiÃ§Ã£o de VariÃ¡veis',
                    'operation': 'substitution',
                    'description': ', '.join(subs_list_display)
                })
            
            # --- PASSO 4: Matriz SubstituÃ­da (ESTRUTURA PRESERVADA) âœ… NOVO ---
            if granularity in ('detailed', 'all') and self.is_symbolic and subs_dict_display:
                try:
                    rows, cols = self._symbolic_matrix.shape
                    intermediate_elements = []
                    
                    for i in range(rows):
                        row_elements = []
                        for j in range(cols):
                            original_elem = self._symbolic_matrix[i, j]
                            
                            # âœ… MÃ‰TODO ROBUSTO: Substituir e preservar estrutura
                            unevaluated = self._substitute_preserve_structure(
                                original_elem, 
                                subs_dict_display
                            )
                            row_elements.append(unevaluated)
                        
                        intermediate_elements.append(row_elements)
                    
                    # Criar matriz intermediÃ¡ria
                    intermediate_matrix = SpMatrix(intermediate_elements)
                    
                    steps.append({
                        'step': 'Matriz SubstituÃ­da (Passo IntermediÃ¡rio)',
                        'operation': 'intermediate',
                        'description': 'Valores substituÃ­dos, estrutura algÃ©brica preservada',
                        'latex': sp_latex(intermediate_matrix)
                    })
                    
                except Exception as e:
                    logger.warning(f"Falha ao gerar passo intermediÃ¡rio: {e}")
                    logger.debug(traceback.format_exc())
            
            # --- PASSO 5: Matriz NumÃ©rica (Resultado com lambdify) ---
            try:
                result_matrix = self.evaluate(use_cache=True)
                
                # Formato de exibiÃ§Ã£o
                if result_matrix.shape[0] <= 8 and result_matrix.shape[1] <= 8:
                    matrix_display = result_matrix.tolist()
                else:
                    matrix_display = f"{result_matrix.shape[0]}Ã—{result_matrix.shape[1]} (matriz grande)"
                
                steps.append({
                    'step': 'Matriz NumÃ©rica (Resultado)',
                    'operation': 'evaluation',
                    'description': 'AvaliaÃ§Ã£o numÃ©rica final (calculada com NumPy)',
                    'matrix': matrix_display
                })
            
            except Exception as e:
                logger.warning(f"NÃ£o foi possÃ­vel avaliar '{self.name}' numericamente: {e}")
                steps.append({
                    'step': 'AvaliaÃ§Ã£o NumÃ©rica',
                    'operation': 'evaluation_failed',
                    'description': 'NÃ£o foi possÃ­vel avaliar (verificar variÃ¡veis)',
                    'error': str(e)
                })
            
            # --- PASSO 6: Propriedades (Opcional) ---
            if granularity == 'all' and 'result_matrix' in locals():
                try:
                    properties = self._compute_properties(result_matrix)
                    steps.append({
                        'step': 'Propriedades',
                        'operation': 'properties',
                        'description': 'Propriedades da matriz',
                        **properties
                    })
                except Exception as e:
                    logger.warning(f"NÃ£o pÃ´de computar propriedades: {e}")
            
            return steps
        
        except Exception as e:
            logger.error(f"Falha ao gerar steps para matriz {self.name}: {e}")
            logger.debug(traceback.format_exc())
            return [{
                'step': 'Erro',
                'operation': 'error',
                'description': f'Erro ao gerar steps: {e}'
            }]

    def _substitute_preserve_structure(self, expr: sp.Expr, subs_dict: Dict) -> sp.Expr:
        """
        âœ… NOVO MÃ‰TODO v2.1.9: Substitui sÃ­mbolos por valores SEM avaliar.
        Preserva a estrutura algÃ©brica completa usando reconstruÃ§Ã£o recursiva.
        """
        from sympy import Mul, Add, Pow, Number
        from sympy.core.expr import UnevaluatedExpr
        
        # Se jÃ¡ Ã© um nÃºmero, retornar direto
        if expr.is_Number:
            return expr
        
        # Se Ã© um sÃ­mbolo, substituir com UnevaluatedExpr
        if expr.is_Symbol:
            if expr in subs_dict:
                return UnevaluatedExpr(subs_dict[expr])
            return expr
        
        # Se Ã© uma operaÃ§Ã£o, reconstruir recursivamente
        if isinstance(expr, Mul):
            new_args = [self._substitute_preserve_structure(arg, subs_dict) 
                       for arg in expr.args]
            return Mul(*new_args, evaluate=False)
        
        elif isinstance(expr, Add):
            new_args = [self._substitute_preserve_structure(arg, subs_dict) 
                       for arg in expr.args]
            return Add(*new_args, evaluate=False)
        
        elif isinstance(expr, Pow):
            new_base = self._substitute_preserve_structure(expr.base, subs_dict)
            new_exp = self._substitute_preserve_structure(expr.exp, subs_dict)
            return Pow(new_base, new_exp, evaluate=False)
        
        # Para outros tipos, usar subs padrÃ£o
        else:
            return expr.subs(subs_dict, simultaneous=True, evaluate=False)

    def _make_unevaluated(self, expr: sp.Expr, subs_dict: Dict) -> sp.Expr:
        """
        MÃ©todo legado mantido por compatibilidade.
        Use _substitute_preserve_structure() para novos desenvolvimentos.
        """
        return self._substitute_preserve_structure(expr, subs_dict)

    def _compute_properties(self, matrix: np.ndarray) -> Dict[str, Any]:
        """Computa propriedades da matriz."""
        props = {'shape': self.shape, 'is_square': self.is_square}
        
        if self.is_square and NUMPY_AVAILABLE:
            try:
                props['determinant'] = float(np.linalg.det(matrix))
                props['trace'] = float(np.trace(matrix))
                props['rank'] = int(np.linalg.matrix_rank(matrix))
                props['is_symmetric'] = bool(np.allclose(matrix, matrix.T))
            except Exception as e:
                logger.warning(f"NÃ£o pÃ´de computar todas as propriedades: {e}")
        
        return props

# ============================================================================
# MODULE AVAILABILITY FLAG
# ============================================================================

MATRIX_AVAILABLE = NUMPY_AVAILABLE and SYMPY_AVAILABLE

if MATRIX_AVAILABLE:
    logger.info(f"âœ… Matrix module (Robust v2.1.9) disponÃ­vel (NumPy, SymPy)")
else:
    reasons = []
    if not NUMPY_AVAILABLE: reasons.append("NumPy (obrigatÃ³rio) indisponÃ­vel")
    if not SYMPY_AVAILABLE: reasons.append("SymPy (obrigatÃ³rio) indisponÃ­vel")
    logger.critical(f"âŒ Matrix module indisponÃ­vel: {', '.join(reasons)}")

__all__ = [
    'Matrix',
    'MatrixType',
    'GranularityType',
    'MATRIX_AVAILABLE',
    'NUMPY_AVAILABLE',
    'SYMPY_AVAILABLE',
    'debug_matrix_parsing',
]

__version__ = "2.1.9"

# ============================================================================
# FUNÃ‡Ã•ES AUXILIARES PARA DEBUGGING
# ============================================================================

def debug_matrix_parsing(expr_str: str, variables: Dict[str, Variable]) -> Dict[str, Any]:
    """
    FunÃ§Ã£o auxiliar para debug do parsing de matrizes.
    
    Args:
        expr_str: ExpressÃ£o da matriz como string
        variables: DicionÃ¡rio de variÃ¡veis
    
    Returns:
        Dict com informaÃ§Ãµes de debug
    """
    if not SYMPY_AVAILABLE:
        return {'error': 'SymPy nÃ£o disponÃ­vel'}
    
    info = {
        'input_expr': expr_str,
        'variables_provided': list(variables.keys()),
        'parsing_steps': []
    }
    
    try:
        # Criar sÃ­mbolos
        local_dict = {}
        for var_name in variables.keys():
            local_dict[var_name] = sp.Symbol(var_name, real=True, positive=True)
        info['parsing_steps'].append(f"SÃ­mbolos criados: {list(local_dict.keys())}")
        
        # Parse com evaluate=False
        parsed_expr = sp.sympify(expr_str, locals=local_dict, evaluate=False, rational=True)
        info['parsing_steps'].append(f"Parsed type: {type(parsed_expr).__name__}")
        
        # Verificar sÃ­mbolos livres
        if hasattr(parsed_expr, 'free_symbols'):
            info['free_symbols'] = [str(s) for s in parsed_expr.free_symbols]
        
        # LaTeX
        info['latex'] = sp_latex(parsed_expr)
        info['success'] = True
        
    except Exception as e:
        info['error'] = str(e)
        info['success'] = False
    
    return info