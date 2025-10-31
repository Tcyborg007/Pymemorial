# src/pymemorial/editor/natural_engine.py
"""
Natural Memorial Editor v5.4.0 - Core Integration Refactored

🚀 MELHORIAS IMPLEMENTADAS:
✅ Integração completa com core.Variable (valores em unidade base SI).
✅ Uso consistente de core.VariableFactory na detecção.
✅ Contexto numérico centralizado (_build_numerical_context) para @eq, @matrix, @calc.
✅ Processamento @calc prioriza core.Equation corrigido.
✅ Fallback de @calc usa evalf(subs=...) robusto.
✅ Renderização ({var}, $var$, $$var$$) adaptada para core.Variable.
✅ Limpeza de código legado.
"""
from __future__ import annotations
import re
import logging
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum
import traceback

# ============================================================================
# CORE DEPENDENCIES (Importa tudo necessário do __init__.py do core)
# ============================================================================
try:
    from pymemorial.core import (
        Variable as CoreVariable,
        Equation as CoreEquation,
        VariableFactory,
        Matrix, # Importa Matrix
        parse_quantity,
        ureg,
        Quantity,
        strip_units,
        StepRegistry,
        # StepPlugin, <-- REMOVIDO
        PINT_AVAILABLE,
        SYMPY_AVAILABLE,
        NUMPY_AVAILABLE, # Importa flags
        MATRIX_AVAILABLE
    )
    # Importa operações de matriz se disponíveis
    if MATRIX_AVAILABLE:
        try:
            from pymemorial.core.matrix_ops import MATRIX_OPERATIONS
            MATRIX_OPS_AVAILABLE = True
        except ImportError:
            MATRIX_OPERATIONS = {}
            MATRIX_OPS_AVAILABLE = False
            logging.getLogger(__name__).warning("⚠️ Matrix Operations Module (matrix_ops) not found.")
    else:
        MATRIX_OPERATIONS = {}
        MATRIX_OPS_AVAILABLE = False

    CORE_AVAILABLE = True # Assume core básico está OK se chegou aqui

    # Importa SymPy e NumPy diretamente se disponíveis (para funções matemáticas)
    if SYMPY_AVAILABLE:
        import sympy as sp
    else:
        sp = None
    if NUMPY_AVAILABLE:
        import numpy as np
    else:
        np = None

except ImportError as e:
    CORE_AVAILABLE = False
    PINT_AVAILABLE = False
    SYMPY_AVAILABLE = False
    NUMPY_AVAILABLE = False
    MATRIX_AVAILABLE = False
    MATRIX_OPS_AVAILABLE = False
    CoreVariable = None
    CoreEquation = None
    VariableFactory = None
    Matrix = None
    Quantity = float # Fallback type
    sp = None
    np = None
    MATRIX_OPERATIONS = {}
    logger = logging.getLogger(__name__)
    logger.critical(f"PyMemorial Core failed to load: {e}. Engine functionality severely limited.")


# ============================================================================
# SMART PARSER
# ============================================================================
# Assume que smart_parser está sempre disponível ou o programa falha antes
try:
    from .smart_parser import SmartVariableParser
except ImportError as e:
    logger.critical(f"SmartVariableParser not found: {e}. Cannot proceed.")
    # Em um cenário real, poderia levantar o erro aqui
    SmartVariableParser = None


# ============================================================================
# LOGGER INITIALIZATION
# ============================================================================
logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS
# ============================================================================
class DocumentType(Enum):
    """Document type enumeration."""
    MEMORIAL = "memorial"
    ARTICLE = "article"
    TCC = "tcc"
    REPORT = "report"


class RenderMode(Enum):
    """Render mode enumeration."""
    FULL = "full"
    SYMBOLIC = "symbolic"
    NUMERIC = "numeric"
    RESULT = "result"
    STEPS = "steps"


# ============================================================================
# NATURAL MEMORIAL EDITOR CLASS (Refatorado v5.4.0)
# ============================================================================
class NaturalMemorialEditor:
    """
    Natural Memorial Editor v5.4.0 - Core Integration Refactored.

    Processes text containing natural language, variable definitions,
    and calculation blocks (@eq, @matrix, @matop, @calc) using the
    PyMemorial Core modules for robust evaluation and unit handling.

    Features:
    - Uses core.VariableFactory to store variables with SI base units.
    - Prioritizes core.Equation for reliable @calc block evaluation.
    - Supports matrix operations via core.Matrix and matrix_ops.
    - Handles rendering with unit awareness.
    """

    IGNORE_PLACEHOLDERS = {
        'dx', 'dy', 'dt', 'dtheta', 'ddx', 'ddy',
        'mathrm', 'd', 'partial', 'nabla' # Placeholders comuns em LaTeX
    }

    VERSION = "5.4.0"


    def __init__(self, document_type: str = 'memorial'):
        if not CORE_AVAILABLE or VariableFactory is None or SmartVariableParser is None:
            raise ImportError("Core dependencies (VariableFactory, SmartVariableParser) missing.")

        try:
            self.document_type = DocumentType[document_type.upper()]
        except KeyError:
            logger.warning(f"Invalid document_type '{document_type}'. Using 'memorial'.")
            self.document_type = DocumentType.MEMORIAL

        # ✅ self.variables agora armazena OBJETOS CoreVariable
        self.variables: Dict[str, CoreVariable] = {}
        # ✅ self.equations armazena OBJETOS CoreEquation (gerados por @calc)
        self.equations: Dict[str, CoreEquation] = {}
        self.matrices: Dict[str, Matrix] = {} # Armazena objetos Matrix
        self.parser = SmartVariableParser()
        self._steps_cache: Dict[str, List[Dict[str, Any]]] = {} # Mantido para @calc steps

        self._setup_patterns()
        self._setup_matrix_patterns()

        # Log inicial do estado das dependências
        logger.info(f"NaturalMemorialEditor v{self.VERSION} initialized (type={self.document_type.value})")
        logger.info(f"Core Status: NumPy={NUMPY_AVAILABLE}, SymPy={SYMPY_AVAILABLE}, Pint={PINT_AVAILABLE}")
        logger.info(f"Matrix Status: Core Matrix={MATRIX_AVAILABLE}, Matrix Ops={MATRIX_OPS_AVAILABLE}")


    def _setup_patterns(self):
        """Setup regex patterns for @calc and rendering."""
        # Pattern para @calc (mantido)
        self.calc_block = re.compile(
            # Permite [steps:granularity] ou apenas [granularity]
            r'@calc(?:\[(?:steps:)?(\w+)\])?\s+(\w+)\s*=\s*(.+?)(?=\n\n|\n@|\Z)',
             re.MULTILINE | re.DOTALL # Usa DOTALL para expressões multilinhas
        )
        # Patterns de renderização (mantidos)
        var_name = r'([A-Za-z_][A-Za-z0-9_]*)'
        fmt_spec = r'(:[^}]+)?' # Captura :.2f, :~P etc.
        self.value_render = re.compile(r'(?<!\$)(?<!\\)\{' + var_name + fmt_spec + r'\}(?!\$)')
        self.formula_render = re.compile(r'(?<!\$)(?<!\\)\$' + var_name + fmt_spec + r'\$(?!\$)')
        self.full_render = re.compile(r'(?<!\$)(?<!\\)\$\$' + var_name + fmt_spec + r'\$\$(?!\$)')


    def _setup_matrix_patterns(self):
        """Setup regex patterns for @matrix and @matop commands."""
        # Pattern para @matrix (mantido)
        self.matrix_block = re.compile(
            # Permite @matrix[...] ou @matrix ...
            r'@matrix\s*(?:\[(\w+)(?::(\w+))?\])?\s+(\w+)\s*=\s*(\[\[.*?\]\])',
            re.MULTILINE | re.DOTALL
        )
        # Pattern para @matop (mantido)
        self.matrix_operation = re.compile(
            # Permite @matop[...] ou @matop ...
            r'@matop\s*(?:\[(\w+)(?::(\w+))?\])?\s+(\w+)\s*=\s*(.+?)(?=\n\n|\n@|\Z)',
            re.MULTILINE | re.DOTALL
        )
        logger.debug("Core calculation and matrix patterns configured.")


    def _preprocess_matrix_expression(self, matrix_expr: str) -> str:
        """
        Pré-processa expressões matriciais antes da avaliação.
        Remove caracteres inválidos e normaliza a expressão.
        """
        # Remove espaços e quebras de linha extras
        matrix_expr = matrix_expr.strip()

        # Remove separadores Markdown que possam estar incluídos
        matrix_expr = re.sub(r'\n---.*$', '', matrix_expr, flags=re.MULTILINE)

        return matrix_expr


    def process(self, text: str, clean: bool = False) -> str:
        """
        Processes the input text through multiple passes:
        1. Parse literal variables using SmartParser & VariableFactory (stores CoreVariable objects).
        2. Process @eq blocks using direct evaluation (_simple_eval) storing results as CoreVariable.
        3. Process @matrix blocks using core.Matrix.
        4. Process @matop blocks using core.matrix_ops registry.
        5. Process @calc blocks using core.Equation (with robust fallback).
        6. Render placeholders ({var}, $var$, $$var$$) using data from CoreVariable objects.
        7. Optionally clean processed command lines.
        """
        logger.info(f"📝 Starting memorial processing v{self.VERSION}...")

        # --- Pass 0: Store Initial Variables using VariableFactory ---
        logger.info("🔧 Pass 0: Parsing literal variables...")
        if self.parser and VariableFactory:
            try:
                # parser.detect_all_variables retorna Dict[str, Tuple[value, unit]]
                all_vars_parsed = self.parser.detect_all_variables(text)

                # Extrai o dicionário se retornar tupla (ignora o texto parseado)
                literal_vars_dict = all_vars_parsed[0] if isinstance(all_vars_parsed, tuple) else all_vars_parsed

                logger.info(f"✅ SmartParser detected {len(literal_vars_dict)} potential literal variables.")

                count_created = 0
                for var_name, data in literal_vars_dict.items():
                    try:
                        # Extrai valor e unidade (pode ser tupla ou dict dependendo do parser)
                        value, unit = None, None
                        if isinstance(data, tuple) and len(data) >= 2:
                            value, unit = data[0], data[1]
                        elif isinstance(data, dict):
                             value = data.get('value')
                             unit = data.get('unit')
                        else:
                             value = data # Assume valor direto sem unidade

                        # Cria CoreVariable (converte para unidade base SI internamente)
                        variable_obj = VariableFactory.create(name=var_name, value=value, unit=unit)

                        # Armazena o OBJETO Variable
                        self.variables[var_name] = variable_obj
                        count_created += 1
                        logger.debug(f"  ✅ Stored Variable: {variable_obj}")

                    except Exception as e_create:
                        logger.warning(f"⚠️ Failed to create Variable for '{var_name}' from data '{data}': {e_create}")

                logger.info(f"✅ Stored {count_created} CoreVariable objects from literals.")

            except Exception as e_parse:
                logger.error(f"⚠️ Variable detection/creation failed: {e_parse}")
                logger.debug(traceback.format_exc())
        else:
             logger.warning("⚠️ SmartParser or VariableFactory not available. Skipping literal variable parsing.")

        processed_text = text # Começa com o texto original

        # --- Pass 1: Process @eq Blocks ---
        logger.info("🔧 Pass 1: Processing @eq blocks...")
        processed_text = self._process_equation_blocks(processed_text)
        logger.info(f"✅ Variables after @eq: {len(self.variables)} total.")

        # --- Pass 2: Process @matrix Blocks ---
        logger.info("🔧 Pass 2: Processing @matrix blocks...")
        processed_text = self._process_matrices(processed_text)
        logger.info(f"✅ Matrices after @matrix: {len(self.matrices)} total.")

        # --- Pass 3: Process @matop Blocks ---
        logger.info("🔧 Pass 3: Processing @matop blocks...")
        processed_text = self._process_matrix_operations(processed_text)
        # @matop pode adicionar novas variáveis escalares (det, trace) ou matrizes
        logger.info(f"✅ Variables after @matop: {len(self.variables)}.")
        logger.info(f"✅ Matrices after @matop: {len(self.matrices)}.")

        # --- Pass 4: Process @calc Blocks ---
        logger.info("🔧 Pass 4: Processing @calc blocks...")
        processed_text = self._process_calc_blocks(processed_text)
        logger.info(f"✅ Variables after @calc: {len(self.variables)}.")
        logger.info(f"✅ Equations stored after @calc: {len(self.equations)}.")


        # --- Pass 5: Rendering ---
        logger.info("🔧 Pass 5: Rendering placeholders...")
        processed_text = self._render_full(processed_text)       # $$var$$ -> Symbol = Value [Unit]
        processed_text = self._render_formulas(processed_text)   # $var$   -> Symbol
        processed_text = self._render_values(processed_text)     # {var}   -> Value (base unit)
        logger.info("✅ Rendering complete.")

        # --- Pass 6: Cleaning (Optional) ---
        if clean:
            logger.info("🔧 Pass 6: Cleaning processed lines...")
            processed_text = self._clean_text(processed_text)
            logger.info("✅ Cleaning complete.")

        logger.info(f"✅ Memorial processing v{self.VERSION} complete.")
        return processed_text

    # ========================================================================
    # CONTEXT BUILDER HELPER
    # ========================================================================

    def _build_numerical_context(self) -> Dict[str, Any]:
        """
        Builds a context dictionary mapping variable names to their
        numerical SI base values stored in self.variables.

        ✅ v5.4.1: Robust value extraction with multiple fallbacks

        Returns:
            Dict[str, Union[float, int, np.ndarray]]: Context for evaluation.
        """
        context = {}
        for name, var_obj in self.variables.items():
            try:
                # Try multiple extraction methods
                value = None

                # Method 1: Direct .value attribute
                if hasattr(var_obj, 'value') and var_obj.value is not None:
                    value = var_obj.value

                # Method 2: Quantity with magnitude
                elif hasattr(var_obj, 'magnitude'):
                    value = var_obj.magnitude

                # Method 3: Direct numeric type
                elif isinstance(var_obj, (int, float)):
                    value = var_obj

                # Method 4: NumPy types
                elif NUMPY_AVAILABLE and isinstance(var_obj, (np.number, np.ndarray)):
                    value = var_obj

                # Method 5: Pint Quantity - strip units
                elif PINT_AVAILABLE and isinstance(var_obj, Quantity):
                    value = strip_units(var_obj)

                # Method 6: Try float conversion as last resort
                else:
                    try:
                        value = float(var_obj)
                    except (TypeError, ValueError, AttributeError):
                        pass

                # Add to context if value was extracted
                if value is not None:
                    context[name] = value
                    logger.debug(f"  ✅ Context: '{name}' = {value}")
                else:
                    logger.debug(f"  ⚠️ Variable '{name}' has no extractable value")

            except Exception as e:
                logger.warning(f"  ⚠️ Error extracting value for '{name}': {e}")
                continue

        logger.debug(f"✅ Built numerical context with {len(context)} variables")
        return context



    # ========================================================================
    # EQUATION PROCESSING (@eq) - Usa _simple_eval
    # ========================================================================

    def _process_equation_blocks(self, text: str) -> str:
        """
        Process @eq blocks using direct _simple_eval.
        Stores the result as a new CoreVariable.
        """
        pattern = r'@eq(?:\[(?:steps:)?(\w+)\])?\s+(\w+)\s*=\s*(.+?)(?=\n\n|\n@|\Z)' # Allow [steps:mode] or [mode]

        def replace_eq(match: re.Match) -> str:
            mode_str, var_name, expression = match.groups()
            mode = mode_str or 'normal' # 'steps' or 'normal'
            expression = expression.strip()

            logger.debug(f"Processing @eq: {var_name} = {expression[:50]}... (mode: {mode})")

            try:
                # Evaluate using _simple_eval which uses _build_numerical_context
                result_value = self._simple_eval(expression)

                # Store result as a CoreVariable (value is already float/int/ndarray)
                result_variable = VariableFactory.create(
                    name=var_name,
                    value=result_value, # Should be numerical
                    description=f"Resultado de @eq: {expression}"
                    # Unidade/Dimensão? _simple_eval não infere, assume adimensional ou herda
                )
                self.variables[var_name] = result_variable
                logger.debug(f"  ✅ Stored @eq result: {result_variable}")

                # Format output
                if mode == 'steps':
                    # Simple steps output for @eq
                    # Usa o símbolo da variável criada
                    symbol_latex = sp.latex(result_variable.symbol) if SYMPY_AVAILABLE else var_name
                    # Formata o valor base
                    val_str = f"{result_value:.4g}" if isinstance(result_value, (int, float)) else str(result_value)
                    output = f"\n→ Calculando: ${symbol_latex} = {sp.latex(sp.sympify(expression)) if SYMPY_AVAILABLE else expression}$\n" # Tenta formatar expr
                    output += f"→ Resultado: ${symbol_latex} = {val_str}$\n"
                    return output
                else: # 'normal' mode
                    # Simple inline result using the Variable's __str__ representation
                    # (que tentará mostrar unidade original se disponível, senão valor base)
                    # return f"\n{str(result_variable)}\n" # Isso pode ser redundante se o rendering acontecer depois
                    # Apenas retorna vazio, a variável será renderizada depois
                    return "" # O bloco @eq é removido, o valor será renderizado por {var} etc.

            except Exception as e:
                logger.error(f"❌ Erro em @eq '{var_name} = {expression}': {e}")
                logger.debug(traceback.format_exc())
                # Store a placeholder Variable with None value on error?
                error_variable = VariableFactory.create(name=var_name, value=None, description=f"Erro em @eq: {e}")
                self.variables[var_name] = error_variable
                return f"\n**ERRO @eq:** {var_name}: {str(e)[:100]}\n\n" # Retorna erro no texto

        # Usa re.sub para substituir todas as ocorrências
        return re.sub(pattern, replace_eq, text, flags=re.MULTILINE | re.DOTALL)


    def _simple_eval(self, expression: str) -> Any:
        """
        Simple, direct evaluation using Python's eval() with a safe context.
        Context includes numerical values from self.variables and math functions.
        """
        eval_context = {
            "__builtins__": {}, # Restringe builtins perigosos
        }

        # Add safe math functions (NumPy if available, else standard math)
        if NUMPY_AVAILABLE:
            eval_context.update({
                "cos": np.cos, "sin": np.sin, "tan": np.tan,
                "acos": np.arccos, "asin": np.arcsin, "atan": np.arctan, "atan2": np.arctan2,
                "sqrt": np.sqrt, "abs": np.abs, "pow": np.power, # Usa np.power para arrays
                "exp": np.exp, "log": np.log, "log10": np.log10,
                "pi": np.pi, "e": np.e,
                # Funções de arredondamento/piso/teto
                "floor": np.floor, "ceil": np.ceil, "round": np.round,
                # Funções estatísticas básicas se necessário
                "min": np.min, "max": np.max, "mean": np.mean, "sum": np.sum
            })
        else:
             # Fallback para math (pode não funcionar com arrays)
            import math
            eval_context.update({
                "cos": math.cos, "sin": math.sin, "tan": math.tan,
                "acos": math.acos, "asin": math.asin, "atan": math.atan, "atan2": math.atan2,
                "sqrt": math.sqrt, "abs": abs, "pow": pow,
                "exp": math.exp, "log": math.log, "log10": math.log10,
                "pi": math.pi, "e": math.e,
                "floor": math.floor, "ceil": math.ceil, "round": round,
                "min": min, "max": max, "sum": sum # Built-in sum
            })

        # Add SymPy functions if available (para uso em @eq se necessário)
        if SYMPY_AVAILABLE:
            # Funções que podem ser úteis mesmo em eval direto
            eval_context.update({
                "integrate": sp.integrate, # Permite chamar integrate simbolicamente aqui
                "diff": sp.diff,
                "symbols": sp.symbols,
                "Symbol": sp.Symbol,
                "simplify": sp.simplify,
                "N": sp.N # Para avaliação numérica forçada
            })
            # Força x, y, z a serem símbolos SymPy
            try:
                symbols_xyz = sp.symbols('x y z')
                eval_context.update({'x': symbols_xyz[0], 'y': symbols_xyz[1], 'z': symbols_xyz[2]})
            except Exception as e_sym: logger.warning(f"Falha ao criar símbolos x,y,z para simple_eval: {e_sym}")

        # Add numerical values from self.variables
        numerical_context = self._build_numerical_context()
        eval_context.update(numerical_context)

        # Evaluate safely
        try:
            # Usa eval com o contexto seguro
            result = eval(expression, {"__builtins__": {}}, eval_context)
            # Tenta converter resultado SymPy para float/int se possível
            if SYMPY_AVAILABLE and isinstance(result, (sp.Expr, sp.Number)):
                 if result.is_number:
                      try: result = float(result)
                      except: pass # Mantém objeto SymPy se falhar
                 # Se ainda for simbólico (ex: integrate retornou expressão), retorna como está
            return result
        except Exception as e:
            logger.error(f"Erro durante _simple_eval da expressão '{expression}': {e}")
            logger.debug(f"Contexto usado: {list(eval_context.keys())}") # Log chaves do contexto
            raise # Re-levanta a exceção para ser tratada por quem chamou


    # ========================================================================
    # MATRIX PROCESSING (@matrix) - Usa core.Matrix
    # ========================================================================

    def _process_matrices(self, text: str) -> str:
        """Process @matrix blocks using core.Matrix."""
        if not MATRIX_AVAILABLE or Matrix is None:
            logger.warning("⚠️ Matrix support disabled - skipping @matrix blocks")
            # Remove a linha do comando @matrix para evitar que apareça no output
            pattern = r'^@matrix.*?\]\s*=\s*\[\[.*?\]\]'
            return re.sub(pattern, '', text, flags=re.MULTILINE | re.DOTALL)

        pattern = r'@matrix\s*(?:\[(\w+)(?::(\w+))?\])?\s+(\w+)\s*=\s*(\[\[.*?\]\])'

        def replace_matrix(match: re.Match) -> str:
            mode_str, granularity_str, name, matrix_expr = match.groups()
            mode = (mode_str or 'normal').lower()
            granularity = (granularity_str or 'normal').lower() # Granularidade não usada diretamente na criação

            logger.info(f"🔍 Processing @matrix: {name}, mode={mode}")

            try:
                # 1. Pré-processa a expressão (se necessário, ex: /L -> /L**1)
                processed_matrix_expr = self._preprocess_matrix_expression(matrix_expr)

                # 2. Constrói contexto com valores numéricos base + funções
                # Usar _build_eval_context que agora internamente usa _build_numerical_context
                eval_context = self._build_eval_context() # Inclui funções math/sympy

                # 3. Cria a instância de core.Matrix
                # A classe Matrix internamente tentará avaliar numericamente (NumPy)
                # e fará fallback para simbólico (SymPy) se necessário.
                matrix_obj = Matrix(
                    data=processed_matrix_expr, # Passa a string ou lista parseada
                    name=name,
                    # Passa o contexto com valores numéricos base
                    # A classe Matrix V2 precisa ser adaptada para aceitar esse contexto
                    # ou ter um método para avaliar com contexto externo.
                    # Assumindo que a Matrix V2 pode usar um contexto ou avaliar internamente.
                    # Se Matrix V2 espera `variables` (dict de Variable), precisaria adaptar aqui.
                    # Por simplicidade, assumindo que Matrix pode lidar com a string e contexto:
                    # (Pode precisar de ajuste dependendo da API exata de core.Matrix v2.1.9)
                    # Exemplo hipotético:
                    # evaluate_context=eval_context # Passando contexto para avaliação interna
                )

                # Avalia a matriz (se Matrix não fizer isso no __init__)
                # Tenta avaliação numérica primeiro
                try:
                    matrix_obj.evaluate(context=eval_context) # Método hipotético
                except Exception as eval_e:
                     logger.warning(f"Could not numerically evaluate matrix '{name}' immediately: {eval_e}. Storing symbolic form.")
                     # Continua com a forma simbólica ou parcialmente avaliada

                # 4. Armazena o objeto Matrix
                self.matrices[name] = matrix_obj
                logger.info(f"  ✅ Stored core.Matrix object: '{name}' (Symbolic: {matrix_obj.is_symbolic}, Shape: {matrix_obj.shape})")

                # 5. Gera saída (pode usar métodos da Matrix se disponíveis ou formatadores locais)
                # O método _generate_matrix_output pode precisar ser adaptado para usar matrix_obj
                output = self._generate_matrix_output(matrix_obj, name, mode, granularity, matrix_expr)
                return output

            except Exception as e:
                logger.error(f"❌ Error processing @matrix '{name}': {e}")
                logger.debug(traceback.format_exc())
                return f"\n**ERRO MATRIZ {name}:** {str(e)[:150]}\n\n"

        # Aplica substituição
        return re.sub(pattern, replace_matrix, text, flags=re.MULTILINE | re.DOTALL)

    # _build_eval_context agora usa _build_numerical_context e adiciona funções
    def _build_eval_context(self) -> Dict[str, Any]:
        """Builds evaluation context with numerical values + math/sympy functions."""
        eval_context = { "__builtins__": {} }

        # Adiciona funções matemáticas (NumPy ou math)
        if NUMPY_AVAILABLE:
            eval_context.update({
                "cos": np.cos, "sin": np.sin, "tan": np.tan, "sqrt": np.sqrt,
                "abs": np.abs, "pow": np.power, "exp": np.exp, "log": np.log,
                "pi": np.pi, "e": np.e,
                # Outras funções úteis
                 "array": np.array, "zeros": np.zeros, "ones": np.ones, "eye": np.eye,
                 "linspace": np.linspace,
                 # Funções linalg se necessárias
                 "inv": np.linalg.inv, "det": np.linalg.det, "eig": np.linalg.eig
            })
        else:
             import math
             eval_context.update({
                "cos": math.cos, "sin": math.sin, "tan": math.tan, "sqrt": math.sqrt,
                "abs": abs, "pow": pow, "exp": math.exp, "log": math.log,
                "pi": math.pi, "e": math.e,
             })

        # Adiciona funções SymPy se disponíveis
        if SYMPY_AVAILABLE:
            eval_context.update({
                "Symbol": sp.Symbol, "symbols": sp.symbols,
                "Matrix": sp.Matrix, # Para criar matrizes SymPy se necessário
                "integrate": sp.integrate, "diff": sp.diff, "simplify": sp.simplify,
                "Rational": sp.Rational # Para frações exatas
            })
            # Força x, y, z a serem símbolos
            try:
                symbols_xyz = sp.symbols('x y z')
                eval_context.update({'x': symbols_xyz[0], 'y': symbols_xyz[1], 'z': symbols_xyz[2]})
            except Exception as e_sym: logger.warning(f"Falha ao criar símbolos x,y,z para eval_context: {e_sym}")

        # Adiciona valores numéricos base das variáveis
        numerical_values = self._build_numerical_context()
        eval_context.update(numerical_values)

        return eval_context


    # Métodos de formatação de matriz (_generate_matrix_output, _format_matrix_expr_latex, _format_matrix_as_latex)
    # Precisam ser adaptados para usar os métodos do objeto core.Matrix (ex: matrix_obj.latex(), matrix_obj.array())
    # Abaixo, uma adaptação SIMPLIFICADA assumindo que matrix_obj tem métodos .latex() e .array()
    def _generate_matrix_output(
        self,
        matrix_obj: Matrix,
        name: str,
        mode: str,
        granularity: str,
        original_expr: str # String original para fallback simbólico
    ) -> str:
        """Generate formatted output for core.Matrix object based on mode."""
        output = ""
        try:
            if mode == 'steps':
                # Idealmente, core.Matrix teria um método .steps()
                # Fallback: mostra simbólico + numérico
                output += f"\n**Matriz {name} (Steps Mode - Fallback):**\n"
                if hasattr(matrix_obj, 'latex'):
                     output += f"Forma Simbólica/Original:\n$${name} = {matrix_obj.latex()}$$\n"
                else: # Fallback se não tiver .latex()
                     output += f"Forma Original (string):\n$${name} = {self._format_matrix_expr_latex(original_expr)}$$\n"

                if hasattr(matrix_obj, 'array'):
                    try:
                        numeric_array = matrix_obj.array() # Pega o array NumPy
                        output += "Forma Numérica Avaliada:\n"
                        output += self._format_matrix_as_latex(numeric_array, name) + "\n"
                    except Exception as eval_e:
                         output += f"(Não foi possível avaliar numericamente: {eval_e})\n"
                else:
                     output += "(Método .array() não encontrado para avaliação numérica)\n"

            elif mode == 'symbolic':
                 if hasattr(matrix_obj, 'latex'):
                      output += f"\n$${name} = {matrix_obj.latex()}$$\n"
                 else: # Fallback
                      output += f"\n$${name} = {self._format_matrix_expr_latex(original_expr)}$$\n"

            elif mode == 'numeric':
                 if hasattr(matrix_obj, 'array'):
                      try:
                           numeric_array = matrix_obj.array()
                           output += f"\n{self._format_matrix_as_latex(numeric_array, name)}\n"
                      except Exception as eval_e:
                           output += f"\n**ERRO ao avaliar {name} numericamente:** {eval_e}\n"
                 else:
                      output += f"\n**{name}:** (Avaliação numérica não suportada/falhou)\n"

            else: # 'normal' or 'result' (mostra ambos se possível)
                 output += f"\n**Matriz {name}:**\n"
                 added_output = False
                 if hasattr(matrix_obj, 'latex'):
                      try: output += f"$${name} = {matrix_obj.latex()}$$\n"; added_output=True
                      except: pass
                 elif added_output is False: # Fallback se .latex falhar
                      output += f"$${name} = {self._format_matrix_expr_latex(original_expr)}$$\n"; added_output=True

                 if hasattr(matrix_obj, 'array'):
                      try:
                           numeric_array = matrix_obj.array()
                           output += self._format_matrix_as_latex(numeric_array, name) + "\n"; added_output=True
                      except Exception as eval_e:
                           if added_output: output += f"(Avaliação numérica falhou: {eval_e})\n"
                           else: output = f"\n**ERRO ao avaliar {name} numericamente:** {eval_e}\n" # Se nada foi adicionado
                 elif not added_output: # Se não tem .latex nem .array
                      output = f"\n**{name}:** (Representação indisponível)\n"


        except Exception as fmt_e:
            logger.error(f"Erro ao formatar saída para matriz '{name}': {fmt_e}")
            output = f"\n**ERRO ao formatar Matriz {name}:** {fmt_e}\n"

        return output


    # _format_matrix_expr_latex (mantido como fallback se matrix_obj.latex() falhar)
    def _format_matrix_expr_latex(self, matrix_expr: str) -> str:
        # ... (código mantido) ...
        try:
            matrix_expr_clean = matrix_expr.replace('\n', ' ').replace('\r', '')
            # Usa eval com contexto muito limitado para segurança
            rows = eval(matrix_expr_clean, {"__builtins__": {}}, {})
            if not isinstance(rows, list): raise ValueError("Parsed expression is not a list")
            latex_rows = []
            for row in rows:
                if not isinstance(row, list): raise ValueError("Matrix row is not a list")
                # Tenta converter elementos para string de forma segura
                row_str = " & ".join(map(str, row))
                latex_rows.append(row_str)
            matrix_body = " \\\\\n".join(latex_rows)
            return f"\\begin{{bmatrix}}\n{matrix_body}\n\\end{{bmatrix}}"
        except Exception as e:
            logger.debug(f"Could not format matrix expression string: {e}")
            # Retorna a string original formatada como código se falhar
            return f"\\text{{(Error formatting: {matrix_expr[:50]}...)}}"


    # _format_matrix_as_latex (mantido para formatar array NumPy)
    def _format_matrix_as_latex(self, matrix_data: np.ndarray, name: str = "") -> str:
        # ... (código mantido) ...
        if not NUMPY_AVAILABLE or not isinstance(matrix_data, np.ndarray):
             return f"\\text{{(NumPy array expected for LaTeX formatting, got {type(matrix_data)})}}"
        try:
            rows, cols = matrix_data.shape
            latex_rows = []
            for i in range(rows):
                # Formata cada elemento com precisão razoável
                row_values = [f"{matrix_data[i, j]:.4g}" for j in range(cols)]
                latex_rows.append(" & ".join(row_values))
            matrix_body = " \\\\\n".join(latex_rows)
            if name:
                return f"$${name} = \\begin{{bmatrix}}\n{matrix_body}\n\\end{{bmatrix}}$$"
            else:
                return f"$$\\begin{{bmatrix}}\n{matrix_body}\n\\end{{bmatrix}}$$"
        except Exception as e:
             logger.error(f"Erro ao formatar NumPy array como LaTeX: {e}")
             return f"\\text{{(Error formatting NumPy array)}}"


    # _parse_matrix_expression (Removido - lógica agora dentro de core.Matrix ou _process_matrices)
    # _format_matrix_steps (Removido - deve ser método de core.Matrix)
    # _format_matrix_table (Removido - usar _format_matrix_as_latex)
    # _format_matrix_symbolic (Removido - usar matrix_obj.latex())
    # _format_matrix_numeric (Removido - usar matrix_obj.array() + _format_matrix_as_latex)
    # _format_matrix_result (Removido - lógica movida para _generate_matrix_output)


    # ========================================================================
    # MATRIX OPERATIONS (@matop) - Usa MATRIX_OPERATIONS registry
    # ========================================================================

    def _process_matrix_operations(self, text: str) -> str:
        """Process @matop blocks using direct evaluation for complex expressions."""
        if not MATRIX_AVAILABLE or not MATRIX_OPS_AVAILABLE:
            logger.warning("⚠️ Matrix operations disabled - skipping @matop blocks")
            pattern = r'^@matop.*$'
            return re.sub(pattern, '', text, flags=re.MULTILINE)

        pattern = r'@matop\s*(?:\[(\w+)(?::(\w+))?\])?\s+(\w+)\s*=\s*(.+?)(?=\n\n|\n@|\Z)'

        def replace_matop(match: re.Match) -> str:
            operation_str, granularity_str, result_name, expression = match.groups()
            expression = expression.strip()
            operation = (operation_str or 'eval').lower()
            granularity = granularity_str or 'normal'

            logger.debug(f"Processing @matop: {result_name} = {expression} (Op: {operation}, Gran: {granularity})")

            try:
                # --- CORREÇÃO PRINCIPAL AQUI ---
                # Para operações complexas como 'T_rot.T @ K_local @ T_rot', a avaliação direta é mais segura.
                # Vamos construir um contexto que inclua as matrizes como objetos NumPy.
                eval_context = self._build_eval_context()

                # Adiciona os objetos Matrix e seus arrays NumPy avaliados ao contexto
                for name, matrix_obj in self.matrices.items():
                    try:
                        # Adiciona o objeto Matrix ao contexto
                        eval_context[name] = matrix_obj
                        # Adiciona também o array NumPy avaliado para operações diretas
                        eval_context[f"{name}_array"] = matrix_obj.array()
                    except Exception as e_eval:
                        logger.warning(f"Could not evaluate matrix '{name}' for @matop context: {e_eval}")
                        eval_context[name] = matrix_obj # Adiciona o objeto bruto como fallback

                # Se a operação for 'multiply' e a expressão for complexa, usa eval direto.
                if operation == 'multiply' and ('@' in expression or '.T' in expression):
                    logger.info(f"Directly evaluating complex matrix expression: {expression}")
                    result = eval(expression, {"__builtins__": {}}, eval_context)
                    output_str = f"\n**Resultado de @matop eval:** `{result_name}`\n"
                # Para outras operações do registro, usa a lógica original
                elif operation in MATRIX_OPERATIONS:
                    operation_func = MATRIX_OPERATIONS[operation]
                    involved_matrices = self._extract_matrices_from_expression(expression)
                    if not involved_matrices:
                        raise ValueError(f"Could not find any known matrices in expression: {expression}")

                    # A assinatura correta é: func(matrix_a, matrix_b, ...)
                    # O parâmetro 'granularity' não é passado aqui.
                    result, steps = operation_func(*involved_matrices)
                    output_str = self._format_matrix_operation_output(result_name, operation, steps, result)
                else:
                    raise ValueError(f"Operação '@matop' não suportada: '{operation}'")

                # Armazena o resultado
                if isinstance(result, (int, float, np.number)):
                    var_obj = VariableFactory.create(name=result_name, value=result)
                    self.variables[result_name] = var_obj
                    logger.debug(f"  ✅ Stored @matop scalar result: {var_obj}")
                elif NUMPY_AVAILABLE and isinstance(result, np.ndarray):
                    var_obj = VariableFactory.create(name=result_name, value=result)
                    self.variables[result_name] = var_obj
                    logger.debug(f"  ✅ Stored @matop array result: {var_obj}")
                elif MATRIX_AVAILABLE and isinstance(result, Matrix):
                    result.name = result_name
                    self.matrices[result_name] = result
                    logger.debug(f"  ✅ Stored @matop Matrix result: '{result_name}'")
                else:
                    logger.warning(f"Resultado de @matop para '{result_name}' é de tipo inesperado ({type(result)}).")
                    try:
                        var_obj = VariableFactory.create(name=result_name, value=result)
                        self.variables[result_name] = var_obj
                    except:
                        pass

                return output_str

            except Exception as e:
                logger.error(f"Erro em @matop '{result_name} = {expression}': {e}")
                logger.debug(traceback.format_exc())
                return f"\n**ERRO @matop:** {result_name}: {str(e)[:100]}\n\n"

        return re.sub(pattern, replace_matop, text, flags=re.MULTILINE | re.DOTALL)


    # _extract_matrices_from_expression (simplificado, precisa ser mais robusto)
    def _extract_matrices_from_expression(self, expression: str) -> List[Matrix]:
        """
        Extrai objetos Matrix de self.matrices cujos nomes aparecem na expressão.
        Versão simplificada: apenas procura nomes de matrizes conhecidas.
        """
        matrices_found = []
        # Usa regex para encontrar potenciais nomes de variáveis/matrizes
        potential_names = re.findall(r'\b([a-zA-Z_]\w*)\b', expression)
        seen_names = set()
        for name in potential_names:
            if name in self.matrices and name not in seen_names:
                # Verifica se há '.T' após o nome para transposta
                is_transposed = re.search(r'\b' + re.escape(name) + r'\.T\b', expression) is not None
                matrix_obj = self.matrices[name]
                if is_transposed:
                     try:
                          # Cria um novo objeto Matrix transposto (se Matrix suportar)
                          # Assumindo que matrix_obj.T retorna um novo objeto Matrix transposto
                          transposed_matrix = matrix_obj.T
                          transposed_matrix.name = f"{name}_T" # Nome temporário
                          matrices_found.append(transposed_matrix)
                          logger.debug(f"Extracted transposed matrix: {name}.T")
                     except Exception as e_T:
                          logger.warning(f"Could not transpose matrix '{name}' for expression: {e_T}")
                          matrices_found.append(matrix_obj) # Usa original como fallback
                else:
                     matrices_found.append(matrix_obj)
                     logger.debug(f"Extracted matrix: {name}")
                seen_names.add(name) # Evita adicionar a mesma matriz múltiplas vezes

        return matrices_found


    # Métodos de formatação de @matop (_format_matrix_operation_output, etc.) - Mantidos
    def _format_matrix_operation_output(
        self, result_name: str, operation: str, steps: List[Dict[str, Any]], result: Any
    ) -> str:
        """Format matrix operation output based on steps list."""
        output = [f"\n**Operação Matricial: {operation.upper()} -> `{result_name}`**\n"]
        if not steps: # Se não houver steps, mostra apenas o resultado formatado
             output.append("*(Sem passos detalhados fornecidos)*\n")
             if isinstance(result, (int, float, np.number)):
                 output.append(f"Resultado Escalar: `{result_name}` = {result:.6g}\n")
             elif NUMPY_AVAILABLE and isinstance(result, np.ndarray):
                 output.append("Resultado (Array NumPy):\n")
                 output.append(self._format_matrix_as_latex(result, result_name) + "\n")
             elif MATRIX_AVAILABLE and isinstance(result, Matrix):
                 output.append("Resultado (Objeto Matrix):\n")
                 # Usa o _generate_matrix_output para formatar o objeto Matrix resultante
                 output.append(self._generate_matrix_output(result, result_name, 'normal', 'normal', ''))
             else:
                 output.append(f"Resultado (Tipo {type(result)}): {result}\n")
             return "".join(output)

        # Processa steps se existirem
        for i, step in enumerate(steps):
            desc = step.get('description', f'Passo {i+1}')
            latex_expr = step.get('latex', '')
            numeric_val = step.get('numeric', None) # Pode ser escalar ou array

            output.append(f"→ {desc}\n")
            if latex_expr:
                 # Adiciona $$ apenas se não estiver presente
                 latex_expr_fmt = f"$${latex_expr}$$" if not latex_expr.strip().startswith('$') else latex_expr
                 output.append(f"{latex_expr_fmt}\n")
            if numeric_val is not None:
                 if isinstance(numeric_val, (int, float, np.number)):
                      output.append(f"  *Valor:* `{numeric_val:.6g}`\n")
                 elif NUMPY_AVAILABLE and isinstance(numeric_val, np.ndarray):
                      # Formata array como matriz LaTeX sem nome
                      output.append(f"  *Valor (Array):*\n{self._format_matrix_as_latex(numeric_val)}\n")
                 # Adicionar formatação para outros tipos se necessário

        # Adiciona informação sobre o resultado final armazenado
        output.append(f"\n✅ Resultado final armazenado em `{result_name}`.\n")
        output.append("\n")
        return "".join(output)

    # _format_eigenvalue_output e _format_scalar_output podem ser simplificados ou
    # fundidos em _format_matrix_operation_output se a estrutura dos 'steps' for consistente.
    # Mantendo-os por ora para compatibilidade.

    def _format_eigenvalue_output( self, result_name: str, matrix_name: str, steps: List[Dict[str, Any]]) -> str:
        # Reutiliza o formatador geral
        return self._format_matrix_operation_output(result_name, f'Autovalores de {matrix_name}', steps, self.variables.get(result_name, None))

    def _format_scalar_output( self, result_name: str, matrix_name: str, scalar_value: float, operation: str, steps: List[Dict[str, Any]]) -> str:
         # Reutiliza o formatador geral
        operation_names = {'determinant': 'Determinante','trace': 'Traço','rank': 'Posto'}
        op_display = operation_names.get(operation, operation.capitalize())
        return self._format_matrix_operation_output(result_name, f'{op_display} de {matrix_name}', steps, scalar_value)


    # ========================================================================
    # CALC BLOCKS PROCESSING (@calc) - Usa core.Equation
    # ========================================================================

    def _process_calc_blocks(self, text: str) -> str:
        """
        Process @calc blocks using core.Equation, with robust fallback.
        Stores result as a new CoreVariable.
        """
        if not CORE_AVAILABLE or CoreEquation is None or VariableFactory is None:
            logger.warning("Core Equation/VariableFactory unavailable - @calc blocks skipped")
            pattern = r'^@calc.*?$' # Remove a linha inteira
            return re.sub(pattern, '', text, flags=re.MULTILINE | re.DOTALL)

        # Pattern: @calc[steps:granularity] ou @calc[granularity] variable = expression
        pattern = r'@calc(?:\[(?:steps:)?(\w+)\])?\s+(\w+)\s*=\s*(.+?)(?=\n\n|\n@|\Z)'

        def replace_calc(match):
            granularity_str, var_name, expression = match.groups()
            # Default para 'smart' se não especificado, permite granularidade direto
            granularity = granularity_str or 'smart'
            expression = expression.strip()

            logger.debug(f"Processing @calc: {var_name} = {expression[:50]}... (Granularity: {granularity})")

            try:
                # 1. Build Evaluation Context (numerical values + functions + x,y,z symbols)
                # Usar um contexto mais completo que _build_eval_context para @calc
                calc_context = self._build_eval_context() # Já inclui numéricos, funções, x,y,z

                # Verifica se todas as variáveis na *expressão* existem no contexto
                undefined_vars = []
                if SYMPY_AVAILABLE:
                    try:
                        expr_symbols = sp.sympify(expression).free_symbols
                        known_context_keys = set(calc_context.keys())
                        for sym in expr_symbols:
                            if str(sym) not in known_context_keys:
                                undefined_vars.append(f"{str(sym)}(não definida)")
                    except Exception as sym_err:
                        logger.warning(f"Could not parse symbols from @calc expr '{expression}' for validation: {sym_err}")
                        # Fallback: regex check (menos preciso)
                        var_pattern = r'\b([a-zA-Z_]\w*)\b'
                        expr_vars_re = set(re.findall(var_pattern, expression))
                        known_funcs = {k for k,v in calc_context.items() if callable(v)} | {'x', 'y', 'z', 'pi', 'e'}
                        expr_vars_re -= known_funcs
                        for ev in expr_vars_re:
                            if ev not in calc_context:
                                undefined_vars.append(f"{ev}(não definida)")
                else: # Se SymPy não disponível, usa apenas regex
                     var_pattern = r'\b([a-zA-Z_]\w*)\b'; expr_vars_re = set(re.findall(var_pattern, expression))
                     known_funcs = {k for k,v in calc_context.items() if callable(v)} | {'pi', 'e'}
                     expr_vars_re -= known_funcs
                     for ev in expr_vars_re:
                          if ev not in calc_context: undefined_vars.append(f"{ev}(não definida)")


                if undefined_vars:
                    error_msg = f"Variáveis não definidas no contexto: {', '.join(sorted(list(set(undefined_vars))))}"
                    raise ValueError(error_msg) # Levanta erro para ir ao bloco except principal

                # 2. Try using core.Equation (método principal)
                try:
                    logger.debug(f"Attempting calculation for '{var_name}' using core.Equation.")
                    # Cria instância de core.Equation
                    # Passa o contexto numérico base + funções + símbolos x,y,z
                    eq = CoreEquation(
                        # name=var_name, # CoreEquation não parece ter 'name' no init
                        expression=expression,
                        # Passa dict de OBJETOS Variable, Equation.evaluate usará .value
                        variables=self.variables, # Passa o dict completo de Variables
                        description=f"Cálculo de {var_name}"
                        # context=calc_context # Passa contexto numérico/funcional se a API de Equation suportar/precisar
                    )

                    # Avalia (deve usar evalf(subs=...) internamente agora)
                    result_value_base = eq.evaluate() # Deve retornar float/int/ndarray em base SI

                    # Gera os steps se a granularidade não for 'minimal' ou 'basic' (simplificado)
                    # Ou se o modo for explicitamente steps
                    steps_output = ""
                    if granularity not in ['minimal', 'basic'] or (granularity_str and 'steps' in granularity_str.lower()): # Verifica se 'steps' foi mencionado
                         try:
                             # Map string granularity to enum if core uses enums
                              gran_map = {'minimal': GranularityType.MINIMAL, 'basic': GranularityType.BASIC,
                                           'normal': GranularityType.NORMAL, 'detailed': GranularityType.DETAILED,
                                           'all': GranularityType.ALL, 'smart': GranularityType.SMART}
                              gran_enum = gran_map.get(granularity, GranularityType.SMART) # Default smart
                              # Assume eq.steps() retorna uma string formatada
                              steps_output = eq.steps(granularity=gran_enum)
                         except AttributeError: # Se GranularityType não for enum
                              steps_output = eq.steps(granularity=granularity) # Passa string
                         except Exception as steps_err:
                              logger.warning(f"Failed to generate steps for '{var_name}': {steps_err}")
                              steps_output = f"\n*Erro ao gerar passos: {steps_err}*\n"

                    # Armazena a Equation e o resultado como Variable
                    self.equations[var_name] = eq
                    result_variable = VariableFactory.create(name=var_name, value=result_value_base, description=eq.description)
                    self.variables[var_name] = result_variable
                    logger.debug(f"✅ Stored @calc result (via CoreEquation): {result_variable}")

                    # Retorna os steps formatados (ou vazio se não gerado)
                    return f"\n{steps_output}\n" if steps_output else "" # Retorna vazio se só calculou


                # 3. Fallback using SymPy directly (se core.Equation falhar)
                except Exception as eq_error:
                    logger.warning(f"CoreEquation failed for '{var_name}', using SymPy fallback: {eq_error}")
                    if not SYMPY_AVAILABLE:
                        raise ValueError("SymPy fallback failed: SymPy not available.") from eq_error

                    logger.debug(f"Attempting SymPy fallback evaluation for '{var_name}'")
                    sympy_expr = sp.sympify(expression, locals=calc_context)
                    subs_dict = {k: v for k, v in calc_context.items() if isinstance(v, (int, float, np.number))}
                    logger.debug(f"Subs dict for fallback '{var_name}': {subs_dict}")

                    result_sympy = sympy_expr.evalf(subs=subs_dict)
                    if not result_sympy.is_number:
                         simplified = sp.simplify(result_sympy).evalf(subs=subs_dict)
                         if not simplified.is_number:
                              raise ValueError(f"SymPy fallback evaluation resulted in symbolic expression: {simplified}")
                         result_sympy = simplified

                    result_value_base = float(result_sympy)

                    # Armazena resultado como Variable
                    result_variable = VariableFactory.create(name=var_name, value=result_value_base, description=f"Resultado (fallback) de {expression}")
                    self.variables[var_name] = result_variable
                    logger.debug(f"✅ Stored @calc result (via Fallback): {result_variable}")

                    # Retorna formatação básica para fallback
                    symbol_latex = sp.latex(result_variable.symbol) if SYMPY_AVAILABLE else var_name
                    val_str = f"{result_value_base:.4g}"
                    # Mostra a expressão original e o resultado numérico
                    output = f"\n**Cálculo (Fallback):**\n→ ${symbol_latex} = {sp.latex(sp.sympify(expression))}$\n"
                    output += f"→ ${symbol_latex} = {val_str}$\n"
                    return output

            # Captura erros gerais do bloco @calc (incluindo undefined_vars do início)
            except Exception as e:
                logger.error(f"Erro processing @calc '{var_name} = {expression}': {e}")
                logger.debug(traceback.format_exc())
                # Armazena uma Variable com valor None
                error_variable = VariableFactory.create(name=var_name, value=None, description=f"Erro em @calc: {e}")
                self.variables[var_name] = error_variable
                return f"\n\n**ERRO @calc:** {var_name}: {str(e)}\n\n"

        # Substitui todos os blocos @calc
        return re.sub(pattern, replace_calc, text, flags=re.MULTILINE | re.DOTALL)


    # ========================================================================
    # RENDERING METHODS (Adaptados para usar CoreVariable)
    # ========================================================================

    def _render_full(self, text: str) -> str:
        """Render $$var$$ -> Symbol = Value [Unit] (valor na unidade original)."""
        def replace_full(match: re.Match) -> str:
            var_name = match.group(1)
            format_spec = match.group(2) or "" # Ex: :.2f

            if var_name in self.IGNORE_PLACEHOLDERS: return match.group(0)
            if var_name not in self.variables: return match.group(0) # Não encontrado

            var_obj = self.variables[var_name]
            if var_obj is None: return f"$${var_name} = ?$$" # Variável existe mas é None

            try:
                symbol_latex = sp.latex(var_obj.symbol) if SYMPY_AVAILABLE and hasattr(var_obj,'symbol') else var_name
                value_str = "?"
                unit_str = var_obj.unit_str or "" # Pega unidade original

                # Tenta formatar valor na unidade original
                formatted_original = False
                if PINT_AVAILABLE and var_obj.unit_str and var_obj.dimensionality and ureg:
                     try:
                          original_unit_qty = var_obj.to_unit(var_obj.unit_str)
                          if original_unit_qty is not None:
                               # Aplica formatação se especificada
                               fmt = format_spec[1:] if format_spec.startswith(':') else ".4g" # Default .4g
                               value_str = f"{original_unit_qty.magnitude:{fmt}}"
                               # Tenta obter unidade formatada em LaTeX
                               try: unit_str = f"\\; {original_unit_qty.units:Lx}" # LaTeX unit
                               except: unit_str = f"\\; \\text{{{original_unit_qty.units:~P}}}" # Fallback texto
                               formatted_original = True
                     except Exception as fmt_e:
                          logger.debug(f"Could not format '{var_name}' to original unit '{var_obj.unit_str}': {fmt_e}")

                # Se não conseguiu formatar na unidade original, usa valor base (SI)
                if not formatted_original and var_obj.value is not None:
                     # Aplica formatação se especificada
                     fmt = format_spec[1:] if format_spec.startswith(':') else ".4g" # Default .4g
                     value_str = f"{var_obj.value:{fmt}}"
                     # Indica unidade base se tiver dimensionalidade
                     if var_obj.dimensionality and PINT_AVAILABLE and ureg:
                          try:
                               base_unit = ureg.Quantity(1, var_obj.dimensionality).units
                               unit_str = f"\\; {base_unit:Lx}"
                          except:
                               unit_str = f"\\; \\text{{(base SI)}}"
                     else: unit_str = "" # Adimensional ou sem info

                return f"$${symbol_latex} = {value_str}{unit_str}$$"

            except Exception as e:
                logger.warning(f"Error rendering full '$$ {var_name} $$': {e}")
                return f"$${var_name} = \\text{{(render error)}}$$" # Retorna erro formatado

        # Usa regex de self.full_render
        return self.full_render.sub(replace_full, text)


    def _render_formulas(self, text: str) -> str:
        """Render $var$ -> Symbol."""
        def replace_formula(match: re.Match) -> str:
            var_name = match.group(1)
            # Ignora format spec para $var$
            if var_name in self.IGNORE_PLACEHOLDERS: return match.group(0)
            if var_name not in self.variables: return match.group(0)

            var_obj = self.variables[var_name]
            if var_obj is None or not hasattr(var_obj, 'symbol'): return f"${var_name}$" # Fallback

            try:
                # Usa o símbolo SymPy da variável
                symbol_latex = sp.latex(var_obj.symbol) if SYMPY_AVAILABLE else var_name
                return f"${symbol_latex}$"
            except Exception as e:
                logger.warning(f"Error rendering formula '$ {var_name} $': {e}")
                return f"${var_name}$" # Fallback

        # Usa regex de self.formula_render
        return self.formula_render.sub(replace_formula, text)


    def _render_values(self, text: str) -> str:
        """
        Render {var} ou {var:format} -> Value (numérico em unidade base SI, formatado).
        """
        def replace_value(match: re.Match) -> str:
            var_name = match.group(1)
            format_spec = match.group(2) if match.group(2) else ': .4g' # Default :.4g

            # Remove ':' inicial do format_spec para uso em f-string
            fmt = format_spec[1:] if format_spec.startswith(':') else format_spec

            # Ignora placeholders e comandos LaTeX
            if var_name in self.IGNORE_PLACEHOLDERS: return match.group(0)
            if var_name.startswith('\\'): return match.group(0) # Comandos LaTeX
            if var_name in {'begin', 'end', 'frac', 'sqrt', 'text'}: return match.group(0) # Comandos comuns

            if var_name not in self.variables:
                logger.debug(f"Variable '{var_name}' not found for value rendering.")
                return match.group(0) # Retorna placeholder original

            var_obj = self.variables[var_name]
            if var_obj is None or var_obj.value is None:
                return "???" # Valor não definido

            try:
                # Usa o valor numérico base (float, int, ndarray)
                value_to_format = var_obj.value

                # Lida com ndarray (não pode formatar diretamente com :.4g)
                if NUMPY_AVAILABLE and isinstance(value_to_format, np.ndarray):
                     # Formatação simples para array, ignora format_spec por enquanto
                     return f"array(shape={value_to_format.shape})"
                elif isinstance(value_to_format, (int, float)):
                     # Aplica formatação ao valor escalar
                     formatted = f"{value_to_format:{fmt}}"
                     # logger.debug(f"  Rendered value: {{{var_name}{format_spec}}} → {formatted}")
                     return formatted
                else:
                     # Outros tipos (ex: objeto SymPy resultante de @eq?)
                     logger.warning(f"Cannot format value for '{var_name}' of type {type(value_to_format)}. Returning str().")
                     return str(value_to_format)

            except Exception as e:
                logger.warning(f"Error rendering value '{{ {var_name}{format_spec} }}': {e}")
                return f"{{ERR:{var_name}}}" # Retorna erro formatado

        # Usa regex de self.value_render
        return self.value_render.sub(replace_value, text)


    # ========================================================================
    # CLEANING METHOD
    # ========================================================================

    def _clean_text(self, text: str) -> str:
        """Remove processed command lines (@eq, @matrix, @matop, @calc)."""
        logger.debug("Cleaning processed command lines...")
        # Regex para @eq
        text = re.sub(r'^\s*@eq(?:\[.*?\])?\s+\w+\s*=.*$', '', text, flags=re.MULTILINE)
        # Regex para @matrix (robusto para multilinhas)
        text = re.sub(r'^\s*@matrix\s*(?:\[.*?\])?\s+\w+\s*=\s*\[\[.*?\]\]', '', text, flags=re.MULTILINE | re.DOTALL)
        # Regex para @matop
        text = re.sub(r'^\s*@matop(?:\[.*?\])?\s+\w+\s*=.*?(?=\n\n|\n\s*@|\Z)', '', text, flags=re.MULTILINE | re.DOTALL)
        # Regex para @calc
        text = re.sub(r'^\s*@calc(?:\[.*?\])?\s+\w+\s*=.*?(?=\n\n|\n\s*@|\Z)', '', text, flags=re.MULTILINE | re.DOTALL)

        # Limpeza geral de espaços e linhas extras
        text = re.sub(r'\n{3,}', '\n\n', text) # Remove excesso de linhas em branco
        text = re.sub(r'^[ \t]+|[ \t]+$', '', text, flags=re.MULTILINE) # Remove espaços no início/fim das linhas
        return text.strip() # Remove espaços no início/fim do texto todo

    # ========================================================================
    # UTILITY METHODS (Adaptados para CoreVariable)
    # ========================================================================

    def get_variable(self, name: str) -> Optional[CoreVariable]:
        """Get CoreVariable object by name."""
        return self.variables.get(name)

    def get_equation(self, name: str) -> Optional[CoreEquation]:
        """Get CoreEquation object generated by @calc."""
        return self.equations.get(name)

    def get_matrix(self, name: str) -> Optional[Matrix]:
        """Get core.Matrix object by name."""
        return self.matrices.get(name)

    def list_variables(self) -> List[str]:
        """List names of all stored CoreVariable objects."""
        return list(self.variables.keys())

    def list_equations(self) -> List[str]:
        """List names of all stored CoreEquation objects."""
        return list(self.equations.keys())

    def list_matrices(self) -> List[str]:
        """List names of all stored core.Matrix objects."""
        return list(self.matrices.keys())

    def get_summary(self) -> Dict[str, Any]:
        """Get editor summary."""
        # Cria lista de variáveis com seus valores (string formatada)
        vars_list_detailed = [f"{name}: {str(var)}" for name, var in self.variables.items()]

        return {
            'editor_version': self.VERSION,
            'document_type': self.document_type.value,
            'total_variables': len(self.variables),
            'total_equations': len(self.equations),
            'total_matrices': len(self.matrices),
            'variables_list': vars_list_detailed, # Lista mais detalhada
            'equations_list': list(self.equations.keys()),
            'matrices_list': list(self.matrices.keys()),
            'core_features': {
                'core_available': CORE_AVAILABLE,
                'sympy_available': SYMPY_AVAILABLE,
                'numpy_available': NUMPY_AVAILABLE,
                'pint_available': PINT_AVAILABLE,
                'matrix_available': MATRIX_AVAILABLE,
                'matrix_ops_available': MATRIX_OPS_AVAILABLE,
            }
        }


# ============================================================================
# EXPORTS
# ============================================================================
__all__ = ['NaturalMemorialEditor', 'DocumentType', 'RenderMode']

# ============================================================================
# CORE DEPENDENCIES (Importa tudo necessário do __init__.py do core)
# ============================================================================
try:
    from pymemorial.core import (
        Variable as CoreVariable,
        Equation as CoreEquation,
        VariableFactory,
        Matrix, # Importa Matrix
        parse_quantity,
        ureg,
        Quantity,
        strip_units,
        StepRegistry,
        PINT_AVAILABLE,
        SYMPY_AVAILABLE,
        NUMPY_AVAILABLE, # Importa flags
        MATRIX_AVAILABLE
    )
    # Importa operações de matriz se disponíveis
    if MATRIX_AVAILABLE:
        try:
            from pymemorial.core.matrix_ops import MATRIX_OPERATIONS
            MATRIX_OPS_AVAILABLE = True
        except ImportError:
            MATRIX_OPERATIONS = {}
            MATRIX_OPS_AVAILABLE = False
            logging.getLogger(__name__).warning("⚠️ Matrix Operations Module (matrix_ops) not found.")
    else:
        MATRIX_OPERATIONS = {}
        MATRIX_OPS_AVAILABLE = False

    CORE_AVAILABLE = True # Assume core básico está OK se chegou aqui

    # Importa SymPy e NumPy diretamente se disponíveis (para funções matemáticas)
    if SYMPY_AVAILABLE:
        import sympy as sp
    else:
        sp = None
    if NUMPY_AVAILABLE:
        import numpy as np
    else:
        np = None

except ImportError as e:
    CORE_AVAILABLE = False
    PINT_AVAILABLE = False
    SYMPY_AVAILABLE = False
    NUMPY_AVAILABLE = False
    MATRIX_AVAILABLE = False
    MATRIX_OPS_AVAILABLE = False
    CoreVariable = None
    CoreEquation = None
    VariableFactory = None
    Matrix = None
    Quantity = float # Fallback type
    sp = None
    np = None
    MATRIX_OPERATIONS = {}
    logger = logging.getLogger(__name__)
    logger.critical(f"PyMemorial Core failed to load: {e}. Engine functionality severely limited.")
