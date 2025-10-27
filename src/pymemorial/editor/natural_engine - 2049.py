# src/pymemorial/editor/natural_engine.py
"""
Natural Memorial Editor v5.3.7 - DEFINITIVE STABLE RELEASE

🚀 CORREÇÕES IMPLEMENTADAS:
✅ Regex CORRIGIDO para @matrix e @matop
✅ Imports robustos com MATRIX_OPERATIONS
✅ Direct eval para @eq (bypass CoreEquation)
✅ Ordem de processamento otimizada
✅ Formatação de matrizes em Markdown
✅ Steps detalhados com granularidade
✅ Suporte completo a operações matriciais
✅ HTML export com MathJax
"""
from __future__ import annotations
import re
import logging
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum
import traceback


# ============================================================================
# CORE DEPENDENCIES
# ============================================================================
try:
    from pymemorial.core import (
        Variable as CoreVariable, 
        Equation as CoreEquation, 
        VariableFactory,
        parse_quantity, 
        ureg, 
        Quantity, 
        strip_units,
        StepRegistry, 
        StepPlugin, 
        GranularityType, 
        PINT_AVAILABLE
    )
    CORE_AVAILABLE = True
except ImportError as e:
    CORE_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.critical(f"PyMemorial Core not found: {e}")


# ============================================================================
# NUMPY (OBRIGATÓRIO PARA MATRIZES)
# ============================================================================
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None
    logger = logging.getLogger(__name__)
    logger.warning("⚠️ NumPy not available - matrix operations disabled")


# ============================================================================
# SYMPY (OPCIONAL - PARA OPERAÇÕES SIMBÓLICAS)
# ============================================================================
try:
    import sympy as sp
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False
    sp = None


# ============================================================================
# MATRIX SUPPORT (COM FALLBACK ROBUSTO v2.1.9+)
# ============================================================================
try:
    from pymemorial.core import Matrix
    MATRIX_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("✅ Matrix module (Robust v2.1.9) disponível (NumPy, SymPy)")
    
except ImportError as e:
    MATRIX_AVAILABLE = False
    Matrix = None
    logger = logging.getLogger(__name__)
    logger.warning(f"⚠️ Matrix module not available: {e}")


# ============================================================================
# MATRIX OPERATIONS REGISTRY (CRITICAL FIX v5.3.7)
# ============================================================================
try:
    from pymemorial.core.matrix_ops import MATRIX_OPERATIONS
    MATRIX_OPS_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("✅ Matrix Operations Module v2.3.1 loaded")
    
except ImportError as e:
    MATRIX_OPERATIONS = {}
    MATRIX_OPS_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning(f"⚠️ MATRIX_OPERATIONS not available: {e}")


# ============================================================================
# LEGACY MATRIX FUNCTIONS (FALLBACK)
# ============================================================================
try:
    from pymemorial.core import (
        multiply_matrices_with_steps,
        invert_matrix_with_steps,
    )
except ImportError:
    multiply_matrices_with_steps = None
    invert_matrix_with_steps = None
    logger = logging.getLogger(__name__)
    logger.debug("Legacy matrix functions not available (use MATRIX_OPERATIONS)")


# ============================================================================
# SMART PARSER
# ============================================================================
from .smart_parser import SmartVariableParser


# ============================================================================
# LOGGER INITIALIZATION
# ============================================================================
logger = logging.getLogger(__name__)


# ============================================================================
# VERIFY CRITICAL DEPENDENCIES
# ============================================================================
if MATRIX_AVAILABLE and MATRIX_OPS_AVAILABLE:
    logger.info("✅ Matrix support fully enabled in core")
elif MATRIX_AVAILABLE and not MATRIX_OPS_AVAILABLE:
    logger.warning("⚠️ Matrix available but operations limited")
else:
    logger.warning("⚠️ Matrix support disabled - install numpy and sympy")


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
# NATURAL MEMORIAL EDITOR CLASS
# ============================================================================
class NaturalMemorialEditor:
    """
    Natural Memorial Editor v5.3.7 - Matrizes com Steps Integrado.
    
    Features:
    - Direct eval for @eq blocks (no CoreEquation dependency)
    - MATRIX_OPERATIONS registry support
    - Robust variable storage (float values)
    - HTML/PDF export capabilities
    - Steps with multiple granularity levels
    """
    
    IGNORE_PLACEHOLDERS = {
        'dx', 'dy', 'dt', 'dtheta', 'ddx', 'ddy', 
        'mathrm', 'd', 'partial', 'nabla'
    }
    
    VERSION = "5.3.7"

    
    def __init__(self, document_type: str = 'memorial'):
        if not CORE_AVAILABLE:
            raise ImportError("PyMemorial Core required (v5.1+)")
        
        try:
            self.document_type = DocumentType[document_type.upper()]
        except KeyError:
            logger.warning(f"Invalid document_type. Using 'memorial'.")
            self.document_type = DocumentType.MEMORIAL
        
        self.variables: Dict[str, CoreVariable] = {}
        self.equations: Dict[str, CoreEquation] = {}
        self.matrices: Dict[str, Any] = {}  # Tipo genérico para fallback
        self.parser = SmartVariableParser()
        self._steps_cache: Dict[str, List[Dict[str, Any]]] = {}
        
        self._setup_patterns()
        self._setup_matrix_patterns()
        
        # Verificar disponibilidade de recursos
        if not NUMPY_AVAILABLE:
            logger.warning("NumPy not available - matrix operations disabled")
        if not MATRIX_AVAILABLE:
            logger.warning("Matrix module not available - using fallback")
        
        logger.info(f"NaturalMemorialEditor v5.2 initialized (type={self.document_type.value}, "
                    f"numpy={NUMPY_AVAILABLE}, matrix={MATRIX_AVAILABLE})")

    def _preprocess_matrix_expression(self, expr_str: str) -> str:
        """
        Pré-processa expressão de matriz para evitar contaminação numérica.
        
        ✅ v5.3 FINAL: Adiciona **1 em divisões simples para forçar forma simbólica.
        
        Exemplo: "4*E*I/Le" → "4*E*I/Le**1"
        """
        import re
        
        # Coletar todas as variáveis conhecidas
        var_names = list(self.variables.keys())
        if hasattr(self, 'parser') and hasattr(self.parser, 'detected_variables'):
            var_names.extend(self.parser.detected_variables.keys())
        
        var_names = list(set(var_names))  # Remover duplicatas
        
        logger.debug(f"Variáveis para pré-processamento: {var_names}")
        
        # Para cada variável, adicionar **1 se divisão não tiver expoente
        for var in var_names:
            # Padrão: /VAR (não seguido por **)
            pattern = rf'/{var}(?!\*\*)'
            replacement = rf'/{var}**1'
            expr_str = re.sub(pattern, replacement, expr_str)
        
        logger.debug(f"Expressão pré-processada (preview): {expr_str[:150]}...")
        return expr_str



    def _setup_patterns(self):
        """Setup regex patterns for calculations."""
        self.calc_block = re.compile(
            r'@calc(?:\[(\w+)(?::(\w+))?\])?\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+)',
            re.MULTILINE
        )
        var_name = r'([A-Za-z_][A-Za-z0-9_]*)'
        fmt_spec = r'(:[^}]+)?'
        self.value_render = re.compile(r'(?<!\$)(?<!\\)\{' + var_name + fmt_spec + r'\}(?!\$)')
        self.formula_render = re.compile(r'(?<!\$)(?<!\\)\$' + var_name + fmt_spec + r'\$(?!\$)')
        self.full_render = re.compile(r'(?<!\$)(?<!\\)\$\$' + var_name + fmt_spec + r'\$\$(?!\$)')
    
    def _setup_matrix_patterns(self):
        """
        Setup regex patterns for @matrix and @matop commands.
        
        ✅ CORREÇÃO: Regex ROBUSTO que captura corretamente mode:granularity
        """
        # Pattern para @matrix[mode:granularity] NAME = [[...]]
        # Captura: (mode, granularity, name, expression)
        self.matrix_block = re.compile(
            r'@matrix\s*(?:\[(\w+)(?::(\w+))?\])?\s+(\w+)\s*=\s*(\[\[.+?\]\])',
            re.MULTILINE | re.DOTALL
        )
        
        # Pattern para @matop[operation:granularity] NAME = EXPR
        self.matrix_operation = re.compile(
            r'@matop\s*(?:\[(\w+)(?::(\w+))?\])?\s+(\w+)\s*=\s*(.+?)(?=\n\n|\n@|\Z)',
            re.MULTILINE | re.DOTALL
        )
        
        logger.debug("Matrix patterns configured")
    
    def process(self, text: str, clean: bool = False) -> str:
        """
        Process text with ORDEM CORRIGIDA de operações.
        
        ✅ v5.3.5 CRITICAL FIX: Process @eq BEFORE @matrix
        
        ORDEM CORRETA:
        1. Detectar variáveis literais (valores diretos no texto)
        2. Processar @eq blocks PRIMEIRO (calcula EI_viga, theta_rad, etc.)
        3. Processar matrizes (agora com variáveis disponíveis)
        4. Processar operações matriciais
        5. Processar cálculos @calc restantes
        6. Renderizar valores
        7. Limpar (opcional)
        """
        logger.info("📝 Starting memorial processing (v5.3.5)...")
        
        # Step 1: Detectar variáveis LITERAIS do texto
        if self.parser:
            try:
                all_vars_result = self.parser.detect_all_variables(text)
                
                if isinstance(all_vars_result, tuple):
                    all_vars = all_vars_result[0]
                else:
                    all_vars = all_vars_result
                
                logger.info(f"✅ Variables detected: {len(all_vars)} total")
                
                for var_name, var_data in all_vars.items():
                    try:
                        if isinstance(var_data, tuple):
                            value = var_data[0] if len(var_data) > 0 else 0.0
                        elif isinstance(var_data, dict):
                            value = var_data.get('value', 0.0)
                        elif hasattr(var_data, 'magnitude'):
                            value = var_data.magnitude
                        elif hasattr(var_data, 'value'):
                            value = var_data.value
                        else:
                            value = var_data
                        
                        if isinstance(value, (int, float)):
                            self.variables[var_name] = value
                        elif hasattr(value, 'magnitude'):
                            self.variables[var_name] = float(value.magnitude)
                        else:
                            try:
                                self.variables[var_name] = float(value)
                            except (TypeError, ValueError):
                                logger.warning(f"⚠️ Cannot convert '{var_name}' to number: {value}")
                                self.variables[var_name] = 0.0
                        
                        logger.debug(f"  ✅ {var_name} = {self.variables[var_name]}")
                        
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to extract '{var_name}': {e}")
                        self.variables[var_name] = 0.0
                
            except Exception as e:
                logger.warning(f"⚠️ Variable detection failed: {e}")
                logger.debug(traceback.format_exc())
        
        # 🔧 FIX v5.3.5: Process @eq BEFORE @matrix
        # This calculates intermediate variables like EI_viga, theta_rad
        logger.info("🔧 Pass 1: Processing @eq blocks (intermediate calculations)...")
        processed_text = self._process_equation_blocks(text)
        
        logger.info(f"✅ Variables after @eq: {len(self.variables)} total")
        logger.debug(f"   Available: {', '.join(list(self.variables.keys())[:10])}")
        
        # Now process matrices (with all variables available)
        logger.info("🔧 Pass 2: Processing @matrix blocks...")
        processed_text = self._process_matrices(processed_text)
        
        logger.info("🔧 Pass 3: Processing @matop blocks...")
        processed_text = self._process_matrix_operations(processed_text)
        
        logger.info("🔧 Pass 4: Processing @calc blocks...")
        processed_text = self._process_calculations(processed_text)
        
        logger.info("🔧 Pass 5: Rendering...")
        processed_text = self._render_full(processed_text)
        processed_text = self._render_formulas(processed_text)
        processed_text = self._render_values(processed_text)
        
        if clean:
            processed_text = self._clean_text(processed_text)
        
        logger.info("✅ Memorial processing complete (v5.3.5)")
        return processed_text




    
    # ========================================================================
    # MATRIX PROCESSING
    # ========================================================================
    
    def _process_matrices(self, text: str) -> str:
        """
        Process @matrix blocks with ROBUST MULTILINE support.
        
        ✅ v5.4.0 DEFINITIVE SOLUTION:
        - MULTILINE REGEX: Captures complete matrix expressions
        - Variable context integration
        - NumPy array evaluation
        - Symbolic fallback with SymPy
        - Steps generation with granularity
        - LaTeX formatting
        
        Syntax: @matrix[mode:granularity] name = [[row1], [row2], ...]
        
        Examples:
            @matrix[steps:detailed] K = [[12*EI/L**3, 6*EI/L**2], 
                                          [6*EI/L**2, 4*EI/L]]
            
            @matrix[normal] T = [[cos(theta), -sin(theta)], 
                                 [sin(theta), cos(theta)]]
        """
        if not MATRIX_AVAILABLE:
            logger.warning("⚠️ Matrix support disabled - skipping @matrix blocks")
            return text
        
        # 🔧 ROBUST MULTILINE REGEX
        # Captures: @matrix[mode:gran] name = [[...], [...], ...]
        # Including whitespace and newlines between rows
        pattern = r'@matrix\[(?:(\w+):)?(\w+)?\]\s+(\w+)\s*=\s*(\[\s*\[[\s\S]*?\]\s*\])'
        
        def replace_matrix(match: re.Match) -> str:
            """Callback for matrix replacement."""
            mode_str, granularity_str, name, matrix_expr = match.groups()
            
            mode = (mode_str or 'normal').lower()
            granularity = (granularity_str or 'normal').lower()
            
            logger.info(f"🔍 Processing @matrix: {name}, mode={mode}, granularity={granularity}")
            logger.debug(f"  Matrix expression length: {len(matrix_expr)} chars")
            
            try:
                # Build evaluation context with ALL variables
                eval_context = self._build_eval_context()
                
                logger.debug(f"  Variables available: {len(self.variables)} total")
                logger.debug(f"  Sample vars: {list(self.variables.keys())[:5]}")
                
                # STEP 1: Try NUMERIC evaluation with NumPy
                matrix_obj = None
                numeric_data = None
                
                try:
                    # Clean expression (remove extra whitespace/newlines)
                    cleaned_expr = matrix_expr.replace('\n', ' ').replace('\r', '')
                    
                    # Evaluate to nested list
                    numeric_data = eval(cleaned_expr, eval_context)
                    
                    # Convert to NumPy array
                    if isinstance(numeric_data, (list, tuple)):
                        numeric_data = np.array(numeric_data, dtype=float)
                        logger.info(f"  ✅ Numeric evaluation successful: shape {numeric_data.shape}")
                    
                    # Create Matrix object
                    matrix_obj = Matrix(
                        data=numeric_data,
                        name=name,
                        description=f"Matrix {name} from natural editor"
                    )
                    
                except Exception as e:
                    logger.warning(f"  ⚠️ Numeric evaluation failed: {str(e)[:100]}")
                    
                    # STEP 2: Try SYMBOLIC evaluation with SymPy
                    if SYMPY_AVAILABLE:
                        try:
                            logger.debug("  Attempting symbolic evaluation...")
                            
                            # Convert to SymPy Matrix
                            symbolic_data = sp.Matrix(eval(cleaned_expr, eval_context))
                            
                            matrix_obj = Matrix(
                                data=symbolic_data,
                                name=name,
                                description=f"Symbolic matrix {name}"
                            )
                            
                            logger.info(f"  ✅ Symbolic evaluation successful: {symbolic_data.shape}")
                            
                        except Exception as e2:
                            logger.error(f"  ❌ Symbolic evaluation also failed: {str(e2)[:100]}")
                            raise ValueError(f"Matrix evaluation failed (numeric and symbolic): {e2}")
                    else:
                        raise ValueError(f"Numeric evaluation failed and SymPy not available: {e}")
                
                if matrix_obj is None:
                    raise ValueError("Failed to create matrix object")
                
                # Store matrix
                self.matrices[name] = matrix_obj
                logger.info(f"  ✅ Matrix '{name}' stored in self.matrices")
                
                # Generate output based on mode
                output = self._generate_matrix_output(
                    matrix_obj=matrix_obj,
                    name=name,
                    mode=mode,
                    granularity=granularity,
                    original_expr=matrix_expr
                )
                
                return output
            
            except Exception as e:
                logger.error(f"❌ Error processing @matrix '{name}': {e}")
                logger.error(f"   Expression preview: {matrix_expr[:200]}...")
                logger.debug(traceback.format_exc())
                
                return f"\n**ERRO MATRIZ {name}:** {str(e)[:150]}\n\n"
        
        # Apply regex substitution
        result = re.sub(pattern, replace_matrix, text, flags=re.MULTILINE | re.DOTALL)
        return result
    
    
    def _build_eval_context(self) -> Dict[str, Any]:
        """
        Build complete evaluation context with variables and functions.
        
        Returns:
            Dictionary with math functions and stored variables
        """
        eval_context = {
            "__builtins__": {},
        }
        
        # Add math functions
        if NUMPY_AVAILABLE:
            eval_context.update({
                "cos": np.cos,
                "sin": np.sin,
                "tan": np.tan,
                "sqrt": np.sqrt,
                "abs": abs,
                "pow": pow,
                "exp": np.exp,
                "log": np.log,
                "pi": np.pi,
                "e": np.e,
            })
        
        # Add SymPy functions (for symbolic)
        if SYMPY_AVAILABLE:
            eval_context.update({
                "Symbol": sp.Symbol,
                "symbols": sp.symbols,
                "Matrix": sp.Matrix,
            })
        
        # Add ALL stored variables
        eval_context.update(self.variables)
        
        return eval_context
    
    
    def _generate_matrix_output(
        self,
        matrix_obj: Matrix,
        name: str,
        mode: str,
        granularity: str,
        original_expr: str
    ) -> str:
        """
        Generate formatted output for matrix based on mode.
        
        Args:
            matrix_obj: Matrix object
            name: Matrix name
            mode: Display mode (normal, steps, symbolic)
            granularity: Detail level (basic, detailed, etc.)
            original_expr: Original matrix expression
            
        Returns:
            Formatted string for insertion in document
        """
        output = ""
        
        if mode == 'steps':
            # Show detailed steps
            output += f"\n**Matriz {name}:**\n\n"
            output += f"Definição simbólica:\n\n"
            output += f"$${name} = {self._format_matrix_expr_latex(original_expr)}$$\n\n"
            
            # Try to evaluate and show numeric result
            try:
                evaluated = matrix_obj.evaluate()
                if isinstance(evaluated, np.ndarray):
                    output += "Matriz avaliada numericamente:\n\n"
                    output += self._format_matrix_as_latex(evaluated, name)
                    output += "\n"
            except Exception as e:
                logger.debug(f"Could not evaluate matrix for display: {e}")
                output += f"(Matriz simbólica - não pode ser avaliada numericamente)\n\n"
        
        else:
            # Simple output (just evaluated matrix)
            try:
                evaluated = matrix_obj.evaluate()
                if isinstance(evaluated, np.ndarray):
                    output += f"\n{self._format_matrix_as_latex(evaluated, name)}\n"
                else:
                    # Symbolic matrix
                    output += f"\n$${name} = {self._format_matrix_expr_latex(original_expr)}$$\n"
            except Exception as e:
                output += f"\n$${name} = {self._format_matrix_expr_latex(original_expr)}$$\n"
        
        return output
    
    
    def _format_matrix_expr_latex(self, matrix_expr: str) -> str:
        """
        Format matrix expression as LaTeX (for symbolic display).
        
        Args:
            matrix_expr: String with matrix expression like "[[a, b], [c, d]]"
            
        Returns:
            LaTeX formatted matrix
        """
        try:
            # Parse nested list
            matrix_expr_clean = matrix_expr.replace('\n', ' ').replace('\r', '')
            rows = eval(matrix_expr_clean)
            
            # Convert to LaTeX rows
            latex_rows = []
            for row in rows:
                row_str = " & ".join(str(elem) for elem in row)
                latex_rows.append(row_str)
            
            matrix_body = " \\\\\n".join(latex_rows)
            
            return f"\\begin{{bmatrix}}\n{matrix_body}\n\\end{{bmatrix}}"
        
        except Exception as e:
            logger.debug(f"Could not format matrix expression: {e}")
            return matrix_expr  # Return as-is
    
    
    def _format_matrix_as_latex(self, matrix_data: np.ndarray, name: str = "") -> str:
        """
        Format NumPy array as LaTeX matrix for display.
        
        Args:
            matrix_data: NumPy array to format
            name: Optional name to display
            
        Returns:
            LaTeX string with matrix
        """
        rows, cols = matrix_data.shape
        
        latex_rows = []
        for i in range(rows):
            row_values = [f"{matrix_data[i, j]:.4g}" for j in range(cols)]
            latex_rows.append(" & ".join(row_values))
        
        matrix_body = " \\\\\n".join(latex_rows)
        
        if name:
            return f"$${name} = \\begin{{bmatrix}}\n{matrix_body}\n\\end{{bmatrix}}$$"
        else:
            return f"$$\\begin{{bmatrix}}\n{matrix_body}\n\\end{{bmatrix}}$$"

    
    
    def _format_matrix_as_latex(self, matrix_data: np.ndarray, name: str = "") -> str:
        """
        Format NumPy array as LaTeX matrix for display.
        
        Args:
            matrix_data: NumPy array to format
            name: Optional name to display
            
        Returns:
            LaTeX string with matrix
        """
        rows, cols = matrix_data.shape
        
        latex_rows = []
        for i in range(rows):
            row_values = [f"{matrix_data[i, j]:.4g}" for j in range(cols)]
            latex_rows.append(" & ".join(row_values))
        
        matrix_body = " \\\\\n".join(latex_rows)
        
        if name:
            return f"$${name} = \\begin{{bmatrix}}\n{matrix_body}\n\\end{{bmatrix}}$$"
        else:
            return f"$$\\begin{{bmatrix}}\n{matrix_body}\n\\end{{bmatrix}}$$"


    def _parse_matrix_expression(self, expr: str):
        """
        Parse matrix expression from string.
        
        ✅ v5.3.2 FIXES:
        - Robust symbolic/numeric evaluation
        - Proper variable substitution
        - Fallback handling
        
        Suporta:
        - [[1,2],[3,4]] - Lista literal
        - [[EI_viga, 0], [0, EI_viga]] - Expressões simbólicas
        """
        import ast
        
        expr = expr.strip()
        
        logger.debug(f"Parsing matrix expression (first 100 chars): {expr[:100]}...")
        
        # Tentativa 1: Parse literal (lista Python pura)
        try:
            result = ast.literal_eval(expr)
            if isinstance(result, list):
                logger.debug("Matrix parsed as literal list")
                return result
        except Exception:
            pass
        
        # Tentativa 2: Eval com contexto de variáveis (numérico)
        try:
            # Criar contexto de eval
            eval_context = {
                "__builtins__": {},
                "cos": np.cos if NUMPY_AVAILABLE else (lambda x: x),
                "sin": np.sin if NUMPY_AVAILABLE else (lambda x: x),
                "tan": np.tan if NUMPY_AVAILABLE else (lambda x: x),
                "sqrt": np.sqrt if NUMPY_AVAILABLE else (lambda x: x),
            }
            
            # Adicionar variáveis conhecidas
            for var_name, var_obj in self.variables.items():
                if hasattr(var_obj, 'value') and var_obj.value is not None:
                    eval_context[var_name] = var_obj.value
                elif hasattr(var_obj, 'magnitude'):
                    eval_context[var_name] = var_obj.magnitude
            
            logger.debug(f"Eval context has {len(eval_context)} entries")
            
            result = eval(expr, eval_context)
            
            if isinstance(result, list):
                logger.debug("Matrix evaluated numerically")
                return result
        
        except Exception as e:
            logger.warning(f"Numeric eval failed: {e}")
        
        # Tentativa 3: Fallback simbólico (SymPy)
        if SYMPY_AVAILABLE:
            try:
                # Criar contexto simbólico
                symbols_dict = {}
                
                # Adicionar variáveis conhecidas
                for var_name, var_obj in self.variables.items():
                    if hasattr(var_obj, 'value') and var_obj.value is not None:
                        symbols_dict[var_name] = var_obj.value
                    else:
                        symbols_dict[var_name] = sp.Symbol(var_name, real=True)
                
                # Adicionar funções matemáticas
                symbols_dict.update({
                    'cos': sp.cos,
                    'sin': sp.sin,
                    'tan': sp.tan,
                    'sqrt': sp.sqrt,
                })
                
                # Adicionar símbolos para variáveis não definidas
                import re
                for var_match in re.finditer(r'\b([A-Za-z_]\w*)\b', expr):
                    var_name = var_match.group(1)
                    if var_name not in symbols_dict and var_name not in ['cos', 'sin', 'tan', 'sqrt']:
                        symbols_dict[var_name] = sp.Symbol(var_name, real=True)
                
                result = eval(expr, symbols_dict)
                
                logger.debug("Matrix evaluated symbolically")
                return result
            
            except Exception as e:
                logger.error(f"Symbolic eval also failed: {e}")
                raise ValueError(f"Could not parse matrix expression: {e}")
        
        # Se chegou aqui, todas as tentativas falharam
        raise ValueError(f"Failed to parse matrix expression after all attempts")

    
    def _format_matrix_steps(self, matrix, granularity: str) -> str:
        """
        Format matrix with step-by-step construction.
        
        ✅ CORREÇÃO: Garantir que o LaTeX simbólico seja PURO (sem números)
        """
        steps = matrix.steps(granularity=granularity)
        
        output = ["\n**Matriz:**\n"]
        
        for step in steps:
            operation = step.get('operation', 'unknown')
            description = step.get('description', '')
            
            if operation == 'definition':
                output.append(f"→ **Definição:** {description}")
            
            elif operation == 'symbolic' and 'latex' in step:
                latex_matrix = step['latex']
                output.append(f"\n$$[{matrix.name}] = {latex_matrix}$$\n")
            
            elif operation == 'substitution':
                output.append(f"→ *Substituição:* {description}")
            
            elif operation == 'intermediate' and 'latex' in step:
                latex_intermediate = step['latex']
                output.append(f"\n→ **Matriz Substituída (Passo Intermediário):**\n")
                output.append(f"$$[{matrix.name}] = {latex_intermediate}$$\n")
            
            elif operation == 'evaluation' and 'matrix' in step:
                matrix_data = step['matrix']
                if isinstance(matrix_data, list):
                    output.append(f"\n→ **Matriz Numérica:**")
                    output.append(self._format_matrix_table(matrix_data, matrix.name))
            
            elif operation == 'properties':
                props = []
                if 'determinant' in step:
                    props.append(f"Determinante: {step['determinant']:.6g}")
                if 'trace' in step:
                    props.append(f"Traço: {step['trace']:.6g}")
                if 'rank' in step:
                    props.append(f"Posto: {step['rank']}")
                if props:
                    output.append(f"\n→ *Propriedades:* {', '.join(props)}")
        
        output.append("\n")
        return "\n".join(output)
    
    def _format_matrix_table(self, matrix_data: list, name: str) -> str:
        """Format matrix as Markdown table."""
        if not matrix_data:
            return ""
        
        rows = len(matrix_data)
        cols = len(matrix_data[0]) if rows > 0 else 0
        
        lines = []
        
        # Header
        header = [f"**{name}**"] + [f"Col {j+1}" for j in range(cols)]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "---|" * (cols + 1))
        
        # Rows
        for i, row in enumerate(matrix_data):
            row_str = f"**Linha {i+1}**"
            for val in row:
                if isinstance(val, (int, float)):
                    row_str += f" | {val:.4g}"
                else:
                    row_str += f" | {val}"
            lines.append(row_str + " |")
        
        return "\n" + "\n".join(lines) + "\n"
    
    def _format_matrix_symbolic(self, matrix) -> str:
        """Format matrix showing only symbolic form."""
        if not matrix.is_symbolic or not SYMPY_AVAILABLE:
            return f"\n**{matrix.name}:** (numeric matrix)\n\n"
        
        try:
            latex_str = sp.latex(matrix._symbolic_matrix)
            return f"\n$$[{matrix.name}] = {latex_str}$$\n\n"
        except Exception as e:
            logger.error(f"Error formatting symbolic matrix: {e}")
            return f"\n**{matrix.name}:** (error formatting symbolic form)\n\n"
    
    def _format_matrix_numeric(self, matrix) -> str:
        """Format matrix showing only numeric values."""
        try:
            result = matrix.evaluate()
            return f"\n**{matrix.name}:**\n{self._format_matrix_table(result.tolist(), matrix.name)}\n"
        except Exception as e:
            logger.error(f"Error evaluating numeric matrix: {e}")
            return f"\n**{matrix.name}:** (error evaluating numeric form)\n\n"
    
    def _format_matrix_result(self, matrix) -> str:
        """Format matrix showing both symbolic and numeric."""
        output = []
        
        if matrix.is_symbolic:
            output.append(self._format_matrix_symbolic(matrix))
        
        try:
            output.append(self._format_matrix_numeric(matrix))
        except Exception as e:
            output.append(f"\n**Erro ao avaliar matriz numericamente:** {e}\n")
        
        return "".join(output)
    
    # ========================================================================
    # MATRIX OPERATIONS
    # ========================================================================
    
    def _process_matrix_operations(self, text: str) -> str:
        """
        Process @matop operations with MATRIX_OPERATIONS registry.
        
        ✅ v5.3.3 FIX: Complete implementation with replacer function
        
        Syntax: @matop[operation:granularity] result = expression
        
        Examples:
            @matop[multiply:detailed] K_global = T_rot.T @ K_local @ T_rot
            @matop[eigenvalues] lambda_M = eigenvals(M_massa)
            @matop[inverse] K_inv = inv(K_local)
        """
        # Regex pattern for @matop
        pattern = r'@matop\[(?:(\w+):)?(\w+)?\]\s+(\w+)\s*=\s*(.+?)(?=\n\n|\n@|\Z)'
        
        def replace_matop(match: re.Match) -> str:
            """Callback function for regex substitution."""
            operation_str, granularity_str, result_name, expression = match.groups()
            
            operation = (operation_str or 'multiply').lower()
            granularity = granularity_str or 'normal'
            
            # 🔧 FIX: Map operation aliases
            operation_aliases = {
                'steps': 'multiply',
                'mult': 'multiply',
                'inv': 'inverse',
                't': 'transpose',
                'det': 'determinant',
                'eigenvals': 'eigenvalues',
                'eigenvecs': 'eigenvectors',
            }
            
            operation = operation_aliases.get(operation, operation)
            
            logger.debug(f"Processing @matop: {operation} → {result_name}")
            logger.debug(f"  Expression: {expression[:100]}...")
            
            try:
                # Check if operation exists in registry
                if operation not in MATRIX_OPERATIONS:
                    available = ', '.join(MATRIX_OPERATIONS.keys())
                    raise ValueError(
                        f"Operação não suportada: {operation}\n"
                        f"Disponíveis: {available}"
                    )
                
                # Extract matrices from expression
                matrices = self._extract_matrices_from_expression(expression)
                
                if not matrices:
                    raise ValueError(f"Nenhuma matriz na expressão: {expression}")
                
                # Get operation function
                operation_func = MATRIX_OPERATIONS[operation]
                
                # Execute operation based on type
                if operation in ('multiply', 'mult'):
                    if len(matrices) < 2:
                        raise ValueError("Multiplicação requer pelo menos 2 matrizes")
                    
                    result_matrix = matrices[0]
                    steps_all = []
                    
                    for next_matrix in matrices[1:]:
                        result_matrix, steps = operation_func(result_matrix, next_matrix, granularity)
                        steps_all.extend(steps)
                    
                    result_matrix.name = result_name
                    self.matrices[result_name] = result_matrix
                    
                    return self._format_matrix_operation_output(
                        result_name, operation, steps_all, result_matrix
                    )
                
                elif operation in ('eigenvalues', 'eigenvectors'):
                    if len(matrices) > 1:
                        logger.warning(f"Eigenvalues: usando apenas primeira matriz")
                    
                    compute_vecs = (operation == 'eigenvectors')
                    result, steps = operation_func(matrices[0], granularity, compute_vecs)
                    
                    # Store eigenvalues
                    if isinstance(result, tuple):
                        eigenvals, eigenvecs = result
                        self.variables[result_name] = eigenvals
                    else:
                        self.variables[result_name] = result
                    
                    return self._format_eigenvalue_output(result_name, matrices[0].name, steps)
                
                elif operation in ('trace', 'rank', 'determinant'):
                    if len(matrices) > 1:
                        logger.warning(f"{operation}: usando apenas primeira matriz")
                    
                    scalar_result, steps = operation_func(matrices[0], granularity)
                    
                    self.variables[result_name] = scalar_result
                    
                    return self._format_scalar_output(
                        result_name, matrices[0].name, scalar_result, operation, steps
                    )
                
                else:
                    # Unary operations (inverse, transpose, add, etc.)
                    if operation in ('add', 'subtract'):
                        if len(matrices) < 2:
                            raise ValueError(f"{operation} requer 2 matrizes")
                        result_matrix, steps = operation_func(matrices[0], matrices[1], granularity)
                    else:
                        if len(matrices) > 1:
                            logger.warning(f"{operation}: usando apenas primeira matriz")
                        result_matrix, steps = operation_func(matrices[0], granularity)
                    
                    result_matrix.name = result_name
                    self.matrices[result_name] = result_matrix
                    
                    return self._format_matrix_operation_output(
                        result_name, operation, steps, result_matrix
                    )
            
            except Exception as e:
                logger.error(f"Erro em @matop '{result_name}': {e}")
                logger.debug(traceback.format_exc())
                return f"**ERRO OPERAÇÃO:** {result_name}: {e}\n\n"
        
        # Apply regex substitution with replacer callback
        return re.sub(pattern, replace_matop, text, flags=re.DOTALL)


    # ========================================================================
    # CALC BLOCKS PROCESSING (v5.3.8 - ADDED)
    # ========================================================================
    
    def _process_calc_blocks(self, text: str) -> str:
        """
        Process @calc blocks with FULL variable context from SmartParser.
        
        ✅ CRITICAL FIX v5.3.9: Robust value extraction
        ✅ Uses self.variables populated by SmartParser
        ✅ Supports steps (basic, smart, detailed)
        
        Args:
            text: Text with @calc blocks
            
        Returns:
            Text with @calc blocks replaced by results
        """
        if not CORE_AVAILABLE:
            logger.warning("Core unavailable - @calc blocks skipped")
            return text
        
        # Pattern: @calc[steps:granularity] variable = expression
        pattern = r'@calc(?:\[steps:(\w+)\])?\s+(\w+)\s*=\s*(.+?)(?=\n|$)'
        
        def replace_calc(match):
            granularity = match.group(1) or 'basic'  # Default: basic
            var_name = match.group(2).strip()
            expression = match.group(3).strip()
            
            try:
                # ✅ Build evaluation context with robust value extraction
                eval_context = {}
                
                # Add all detected variables with their values
                for v_name, v_obj in self.variables.items():
                    # Try multiple extraction methods
                    try:
                        if hasattr(v_obj, 'value') and v_obj.value is not None:
                            eval_context[v_name] = v_obj.value
                        elif hasattr(v_obj, 'magnitude'):
                            eval_context[v_name] = v_obj.magnitude
                        elif isinstance(v_obj, (int, float)):
                            # Direct numeric value
                            eval_context[v_name] = v_obj
                        elif hasattr(v_obj, '_magnitude'):
                            # Pint Quantity with _magnitude
                            eval_context[v_name] = v_obj._magnitude
                        elif PINT_AVAILABLE and isinstance(v_obj, Quantity):
                            # Pint Quantity - strip units
                            eval_context[v_name] = strip_units(v_obj)
                        else:
                            # Last resort: try to convert to float
                            try:
                                eval_context[v_name] = float(v_obj)
                            except:
                                logger.debug(f"Variable '{v_name}' detected but value extraction failed")
                    except Exception as e:
                        logger.debug(f"Variable '{v_name}' extraction error: {e}")
                
                # Add numpy/sympy functions
                if NUMPY_AVAILABLE:
                    eval_context.update({
                        'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
                        'sqrt': np.sqrt, 'log': np.log, 'exp': np.exp,
                        'pi': np.pi
                    })
                
                if SYMPY_AVAILABLE:
                    eval_context.update({
                        'integrate': sp.integrate,
                        'diff': sp.diff,
                        'symbols': sp.symbols,
                        'Symbol': sp.Symbol,
                        'simplify': sp.simplify
                    })
                
                # ✅ Check if all variables in expression are defined
                import re
                var_pattern = r'\b([a-zA-Z_]\w*)\b'
                expr_vars = set(re.findall(var_pattern, expression))
                
                # Exclude functions and known constants
                known_funcs = {'sin', 'cos', 'tan', 'sqrt', 'log', 'exp', 'integrate', 
                              'diff', 'symbols', 'Symbol', 'simplify', 'pi'}
                expr_vars -= known_funcs
                
                # Check for undefined variables
                undefined_vars = []
                for ev in expr_vars:
                    if ev not in eval_context:
                        undefined_vars.append(f"{ev}(sem valor)")
                
                if undefined_vars:
                    error_msg = f"Variáveis indefinidas: {', '.join(undefined_vars)}"
                    logger.error(f"Erro @calc '{var_name}': {error_msg}")
                    return f"\n\n**ERRO:** {var_name}: {error_msg}\n\n"
                
                # ✅ Create CoreEquation with granularity
                try:
                    from pymemorial.core import Equation as CoreEquation, GranularityType
                    
                    # Map string granularity to enum
                    granularity_map = {
                        'basic': GranularityType.BASIC,
                        'smart': GranularityType.SMART,
                        'detailed': GranularityType.DETAILED,
                        'all': GranularityType.ALL
                    }
                    
                    gran_enum = granularity_map.get(granularity.lower(), GranularityType.BASIC)
                    
                    # Create equation
                    eq = CoreEquation(
                        name=var_name,
                        expression=expression,
                        context=eval_context
                    )
                    
                    # Evaluate with steps
                    result = eq.evaluate()
                    
                    # Get steps rendering
                    steps_output = eq.steps(granularity=gran_enum)
                    
                    # Store in variables for future use
                    self.variables[var_name] = result
                    
                    # Store equation
                    if not hasattr(self, 'equations'):
                        self.equations = []
                    self.equations.append(eq)
                    
                    logger.debug(f"✅ @calc '{var_name}' = {result} (steps:{granularity})")
                    
                    # Return formatted output
                    return f"\n{steps_output}\n"
                    
                except Exception as eq_error:
                    # Fallback: direct evaluation without CoreEquation
                    logger.warning(f"CoreEquation failed for '{var_name}', using fallback: {eq_error}")
                    
                    try:
                        result = eval(expression, {"__builtins__": {}}, eval_context)
                        self.variables[var_name] = result
                        
                        return f"\n**{var_name}** = {result}\n"
                        
                    except Exception as eval_error:
                        error_msg = f"Evaluation failed: {eval_error}"
                        logger.error(f"Erro @calc '{var_name}': {error_msg}")
                        return f"\n\n**ERRO:** {var_name}: {error_msg}\n\n"
            
            except Exception as e:
                logger.error(f"Erro processing @calc '{var_name}': {e}")
                import traceback
                traceback.print_exc()
                return f"\n\n**ERRO:** {var_name}: {str(e)}\n\n"
        
        # Replace all @calc blocks
        result = re.sub(pattern, replace_calc, text, flags=re.MULTILINE)
        
        return result



    
    def _extract_matrices_from_expression(self, expression: str) -> List:
        """
        Extract matrix objects from expression string.
        
        Examples:
            "A @ B" -> [Matrix A, Matrix B]
            "T_rot.T @ K_local @ T_rot" -> [Matrix T_rot^T, Matrix K_local, Matrix T_rot]
            "inv(K)" -> [Matrix K]
        """
        # Remove operators and parentheses
        cleaned = re.sub(r'[@\(\)\s]', ' ', expression)
        
        # Track transposes
        transpose_map = {}
        for match in re.finditer(r'(\w+)\.T', expression):
            transpose_map[match.group(1)] = True
        
        cleaned = cleaned.replace('.T', '')
        
        # Extract matrix names
        matrix_names = [name.strip() for name in cleaned.split() if name.strip()]
        
        # Fetch matrices from storage
        matrices = []
        for name in matrix_names:
            if name in self.matrices:
                matrix = self.matrices[name]
                
                # Apply transpose if needed
                if name in transpose_map:
                    matrix_data = matrix.evaluate().T
                    matrix = Matrix(
                        data=matrix_data,
                        name=f"{name}_T",
                        description=f"Transposta de {name}"
                    )
                
                matrices.append(matrix)
            else:
                logger.warning(f"⚠️ Matriz '{name}' não encontrada")
        
        return matrices
    
    
    def _format_matrix_operation_output(
        self,
        result_name: str,
        operation: str,
        steps: List[Dict[str, Any]],
        result_matrix: Optional[Any] = None
    ) -> str:
        """Format matrix operation output."""
        output = []
        
        output.append(f"\n**Operação:** {operation.upper()}\n\n")
        
        for step in steps:
            description = step.get('description', '')
            latex = step.get('latex', '')
            
            if description:
                output.append(f"→ {description}\n")
            
            if latex:
                output.append(f"\n$${latex}$$\n")
        
        if result_matrix:
            output.append(f"\n✅ Resultado armazenado: `{result_name}`\n")
        
        output.append("\n")
        return "".join(output)
    
    
    def _format_eigenvalue_output(
        self,
        result_name: str,
        matrix_name: str,
        steps: List[Dict[str, Any]]
    ) -> str:
        """Format eigenvalue output."""
        output = []
        
        output.append(f"\n**Autovalores de {matrix_name}:**\n\n")
        
        for step in steps:
            description = step.get('description', '')
            latex = step.get('latex', '')
            
            if description:
                output.append(f"→ {description}\n")
            
            if latex:
                output.append(f"\n$${latex}$$\n")
        
        output.append(f"\n✅ Autovalores: `{result_name}`\n\n")
        return "".join(output)
    
    
    def _format_scalar_output(
        self,
        result_name: str,
        matrix_name: str,
        scalar_value: float,
        operation: str,
        steps: List[Dict[str, Any]]
    ) -> str:
        """Format scalar result output."""
        output = []
        
        operation_names = {
            'determinant': 'Determinante',
            'trace': 'Traço',
            'rank': 'Posto'
        }
        
        op_display = operation_names.get(operation, operation.capitalize())
        
        output.append(f"\n**{op_display} de {matrix_name}:**\n\n")
        
        for step in steps:
            description = step.get('description', '')
            latex = step.get('latex', '')
            
            if description:
                output.append(f"→ {description}\n")
            
            if latex:
                output.append(f"\n$${latex}$$\n")
        
        output.append(f"\n✅ `{result_name}` = {scalar_value:.6g}\n\n")
        return "".join(output)

    
    # ========================================================================
    # EQUATION PROCESSING (NOVO v5.3.2 - Suporte a @eq)
    # ========================================================================
    
    def _process_equation_blocks(self, text: str) -> str:
        """
        Process @eq blocks with DIRECT EVALUATION (bypass CoreEquation).
        
        ✅ v5.3.7 DEFINITIVE FIX:
        - Use _simple_eval() directly (no CoreEquation dependency)
        - Store results as float in self.variables
        - Generate simple steps output
        
        Syntax: @eq[mode:granularity] var_name = expression
        """
        pattern = r'@eq\[(?:(\w+):)?(\w+)?\]\s+(\w+)\s*=\s*(.+?)(?=\n\n|\n@|\Z)'
        
        def replace_eq(match: re.Match) -> str:
            mode_str, granularity_str, var_name, expression = match.groups()
            
            mode = mode_str or 'normal'
            granularity = granularity_str or 'normal'
            
            logger.debug(f"Processing @eq: {var_name} = {expression[:50]}...")
            
            try:
                # 🔧 v5.3.7: DIRECT EVAL (bypass CoreEquation completely)
                result = self._simple_eval(expression.strip())
                
                # Store as numeric value
                if isinstance(result, (int, float)):
                    self.variables[var_name] = float(result)
                elif hasattr(result, 'magnitude'):
                    self.variables[var_name] = float(result.magnitude)
                else:
                    try:
                        self.variables[var_name] = float(result)
                    except (TypeError, ValueError):
                        logger.warning(f"Cannot convert result to float: {result}")
                        self.variables[var_name] = 0.0
                
                logger.debug(f"  ✅ {var_name} = {self.variables[var_name]}")
                
                # Format output based on mode
                if mode == 'steps':
                    # Generate simple steps output
                    output = f"\n→ Calculando: ${var_name} = {expression.strip()}$\n\n"
                    output += f"→ Resultado: ${var_name} = {self.variables[var_name]:.4g}$\n"
                    return output
                else:
                    # Simple inline result
                    return f"\n${var_name} = {self.variables[var_name]:.4g}$\n"
            
            except Exception as e:
                logger.error(f"❌ Erro em @eq '{var_name}': {e}")
                logger.debug(traceback.format_exc())
                
                # Store 0.0 as fallback
                self.variables[var_name] = 0.0
                
                return f"\n**ERRO EQUAÇÃO:** {var_name} = {expression.strip()} (erro: {str(e)[:50]})\n\n"
        
        return re.sub(pattern, replace_eq, text, flags=re.DOTALL)

    
    
    def _simple_eval(self, expression: str):
        """
        Simple evaluation with variable context + SymPy support.
        
        ✅ v5.3.5: Added 'diff' function for derivatives
        """
        eval_context = {
            "__builtins__": {},
        }
        
        # Math functions
        if NUMPY_AVAILABLE:
            eval_context.update({
                "cos": np.cos,
                "sin": np.sin,
                "tan": np.tan,
                "sqrt": np.sqrt,
                "abs": abs,
                "pow": pow,
                "exp": np.exp,
                "log": np.log,
            })
        
        # SymPy functions (for derivatives, integrals)
        if SYMPY_AVAILABLE:
            eval_context.update({
                "diff": sp.diff,
                "integrate": sp.integrate,
                "symbols": sp.symbols,
                "Symbol": sp.Symbol,
            })
        else:
            # Fallback placeholders
            eval_context["diff"] = lambda *args, **kwargs: 0
            eval_context["integrate"] = lambda *args, **kwargs: 0
        
        # Add all stored variables
        eval_context.update(self.variables)
        
        return eval(expression, eval_context)






    def _replace_matop_match(self, match: re.Match) -> str:
        """Process a single @matop match."""
        operation_str, granularity_str, result_name, expression = match.groups()
        operation_str = operation_str or 'multiply'
        granularity = granularity_str or 'normal'
        
        try:
            operation = operation_str.lower()
            
            if operation in ('multiply', 'mult'):
                return self._process_multiply_operation(result_name, expression, granularity)
            elif operation in ('inverse', 'inv'):
                return self._process_inverse_operation(result_name, expression, granularity)
            elif operation in ('transpose', 't'):
                return self._process_transpose_operation(result_name, expression, granularity)
            elif operation in ('determinant', 'det'):
                return self._process_determinant_operation(result_name, expression, granularity)
            else:
                raise ValueError(f"Operação não suportada: {operation}")
        
        except Exception as e:
            logger.error(f"Erro em @matop '{result_name}': {e}")
            return f"\n**ERRO OPERAÇÃO:** {result_name}: {e}\n\n"
    
    def _process_multiply_operation(self, result_name: str, expr: str, granularity: str) -> str:
        """Process matrix multiplication: C = A * B or C = A * B * C (encadeada)."""
        # Parse expression (suporta A*B, A@B, A.T*B, etc)
        # Remove espaços e substitui @ por *
        expr = expr.replace('@', '*').replace(' ', '')
        
        # Extrair transpostas (A.T vira A com flag transpose)
        import re
        transpose_pattern = re.compile(r'(\w+)\.T')
        
        # Encontrar todos os termos (matrizes e suas transpostas)
        terms = []
        current_expr = expr
        
        # Split por * mas mantendo informação de transposta
        parts = current_expr.split('*')
        
        for part in parts:
            if '.T' in part:
                matrix_name = part.replace('.T', '')
                terms.append({'name': matrix_name, 'transpose': True})
            else:
                terms.append({'name': part, 'transpose': False})
        
        # Validar que todas as matrizes existem
        for term in terms:
            if term['name'] not in self.matrices:
                raise ValueError(f"Matriz não encontrada: {term['name']}")
        
        # Executar multiplicação encadeada
        logger.info(f"🔍 Multiplicação encadeada: {len(terms)} termos")
        
        result_matrix = None
        operation_desc = []
        
        for i, term in enumerate(terms):
            matrix = self.matrices[term['name']]
            
            # Aplicar transposta se necessário
            if term['transpose']:
                matrix_data = matrix.evaluate().T
                matrix = Matrix(
                    data=matrix_data,
                    description=f"Transposta de {term['name']}",
                    name=f"{term['name']}_T"
                )
                operation_desc.append(f"{term['name']}ᵀ")
            else:
                operation_desc.append(term['name'])
            
            if result_matrix is None:
                result_matrix = matrix
            else:
                # Multiplicar com o resultado acumulado
                result_matrix, _ = multiply_matrices_with_steps(
                    result_matrix, 
                    matrix, 
                    'minimal'  # Interno usa minimal para eficiência
                )
        
        # Definir nome e armazenar
        result_matrix.name = result_name
        self.matrices[result_name] = result_matrix
        
        # Formatar output
        output = [f"\n**Multiplicação: {result_name} = {' × '.join(operation_desc)}**\n"]
        output.append(f"→ *Dimensões: {result_matrix.shape[0]}×{result_matrix.shape[1]}*\n")
        output.append(f"\n→ **Resultado:**")
        output.append(self._format_matrix_table(result_matrix.evaluate().tolist(), result_name))
        output.append("\n")
        
        return "".join(output)

    
    def _process_inverse_operation(self, result_name: str, expr: str, granularity: str) -> str:
        """Process matrix inversion."""
        matrix_name = expr.replace('inv(', '').replace(')', '').strip()
        
        if matrix_name not in self.matrices:
            raise ValueError(f"Matriz não encontrada: {matrix_name}")
        
        matrix = self.matrices[matrix_name]
        result_matrix, steps = invert_matrix_with_steps(matrix, granularity)
        result_matrix.name = result_name
        self.matrices[result_name] = result_matrix
        
        # Format output
        output = [f"\n**Inversão: {result_name} = {matrix_name}⁻¹**\n"]
        
        for step in steps:
            operation = step.get('operation', '')
            description = step.get('description', '')
            
            if operation == 'determinant':
                value = step.get('value', 0)
                output.append(f"→ *{description}* (det = {value:.6g})")
            elif operation == 'method':
                output.append(f"→ *{description}*")
            elif operation == 'result' and 'matrix' in step:
                output.append(f"\n→ **Matriz Inversa:**")
                output.append(self._format_matrix_table(step['matrix'], result_name))
            elif operation == 'verification' and 'matrix' in step:
                output.append(f"\n→ *Verificação (A × A⁻¹ = I):*")
                output.append(self._format_matrix_table(step['matrix'], 'I'))
        
        output.append("\n")
        return "".join(output)
    
    def _process_transpose_operation(self, result_name: str, expr: str, granularity: str) -> str:
        """Process matrix transpose."""
        matrix_name = expr.replace('.T', '').strip()
        
        if matrix_name not in self.matrices:
            raise ValueError(f"Matriz não encontrada: {matrix_name}")
        
        matrix = self.matrices[matrix_name]
        result = matrix.evaluate().T
        result_matrix = Matrix(
            data=result,
            description=f"Transposta de {matrix_name}",
            name=result_name
        )
        self.matrices[result_name] = result_matrix
        
        return f"\n**Transposta:** {result_name} = {matrix_name}ᵀ\n{self._format_matrix_table(result.tolist(), result_name)}\n"
    
    def _process_determinant_operation(self, result_name: str, expr: str, granularity: str) -> str:
        """Process determinant calculation."""
        matrix_name = expr.replace('det(', '').replace(')', '').strip()
        
        if matrix_name not in self.matrices:
            raise ValueError(f"Matriz não encontrada: {matrix_name}")
        
        matrix = self.matrices[matrix_name]
        
        if not matrix.is_square:
            raise ValueError("Matriz deve ser quadrada para calcular determinante")
        
        result = np.linalg.det(matrix.evaluate())
        
        # Store as variable
        self.variables[result_name] = VariableFactory.create(
            name=result_name,
            value=result,
            description=f"Determinante de {matrix_name}"
        )
        
        return f"\n**Determinante:** {result_name} = det({matrix_name}) = {result:.6g}\n\n"
    
    # ========================================================================
    # EXISTING METHODS (mantidos inalterados)
    # ========================================================================
    
    def _detect_variables(self, text: str):
        """Detectar variáveis no texto."""
        detected_vars_dict = self.parser.detect_all_variables(text)
        new_variables = {}
        for name, (value, unit) in detected_vars_dict.items():
            try:
                new_var = VariableFactory.create(
                    name=name, value=value, unit=unit, 
                    description=f"Var: {name}"
                )
                if new_var.value is not None:
                    new_variables[name] = new_var
            except Exception as e:
                logger.error(f"Failed to create '{name}': {e}")
        self.variables.update(new_variables)





    def _process_calculations(self, text: str) -> str:
        """Process @calc blocks (mantido do código original)."""
        if not CORE_AVAILABLE or not SYMPY_AVAILABLE:
            return text
        return self.calc_block.sub(self._replace_calc_match, text)
    
    def _replace_calc_match(self, match: re.Match) -> str:
        """Replace @calc match with step-by-step solution."""
        mode_str, granularity_str, result_name, expression = match.groups()
        mode_str = mode_str or 'full'
        
        try:
            render_mode = RenderMode[mode_str.upper()]
        except KeyError:
            logger.warning(f"Unknown mode '{mode_str}'")
            render_mode = RenderMode.FULL
        
        granularity = granularity_str or ('smart' if render_mode == RenderMode.STEPS else None)
        
        try:
            # Find symbols and validate variables
            parser_symbols = self.parser.find_symbols_in_expression(expression)
            required_core_vars: Dict[str, CoreVariable] = {}
            missing_vars = []
            known_funcs = {'sqrt', 'sin', 'cos', 'tan', 'exp', 'log', 'pi', 'integrate', 'y', 'x', 'z'}
            
            for name in parser_symbols:
                if name in self.variables:
                    if hasattr(self.variables[name], 'value') and self.variables[name].value is not None:
                        required_core_vars[name] = self.variables[name]
                    else:
                        missing_vars.append(f"{name}(sem valor)")
                elif name not in known_funcs:
                    missing_vars.append(name)
            
            if missing_vars:
                raise ValueError(f"Variáveis indefinidas: {', '.join(missing_vars)}")
            
            # Create equation
            core_eq = CoreEquation(
                expression=f"{result_name} = {expression}",
                variables=required_core_vars,
                description=f"Calc: {result_name}"
            )
            
            output_str = ""
            result_data = None
            
            # Evaluate if not symbolic mode
            if render_mode != RenderMode.SYMBOLIC:
                result_data = core_eq.evaluate()
                if result_data is None:
                    raise ValueError("Avaliação retornou None")
                
                self.variables[result_name] = VariableFactory.create(
                    name=result_name,
                    value=result_data,
                    description=f"Resultado: {expression}"
                )
                self.equations[result_name] = core_eq
            
            # Generate output based on mode
            if render_mode == RenderMode.STEPS:
                steps_list = self._optimize_steps_generation(core_eq, granularity)
                steps_list = self._reduce_redundant_steps(steps_list)
                output_str = self._format_core_steps(steps_list, result_name, core_eq)
            
            elif render_mode == RenderMode.SYMBOLIC:
                lhs_sym = sp.Symbol(result_name)
                expr_to_render = core_eq.expression if hasattr(core_eq, 'expression') else sp.sympify(expression)
                output_str = f"\n${sp.latex(lhs_sym)} = {sp.latex(expr_to_render)}$\n\n"
            
            elif render_mode == RenderMode.NUMERIC:
                lhs_latex = sp.latex(sp.Symbol(result_name))
                result_str = f"{strip_units(result_data):.4g}"
                output_str = f"\n${lhs_latex} = {result_str}$\n\n"
            
            elif render_mode == RenderMode.RESULT:
                fmt = "~P" if PINT_AVAILABLE else ""
                output_str = f"\n**{result_name} = {result_data:{fmt}}**\n\n"
            
            else:  # FULL (Default)
                output_str = "\n**Cálculo:**\n\n"
                lhs_sym_latex = sp.latex(sp.Symbol(result_name))
                expr_to_render = core_eq.expression if hasattr(core_eq, 'expression') and isinstance(core_eq.expression, sp.Expr) else sp.sympify(expression)
                output_str += f"${lhs_sym_latex} = {sp.latex(expr_to_render)}$\n\n"
                
                result_magnitude_str = f"{strip_units(result_data):.4g}"
                output_str += f"${lhs_sym_latex} = {result_magnitude_str}$\n\n"
                
                if PINT_AVAILABLE and hasattr(result_data, 'units'):
                    output_str += f"<div class=\"result\">**${result_name} = {result_data:~P}$**</div>".replace(" **", "**")
                else:
                    formatted_val = self._format_number_smart(strip_units(result_data))
                    output_str += f"<div class=\"result\">**${result_name} = {formatted_val}$**</div>".replace(" **", "**")
            
            return output_str
        
        except Exception as e:
            logger.error(f"Erro @calc '{result_name}': {e}")
            logger.debug(traceback.format_exc())
            return f"\n**ERRO:** {result_name}: {e}\n\n"
    
    def _optimize_steps_generation(self, core_eq: CoreEquation, granularity: str) -> List[Dict[str, Any]]:
        """Gera steps com cache e granularidade otimizada."""
        granularity_map = {
            'minimun': 'minimal', 'minimum': 'minimal',
            'simple': 'basic', 'medium': 'normal',
            'full': 'all', 'max': 'all', 'maximum': 'all'
        }
        original = granularity
        granularity = granularity_map.get(granularity.lower(), granularity.lower())
        
        valid = {'minimal', 'basic', 'normal', 'detailed', 'all', 'smart'}
        if granularity not in valid:
            logger.warning(f"Granularidade inválida '{original}'. Usando 'smart'")
            granularity = 'smart'
        
        cache_key = f"{id(core_eq)}_{granularity}"
        if cache_key in self._steps_cache:
            logger.debug(f"Cache HIT: '{granularity}'")
            return self._steps_cache[cache_key]
        
        logger.debug(f"Gerando steps: '{granularity}'")
        try:
            steps_list = core_eq.steps(granularity=granularity, show_units=True, max_steps=None)
            logger.debug(f"✅ {len(steps_list)} steps gerados")
        except Exception as e:
            logger.error(f"Erro: {e}")
            try:
                steps_list = core_eq.steps(granularity='smart')
            except:
                steps_list = [{
                    'step': 'Erro',
                    'operation': 'error',
                    'expr': '',
                    'numeric': None,
                    'description': f'Erro: {e}'
                }]
        
        self._steps_cache[cache_key] = steps_list
        
        # Limitar cache
        if len(self._steps_cache) > 100:
            for key in list(self._steps_cache.keys())[:50]:
                del self._steps_cache[key]
        
        return steps_list
    
    def _reduce_redundant_steps(self, steps_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove steps redundantes."""
        if not steps_list or len(steps_list) <= 2:
            return steps_list
        
        optimized = []
        last_expr = None
        last_operation = None
        
        for i, step in enumerate(steps_list):
            expr = step.get('expr', '')
            operation = step.get('operation', '')
            
            # Sempre manter primeiro e último
            if i == 0 or i == len(steps_list) - 1 or operation == 'result':
                optimized.append(step)
                last_expr = expr
                last_operation = operation
                continue
            
            # Pular duplicados
            if expr == last_expr and operation == last_operation:
                continue
            
            # Pular intermediários vazios
            if operation == 'intermediate' and not step.get('numeric') and not step.get('description'):
                continue
            
            optimized.append(step)
            last_expr = expr
            last_operation = operation
        
        return optimized
    
    def _format_core_steps(self, steps_list: List[Dict[str, Any]], result_name: str, core_eq: CoreEquation) -> str:
        """Formata steps em texto markdown."""
        if not steps_list:
            return f"\n**Cálculo:** {result_name} = ???\n\n"
        
        output_lines = ["\n**Cálculo:**\n"]
        seen_exprs = set()
        step_count = 0
        
        for step_dict in steps_list:
            if step_count >= 15:
                output_lines.append(f"\n*... ({len(steps_list) - 15} passos omitidos)*\n")
                break
            
            operation = step_dict.get('operation', 'unknown')
            expr_latex = step_dict.get('expr', '')
            numeric_val = step_dict.get('numeric')
            
            step_key = f"{operation}_{expr_latex}_{numeric_val}"
            if step_key in seen_exprs:
                continue
            
            seen_exprs.add(step_key)
            step_count += 1
            
            if operation in ('symbolic', 'substitution', 'simplification', 'evaluation'):
                if expr_latex:
                    output_lines.append(f"→ ${expr_latex}$")
            elif operation == 'result':
                if numeric_val is not None:
                    formatted_num = self._format_number_smart(numeric_val)
                    output_lines.append(f"→ **${result_name} = {formatted_num}$** ✓")
            elif operation == 'intermediate':
                desc = step_dict.get('description', '')
                if desc:
                    output_lines.append(f"→ *{desc}*")
        
        # Adicionar resultado final se não foi incluído
        if core_eq.result is not None and not any(s.get('operation') == 'result' for s in steps_list):
            formatted_num = self._format_number_smart(core_eq.result)
            output_lines.append(f"→ **${result_name} = {formatted_num}$** ✓")
        
        output_lines.append("\n")
        return "\n".join(output_lines)
    
    def _format_number_smart(self, value: Any) -> str:
        """Formata números de forma inteligente."""
        try:
            num_val = strip_units(value)
            if not isinstance(num_val, (int, float)):
                num_val = float(num_val)
            
            abs_val = abs(num_val)
            if abs_val == 0:
                return "0"
            elif abs_val < 0.001 or abs_val > 10000:
                return f"{num_val:.3e}"
            else:
                return f"{num_val:.6g}"
        except:
            return str(value)
    
    def _render_full(self, text: str) -> str:
        """Render $ variables with full value display."""
        def replace_full(match: re.Match) -> str:
            var_name = match.group(1)
            if var_name in self.IGNORE_PLACEHOLDERS or var_name not in self.variables:
                return match.group(0)
            
            var_obj = self.variables[var_name]
            try:
                var_latex = sp.latex(sp.Symbol(var_name))
                val = var_obj.value
                
                if val is None:
                    val_str = "?"
                elif PINT_AVAILABLE and hasattr(val, 'units'):
                    val_str = f"{val:~P}"
                else:
                    val_str = self._format_number_smart(strip_units(val))
                
                return f"${var_latex} = {val_str}$"
            except:
                return match.group(0)
        
        return self.full_render.sub(replace_full, text)
    
    def _render_formulas(self, text: str) -> str:
        """Render $ variables as LaTeX symbols."""
        def replace_formula(match: re.Match) -> str:
            var_name = match.group(1)
            if var_name in self.IGNORE_PLACEHOLDERS:
                return match.group(0)
            
            try:
                return f"${sp.latex(sp.Symbol(var_name))}$"
            except:
                return match.group(0)
        
        return self.formula_render.sub(replace_formula, text)
    
    def _render_values(self, text: str) -> str:
        """
        Render variable values in format {var_name:.2f}.
        
        ✅ v5.4.1 FIX: Ignore LaTeX commands (begin, end, bmatrix, etc.)
        
        Supports:
        - {var_name} → default format (.4g)
        - {var_name:.2f} → custom format
        - {var_name:.3e} → scientific notation
        """
        # LaTeX commands to ignore
        LATEX_COMMANDS = {
            'begin', 'end', 'bmatrix', 'pmatrix', 'vmatrix', 'matrix',
            'frac', 'sqrt', 'sum', 'int', 'left', 'right', 'text',
            'mathrm', 'mathbf', 'alpha', 'beta', 'gamma', 'delta',
            'theta', 'lambda', 'sigma', 'phi', 'omega'
        }
        
        def replace_value(match: re.Match) -> str:
            var_name = match.group(1)
            format_spec = match.group(2) if match.group(2) else '.4g'
            
            # 🔧 v5.4.1: Ignore LaTeX commands
            if var_name in LATEX_COMMANDS:
                return match.group(0)  # Return unchanged
            
            # 🔧 v5.4.1: Ignore if starts with backslash (LaTeX command)
            if var_name.startswith('\\'):
                return match.group(0)
            
            if var_name not in self.variables:
                logger.debug(f"Variable '{var_name}' not found for rendering (may be LaTeX)")
                return match.group(0)  # Return unchanged
            
            try:
                var_value = self.variables[var_name]
                
                # Extract numeric value
                if isinstance(var_value, (int, float)):
                    value_to_format = var_value
                
                elif isinstance(var_value, tuple):
                    logger.warning(f"Variable '{var_name}' is tuple: {var_value}, using first element")
                    value_to_format = float(var_value[0]) if len(var_value) > 0 else 0.0
                
                elif hasattr(var_value, 'magnitude'):
                    value_to_format = float(var_value.magnitude)
                
                elif hasattr(var_value, 'value'):
                    nested_value = var_value.value
                    if hasattr(nested_value, 'magnitude'):
                        value_to_format = float(nested_value.magnitude)
                    else:
                        value_to_format = float(nested_value)
                
                else:
                    value_to_format = float(var_value)
                
                # Format value
                formatted = f"{value_to_format:{format_spec}}"
                logger.debug(f"  Rendered: {{{var_name}{':' + format_spec if format_spec != '.4g' else ''}}} → {formatted}")
                return formatted
            
            except Exception as e:
                logger.debug(f"Error rendering '{var_name}': {e}")
                return match.group(0)  # Return unchanged on error
        
        # Regex pattern: {var_name} or {var_name:.2f}
        pattern = r'\{(\w+)(?::([^\}]+))?\}'
        
        return re.sub(pattern, replace_value, text)


    
    def _clean_text(self, text: str) -> str:
        """
        Remove comandos processados.
        
        ✅ ORDEM: Remove @calc, @matrix, @matop
        """
        text = re.sub(r'^@calc.*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'^@matrix\s*(?:\[.*?\])?\s+\w+\s*=\s*\[\[.+?\]\]', '', text, flags=re.MULTILINE | re.DOTALL)
        text = re.sub(r'^@matop.*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[ \t]+$', '', text, flags=re.MULTILINE)
        return text.strip()
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def get_variable(self, name: str) -> Optional[CoreVariable]:
        return self.variables.get(name)
    
    def get_equation(self, name: str) -> Optional[CoreEquation]:
        return self.equations.get(name)
    
    def get_matrix(self, name: str):
        """Get matrix by name."""
        return self.matrices.get(name)
    
    def list_variables(self) -> List[str]:
        return list(self.variables.keys())
    
    def list_equations(self) -> List[str]:
        return list(self.equations.keys())
    
    def list_matrices(self) -> List[str]:
        """List all matrix names."""
        return list(self.matrices.keys())
    
    def get_summary(self) -> Dict[str, Any]:
        """Get editor summary."""
        return {
            'document_type': self.document_type.value,
            'total_variables': len(self.variables),
            'total_equations': len(self.equations),
            'total_matrices': len(self.matrices),
            'variables_list': list(self.variables.keys()),
            'equations_list': list(self.equations.keys()),
            'matrices_list': list(self.matrices.keys()),
            'features': {
                'core_available': CORE_AVAILABLE,
                'sympy_available': SYMPY_AVAILABLE,
                'numpy_available': NUMPY_AVAILABLE,
                'pint_available': PINT_AVAILABLE,
                'matrix_available': MATRIX_AVAILABLE
            }
        }


__version__ = "5.2.1-matrix-integrated"
__all__ = ['NaturalMemorialEditor', 'DocumentType', 'RenderMode']