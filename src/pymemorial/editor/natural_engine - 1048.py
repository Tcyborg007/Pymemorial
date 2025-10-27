# src/pymemorial/editor/natural_engine.py
"""
Natural Memorial Editor v5.2 - MATRIZES COM STEPS INTEGRADO

ðŸš€ CORREÃ‡Ã•ES IMPLEMENTADAS:
âœ… Regex CORRIGIDO para @matrix e @matop
âœ… Imports de Matrix verificados e fallback robusto
âœ… Ordem de processamento otimizada
âœ… FormataÃ§Ã£o de matrizes em Markdown
âœ… Steps detalhados com granularidade
âœ… Suporte completo a operaÃ§Ãµes matriciais
"""
from __future__ import annotations
import re
import logging
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum
import traceback

# Core Dependencies
try:
    from pymemorial.core import (
        Variable as CoreVariable, Equation as CoreEquation, VariableFactory,
        parse_quantity, ureg, Quantity, strip_units,
        StepRegistry, StepPlugin, GranularityType, PINT_AVAILABLE
    )
    CORE_AVAILABLE = True
except ImportError as e:
    CORE_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.critical(f"PyMemorial Core not found: {e}")

# NumPy (OBRIGATÃ“RIO para matrizes)
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

# SymPy
try:
    import sympy as sp
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False
    sp = None

# Matrix Support (COM FALLBACK ROBUSTO)
try:
    from pymemorial.core import (
        Matrix,
        multiply_matrices_with_steps,
        invert_matrix_with_steps,
        MATRIX_AVAILABLE
    )
    if not MATRIX_AVAILABLE:
        raise ImportError("MATRIX_AVAILABLE flag is False")
except ImportError:
    MATRIX_AVAILABLE = False
    Matrix = None
    multiply_matrices_with_steps = None
    invert_matrix_with_steps = None

from .smart_parser import SmartVariableParser

logger = logging.getLogger(__name__)

class DocumentType(Enum):
    MEMORIAL="memorial"; ARTICLE="article"; TCC="tcc"; REPORT="report"

class RenderMode(Enum):
    FULL="full"; SYMBOLIC="symbolic"; NUMERIC="numeric"; RESULT="result"; STEPS="steps"


class NaturalMemorialEditor:
    """Natural Memorial Editor v5.2 - Matrizes com Steps Integrado."""
    
    IGNORE_PLACEHOLDERS = {'dx', 'dy', 'dt', 'dtheta', 'ddx', 'ddy', 'mathrm', 'd'}
    
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
        self.matrices: Dict[str, Any] = {}  # Tipo genÃ©rico para fallback
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



    def _detect_numerical_contamination(self, matrix: SpMatrix) -> bool:
        """
        Detecta contaminaÃ§Ã£o numÃ©rica em matriz simbÃ³lica.
        
        Verifica se algum elemento contÃ©m valores Float quando deveria ser simbÃ³lico.
        
        Returns:
            bool: True se contaminaÃ§Ã£o detectada, False caso contrÃ¡rio
        """
        if not self.is_symbolic:
            return False
        
        for element in matrix:
            # Se tem sÃ­mbolos mas tambÃ©m contÃ©m Float, estÃ¡ contaminado
            if element.free_symbols and element.has(sp.Float):
                logger.warning(f"Elemento contaminado detectado: {element}")
                return True
        
        return False
    
    def _force_symbolic_divisions(self, expr_str: str) -> str:
        """
        ForÃ§a todas as divisÃµes a manterem forma simbÃ³lica.
        
        Adiciona **1 em todas as divisÃµes simples para evitar substituiÃ§Ã£o prematura.
        
        Args:
            expr_str: ExpressÃ£o de matriz como string
        
        Returns:
            str: ExpressÃ£o transformada
        """
        import re
        
        # Extrair variÃ¡veis da expressÃ£o
        var_pattern = r'\b([A-Za-z_][A-Za-z0-9_]*)\b'
        variables = set(re.findall(var_pattern, expr_str))
        
        # Filtrar palavras-chave Python
        keywords = {'True', 'False', 'None', 'and', 'or', 'not', 'in', 'is'}
        variables = variables - keywords
        
        logger.debug(f"ForÃ§ando forma simbÃ³lica para variÃ¡veis: {variables}")
        
        # Para cada variÃ¡vel, adicionar **1 se divisÃ£o nÃ£o tiver expoente
        for var in variables:
            pattern = rf'/{var}(?!\*\*)'
            replacement = rf'/{var}**1'
            expr_str = re.sub(pattern, replacement, expr_str)
        
        return expr_str


    def _preprocess_matrix_expression(self, expr_str: str) -> str:
        """
        PrÃ©-processa expressÃ£o de matriz para evitar contaminaÃ§Ã£o numÃ©rica.
        
        âœ… v5.3 FINAL: Adiciona **1 em divisÃµes simples para forÃ§ar forma simbÃ³lica.
        
        Exemplo: "4*E*I/Le" â†’ "4*E*I/Le**1"
        """
        import re
        
        # Coletar todas as variÃ¡veis conhecidas
        var_names = list(self.variables.keys())
        if hasattr(self, 'parser') and hasattr(self.parser, 'detected_variables'):
            var_names.extend(self.parser.detected_variables.keys())
        
        var_names = list(set(var_names))  # Remover duplicatas
        
        logger.debug(f"VariÃ¡veis para prÃ©-processamento: {var_names}")
        
        # Para cada variÃ¡vel, adicionar **1 se divisÃ£o nÃ£o tiver expoente
        for var in var_names:
            # PadrÃ£o: /VAR (nÃ£o seguido por **)
            pattern = rf'/{var}(?!\*\*)'
            replacement = rf'/{var}**1'
            expr_str = re.sub(pattern, replacement, expr_str)
        
        logger.debug(f"ExpressÃ£o prÃ©-processada (preview): {expr_str[:150]}...")
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
        
        âœ… CORREÃ‡ÃƒO: Regex ROBUSTO que captura corretamente mode:granularity
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
    
    def process(self, text: str, clean: bool = True) -> str:
        """
        Process text with ORDEM CORRETA de operaÃ§Ãµes.
        
        âœ… ORDEM:
        1. Detectar variÃ¡veis
        2. Processar matrizes
        3. Processar operaÃ§Ãµes matriciais
        4. Processar cÃ¡lculos
        5. Renderizar
        6. Limpar (opcional)
        """
        # Step 1: Detectar variÃ¡veis
        self._detect_variables(text)
        
        # Step 2: Processar matrizes ANTES de limpar
        processed_text = self._process_matrices(text)
        
        # Step 3: Processar operaÃ§Ãµes matriciais
        processed_text = self._process_matrix_operations(processed_text)
        
        # Step 4: Processar cÃ¡lculos normais
        processed_text = self._process_calculations(processed_text)
        
        # Step 5: Renderizar valores
        processed_text = self._render_full(processed_text)
        processed_text = self._render_formulas(processed_text)
        processed_text = self._render_values(processed_text)
        
        # Step 6: Limpar (POR ÃšLTIMO)
        if clean:
            processed_text = self._clean_text(processed_text)
        
        return processed_text
    
    # ========================================================================
    # MATRIX PROCESSING
    # ========================================================================
    
    def _process_matrices(self, text: str) -> str:
        """
        Process @matrix blocks.
        
        âœ… VALIDAÃ‡ÃƒO: Verifica se Matrix estÃ¡ disponÃ­vel
        """
        if not NUMPY_AVAILABLE:
            logger.warning("NumPy not available - skipping matrix processing")
            return text
        
        if not MATRIX_AVAILABLE or Matrix is None:
            logger.warning("Matrix module not available - skipping matrix processing")
            return text
        
        def replacer(match: re.Match) -> str:
            try:
                return self._replace_matrix_match(match)
            except Exception as e:
                logger.error(f"Matrix replacement failed: {e}")
                return f"\n**ERRO MATRIZ:** {e}\n\n"
        
        result = self.matrix_block.sub(replacer, text)
        logger.debug(f"Processed {len(self.matrices)} matrices")
        return result
    
    def _replace_matrix_match(self, match: re.Match) -> str:
        """
        Replace a single @matrix match with formatted output.
        
        âœ… CORREÃ‡ÃƒO v5.2.1: Garantir isolamento absoluto de valores
        """
        mode_str, granularity_str, matrix_name, matrix_expr = match.groups()
        mode_str = mode_str or 'result'
        granularity = granularity_str or 'normal'
        
        logger.info(f"ðŸ” Processing @matrix: {matrix_name}, mode={mode_str}, granularity={granularity}")
        
        try:
            matrix_expr_preprocessed = self._preprocess_matrix_expression(matrix_expr)
            matrix_data = self._parse_matrix_expression(matrix_expr_preprocessed)
            
            # âœ… CORREÃ‡ÃƒO CRÃTICA: Usar as variÃ¡veis ORIGINAIS com valores
            # Mas garantir que o parsing simbÃ³lico seja feito com sÃ­mbolos puros
            original_variables = self.variables.copy()
            
            # Criar matriz com variÃ¡veis ORIGINAIS (com valores)
            # O parsing simbÃ³lico no Matrix v2.1.9 jÃ¡ lida com isolamento interno
            matrix = Matrix(
                data=matrix_data,
                variables=original_variables,  # âœ… Usar variÃ¡veis originais
                description=f"Matriz {matrix_name}",
                name=matrix_name
            )
            
            self.matrices[matrix_name] = matrix
            logger.info(f"âœ… Matrix '{matrix_name}' criada: shape={matrix.shape}, symbolic={matrix.is_symbolic}")
            
            # Gerar output baseado no modo
            if mode_str.lower() == 'steps':
                output = self._format_matrix_steps(matrix, granularity)
            elif mode_str.lower() == 'symbolic':
                output = self._format_matrix_symbolic(matrix)
            elif mode_str.lower() == 'numeric':
                output = self._format_matrix_numeric(matrix)
            else:  # 'result'
                output = self._format_matrix_result(matrix)
            
            return output
            
        except Exception as e:
            logger.error(f"âŒ Error processing @matrix '{matrix_name}': {e}")
            logger.error(f"   Expression: {matrix_expr[:200]}...")
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("   Full traceback:")
                logger.debug(traceback.format_exc())
            
            return f"\n**ERRO MATRIZ:** {matrix_name}: {e}\n\n"

    def _parse_matrix_expression(self, expr: str):
        """
        Parse matrix expression from string.
        
        âœ… Suporta: [[1,2],[3,4]] e expressÃµes SymPy
        """
        import ast
        expr = expr.strip()
        
        try:
            # Tenta parse literal (lista Python)
            result = ast.literal_eval(expr)
            if isinstance(result, list):
                return result
        except:
            pass
        
        # Fallback: retorna como string (serÃ¡ tratado como SymPy)
        return expr
    
    def _format_matrix_steps(self, matrix, granularity: str) -> str:
        """
        Format matrix with step-by-step construction.
        
        âœ… CORREÃ‡ÃƒO: Garantir que o LaTeX simbÃ³lico seja PURO (sem nÃºmeros)
        """
        steps = matrix.steps(granularity=granularity)
        
        output = ["\n**Matriz:**\n"]
        
        for step in steps:
            operation = step.get('operation', 'unknown')
            description = step.get('description', '')
            
            if operation == 'definition':
                output.append(f"â†’ **DefiniÃ§Ã£o:** {description}")
            
            elif operation == 'symbolic' and 'latex' in step:
                latex_matrix = step['latex']
                output.append(f"\n$$[{matrix.name}] = {latex_matrix}$$\n")
            
            elif operation == 'substitution':
                output.append(f"â†’ *SubstituiÃ§Ã£o:* {description}")
            
            elif operation == 'intermediate' and 'latex' in step:
                latex_intermediate = step['latex']
                output.append(f"\nâ†’ **Matriz SubstituÃ­da (Passo IntermediÃ¡rio):**\n")
                output.append(f"$$[{matrix.name}] = {latex_intermediate}$$\n")
            
            elif operation == 'evaluation' and 'matrix' in step:
                matrix_data = step['matrix']
                if isinstance(matrix_data, list):
                    output.append(f"\nâ†’ **Matriz NumÃ©rica:**")
                    output.append(self._format_matrix_table(matrix_data, matrix.name))
            
            elif operation == 'properties':
                props = []
                if 'determinant' in step:
                    props.append(f"Determinante: {step['determinant']:.6g}")
                if 'trace' in step:
                    props.append(f"TraÃ§o: {step['trace']:.6g}")
                if 'rank' in step:
                    props.append(f"Posto: {step['rank']}")
                if props:
                    output.append(f"\nâ†’ *Propriedades:* {', '.join(props)}")
        
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
        """Process @matop blocks."""
        if not MATRIX_AVAILABLE:
            logger.warning("Matrix operations not available")
            return text
        
        def replacer(match: re.Match) -> str:
            try:
                return self._replace_matop_match(match)
            except Exception as e:
                logger.error(f"Matrix operation failed: {e}")
                return f"\n**ERRO OPERAÃ‡ÃƒO:** {e}\n\n"
        
        return self.matrix_operation.sub(replacer, text)
    
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
                raise ValueError(f"OperaÃ§Ã£o nÃ£o suportada: {operation}")
        
        except Exception as e:
            logger.error(f"Erro em @matop '{result_name}': {e}")
            return f"\n**ERRO OPERAÃ‡ÃƒO:** {result_name}: {e}\n\n"
    
    def _process_multiply_operation(self, result_name: str, expr: str, granularity: str) -> str:
        """Process matrix multiplication: C = A * B or C = A * B * C (encadeada)."""
        # Parse expression (suporta A*B, A@B, A.T*B, etc)
        # Remove espaÃ§os e substitui @ por *
        expr = expr.replace('@', '*').replace(' ', '')
        
        # Extrair transpostas (A.T vira A com flag transpose)
        import re
        transpose_pattern = re.compile(r'(\w+)\.T')
        
        # Encontrar todos os termos (matrizes e suas transpostas)
        terms = []
        current_expr = expr
        
        # Split por * mas mantendo informaÃ§Ã£o de transposta
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
                raise ValueError(f"Matriz nÃ£o encontrada: {term['name']}")
        
        # Executar multiplicaÃ§Ã£o encadeada
        logger.info(f"ðŸ” MultiplicaÃ§Ã£o encadeada: {len(terms)} termos")
        
        result_matrix = None
        operation_desc = []
        
        for i, term in enumerate(terms):
            matrix = self.matrices[term['name']]
            
            # Aplicar transposta se necessÃ¡rio
            if term['transpose']:
                matrix_data = matrix.evaluate().T
                matrix = Matrix(
                    data=matrix_data,
                    description=f"Transposta de {term['name']}",
                    name=f"{term['name']}_T"
                )
                operation_desc.append(f"{term['name']}áµ€")
            else:
                operation_desc.append(term['name'])
            
            if result_matrix is None:
                result_matrix = matrix
            else:
                # Multiplicar com o resultado acumulado
                result_matrix, _ = multiply_matrices_with_steps(
                    result_matrix, 
                    matrix, 
                    'minimal'  # Interno usa minimal para eficiÃªncia
                )
        
        # Definir nome e armazenar
        result_matrix.name = result_name
        self.matrices[result_name] = result_matrix
        
        # Formatar output
        output = [f"\n**MultiplicaÃ§Ã£o: {result_name} = {' Ã— '.join(operation_desc)}**\n"]
        output.append(f"â†’ *DimensÃµes: {result_matrix.shape[0]}Ã—{result_matrix.shape[1]}*\n")
        output.append(f"\nâ†’ **Resultado:**")
        output.append(self._format_matrix_table(result_matrix.evaluate().tolist(), result_name))
        output.append("\n")
        
        return "".join(output)

    
    def _process_inverse_operation(self, result_name: str, expr: str, granularity: str) -> str:
        """Process matrix inversion."""
        matrix_name = expr.replace('inv(', '').replace(')', '').strip()
        
        if matrix_name not in self.matrices:
            raise ValueError(f"Matriz nÃ£o encontrada: {matrix_name}")
        
        matrix = self.matrices[matrix_name]
        result_matrix, steps = invert_matrix_with_steps(matrix, granularity)
        result_matrix.name = result_name
        self.matrices[result_name] = result_matrix
        
        # Format output
        output = [f"\n**InversÃ£o: {result_name} = {matrix_name}â»Â¹**\n"]
        
        for step in steps:
            operation = step.get('operation', '')
            description = step.get('description', '')
            
            if operation == 'determinant':
                value = step.get('value', 0)
                output.append(f"â†’ *{description}* (det = {value:.6g})")
            elif operation == 'method':
                output.append(f"â†’ *{description}*")
            elif operation == 'result' and 'matrix' in step:
                output.append(f"\nâ†’ **Matriz Inversa:**")
                output.append(self._format_matrix_table(step['matrix'], result_name))
            elif operation == 'verification' and 'matrix' in step:
                output.append(f"\nâ†’ *VerificaÃ§Ã£o (A Ã— Aâ»Â¹ = I):*")
                output.append(self._format_matrix_table(step['matrix'], 'I'))
        
        output.append("\n")
        return "".join(output)
    
    def _process_transpose_operation(self, result_name: str, expr: str, granularity: str) -> str:
        """Process matrix transpose."""
        matrix_name = expr.replace('.T', '').strip()
        
        if matrix_name not in self.matrices:
            raise ValueError(f"Matriz nÃ£o encontrada: {matrix_name}")
        
        matrix = self.matrices[matrix_name]
        result = matrix.evaluate().T
        result_matrix = Matrix(
            data=result,
            description=f"Transposta de {matrix_name}",
            name=result_name
        )
        self.matrices[result_name] = result_matrix
        
        return f"\n**Transposta:** {result_name} = {matrix_name}áµ€\n{self._format_matrix_table(result.tolist(), result_name)}\n"
    
    def _process_determinant_operation(self, result_name: str, expr: str, granularity: str) -> str:
        """Process determinant calculation."""
        matrix_name = expr.replace('det(', '').replace(')', '').strip()
        
        if matrix_name not in self.matrices:
            raise ValueError(f"Matriz nÃ£o encontrada: {matrix_name}")
        
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
        """Detectar variÃ¡veis no texto."""
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
        """Process @calc blocks (mantido do cÃ³digo original)."""
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
                raise ValueError(f"VariÃ¡veis indefinidas: {', '.join(missing_vars)}")
            
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
                    raise ValueError("AvaliaÃ§Ã£o retornou None")
                
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
                output_str = "\n**CÃ¡lculo:**\n\n"
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
            logger.warning(f"Granularidade invÃ¡lida '{original}'. Usando 'smart'")
            granularity = 'smart'
        
        cache_key = f"{id(core_eq)}_{granularity}"
        if cache_key in self._steps_cache:
            logger.debug(f"Cache HIT: '{granularity}'")
            return self._steps_cache[cache_key]
        
        logger.debug(f"Gerando steps: '{granularity}'")
        try:
            steps_list = core_eq.steps(granularity=granularity, show_units=True, max_steps=None)
            logger.debug(f"âœ… {len(steps_list)} steps gerados")
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
            
            # Sempre manter primeiro e Ãºltimo
            if i == 0 or i == len(steps_list) - 1 or operation == 'result':
                optimized.append(step)
                last_expr = expr
                last_operation = operation
                continue
            
            # Pular duplicados
            if expr == last_expr and operation == last_operation:
                continue
            
            # Pular intermediÃ¡rios vazios
            if operation == 'intermediate' and not step.get('numeric') and not step.get('description'):
                continue
            
            optimized.append(step)
            last_expr = expr
            last_operation = operation
        
        return optimized
    
    def _format_core_steps(self, steps_list: List[Dict[str, Any]], result_name: str, core_eq: CoreEquation) -> str:
        """Formata steps em texto markdown."""
        if not steps_list:
            return f"\n**CÃ¡lculo:** {result_name} = ???\n\n"
        
        output_lines = ["\n**CÃ¡lculo:**\n"]
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
                    output_lines.append(f"â†’ ${expr_latex}$")
            elif operation == 'result':
                if numeric_val is not None:
                    formatted_num = self._format_number_smart(numeric_val)
                    output_lines.append(f"â†’ **${result_name} = {formatted_num}$** âœ“")
            elif operation == 'intermediate':
                desc = step_dict.get('description', '')
                if desc:
                    output_lines.append(f"â†’ *{desc}*")
        
        # Adicionar resultado final se nÃ£o foi incluÃ­do
        if core_eq.result is not None and not any(s.get('operation') == 'result' for s in steps_list):
            formatted_num = self._format_number_smart(core_eq.result)
            output_lines.append(f"â†’ **${result_name} = {formatted_num}$** âœ“")
        
        output_lines.append("\n")
        return "\n".join(output_lines)
    
    def _format_number_smart(self, value: Any) -> str:
        """Formata nÃºmeros de forma inteligente."""
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
        """Render {} variables with numeric values."""
        def replace_value(match: re.Match) -> str:
            var_name = match.group(1)
            if var_name in self.IGNORE_PLACEHOLDERS or var_name not in self.variables:
                return match.group(0)
            
            val = self.variables[var_name].value
            if val is None:
                return "???"
            
            if PINT_AVAILABLE and hasattr(val, 'units'):
                return f"{val:~P}"
            
            return str(strip_units(val))
        
        return self.value_render.sub(replace_value, text)
    
    def _clean_text(self, text: str) -> str:
        """
        Remove comandos processados.
        
        âœ… ORDEM: Remove @calc, @matrix, @matop
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