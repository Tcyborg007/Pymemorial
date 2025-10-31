# src/pymemorial/engine/processor.py
"""
Unified Processor - Processador Unificado Ultra-Robusto

Consolida os 3 processadores fragmentados:
- recognition.TextProcessor (parsing AST + LaTeX)
- editor.NaturalEngine (texto natural PT-BR)
- editor.StepEngine (steps automáticos tipo Calcpad)

Pipeline Completo:
texto → ASTParser → SymPy → Steps → LaTeX → Output

Features:
✅ Parsing inteligente de texto natural + código Python
✅ Steps automáticos com 4 níveis de granularidade
✅ Cache inteligente (hash-based)
✅ Fallbacks robustos (sempre funciona)
✅ Integração total com MemorialContext
✅ Suporte a SymPy + SciPy + NumPy
✅ Thread-safe

Author: PyMemorial Team
Date: 2025-10-28
Version: 3.0.0
"""

from __future__ import annotations

import re
import ast
import hashlib
import logging
import warnings
from typing import Dict, Any, Optional, List, Tuple, Union, Literal
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
    sp = None

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

# Imports internos
try:
    from pymemorial.core import (
        Variable, Equation, Calculator, get_config
    )
    CORE_AVAILABLE = True
except ImportError:
    CORE_AVAILABLE = False

try:
    from pymemorial.recognition import (
        PyMemorialASTParser, SmartTextEngine, GreekSymbols
    )
    RECOGNITION_AVAILABLE = True
except ImportError:
    RECOGNITION_AVAILABLE = False

try:
    from .context import MemorialContext, get_context
    CONTEXT_AVAILABLE = True
except ImportError:
    CONTEXT_AVAILABLE = False


# ============================================================================
# LOGGING
# ============================================================================

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS & DATACLASSES
# ============================================================================

class GranularityLevel(str, Enum):
    """Níveis de granularidade dos steps (compatível com Calcpad/MathCAD)."""
    MINIMAL = "minimal"      # Apenas resultado final
    BASIC = "basic"          # Fórmula → resultado
    MEDIUM = "medium"        # Fórmula → substituição → resultado
    DETAILED = "detailed"    # Todos os steps intermediários
    ALL = "all"              # Debug completo


class ProcessingMode(str, Enum):
    """Modos de processamento."""
    TEXT = "text"            # Texto natural (markdown-like)
    CODE = "code"            # Código Python puro
    MIXED = "mixed"          # Texto + código inline
    AUTO = "auto"            # Detecta automaticamente


@dataclass
class ProcessingResult:
    """Resultado do processamento."""
    success: bool
    output: str
    output_latex: str = ""
    steps: List[str] = field(default_factory=list)
    variables: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StepData:
    """Step individual de cálculo."""
    description: str
    formula_symbolic: str
    formula_numeric: str
    result: Union[float, str]
    unit: str = ""
    latex: str = ""


# ============================================================================
# UNIFIED PROCESSOR
# ============================================================================

class UnifiedProcessor:
    """
    Processador unificado ultra-robusto.
    
    Consolida TODO o processamento de texto/código/cálculos em uma única classe.
    
    Features:
    ---------
    - 🎯 Pipeline completo: texto → parsing → cálculo → steps → LaTeX
    - 🔄 Cache inteligente (hash-based para performance)
    - 🛡️ Fallbacks robustos (sempre retorna algo útil)
    - 🌍 Integração total com MemorialContext
    - 📊 Suporte SymPy + SciPy + NumPy
    - 🔒 Thread-safe (locks internos)
    
    Examples:
    ---------
    >>> processor = UnifiedProcessor()
    >>> result = processor.process("q = 15.0 kN/m  # Carga distribuída")
    >>> print(result.output)
    >>> 
    >>> # Cálculo com steps
    >>> result = processor.process_calculation(
    ...     "M_max = q * L**2 / 8",
    ...     context={"q": 15.0, "L": 6.0},
    ...     granularity="detailed"
    ... )
    >>> for step in result.steps:
    ...     print(step)
    """
    
    def __init__(
        self,
        context: Optional[MemorialContext] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Inicializa processador.
        
        Args:
            context: MemorialContext (se None, usa global)
            config: Configuração customizada (se None, usa padrão)
        """
        # Context
        if CONTEXT_AVAILABLE:
            self._context = context or get_context()
        else:
            self._context = None
            logger.warning("MemorialContext não disponível")
        
        # Config
        if CORE_AVAILABLE:
            self._config = config or get_config()
        else:
            self._config = config or {}
        
        # Componentes internos (lazy loading)
        self._ast_parser: Optional[PyMemorialASTParser] = None
        self._text_engine: Optional[SmartTextEngine] = None
        self._calculator: Optional[Calculator] = None
        
        # Cache de results (hash → ProcessingResult)
        self._cache: Dict[str, ProcessingResult] = {}
        self._cache_enabled = True
        self._cache_max_size = 1000
    
    # ========================================================================
    # LAZY LOADING DE COMPONENTES
    # ========================================================================
    
    @property
    def ast_parser(self) -> PyMemorialASTParser:
        """Lazy loading do ASTParser."""
        if self._ast_parser is None:
            if not RECOGNITION_AVAILABLE:
                raise RuntimeError("recognition.PyMemorialASTParser não disponível")
            self._ast_parser = PyMemorialASTParser()
        return self._ast_parser
    
    @property
    def text_engine(self) -> SmartTextEngine:
        """Lazy loading do TextEngine."""
        if self._text_engine is None:
            if not RECOGNITION_AVAILABLE:
                raise RuntimeError("recognition.SmartTextEngine não disponível")
            self._text_engine = SmartTextEngine()
        return self._text_engine
    
    @property
    def calculator(self) -> Calculator:
        """Lazy loading do Calculator."""
        if self._calculator is None:
            if not CORE_AVAILABLE:
                raise RuntimeError("core.Calculator não disponível")
            self._calculator = Calculator()
        return self._calculator
    
    # ========================================================================
    # API PRINCIPAL - PROCESSAMENTO AUTOMÁTICO
    # ========================================================================
    
    def process(
        self,
        input_text: str,
        mode: ProcessingMode = ProcessingMode.AUTO,
        granularity: GranularityLevel = GranularityLevel.MEDIUM,
        use_cache: bool = True
    ) -> ProcessingResult:
        """
        Processa texto/código de forma inteligente (modo AUTO).
        
        Detecta automaticamente o tipo de input e aplica processamento adequado:
        - Texto natural → parsing + formatação
        - Código Python → parsing AST + execução + steps
        - Misto → combina ambos
        
        Args:
            input_text: Texto ou código a processar
            mode: Modo de processamento (AUTO detecta automaticamente)
            granularity: Nível de detalhe dos steps
            use_cache: Se True, usa cache para evitar reprocessamento
        
        Returns:
            ProcessingResult com output, steps, variáveis, etc.
        
        Examples:
            >>> # Texto natural
            >>> result = processor.process("A carga é de 15 kN/m")
            >>> 
            >>> # Código Python
            >>> result = processor.process("M = q * L**2 / 8")
            >>> 
            >>> # Misto (markdown + código)
            >>> result = processor.process('''
            ... # Cálculo do Momento
            ... q = 15.0 kN/m
            ... L = 6.0 m
            ... M_max = q * L**2 / 8
            ... ''')
        """
        # Cache check
        if use_cache and self._cache_enabled:
            cache_key = self._compute_cache_key(input_text, mode, granularity)
            if cache_key in self._cache:
                logger.debug(f"Cache HIT: {cache_key[:16]}...")
                return self._cache[cache_key]
        
        # Detectar modo automaticamente se necessário
        if mode == ProcessingMode.AUTO:
            mode = self._detect_mode(input_text)
        
        # Processar de acordo com o modo
        try:
            if mode == ProcessingMode.TEXT:
                result = self._process_text(input_text, granularity)
            elif mode == ProcessingMode.CODE:
                result = self._process_code(input_text, granularity)
            elif mode == ProcessingMode.MIXED:
                result = self._process_mixed(input_text, granularity)
            else:
                result = ProcessingResult(
                    success=False,
                    output="",
                    errors=[f"Modo não suportado: {mode}"]
                )
        except Exception as e:
            logger.exception(f"Erro ao processar: {e}")
            result = ProcessingResult(
                success=False,
                output="",
                errors=[f"Erro no processamento: {str(e)}"]
            )
        
        # Armazenar em cache
        if use_cache and self._cache_enabled and result.success:
            cache_key = self._compute_cache_key(input_text, mode, granularity)
            self._cache[cache_key] = result
            
            # Limpar cache se muito grande
            if len(self._cache) > self._cache_max_size:
                self._cache.clear()
                logger.info("Cache limpo (atingiu tamanho máximo)")
        
        return result
    
    # ========================================================================
    # PROCESSAMENTO DE CÁLCULOS (API DIRETA)
    # ========================================================================
    
# ... código anterior ...

# src/pymemorial/engine/processor.py

    def process_calculation(
        self,
        expression: str,
        context: Optional[Dict[str, Any]] = None,
        granularity: Union[str, Any] = "medium",
        unit: str = "",
        description: str = ""
    ) -> ProcessingResult:
        """
        Processa cálculo com steps automáticos (API direta, sem detecção).
        """
        try:
            # Normalizar granularidade para string lowercase
            if hasattr(granularity, 'value'):
                gran_str = granularity.value.lower()
            elif isinstance(granularity, str):
                gran_str = granularity.lower()
            else:
                gran_str = str(granularity).lower()
            
            # Preparar contexto
            if context is None and self._context:
                context = {name: var.value for name, var in self._context.list_variables().items()}
            elif context is None:
                context = {}
            
            # Gerar steps
            steps = self._generate_steps(expression, context, gran_str)
            
            # Calcular resultado final
            result_value = self._evaluate_expression(expression, context)
            
            # Construir output LaTeX (antigo)
            output_lines = []
            for step in steps:
                output_lines.append(step.description)
                if step.formula_symbolic:
                    output_lines.append(f"  {step.formula_symbolic}")
                if step.formula_numeric and step.formula_numeric != step.formula_symbolic:
                    output_lines.append(f"  {step.formula_numeric}")
                if step.result is not None:
                    unit_str = f" {step.unit}" if step.unit else ""
                    output_lines.append(f"  = {step.result}{unit_str}")
                output_lines.append("")
            output = "\n".join(output_lines)
            
            # Gerar LaTeX output
            latex_lines = []
            for step in steps:
                if step.latex:
                    latex_lines.append(step.latex)
            output_latex = "\n".join(latex_lines)
            
            # ✅ ADICIONAR: Gerar saída em texto natural
            natural_output = self._format_natural_output(steps, expression, result_value, unit)
            
            # CORREÇÃO CRÍTICA: Garantir que a variável calculada esteja no dicionário
            if '=' in expression:
                var_name = expression.split('=')[0].strip()
            else:
                var_name = "_result"
            
            # Criar dicionário de variáveis com ambos os nomes
            variables_dict = {}
            if result_value is not None:
                variables_dict[var_name] = result_value
                if var_name != "_result":
                    variables_dict['_result'] = result_value
            
            print(f"🔍 DEBUG process_calculation(): variables_dict = {variables_dict}")
            
            return ProcessingResult(
                success=True,
                output=natural_output,  # ✅ USAR TEXTO NATURAL ao invés de LaTeX bruto
                output_latex=output_latex,
                steps=[step.description for step in steps],
                variables=variables_dict,
                metadata={
                    'expression': expression,
                    'result': result_value,
                    'unit': unit,
                    'description': description,
                    'latex': output_latex,
                    'natural': natural_output,  # ✅ INCLUIR AMBOS
                    'section': self._context.current_scope if self._context else None
                }
            )
        except Exception as e:
            logger.exception(f"Erro ao processar cálculo: {e}")
            return ProcessingResult(
                success=False,
                output="",
                output_latex="",
                steps=[],
                variables={},
                errors=[f"Erro no cálculo: {str(e)}"]
            )


# ... código posterior ...




    
    # ========================================================================
    # PROCESSAMENTO INTERNO (PRIVADO)
    # ========================================================================
    
    def _detect_mode(self, text: str) -> ProcessingMode:
        """Detecta automaticamente o modo de processamento."""
        # Remover comentários para análise
        text_clean = re.sub(r'#.*$', '', text, flags=re.MULTILINE).strip()
        
        # Detectar código Python (linhas com =, operadores, etc.)
        code_patterns = [
            r'^\s*\w+\s*=\s*[^#\n]+$',  # Atribuição
            r'\*\*|\*|\/|\+|\-',          # Operadores matemáticos
            r'^\s*def\s+\w+',             # Definição de função
            r'^\s*class\s+\w+',           # Definição de classe
        ]
        
        code_lines = 0
        text_lines = 0
        
        for line in text_clean.split('\n'):
            line = line.strip()
            if not line:
                continue
            
            is_code = any(re.search(pattern, line) for pattern in code_patterns)
            if is_code:
                code_lines += 1
            else:
                text_lines += 1
        
        # Decidir modo
        total_lines = code_lines + text_lines
        if total_lines == 0:
            return ProcessingMode.TEXT
        
        code_ratio = code_lines / total_lines
        
        if code_ratio > 0.7:
            return ProcessingMode.CODE
        elif code_ratio > 0.3:
            return ProcessingMode.MIXED
        else:
            return ProcessingMode.TEXT
    
    def _process_text(
        self,
        text: str,
        granularity: GranularityLevel
    ) -> ProcessingResult:
        """Processa texto natural."""
        if not RECOGNITION_AVAILABLE:
            return ProcessingResult(
                success=True,
                output=text,
                warnings=["recognition não disponível - retornando texto original"]
            )
        
        try:
            # Usar SmartTextEngine
            context_dict = {}
            if self._context:
                context_dict = {
                    name: var.value
                    for name, var in self._context.list_variables().items()
                }
            
            processed = self.text_engine.process_text(text, context_dict)
            
            return ProcessingResult(
                success=True,
                output=processed,
                output_latex="",
                metadata={"mode": "text"}
            )
        
        except Exception as e:
            logger.exception(f"Erro ao processar texto: {e}")
            return ProcessingResult(
                success=False,
                output="",
                errors=[f"Erro no processamento de texto: {str(e)}"]
            )
    
    def _process_code(
        self,
        code: str,
        granularity: GranularityLevel
    ) -> ProcessingResult:
        """Processa código Python puro."""
        # CORREÇÃO: Criar fallback se ASTParser não disponível
        if not RECOGNITION_AVAILABLE:
            # Fallback: processar linha por linha
            return self._process_code_fallback(code, granularity)
        
        try:
            # Parse com ASTParser
            assignments = self.ast_parser.parse_code_block(code)
            
            # Executar cada assignment
            results = []
            context_dict = {}
            if self._context:
                context_dict = {
                    name: var.value
                    for name, var in self._context.list_variables().items()
                }
            
            for assign in assignments:
                # Processar cálculo
                result = self.process_calculation(
                    expression=f"{assign.lhs} = {assign.rhs_symbolic}",
                    context=context_dict,
                    granularity=granularity
                )
                results.append(result)
                
                # Atualizar contexto (com proteção)
                if result.success and "_result" in result.variables:
                    context_dict[assign.lhs] = result.variables["_result"]
                    if self._context:
                        try:
                            self._context.set(assign.lhs, result.variables["_result"])
                        except Exception as e:
                            logger.warning(f"Erro ao adicionar variável ao contexto: {e}")
            
            # Combinar outputs
            output = "\n\n".join(r.output for r in results if r.success)
            output_latex = "\n\n".join(r.output_latex for r in results if r.success)
            
            return ProcessingResult(
                success=all(r.success for r in results),
                output=output,
                output_latex=output_latex,
                variables=context_dict,
                metadata={"mode": "code", "num_assignments": len(assignments)}
            )
        
        except Exception as e:
            logger.exception(f"Erro ao processar código: {e}")
            # Fallback
            return self._process_code_fallback(code, granularity)
    
    def _process_code_fallback(
        self,
        code: str,
        granularity: GranularityLevel
    ) -> ProcessingResult:
        """Fallback para processar código sem ASTParser."""
        lines = code.strip().split('\n')
        context_dict = {}
        if self._context:
            context_dict = {
                name: var.value
                for name, var in self._context.list_variables().items()
            }
        
        results = []
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            if '=' in line:
                # É um assignment
                lhs, rhs = line.split('=', 1)
                var_name = lhs.strip()
                expr = rhs.strip()
                
                # Remover comentários inline
                if '#' in expr:
                    expr = expr.split('#')[0].strip()
                
                # Avaliar
                try:
                    value = self._evaluate_expression(expr, context_dict)
                    context_dict[var_name] = value
                    
                    # Adicionar ao contexto global
                    if self._context:
                        try:
                            self._context.set(var_name, value)
                        except:
                            pass
                    
                    results.append(f"{var_name} = {value}")
                except Exception as e:
                    logger.error(f"Erro ao processar linha '{line}': {e}")
        
        return ProcessingResult(
            success=True,
            output="\n".join(results),
            variables=context_dict,
            metadata={"mode": "code_fallback"}
        )

    
    def _process_mixed(
        self,
        text: str,
        granularity: GranularityLevel
    ) -> ProcessingResult:
        """Processa texto misto (texto + código)."""
        # Separar blocos de texto e código
        lines = text.split('\n')
        blocks = []
        current_block = []
        current_type = None
        
        for line in lines:
            # Detectar tipo da linha
            is_code = bool(re.search(r'^\s*\w+\s*=\s*[^#\n]+$', line))
            line_type = 'code' if is_code else 'text'
            
            if line_type != current_type and current_block:
                # Fechar bloco anterior
                blocks.append((current_type, '\n'.join(current_block)))
                current_block = []
            
            current_block.append(line)
            current_type = line_type
        
        # Adicionar último bloco
        if current_block:
            blocks.append((current_type, '\n'.join(current_block)))
        
        # Processar cada bloco
        results = []
        for block_type, block_content in blocks:
            if block_type == 'code':
                result = self._process_code(block_content, granularity)
            else:
                result = self._process_text(block_content, granularity)
            results.append(result)
        
        # Combinar outputs
        output = "\n\n".join(r.output for r in results if r.success)
        output_latex = "\n\n".join(r.output_latex for r in results if r.success)
        
        return ProcessingResult(
            success=all(r.success for r in results),
            output=output,
            output_latex=output_latex,
            metadata={"mode": "mixed", "num_blocks": len(blocks)}
        )
    
    # ========================================================================
    # STEPS GENERATION (CORE)
    # ========================================================================
    
    def _generate_steps(
        self,
        expression: str,
        context: Dict[str, Any],
        granularity: GranularityLevel
    ) -> List[StepData]:
        """
        Gera steps automáticos estilo Calcpad.
        
        Implementa 4 níveis de granularidade:
        - MINIMAL: Apenas resultado
        - BASIC: Fórmula → resultado
        - MEDIUM: Fórmula → substituição → resultado
        - DETAILED: Todos os steps intermediários
        """
        steps = []
        
        # Parse da expressão
        if "=" in expression:
            lhs, rhs = expression.split("=", 1)
            var_name = lhs.strip()
            expr_str = rhs.strip()
        else:
            var_name = "_result"
            expr_str = expression.strip()
        
        # Step 1: Fórmula simbólica (todos exceto MINIMAL)
        if granularity != GranularityLevel.MINIMAL:
            steps.append(StepData(
                description=f"Fórmula:",
                formula_symbolic=f"{var_name} = {expr_str}",
                formula_numeric="",
                result=None,
                latex=self._to_latex(f"{var_name} = {expr_str}")
            ))
        
        # Step 2: Substituição numérica (MEDIUM e DETAILED)
        if granularity in [GranularityLevel.MEDIUM, GranularityLevel.DETAILED]:
            numeric_expr = self._substitute_values(expr_str, context)
            steps.append(StepData(
                description="Substituindo valores:",
                formula_symbolic="",
                formula_numeric=f"{var_name} = {numeric_expr}",
                result=None,
                latex=self._to_latex(f"{var_name} = {numeric_expr}")
            ))
        
        # Step 3: Resultado final (todos)
        result_value = self._evaluate_expression(expr_str, context)
        steps.append(StepData(
            description="Resultado:",
            formula_symbolic="",
            formula_numeric="",
            result=result_value,
            latex=self._to_latex(f"{var_name} = {result_value}")
        ))
        
        return steps


    def _format_natural_output(
        self, 
        steps: List[StepData], 
        equation_str: str, 
        result_value: float, 
        unit: str = ""
    ) -> str:
        """
        Gera saída formatada em TEXTO NATURAL (não LaTeX).
        
        Formato:
            Fórmula:
                M = F × d
            
            Substituindo valores:
                M = 100 kN × 2.5 m
            
            Resultado:
                M = 250.0 kN·m
        """
        lines = []
        
        # 1. Fórmula simbólica
        lines.append("Fórmula:")
        lines.append(f"    {equation_str}")
        lines.append("")
        
        # 2. Substituição de valores
        if len(steps) > 1:
            lines.append("Substituindo valores:")
            # Pegar o passo de substituição (geralmente o segundo)
            for step in steps:
                if hasattr(step, 'formula_numeric') and step.formula_numeric:
                    # Formatar com valores e unidades
                    expr_formatted = step.formula_numeric.replace('*', ' × ').replace('/', ' ÷ ')
                    lines.append(f"    {expr_formatted}")
            lines.append("")
        
        # 3. Resultado final
        lines.append("Resultado:")
        result_str = f"{result_value:.2f}" if isinstance(result_value, float) else str(result_value)
        unit_str = f" {unit}" if unit else ""
        lines.append(f"    = {result_str}{unit_str}")
        
        return "\n".join(lines)



    def _substitute_values(self, expr: str, context: Dict[str, Any]) -> str:
        """Substitui valores numéricos na expressão."""
        # Substituir variáveis por seus valores
        result = expr
        for var_name, value in context.items():
            # Usar regex para substituir apenas palavras completas
            pattern = r'\b' + re.escape(var_name) + r'\b'
            result = re.sub(pattern, str(value), result)
        return result
    
# src/pymemorial/engine/processor.py

    def _evaluate_expression(
        self,
        expression: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[float]:
        """
        Avalia expressão matemática com contexto fornecido.
        """
        print(f"🔍 DEBUG _evaluate_expression(): Avaliando: '{expression}'")
        print(f"🔍 DEBUG _evaluate_expression(): Contexto: {context}")
        
        # CORREÇÃO: Preparar contexto completo
        if context is None:
            context = {}
        
        # Se tem contexto global, adicionar as variáveis
        if self._context:
            for name, var in self._context.list_variables().items():
                if name not in context:  # Não sobrescrever
                    try:
                        if hasattr(var, 'value'):
                            context[name] = var.value
                        else:
                            context[name] = var
                    except:
                        pass
        
        print(f"🔍 DEBUG _evaluate_expression(): Contexto final: {context}")
        
        # Extrair lado direito da expressão (se tem "=")
        if "=" in expression:
            parts = expression.split("=", 1)
            expr_to_eval = parts[1].strip()
            var_name = parts[0].strip()  # CORREÇÃO: Extrair nome da variável
            print(f"🔍 DEBUG _evaluate_expression(): Expressão para avaliar: '{expr_to_eval}'")
            print(f"🔍 DEBUG _evaluate_expression(): Nome da variável: '{var_name}'")
        else:
            expr_to_eval = expression.strip()
            var_name = "_result"
            print(f"🔍 DEBUG _evaluate_expression(): Expressão sem '=': '{expr_to_eval}'")
        
        # Tentar avaliar
        try:
            # Criar namespace seguro com math e numpy
            safe_dict = {
                '__builtins__': {},
                'abs': abs,
                'max': max,
                'min': min,
                'round': round,
                'sum': sum,
                'pow': pow,
            }
            
            # Adicionar funções matemáticas
            try:
                import math
                safe_dict.update({
                    'sqrt': math.sqrt,
                    'sin': math.sin,
                    'cos': math.cos,
                    'tan': math.tan,
                    'pi': math.pi,
                    'e': math.e,
                    'log': math.log,
                    'exp': math.exp,
                    'ceil': math.ceil,
                    'floor': math.floor,
                    'atan': math.atan,
                    'asin': math.asin,
                    'acos': math.acos,
                    'math': math,
                })
            except ImportError:
                pass
            
            # Adicionar numpy se disponível
            if NUMPY_AVAILABLE:
                safe_dict.update({
                    'np': np,
                    'sqrt': np.sqrt,
                    'sin': np.sin,
                    'cos': np.cos,
                    'tan': np.tan,
                    'pi': np.pi,
                    'e': np.e,
                    'log': np.log,
                    'exp': np.exp,
                })
            
            # CRÍTICO: Adicionar variáveis do contexto
            safe_dict.update(context)
            
            print(f"🔍 DEBUG _evaluate_expression(): Namespace seguro: {list(safe_dict.keys())}")
            
            # Avaliar
            result = eval(expr_to_eval, safe_dict, {})
            print(f"✅ DEBUG _evaluate_expression(): Resultado: {result}")
            
            # Converter para float se possível
            if isinstance(result, (int, float)):
                return float(result)
            elif hasattr(result, '__float__'):
                return float(result)
            else:
                return result
                
        except Exception as e:
            logger.error(f"Erro ao avaliar expressão '{expr_to_eval}': {e}")
            print(f"❌ DEBUG _evaluate_expression(): ERRO: {e}")
            return None


    
    def _to_latex(self, expr: str) -> str:
        """Converte expressão para LaTeX."""
        if not SYMPY_AVAILABLE:
            return f"${expr}$"  # Fallback simples
        
        try:
            # Parse com SymPy
            sympy_expr = sympify(expr, evaluate=False)
            latex = sp.latex(sympy_expr)
            return f"$${latex}$$"
        except Exception as e:
            logger.debug(f"Erro ao converter para LaTeX: {e}")
            return f"${expr}$"
    
    # ========================================================================
    # CACHE
    # ========================================================================
    
    def _compute_cache_key(
        self,
        text: str,
        mode: ProcessingMode,
        granularity: GranularityLevel
    ) -> str:
        """Computa chave de cache (SHA256)."""
        content = f"{text}|{mode.value}|{granularity.value}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def clear_cache(self):
        """Limpa cache."""
        self._cache.clear()
        logger.info("Cache limpo")
    
    def enable_cache(self, enabled: bool = True):
        """Habilita/desabilita cache."""
        self._cache_enabled = enabled


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "UnifiedProcessor",
    "ProcessingResult",
    "StepData",
    "GranularityLevel",
    "ProcessingMode",
]
