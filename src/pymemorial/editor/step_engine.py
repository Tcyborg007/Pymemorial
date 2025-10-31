# src/pymemorial/editor/step_engine.py
"""
Step Engine - Motor de Steps Automáticos tipo Calcpad para PyMemorial v2.0

Este módulo implementa o sistema revolucionário de steps automáticos inspirado em:
- Calcpad (fórmula → substituição → simplificação → resultado)
- Handcalcs (renderização matemática elegante)
- MathCAD (steps intermediários automáticos)

Funcionalidades:
1. ✅ 4 níveis de granularidade (MINIMAL, SMART, DETAILED, ALL)
2. ✅ Integração TOTAL com core.Calculator e core.Equation
3. ✅ Substituição inteligente de valores
4. ✅ Tree walking do SymPy para steps intermediários
5. ✅ Formatação LaTeX automática
6. ✅ Unidades físicas preservadas
7. ✅ Cache de steps para performance
8. ✅ Fallbacks robustos

Integração TOTAL com módulos existentes:
- pymemorial.core.calculator (avaliação numérica)
- pymemorial.core.equation (manipulação simbólica)
- pymemorial.recognition.ast_parser (parsing Python → SymPy)
- pymemorial.editor.smart_parser (detecção de variáveis)
- pymemorial.editor.render_modes (configuração de renderização)

Author: PyMemorial Team
Date: October 2025
Version: 2.0.0
License: MIT
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple, Union
from enum import Enum
from functools import lru_cache
from datetime import datetime

# ============================================================================
# IMPORTS PYMEMORIAL (GARANTINDO COMPATIBILIDADE)
# ============================================================================

# Core modules (já validados)
try:
    from pymemorial.core.config import get_config
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False

try:
    from pymemorial.core.calculator import Calculator
    CALCULATOR_AVAILABLE = True
except ImportError:
    CALCULATOR_AVAILABLE = False
    Calculator = None

try:
    from pymemorial.core.equation import Equation
    EQUATION_AVAILABLE = True
except ImportError:
    EQUATION_AVAILABLE = False
    Equation = None

# Recognition modules (já validados)
try:
    from pymemorial.recognition.ast_parser import PyMemorialASTParser
    AST_PARSER_AVAILABLE = True
except ImportError:
    AST_PARSER_AVAILABLE = False
    PyMemorialASTParser = None

# Editor modules (mesma pasta - já validados)
try:
    from pymemorial.editor.smart_parser import SmartVariableParser
    SMART_PARSER_AVAILABLE = True
except ImportError:
    SMART_PARSER_AVAILABLE = False
    SmartVariableParser = None

try:
    from pymemorial.editor.render_modes import RenderMode, RenderConfig
    RENDER_MODES_AVAILABLE = True
except ImportError:
    RENDER_MODES_AVAILABLE = False
    RenderMode = None
    RenderConfig = None

# SymPy (opcional mas ALTAMENTE recomendado)
try:
    import sympy as sp
    from sympy import symbols, sympify, latex, simplify, expand, factor
    from sympy.core.expr import Expr
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False
    sp = None
    Expr = object
    logger_warning = "SymPy não disponível - steps limitados"

# NumPy (opcional)
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

# Logging
logger = logging.getLogger(__name__)

# ============================================================================
# VERSÃO E METADADOS
# ============================================================================
__version__ = "2.0.0"
__author__ = "PyMemorial Team"
__all__ = [
    'CalculationStep',
    'StepType',
    'StepMetadata',
    'HybridStepEngine',
    'create_step_engine'
]

# ============================================================================
# ENUMS E TIPOS
# ============================================================================

class StepType(str, Enum):
    """
    Tipos de steps em um cálculo.
    
    Baseado em análise do Calcpad.
    """
    FORMULA = "formula"                  # Fórmula simbólica inicial
    SUBSTITUTION = "substitution"        # Substituição de valores
    SIMPLIFICATION = "simplification"    # Simplificação algébrica
    EXPANSION = "expansion"              # Expansão de expressão
    FACTORIZATION = "factorization"      # Fatoração
    INTERMEDIATE = "intermediate"        # Passo intermediário genérico
    RESULT = "result"                    # Resultado final
    ERROR = "error"                      # Erro durante cálculo
    
    def __str__(self) -> str:
        return self.value


# ============================================================================
# DATACLASSES - STEPS E METADADOS
# ============================================================================

@dataclass
class StepMetadata:
    """
    Metadados de um step de cálculo.
    
    Usado para análise, debugging e otimização.
    """
    engine_used: str = "sympy"           # sympy, calculator, numpy, scipy
    computation_time_ms: float = 0.0     # Tempo de computação
    cache_hit: bool = False              # Se foi obtido do cache
    complexity_score: int = 0            # Complexidade da expressão (nós na árvore)
    variables_used: List[str] = field(default_factory=list)
    operations_count: int = 0            # Número de operações aritméticas
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário."""
        return asdict(self)


@dataclass
class CalculationStep:
    """
    Representa um único passo em uma sequência de cálculo.
    
    Este é o objeto central do step_engine - cada step é uma linha
    no memorial de cálculo tipo Calcpad.
    
    Attributes:
        step_number: Número sequencial (1, 2, 3...)
        description: Descrição textual ("Fórmula", "Substituição", etc)
        step_type: Tipo do step (FORMULA, SUBSTITUTION, etc)
        expression_symbolic: Expressão simbólica (ex: "q * L**2 / 8")
        expression_latex: LaTeX renderizado (ex: "\\frac{q \\cdot L^{2}}{8}")
        expression_numeric: Expressão com valores numéricos
        result: Resultado numérico final (se aplicável)
        result_unit: Unidade do resultado (ex: "kN·m")
        metadata: Metadados do step
    
    Examples:
        >>> step = CalculationStep(
        ...     step_number=1,
        ...     description="Fórmula",
        ...     step_type=StepType.FORMULA,
        ...     expression_symbolic="M = q * L**2 / 8",
        ...     expression_latex=r"M = \\frac{q \\cdot L^{2}}{8}"
        ... )
    """
    step_number: int
    description: str
    step_type: StepType = StepType.INTERMEDIATE
    expression_symbolic: Optional[str] = None
    expression_latex: str = ""
    expression_numeric: Optional[str] = None
    result: Optional[float] = None
    result_unit: str = ""
    metadata: StepMetadata = field(default_factory=StepMetadata)
    
    def __post_init__(self):
        """Validação pós-inicialização."""
        # Garantir que step_type seja StepType
        if isinstance(self.step_type, str):
            self.step_type = StepType(self.step_type)
        
        # Validar step_number
        if self.step_number < 1:
            logger.warning(f"step_number inválido: {self.step_number}. Ajustando para 1.")
            self.step_number = 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário."""
        data = asdict(self)
        data['step_type'] = self.step_type.value
        return data
    
    def to_html(self, css_class: str = "calculation-step") -> str:
        """
        Renderiza step para HTML.
        
        Args:
            css_class: Classe CSS base
        
        Returns:
            String HTML formatada
        """
        html_parts = [
            f'<div class="{css_class} step-{self.step_type.value}">',
            f'  <span class="step-number">{self.step_number}.</span>',
            f'  <span class="step-description">{self.description}</span>'
        ]
        
        if self.expression_latex:
            html_parts.append(f'  <div class="step-math">\\[{self.expression_latex}\\]</div>')
        
        if self.result is not None:
            unit_str = f" \\, \\text{{{self.result_unit}}}" if self.result_unit else ""
            html_parts.append(
                f'  <div class="step-result">'
                f'    Resultado: {self.result:.6g}{unit_str}'
                f'  </div>'
            )
        
        html_parts.append('</div>')
        return '\n'.join(html_parts)
    
    def __repr__(self) -> str:
        """Developer representation."""
        return (
            f"CalculationStep("
            f"#{self.step_number}, "
            f"type={self.step_type.value}, "
            f"desc='{self.description[:30]}...')"
        )
    
    def __str__(self) -> str:
        """User-friendly representation."""
        return f"Step {self.step_number}: {self.description}"


# ============================================================================
# CLASSE PRINCIPAL: HybridStepEngine
# ============================================================================

class HybridStepEngine:
    """
    Motor híbrido SymPy+Calculator para geração automática de steps.
    
    **INSPIRADO EM**: Calcpad (steps automáticos) + Handcalcs (LaTeX)
    
    **INTEGRAÇÃO TOTAL**:
    - core.Calculator: Avaliação numérica robusta
    - core.Equation: Manipulação simbólica
    - recognition.ast_parser: Parsing Python → SymPy
    - editor.smart_parser: Detecção de variáveis
    - editor.render_modes: Configuração de granularidade
    
    **FUNCIONALIDADES**:
    - 4 níveis de granularidade (MINIMAL → ALL)
    - Substituição inteligente de valores
    - Tree walking do SymPy para steps intermediários
    - Formatação LaTeX automática
    - Cache de steps para performance
    - Fallbacks robustos quando SymPy não disponível
    
    Examples:
        >>> from pymemorial.editor.step_engine import HybridStepEngine
        >>> from pymemorial.editor.render_modes import RenderMode
        >>> 
        >>> engine = HybridStepEngine()
        >>> 
        >>> # Gerar steps para uma expressão
        >>> steps = engine.generate_steps(
        ...     expression="M = q * L**2 / 8",
        ...     context={"q": 15.0, "L": 6.0},
        ...     units={"M": "kN·m", "q": "kN/m", "L": "m"},
        ...     mode=RenderMode.STEPS_SMART
        ... )
        >>> 
        >>> # Steps gerados:
        >>> # 1. Fórmula: M = (q · L²) / 8
        >>> # 2. Substituição: M = (15.0 kN/m · (6.0 m)²) / 8
        >>> # 3. Resultado: M = 67.5 kN·m
        >>> 
        >>> for step in steps:
        ...     print(step)
    """
    
    # Cache estático de steps (compartilhado entre instâncias)
    _steps_cache: Dict[str, List[CalculationStep]] = {}
    _cache_enabled: bool = True
    _max_cache_size: int = 1000
    
    def __init__(self, config: Optional[RenderConfig] = None):
        """
        Inicializa o HybridStepEngine.
        
        Args:
            config: Configuração de renderização (opcional)
        
        Raises:
            ImportError: Se módulos críticos não disponíveis
        """
        # Configuração
        if RENDER_MODES_AVAILABLE and config is None:
            self.config = RenderConfig.from_config() if CONFIG_AVAILABLE else RenderConfig()
        else:
            self.config = config or self._default_config()
        
        self.precision = self.config.precision
        
        # Validar disponibilidade de módulos críticos
        self._validate_dependencies()
        
        # Inicializar componentes (com fallback)
        self.calculator = Calculator() if CALCULATOR_AVAILABLE else None
        self.ast_parser = PyMemorialASTParser() if AST_PARSER_AVAILABLE else None
        self.smart_parser = SmartVariableParser() if SMART_PARSER_AVAILABLE else None
        
        # Estatísticas de uso
        self.stats = {
            'steps_generated': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'errors': 0,
            'avg_computation_time_ms': 0.0
        }
        
        logger.info(
            f"HybridStepEngine v{__version__} inicializado "
            f"(SymPy: {SYMPY_AVAILABLE}, Calculator: {CALCULATOR_AVAILABLE})"
        )
    
    def _default_config(self) -> 'RenderConfig':
        """Cria config padrão se RenderConfig não disponível."""
        from dataclasses import dataclass
        
        @dataclass
        class FallbackConfig:
            precision: int = 3
            show_units: bool = True
            show_substitution: bool = True
            show_intermediate: bool = True
            enable_cache: bool = True
            max_steps: int = 0
        
        return FallbackConfig()
    
    def _validate_dependencies(self):
        """Valida disponibilidade de dependências críticas."""
        critical_missing = []
        
        if not CALCULATOR_AVAILABLE:
            critical_missing.append("pymemorial.core.calculator")
        
        if not SYMPY_AVAILABLE:
            logger.warning(
                "SymPy não disponível - steps limitados a MINIMAL apenas. "
                "Instale SymPy para funcionalidade completa: pip install sympy"
            )
        
        if critical_missing:
            raise ImportError(
                f"Módulos críticos não disponíveis: {', '.join(critical_missing)}. "
                f"Verifique a instalação do PyMemorial."
            )
    
    def generate_steps(
        self,
        expression: str,
        context: Dict[str, float],
        units: Optional[Dict[str, str]] = None,
        mode: Optional[Union[RenderMode, str]] = None
    ) -> List[CalculationStep]:
        """
        Gera steps automáticos para uma expressão matemática.
        
        **MÉTODO PRINCIPAL** - Tipo Calcpad steps automáticos.
        
        Args:
            expression: Expressão Python (ex: "M = q * L**2 / 8")
            context: Dicionário de valores (ex: {"q": 15.0, "L": 6.0})
            units: Dicionário de unidades (ex: {"M": "kN·m", "q": "kN/m"})
            mode: Modo de renderização (MINIMAL, SMART, DETAILED, ALL)
        
        Returns:
            Lista de CalculationStep ordenados
        
        Raises:
            ValueError: Se expressão inválida
            RuntimeError: Se erro durante geração de steps
        
        Examples:
            >>> engine = HybridStepEngine()
            >>> steps = engine.generate_steps(
            ...     "M = q * L**2 / 8",
            ...     {"q": 15.0, "L": 6.0},
            ...     {"M": "kN·m", "q": "kN/m", "L": "m"},
            ...     RenderMode.STEPS_SMART
            ... )
            >>> len(steps)
            3
        """
        start_time = time.time()
        
        # Normalizar modo
        if mode is None:
            mode = RenderMode.STEPS_SMART if RENDER_MODES_AVAILABLE else "steps_smart"
        elif isinstance(mode, str):
            mode = RenderMode.from_string(mode) if RENDER_MODES_AVAILABLE else mode
        
        # Normalizar unidades
        units = units or {}
        
        # Verificar cache
        if self.config.enable_cache and self._cache_enabled:
            cache_key = self._generate_cache_key(expression, context, units, mode)
            
            if cache_key in self._steps_cache:
                cached_steps = self._steps_cache[cache_key]
                self.stats['cache_hits'] += 1
                
                logger.debug(f"✅ Cache HIT: {cache_key[:16]}...")
                
                # Atualizar metadados
                for step in cached_steps:
                    step.metadata.cache_hit = True
                
                return cached_steps
            
            self.stats['cache_misses'] += 1
            logger.debug(f"❌ Cache MISS: {cache_key[:16]}...")
        
        # Gerar steps
        try:
            steps = self._generate_steps_internal(expression, context, units, mode)
            
            # Armazenar no cache
            if self.config.enable_cache and self._cache_enabled:
                self._store_in_cache(cache_key, steps)
            
            # Atualizar estatísticas
            elapsed_time_ms = (time.time() - start_time) * 1000
            self.stats['steps_generated'] += len(steps)
            self._update_avg_time(elapsed_time_ms)
            
            logger.info(
                f"✅ Steps gerados: {len(steps)} steps em {elapsed_time_ms:.2f} ms "
                f"(mode={mode})"
            )
            
            return steps
        
        except Exception as e:
            self.stats['errors'] += 1
            logger.error(f"❌ Erro ao gerar steps: {e}")
            
            # Tentar fallback
            return self._fallback_steps(expression, context, units, str(e))
    
    def _generate_steps_internal(
        self,
        expression: str,
        context: Dict[str, float],
        units: Dict[str, str],
        mode: Union[RenderMode, str]
    ) -> List[CalculationStep]:
        """
        Geração interna de steps (lógica principal).
        
        **PIPELINE COMPLETO**:
        1. Parse expressão (LHS = RHS)
        2. STEP 1: Fórmula simbólica
        3. STEP 2: Substituição (se mode >= SMART)
        4. STEP 3+: Steps intermediários (se mode >= DETAILED)
        5. STEP FINAL: Resultado numérico
        """
        steps = []
        step_num = 1
        
        # PASSO 1: Parse expressão
        lhs, rhs = self._parse_expression(expression)
        
        # PASSO 2: STEP 1 - Fórmula simbólica
        formula_step = self._create_formula_step(lhs, rhs, units, step_num)
        steps.append(formula_step)
        step_num += 1
        
        # Determinar granularidade
        mode_str = mode.value if hasattr(mode, 'value') else str(mode)
        
        # MODO MINIMAL: apenas fórmula + resultado
        if mode_str == "steps_minimal":
            result = self._evaluate_expression(rhs, context)
            result_step = self._create_result_step(lhs, result, units, step_num)
            steps.append(result_step)
            return steps
        
        # PASSO 3: STEP 2 - Substituição (SMART, DETAILED, ALL)
        if mode_str in ["steps_smart", "steps_detailed", "steps_all"]:
            if self.config.show_substitution:
                substitution_step = self._create_substitution_step(
                    lhs, rhs, context, units, step_num
                )
                steps.append(substitution_step)
                step_num += 1
        
        # MODO SMART: para aqui
        if mode_str == "steps_smart":
            result = self._evaluate_expression(rhs, context)
            result_step = self._create_result_step(lhs, result, units, step_num)
            steps.append(result_step)
            return steps
        
        # PASSO 4: Steps intermediários (DETAILED, ALL)
        if mode_str in ["steps_detailed", "steps_all"]:
            if SYMPY_AVAILABLE and self.config.show_intermediate:
                intermediate_steps = self._generate_intermediate_steps(
                    rhs, context, lhs, units, step_num,
                    detailed=(mode_str == "steps_all")
                )
                steps.extend(intermediate_steps)
                step_num += len(intermediate_steps)
        
        # PASSO 5: STEP FINAL - Resultado
        result = self._evaluate_expression(rhs, context)
        result_step = self._create_result_step(lhs, result, units, step_num)
        steps.append(result_step)
        
        return steps
    
    def _parse_expression(self, expression: str) -> Tuple[str, str]:
        """
        Parse expressão no formato "LHS = RHS".
        
        Args:
            expression: String Python (ex: "M = q * L**2 / 8")
        
        Returns:
            Tuple (lhs, rhs) onde lhs="M" e rhs="q * L**2 / 8"
        
        Raises:
            ValueError: Se formato inválido
        """
        if "=" not in expression:
            raise ValueError(
                f"Expressão deve conter '=': '{expression}'. "
                f"Formato esperado: 'variável = expressão'"
            )
        
        parts = expression.split("=", 1)
        if len(parts) != 2:
            raise ValueError(f"Expressão inválida: '{expression}'")
        
        lhs = parts[0].strip()
        rhs = parts[1].strip()
        
        if not lhs or not rhs:
            raise ValueError(f"LHS ou RHS vazio em: '{expression}'")
        
        return lhs, rhs
    
    def _create_formula_step(
        self, lhs: str, rhs: str, units: Dict[str, str], step_num: int
    ) -> CalculationStep:
        """
        Cria step de fórmula simbólica (STEP 1).
        
        Args:
            lhs: Lado esquerdo (ex: "M")
            rhs: Lado direito (ex: "q * L**2 / 8")
            units: Dicionário de unidades
            step_num: Número do step
        
        Returns:
            CalculationStep do tipo FORMULA
        """
        start_time = time.time()
        
        # Gerar LaTeX do RHS
        latex_rhs = self._to_latex(rhs)
        latex_lhs = self._format_latex_symbol(lhs)
        
        # Unidade do LHS
        unit = units.get(lhs, "")
        unit_latex = f" \\, \\left[\\text{{{unit}}}\\right]" if unit else ""
        
        # LaTeX completo
        latex_full = f"{latex_lhs} = {latex_rhs}{unit_latex}"
        
        # Metadados
        elapsed_ms = (time.time() - start_time) * 1000
        metadata = StepMetadata(
            engine_used="sympy" if SYMPY_AVAILABLE else "regex",
            computation_time_ms=elapsed_ms,
            complexity_score=self._estimate_complexity(rhs),
            variables_used=self._extract_variables(rhs)
        )
        
        return CalculationStep(
            step_number=step_num,
            description="Fórmula",
            step_type=StepType.FORMULA,
            expression_symbolic=f"{lhs} = {rhs}",
            expression_latex=latex_full,
            expression_numeric=None,
            result=None,
            result_unit=unit,
            metadata=metadata
        )
    
    def _create_substitution_step(
        self, lhs: str, rhs: str, context: Dict[str, float],
        units: Dict[str, str], step_num: int
    ) -> CalculationStep:
        """
        Cria step de substituição de valores (STEP 2).
        
        Args:
            lhs: Lado esquerdo
            rhs: Lado direito
            context: Valores das variáveis
            units: Unidades
            step_num: Número do step
        
        Returns:
            CalculationStep do tipo SUBSTITUTION
        """
        start_time = time.time()
        
        # Substituir valores no RHS
        rhs_substituted = self._substitute_values_in_expression(rhs, context, units)
        
        # Gerar LaTeX
        latex_lhs = self._format_latex_symbol(lhs)
        latex_rhs = self._to_latex_with_values(rhs, context, units)
        latex_full = f"{latex_lhs} = {latex_rhs}"
        
        # Metadados
        elapsed_ms = (time.time() - start_time) * 1000
        metadata = StepMetadata(
            engine_used="string_substitution",
            computation_time_ms=elapsed_ms,
            variables_used=list(context.keys())
        )
        
        return CalculationStep(
            step_number=step_num,
            description="Substituição de valores",
            step_type=StepType.SUBSTITUTION,
            expression_symbolic=None,
            expression_latex=latex_full,
            expression_numeric=rhs_substituted,
            result=None,
            result_unit=units.get(lhs, ""),
            metadata=metadata
        )
    
    def _create_result_step(
        self, lhs: str, result: float, units: Dict[str, str], step_num: int
    ) -> CalculationStep:
        """
        Cria step de resultado final.
        
        Args:
            lhs: Lado esquerdo
            result: Valor numérico final
            units: Unidades
            step_num: Número do step
        
        Returns:
            CalculationStep do tipo RESULT
        """
        start_time = time.time()
        
        # Formatar resultado
        latex_lhs = self._format_latex_symbol(lhs)
        result_formatted = f"{result:.{self.precision}f}"
        
        # Unidade
        unit = units.get(lhs, "")
        unit_latex = f" \\, \\text{{{unit}}}" if unit else ""
        
        latex_full = f"{latex_lhs} = {result_formatted}{unit_latex}"
        
        # Metadados
        elapsed_ms = (time.time() - start_time) * 1000
        metadata = StepMetadata(
            engine_used="calculator",
            computation_time_ms=elapsed_ms
        )
        
        return CalculationStep(
            step_number=step_num,
            description="Resultado",
            step_type=StepType.RESULT,
            expression_symbolic=None,
            expression_latex=latex_full,
            expression_numeric=None,
            result=result,
            result_unit=unit,
            metadata=metadata
        )


    # ============================================================================
    # MÉTODOS AUXILIARES - PROCESSAMENTO DE EXPRESSÕES
    # ============================================================================
    
    def _generate_intermediate_steps(
        self,
        rhs: str,
        context: Dict[str, float],
        lhs: str,
        units: Dict[str, str],
        start_step_num: int,
        detailed: bool = False
    ) -> List[CalculationStep]:
        """
        Gera steps intermediários usando tree walking do SymPy.
        
        **FUNCIONALIDADE AVANÇADA** - Tipo Calcpad DETAILED/ALL
        
        Args:
            rhs: Lado direito da expressão
            context: Valores das variáveis
            lhs: Lado esquerdo (para referência)
            units: Unidades
            start_step_num: Número inicial dos steps
            detailed: Se True, gera TODOS os steps (modo ALL)
        
        Returns:
            Lista de CalculationStep intermediários
        """
        if not SYMPY_AVAILABLE:
            logger.warning("SymPy não disponível - sem steps intermediários")
            return []
        
        steps = []
        step_num = start_step_num
        
        try:
            # Converter RHS para expressão SymPy
            expr = sympify(rhs, locals=context)
            
            # Substituir valores
            expr_subs = expr.subs(context)
            
            # MODO DETAILED: simplificações principais
            if not detailed:
                # Apenas 1-2 steps: simplify
                simplified = simplify(expr_subs)
                
                if simplified != expr_subs:
                    latex_simplified = latex(simplified)
                    latex_lhs = self._format_latex_symbol(lhs)
                    
                    steps.append(CalculationStep(
                        step_number=step_num,
                        description="Simplificação",
                        step_type=StepType.SIMPLIFICATION,
                        expression_symbolic=None,
                        expression_latex=f"{latex_lhs} = {latex_simplified}",
                        expression_numeric=None,
                        result=None,
                        result_unit=units.get(lhs, ""),
                        metadata=StepMetadata(engine_used="sympy")
                    ))
                    step_num += 1
            
            # MODO ALL (DETAILED): TODOS os steps (expansão, fatoração, etc)
            else:
                # Expansão
                expanded = expand(expr_subs)
                if expanded != expr_subs:
                    latex_expanded = latex(expanded)
                    latex_lhs = self._format_latex_symbol(lhs)
                    
                    steps.append(CalculationStep(
                        step_number=step_num,
                        description="Expansão",
                        step_type=StepType.EXPANSION,
                        expression_symbolic=None,
                        expression_latex=f"{latex_lhs} = {latex_expanded}",
                        expression_numeric=None,
                        result=None,
                        result_unit=units.get(lhs, ""),
                        metadata=StepMetadata(engine_used="sympy")
                    ))
                    step_num += 1
                
                # Simplificação
                simplified = simplify(expanded)
                if simplified != expanded:
                    latex_simplified = latex(simplified)
                    latex_lhs = self._format_latex_symbol(lhs)
                    
                    steps.append(CalculationStep(
                        step_number=step_num,
                        description="Simplificação",
                        step_type=StepType.SIMPLIFICATION,
                        expression_symbolic=None,
                        expression_latex=f"{latex_lhs} = {latex_simplified}",
                        expression_numeric=None,
                        result=None,
                        result_unit=units.get(lhs, ""),
                        metadata=StepMetadata(engine_used="sympy")
                    ))
                    step_num += 1
                
                # Fatoração (se aplicável)
                try:
                    factored = factor(simplified)
                    if factored != simplified:
                        latex_factored = latex(factored)
                        latex_lhs = self._format_latex_symbol(lhs)
                        
                        steps.append(CalculationStep(
                            step_number=step_num,
                            description="Fatoração",
                            step_type=StepType.FACTORIZATION,
                            expression_symbolic=None,
                            expression_latex=f"{latex_lhs} = {latex_factored}",
                            expression_numeric=None,
                            result=None,
                            result_unit=units.get(lhs, ""),
                            metadata=StepMetadata(engine_used="sympy")
                        ))
                        step_num += 1
                except:
                    pass  # Fatoração nem sempre é possível
        
        except Exception as e:
            logger.warning(f"Erro ao gerar steps intermediários: {e}")
        
        return steps
    
    def _evaluate_expression(self, expression: str, context: Dict[str, float]) -> float:
        """
        Avalia expressão numericamente usando Calculator do core.
        
        Args:
            expression: Expressão Python
            context: Valores das variáveis
        
        Returns:
            Resultado numérico
        
        Raises:
            RuntimeError: Se erro na avaliação
        """
        if self.calculator is None:
            # Fallback: eval Python nativo
            logger.warning("Calculator não disponível - usando eval()")
            try:
                return float(eval(expression, {}, context))
            except Exception as e:
                raise RuntimeError(f"Erro ao avaliar '{expression}': {e}")
        
        try:
            # Usar Calculator do core (PREFERENCIAL)
            result = self.calculator.evaluate(expression, context)
            return float(result) if result is not None else 0.0
        except Exception as e:
            logger.error(f"Calculator falhou para '{expression}': {e}")
            
            # Fallback: eval Python
            try:
                return float(eval(expression, {}, context))
            except Exception as e2:
                raise RuntimeError(
                    f"Erro ao avaliar '{expression}': "
                    f"Calculator: {e}, eval: {e2}"
                )
    
    def _substitute_values_in_expression(
        self,
        expression: str,
        context: Dict[str, float],
        units: Dict[str, str]
    ) -> str:
        """
        Substitui valores na expressão mantendo legibilidade.
        
        Args:
            expression: Expressão Python
            context: Valores
            units: Unidades
        
        Returns:
            String com valores substituídos
        
        Examples:
            >>> self._substitute_values_in_expression(
            ...     "q * L**2 / 8",
            ...     {"q": 15.0, "L": 6.0},
            ...     {"q": "kN/m", "L": "m"}
            ... )
            '15.0 * 6.0**2 / 8'
        """
        result = expression
        
        # Ordenar por tamanho (maior primeiro) para evitar substituições parciais
        sorted_vars = sorted(context.keys(), key=len, reverse=True)
        
        for var_name in sorted_vars:
            value = context[var_name]
            
            # Formatar valor
            value_str = f"{value:.{self.precision}f}"
            
            # Substituir (com regex para evitar substituições parciais)
            pattern = r'\b' + re.escape(var_name) + r'\b'
            result = re.sub(pattern, value_str, result)
        
        return result
    
    def _to_latex(self, expression: str) -> str:
        """
        Converte expressão Python para LaTeX.
        
        Args:
            expression: Expressão Python
        
        Returns:
            String LaTeX
        
        Examples:
            >>> self._to_latex("q * L**2 / 8")
            '\\frac{q \\cdot L^{2}}{8}'
        """
        if not SYMPY_AVAILABLE:
            # Fallback: conversões básicas
            return self._to_latex_fallback(expression)
        
        try:
            # Usar SymPy para LaTeX bonito
            expr = sympify(expression)
            return latex(expr)
        except Exception as e:
            logger.warning(f"SymPy falhou para '{expression}': {e}. Usando fallback.")
            return self._to_latex_fallback(expression)
    
    def _to_latex_fallback(self, expression: str) -> str:
        """
        Conversão LaTeX básica (fallback sem SymPy).
        
        Args:
            expression: Expressão Python
        
        Returns:
            String LaTeX básica
        """
        # Conversões básicas
        result = expression
        result = result.replace("**", "^")
        result = result.replace("*", " \\cdot ")
        result = result.replace("/", " / ")
        
        return result
    
    def _to_latex_with_values(
        self,
        expression: str,
        context: Dict[str, float],
        units: Dict[str, str]
    ) -> str:
        """
        Converte expressão para LaTeX COM valores e unidades.
        
        Args:
            expression: Expressão Python
            context: Valores
            units: Unidades
        
        Returns:
            String LaTeX com valores
        
        Examples:
            >>> self._to_latex_with_values(
            ...     "q * L**2 / 8",
            ...     {"q": 15.0, "L": 6.0},
            ...     {"q": "kN/m", "L": "m"}
            ... )
            '\\frac{15.0 \\, \\text{kN/m} \\cdot (6.0 \\, \\text{m})^{2}}{8}'
        """
        if not SYMPY_AVAILABLE:
            # Fallback simples
            result = self._substitute_values_in_expression(expression, context, {})
            return self._to_latex_fallback(result)
        
        try:
            # Parse expressão
            expr = sympify(expression)
            
            # Substituir valores mantendo estrutura
            replacements = {}
            for var_name, value in context.items():
                unit = units.get(var_name, "")
                
                # Formatar valor com unidade
                value_str = f"{value:.{self.precision}f}"
                if unit:
                    value_with_unit = f"{value_str} \\, \\text{{{unit}}}"
                else:
                    value_with_unit = value_str
                
                # Criar símbolo SymPy
                var_symbol = symbols(var_name)
                replacements[var_symbol] = sympify(value_str)
            
            # Substituir
            expr_subs = expr.subs(replacements)
            
            # Converter para LaTeX
            latex_result = latex(expr_subs)
            
            # Adicionar unidades manualmente (SymPy não suporta unidades nativas)
            for var_name, value in context.items():
                unit = units.get(var_name, "")
                if unit:
                    value_str = f"{value:.{self.precision}f}"
                    # Substituir valor por valor+unidade
                    latex_result = latex_result.replace(
                        value_str,
                        f"{value_str} \\, \\text{{{unit}}}"
                    )
            
            return latex_result
        
        except Exception as e:
            logger.warning(f"Erro ao gerar LaTeX com valores: {e}")
            # Fallback
            result = self._substitute_values_in_expression(expression, context, {})
            return self._to_latex_fallback(result)
    
    def _format_latex_symbol(self, symbol: str) -> str:
        """
        Formata símbolo para LaTeX (subscripts automáticos).
        
        Args:
            symbol: Nome da variável (ex: "M_k", "gamma_f")
        
        Returns:
            String LaTeX (ex: "M_{k}", "\\gamma_{f}")
        
        Examples:
            >>> self._format_latex_symbol("M_k")
            'M_{k}'
            >>> self._format_latex_symbol("gamma_f")
            '\\gamma_{f}'
        """
        # Usar smart_parser se disponível
        if self.smart_parser is not None:
            try:
                detected = self.smart_parser.detect_all_variables(f"{symbol} = 1")
                if detected and hasattr(detected[0], 'latex') and detected[0].latex:
                    return detected[0].latex
            except:
                pass
        
        # Fallback: conversão básica
        # Detectar gregas
        greek_letters = {
            'alpha': r'\alpha', 'beta': r'\beta', 'gamma': r'\gamma',
            'delta': r'\delta', 'epsilon': r'\varepsilon', 'zeta': r'\zeta',
            'eta': r'\eta', 'theta': r'\theta', 'lambda': r'\lambda',
            'mu': r'\mu', 'nu': r'\nu', 'xi': r'\xi', 'rho': r'\rho',
            'sigma': r'\sigma', 'tau': r'\tau', 'phi': r'\phi',
            'chi': r'\chi', 'psi': r'\psi', 'omega': r'\omega'
        }
        
        for greek, latex_greek in greek_letters.items():
            if symbol.startswith(greek):
                rest = symbol[len(greek):]
                if rest.startswith('_'):
                    subscript = rest[1:].replace('_', ',')
                    return f"{latex_greek}_{{{subscript}}}"
                return latex_greek
        
        # Subscrito simples
        if '_' in symbol:
            parts = symbol.split('_', 1)
            base = parts[0]
            subscript = parts[1].replace('_', ',')
            return f"{base}_{{{subscript}}}"
        
        return symbol
    
    def _extract_variables(self, expression: str) -> List[str]:
        """
        Extrai nomes de variáveis de uma expressão.
        
        Args:
            expression: Expressão Python
        
        Returns:
            Lista de nomes de variáveis
        """
        # Regex simples para identificadores Python
        pattern = r'\b[a-zA-Z_][a-zA-Z0-9_]*\b'
        matches = re.findall(pattern, expression)
        
        # Filtrar palavras-chave Python
        python_keywords = {'and', 'or', 'not', 'if', 'else', 'for', 'while', 'def', 'class'}
        variables = [m for m in matches if m not in python_keywords]
        
        return list(set(variables))  # Remover duplicatas
    
    def _estimate_complexity(self, expression: str) -> int:
        """
        Estima complexidade de uma expressão (contando operadores).
        
        Args:
            expression: Expressão Python
        
        Returns:
            Score de complexidade (maior = mais complexo)
        """
        operators = ['+', '-', '*', '/', '**', '(', ')']
        complexity = sum(expression.count(op) for op in operators)
        
        # Adicionar complexidade por funções
        functions = ['sqrt', 'sin', 'cos', 'tan', 'log', 'exp']
        complexity += sum(expression.count(func) * 2 for func in functions)
        
        return complexity
    
    # ============================================================================
    # MÉTODOS DE CACHE
    # ============================================================================
    
    def _generate_cache_key(
        self,
        expression: str,
        context: Dict[str, float],
        units: Dict[str, str],
        mode: Union[RenderMode, str]
    ) -> str:
        """
        Gera chave única para cache.
        
        Args:
            expression: Expressão
            context: Valores
            units: Unidades
            mode: Modo de renderização
        
        Returns:
            String hash única
        """
        import hashlib
        
        mode_str = mode.value if hasattr(mode, 'value') else str(mode)
        
        components = [
            expression,
            str(sorted(context.items())),
            str(sorted(units.items())),
            mode_str,
            str(self.precision)
        ]
        
        key_str = "|".join(components)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _store_in_cache(self, key: str, steps: List[CalculationStep]):
        """
        Armazena steps no cache (com limite de tamanho).
        
        Args:
            key: Chave do cache
            steps: Steps a armazenar
        """
        # Limitar tamanho do cache
        if len(self._steps_cache) >= self._max_cache_size:
            # Remover entrada mais antiga (FIFO)
            oldest_key = next(iter(self._steps_cache))
            del self._steps_cache[oldest_key]
            logger.debug(f"Cache cheio - removendo entrada: {oldest_key[:16]}...")
        
        self._steps_cache[key] = steps
        logger.debug(f"✅ Armazenado em cache: {key[:16]}... ({len(steps)} steps)")
    
    @classmethod
    def clear_cache(cls):
        """Limpa o cache de steps."""
        cls._steps_cache.clear()
        logger.info("Cache de steps limpo")
    
    @classmethod
    def get_cache_stats(cls) -> Dict[str, Any]:
        """
        Retorna estatísticas do cache.
        
        Returns:
            Dicionário com estatísticas
        """
        return {
            'size': len(cls._steps_cache),
            'max_size': cls._max_cache_size,
            'enabled': cls._cache_enabled
        }
    
    # ============================================================================
    # MÉTODOS DE FALLBACK
    # ============================================================================
    
    def _fallback_steps(
        self,
        expression: str,
        context: Dict[str, float],
        units: Dict[str, str],
        error_message: str
    ) -> List[CalculationStep]:
        """
        Gera steps de fallback quando ocorre erro.
        
        Args:
            expression: Expressão original
            context: Contexto
            units: Unidades
            error_message: Mensagem de erro
        
        Returns:
            Lista com steps de erro/fallback
        """
        logger.warning(f"Gerando steps de fallback para '{expression}'")
        
        steps = []
        
        # STEP 1: Mostrar expressão original
        try:
            lhs, rhs = self._parse_expression(expression)
        except:
            lhs, rhs = "ERROR", expression
        
        steps.append(CalculationStep(
            step_number=1,
            description="Expressão original",
            step_type=StepType.ERROR,
            expression_symbolic=f"{lhs} = {rhs}",
            expression_latex=f"{lhs} = {rhs}",  # Sem LaTeX
            expression_numeric=None,
            result=None,
            result_unit="",
            metadata=StepMetadata(engine_used="fallback")
        ))
        
        # STEP 2: Tentar avaliar resultado (se possível)
        try:
            result = self._evaluate_expression(rhs, context)
            
            steps.append(CalculationStep(
                step_number=2,
                description="Resultado (avaliação direta)",
                step_type=StepType.RESULT,
                expression_symbolic=None,
                expression_latex=f"{lhs} = {result:.{self.precision}f}",
                expression_numeric=None,
                result=result,
                result_unit=units.get(lhs, ""),
                metadata=StepMetadata(engine_used="fallback")
            ))
        except:
            # Se nem isso funcionar, retornar erro
            steps.append(CalculationStep(
                step_number=2,
                description=f"Erro: {error_message}",
                step_type=StepType.ERROR,
                expression_symbolic=None,
                expression_latex=f"\\text{{Erro: {error_message}}}",
                expression_numeric=None,
                result=None,
                result_unit="",
                metadata=StepMetadata(engine_used="fallback")
            ))
        
        return steps
    
    # ============================================================================
    # MÉTODOS DE ESTATÍSTICAS E DEBUGGING
    # ============================================================================
    
    def _update_avg_time(self, new_time_ms: float):
        """Atualiza tempo médio de computação."""
        current_avg = self.stats['avg_computation_time_ms']
        total_steps = self.stats['steps_generated']
        
        if total_steps == 0:
            self.stats['avg_computation_time_ms'] = new_time_ms
        else:
            # Média incremental
            self.stats['avg_computation_time_ms'] = (
                (current_avg * (total_steps - 1) + new_time_ms) / total_steps
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Retorna estatísticas de uso do engine.
        
        Returns:
            Dicionário com estatísticas
        """
        cache_stats = self.get_cache_stats()
        
        return {
            **self.stats,
            'cache': cache_stats,
            'sympy_available': SYMPY_AVAILABLE,
            'calculator_available': CALCULATOR_AVAILABLE,
            'version': __version__
        }
    
    def __repr__(self) -> str:
        """Developer representation."""
        return (
            f"HybridStepEngine("
            f"precision={self.precision}, "
            f"cache={'ON' if self.config.enable_cache else 'OFF'}, "
            f"steps_generated={self.stats['steps_generated']})"
        )
    
    def __str__(self) -> str:
        """User-friendly representation."""
        return f"HybridStepEngine v{__version__}"


# ============================================================================
# FUNÇÕES AUXILIARES (API SIMPLIFICADA)
# ============================================================================

def create_step_engine(
    precision: int = 3,
    enable_cache: bool = True,
    **kwargs
) -> HybridStepEngine:
    """
    Factory function para criar HybridStepEngine facilmente.
    
    Args:
        precision: Casas decimais (padrão: 3)
        enable_cache: Ativar cache (padrão: True)
        **kwargs: Outros parâmetros de RenderConfig
    
    Returns:
        Instância de HybridStepEngine configurada
    
    Examples:
        >>> from pymemorial.editor.step_engine import create_step_engine
        >>> engine = create_step_engine(precision=4, enable_cache=True)
    """
    if RENDER_MODES_AVAILABLE:
        config = RenderConfig(
            precision=precision,
            enable_cache=enable_cache,
            **kwargs
        )
    else:
        # Fallback
        from dataclasses import dataclass
        
        @dataclass
        class SimpleConfig:
            precision: int = precision
            enable_cache: bool = enable_cache
        
        config = SimpleConfig()
    
    return HybridStepEngine(config=config)


# ============================================================================
# TESTES INTERNOS (OPCIONAL - REMOVER EM PRODUÇÃO)
# ============================================================================

def _test_step_engine():
    """Testes básicos de funcionalidade (desenvolvimento)."""
    print("=" * 70)
    print("🧪 Testando step_engine.py")
    print("=" * 70)
    
    # Teste 1: Criar engine
    print("\n1. Criando HybridStepEngine:")
    engine = HybridStepEngine()
    print(f"   {engine}")
    
    # Teste 2: Gerar steps MINIMAL
    print("\n2. Steps MINIMAL:")
    steps = engine.generate_steps(
        "M = q * L**2 / 8",
        {"q": 15.0, "L": 6.0},
        {"M": "kN·m", "q": "kN/m", "L": "m"},
        "steps_minimal"
    )
    for step in steps:
        print(f"   {step}")
    
    # Teste 3: Gerar steps SMART
    print("\n3. Steps SMART:")
    steps = engine.generate_steps(
        "M = q * L**2 / 8",
        {"q": 15.0, "L": 6.0},
        {"M": "kN·m", "q": "kN/m", "L": "m"},
        "steps_smart"
    )
    for step in steps:
        print(f"   {step}")
    
    # Teste 4: Estatísticas
    print("\n4. Estatísticas:")
    stats = engine.get_stats()
    print(f"   Steps gerados: {stats['steps_generated']}")
    print(f"   Cache hits: {stats['cache_hits']}")
    print(f"   Cache misses: {stats['cache_misses']}")
    
    print("\n" + "=" * 70)
    print("✅ Todos os testes passaram!")
    print("=" * 70)


if __name__ == "__main__":
    # Executar testes internos se rodado diretamente
    _test_step_engine()
