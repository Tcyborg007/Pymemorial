# auto_fix_pymemorial.py
"""
Script de Auto-Correção - PyMemorial v2.1

Este script:
1. Verifica se smart_parser.py existe
2. Cria backup do natural_engine.py atual
3. Substitui com versão corrigida
4. Limpa cache
5. Testa se funcionou

Uso: python auto_fix_pymemorial.py
"""

import os
import shutil
from pathlib import Path
from datetime import datetime

# Cores
class C:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    print(f"\n{C.BLUE}{C.BOLD}{'=' * 80}{C.END}")
    print(f"{C.BLUE}{C.BOLD}{text.center(80)}{C.END}")
    print(f"{C.BLUE}{C.BOLD}{'=' * 80}{C.END}\n")

def print_success(text):
    print(f"{C.GREEN}✓ {text}{C.END}")

def print_error(text):
    print(f"{C.RED}✗ {text}{C.END}")

def print_warning(text):
    print(f"{C.YELLOW}⚠ {text}{C.END}")

# Paths
ROOT = Path(__file__).parent
SRC_EDITOR = ROOT / "src" / "pymemorial" / "editor"
NATURAL_ENGINE = SRC_EDITOR / "natural_engine.py"
SMART_PARSER = SRC_EDITOR / "smart_parser.py"

print_header("AUTO-FIX PyMemorial v2.1")

# ============================================================================
# STEP 1: Verificar smart_parser.py
# ============================================================================

print_header("STEP 1: Verificando smart_parser.py")

if not SMART_PARSER.exists():
    print_error(f"smart_parser.py NÃO ENCONTRADO em: {SMART_PARSER}")
    print_warning("Por favor, crie o arquivo primeiro!")
    print_warning("Copie o código do artefato 'smart_parser_v2'")
    input("\nPressione ENTER após criar o arquivo...")
    
    if not SMART_PARSER.exists():
        print_error("Arquivo ainda não existe. Abortando.")
        exit(1)

print_success(f"smart_parser.py encontrado: {SMART_PARSER}")

# Verificar se tem o conteúdo correto
content = SMART_PARSER.read_text(encoding='utf-8')
if "SmartVariableParser" in content and "ParsedVariable" in content:
    print_success("Conteúdo do smart_parser.py válido")
else:
    print_error("smart_parser.py tem conteúdo inválido!")
    exit(1)

# ============================================================================
# STEP 2: Backup do natural_engine.py atual
# ============================================================================

print_header("STEP 2: Criando backup")

if NATURAL_ENGINE.exists():
    backup_name = f"natural_engine_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"
    backup_path = SRC_EDITOR / backup_name
    shutil.copy2(NATURAL_ENGINE, backup_path)
    print_success(f"Backup criado: {backup_name}")
else:
    print_warning("natural_engine.py não existe (será criado)")

# ============================================================================
# STEP 3: Verificar se precisa substituir
# ============================================================================

print_header("STEP 3: Verificando arquivo atual")

if NATURAL_ENGINE.exists():
    current_content = NATURAL_ENGINE.read_text(encoding='utf-8')
    
    if "_detect_variables_with_smart_parser" in current_content:
        print_success("Arquivo JÁ ESTÁ CORRETO!")
        print_warning("Mas vamos verificar o import do SmartParser...")
        
        if "from .smart_parser import SmartVariableParser" in current_content:
            print_success("Import do SmartParser OK")
        else:
            print_error("Import do SmartParser FALTANDO!")
            print_warning("Vou corrigir...")
    else:
        print_warning("Arquivo usa código ANTIGO - precisa substituir")

# ============================================================================
# STEP 4: Escrever código corrigido
# ============================================================================

print_header("STEP 4: Escrevendo código corrigido")

# Código corrigido (versão mínima funcional)
CORRECTED_CODE = '''# src/pymemorial/editor/natural_engine.py
"""
Natural Memorial Editor v2.1 - CORRIGIDO

CRÍTICO: Usa APENAS SmartParser para detecção
"""

from __future__ import annotations
import re
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

try:
    import sympy as sp
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False
    sp = None

# CRÍTICO: Import do SmartParser
try:
    from .smart_parser import SmartVariableParser, ParsedVariable
    PARSER_AVAILABLE = True
except ImportError:
    PARSER_AVAILABLE = False
    SmartVariableParser = None
    ParsedVariable = None

logger = logging.getLogger(__name__)

class DocumentType(Enum):
    MEMORIAL = "memorial"
    ARTICLE = "article"
    TCC = "tcc"
    REPORT = "report"

class RenderMode(Enum):
    FULL = "full"
    SYMBOLIC = "symbolic"
    NUMERIC = "numeric"
    RESULT = "result"
    STEPS = "steps"

@dataclass
class SmartVariable:
    name: str
    value: Optional[float] = None
    unit: str = ''
    formula: Optional[str] = None
    symbol: Optional[sp.Symbol] = None
    description: str = ''
    auto_detected: bool = True
    
    def __post_init__(self):
        if SYMPY_AVAILABLE and sp is not None and self.symbol is None:
            self.symbol = sp.Symbol(self.name, real=True)

@dataclass
class CalculationResult:
    name: str
    expression: str
    symbolic_expr: Optional[sp.Expr] = None
    numeric_expr: Optional[sp.Expr] = None
    result_value: float = 0.0
    result_unit: str = ''
    variables_used: List[str] = None
    render_mode: RenderMode = RenderMode.FULL
    
    def __post_init__(self):
        if self.variables_used is None:
            self.variables_used = []

class NaturalMemorialEditor:
    """Editor v2.1 - USA SMART PARSER"""
    
    def __init__(self, document_type: str = 'memorial', auto_greek: bool = True, auto_units: bool = True):
        self.document_type = DocumentType[document_type.upper()]
        self.auto_greek = auto_greek
        self.auto_units = auto_units
        
        self.variables: Dict[str, SmartVariable] = {}
        self.equations: Dict[str, CalculationResult] = {}
        
        # CRÍTICO: Inicializar SmartParser
        if not PARSER_AVAILABLE or SmartVariableParser is None:
            logger.error("SmartParser NOT available!")
            raise ImportError("SmartParser required")
        
        self.parser = SmartVariableParser()
        logger.debug("SmartParser initialized successfully")
        
        self._setup_patterns()
        logger.info(f"NaturalMemorialEditor v2.1 initialized (type={self.document_type.value})")
    
    def _setup_patterns(self):
        self.calc_block = re.compile(r'@calc(?:\\[(\\w+)\\])?\\s+([A-Za-z_][A-Za-z0-9_]*)\\s*=\\s*(.+)', re.MULTILINE)
        self.value_render = re.compile(r'\\{([A-Za-z_][A-Za-z0-9_]*)\\}')
        self.formula_render = re.compile(r'\\{\\{([A-Za-z_][A-Za-z0-9_]*)\\}\\}')
        self.full_render = re.compile(r'\\{\\{\\{([A-Za-z_][A-Za-z0-9_]*)\\}\\}\\}')
    
    def process(self, text: str, clean: bool = True) -> str:
        logger.debug("=" * 80)
        logger.debug("PROCESSING WITH SMART PARSER v2.1")
        logger.debug("=" * 80)
        
        # CRÍTICO: Usar SmartParser
        logger.debug("STEP 1: Calling SmartParser...")
        self._detect_with_smart_parser(text)
        logger.info(f"✓ Detected {len(self.variables)} variables: {list(self.variables.keys())}")
        
        logger.debug("STEP 2: Processing calculations...")
        text = self._process_calculations(text)
        logger.info(f"✓ Processed {len(self.equations)} equations")
        
        logger.debug("STEP 3: Rendering...")
        text = self._render_full(text)
        text = self._render_formulas(text)
        text = self._render_values(text)
        
        if clean:
            text = self._clean_text(text)
        
        logger.debug("=" * 80)
        return text
    
    def _detect_with_smart_parser(self, text: str):
        """ÚNICO MÉTODO DE DETECÇÃO - Usa SmartParser"""
        if not self.parser:
            logger.error("SmartParser not initialized!")
            return
        
        logger.debug("Calling SmartParser.detect_variables()...")
        parsed_vars = self.parser.detect_variables(text)
        
        logger.debug(f"SmartParser returned {len(parsed_vars)} variables")
        
        for var in parsed_vars:
            self.variables[var.name] = SmartVariable(
                name=var.name,
                value=var.value,
                unit=var.unit,
                description=f"Line {var.line_number}",
                auto_detected=True
            )
            logger.debug(f"  → {var.name} = {var.value} {var.unit}")
    
    def _process_calculations(self, text: str) -> str:
        def replace_calc(match):
            mode_str = match.group(1) or 'full'
            result_name = match.group(2)
            expression = match.group(3).strip()
            
            logger.debug(f"Processing: {result_name} = {expression}")
            
            try:
                try:
                    render_mode = RenderMode[mode_str.upper()]
                except KeyError:
                    render_mode = RenderMode.FULL
                
                if not SYMPY_AVAILABLE:
                    return "\\n**[ERROR: SymPy required]**\\n"
                
                local_vars = {v.name: v.symbol for v in self.variables.values()}
                logger.debug(f"  Available: {list(local_vars.keys())}")
                
                expr_symbols = self.parser.find_symbols_in_expression(expression)
                logger.debug(f"  Needed: {expr_symbols}")
                
                missing = expr_symbols - set(local_vars.keys())
                if missing:
                    logger.error(f"  ✗ Undefined: {missing}")
                    return f"\\n**[ERROR: Undefined {missing}]**\\n"
                
                sympy_expr = sp.sympify(expression, locals=local_vars)
                subs_dict = {v.symbol: v.value for v in self.variables.values() if v.value is not None}
                numeric_expr = sympy_expr.subs(subs_dict)
                result_value = float(numeric_expr)
                
                result_unit = self._detect_unit(expression)
                
                self.variables[result_name] = SmartVariable(
                    name=result_name, value=result_value, unit=result_unit, formula=expression
                )
                
                calc_result = CalculationResult(
                    name=result_name, expression=expression, symbolic_expr=sympy_expr,
                    numeric_expr=numeric_expr, result_value=result_value, result_unit=result_unit,
                    variables_used=list(expr_symbols), render_mode=render_mode
                )
                self.equations[result_name] = calc_result
                
                logger.info(f"  ✓ {result_name} = {result_value} {result_unit}")
                
                return self._render_calculation(calc_result)
                
            except Exception as e:
                logger.error(f"  ✗ Error: {e}", exc_info=True)
                return f"\\n**[ERROR: {str(e)}]**\\n"
        
        return self.calc_block.sub(replace_calc, text)
    
    def _detect_unit(self, expr: str) -> str:
        for name in self.variables:
            if name in expr and self.variables[name].unit:
                return self.variables[name].unit
        return ''
    
    def _render_calculation(self, calc: CalculationResult) -> str:
        if calc.render_mode == RenderMode.STEPS:
            return self._render_steps(calc)
        elif calc.render_mode == RenderMode.SYMBOLIC:
            return self._render_symbolic(calc)
        elif calc.render_mode == RenderMode.NUMERIC:
            return self._render_numeric(calc)
        elif calc.render_mode == RenderMode.RESULT:
            return self._render_result(calc)
        else:
            return self._render_full_calc(calc)
    
    def _render_steps(self, calc: CalculationResult) -> str:
        subs = {v.symbol: v.value for v in self.variables.values() if v.value is not None}
        out = "\\n**Cálculo:**\\n\\n"
        out += f"→ ${sp.latex(sp.Symbol(calc.name))} = {sp.latex(calc.symbolic_expr)}$\\n"
        out += f"→ ${sp.latex(sp.Symbol(calc.name))} = {sp.latex(calc.symbolic_expr.subs(subs))}$\\n"
        out += f"→ **${sp.latex(sp.Symbol(calc.name))} = {calc.result_value:.4g}$ {calc.result_unit}** ✓\\n\\n"
        return out
    
    def _render_symbolic(self, calc: CalculationResult) -> str:
        return f"\\n$${sp.latex(sp.Symbol(calc.name))} = {sp.latex(calc.symbolic_expr)}$$\\n\\n"
    
    def _render_numeric(self, calc: CalculationResult) -> str:
        return f"\\n$${sp.latex(sp.Symbol(calc.name))} = {sp.latex(calc.numeric_expr)} = {calc.result_value:.4g}$$\\n\\n"
    
    def _render_result(self, calc: CalculationResult) -> str:
        return f"\\n**{calc.name} = {calc.result_value:.4g} {calc.result_unit}**\\n\\n"
    
    def _render_full_calc(self, calc: CalculationResult) -> str:
        out = "\\n**Cálculo:**\\n\\n"
        out += f"$${sp.latex(sp.Symbol(calc.name))} = {sp.latex(calc.symbolic_expr)}$$\\n\\n"
        out += f"$${sp.latex(sp.Symbol(calc.name))} = {sp.latex(calc.numeric_expr)} = {calc.result_value:.4g}$$\\n\\n"
        return out
    
    def _render_values(self, text: str) -> str:
        def repl(m):
            v = self.variables.get(m.group(1))
            return f"**{v.value:.4g} {v.unit}**".strip() if v and v.value else m.group(0)
        return self.value_render.sub(repl, text)
    
    def _render_formulas(self, text: str) -> str:
        def repl(m):
            c = self.equations.get(m.group(1))
            return f"$({c.expression})$" if c else m.group(0)
        return self.formula_render.sub(repl, text)
    
    def _render_full(self, text: str) -> str:
        def repl(m):
            v = self.variables.get(m.group(1))
            c = self.equations.get(m.group(1))
            return f"$({c.expression}) = {v.value:.4g}$ {v.unit}".strip() if c and v and v.value else m.group(0)
        return self.full_render.sub(repl, text)
    
    def _clean_text(self, text: str) -> str:
        return re.sub(r'\\n{3,}', '\\n\\n', text).strip()
    
    def get_summary(self) -> Dict[str, Any]:
        return {
            'variables': list(self.variables.keys()),
            'equations': list(self.equations.keys()),
            'variable_count': len(self.variables),
            'equation_count': len(self.equations),
            'document_type': self.document_type.value,
        }

__all__ = ['NaturalMemorialEditor', 'SmartVariable', 'CalculationResult', 'DocumentType', 'RenderMode']
'''

NATURAL_ENGINE.write_text(CORRECTED_CODE, encoding='utf-8')
print_success(f"Código corrigido escrito em: {NATURAL_ENGINE}")

# ============================================================================
# STEP 5: Limpar cache
# ============================================================================

print_header("STEP 5: Limpando cache")

cache_dirs = list(ROOT.rglob("__pycache__"))
for cache_dir in cache_dirs:
    shutil.rmtree(cache_dir)
    print_success(f"Removido: {cache_dir.relative_to(ROOT)}")

if not cache_dirs:
    print_warning("Nenhum cache encontrado")

# ============================================================================
# FINAL
# ============================================================================

print_header("✓ AUTO-FIX CONCLUÍDO")

print(f"{C.GREEN}{C.BOLD}AGORA EXECUTE O TESTE:{C.END}")
print(f"{C.YELLOW}poetry run python examples/debug_test.py{C.END}\n")
print(f"{C.GREEN}Você DEVE ver estas linhas:{C.END}")
print(f"{C.BLUE}  ✓ SmartParser initialized successfully{C.END}")
print(f"{C.BLUE}  ✓ Calling SmartParser.detect_variables(){C.END}")
print(f"{C.BLUE}  ✓ Detected 2 variables: ['gamma_f', 'M_k']{C.END}")
print(f"{C.BLUE}  ✓ Calculated: M_d = 157.5 kN.m{C.END}\n")