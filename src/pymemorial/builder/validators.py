# src/pymemorial/builder/validators.py
"""
Validadores aprimorados para builder (v2.0: Full Cycles + Compliance).

Valida vars/sections/equations com Tarjan cycles, norm checks (gamma_s >=1.0 NBR).
Compatível 100%; fluent validate_chain().

Example:
    validator = MemorialValidator()
    report = validator.validate_chain({'M_k': 150}, [eq], "template")
"""

from typing import List, Set, Tuple, Dict, Optional, Any
import re
import logging
from collections import defaultdict

# FIX: Import MVP correto
try:
    from ..recognition import get_engine
    RECOGNITION_AVAILABLE = True
except ImportError:
    RECOGNITION_AVAILABLE = False
    get_engine = None

_logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Erro de validação (enhanced: + type_hint)."""
    def __init__(self, message: str, type_hint: Optional[str] = None):
        self.type_hint = type_hint
        super().__init__(message)


class ValidationReport:
    """Report de validação (list errors + suggestions)."""
    def __init__(self, valid: bool, errors: List[Tuple[str, str]], suggestions: List[Dict]):
        self.valid = valid
        self.errors = errors  # (element, message)
        self.suggestions = suggestions  # [{'fix': str, 'reason': str}]


class MemorialValidator:
    """Validador aprimorado com cycles Tarjan, norm compliance."""
    
    @staticmethod
    def validate_variable_name(name: str) -> bool:
        """Valida nome (Python rules + reserved)."""
        pattern = r'^[a-zA-Z_][a-zA-Z0-9_]*$'
        if not re.match(pattern, name):
            raise ValidationError(
                f"Nome inválido '{name}'. Deve começar com letra/_ e alfanum/_ apenas.",
                type_hint='var_name'
            )
        
        reserved = {'and', 'or', 'not', 'in', 'is', 'for', 'if', 'else', 'def', 'class', 'return'}
        if name in reserved:
            raise ValidationError(f"'{name}' é reserved Python.", type_hint='reserved')
        
        return True
    
    @staticmethod
    def validate_section_level(level: int, parent_level: int = 0) -> bool:
        """Valida level (no salto >1)."""
        if level < 1:
            raise ValidationError("Level >=1.", type_hint='section_level')
        if level > parent_level + 1 and parent_level > 0:
            raise ValidationError(
                f"Salto inválido {parent_level} → {level}. Incremento 1.",
                type_hint='level_jump'
            )
        return True
    
    @staticmethod
    def check_circular_references(variables: Dict, equations: List) -> Tuple[bool, List[List[str]]]:
        """Full Tarjan cycles em deps vars/equations."""
        # Graph: var → deps
        graph = defaultdict(list)
        
        for eq in equations:
            # Assumir lhs = primeira var
            lhs = eq.variables[0].name if hasattr(eq, 'variables') and eq.variables else None
            if lhs:
                # RHS vars (símbolos SymPy)
                rhs_vars = []
                if hasattr(eq, 'expression') and hasattr(eq.expression, 'free_symbols'):
                    rhs_vars = [str(s) for s in eq.expression.free_symbols]
                
                graph[lhs].extend(rhs_vars)
                for dep in rhs_vars:
                    if dep in variables:
                        graph[dep]  # Touch to include
        
        # Tarjan algo (disc, low, stack for SCC)
        disc = {}
        low = {}
        stack = []
        on_stack = set()
        cycles = []
        time_counter = [0]  # Usar list para mutabilidade em nested func
        
        def tarjan(node):
            disc[node] = low[node] = time_counter[0]
            time_counter[0] += 1
            stack.append(node)
            on_stack.add(node)
            
            for child in graph[node]:
                if child not in disc:
                    tarjan(child)
                    low[node] = min(low[node], low[child])
                elif child in on_stack:
                    low[node] = min(low[node], disc[child])
            
            if low[node] == disc[node]:
                cycle = []
                while True:
                    u = stack.pop()
                    on_stack.remove(u)
                    cycle.append(u)
                    if u == node:
                        break
                if len(cycle) > 1:
                    cycles.append(cycle)
        
        for node in graph:
            if node not in disc:
                tarjan(node)
        
        has_cycle = bool(cycles)
        if has_cycle:
            _logger.warning(f"Found {len(cycles)} cycles: {cycles}")
        else:
            _logger.debug("No cycles detected")
        
        return has_cycle, cycles
    
    @staticmethod
    def check_undefined_variables(equations: List, defined_vars: Set[str]) -> List[str]:
        """Undefined vars."""
        undefined = []
        for eq in equations:
            symbols = []
            if hasattr(eq, 'expression') and hasattr(eq.expression, 'free_symbols'):
                symbols = eq.expression.free_symbols
            
            for symbol in symbols:
                sym_str = str(symbol)
                if sym_str not in defined_vars:
                    undefined.append(sym_str)
        
        return list(set(undefined))
    
    @staticmethod
    def validate_norm_compliance(variables: Dict[str, Any], norm: str = "NBR6118_2023") -> List[Tuple[str, str]]:
        """Norm checks (ex: gamma_s >=1.0)."""
        errors = []
        factors = {'NBR6118_2023': {'safety_factor': 1.0}}  # Min value
        
        for name, var in variables.items():
            if 'gamma' in name.lower() or 'safety' in name.lower():
                val = getattr(var, 'value', 0) if hasattr(var, 'value') else 0
                if hasattr(val, 'magnitude'):
                    val = val.magnitude
                
                min_val = factors.get(norm, {}).get('safety_factor', 1.0)
                if val < min_val:
                    errors.append((name, f"{val} < {min_val} (norm {norm})"))
        
        return errors
    
    @staticmethod
    def suggest_corrections(report: ValidationReport) -> List[Dict[str, str]]:
        """Suggest fixes (stub para MVP)."""
        suggestions = []
        
        # Stub simples: detecta undefined vars
        for elem, msg in report.errors:
            if 'undefined' in msg.lower():
                suggestions.append({
                    'fix': f'add_variable("{elem}", value=0)',
                    'reason': f'Variable {elem} is undefined',
                    'type': 'undefined_var'
                })
        
        return suggestions
    
    @staticmethod
    def validate_template(template: str) -> Tuple[bool, List[str]]:
        """Valida template ({{var}} syntax)."""
        # Pattern para {{var}}
        placeholder_pattern = r'\{\{([a-zA-Z_][a-zA-Z0-9_]*)\}\}'
        placeholders = re.findall(placeholder_pattern, template)
        
        # Detecta malformados {var}
        malformed_pattern = r'(?<!\{)\{(?!\{)|(?<!\})\}(?!\})'
        malformed = re.findall(malformed_pattern, template)
        
        is_valid = len(malformed) == 0
        return is_valid, placeholders
    
    @staticmethod
    def validate_chain(
        variables: Dict[str, Any],
        equations: List,
        template: str = ""
    ) -> ValidationReport:
        """Fluent chain validation (all + report/suggest)."""
        errors = []
        
        # 1. Validate variable names
        for name in variables:
            try:
                MemorialValidator.validate_variable_name(name)
            except ValidationError as e:
                errors.append((name, str(e)))
        
        # 2. Check circular references
        has_cycle, cycles = MemorialValidator.check_circular_references(variables, equations)
        if has_cycle:
            errors.extend([(c[0], f"Cycle: {' → '.join(c)}") for c in cycles])
        
        # 3. Check undefined variables
        undefined = MemorialValidator.check_undefined_variables(equations, set(variables.keys()))
        if undefined:
            errors.extend([(u, f"Undefined variable") for u in undefined])
        
        # 4. Norm compliance
        norm_errors = MemorialValidator.validate_norm_compliance(variables)
        if norm_errors:
            errors.extend(norm_errors)
        
        # 5. Template validation
        if template:
            is_valid_tpl, _ = MemorialValidator.validate_template(template)
            if not is_valid_tpl:
                errors.append(("template", "Invalid template syntax"))
        
        valid = len(errors) == 0
        report = ValidationReport(valid, errors, [])
        suggestions = MemorialValidator.suggest_corrections(report)
        report.suggestions = suggestions
        
        return report


__all__ = [
    'ValidationError',
    'ValidationReport',
    'MemorialValidator',
]
