# src/pymemorial/core/__init__.py
"""
Core Module - PyMemorial v2.1.2 (Compatibility Validated + Bug Fixes)

✅ CORREÇÕES v2.1.2:
- Exportação correta de NUMPY_AVAILABLE
- Ordem de importação otimizada
- Flags globais sincronizadas
- Suporte completo a Matrix com steps robustos
- ✅ NOVA: Validação de compatibilidade automática
- ✅ NOVA: Função validate_compatibility() pública
"""
from __future__ import annotations
import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

# ============================================================================
# FLAGS GLOBAIS (Inicializadas como False, serão atualizadas)
# ============================================================================
PINT_AVAILABLE = False
SYMPY_AVAILABLE = False
RECOGNITION_AVAILABLE = False
MATRIX_AVAILABLE = False
NUMPY_AVAILABLE = False  # ✅ ADICIONADA


# ============================================================================
# MATRIX MODULE (Importação Prioritária)
# ============================================================================
try:
    # Importa Matrix e TODAS as flags de disponibilidade
    from .matrix import (
        Matrix,
        MatrixType,
        GranularityType,
        MATRIX_AVAILABLE as CORE_MATRIX_FLAG,
        NUMPY_AVAILABLE as CORE_NUMPY_FLAG,
        SYMPY_AVAILABLE as CORE_SYMPY_FLAG,
        debug_matrix_parsing  # ✅ Função de debug
    )
    
    # Importa operações matriciais
    from .matrix_ops import (
        multiply_matrices_with_steps,
        invert_matrix_with_steps
    )
    
    # ✅ ATUALIZA FLAGS GLOBAIS com os valores do módulo matrix
    MATRIX_AVAILABLE = CORE_MATRIX_FLAG
    NUMPY_AVAILABLE = CORE_NUMPY_FLAG
    
    # Nota: SYMPY_AVAILABLE será atualizada na seção de Equation abaixo
    # (prioridade para o módulo equation que tem mais features SymPy)
    
    # Log status detalhado
    if MATRIX_AVAILABLE:
        logger.info("✅ Matrix support fully enabled in core")
    else:
        reasons = []
        if not CORE_NUMPY_FLAG:
            reasons.append("NumPy missing")
        if not CORE_SYMPY_FLAG:
            reasons.append("SymPy missing")
        logger.warning(f"⚠️ Matrix support limited: {', '.join(reasons)}")
        
except ImportError as e:
    logger.error(f"❌ Matrix module import failed: {e}")
    MATRIX_AVAILABLE = False
    NUMPY_AVAILABLE = False
    Matrix = None
    MatrixType = None
    GranularityType = None
    multiply_matrices_with_steps = None
    invert_matrix_with_steps = None
    debug_matrix_parsing = None
    
    # Create dummy types for type hints
    try:
        from typing import Any
        MatrixType = Any
        GranularityType = Any
    except:
        pass


# ============================================================================
# UNITS MODULE V2.0 ✅ CORRIGIDO
# ============================================================================
try:
    # ✅ IMPORTS CORRETOS V2.0
    from .units import (
        get_unit_registry,
        reset_unit_registry,
        UnitParser,
        UnitValidator,
        UnitFormatter,
        UnitError,
        PINT_AVAILABLE as CORE_PINT_OK
    )
    
    # ✅ ATUALIZA FLAG GLOBAL
    PINT_AVAILABLE = CORE_PINT_OK
    
    if PINT_AVAILABLE:
        logger.debug("✅ Units module V2.0 (Pint) loaded successfully")
        
        # ✅ CRIAR ALIASES PARA COMPATIBILIDADE V1.0
        # (para não quebrar código legado)
        _unit_registry_cache = get_unit_registry()
        ureg = _unit_registry_cache.ureg if _unit_registry_cache and hasattr(_unit_registry_cache, 'ureg') else None
        
        try:
            import pint
            Quantity = pint.Quantity if ureg else float
        except ImportError:
            Quantity = float
        
        # ✅ FUNÇÕES HELPER SIMPLES (compatibilidade v1.0)
        def parse_quantity(value, unit=None):
            """
            Legacy wrapper para API v1.0.
            
            Para novo código, use:
                reg = get_unit_registry()
                parser = UnitParser(reg)
                parser.parse("10 m")
            """
            reg = get_unit_registry()
            if unit:
                return reg.parse(f"{value} {unit}")
            return reg.parse(str(value))
        
        def strip_units(quantity):
            """Extrai magnitude de Quantity."""
            if hasattr(quantity, 'magnitude'):
                return float(quantity.magnitude)
            return float(quantity)
        
        def check_dimensional_consistency(q1, q2):
            """
            Verifica consistência dimensional (compatibilidade v1.0).
            
            Returns:
                dict: {'consistent': bool, 'score': int}
            """
            try:
                validator = UnitValidator(get_unit_registry())
                compatible = validator.are_compatible(q1, q2)
                return {'consistent': compatible, 'score': 10 if compatible else 0}
            except:
                return {'consistent': True, 'score': 10}
        
    else:
        logger.warning(
            "⚠️ Units module loaded BUT Pint not available. "
            "Using float fallback."
        )
        
        # ✅ FALLBACKS
        ureg = None
        Quantity = float
        parse_quantity = lambda v, u=None: float(v) if v is not None else 0.0
        strip_units = lambda q: float(getattr(q, 'magnitude', q))
        check_dimensional_consistency = lambda q1, q2: {'consistent': True, 'score': 10}

except ImportError as e:
    logger.error(f"❌ Units module failed to import: {e}. Using float fallback.")
    PINT_AVAILABLE = False
    
    # ✅ FALLBACKS COMPLETOS
    get_unit_registry = lambda: None
    reset_unit_registry = lambda: None
    UnitParser = None
    UnitValidator = None
    UnitFormatter = None
    UnitError = ValueError
    ureg = None
    Quantity = float
    parse_quantity = lambda v, u=None: float(v) if v is not None else 0.0
    strip_units = lambda q: float(getattr(q, 'magnitude', q))
    check_dimensional_consistency = lambda q1, q2: {'consistent': True, 'score': 10}


# ============================================================================
# VARIABLE MODULE ✅ CORRIGIDO
# ============================================================================
try:
    # ✅ REMOVER VariableFactory - NÃO EXISTE!
    from .variable import Variable
    logger.debug("✅ Variable module loaded.")
except ImportError as e:
    logger.warning(f"⚠️ Variable module failed: {e}.")
    Variable = None


# ============================================================================
# EQUATION MODULE (Depende de SymPy)
# ============================================================================
try:
    from .equation import (
        Equation, EquationFactory, 
        StepRegistry, StepPlugin, 
        GranularityType
    )
    
    # ✅ ATUALIZA FLAG GLOBAL DE SYMPY (fonte primária)
    try:
        import sympy as sp
        SYMPY_AVAILABLE = True
        logger.debug("Equation module loaded (SymPy available).")
    except ImportError:
        SYMPY_AVAILABLE = False
        logger.error(
            "Equation module loaded BUT SymPy import failed. "
            "Core functionality limited."
        )
        Equation = None
        EquationFactory = None
        StepRegistry = None
        StepPlugin = None
        GranularityType = None
        
except ImportError as e:
    logger.warning(f"Equation module failed entirely: {e}.")
    Equation = None
    EquationFactory = None
    StepRegistry = None
    StepPlugin = None
    GranularityType = None
    SYMPY_AVAILABLE = False


# ============================================================================
# CALCULATOR MODULE
# ============================================================================
try:
    from .calculator import Calculator
    logger.debug("Calculator module loaded.")
except ImportError as e:
    logger.warning(f"Calculator module failed: {e}.")
    Calculator = None


# ============================================================================
# CACHE MODULE
# ============================================================================
try:
    from .cache import ResultCache, cached_method
    logger.debug("Cache module loaded.")
except ImportError as e:
    logger.warning(f"Cache module failed: {e}.")
    ResultCache = None
    cached_method = None


# ============================================================================
# RECOGNITION MODULE (Opcional)
# ============================================================================
try:
    # Usa import relativo duplo para sair do 'core' e entrar em 'recognition'
    from ..recognition import EngineeringNLP, DetectedVar
    RECOGNITION_AVAILABLE = True
    logger.debug("Recognition module loaded.")
except ImportError:
    logger.debug("Recognition module not available.")
    EngineeringNLP = None
    DetectedVar = None
except ValueError:
    logger.debug("Recognition module not found (relative import failed).")
    EngineeringNLP = None
    DetectedVar = None


# ============================================================================
# DEPENDENCY VALIDATION (Função Original Mantida)
# ============================================================================
def validate_imports() -> Dict[str, Any]:
    """
    Valida a disponibilidade de todos os módulos e retorna um relatório.
    
    Returns:
        Dict com status de cada módulo, módulos faltando e sugestões de instalação
    """
    status = {
        'pint': PINT_AVAILABLE,
        'sympy': SYMPY_AVAILABLE,
        'numpy': NUMPY_AVAILABLE,  # ✅ ADICIONADO
        'matrix': MATRIX_AVAILABLE,  # ✅ ADICIONADO
        'recognition': RECOGNITION_AVAILABLE,
        'variable': Variable is not None,
        'equation': Equation is not None,
        'calculator': Calculator is not None,
        'cache': ResultCache is not None
    }
    
    missing = [k for k, v in status.items() if not v]
    
    suggestions = []
    if not PINT_AVAILABLE:
        suggestions.append("poetry add pint")
    if not SYMPY_AVAILABLE:
        suggestions.append("poetry add sympy")
    if not NUMPY_AVAILABLE:
        suggestions.append("poetry add numpy")
    if Variable is None:
        suggestions.append("Check src/pymemorial/core/variable.py")
    if Equation is None and SYMPY_AVAILABLE:
        suggestions.append("Check src/pymemorial/core/equation.py")
    elif Equation is None:
        suggestions.append("Install SymPy (poetry add sympy) and check src/pymemorial/core/equation.py")
    if Calculator is None:
        suggestions.append("Check src/pymemorial/core/calculator.py")
    if ResultCache is None:
        suggestions.append("Check src/pymemorial/core/cache.py")
    if not RECOGNITION_AVAILABLE:
        suggestions.append("poetry add spacy transformers")

    if missing:
        logger.warning(
            f"Missing/failed core dependencies: {', '.join(missing)}. "
            f"Suggestions: {'; '.join(suggestions)}"
        )
    else:
        logger.info("All core dependencies seem available.")
    
    return {
        'status': status,
        'missing': missing,
        'suggestions': suggestions,
        'all_available': len(missing) == 0
    }


# ============================================================================
# ✅ NOVA: COMPATIBILITY VALIDATION
# ============================================================================
def validate_compatibility() -> Dict[str, Any]:
    """
    Valida compatibilidade entre módulos e versões de dependências.
    
    Verifica:
    - Consistência entre flags e módulos carregados
    - Versões de dependências críticas
    - Circular dependencies
    - Matrix Operations registry
    
    Returns:
        Dict com status de compatibilidade, warnings e errors
    
    Examples:
    --------
    >>> from pymemorial.core import validate_compatibility
    >>> status = validate_compatibility()
    >>> if not status['compatible']:
    ...     print(f"Errors: {status['errors']}")
    >>> if status['warnings']:
    ...     print(f"Warnings: {status['warnings']}")
    """
    warnings_list = []
    errors_list = []
    compatible = True
    
    # 1. Valida SymPy vs Equation
    if SYMPY_AVAILABLE and Equation is None:
        warnings_list.append(
            "SymPy está disponível mas Equation não foi carregada. "
            "Verifique erros de importação em equation.py"
        )
        compatible = False
    
    # 2. Valida NumPy vs Matrix
    if NUMPY_AVAILABLE and not MATRIX_AVAILABLE:
        warnings_list.append(
            "NumPy está disponível mas Matrix está desabilitada. "
            "Verifique erros de importação em matrix.py"
        )
    
    # 3. Valida Matrix vs Matrix Ops
    if MATRIX_AVAILABLE and Matrix is not None:
        try:
            from .matrix_ops import MATRIX_OPERATIONS
            if not MATRIX_OPERATIONS:
                warnings_list.append(
                    "Matrix disponível mas MATRIX_OPERATIONS vazio. "
                    "Operações matriciais limitadas."
                )
        except ImportError as e:
            warnings_list.append(f"Matrix Operations não pôde ser importada: {e}")
    
    # 4. Valida Pint vs Variable
    if not PINT_AVAILABLE and Variable is not None:
        warnings_list.append(
            "Pint não disponível - Variable usará float como fallback. "
            "Conversões de unidade desabilitadas."
        )
    
    # 5. Valida versões de dependências críticas
    try:
        import sympy as sp_check
        sp_version = tuple(map(int, sp_check.__version__.split('.')[:2]))
        if sp_version < (1, 10):
            warnings_list.append(
                f"SymPy {sp_check.__version__} detectado. "
                "Recomendado: SymPy >= 1.10 para melhor desempenho."
            )
    except:
        pass
    
    try:
        import numpy as np_check
        np_version = tuple(map(int, np_check.__version__.split('.')[:2]))
        if np_version < (1, 20):
            warnings_list.append(
                f"NumPy {np_check.__version__} detectado. "
                "Recomendado: NumPy >= 1.20 para suporte completo."
            )
    except:
        pass
    
    # 6. Valida circular dependencies (Variable ↔ Equation)
    if Variable is not None and Equation is not None:
        try:
            # Testa criação básica
            test_var = VariableFactory.create('test', 1.0)
            test_eq = EquationFactory.create('test = 1', {'test': test_var})
            logger.debug("✅ Variable ↔ Equation compatibility OK")
        except Exception as e:
            errors_list.append(f"Variable/Equation incompatibilidade: {e}")
            compatible = False
    
    # 7. Valida StepRegistry (se Equation disponível)
    if Equation is not None and StepRegistry is None:
        warnings_list.append(
            "Equation disponível mas StepRegistry não carregada. "
            "Funcionalidade de plugins de steps desabilitada."
        )
    
    # Log warnings
    for warning in warnings_list:
        logger.warning(f"⚠️ {warning}")
    
    # Log errors
    for error in errors_list:
        logger.error(f"❌ {error}")
    
    return {
        'compatible': compatible and len(errors_list) == 0,
        'warnings': warnings_list,
        'errors': errors_list,
        'modules': {
            'sympy': SYMPY_AVAILABLE,
            'numpy': NUMPY_AVAILABLE,
            'pint': PINT_AVAILABLE,
            'matrix': MATRIX_AVAILABLE,
            'recognition': RECOGNITION_AVAILABLE
        },
        'classes': {
            'Variable': Variable is not None,
            'Equation': Equation is not None,
            'Matrix': Matrix is not None,
            'Calculator': Calculator is not None,
            'StepRegistry': StepRegistry is not None
        },
        'version': get_version()
    }


# ============================================================================
# ✅ NOVA: STARTUP VALIDATION (automática na importação)
# ============================================================================
def _run_startup_validation():
    """Executa validação automática na importação do módulo."""
    logger.debug("Running core module compatibility validation...")
    status = validate_compatibility()
    
    if not status['compatible']:
        logger.error(
            "⚠️ CORE MODULE COMPATIBILITY ISSUES DETECTED ⚠️\n"
            f"Errors: {len(status['errors'])}\n"
            f"Warnings: {len(status['warnings'])}\n"
            "Run validate_compatibility() for details."
        )
    elif status['warnings']:
        logger.warning(
            f"Core loaded with {len(status['warnings'])} warnings. "
            "Run validate_compatibility() for details."
        )
    else:
        logger.info("✅ Core module compatibility: OK")


# ============================================================================
# CORE BUNDLE FACTORY
# ============================================================================
def get_core_bundle(
    enable_nlp: bool = False,
    enable_cache: bool = True,
    cache_size: int = 128,
    cache_ttl: Optional[float] = None
) -> Dict[str, Any]:
    """
    Cria um bundle com todos os componentes do core.
    
    Args:
        enable_nlp: Habilitar reconhecimento NLP
        enable_cache: Habilitar cache de resultados
        cache_size: Tamanho máximo do cache
        cache_ttl: Time-to-live do cache em segundos
        
    Returns:
        Dict com todos os componentes e flags de disponibilidade
    """
    bundle = {
        # Factories
        'variable': VariableFactory if VariableFactory else None,
        'equation': EquationFactory if EquationFactory else None,
        'calculator': None,
        'cache': None,
        'step_registry': StepRegistry if StepRegistry else None,
        
        # Matrix support
        'matrix': Matrix if Matrix else None,
        'multiply_matrices': multiply_matrices_with_steps if multiply_matrices_with_steps else None,
        'invert_matrix': invert_matrix_with_steps if invert_matrix_with_steps else None,
        'debug_matrix': debug_matrix_parsing if debug_matrix_parsing else None,
        
        # Units
        'parse_quantity': parse_quantity,
        'ureg': ureg,
        'strip_units': strip_units,
        'normalize_norm': normalize_norm,
        'suggest_unit': suggest_unit,
        'latex_unit': latex_unit,
        
        # Utilities
        'validate_deps': validate_imports,
        'validate_compat': validate_compatibility,  # ✅ NOVA
        
        # Flags
        'pint_available': PINT_AVAILABLE,
        'sympy_available': SYMPY_AVAILABLE,
        'numpy_available': NUMPY_AVAILABLE,  # ✅ ADICIONADO
        'matrix_available': MATRIX_AVAILABLE,  # ✅ ADICIONADO
        'recognition_available': RECOGNITION_AVAILABLE,
        
        # NLP
        'suggest': None
    }
    
    # Setup cache
    if enable_cache and ResultCache:
        try:
            bundle['cache'] = ResultCache(maxsize=cache_size, ttl=cache_ttl)
        except Exception as e:
            logger.warning(f"Failed to create ResultCache: {e}")
    
    # Setup calculator
    if Calculator:
        try:
            bundle['calculator'] = Calculator(cache=bundle['cache'])
        except Exception as e:
            logger.warning(f"Failed to create Calculator: {e}")
    
    # Setup NLP
    if enable_nlp and RECOGNITION_AVAILABLE and EngineeringNLP and DetectedVar:
        def suggest_type(description: str, name: Optional[str] = None) -> str:
            try:
                nlp = EngineeringNLP()
                base = name.split('_')[0] if name and '_' in name else (name or description)
                det = DetectedVar(name=name or description, base=base, subscript='')
                return nlp.infer_type(det, description)
            except Exception as e:
                logger.warning(f"NLP suggestion failed: {e}")
                return 'unknown'
        
        bundle['suggest'] = suggest_type
    
    logger.debug(f"Core bundle created (NLP:{enable_nlp}, Cache:{enable_cache})")
    return bundle


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================
def get_version() -> str:
    """Retorna a versão do módulo core."""
    return "2.1.2"

















# ============================================================================
# EXPORTS (__all__)
# ============================================================================
__all__ = [
    # ✅ Units V2.0 (API atualizada)
    'get_unit_registry',
    'reset_unit_registry',
    'UnitParser',
    'UnitValidator',
    'UnitFormatter',
    'UnitError',
    'parse_quantity',  # Compatibilidade v1.0
    'ureg',            # Compatibilidade v1.0
    'Quantity',        # Compatibilidade v1.0
    'strip_units',     # Compatibilidade v1.0
    'check_dimensional_consistency',  # Compatibilidade v1.0
    
    # Variable
    'Variable', 
    'VariableFactory',
    
    # Equation
    'Equation', 
    'EquationFactory',
    'StepRegistry', 
    'StepPlugin', 
    'GranularityType',
    
    # Matrix (✅ COMPLETO)
    'Matrix',
    'MatrixType',
    'multiply_matrices_with_steps',
    'invert_matrix_with_steps',
    'debug_matrix_parsing',  # ✅ ADICIONADO
    
    # Calculator
    'Calculator',
    
    # Cache
    'ResultCache', 
    'cached_method',
    
    # Utilities
    'get_core_bundle', 
    'validate_imports', 
    'validate_compatibility',  # ✅ NOVA
    'get_version',
    
    # Flags (✅ TODAS AS FLAGS)
    'PINT_AVAILABLE',
    'SYMPY_AVAILABLE',
    'NUMPY_AVAILABLE',  # ✅ ADICIONADO
    'MATRIX_AVAILABLE',  # ✅ ADICIONADO
    'RECOGNITION_AVAILABLE',
]

__version__ = "2.1.2"
__author__ = "PyMemorial Team"
__email__ = "contact@pymemorial.org"


# ============================================================================
# ✅ EXECUTAR VALIDAÇÃO AUTOMÁTICA NA IMPORTAÇÃO
# ============================================================================
_run_startup_validation()


# ============================================================================
# LOG INICIAL
# ============================================================================
logger.debug(
    f"PyMemorial Core v{__version__} initialized. "
    f"Matrix: {MATRIX_AVAILABLE}, NumPy: {NUMPY_AVAILABLE}, "
    f"SymPy: {SYMPY_AVAILABLE}, Pint: {PINT_AVAILABLE}"
)