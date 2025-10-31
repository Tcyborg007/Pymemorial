"""
PyMemorial v2.0 - Configuração Global de Testes Unitários
==========================================================

Fixtures compartilhadas e configuração do pytest.

Author: PyMemorial Team
Date: 2025-10-27
Status: v2.0 Implementation
"""

import pytest
import sys
import warnings
from pathlib import Path
from typing import Dict, Any, Optional

# ============================================================================
# CONFIGURAÇÃO DE PATH
# ============================================================================

# Adicionar src ao PYTHONPATH para imports funcionarem
src_path = Path(__file__).parent.parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))


# ============================================================================
# IMPORTS CONDICIONAIS (Compatibilidade v1.0 ↔ v2.0)
# ============================================================================

# Config module (pode não existir ainda se em desenvolvimento)
try:
    from pymemorial.core.config import (
        get_config,
        reset_config,
        set_option,
        PyMemorialConfig
    )
    CONFIG_AVAILABLE = True
except (ImportError, ModuleNotFoundError, AttributeError):
    CONFIG_AVAILABLE = False
    warnings.warn(
        "pymemorial.core.config não disponível. "
        "Usando mocks temporários para compatibilidade.",
        UserWarning
    )
    
    # Mock Config (fallback temporário)
    class MockDisplayConfig:
        precision = 2
        latex_mode = True
        unit_system = "SI"
        language = "pt_BR"
        scientific_notation = False
    
    class MockCalculationConfig:
        granularity = "medium"
        auto_simplify = True
        cache_enabled = True
    
    class MockConfig:
        display = MockDisplayConfig()
        calculation = MockCalculationConfig()
    
    def get_config():
        """Mock get_config."""
        return MockConfig()
    
    def reset_config():
        """Mock reset_config."""
        pass
    
    def set_option(key: str, value: Any):
        """Mock set_option."""
        pass


# Variable module (deve existir)
try:
    from pymemorial.core.variable import Variable
    VARIABLE_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    VARIABLE_AVAILABLE = False
    Variable = None


# Units module (deve existir)
try:
    from pymemorial.core.units import (
        get_unit_registry,
        UnitParser,
        UnitValidator,
        UnitFormatter
    )
    UNITS_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    UNITS_AVAILABLE = False


# Equation module (pode não existir ainda)
try:
    from pymemorial.core.equation import Equation
    EQUATION_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    EQUATION_AVAILABLE = False
    Equation = None


# Calculator module (pode não existir ainda)
try:
    from pymemorial.core.calculator import Calculator
    CALCULATOR_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    CALCULATOR_AVAILABLE = False
    Calculator = None


# SymPy (dependency externa)
try:
    import sympy as sp
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False
    sp = None


# NumPy (dependency externa)
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None


# Pint (dependency externa)
try:
    import pint
    PINT_AVAILABLE = True
except ImportError:
    PINT_AVAILABLE = False


# ============================================================================
# FIXTURES GLOBAIS - LIFECYCLE
# ============================================================================

@pytest.fixture(autouse=True)
def reset_global_state():
    """
    Reseta estado global após cada teste.
    
    Executado automaticamente para TODOS os testes.
    Garante isolamento entre testes.
    """
    # Setup: Antes do teste (não precisa fazer nada)
    yield
    
    # Teardown: Após o teste
    if CONFIG_AVAILABLE:
        try:
            reset_config()
        except Exception as e:
            warnings.warn(f"Erro ao resetar config: {e}", UserWarning)
    
    # Limpar caches (se disponível)
    if UNITS_AVAILABLE:
        try:
            # Resetar registry de unidades (se existir método)
            from pymemorial.core.units import reset_unit_registry
            reset_unit_registry()
        except (ImportError, AttributeError):
            pass


@pytest.fixture
def clean_config():
    """
    Fornece config limpo para testes que precisam isolar configuração.
    
    Examples:
        def test_custom_precision(clean_config):
            set_option("display.precision", 4)
            # Teste com precisão personalizada
    """
    if CONFIG_AVAILABLE:
        reset_config()
        return get_config()
    else:
        return MockConfig()


# ============================================================================
# FIXTURES - VARIÁVEIS
# ============================================================================

@pytest.fixture
def simple_var():
    """Fixture: Variável simples sem unidade."""
    if not VARIABLE_AVAILABLE:
        pytest.skip("Variable module não disponível")
    return Variable("x", 10)


@pytest.fixture
def var_with_unit():
    """Fixture: Variável com unidade."""
    if not VARIABLE_AVAILABLE:
        pytest.skip("Variable module não disponível")
    return Variable("F", 100, unit="kN")


@pytest.fixture
def simple_vars() -> Dict[str, 'Variable']:
    """Fixture: Dicionário de variáveis simples."""
    if not VARIABLE_AVAILABLE:
        pytest.skip("Variable module não disponível")
    return {
        "x": Variable("x", 10),
        "y": Variable("y", 5),
        "z": Variable("z", 2)
    }


@pytest.fixture
def vars_with_units() -> Dict[str, 'Variable']:
    """Fixture: Dicionário de variáveis com unidades."""
    if not VARIABLE_AVAILABLE:
        pytest.skip("Variable module não disponível")
    return {
        "F": Variable("F", 100, unit="kN"),
        "L": Variable("L", 6, unit="m"),
        "q": Variable("q", 15, unit="kN/m"),
        "gamma_f": Variable("gamma_f", 1.4)  # Adimensional
    }


@pytest.fixture
def engineering_vars() -> Dict[str, 'Variable']:
    """Fixture: Variáveis típicas de engenharia estrutural."""
    if not VARIABLE_AVAILABLE:
        pytest.skip("Variable module não disponível")
    return {
        "f_ck": Variable("f_ck", 30, unit="MPa", description="Resistência característica do concreto"),
        "f_yk": Variable("f_yk", 500, unit="MPa", description="Resistência característica do aço"),
        "b_w": Variable("b_w", 0.20, unit="m", description="Largura da viga"),
        "h": Variable("h", 0.50, unit="m", description="Altura da viga"),
        "d": Variable("d", 0.45, unit="m", description="Altura útil"),
        "M_d": Variable("M_d", 150, unit="kN*m", description="Momento de cálculo")
    }


# ============================================================================
# FIXTURES - EQUATION
# ============================================================================

@pytest.fixture
def simple_equation():
    """Fixture: Equação simples."""
    if not EQUATION_AVAILABLE:
        pytest.skip("Equation module não disponível")
    return Equation("x + y")


@pytest.fixture
def quadratic_equation():
    """Fixture: Equação quadrática."""
    if not EQUATION_AVAILABLE:
        pytest.skip("Equation module não disponível")
    return Equation("x**2 + 2*x + 1")


@pytest.fixture
def equation_with_vars(simple_vars):
    """Fixture: Equação com variáveis."""
    if not EQUATION_AVAILABLE:
        pytest.skip("Equation module não disponível")
    return Equation("x * y + z", locals_dict=simple_vars)


# ============================================================================
# FIXTURES - CALCULATOR
# ============================================================================

@pytest.fixture
def calculator_empty():
    """Fixture: Calculator vazio."""
    if not CALCULATOR_AVAILABLE:
        pytest.skip("Calculator module não disponível")
    return Calculator()


@pytest.fixture
def calculator_with_vars(simple_vars):
    """Fixture: Calculator com variáveis."""
    if not CALCULATOR_AVAILABLE:
        pytest.skip("Calculator module não disponível")
    return Calculator(variables=simple_vars)


# ============================================================================
# FIXTURES - UTILITIES
# ============================================================================

@pytest.fixture
def temp_dir(tmp_path):
    """
    Fixture: Diretório temporário para testes de I/O.
    
    Usa tmp_path do pytest (automático cleanup).
    """
    return tmp_path


@pytest.fixture
def mock_logger(monkeypatch):
    """Fixture: Mock de logger para testes."""
    import logging
    
    class MockHandler(logging.Handler):
        def __init__(self):
            super().__init__()
            self.messages = []
        
        def emit(self, record):
            self.messages.append(self.format(record))
    
    handler = MockHandler()
    logger = logging.getLogger("pymemorial")
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    
    yield handler
    
    logger.removeHandler(handler)


# ============================================================================
# MARKERS PERSONALIZADOS
# ============================================================================

# tests/unit/test_core/conftest.py (TRECHO CORRIGIDO)

# ============================================================================
# VERIFICAÇÃO DE DEPENDÊNCIAS (com fallbacks)
# ============================================================================

def pytest_configure(config):
    """Verifica dependências antes de rodar testes."""
    
    print("\nPyMemorial v2.0 Test Suite")
    print(f"Python: {sys.version.split()[0]}")
    print("\nDependencies disponíveis:")
    
    # Config
    try:
        from pymemorial.core.config import get_config
        print("  - Config:     ✅")
    except ImportError:
        print("  - Config:     ❌")
    
    # Variable
    try:
        from pymemorial.core.variable import Variable
        print("  - Variable:   ✅")
    except ImportError:
        print("  - Variable:   ❌")
    
    # Units
    try:
        from pymemorial.core.units import get_ureg
        print("  - Units:      ✅")
    except ImportError:
        print("  - Units:      ❌")
    
    # Equation (SEM VERIFICAR StepRegistry!)
    try:
        from pymemorial.core.equation import Equation
        print("  - Equation:   ✅")
    except ImportError as e:
        print(f"  - Equation:   ⚠️ (parcial: {e})")
    
    # Calculator
    try:
        from pymemorial.core.calculator import Calculator
        print("  - Calculator: ✅")
    except ImportError:
        print("  - Calculator: ❌")
    
    # SymPy
    try:
        import sympy
        print("  - SymPy:      ✅")
    except ImportError:
        print("  - SymPy:      ❌")
    
    # NumPy
    try:
        import numpy
        print("  - NumPy:      ✅")
    except ImportError:
        print("  - NumPy:      ❌")
    
    # Pint
    try:
        import pint
        print("  - Pint:       ✅")
    except ImportError:
        print("  - Pint:       ❌")
    
    print()


# ============================================================================
# FIXTURES BÁSICAS
# ============================================================================

@pytest.fixture
def ureg():
    """Unit registry Pint."""
    from pymemorial.core.units import get_ureg
    return get_ureg()


@pytest.fixture
def simple_variable():
    """Variable simples para testes."""
    from pymemorial.core.variable import Variable
    return Variable("x", 10, unit="m", description="Test variable")


@pytest.fixture
def calculator_instance():
    """Calculator instance for tests."""
    try:
        from pymemorial.core.calculator import Calculator
        return Calculator()
    except ImportError:
        pytest.skip("Calculator not available")


# ❌ REMOVER OU COMENTAR (StepRegistry não existe ainda):
# @pytest.fixture
# def step_registry():
#     """StepRegistry instance for tests."""
#     from pymemorial.core.equation import StepRegistry
#     return StepRegistry()

# ============================================================================
# HOOKS PYTEST
# ============================================================================

def pytest_collection_modifyitems(config, items):
    """
    Modifica collection de testes.
    
    - Auto-skip testes que requerem dependencies não instaladas
    - Adiciona markers automáticos
    """
    for item in items:
        # Auto-skip se dependency faltando
        if "requires_sympy" in item.keywords and not SYMPY_AVAILABLE:
            item.add_marker(
                pytest.mark.skip(reason="SymPy não instalado")
            )
        
        if "requires_pint" in item.keywords and not PINT_AVAILABLE:
            item.add_marker(
                pytest.mark.skip(reason="Pint não instalado")
            )
        
        # Adicionar marker 'unit' automaticamente se não tiver nenhum
        if not any(marker in item.keywords for marker in ["unit", "integration", "smoke"]):
            item.add_marker(pytest.mark.unit)


# ============================================================================
# ASSERTIONS CUSTOMIZADAS
# ============================================================================

def assert_almost_equal(actual: float, expected: float, tolerance: float = 1e-6):
    """
    Assert com tolerância para floats.
    
    Examples:
        assert_almost_equal(result.value, 67.5, tolerance=0.01)
    """
    assert abs(actual - expected) < tolerance, (
        f"Valores não são aproximadamente iguais:\n"
        f"  Esperado: {expected}\n"
        f"  Recebido: {actual}\n"
        f"  Diferença: {abs(actual - expected)} (tolerância: {tolerance})"
    )


# ============================================================================
# INFORMAÇÕES DE AMBIENTE (Debug)
# ============================================================================

def pytest_report_header(config):
    """Adiciona informações ao header do pytest."""
    info = [
        f"PyMemorial v2.0 Test Suite",
        f"Python: {sys.version.split()[0]}",
        f"",
        f"Dependencies disponíveis:",
        f"  - Config:     {'✅' if CONFIG_AVAILABLE else '❌'}",
        f"  - Variable:   {'✅' if VARIABLE_AVAILABLE else '❌'}",
        f"  - Units:      {'✅' if UNITS_AVAILABLE else '❌'}",
        f"  - Equation:   {'✅' if EQUATION_AVAILABLE else '❌'}",
        f"  - Calculator: {'✅' if CALCULATOR_AVAILABLE else '❌'}",
        f"  - SymPy:      {'✅' if SYMPY_AVAILABLE else '❌'}",
        f"  - NumPy:      {'✅' if NUMPY_AVAILABLE else '❌'}",
        f"  - Pint:       {'✅' if PINT_AVAILABLE else '❌'}",
    ]
    return "\n".join(info)


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Fixtures
    'reset_global_state',
    'clean_config',
    'simple_var',
    'var_with_unit',
    'simple_vars',
    'vars_with_units',
    'engineering_vars',
    'simple_equation',
    'quadratic_equation',
    'equation_with_vars',
    'calculator_empty',
    'calculator_with_vars',
    'temp_dir',
    'mock_logger',
    # Utilities
    'assert_almost_equal',
    # Flags
    'CONFIG_AVAILABLE',
    'VARIABLE_AVAILABLE',
    'UNITS_AVAILABLE',
    'EQUATION_AVAILABLE',
    'CALCULATOR_AVAILABLE',
    'SYMPY_AVAILABLE',
    'NUMPY_AVAILABLE',
    'PINT_AVAILABLE',
]
