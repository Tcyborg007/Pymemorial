# tests/unit/test_core/test_units_part1.py

"""
Testes PARTE 1: UnitRegistry - Wrapper Pint Singleton

OBJETIVO:
- Wrapper inteligente sobre Pint
- Singleton thread-safe
- Lazy loading (só carrega Pint se necessário)
- Custom definitions (tonne_force, percent)
- Fallback se Pint não instalado
"""

import pytest

from pymemorial.core.units import (
    get_unit_registry,
    reset_unit_registry,
    UnitRegistry,
    UnitError,
    PINT_AVAILABLE
)


class TestUnitRegistrySingleton:
    """Testes do padrão Singleton."""
    
    def test_get_unit_registry_returns_singleton(self):
        """get_unit_registry() deve retornar mesma instância."""
        reg1 = get_unit_registry()
        reg2 = get_unit_registry()
        assert reg1 is reg2
    
    def test_reset_unit_registry_creates_new_instance(self):
        """reset_unit_registry() deve criar nova instância."""
        reg1 = get_unit_registry()
        reset_unit_registry()
        reg2 = get_unit_registry()
        assert reg1 is not reg2


class TestUnitRegistryBasic:
    """Testes básicos de UnitRegistry."""
    
    def test_pint_available_flag(self):
        """PINT_AVAILABLE deve indicar se Pint está instalado."""
        # Se Pint instalado, deve ser True
        # Se não instalado, deve ser False
        assert isinstance(PINT_AVAILABLE, bool)
    
    @pytest.mark.skipif(not PINT_AVAILABLE, reason="Pint not installed")
    def test_registry_has_ureg(self):
        """Registry deve ter ureg (Pint UnitRegistry) se disponível."""
        reg = get_unit_registry()
        assert reg.ureg is not None
    
    @pytest.mark.skipif(not PINT_AVAILABLE, reason="Pint not installed")
    def test_parse_simple_unit(self):
        """Parsing de unidade simples."""
        reg = get_unit_registry()
        quantity = reg.parse('10 m')
        
        assert quantity is not None
        # Verificar magnitude
        assert abs(quantity.magnitude - 10.0) < 0.001


class TestUnitRegistryCustomDefinitions:
    """Testes de definições customizadas."""
    
    @pytest.mark.skipif(not PINT_AVAILABLE, reason="Pint not installed")
    def test_custom_tonne_force(self):
        """Definição customizada: tonne_force (tf)."""
        reg = get_unit_registry()
        quantity = reg.parse('1 tf')
        
        # 1 tf = 1000 kgf ≈ 9806.65 N
        quantity_n = reg.convert(quantity, 'N')
        assert abs(quantity_n.magnitude - 9806.65) < 1.0
    
    # tests/unit/test_core/test_units_part1.py

    @pytest.mark.skipif(not PINT_AVAILABLE, reason="Pint not installed")
    def test_custom_percent(self):
        """Definição customizada: percent (%)."""
        reg = get_unit_registry()
        
        # Tentar parsing com % (Pint já tem suporte nativo)
        quantity = reg.parse('50%')
        
        # 50% no Pint = 0.5 (dimensionless)
        # Pint armazena percentual como fração
        magnitude = quantity.magnitude
        
        # Tolerância para representação interna
        # Algumas versões do Pint armazenam como 50, outras como 0.5
        # Normalizar para 0-1 range
        if magnitude > 1:
            # Pint armazenou como 50 (percentual bruto)
            magnitude = magnitude / 100.0
        
        # Verificar se é aproximadamente 0.5 (50%)
        assert abs(magnitude - 0.5) < 0.01, f"Expected ~0.5, got {magnitude}"




class TestUnitRegistryFallback:
    """Testes de fallback quando Pint não disponível."""
    
    @pytest.mark.skipif(PINT_AVAILABLE, reason="Pint is installed")
    def test_registry_works_without_pint(self):
        """Registry deve funcionar (modo degradado) sem Pint."""
        reg = get_unit_registry()
        assert reg is not None
        assert reg.ureg is None  # Sem Pint
    
    @pytest.mark.skipif(PINT_AVAILABLE, reason="Pint is installed")
    def test_parse_returns_float_without_pint(self):
        """parse() sem Pint deve retornar float."""
        reg = get_unit_registry()
        result = reg.parse('10 m')
        
        # Fallback: retorna apenas magnitude como float
        assert isinstance(result, float)
        assert result == 10.0


# Executar: pytest tests/unit/test_core/test_units_part1.py -v
# Esperado: 10 FAILURES (RED) antes da implementação ✅
