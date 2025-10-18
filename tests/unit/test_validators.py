"""Testes de validadores."""
import pytest
from pymemorial.builder.validators import (
    MemorialValidator,
    ValidationError,
    validate_template,
)


def test_validate_variable_name_valid():
    """Testa validação de nome válido."""
    validator = MemorialValidator()
    assert validator.validate_variable_name("fck") is True
    assert validator.validate_variable_name("alpha_c") is True
    assert validator.validate_variable_name("_private") is True


def test_validate_variable_name_invalid():
    """Testa validação de nome inválido."""
    validator = MemorialValidator()
    
    with pytest.raises(ValidationError, match="inválido"):
        validator.validate_variable_name("123var")  # Começa com número
    
    with pytest.raises(ValidationError, match="inválido"):
        validator.validate_variable_name("var-name")  # Contém hífen


def test_validate_variable_name_reserved():
    """Testa rejeição de palavras reservadas."""
    validator = MemorialValidator()
    
    with pytest.raises(ValidationError, match="palavra reservada"):
        validator.validate_variable_name("for")


def test_validate_section_level_valid():
    """Testa validação de nível de seção."""
    validator = MemorialValidator()
    assert validator.validate_section_level(1) is True
    assert validator.validate_section_level(2, parent_level=1) is True


def test_validate_section_level_invalid():
    """Testa nível de seção inválido."""
    validator = MemorialValidator()
    
    with pytest.raises(ValidationError, match="deve ser >= 1"):
        validator.validate_section_level(0)
    
    with pytest.raises(ValidationError, match="Salto de nível"):
        validator.validate_section_level(3, parent_level=1)


def test_check_circular_references():
    """Testa detecção de referências circulares."""
    validator = MemorialValidator()
    has_cycles, cycles = validator.check_circular_references({}, [])
    assert has_cycles is False
    assert cycles == []


def test_check_undefined_variables():
    """Testa detecção de variáveis não definidas."""
    import sympy as sp
    from pymemorial.core import Equation
    
    # Equação com variável não definida
    x, y = sp.symbols('x y')
    eq = Equation(expression=x + y, variables={})
    
    validator = MemorialValidator()
    undefined = validator.check_undefined_variables([eq], defined_vars={'x'})
    
    assert 'y' in undefined


def test_validate_template():
    """Testa validação de template."""
    is_valid, vars_req = validate_template("Valor: {{x}}")
    assert is_valid is True
    assert vars_req == ['x']


def test_validate_template_malformed():
    """Testa detecção de template malformado."""
    is_valid, vars_req = validate_template("Valor: {x}")
    assert is_valid is False
