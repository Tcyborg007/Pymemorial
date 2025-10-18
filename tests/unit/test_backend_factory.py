"""Testes do factory de backends."""
import pytest
from pymemorial.backends.factory import BackendFactory


def test_available_backends():
    """Testa listagem de backends disponíveis."""
    backends = BackendFactory.available_backends()
    assert isinstance(backends, list)
    # Pelo menos um deve estar disponível no ambiente de testes
    # (configurado no pyproject.toml com extras)


def test_create_invalid_backend():
    """Testa criação de backend inválido."""
    with pytest.raises(ValueError, match="não suportado"):
        BackendFactory.create("invalid_backend")


@pytest.mark.skipif(
    "pynite" not in BackendFactory.available_backends(),
    reason="PyNite não instalado"
)
def test_create_pynite():
    """Testa criação de Pynite backend."""
    backend = BackendFactory.create("pynite")
    assert backend.name == "Pynite"


@pytest.mark.skipif(
    "opensees" not in BackendFactory.available_backends(),
    reason="OpenSees não instalado"
)
def test_create_opensees():
    """Testa criação de OpenSees backend."""
    backend = BackendFactory.create("opensees")
    assert backend.name == "OpenSees"
