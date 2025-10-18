# tests/test_pynite_backend.py
import importlib.util
import pytest

pytestmark = pytest.mark.backends

PyNite_available = importlib.util.find_spec("PyNite") is not None

@pytest.mark.skipif(not PyNite_available, reason="PyNite indisponível neste Python")
def test_pynite_backend_initialize():
    from pymemorial.backends.pynite import PyNiteBackend
    be = PyNiteBackend()
    be.initialize()
    assert be._model is not None
