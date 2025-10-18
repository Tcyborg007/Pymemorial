# src/pymemorial/backends/pynite.py
from importlib.util import find_spec

class PyNiteBackend:
    """Stub do backend PyNite. Importa PyNite só quando necessário."""

    def __init__(self):
        self._model = None  # será uma instância de FEModel3D quando disponível

    def initialize(self):
        # Import local evita erro em ambientes onde PyNite não está presente (ex.: Python 3.13)
        try:
            from PyNite import FEModel3D
        except Exception as e:
            raise ImportError(
                "PyNite não está disponível neste ambiente. "
                "Use Python 3.12 ou instale o extra em um venv compatível."
            ) from e
        self._model = FEModel3D()

    @property
    def available(self) -> bool:
        """Retorna True se PyNite pode ser importado neste ambiente."""
        try:
            return find_spec("PyNite") is not None
        except Exception:
            return False
