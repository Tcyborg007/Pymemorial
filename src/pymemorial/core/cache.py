"""Cache de resultados para performance."""
from functools import lru_cache
from typing import Any, Hashable

class ResultCache:
    """Cache simples para resultados de equações."""
    
    def __init__(self, maxsize=128):
        self._cache = {}
        self.maxsize = maxsize
    
    def get(self, key: Hashable) -> Any:
        """Recupera valor do cache."""
        return self._cache.get(key)
    
    def set(self, key: Hashable, value: Any):
        """Armazena valor no cache."""
        if len(self._cache) >= self.maxsize:
            # Remove primeiro item (FIFO simples)
            self._cache.pop(next(iter(self._cache)))
        self._cache[key] = value
    
    def clear(self):
        """Limpa o cache."""
        self._cache.clear()
