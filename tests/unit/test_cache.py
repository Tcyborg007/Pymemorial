"""Testes de cache."""
from pymemorial.core.cache import ResultCache

def test_cache_set_get():
    """Testa armazenamento e recuperação."""
    cache = ResultCache()
    cache.set("key1", 42)
    assert cache.get("key1") == 42

def test_cache_maxsize():
    """Testa limite de tamanho."""
    cache = ResultCache(maxsize=2)
    cache.set("a", 1)
    cache.set("b", 2)
    cache.set("c", 3)  # Deve remover "a"
    
    assert cache.get("a") is None
    assert cache.get("c") == 3

def test_cache_clear():
    """Testa limpeza."""
    cache = ResultCache()
    cache.set("x", 100)
    cache.clear()
    assert cache.get("x") is None
