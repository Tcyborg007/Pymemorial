def pytest_configure(config):
    config.addinivalue_line("markers", "backends: testes que exigem backends (PyNite/OpenSees)")
    config.addinivalue_line("markers", "opensees: testes que exigem OpenSees/opstool")
