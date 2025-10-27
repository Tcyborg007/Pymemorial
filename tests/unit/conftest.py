import os
import pytest
from pymemorial.core.config import reset_config

@pytest.fixture(autouse=True)
def _reset_config_between_tests(monkeypatch):
    monkeypatch.setenv('PYMEMORIAL_DISABLE_AUTOLOAD', '1')
    reset_config()
    yield
    reset_config()
