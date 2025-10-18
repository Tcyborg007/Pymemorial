# tests/test_smoke.py
import subprocess
import sys

def test_import_core():
    """Testa import e __version__ (cobertura direta)."""
    import pymemorial
    assert hasattr(pymemorial, "__version__")
    assert isinstance(pymemorial.__version__, str)
    assert len(pymemorial.__version__) > 0

def test_cli_version():
    """Testa CLI via subprocess (funcional, sem cobertura)."""
    out = subprocess.check_output(
        [sys.executable, "-m", "pymemorial.cli", "version"],
        stderr=subprocess.STDOUT
    ).decode()
    assert "PyMemorial" in out

def test_cli_import():
    """Testa import direto da CLI (cobertura)."""
    from pymemorial.cli import main
    assert callable(main)
