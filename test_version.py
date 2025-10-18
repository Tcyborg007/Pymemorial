import subprocess
import sys

def test_import_core():
    import pymemorial
    assert hasattr(pymemorial, "__version__")

def test_cli_version():
    out = subprocess.check_output([sys.executable, "-m", "pymemorial.cli", "version"]).decode()
    assert "PyMemorial" in out
