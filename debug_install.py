#!/usr/bin/env python
"""Debug de compatibilidade do PyMemorial"""
import sys
import subprocess
import textwrap
from pathlib import Path

def p(msg=""):
    print(msg, flush=True)

def header(t):
    p("=" * 78)
    p(t)
    p("=" * 78)

def find_version_from_metadata(dist_name):
    try:
        from importlib.metadata import version
        return version(dist_name)
    except Exception:
        return None

def check_import(modname):
    try:
        mod = __import__(modname)
        return True, mod
    except Exception as e:
        return False, e

def find_spec_path(modname):
    try:
        from importlib.util import find_spec
        spec = find_spec(modname)
        return getattr(spec, "origin", None)
    except Exception:
        return None

def read_pyproject():
    here = Path.cwd()
    f = here / "pyproject.toml"
    if not f.exists():
        return None, "pyproject.toml não encontrado"
    try:
        import tomllib
        data = tomllib.loads(f.read_text(encoding="utf-8"))
        return data, None
    except Exception as e:
        return None, f"Falha ao ler pyproject.toml: {e}"

def py_range_ok(py_req_str):
    ver = sys.version_info
    ok_min = (ver.major > 3) or (ver.major == 3 and ver.minor >= 10)
    ok_max = not (ver.major > 3 or (ver.major == 3 and ver.minor >= 14))
    return ok_min and ok_max

def run_cli_version():
    try:
        out = subprocess.check_output([sys.executable, "-m", "pymemorial.cli", "version"], stderr=subprocess.STDOUT)
        return True, out.decode(errors="replace").strip()
    except subprocess.CalledProcessError as e:
        return False, e.output.decode(errors="replace") if e.output else str(e)
    except Exception as e:
        return False, str(e)

def check_optional(modnames):
    results = []
    for label, modname in modnames:
        ok, obj = check_import(modname)
        path = find_spec_path(modname) if ok else None
        base = modname.split(".")[0]
        ver_s = find_version_from_metadata(base)
        err_msg = None if ok else str(obj)
        results.append((label, ok, ver_s, path, err_msg))
    return results

def main():
    header("AMBIENTE")
    p(f"Python: {sys.version}")
    p(f"Exec:   {sys.executable}")
    p("sys.path (top 8):")
    for i, sp in enumerate(sys.path[:8], 1):
        p(f"  {i:2d}. {sp}")
    p()

    header("PYPROJECT E COMPATIBILIDADE")
    data, err = read_pyproject()
    if err:
        p(f"✗ {err}")
    else:
        pj = data.get("project", {})
        py_req = pj.get("requires-python", "<desconhecido>")
        p(f"requires-python: {py_req}")
        p(f"Compatível? {'SIM' if py_range_ok(py_req) else 'NÃO'}")
        opt = pj.get("optional-dependencies", {})
        if opt:
            p("optional-dependencies (extras):")
            for k, v in opt.items():
                p(f"  - {k}: {', '.join(v[:2])}{'...' if len(v) > 2 else ''}")
    p()

    header("PACOTE RAIZ E VERSÃO")
    ok_pkg, obj_pkg = check_import("pymemorial")
    if not ok_pkg:
        p(f"✗ Falha importando pymemorial: {obj_pkg}")
        sys.exit(1)
    
    import pymemorial
    p(f"Local: {getattr(pymemorial, '__file__', None)}")
    p(f"Tem __version__? {hasattr(pymemorial, '__version__')}")
    p(f"__version__: {getattr(pymemorial, '__version__', None)}")
    meta_ver = find_version_from_metadata("pymemorial")
    p(f"Versão via metadata: {meta_ver}")
    p()

    header("CLI")
    ok_cli, out = run_cli_version()
    p("Comando: python -m pymemorial.cli version")
    p(f"Saída: {out}")
    p(f"CLI OK? {'SIM' if ok_cli else 'NÃO'}")
    p()

    header("BIBLIOTECAS BASE")
    base_mods = [
        ("sympy", "sympy"),
        ("numpy", "numpy"), 
        ("pandas", "pandas"),
        ("scipy", "scipy"),
        ("matplotlib", "matplotlib"),
        ("pint", "pint"),
        ("rich", "rich"),
        ("click", "click"),
    ]
    
    for label, ok, ver, path, err in check_optional(base_mods):
        status = "OK" if ok else "NOK"
        p(f"{label:<12} -> {status:3} | ver={ver or 'N/A'}")
        if not ok and err:
            p(f"    Erro: {err}")
    p()

    header("BACKENDS E OPCIONAIS")
    py_maj, py_min = sys.version_info[:2]
    p(f"Python: {py_maj}.{py_min} ({'stack completo recomendado' if py_min == 12 else 'core leve'})")
    
    backends = [
        ("Pynite", "Pynite"),
        ("OpenSees", "openseespy.opensees"),
        ("SectionProps", "sectionproperties"), 
        ("PyVista", "pyvista"),
        ("Pyomo", "pyomo"),
        ("WeasyPrint", "weasyprint"),
    ]
    
    for label, ok, ver, path, err in check_optional(backends):
        status = "OK" if ok else "NOK"
        p(f"{label:<12} -> {status:3} | ver={ver or 'N/A'}")
        if not ok and err and "No module" not in err:
            p(f"    Erro: {err}")
    p()

    header("TESTES")
    tests_dir = Path("tests")
    if not tests_dir.exists():
        p("✗ Pasta 'tests' não encontrada")
    else:
        files = sorted(p.name for p in tests_dir.glob("test_*.py"))
        p(f"Arquivos: {', '.join(files) if files else '(nenhum)'}")
        if "test_smoke.py" in files:
            p("✓ test_smoke.py detectado")
    p()

    header("RESUMO")
    ok_core = hasattr(pymemorial, "__version__") and ok_cli
    p(f"✓ Core funcional: {'SIM' if ok_core else 'NÃO'}")
    p(f"✓ Python compatível: {'SIM' if py_range_ok('>=3.10,<3.14') else 'NÃO'}")
    p("Para backends completos: Python 3.12 + extras")
    p("Para core leve: Python 3.13 (sem algunos backends)")
    p("=" * 78)

if __name__ == "__main__":
    main()
