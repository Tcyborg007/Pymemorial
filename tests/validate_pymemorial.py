# tests/validate_pymemorial.py
"""
Script de Validação Simplificado - PyMemorial v1.0
Captura TODOS os erros e sempre gera output
"""

import sys
from pathlib import Path

# Adicionar src ao path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

print("="*80)
print("VALIDAÇÃO PYMEMORIAL v1.0")
print(f"Python: {sys.version}")
print(f"Path: {sys.path[0]}")
print("="*80)

def safe_test(test_name, test_func):
    """Executa teste com captura de erros."""
    print(f"\n### {test_name} ###")
    try:
        result = test_func()
        if result:
            print(f"✅ PASSOU")
        else:
            print(f"❌ FALHOU")
        return result
    except Exception as e:
        print(f"❌ ERRO: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

# ============================================================================
# TESTE 1: Import pymemorial
# ============================================================================
def test_import_pymemorial():
    try:
        import pymemorial
        print(f"  Pymemorial path: {pymemorial.__file__}")
        return True
    except ImportError as e:
        print(f"  ImportError: {e}")
        return False

# ============================================================================
# TESTE 2: Import recognition
# ============================================================================
def test_import_recognition():
    try:
        from pymemorial.recognition import VariableParser, TextProcessor
        print(f"  VariableParser: {VariableParser}")
        print(f"  TextProcessor: {TextProcessor}")
        return True
    except ImportError as e:
        print(f"  ImportError: {e}")
        return False

# ============================================================================
# TESTE 3: Import core
# ============================================================================
def test_import_core():
    try:
        from pymemorial.core import Variable, Equation
        print(f"  Variable: {Variable}")
        print(f"  Equation: {Equation}")
        return True
    except ImportError as e:
        print(f"  ImportError: {e}")
        return False

# ============================================================================
# TESTE 4: Variable tem .symbol?
# ============================================================================
def test_variable_symbol():
    try:
        from pymemorial.core import Variable
        v = Variable('test', 1.0, 'm')
        has_symbol = hasattr(v, 'symbol')
        print(f"  Variable.symbol existe: {has_symbol}")
        print(f"  Atributos: {[a for a in dir(v) if not a.startswith('_')][:10]}")
        return True  # Sempre passa (só queremos saber se tem)
    except Exception as e:
        print(f"  Erro: {e}")
        return False

# ============================================================================
# TESTE 5: Import document
# ============================================================================
def test_import_document():
    try:
        from pymemorial.document import BaseDocument, Memorial
        print(f"  BaseDocument: {BaseDocument}")
        print(f"  Memorial: {Memorial}")
        return True
    except ImportError as e:
        print(f"  ImportError: {e}")
        return False

# ============================================================================
# TESTE 6: Import sections
# ============================================================================
def test_import_sections():
    try:
        from pymemorial.sections import SectionFactory
        print(f"  SectionFactory: {SectionFactory}")
        return True
    except ImportError as e:
        print(f"  ImportError: {e}")
        return False

# ============================================================================
# TESTE 7: Import builder
# ============================================================================
def test_import_builder():
    try:
        from pymemorial.builder import MemorialBuilder
        print(f"  MemorialBuilder: {MemorialBuilder}")
        return True
    except ImportError as e:
        print(f"  ImportError: {e}")
        return False

# ============================================================================
# TESTE 8: Import backends
# ============================================================================
def test_import_backends():
    try:
        from pymemorial.backends import BackendFactory
        print(f"  BackendFactory: {BackendFactory}")
        return True
    except ImportError as e:
        print(f"  ImportError: {e}")
        return False

# ============================================================================
# EXECUTAR TESTES
# ============================================================================
if __name__ == '__main__':
    results = {}
    
    results['1. Import pymemorial'] = safe_test('TESTE 1: Import pymemorial', test_import_pymemorial)
    results['2. Import recognition'] = safe_test('TESTE 2: Import recognition', test_import_recognition)
    results['3. Import core'] = safe_test('TESTE 3: Import core', test_import_core)
    results['4. Variable.symbol?'] = safe_test('TESTE 4: Variable.symbol?', test_variable_symbol)
    results['5. Import document'] = safe_test('TESTE 5: Import document', test_import_document)
    results['6. Import sections'] = safe_test('TESTE 6: Import sections', test_import_sections)
    results['7. Import builder'] = safe_test('TESTE 7: Import builder', test_import_builder)
    results['8. Import backends'] = safe_test('TESTE 8: Import backends', test_import_backends)
    
    # Relatório
    print("\n" + "="*80)
    print("RELATÓRIO FINAL")
    print("="*80)
    
    passed = sum(1 for r in results.values() if r)
    failed = sum(1 for r in results.values() if not r)
    
    print(f"\n✅ PASSOU: {passed}/{len(results)}")
    print(f"❌ FALHOU: {failed}/{len(results)}")
    
    print("\nDetalhes:")
    for test_name, result in results.items():
        status = "✅" if result else "❌"
        print(f"  {status} {test_name}")
    
    print("\n" + "="*80)
    print("FIM DA VALIDAÇÃO")
    print("="*80)
