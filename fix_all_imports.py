#!/usr/bin/env python3
"""
Script de Correção Automática - PyMemorial v3.0
Corrige TODOS os problemas de import identificados.

Uso:
    python fix_all_imports.py

Author: PyMemorial Team
Date: 2025-10-21
"""

from pathlib import Path
import re


def fix_patterns_py():
    """FIX 1: Adicionar funções utilitárias ao patterns.py"""
    filepath = Path("src/pymemorial/recognition/patterns.py")
    
    if not filepath.exists():
        print(f"❌ {filepath} não encontrado!")
        return False
    
    content = filepath.read_text(encoding='utf-8')
    
    # Verificar se já tem as funções
    if "def find_variables" in content:
        print(f"✅ {filepath} já está correto!")
        return True
    
    # Adicionar funções utilitárias
    utility_functions = '''

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def find_variables(text: str) -> List[str]:
    """Find all variable names in text."""
    return VARNAME.findall(text)


def find_numbers(text: str) -> List[float]:
    """Find all numbers in text."""
    matches = NUMBER.findall(text)
    return [float(m[0] if isinstance(m, tuple) else m) for m in matches]


def find_placeholders(text: str) -> List[str]:
    """Find all {var} placeholders."""
    return PLACEHOLDER.findall(text)


def has_greek_letters(text: str) -> bool:
    """Check if text contains Greek letters."""
    return bool(GREEKLETTER.search(text))
'''
    
    # Inserir antes de __all__
    if "__all__" in content:
        content = content.replace("# ============================================================================\n# EXPORTS", 
                                  utility_functions + "\n\n# ============================================================================\n# EXPORTS")
    else:
        content += utility_functions
    
    # Atualizar __all__ se existir
    if "__all__ = [" in content:
        all_section = '''__all__ = [
    # Patterns v2.0
    'PLACEHOLDER',
    'VARNAME',
    'GREEKLETTER',
    'NUMBER',
    # Patterns v3.0
    'VALUEDISPLAYPATTERN',
    'FORMULADISPLAYPATTERN',
    'EQUATIONBLOCKPATTERN',
    # Functions
    'find_variables',
    'find_numbers',
    'find_placeholders',
    'has_greek_letters',
]'''
        content = re.sub(r'__all__\s*=\s*\[.*?\]', all_section, content, flags=re.DOTALL)
    
    # Backup
    backup = filepath.with_suffix('.py.backup')
    backup.write_text(content, encoding='utf-8')
    
    # Salvar
    filepath.write_text(content, encoding='utf-8')
    print(f"✅ Corrigido: {filepath}")
    print(f"   Backup: {backup}")
    return True


def fix_recognition_init():
    """FIX 2: Adicionar PLACEHOLDER aos exports do recognition/__init__.py"""
    filepath = Path("src/pymemorial/recognition/__init__.py")
    
    if not filepath.exists():
        print(f"❌ {filepath} não encontrado!")
        return False
    
    content = filepath.read_text(encoding='utf-8')
    
    # Verificar se já tem PLACEHOLDER nos imports
    if "from .patterns import (\n    # Patterns v2.0\n    PLACEHOLDER," in content:
        print(f"✅ {filepath} já está correto!")
        return True
    
    # Corrigir imports de patterns
    patterns_import_old = '''try:
    from .patterns import (
        find_variables,
        find_numbers,
        find_placeholders,
        has_greek_letters,
    )
    PATTERNS_V3_AVAILABLE = False'''
    
    patterns_import_new = '''try:
    from .patterns import (
        # Patterns v2.0
        PLACEHOLDER,
        VARNAME,
        GREEKLETTER,
        NUMBER,
        # Patterns v3.0
        VALUEDISPLAYPATTERN,
        FORMULADISPLAYPATTERN,
        EQUATIONBLOCKPATTERN,
        # Functions
        find_variables,
        find_numbers,
        find_placeholders,
        has_greek_letters,
    )
    PATTERNS_V3_AVAILABLE = True'''
    
    content = content.replace(patterns_import_old, patterns_import_new)
    
    # Adicionar aos exports __all__
    if '"find_variables",' in content and '"PLACEHOLDER",' not in content:
        # Adicionar PLACEHOLDER antes de find_variables
        content = content.replace(
            '    # ========== Utils ==========\n    "GreekSymbols",',
            '''    # ========== Utils ==========
    "GreekSymbols",
    "VariableParser",
    "ParsedVariable",
    "ASCII_TO_GREEK",
    "GREEK_TO_ASCII",
    
    # ========== Patterns v3.0 ==========
    "PLACEHOLDER",
    "VARNAME",
    "GREEKLETTER",
    "NUMBER",
    "VALUEDISPLAYPATTERN",
    "FORMULADISPLAYPATTERN",
    "EQUATIONBLOCKPATTERN",'''
        )
        
        # Remover duplicatas
        content = content.replace(
            '''    "VariableParser",
    "ParsedVariable",
    "ASCII_TO_GREEK",
    "GREEK_TO_ASCII",
    "find_variables",''',
            '''    "find_variables",'''
        )
    
    # Backup
    backup = filepath.with_suffix('.py.backup')
    backup.write_text(content, encoding='utf-8')
    
    # Salvar
    filepath.write_text(content, encoding='utf-8')
    print(f"✅ Corrigido: {filepath}")
    print(f"   Backup: {backup}")
    return True


def fix_units_py():
    """FIX 3: Adicionar strip_units ao units.py"""
    filepath = Path("src/pymemorial/core/units.py")
    
    if not filepath.exists():
        print(f"❌ {filepath} não encontrado!")
        return False
    
    content = filepath.read_text(encoding='utf-8')
    
    # Verificar se já tem strip_units
    if "def strip_units" in content:
        print(f"✅ {filepath} já tem strip_units!")
        return True
    
    # Adicionar função strip_units
    strip_units_code = '''

def strip_units(value):
    """
    Remove unidades de um valor, retornando apenas o número.
    
    Args:
        value: Valor com ou sem unidade (float, Quantity, Variable, etc.)
    
    Returns:
        float: Valor numérico sem unidade
    
    Examples:
        >>> strip_units(150.0)
        150.0
        >>> strip_units(Q_(150, 'kN'))
        150.0
    """
    # Se for Quantity do Pint
    if hasattr(value, 'magnitude'):
        return float(value.magnitude)
    
    # Se for Variable do PyMemorial
    if hasattr(value, 'value'):
        return float(value.value)
    
    # Se já for número
    try:
        return float(value)
    except (TypeError, ValueError):
        raise TypeError(f"Cannot strip units from {type(value).__name__}")
'''
    
    # Adicionar antes de __all__ ou no final
    if "__all__" in content:
        content = content.replace("__all__ = [", strip_units_code + "\n\n__all__ = [")
        # Adicionar ao __all__
        content = content.replace("__all__ = [", "__all__ = [\n    'strip_units',")
    else:
        content += strip_units_code
    
    # Backup
    backup = filepath.with_suffix('.py.backup')
    backup.write_text(content, encoding='utf-8')
    
    # Salvar
    filepath.write_text(content, encoding='utf-8')
    print(f"✅ Corrigido: {filepath}")
    print(f"   Backup: {backup}")
    return True


def main():
    """Executa todas as correções"""
    print("=" * 60)
    print("PyMemorial v3.0 - Correção Automática de Imports")
    print("=" * 60)
    print()
    
    results = []
    
    print("🔧 FIX 1: patterns.py")
    results.append(("patterns.py", fix_patterns_py()))
    print()
    
    print("🔧 FIX 2: recognition/__init__.py")
    results.append(("recognition/__init__.py", fix_recognition_init()))
    print()
    
    print("🔧 FIX 3: units.py")
    results.append(("units.py", fix_units_py()))
    print()
    
    print("=" * 60)
    print("RESUMO")
    print("=" * 60)
    for name, success in results:
        status = "✅" if success else "❌"
        print(f"{status} {name}")
    
    all_success = all(r[1] for r in results)
    if all_success:
        print()
        print("🎉 TODAS AS CORREÇÕES APLICADAS COM SUCESSO!")
        print()
        print("📝 Próximos passos:")
        print("   1. poetry run python -c \"from pymemorial.recognition import PLACEHOLDER; print('✅ OK!')\"")
        print("   2. poetry run pytest tests/ -v")
    else:
        print()
        print("⚠️  Algumas correções falharam. Verifique os erros acima.")
    
    return all_success


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
