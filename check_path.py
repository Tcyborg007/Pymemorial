# check_path.py
import pymemorial
import inspect
import sys

print("="*60)
print(f"VERSÃO DO PYTHON: {sys.version}")
print("="*60)
print("CAMINHOS DE IMPORTAÇÃO (sys.path):")
for p in sys.path:
    print(f"  - {p}")
print("="*60)

print("LOCALIZANDO ARQUIVOS DA BIBLIOTECA 'pymemorial'...")
print("-"*60)

try:
    print(f"🔍 Verificando 'pymemorial' (pacote raiz):")
    print(f"  > {pymemorial}")
    print(f"  > ARQUIVO: {inspect.getfile(pymemorial)}")
except Exception as e:
    print(f"  ❌ FALHA: {e}")

print("-"*60)

try:
    from pymemorial import eng_memorial
    print(f"🔍 Verificando 'pymemorial.eng_memorial':")
    print(f"  > {eng_memorial}")
    print(f"  > ARQUIVO: {inspect.getfile(eng_memorial)}")
except Exception as e:
    print(f"  ❌ FALHA: {e}")

print("-"*60)

try:
    from pymemorial.engine import context
    print(f"🔍 Verificando 'pymemorial.engine.context':")
    print(f"  > {context}")
    print(f"  > ARQUIVO: {inspect.getfile(context)}")
except Exception as e:
    print(f"  ❌ FALHA: {e}")

print("="*60)