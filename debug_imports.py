# debug_imports.py
"""
Script de Debug para Validação de Importações do PyMemorial.
Execute com: poetry run python debug_imports.py
"""

import logging
import sys
import importlib

# Configuração básica de logging para ver mensagens de erro/debug
logging.basicConfig(level=logging.DEBUG, format='%(levelname)-8s | %(name)-25s | %(message)s')
logger = logging.getLogger("DEBUG_TEST")

logger.info("Iniciando teste de importações...")
logger.info(f"Python sys.path: {sys.path}") # Mostra onde o Python está procurando

# --- Teste 1: Dependências Externas ---
logger.info("--- Teste 1: Dependências Externas ---")
try:
    import pint
    logger.info("✅ pint importado com sucesso.")
    # Verifica a versão (útil para debug)
    try: logger.info(f"   Versão do Pint: {pint.__version__}")
    except AttributeError: logger.warning("   Não foi possível obter a versão do Pint.")
except ImportError as e:
    logger.error(f"❌ FALHA ao importar pint: {e}")
    sys.exit(1) # Aborta se dependência crucial faltar

try:
    import sympy
    logger.info("✅ sympy importado com sucesso.")
    try: logger.info(f"   Versão do Sympy: {sympy.__version__}")
    except AttributeError: logger.warning("   Não foi possível obter a versão do Sympy.")
except ImportError as e:
    logger.error(f"❌ FALHA ao importar sympy: {e}")
    sys.exit(1)

# --- Teste 2: Módulos Individuais do Core ---
logger.info("--- Teste 2: Módulos Individuais do Core ---")
core_units = None
try:
    # Tenta importar o módulo específico
    core_units = importlib.import_module("pymemorial.core.units")
    logger.info("✅ pymemorial.core.units importado com sucesso.")
    # Verifica se a flag PINT_OK foi definida DENTRO dele
    if hasattr(core_units, 'PINT_OK'):
        logger.info(f"   Flag PINT_OK encontrada em units.py: {core_units.PINT_OK}")
    else:
        logger.error("   ❌ ERRO: Flag PINT_OK NÃO encontrada em units.py!")
    # Verifica se ureg foi criado
    if hasattr(core_units, 'ureg'):
         logger.info(f"   ureg encontrado em units.py: {type(core_units.ureg)}")
    else:
         logger.error("   ❌ ERRO: ureg NÃO encontrado em units.py!")

except ImportError as e:
    logger.error(f"❌ FALHA ao importar pymemorial.core.units: {e}")
except Exception as e:
    logger.error(f"❌ ERRO INESPERADO ao importar pymemorial.core.units: {e}", exc_info=True)

core_equation = None
try:
    core_equation = importlib.import_module("pymemorial.core.equation")
    logger.info("✅ pymemorial.core.equation importado com sucesso.")
    # Verifica se StepRegistry foi definido
    if hasattr(core_equation, 'StepRegistry'):
        logger.info(f"   StepRegistry encontrado em equation.py: {type(core_equation.StepRegistry)}")
    else:
        logger.error("   ❌ ERRO: StepRegistry NÃO encontrado em equation.py!")
except ImportError as e:
    logger.error(f"❌ FALHA ao importar pymemorial.core.equation: {e}")
except Exception as e:
    logger.error(f"❌ ERRO INESPERADO ao importar pymemorial.core.equation: {e}", exc_info=True)


# --- Teste 3: Pacote Core (__init__.py) ---
logger.info("--- Teste 3: Pacote Core (core/__init__.py) ---")
pymemorial_core = None
try:
    # Importa o pacote, o que executa o __init__.py
    pymemorial_core = importlib.import_module("pymemorial.core")
    logger.info("✅ pymemorial.core importado com sucesso.")

    # Verifica se as flags e classes foram exportadas corretamente pelo __init__.py
    if hasattr(pymemorial_core, 'PINT_AVAILABLE'):
        logger.info(f"   Flag PINT_AVAILABLE exportada por core/__init__.py: {pymemorial_core.PINT_AVAILABLE}")
    else:
        logger.error("   ❌ ERRO: Flag PINT_AVAILABLE NÃO exportada por core/__init__.py!")

    if hasattr(pymemorial_core, 'SYMPY_AVAILABLE'):
        logger.info(f"   Flag SYMPY_AVAILABLE exportada por core/__init__.py: {pymemorial_core.SYMPY_AVAILABLE}")
    else:
        logger.error("   ❌ ERRO: Flag SYMPY_AVAILABLE NÃO exportada por core/__init__.py!")

    if hasattr(pymemorial_core, 'StepRegistry'):
        logger.info(f"   StepRegistry exportado por core/__init__.py: {type(pymemorial_core.StepRegistry)}")
    else:
        logger.error("   ❌ ERRO: StepRegistry NÃO exportado por core/__init__.py!")

    if hasattr(pymemorial_core, 'ureg'):
         logger.info(f"   ureg exportado por core/__init__.py: {type(pymemorial_core.ureg)}")
    else:
         # Pode ser None se Pint não carregou, mas deve existir
         if hasattr(pymemorial_core,'PINT_AVAILABLE') and not pymemorial_core.PINT_AVAILABLE:
              logger.warning("   ureg é None (esperado, pois Pint não carregou).")
         else:
              logger.error("   ❌ ERRO: ureg NÃO exportado por core/__init__.py!")


except ImportError as e:
    logger.error(f"❌ FALHA CRÍTICA ao importar pymemorial.core: {e}", exc_info=True)
    sys.exit(1) # Aborta se o core falhar
except Exception as e:
    logger.error(f"❌ ERRO INESPERADO ao importar pymemorial.core: {e}", exc_info=True)
    sys.exit(1)


# --- Teste 4: Módulos Individuais do Editor ---
logger.info("--- Teste 4: Módulos Individuais do Editor ---")
editor_parser = None
try:
    editor_parser = importlib.import_module("pymemorial.editor.smart_parser")
    logger.info("✅ pymemorial.editor.smart_parser importado com sucesso.")
    if hasattr(editor_parser, 'SmartVariableParser'):
         logger.info(f"   Classe SmartVariableParser encontrada.")
    else:
         logger.error(f"   ❌ ERRO: Classe SmartVariableParser NÃO encontrada.")
except ImportError as e:
    logger.error(f"❌ FALHA ao importar pymemorial.editor.smart_parser: {e}")
except Exception as e:
    logger.error(f"❌ ERRO INESPERADO ao importar pymemorial.editor.smart_parser: {e}", exc_info=True)

editor_engine = None
try:
    editor_engine = importlib.import_module("pymemorial.editor.natural_engine")
    logger.info("✅ pymemorial.editor.natural_engine importado com sucesso.")
    # Verifica se ele conseguiu importar o core corretamente
    if hasattr(editor_engine, 'CORE_AVAILABLE'):
         logger.info(f"   Flag CORE_AVAILABLE em natural_engine.py: {editor_engine.CORE_AVAILABLE}")
         if not editor_engine.CORE_AVAILABLE:
              logger.error("   ❌ ERRO: natural_engine.py reporta que o Core NÃO está disponível!")
    else:
         logger.error("   ❌ ERRO: Flag CORE_AVAILABLE NÃO encontrada em natural_engine.py!")

    if hasattr(editor_engine, 'PINT_AVAILABLE'):
         logger.info(f"   Flag PINT_AVAILABLE em natural_engine.py: {editor_engine.PINT_AVAILABLE}")
    else:
         logger.error("   ❌ ERRO: Flag PINT_AVAILABLE NÃO encontrada em natural_engine.py!")

    if hasattr(editor_engine, 'NaturalMemorialEditor'):
         logger.info(f"   Classe NaturalMemorialEditor encontrada.")
    else:
         logger.error(f"   ❌ ERRO: Classe NaturalMemorialEditor NÃO encontrada.")

except ImportError as e:
    logger.error(f"❌ FALHA ao importar pymemorial.editor.natural_engine: {e}", exc_info=True)
except Exception as e:
    logger.error(f"❌ ERRO INESPERADO ao importar pymemorial.editor.natural_engine: {e}", exc_info=True)


# --- Teste 5: Pacote Editor (__init__.py) ---
logger.info("--- Teste 5: Pacote Editor (editor/__init__.py) ---")
pymemorial_editor = None
try:
    pymemorial_editor = importlib.import_module("pymemorial.editor")
    logger.info("✅ pymemorial.editor importado com sucesso.")

    # Verifica se ele exportou o NaturalMemorialEditor
    if hasattr(pymemorial_editor, 'NaturalMemorialEditor') and pymemorial_editor.NaturalMemorialEditor is not None:
        logger.info(f"   NaturalMemorialEditor exportado por editor/__init__.py: {type(pymemorial_editor.NaturalMemorialEditor)}")
    else:
        logger.error("   ❌ ERRO: NaturalMemorialEditor NÃO exportado ou é None em editor/__init__.py!")

    if hasattr(pymemorial_editor, 'EDITOR_AVAILABLE'):
         logger.info(f"   Flag EDITOR_AVAILABLE exportada por editor/__init__.py: {pymemorial_editor.EDITOR_AVAILABLE}")
         if not pymemorial_editor.EDITOR_AVAILABLE:
              logger.error("   ❌ ERRO: editor/__init__.py reporta que o Editor NÃO está disponível!")
    else:
         logger.error("   ❌ ERRO: Flag EDITOR_AVAILABLE NÃO exportada por editor/__init__.py!")

except ImportError as e:
    logger.error(f"❌ FALHA ao importar pymemorial.editor: {e}", exc_info=True)
except Exception as e:
    logger.error(f"❌ ERRO INESPERADO ao importar pymemorial.editor: {e}", exc_info=True)


# --- Teste 6: Instanciação (se o editor foi exportado) ---
logger.info("--- Teste 6: Instanciação do NaturalMemorialEditor ---")
if pymemorial_editor and hasattr(pymemorial_editor, 'NaturalMemorialEditor') and pymemorial_editor.NaturalMemorialEditor is not None:
    try:
        editor_instance = pymemorial_editor.NaturalMemorialEditor()
        logger.info(f"✅ Instância de NaturalMemorialEditor criada com sucesso: {type(editor_instance)}")
    except ImportError as ie: # Captura o ImportError que estava ocorrendo no __init__
         logger.error(f"❌ FALHA ao instanciar NaturalMemorialEditor: {ie}", exc_info=True)
    except Exception as e:
        logger.error(f"❌ ERRO INESPERADO ao instanciar NaturalMemorialEditor: {e}", exc_info=True)
else:
    logger.error("❌ Pulo da instanciação: NaturalMemorialEditor não foi carregado/exportado corretamente.")


logger.info("--- Teste de Importações Concluído ---")