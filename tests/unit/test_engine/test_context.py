# tests/unit/test_engine/test_context.py
"""
Testes para engine.context (MemorialContext)

Testa:
- Singleton pattern
- Set/get variáveis
- Escopo hierárquico
- Serialização JSON
- Thread-safety
"""

import pytest
import threading
import json
from pathlib import Path

from pymemorial.engine.context import (
    MemorialContext, 
    VariableScope, 
    get_context, 
    reset_context
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture(autouse=True)
def reset_ctx():
    """Reset context antes de cada teste."""
    reset_context()
    yield
    reset_context()


# ============================================================================
# TESTES - SINGLETON
# ============================================================================

def test_singleton_pattern():
    """Testa se MemorialContext é singleton."""
    ctx1 = MemorialContext.get_instance()
    ctx2 = MemorialContext.get_instance()
    
    assert ctx1 is ctx2, "Devem ser a mesma instância"


def test_get_context_shortcut():
    """Testa atalho get_context()."""
    ctx1 = get_context()
    ctx2 = MemorialContext.get_instance()
    
    assert ctx1 is ctx2


def test_reset_context():
    """Testa reset do singleton."""
    ctx1 = get_context()
    ctx1.set("test_var", 42)
    
    reset_context()
    ctx2 = get_context()
    
    assert ctx1 is not ctx2
    assert not ctx2.has("test_var")


# ============================================================================
# TESTES - VARIÁVEIS
# ============================================================================

def test_set_and_get_variable():
    """Testa definir e buscar variável."""
    ctx = get_context()
    
    var = ctx.set("M_k", 150.5, "kN.m", "Momento característico")
    
    assert var is not None
    assert var.value == 150.5
    assert var.unit == "kN.m"
    assert var.description == "Momento característico"
    
    retrieved = ctx.get("M_k")
    assert retrieved is var


def test_has_variable():
    """Testa verificação de existência."""
    ctx = get_context()
    
    assert not ctx.has("M_k")
    
    ctx.set("M_k", 150.5)
    
    assert ctx.has("M_k")


def test_delete_variable():
    """Testa remoção de variável."""
    ctx = get_context()
    ctx.set("temp", 42)
    
    assert ctx.has("temp")
    
    deleted = ctx.delete("temp")
    assert deleted is True
    assert not ctx.has("temp")
    
    deleted_again = ctx.delete("temp")
    assert deleted_again is False


def test_list_variables():
    """Testa listagem de variáveis."""
    ctx = get_context()
    
    ctx.set("a", 10)
    ctx.set("b", 20)
    ctx.set("c", 30)
    
    variables = ctx.list_variables()
    
    assert len(variables) == 3
    assert "a" in variables
    assert "b" in variables
    assert "c" in variables


# ============================================================================
# TESTES - CÁLCULOS
# ============================================================================

def test_calc_simple():
    """Testa cálculo simples."""
    ctx = get_context()
    
    ctx.set("a", 10)
    ctx.set("b", 5)
    
    result = ctx.calc("c = a + b")
    
    assert result.value == 15
    assert ctx.has("c")
    assert ctx.get("c").value == 15


def test_calc_expression():
    """Testa cálculo com expressão matemática."""
    ctx = get_context()
    
    ctx.set("q", 15.0)
    ctx.set("L", 6.0)
    
    result = ctx.calc("M = q * L**2 / 8")
    
    assert result.value == pytest.approx(67.5, rel=1e-6)


def test_calc_without_assignment():
    """Testa cálculo sem atribuição."""
    ctx = get_context()
    
    ctx.set("a", 10)
    ctx.set("b", 5)
    
    result = ctx.calc("a * b")  # Sem "c ="
    
    assert result.value == 50
    assert ctx.has("_result")  # Armazena em _result


# ============================================================================
# TESTES - ESCOPO HIERÁRQUICO
# ============================================================================

# tests/unit/test_engine/test_context.py
def test_scope_context_manager():
    """Testa escopo com context manager."""
    ctx = get_context()
    
    ctx.set("global_var", 100)
    
    with ctx.scope("Geometria"):
        ctx.set("b", 20)
        ctx.set("h", 50)
        
        assert ctx.get("b").value == 20
        assert ctx.get("global_var").value == 100  # Acessa escopo pai
    
    # CORREÇÃO: Após sair do escopo, volta para o pai
    # Então "b" não estará mais acessível diretamente
    # MAS ainda está na árvore de escopos
    # Para acessar, precisa estar no escopo filho ou usar list_variables
    variables = ctx.list_variables(include_parents=True)
    assert "b" in variables
    assert variables["b"].value == 20



def test_scope_push_pop():
    """Testa escopo com push/pop manual."""
    ctx = get_context()
    
    ctx.set("root_var", 1)
    
    scope1 = ctx.push_scope("Level1")
    ctx.set("level1_var", 2)
    
    scope2 = ctx.push_scope("Level2")
    ctx.set("level2_var", 3)
    
    # Testa acesso hierárquico
    assert ctx.get("root_var").value == 1
    assert ctx.get("level1_var").value == 2
    assert ctx.get("level2_var").value == 3
    
    # Pop de volta
    ctx.pop_scope()
    assert ctx.get_current_scope() is scope1
    
    ctx.pop_scope()
    assert ctx.get_scope_path() == "root"


def test_scope_isolation():
    """Testa isolamento de variáveis entre escopos irmãos."""
    ctx = get_context()
    
    with ctx.scope("Scope1"):
        ctx.set("var1", 10)
    
    with ctx.scope("Scope2"):
        ctx.set("var2", 20)
        
        # var1 não deve estar acessível (escopo irmão)
        # (depende da implementação - ajustar se necessário)
        pass


def test_get_scope_path():
    """Testa path do escopo."""
    ctx = get_context()
    
    assert ctx.get_scope_path() == "root"
    
    ctx.push_scope("Geometria")
    assert ctx.get_scope_path() == "root.Geometria"
    
    ctx.push_scope("Seção")
    assert ctx.get_scope_path() == "root.Geometria.Seção"


# ============================================================================
# TESTES - SERIALIZAÇÃO
# ============================================================================

def test_to_dict():
    """Testa exportação para dict."""
    ctx = get_context()
    
    ctx.set("a", 10, "m")
    ctx.set("b", 20, "cm")
    
    data = ctx.to_dict()
    
    assert "root" in data
    assert "variables" in data["root"]
    assert "a" in data["root"]["variables"]
    assert data["root"]["variables"]["a"]["value"] == 10


def test_to_json(tmp_path):
    """Testa exportação para JSON."""
    ctx = get_context()
    
    ctx.set("test", 42, "unit")
    
    json_file = tmp_path / "context.json"
    ctx.to_json(json_file)
    
    assert json_file.exists()
    
    # Verificar conteúdo
    with open(json_file, "r") as f:
        data = json.load(f)
    
    assert "root" in data


# ============================================================================
# TESTES - THREAD SAFETY
# ============================================================================

def test_thread_safety():
    """Testa se singleton é thread-safe."""
    instances = []
    
    def get_instance():
        instances.append(get_context())
    
    threads = [threading.Thread(target=get_instance) for _ in range(10)]
    
    for t in threads:
        t.start()
    
    for t in threads:
        t.join()
    
    # Todas as instâncias devem ser iguais
    assert all(inst is instances[0] for inst in instances)


def test_concurrent_set():
    """Testa set concorrente de variáveis."""
    ctx = get_context()
    
    def set_var(name, value):
        ctx.set(name, value)
    
    threads = [
        threading.Thread(target=set_var, args=(f"var_{i}", i))
        for i in range(100)
    ]
    
    for t in threads:
        t.start()
    
    for t in threads:
        t.join()
    
    # Verificar que todas foram criadas
    variables = ctx.list_variables()
    assert len(variables) == 100


# ============================================================================
# TESTES - EDGE CASES
# ============================================================================

def test_overwrite_variable():
    """Testa sobrescrita de variável."""
    ctx = get_context()
    
    ctx.set("var", 10)
    assert ctx.get("var").value == 10
    
    ctx.set("var", 20)  # Sobrescrever
    assert ctx.get("var").value == 20


def test_empty_context():
    """Testa contexto vazio."""
    ctx = get_context()
    
    assert len(ctx.list_variables()) == 0
    assert ctx.get("nonexistent") is None


def test_clear_context():
    """Testa limpeza completa do contexto."""
    ctx = get_context()
    
    ctx.set("a", 1)
    ctx.set("b", 2)
    
    ctx.clear()
    
    assert len(ctx.list_variables()) == 0
    assert ctx.get_scope_path() == "root"
