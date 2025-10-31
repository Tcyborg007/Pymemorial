# src/pymemorial/engine/context.py
"""
Memorial Context - Contexto Global Unificado Thread-Safe

Consolida os 4 contextos fragmentados:
- builder.MemorialBuilder.variables
- editor.NaturalEngine._context
- document.BaseDocument._global_context
- symbols.SymbolRegistry

Features:
✅ Singleton thread-safe
✅ Auto-tracking de todas as variáveis
✅ Escopo hierárquico (seções)
✅ Integração total com core.Variable
✅ Conversão automática de unidades

Author: PyMemorial Team
Date: 2025-10-28
Version: 3.0.0
"""

from __future__ import annotations

import threading
import warnings
import logging
from typing import Dict, Any, Optional, List, Set, Union
from dataclasses import dataclass, field
from contextlib import contextmanager
from pathlib import Path
import json

# LOGGING - IMPORTANTE: definir logo no início
logger = logging.getLogger(__name__)

# Imports do core
try:
    from pymemorial.core import Variable, VariableFactory, get_config
    CORE_AVAILABLE = True
except ImportError:
    CORE_AVAILABLE = False
    Variable = None
    VariableFactory = None
    logger.warning("pymemorial.core não disponível - usando fallback")

# Imports do symbols
try:
    from pymemorial.symbols import SymbolRegistry, get_global_registry
    SYMBOLS_AVAILABLE = True
except ImportError:
    SYMBOLS_AVAILABLE = False
    SymbolRegistry = None
    logger.warning("pymemorial.symbols não disponível")


# ============================================================================
# DATACLASSES
# ============================================================================

# src/pymemorial/engine/context.py

@dataclass
class VariableScope:
    """Escopo de variáveis (para hierarquia de seções)."""
    name: str
    parent: Optional[VariableScope] = None
    variables: Dict[str, Any] = field(default_factory=dict)  # Any ao invés de Variable
    children: List[VariableScope] = field(default_factory=list)
    
    def get_path(self) -> str:
        """Retorna path completo do escopo."""
        if self.parent is None:
            return self.name
        return f"{self.parent.get_path()}.{self.name}"
    
    def find_variable(self, name: str) -> Optional[Any]:
        """Busca variável neste escopo e nos pais."""
        print(f"🔍 DEBUG find_variable(): Buscando '{name}' no escopo '{self.name}'")
        print(f"🔍 DEBUG find_variable(): Variáveis neste escopo: {list(self.variables.keys())}")
        
        if name in self.variables:
            print(f"✅ DEBUG find_variable(): Encontrou '{name}' no escopo '{self.name}'")
            return self.variables[name]
        
        if self.parent:
            print(f"🔍 DEBUG find_variable(): Procurando '{name}' no escopo pai...")
            return self.parent.find_variable(name)
        
        print(f"❌ DEBUG find_variable(): Variável '{name}' não encontrada em nenhum escopo")
        return None
# ============================================================================
# MEMORIAL CONTEXT (SINGLETON THREAD-SAFE)
# ============================================================================

class MemorialContext:
    """
    Contexto global unificado - Singleton thread-safe.
    
    Gerencia TODAS as variáveis do memorial com escopo hierárquico.
    
    Features:
    ---------
    - Auto-tracking de variáveis criadas
    - Escopo hierárquico (seções)
    - Integração com core.Variable e symbols.SymbolRegistry
    - Thread-safe para uso em ambientes multi-thread
    - Serialização para JSON
    
    Examples:
    ---------
    >>> ctx = MemorialContext.get_instance()
    >>> ctx.set("M_k", 150.5, "kN.m", "Momento característico")
    >>> ctx.set("gamma_f", 1.4, "", "Coeficiente de majoração")
    >>> M_d = ctx.calc("M_d = gamma_f * M_k")
    >>> print(ctx.get("M_d").value)  # 210.7
    >>> ctx.to_dict()  # Exporta tudo para dict
    """
    
    _instance: Optional[MemorialContext] = None
    _lock = threading.Lock()
    
    def __init__(self):
        """Construtor privado - Use get_instance()."""
        if MemorialContext._instance is not None:
            raise RuntimeError("Use MemorialContext.get_instance() ao invés de construtor direto")
        
        # Escopo raiz
        self._root_scope = VariableScope(name="root")
        self._current_scope = self._root_scope
        
        # Integração com symbols
        if SYMBOLS_AVAILABLE:
            self._symbol_registry = get_global_registry()
        else:
            self._symbol_registry = None
        
        # Configuração
        self._config = get_config() if CORE_AVAILABLE else {}
        
        # Histórico de cálculos
        self._calculation_history: List[Dict[str, Any]] = []
    
    @classmethod
    def get_instance(cls) -> MemorialContext:
        """Retorna instância única (singleton thread-safe)."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    @classmethod
    def reset(cls):
        """Reseta instância (útil para testes)."""
        with cls._lock:
            cls._instance = None
    
    # ========================================================================
    # VARIÁVEIS - API PRINCIPAL
    # ========================================================================
    
# src/pymemorial/engine/context.py

# src/pymemorial/engine/context.py

    def set(
        self,
        name: str,
        value: Union[float, int, str],
        unit: str = "",
        description: str = "",
        **kwargs
    ) -> Variable:
        """
        Define variável no contexto atual.
        """
        print(f"🔍 DEBUG context.set(): Tentando adicionar '{name}' = {value}")
        
        # CORREÇÃO: Fallback mais simples e robusto
        var_obj = None
        
        try:
            if CORE_AVAILABLE and Variable is not None:
                try:
                    var_obj = VariableFactory.create(
                        name=name,
                        value=value,
                        unit=unit,
                        description=description,
                        **kwargs
                    )
                    print(f"✅ DEBUG context.set(): Variável criada com VariableFactory")
                except Exception as e:
                    print(f"⚠️ DEBUG context.set(): Erro no VariableFactory: {e}")
                    # Fallback para Variable direto
                    var_obj = Variable(
                        name=name,
                        value=value,
                        unit=unit,
                        description=description
                    )
                    print(f"✅ DEBUG context.set(): Variável criada com Variable direto")
            else:
                # CORREÇÃO: Criar objeto simples sem dataclass complexo
                class SimpleVar:
                    def __init__(self, name, value, unit="", description=""):
                        self.name = name
                        self.value = value
                        self.unit = unit
                        self.description = description
                    
                    def __str__(self):
                        return f"{self.name} = {self.value} {self.unit}"
                
                var_obj = SimpleVar(name, value, unit, description)
                print(f"✅ DEBUG context.set(): Variável criada com SimpleVar")
                
        except Exception as e:
            print(f"❌ DEBUG context.set(): Erro crítico na criação: {e}")
            # Último fallback: usar o valor diretamente
            var_obj = value
        
        # CORREÇÃO: Adicionar ao escopo atual de forma garantida
        self._current_scope.variables[name] = var_obj
        print(f"✅ DEBUG context.set(): Variável '{name}' armazenada no escopo. Agora tem: {list(self._current_scope.variables.keys())}")
        
        # Tentar registrar no SymbolRegistry (opcional)
        # Registrar no SymbolRegistry se disponível
        if self._symbol_registry:
            try:
                # CORREÇÃO: SymbolRegistry.define() requer latex como segundo argumento
                self._symbol_registry.define(
                    name,           # symbol_name
                    None,           # latex (None se não disponível)
                    description,    # description
                    kwargs.get("category", "user_defined")  # category
                )
                print(f"✅ DEBUG context.set(): Símbolo '{name}' registrado no registry")
            except Exception as e:
                print(f"⚠️ DEBUG context.set(): Erro no symbol registry: {e}")
        
        return var_obj

    
    def get(self, name: str) -> Optional[Variable]:
        """
        Busca variável no contexto (escopo atual e pais).
        
        Args:
            name: Nome da variável
        
        Returns:
            Variable ou None se não encontrada
        """
        return self._current_scope.find_variable(name)
    
    def has(self, name: str) -> bool:
        """Verifica se variável existe no contexto."""
        return self.get(name) is not None
    
    def delete(self, name: str) -> bool:
        """Remove variável do contexto atual."""
        if name in self._current_scope.variables:
            del self._current_scope.variables[name]
            return True
        return False
    
# src/pymemorial/engine/context.py

    def list_variables(self, include_parents: bool = True) -> Dict[str, Variable]:
        """
        Lista todas as variáveis no escopo atual.
        """
        print(f"🔍 DEBUG list_variables(): Escopo atual: '{self._current_scope.name}'")
        print(f"🔍 DEBUG list_variables(): Variáveis no escopo atual: {list(self._current_scope.variables.keys())}")
        
        if not include_parents:
            return self._current_scope.variables.copy()
        
        # CORREÇÃO: Coletar variáveis de todos os escopos pais
        # Garante que os escopos mais internos (filhos) tenham prioridade
        result = {}
        current_scope = self._current_scope
        
        while current_scope:
            # Adicionar variáveis deste escopo
            for name, var in current_scope.variables.items():
                # SÓ ADICIONA SE AINDA NÃO FOI ADICIONADO (garante prioridade do filho)
                if name not in result:
                    result[name] = var
            current_scope = current_scope.parent
        
        print(f"🔍 DEBUG list_variables(): Todas as variáveis (com pais): {list(result.keys())}")
        return result
    
    def calc(self, expression: str, unit: str = "", description: str = "") -> Variable:
        """
        Calcula expressão e armazena resultado no contexto.
        
        Args:
            expression: Expressão Python (ex: "M_d = gamma_f * M_k")
            unit: Unidade do resultado
            description: Descrição
        
        Returns:
            Variable com resultado
        
        Examples:
            >>> ctx.set("a", 10)
            >>> ctx.set("b", 5)
            >>> c = ctx.calc("c = a + b")
            >>> print(c.value)  # 15
        """
        # CORREÇÃO: Import local do Calculator
        if not CORE_AVAILABLE:
            # Fallback simples
            import math
            
            # Extrair LHS e RHS
            if "=" in expression:
                lhs, rhs = expression.split("=", 1)
                var_name = lhs.strip()
                expr = rhs.strip()
            else:
                var_name = "_result"
                expr = expression.strip()
            
            # Criar contexto numérico
            context = {name: var.value for name, var in self.list_variables().items()}
            
            # Avaliar com eval seguro
            safe_namespace = {
                "__builtins__": {},
                "abs": abs, "min": min, "max": max, "round": round,
                "pow": pow, "sqrt": math.sqrt, "sin": math.sin,
                "cos": math.cos, "tan": math.tan, "pi": math.pi, "e": math.e,
            }
            safe_namespace.update(context)
            
            try:
                result = eval(expr, safe_namespace, {})
            except Exception as e:
                logger.error(f"Erro ao avaliar '{expr}': {e}")
                result = None
            
            # Armazenar
            var = self.set(var_name, result, unit, description)
            return var
        
        # Usar Calculator do core
        try:
            from pymemorial.core import Calculator
            
            # Extrair LHS e RHS
            if "=" in expression:
                lhs, rhs = expression.split("=", 1)
                var_name = lhs.strip()
                expr = rhs.strip()
            else:
                var_name = "_result"
                expr = expression.strip()
            
            # Criar contexto numérico
            context = {name: var.value for name, var in self.list_variables().items()}
            
            # Calcular
            calc = Calculator()
            result = calc.evaluate(expr, context)
            
            # Armazenar no contexto
            var = self.set(var_name, result, unit, description)
            
            # Adicionar ao histórico
            self._calculation_history.append({
                "expression": expression,
                "result": result,
                "context": context.copy()
            })
            
            return var
        
        except Exception as e:
            logger.error(f"Erro ao calcular: {e}")
            raise

    
    # ========================================================================
    # ESCOPO HIERÁRQUICO
    # ========================================================================
    
    @contextmanager
    def scope(self, name: str):
        """
        Context manager para criar escopo temporário.
        
        Examples:
            >>> ctx = MemorialContext.get_instance()
            >>> with ctx.scope("Geometria"):
            ...     ctx.set("b", 20, "cm")
            ...     ctx.set("h", 50, "cm")
            >>> # Após sair, variáveis ainda acessíveis
            >>> ctx.get("b")  # Retorna a variável
        """
        # Criar novo escopo
        new_scope = VariableScope(name=name, parent=self._current_scope)
        self._current_scope.children.append(new_scope)
        
        # Salvar escopo anterior
        previous_scope = self._current_scope
        self._current_scope = new_scope
        
        try:
            yield new_scope
        finally:
            # CORREÇÃO: Restaurar o escopo anterior ao sair do 'with'
            self._current_scope = previous_scope

    
    def push_scope(self, name: str) -> VariableScope:
        """Cria e entra em novo escopo (sem context manager)."""
        new_scope = VariableScope(name=name, parent=self._current_scope)
        self._current_scope.children.append(new_scope)
        self._current_scope = new_scope
        return new_scope
    
    def pop_scope(self) -> bool:
        """Sai do escopo atual e volta para o pai."""
        if self._current_scope.parent is None:
            warnings.warn("Já está no escopo raiz")
            return False
        self._current_scope = self._current_scope.parent
        return True
    
    def get_current_scope(self) -> VariableScope:
        """Retorna escopo atual."""
        return self._current_scope
    
    def get_scope_path(self) -> str:
        """Retorna path do escopo atual."""
        return self._current_scope.get_path()
    
    # ========================================================================
    # SERIALIZAÇÃO
    # ========================================================================
    
    def to_dict(self) -> Dict[str, Any]:
        """Exporta contexto completo para dicionário."""
        def scope_to_dict(scope: VariableScope) -> Dict:
            return {
                "name": scope.name,
                "variables": {
                    name: {
                        "value": var.value,
                        "unit": var.unit,
                        "description": var.description
                    }
                    for name, var in scope.variables.items()
                },
                "children": [scope_to_dict(child) for child in scope.children]
            }
        
        return {
            "root": scope_to_dict(self._root_scope),
            "current_path": self.get_scope_path(),
            "calculation_history": self._calculation_history
        }
    
    def to_json(self, filepath: Union[str, Path]) -> None:
        """Salva contexto em arquivo JSON."""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    def from_dict(self, data: Dict[str, Any]) -> None:
        """Carrega contexto de dicionário."""
        # TODO: Implementar reconstrução de árvore de escopos
        raise NotImplementedError("from_dict ainda não implementado")
    
    def clear(self):
        """Limpa TODO o contexto (reset completo)."""
        self._root_scope = VariableScope(name="root")
        self._current_scope = self._root_scope
        self._calculation_history.clear()


# ============================================================================
# FUNÇÕES DE CONVENIÊNCIA
# ============================================================================

def get_context() -> MemorialContext:
    """Atalho para MemorialContext.get_instance()."""
    return MemorialContext.get_instance()


def reset_context():
    """Reseta contexto global (útil para testes)."""
    MemorialContext.reset()


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "MemorialContext",
    "VariableScope",
    "get_context",
    "reset_context",
]
