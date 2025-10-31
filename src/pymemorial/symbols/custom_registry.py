# src/pymemorial/symbols/custom_registry.py

"""
Registry inteligente de símbolos customizados.

AUTO-APRENDIZADO:
- Detecta símbolos automaticamente do código
- Gera LaTeX inteligente (gamma_s → \\gamma_{s})
- Persiste em ~/.pymemorial/symbols.json

INTEGRAÇÃO:
- Funciona com ast_parser para conversão LaTeX
- Usa config.py para greek_style
- Suporta busca fuzzy e categorização

Examples:
    >>> from pymemorial.symbols import get_global_registry
    >>> 
    >>> registry = get_global_registry()
    >>> 
    >>> # Definir símbolo manualmente
    >>> registry.define('M_d', latex=r'M_{d}', description='Momento de cálculo')
    >>> 
    >>> # Auto-aprender de código
    >>> code = '''
    ... gamma_c = 1.4
    ... f_cd = 30
    ... '''
    >>> registry.learn_from_code(code)
    >>> 
    >>> # Buscar símbolo
    >>> sym = registry.get('gamma_c')
    >>> print(sym.latex)
    \\gamma_{c}
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

from pymemorial.core.config import get_config
from pymemorial.recognition.greek import ASCII_TO_GREEK
from pymemorial.recognition.ast_parser import PyMemorialASTParser, ParsedAssignment
__all__ = [
    'Symbol',
    'SymbolRegistry',
    'RegistryError',
    'get_global_registry',
    'reset_global_registry',
    'get_registry'  # ← ADICIONADO!
]

# =============================================================================
# EXCEÇÕES
# =============================================================================

class RegistryError(Exception):
    """Erro de operação do registry de símbolos."""
    pass

# =============================================================================
# ESTRUTURAS DE DADOS
# =============================================================================

@dataclass
class Symbol:
    """
    Representa um símbolo matemático/de engenharia.
    
    Attributes:
        name: Nome Python do símbolo (ex: gamma_c, M_d)
        latex: Representação LaTeX (ex: \\gamma_{c}, M_{d})
        description: Descrição do símbolo (opcional)
        category: Categoria/norma (ex: 'nbr6118', 'structural')
        aliases: Nomes alternativos (ex: ['γc', 'gama_c'])
    
    Examples:
        >>> sym = Symbol(
        ...     name='gamma_c',
        ...     latex=r'\\gamma_{c}',
        ...     description='Coeficiente de majoração do concreto'
        ... )
    """
    name: str
    latex: str
    description: Optional[str] = None
    category: Optional[str] = None
    aliases: List[str] = field(default_factory=list)
    
    def __repr__(self) -> str:
        return f"Symbol(name='{self.name}', latex='{self.latex}')"

# =============================================================================
# REGISTRY PRINCIPAL
# =============================================================================

class SymbolRegistry:
    """
    Registry inteligente de símbolos customizados.
    
    FUNCIONALIDADES:
    - define(): Definir símbolo manualmente
    - learn_from_code(): Auto-aprender símbolos do código
    - get(): Obter símbolo por nome ou alias
    - has(): Verificar se símbolo existe
    - search(): Buscar símbolos (fuzzy)
    - save_to_file/load_from_file(): Persistência JSON
    
    Attributes:
        symbols: Dicionário de símbolos {name: Symbol}
        alias_map: Mapa de aliases para nomes reais
        registry_file: Path do arquivo JSON de persistência
        auto_save: Salvar automaticamente após mudanças
    
    Examples:
        >>> registry = SymbolRegistry()
        >>> registry.define('M_d', latex=r'M_{d}')
        >>> registry.has('M_d')
        True
    """
    
    # Símbolos builtin Python para ignorar
    PYTHON_BUILTINS = {
        'abs', 'all', 'any', 'ascii', 'bin', 'bool', 'breakpoint', 'bytearray',
        'bytes', 'callable', 'chr', 'classmethod', 'compile', 'complex',
        'delattr', 'dict', 'dir', 'divmod', 'enumerate', 'eval', 'exec',
        'filter', 'float', 'format', 'frozenset', 'getattr', 'globals',
        'hasattr', 'hash', 'help', 'hex', 'id', 'input', 'int', 'isinstance',
        'issubclass', 'iter', 'len', 'list', 'locals', 'map', 'max',
        'memoryview', 'min', 'next', 'object', 'oct', 'open', 'ord', 'pow',
        'print', 'property', 'range', 'repr', 'reversed', 'round', 'set',
        'setattr', 'slice', 'sorted', 'staticmethod', 'str', 'sum', 'super',
        'tuple', 'type', 'vars', 'zip', '__import__', '__name__', '__file__',
        # Módulos comuns
        'math', 'numpy', 'np', 'scipy', 'sp', 'sqrt', 'sin', 'cos', 'tan',
        'exp', 'log', 'log10', 'pi', 'e'
    }
    
    def __init__(
        self,
        registry_file: Optional[Path] = None,
        auto_save: bool = False
    ):
        """
        Inicializa registry de símbolos.
        
        Args:
            registry_file: Path do arquivo JSON (padrão: ~/.pymemorial/symbols.json)
            auto_save: Salvar automaticamente após mudanças
        """
        self.symbols: Dict[str, Symbol] = {}
        self.alias_map: Dict[str, str] = {}  # alias -> real_name
        
        # Path do arquivo de persistência
        if registry_file is None:
            config_dir = Path.home() / '.pymemorial'
            self.registry_file = config_dir / 'symbols.json'
        else:
            self.registry_file = registry_file
        
        self.auto_save = auto_save
        
        # Tentar carregar símbolos existentes
        if self.registry_file.exists():
            try:
                self.load_from_file(self.registry_file)
            except Exception:
                # Se falhar, continuar com registry vazio
                pass
    
    def define(
        self,
        name: str,
        latex: str,
        description: Optional[str] = None,
        category: Optional[str] = None,
        aliases: Optional[List[str]] = None,
        overwrite: bool = False
    ) -> Symbol:
        """
        Define um símbolo manualmente.
        
        Args:
            name: Nome Python do símbolo
            latex: Representação LaTeX
            description: Descrição (opcional)
            category: Categoria (opcional)
            aliases: Nomes alternativos (opcional)
            overwrite: Sobrescrever se já existir
        
        Returns:
            Symbol criado
        
        Raises:
            RegistryError: Se símbolo já existe e overwrite=False
        
        Examples:
            >>> registry = SymbolRegistry()
            >>> registry.define('M_d', latex=r'M_{d}', description='Momento')
        """
        if self.has(name) and not overwrite:
            raise RegistryError(
                f"Symbol '{name}' is already defined. "
                f"Use overwrite=True to replace."
            )
        
        if aliases is None:
            aliases = []
        
        # Criar símbolo
        symbol = Symbol(
            name=name,
            latex=latex,
            description=description,
            category=category,
            aliases=aliases
        )
        
        # Registrar símbolo
        self.symbols[name] = symbol
        
        # Registrar aliases
        for alias in aliases:
            self.alias_map[alias] = name
        
        # Auto-save se ativado
        if self.auto_save:
            self.save_to_file(self.registry_file)
        
        return symbol
    
    def get(self, name: str) -> Optional[Symbol]:
        """
        Obtém símbolo por nome ou alias.
        
        Args:
            name: Nome ou alias do símbolo
        
        Returns:
            Symbol se encontrado, None caso contrário
        
        Examples:
            >>> registry = SymbolRegistry()
            >>> registry.define('gamma_c', latex=r'\\gamma_{c}', aliases=['γc'])
            >>> sym = registry.get('γc')  # Busca por alias
            >>> sym.name
            'gamma_c'
        """
        # Tentar como nome direto
        if name in self.symbols:
            return self.symbols[name]
        
        # Tentar como alias
        if name in self.alias_map:
            real_name = self.alias_map[name]
            return self.symbols.get(real_name)
        
        return None
    
    def has(self, name: str) -> bool:
        """
        Verifica se símbolo existe (por nome ou alias).
        
        Args:
            name: Nome ou alias
        
        Returns:
            True se existe, False caso contrário
        """
        return self.get(name) is not None
    
    def list_all(self) -> List[Symbol]:
        """
        Lista todos os símbolos registrados.
        
        Returns:
            Lista de Symbol
        
        Examples:
            >>> registry = SymbolRegistry()
            >>> registry.define('M_d', latex=r'M_{d}')
            >>> registry.define('V_d', latex=r'V_{d}')
            >>> len(registry.list_all())
            2
        """
        return list(self.symbols.values())
    
    def learn_from_code(self, code: str) -> List[Symbol]:
        """
        Aprende símbolos automaticamente do código Python.
        
        LÓGICA DE APRENDIZADO:
        1. Parse código com ast_parser
        2. Extrair variáveis de todas expressões
        3. Filtrar builtins e módulos
        4. Gerar LaTeX automático para novos símbolos
        5. Registrar símbolos aprendidos
        
        Args:
            code: Código Python fonte
        
        Returns:
            Lista de Symbol aprendidos (novos)
        
        Examples:
            >>> registry = SymbolRegistry()
            >>> code = '''
            ... gamma_c = 1.4
            ... M_d = 150
            ... '''
            >>> learned = registry.learn_from_code(code)
            >>> len(learned)
            2
        """
        parser = PyMemorialASTParser()
        
        # Parse código
        assignments = parser.parse_code_block(code)
        
        # Coletar todas as variáveis
        all_vars = set()
        
        for assign in assignments:
            # Adicionar LHS
            all_vars.add(assign.lhs)
            
            # Adicionar variáveis do RHS
            rhs_vars = parser.extract_variables(assign.rhs_symbolic)
            all_vars.update(rhs_vars)
        
        # Filtrar builtins e já existentes
        new_symbols = []
        
        for var_name in all_vars:
            # Ignorar builtins
            if var_name in self.PYTHON_BUILTINS:
                continue
            
            # Ignorar já existentes
            if self.has(var_name):
                continue
            
            # Gerar LaTeX automático
            latex = self._auto_generate_latex(var_name)
            
            # Definir símbolo
            symbol = self.define(
                name=var_name,
                latex=latex,
                description=f'Auto-learned from code',
                overwrite=False
            )
            
            new_symbols.append(symbol)
        
        return new_symbols
    
    def search(
        self,
        query: Optional[str] = None,
        category: Optional[str] = None
    ) -> List[Symbol]:
        """
        Busca símbolos (fuzzy search).
        
        Args:
            query: Texto para buscar no nome (opcional)
            category: Filtrar por categoria (opcional)
        
        Returns:
            Lista de Symbol encontrados
        
        Examples:
            >>> registry = SymbolRegistry()
            >>> registry.define('M_d', latex=r'M_{d}', category='structural')
            >>> registry.define('M_k', latex=r'M_{k}', category='structural')
            >>> results = registry.search('M')
            >>> len(results)
            2
        """
        results = []
        
        for symbol in self.symbols.values():
            # Filtro de categoria
            if category is not None and symbol.category != category:
                continue
            
            # Filtro de query (substring no nome)
            if query is not None and query not in symbol.name:
                continue
            
            results.append(symbol)
        
        return results
    
    def save_to_file(self, filepath: Optional[Path] = None) -> None:
        """
        Salva registry em arquivo JSON.
        
        Args:
            filepath: Path do arquivo (padrão: self.registry_file)
        
        Examples:
            >>> registry = SymbolRegistry()
            >>> registry.define('M_d', latex=r'M_{d}')
            >>> registry.save_to_file(Path('symbols.json'))
        """
        if filepath is None:
            filepath = self.registry_file
        
        # Criar diretório se não existir
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Converter símbolos para dict
        data = {}
        for name, symbol in self.symbols.items():
            data[name] = asdict(symbol)
        
        # Salvar JSON
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def load_from_file(self, filepath: Path) -> None:
        """
        Carrega registry de arquivo JSON.
        
        Args:
            filepath: Path do arquivo
        
        Examples:
            >>> registry = SymbolRegistry()
            >>> registry.load_from_file(Path('symbols.json'))
        """
        if not filepath.exists():
            return
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Restaurar símbolos
        for name, symbol_dict in data.items():
            symbol = Symbol(**symbol_dict)
            self.symbols[name] = symbol
            
            # Restaurar aliases
            for alias in symbol.aliases:
                self.alias_map[alias] = name
    
    # =========================================================================
    # MÉTODOS AUXILIARES PRIVADOS
    # =========================================================================
    
    def _auto_generate_latex(self, name: str) -> str:
        """
        Gera LaTeX automaticamente para símbolo.
        
        REGRAS:
        1. Detectar letra grega (gamma, alpha) → \\gamma, \\alpha
        2. Detectar subscrito (gamma_s) → \\gamma_{s}
        3. Símbolo simples (M) → M
        
        Args:
            name: Nome Python do símbolo
        
        Returns:
            String LaTeX
        
        Examples:
            >>> registry = SymbolRegistry()
            >>> registry._auto_generate_latex('gamma_c')
            '\\\\gamma_{c}'
            >>> registry._auto_generate_latex('f_cd')
            'f_{cd}'
        """
        # Usar parser para gerar LaTeX
        parser = PyMemorialASTParser()
        latex = parser.to_latex(name)
        return latex

# =============================================================================
# SINGLETON GLOBAL
# =============================================================================

_global_registry: Optional[SymbolRegistry] = None

def get_global_registry() -> SymbolRegistry:
    """
    Obtém instância singleton do registry global.
    
    Returns:
        Instância única de SymbolRegistry
    
    Examples:
        >>> from pymemorial.symbols import get_global_registry
        >>> registry = get_global_registry()
        >>> registry.define('M_d', latex=r'M_{d}')
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = SymbolRegistry()
    return _global_registry

def get_registry() -> SymbolRegistry:
    """
    Alias para get_global_registry() (compatibilidade).
    
    Returns:
        Instância singleton do registry
    
    Examples:
        >>> from pymemorial.symbols import get_registry
        >>> registry = get_registry()
    """
    return get_global_registry()

def reset_global_registry() -> None:
    """
    Reseta registry global (útil para testes).
    
    Examples:
        >>> from pymemorial.symbols import reset_global_registry
        >>> reset_global_registry()
    """
    global _global_registry
    _global_registry = None
