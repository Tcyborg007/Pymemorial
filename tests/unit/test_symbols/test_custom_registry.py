# tests/unit/test_symbols/test_custom_registry.py

"""
Testes completos para symbols/custom_registry.py

Registry inteligente que aprende símbolos customizados automaticamente.

FUNCIONALIDADES TESTADAS:
- Definição manual de símbolos (define)
- Auto-aprendizado de símbolos (learn_from_code)
- Lookup de símbolos (get, has)
- Persistência JSON
- Integração com ast_parser
- Busca fuzzy de símbolos
"""

import pytest
import json
import tempfile
from pathlib import Path

from pymemorial.symbols.custom_registry import (
    SymbolRegistry,
    Symbol,
    RegistryError,
    get_global_registry,
    reset_global_registry
)


class TestSymbolDataclass:
    """Testes da dataclass Symbol."""
    
    def test_symbol_creation_minimal(self):
        """Criar símbolo com dados mínimos."""
        sym = Symbol(name='M_d', latex=r'M_{d}')
        assert sym.name == 'M_d'
        assert sym.latex == r'M_{d}'
        assert sym.description is None
        assert sym.category is None
    
    def test_symbol_creation_full(self):
        """Criar símbolo com todos os dados."""
        sym = Symbol(
            name='gamma_c',
            latex=r'\gamma_{c}',
            description='Coeficiente de majoração do concreto',
            category='nbr6118',
            aliases=['γc', 'gama_c']
        )
        assert sym.name == 'gamma_c'
        assert sym.description is not None
        assert 'γc' in sym.aliases
    
    def test_symbol_repr(self):
        """Símbolo deve ter repr legível."""
        sym = Symbol(name='f_cd', latex=r'f_{cd}')
        repr_str = repr(sym)
        assert 'f_cd' in repr_str
        assert r'f_{cd}' in repr_str


class TestSymbolRegistrySingleton:
    """Testes do padrão Singleton."""
    
    def test_get_global_registry_returns_singleton(self):
        """get_global_registry() deve retornar sempre a mesma instância."""
        reg1 = get_global_registry()
        reg2 = get_global_registry()
        assert reg1 is reg2
    
    def test_reset_global_registry_creates_new_instance(self):
        """reset_global_registry() deve criar nova instância."""
        reg1 = get_global_registry()
        reset_global_registry()
        reg2 = get_global_registry()
        assert reg1 is not reg2


class TestSymbolRegistryDefine:
    """Testes de definição manual de símbolos."""
    
    def test_define_simple_symbol(self):
        """Definir símbolo simples."""
        registry = SymbolRegistry()
        registry.define('M_d', latex=r'M_{d}')
        
        assert registry.has('M_d')
        sym = registry.get('M_d')
        assert sym.name == 'M_d'
        assert sym.latex == r'M_{d}'
    
    def test_define_with_description(self):
        """Definir símbolo com descrição."""
        registry = SymbolRegistry()
        registry.define(
            'f_cd',
            latex=r'f_{cd}',
            description='Resistência de cálculo do concreto à compressão'
        )
        
        sym = registry.get('f_cd')
        assert 'compressão' in sym.description
    
    def test_define_with_aliases(self):
        """Definir símbolo com aliases."""
        registry = SymbolRegistry()
        registry.define(
            'gamma_c',
            latex=r'\gamma_{c}',
            aliases=['γc', 'gama_c']
        )
        
        # Deve encontrar por nome ou alias
        assert registry.has('gamma_c')
        assert registry.has('γc')
        assert registry.has('gama_c')
    
    def test_define_duplicate_raises_error(self):
        """Definir símbolo duplicado deve levantar erro."""
        registry = SymbolRegistry()
        registry.define('M_d', latex=r'M_{d}')
        
        with pytest.raises(RegistryError, match="already defined"):
            registry.define('M_d', latex=r'M_{d}')
    
    def test_define_duplicate_with_overwrite(self):
        """Sobrescrever símbolo existente."""
        registry = SymbolRegistry()
        registry.define('M_d', latex=r'M_{d}', description='Original')
        registry.define('M_d', latex=r'M_{d,new}', overwrite=True)
        
        sym = registry.get('M_d')
        assert sym.latex == r'M_{d,new}'


class TestSymbolRegistryLookup:
    """Testes de busca de símbolos."""
    
    def test_get_existing_symbol(self):
        """Obter símbolo existente."""
        registry = SymbolRegistry()
        registry.define('M_d', latex=r'M_{d}')
        
        sym = registry.get('M_d')
        assert sym is not None
        assert sym.name == 'M_d'
    
    def test_get_nonexistent_returns_none(self):
        """Obter símbolo inexistente retorna None."""
        registry = SymbolRegistry()
        sym = registry.get('nonexistent')
        assert sym is None
    
    def test_get_by_alias(self):
        """Obter símbolo por alias."""
        registry = SymbolRegistry()
        registry.define('gamma_c', latex=r'\gamma_{c}', aliases=['γc'])
        
        sym = registry.get('γc')
        assert sym is not None
        assert sym.name == 'gamma_c'
    
    def test_has_checks_existence(self):
        """has() verifica existência de símbolo."""
        registry = SymbolRegistry()
        registry.define('M_d', latex=r'M_{d}')
        
        assert registry.has('M_d') is True
        assert registry.has('nonexistent') is False
    
    def test_list_all_symbols(self):
        """Listar todos os símbolos."""
        registry = SymbolRegistry()
        registry.define('M_d', latex=r'M_{d}')
        registry.define('V_d', latex=r'V_{d}')
        registry.define('N_d', latex=r'N_{d}')
        
        all_symbols = registry.list_all()
        assert len(all_symbols) == 3
        assert all(isinstance(s, Symbol) for s in all_symbols)


class TestSymbolRegistryAutoLearning:
    """Testes de auto-aprendizado de símbolos."""
    
    def test_learn_from_simple_code(self):
        """Aprender símbolos de código simples."""
        registry = SymbolRegistry()
        code = '''
gamma_c = 1.4
gamma_s = 1.15
M_d = gamma_f * M_k
'''
        registry.learn_from_code(code)
        
        # Deve ter aprendido símbolos com subscrito
        assert registry.has('gamma_c')
        assert registry.has('gamma_s')
        assert registry.has('M_d')
        assert registry.has('gamma_f')
        assert registry.has('M_k')
    
    def test_learn_latex_auto_generation(self):
        """LaTeX deve ser auto-gerado para símbolos aprendidos."""
        registry = SymbolRegistry()
        code = 'f_cd = 30  # MPa'
        registry.learn_from_code(code)
        
        sym = registry.get('f_cd')
        # LaTeX auto-gerado para subscrito
        assert sym.latex == r'f_{cd}'
    
    def test_learn_greek_auto_detection(self):
        """Letras gregas devem ser detectadas automaticamente."""
        registry = SymbolRegistry()
        code = 'alpha_s = 0.85'
        registry.learn_from_code(code)
        
        sym = registry.get('alpha_s')
        # Deve reconhecer 'alpha' como letra grega
        assert r'\alpha' in sym.latex
    
    def test_learn_ignores_builtins(self):
        """Não deve aprender símbolos builtin do Python."""
        registry = SymbolRegistry()
        code = '''
import math
result = math.sqrt(16)
'''
        registry.learn_from_code(code)
        
        # Não deve ter aprendido 'math' ou 'sqrt'
        assert not registry.has('math')
        assert not registry.has('sqrt')


class TestSymbolRegistryPersistence:
    """Testes de persistência em JSON."""
    
    @pytest.fixture
    def temp_file(self):
        """Fixture: arquivo temporário."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_path = Path(f.name)
        yield temp_path
        if temp_path.exists():
            temp_path.unlink()
    

    def test_save_registry(self, temp_file):
        """Salvar registry em JSON."""
        registry = SymbolRegistry()
        registry.define('M_d', latex=r'M_{d}', description='Momento de cálculo')
        registry.define('V_d', latex=r'V_{d}')
        
        registry.save_to_file(temp_file)
        
        assert temp_file.exists()
        
        # Verificar conteúdo JSON
        with open(temp_file, encoding='utf-8') as f:  # ← ADICIONAR encoding='utf-8'
            data = json.load(f)
        
        assert 'M_d' in data
        assert data['M_d']['latex'] == r'M_{d}'
        assert data['M_d']['description'] == 'Momento de cálculo'
    
    def test_load_registry(self, temp_file):
        """Carregar registry de JSON."""
        # Criar JSON manualmente
        data = {
            'M_d': {
                'name': 'M_d',
                'latex': r'M_{d}',
                'description': 'Momento',
                'category': 'structural',
                'aliases': ['Md']
            }
        }
        
        with open(temp_file, 'w') as f:
            json.dump(data, f)
        
        # Carregar
        registry = SymbolRegistry()
        registry.load_from_file(temp_file)
        
        assert registry.has('M_d')
        sym = registry.get('M_d')
        assert sym.description == 'Momento'
        assert 'Md' in sym.aliases
    
    def test_auto_save_after_define(self, temp_file, monkeypatch):
        """Auto-salvar após definir símbolo."""
        registry = SymbolRegistry(registry_file=temp_file, auto_save=True)
        
        registry.define('M_d', latex=r'M_{d}')
        
        # Deve ter salvado automaticamente
        assert temp_file.exists()
        
        with open(temp_file) as f:
            data = json.load(f)
        
        assert 'M_d' in data


class TestSymbolRegistryIntegration:
    """Testes de integração com outros módulos."""
    
    def test_integration_with_ast_parser(self):
        """Integração com ast_parser.to_latex()."""
        from pymemorial.recognition.ast_parser import PyMemorialASTParser
        
        registry = get_global_registry()
        registry.define('M_d', latex=r'M_{d,custom}')
        
        parser = PyMemorialASTParser()
        
        # Parser deve usar símbolo do registry (futuro)
        # Por ora, testa que registry existe
        assert registry.has('M_d')


class TestSymbolRegistrySearch:
    """Testes de busca fuzzy."""
    
    def test_search_by_partial_name(self):
        """Buscar símbolos por nome parcial."""
        registry = SymbolRegistry()
        registry.define('M_d', latex=r'M_{d}')
        registry.define('M_k', latex=r'M_{k}')
        registry.define('M_Rd', latex=r'M_{Rd}')
        registry.define('V_d', latex=r'V_{d}')
        
        # Buscar símbolos que contenham 'M'
        results = registry.search('M')
        assert len(results) == 3
        assert all('M' in s.name for s in results)
    
    def test_search_by_category(self):
        """Buscar símbolos por categoria."""
        registry = SymbolRegistry()
        registry.define('gamma_c', latex=r'\gamma_{c}', category='nbr6118')
        registry.define('gamma_s', latex=r'\gamma_{s}', category='nbr6118')
        registry.define('phi', latex=r'\phi', category='aci318')
        
        results = registry.search(category='nbr6118')
        assert len(results) == 2


# Executar: pytest tests/unit/test_symbols/test_custom_registry.py -v
# Esperado: TODOS FALHAM (RED) ✅
# Próximo passo: implementar custom_registry.py para GREEN
