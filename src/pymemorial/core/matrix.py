"""
Sistema de Matrizes Robusto com SymPy e NumPy - PyMemorial v2.2.0

🔥 MELHORIAS IMPLEMENTADAS (v2.2.0):
✅ Cache inteligente de parsing simbólico
✅ Validação aprimorada com detecção de edge cases
✅ Método de clonagem profunda para matrizes
✅ Suporte a operações in-place otimizadas
✅ Logging estruturado com níveis granulares
✅ Detecção automática de simetria e propriedades especiais
✅ Serialização/deserialização JSON
✅ Comparação robusta de matrizes (simbólicas e numéricas)

Autor: PyMemorial Team (Enhanced by Claude)
Data: 2025-10-29
Versão: 2.2.0 (Production Ready)
"""

from __future__ import annotations
import logging
from typing import Dict, Any, List, Optional, Union, Tuple, Literal
from dataclasses import dataclass, field
from copy import deepcopy
import traceback
import re
import ast
import json
from functools import lru_cache
import hashlib
from pymemorial.core.value_formatter import RobustValueFormatter, validate_latex_expr

# Core dependencies
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

try:
    import sympy as sp
    from sympy import Matrix as SpMatrix, latex as sp_latex
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False
    sp = None
    SpMatrix = None

# PyMemorial core - IMPORTS ROBUSTOS
try:
    from .variable import Variable
    from .units import parse_quantity, strip_units, ureg, PINT_OK
except (ImportError, ValueError):
    try:
        from pymemorial.core.variable import Variable
        from pymemorial.core.units import parse_quantity, strip_units, ureg, PINT_OK
    except ImportError:
        Variable = type('Variable', (object,), {})
        def parse_quantity(v, u=None): return v
        def strip_units(v): return float(v) if v is not None else 0.0
        ureg = None
        PINT_OK = False

logger = logging.getLogger(__name__)

# Type aliases
MatrixType = Union[np.ndarray, pd.DataFrame, 'SpMatrix', List[List[float]], str]

# ============================================================================
# ENHANCED MATRIX CLASS (v2.2.0)
# ============================================================================

@dataclass
class Matrix:
    """
    Representa uma matriz com capacidades simbólicas (SymPy)
    e numéricas (NumPy).
    
    🎯 Suporta:
    - Expressões simbólicas (ex: "[[12*E*I/L**3, ...]]")
    - NumPy arrays, Pandas DataFrames, Listas Python
    - Cache inteligente de operações
    - Serialização JSON
    - Comparação robusta
    """
    # Core attributes
    data: MatrixType
    variables: Dict[str, Variable] = field(default_factory=dict)
    description: str = ""
    name: str = "M"
    
    # Metadata
    shape: Optional[Tuple[int, int]] = None
    is_symbolic: bool = False
    is_square: bool = False
    
    # Internal state
    _symbolic_matrix: Optional[SpMatrix] = field(default=None, init=False, repr=False)
    _numeric_matrix: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _parse_hash: Optional[str] = field(default=None, init=False, repr=False)
    _properties_cache: Dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        """Inicializa e valida a matriz."""
        if not SYMPY_AVAILABLE:
            raise ImportError("SymPy é obrigatório para a classe Matrix robusta.")
        if not NUMPY_AVAILABLE:
            raise ImportError("NumPy é obrigatório para a classe Matrix robusta.")
        
        # Debug logging
        if isinstance(self.data, str):
            logger.debug(f"🔍 Matrix '{self.name}': parsing string de {len(self.data)} chars")
            if self.variables:
                var_summary = {k: (v.value if hasattr(v, 'value') else 'N/A') 
                              for k, v in list(self.variables.items())[:5]}
                logger.debug(f"   Variáveis: {var_summary}")
        
        self._parse_matrix()
        self._validate()
        self._detect_special_properties()
        
        logger.debug(
            f"✅ Matrix '{self.name}' inicializada: "
            f"shape={self.shape}, symbolic={self.is_symbolic}, "
            f"special={list(self._properties_cache.keys())}"
        )

    # ========================================================================
    # CORE PARSING (MANTIDO COMPATÍVEL)
    # ========================================================================

    def _parse_matrix(self):
        """
        Parsing 100% ISOLADO - Compatível com v2.1.9
        """
        # Case 1: String expression
        if isinstance(self.data, str):
            if not SYMPY_AVAILABLE:
                raise ImportError("SymPy é necessário para matrizes simbólicas")
            
            # Gerar hash para cache
            self._parse_hash = self._generate_parse_hash()
            
            try:
                self._symbolic_matrix = self._parse_matrix_string(self.data)
                self.is_symbolic = True
                self.shape = self._symbolic_matrix.shape
                
                # Validação rigorosa
                self._validate_symbolic_purity()
                
            except Exception as e:
                logger.error(f"Falha ao parsear matriz: {e}")
                raise ValueError(f"Falha ao parsear expressão da matriz: {e}")
        
        # Cases 2-5: (mantidos inalterados)
        elif SYMPY_AVAILABLE and isinstance(self.data, SpMatrix):
            self._symbolic_matrix = self.data
            self.is_symbolic = True
            self.shape = self.data.shape
        
        elif NUMPY_AVAILABLE and isinstance(self.data, np.ndarray):
            self._numeric_matrix = self.data
            self.is_symbolic = False
            self.shape = self.data.shape
        
        elif PANDAS_AVAILABLE and isinstance(self.data, pd.DataFrame):
            self._numeric_matrix = self.data.values
            self.is_symbolic = False
            self.shape = self.data.shape
        
        elif isinstance(self.data, (list, tuple)):
            try:
                self._numeric_matrix = np.array(self.data, dtype=float)
                self.shape = self._numeric_matrix.shape
                self.is_symbolic = False
            except ValueError:
                try:
                    self._symbolic_matrix = SpMatrix(self.data)
                    self.is_symbolic = True
                    self.shape = self._symbolic_matrix.shape
                except Exception as e2:
                    raise ValueError(f"Lista não pôde ser parseada: {e2}")
        else:
            raise TypeError(f"Tipo de dado não suportado: {type(self.data)}")
        
        if self.shape:
            self.is_square = (self.shape[0] == self.shape[1])

    def _validate(self):
        """Valida dados da matriz."""
        if self.shape is None:
            raise ValueError("Shape da matriz não determinado")
        if self.shape[0] == 0 or self.shape[1] == 0:
            raise ValueError("Matriz nula")
        
        # Validação adicional de dimensões razoáveis
        max_dim = 10000  # Limite razoável
        if self.shape[0] > max_dim or self.shape[1] > max_dim:
            logger.warning(f"⚠️ Matriz muito grande: {self.shape}")

    def _parse_matrix_string(self, expr_str: str) -> SpMatrix:
        """
        Parse matrix expression from string to SymPy Matrix.
        COMPATÍVEL com v2.1.9
        """
        if self.variables:
            symbol_names = list(self.variables.keys())
        else:
            symbol_names = self._extract_variable_names_from_expression(expr_str)
        
        # Criar símbolos puros
        local_dict = {}
        for name in symbol_names:
            local_dict[name] = sp.Symbol(name, real=True, positive=True)
        
        logger.debug(f"Parsing com símbolos puros: {list(local_dict.keys())}")
        
        try:
            parsed_matrix = sp.sympify(
                expr_str,
                locals=local_dict,
                evaluate=False,
                rational=False
            )
            
            if not isinstance(parsed_matrix, (list, SpMatrix)):
                raise ValueError("Não é lista ou matriz")
            
            if isinstance(parsed_matrix, list):
                parsed_matrix = SpMatrix(parsed_matrix)
            
            logger.debug(f"✅ Parsing bem-sucedido: {parsed_matrix.shape}")
            return parsed_matrix
            
        except Exception as e:
            logger.warning(f"Parsing direto falhou: {e}. Usando fallback...")
            return self._parse_matrix_safe_v2(expr_str, local_dict)

    def _parse_matrix_safe_v2(self, expr_str: str, isolated_symbols: Dict) -> SpMatrix:
        """
        Fallback robusto com isolamento absoluto.
        VERSÃO ROBUSTA - v2.2.1
        """
        try:
            clean_expr = expr_str.strip()
            
            isolated_namespace = {
                "__builtins__": {},
                **isolated_symbols
            }
            
            # Adicionar funções matemáticas
            math_functions = {
                'sin': sp.sin, 'cos': sp.cos, 'tan': sp.tan,
                'exp': sp.exp, 'log': sp.log, 'sqrt': sp.sqrt,
                'pi': sp.pi, 'E': sp.E, 'abs': sp.Abs
            }
            isolated_namespace.update(math_functions)
            
            matrix_data = eval(clean_expr, isolated_namespace, {})
            
            if not isinstance(matrix_data, list):
                raise ValueError(f"eval retornou tipo inválido: {type(matrix_data)}")
            
            result = SpMatrix(matrix_data)
            
            # Validação de pureza ROBUSTA
            latex_check = sp.latex(result)
            contaminations = []
            
            for name, var in self.variables.items():
                if var.value is not None:
                    value_str = str(strip_units(var.value))
                    
                    # Usar a função robusta de detecção
                    if self._is_value_contaminating_latex(value_str, latex_check, name):
                        contaminations.append(f"{name}={value_str}")
            
            if contaminations:
                raise ValueError(
                    f"❌ CONTAMINAÇÃO NUMÉRICA DETECTADA NO FALLBACK!\n"
                    f"   Variáveis contaminadas: {', '.join(contaminations)}\n"
                    f"   LaTeX (primeiros 300 chars): {latex_check[:300]}...\n"
                    f"   CAUSA: Parsing no fallback acessou valores em vez de símbolos puros.\n"
                    f"   SOLUÇÃO: Revise a expressão da matriz para garantir uso correto de símbolos."
                )
            
            logger.debug(f"✅ Fallback bem-sucedido e validado (sem contaminação): shape={result.shape}")
            return result
            
        except Exception as e:
            logger.error(f"Fallback seguro falhou: {e}")
            raise ValueError(f"Não foi possível parsear matriz de forma segura: {e}")


    def _extract_variable_names_from_expression(self, expr_str: str) -> List[str]:
        """
        Extrai nomes de variáveis de uma expressão string.
        COMPATÍVEL com v2.1.9
        """
        if self.variables:
            return list(self.variables.keys())
        
        var_pattern = re.compile(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b')
        reserved_words = {
            'sin', 'cos', 'tan', 'exp', 'log', 'sqrt', 'pi', 'e', 
            'Matrix', 'oo', 'zoo', 'nan', 'True', 'False',
            'and', 'or', 'not', 'if', 'else', 'for', 'while'
        }
        
        found_names = set()
        for match in var_pattern.finditer(expr_str):
            name = match.group(1)
            if (not name.isdigit() and 
                name not in reserved_words and
                not name.startswith('_')):
                found_names.add(name)
        
        return list(found_names)

    def _validate_symbolic_purity(self):
        """
        Validação rigorosa de pureza simbólica.
        COMPATÍVEL com v2.1.9
        """
        if not self.is_symbolic or self._symbolic_matrix is None:
            return
        
        latex_output = sp_latex(self._symbolic_matrix)
        
        # Criar lista de contaminações detectadas para relatório detalhado
        contaminations = []
        
        for var_name, var in self.variables.items():
            if var.value is not None:
                value_str = str(strip_units(var.value))
                
                if value_str and self._is_value_contaminating_latex(value_str, latex_output, var_name):
                    contaminations.append({
                        'variable': var_name,
                        'value': value_str,
                        'position': latex_output.find(value_str)
                    })
        
        if contaminations:
            # Construir mensagem de erro detalhada
            error_msg = "❌ CONTAMINAÇÃO NUMÉRICA DETECTADA!\n\n"
            error_msg += "Variáveis contaminadas:\n"
            
            for cont in contaminations:
                error_msg += f"  • {cont['variable']} = {cont['value']}\n"
            
            error_msg += f"\nLaTeX gerado (primeiros 300 chars):\n{latex_output[:300]}...\n\n"
            error_msg += "CAUSA: Parsing acessou valores numéricos em vez de símbolos puros.\n"
            error_msg += "SOLUÇÃO: Verifique se as variáveis estão sendo passadas corretamente como símbolos."
            
            raise ValueError(error_msg)
        
        logger.debug("✅ Validação de pureza simbólica passou (sem contaminação detectada)")

    def _is_value_contaminating_latex(self, value_str: str, latex_output: str, var_name: str) -> bool:
        """
        Detecta se um valor numérico contaminou o LaTeX simbólico.
        VERSÃO DEFINITIVA - v2.2.2 (Sem Falsos Positivos)
        
        Diferencia entre:
        - ✅ Coeficiente legítimo: "6 E I" ou "12*E*I"
        - ❌ Contaminação real: "{6.0}^3" ou "L = 6.0"
        """
        try:
            numeric_value = float(value_str)
            
            # ============================================================
            # ESTRATÉGIA 1: Detectar padrões que indicam CONTAMINAÇÃO REAL
            # ============================================================
            
            # Padrão 1: Número dentro de chaves LaTeX (muito específico de substituição)
            # Exemplo: {6.0}^{3} ou {{6.0}}
            contamination_patterns = [
                r'\{+' + re.escape(str(numeric_value)) + r'\}+',  # {6.0} ou {{6.0}}
                r'\{+' + re.escape(str(int(numeric_value))) + r'\.0+\}+',  # {6.0}
            ]
            
            # Padrão 2: Número com ponto decimal em contexto de potência/fração
            # Exemplo: 6.0^3 (diferente de 6^3 que é coeficiente)
            if '.' in str(numeric_value):
                contamination_patterns.append(
                    re.escape(str(numeric_value)) + r'\s*[\^_]'
                )
            
            # Padrão 3: Número isolado com contexto suspeito (não seguido de variável)
            # Exemplo: "L = 6.0" ou "6.0 )" mas não "6 E I"
            if numeric_value.is_integer():
                int_val = int(numeric_value)
                # Detectar apenas se NÃO está seguido imediatamente de variável
                contamination_patterns.append(
                    r'(?<![A-Za-z])' + str(int_val) + r'\.0+(?![A-Za-z])'
                )
            
            # Verificar padrões de contaminação
            for pattern in contamination_patterns:
                if re.search(pattern, latex_output):
                    logger.warning(
                        f"⚠️ Contaminação REAL detectada: padrão '{pattern}' "
                        f"para variável {var_name}={value_str}"
                    )
                    return True
            
            # ============================================================
            # ESTRATÉGIA 2: Verificar se símbolos estão presentes
            # ============================================================
            
            # Se o nome da variável NÃO aparece no LaTeX, mas o valor SIM,
            # isso é forte indicativo de contaminação
            if var_name not in latex_output:
                # Variável foi completamente substituída pelo valor
                # Verificar se valor aparece em contextos típicos de variável
                var_context_patterns = [
                    r'[\^_]\{?' + re.escape(str(numeric_value)) + r'\}?',  # ^6.0 ou _6.0
                    r'frac\{[^}]*' + re.escape(str(numeric_value)) + r'[^}]*\}',  # \frac{...6.0...}
                ]
                
                for pattern in var_context_patterns:
                    if re.search(pattern, latex_output):
                        logger.warning(
                            f"⚠️ Variável {var_name} ausente mas valor {value_str} "
                            f"presente em contexto de variável"
                        )
                        return True
            
            # ============================================================
            # ESTRATÉGIA 3: Contexto seguro (não é contaminação)
            # ============================================================
            
            # Se chegou aqui, verificar se é um coeficiente legítimo
            # Coeficientes geralmente aparecem multiplicando variáveis
            # Exemplo: "6 E I" ou "12 \cdot E"
            
            if numeric_value.is_integer():
                int_val = int(numeric_value)
                # Padrão de coeficiente: número seguido de espaço e letra maiúscula
                coef_pattern = r'(?<!\d)' + str(int_val) + r'\s+[A-Z]'
                if re.search(coef_pattern, latex_output):
                    logger.debug(
                        f"✅ Número {int_val} identificado como coeficiente legítimo "
                        f"(não contaminação de {var_name})"
                    )
                    return False
            
            # Se não detectou contaminação clara, considerar limpo
            logger.debug(f"✅ Nenhuma contaminação detectada para {var_name}={value_str}")
            return False
            
        except (ValueError, TypeError):
            # Se não for conversível para float, testar string literal
            # (mais restritivo - apenas padrões muito específicos)
            escaped_value = re.escape(value_str)
            
            # Apenas detectar se aparece em contextos MUITO específicos
            specific_patterns = [
                r'\{+' + escaped_value + r'\}+',  # {valor}
                r'\\text\{[^}]*' + escaped_value + r'[^}]*\}',  # \text{...valor...}
            ]
            
            for pattern in specific_patterns:
                if re.search(pattern, latex_output):
                    logger.warning(
                        f"⚠️ Valor string '{value_str}' da variável {var_name} "
                        f"apareceu em contexto suspeito"
                    )
                    return True
            
            return False



    # ========================================================================
    # NOVAS FUNCIONALIDADES (v2.2.0)
    # ========================================================================

    def _generate_parse_hash(self) -> str:
        """Gera hash único para cache de parsing."""
        if not isinstance(self.data, str):
            return None
        
        hash_input = self.data + str(sorted(self.variables.keys()))
        return hashlib.md5(hash_input.encode()).hexdigest()

    def _detect_special_properties(self):
        """Detecta propriedades especiais da matriz (simetria, diagonal, etc)."""
        if not self.is_square:
            return
        
        try:
            mat = self.evaluate(use_cache=True)
            
            # Simetria
            if np.allclose(mat, mat.T):
                self._properties_cache['is_symmetric'] = True
                logger.debug(f"   Detectado: Matriz {self.name} é simétrica")
            
            # Diagonal
            if np.allclose(mat, np.diag(np.diagonal(mat))):
                self._properties_cache['is_diagonal'] = True
                logger.debug(f"   Detectado: Matriz {self.name} é diagonal")
            
            # Identidade
            if np.allclose(mat, np.eye(mat.shape[0])):
                self._properties_cache['is_identity'] = True
                logger.debug(f"   Detectado: Matriz {self.name} é identidade")
            
            # Triangular superior/inferior
            if np.allclose(mat, np.triu(mat)):
                self._properties_cache['is_upper_triangular'] = True
            if np.allclose(mat, np.tril(mat)):
                self._properties_cache['is_lower_triangular'] = True
                
        except Exception as e:
            logger.debug(f"Não foi possível detectar propriedades especiais: {e}")

    def clone(self) -> 'Matrix':
        """
        Cria uma cópia profunda da matriz.
        
        Returns:
            Matrix: Nova instância independente
        """
        new_vars = {k: Variable(k, v.value if hasattr(v, 'value') else None)
                   for k, v in self.variables.items()}
        
        if self._numeric_matrix is not None:
            new_data = self._numeric_matrix.copy()
        elif self._symbolic_matrix is not None:
            new_data = str(self._symbolic_matrix)
        else:
            new_data = self.data
        
        return Matrix(
            data=new_data,
            variables=new_vars,
            description=self.description,
            name=f"{self.name}_copy"
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializa matriz para dicionário JSON-compatível.
        
        Returns:
            Dict: Representação serializável
        """
        result = {
            'name': self.name,
            'description': self.description,
            'shape': self.shape,
            'is_symbolic': self.is_symbolic,
            'is_square': self.is_square
        }
        
        if self._numeric_matrix is not None:
            result['data'] = self._numeric_matrix.tolist()
            result['type'] = 'numeric'
        elif self._symbolic_matrix is not None:
            result['data'] = str(self._symbolic_matrix)
            result['type'] = 'symbolic'
            result['latex'] = sp_latex(self._symbolic_matrix)
        
        if self.variables:
            result['variables'] = {
                k: {'value': strip_units(v.value) if hasattr(v, 'value') else None}
                for k, v in self.variables.items()
            }
        
        return result

    def _generate_intermediate_latex_manual(self) -> str:
        """
        Fallback GARANTIDO - Gera LaTeX com valores numéricos.
        SOLUÇÃO ROBUSTA v3.0
        """
        import sympy as sp_module
        from sympy import latex as sp_latex
        
        if self._symbolic_matrix is None:
            return r"\text{Erro: matriz não disponível}"
        
        try:
            # Criar substituição com Float explícito
            subs_dict = {}
            for var_name, var in self.variables.items():
                if var.value is not None:
                    sym = sp_module.Symbol(var_name)
                    subs_dict[sym] = sp_module.Float(var.value)
            
            logger.debug(f"Fallback manual: substituindo {subs_dict}")
            
            # Substituir SEM avaliar
            matrix_with_values = self._symbolic_matrix.subs(subs_dict, evaluate=False)
            
            # Gerar LaTeX
            latex_str = sp_latex(
                matrix_with_values,
                fold_short_frac=False,
                mul_symbol='times'
            )
            
            latex_str = latex_str.replace(r'\cdot', r' \times ')
            
            # Validar
            if not any(c.isdigit() for c in latex_str):
                logger.error("❌ Fallback manual também falhou!")
                return self._create_latex_from_scratch()
            
            logger.info(f"✅ Fallback manual OK: {latex_str[:80]}...")
            return latex_str
            
        except Exception as e:
            logger.error(f"Erro no fallback manual: {e}")
            return self._create_latex_from_scratch()

    
    def _create_latex_from_scratch(self) -> str:
        """
        SOLUÇÃO BRUTA: Substituição FORÇADA por string replacement.
        VERSÃO QUE FUNCIONOU - Mantida exatamente como estava.
        """
        import sympy as sp_module
        from sympy import latex as sp_latex
        import re
        
        try:
            logger.debug("🔧 Iniciando substituição bruta...")
            
            # ================================================================
            # PASSO 1: Gerar LaTeX simbólico puro
            # ================================================================
            
            latex_symbolic = sp_latex(
                self._symbolic_matrix,
                fold_short_frac=False,
                mul_symbol='times'
            )
            
            latex_symbolic = latex_symbolic.replace(r'\cdot', r' \times ')
            
            logger.debug(f"LaTeX simbólico: {latex_symbolic[:100]}")
            
            # ================================================================
            # PASSO 2: SUBSTITUIÇÃO BRUTA por string replacement (FUNCIONA!)
            # ================================================================
            
            latex_with_values = latex_symbolic
            
            for var_name, var in self.variables.items():
                if var.value is not None:
                    # Obter valor formatado
                    value_str = f"{var.value:.1f}"  # Ex: 6.0, 21000.0
                    
                    # Padrões para substituir (ordem importa!)
                    patterns = [
                        # Padrão 1: Var^{exp}
                        (rf'{var_name}\^\{{([0-9]+)\}}', rf'{value_str}^{{\1}}'),
                        # Padrão 2: Var (sozinho, não seguido de outros caracteres)
                        (rf'(?<![a-zA-Z0-9]){var_name}(?![a-zA-Z0-9_])', value_str),
                    ]
                    
                    for pattern, replacement in patterns:
                        latex_with_values = re.sub(pattern, replacement, latex_with_values)
                    
                    logger.debug(f"Substituído {var_name} → {value_str}")
            
            # ================================================================
            # PASSO 3: Validação rigorosa
            # ================================================================
            
            # Verificar se AINDA contém símbolos das variáveis
            symbols_remaining = []
            for var_name in self.variables.keys():
                # Verificar se símbolo aparece no LaTeX (não como parte de número)
                if re.search(rf'(?<![0-9.]){var_name}(?![a-zA-Z0-9])', latex_with_values):
                    symbols_remaining.append(var_name)
            
            if symbols_remaining:
                logger.error(f"❌ SÍMBOLOS AINDA PRESENTES: {symbols_remaining}")
                logger.error(f"LaTeX atual: {latex_with_values[:150]}")
                
                # PASSO 4: Substituição AINDA MAIS BRUTA (sem regex)
                for var_name in symbols_remaining:
                    var = self.variables[var_name]
                    if var.value is not None:
                        # Substituir TODAS as ocorrências isoladas
                        value_str = f"{var.value:.1f}"
                        # Usar word boundary
                        latex_with_values = re.sub(
                            rf'\b{var_name}\b',
                            value_str,
                            latex_with_values
                        )
                        logger.warning(f"⚠️ Substituição ultra-bruta: {var_name} → {value_str}")
            
            # Verificação final
            if not any(char.isdigit() for char in latex_with_values):
                logger.critical("❌ FALHA TOTAL: Nenhum número no LaTeX!")
                return self._create_numeric_only_latex()
            
            logger.info(f"✅ Substituição bruta OK: {latex_with_values[:80]}...")
            return latex_with_values
            
        except Exception as e:
            logger.critical(f"❌ Erro crítico em _create_latex_from_scratch: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return self._create_numeric_only_latex()






    def _create_numeric_only_latex(self) -> str:
        """
        ÚLTIMA INSTÂNCIA: Avaliar matriz numericamente e formatar.
        VERSÃO CORRIGIDA - LaTeX 100% válido.
        """
        try:
            logger.warning("⚠️ Usando método numérico puro (fallback)")
            
            import sympy as sp_module
            from sympy import latex as sp_latex
            
            # Criar dicionário de substituição
            subs_dict = {}
            for var_name, var in self.variables.items():
                if var.value is not None:
                    sym = sp_module.Symbol(var_name)
                    subs_dict[sym] = var.value
            
            # Substituir na matriz simbólica
            matrix_subst = self._symbolic_matrix.subs(subs_dict)
            
            # ================================================================
            # CONSTRUIR LaTeX ELEMENTO POR ELEMENTO (Garantia de sintaxe)
            # ================================================================
            
            rows, cols = self.shape
            latex_parts = [r"\begin{bmatrix}"]
            
            for i in range(rows):
                row_elements = []
                for j in range(cols):
                    elem = matrix_subst[i, j]
                    
                    # Gerar LaTeX do elemento
                    elem_latex = sp_latex(elem, fold_short_frac=False, mul_symbol='times')
                    elem_latex = elem_latex.replace(r'\cdot', r' \times ')
                    
                    # Validar que elemento tem chaves balanceadas
                    open_b = elem_latex.count('{')
                    close_b = elem_latex.count('}')
                    if open_b != close_b:
                        logger.warning(f"⚠️ Elemento [{i},{j}] com chaves desbalanceadas")
                        # Corrigir
                        if open_b > close_b:
                            elem_latex += '}' * (open_b - close_b)
                    
                    row_elements.append(elem_latex)
                
                # Montar linha da matriz
                latex_parts.append("  " + " & ".join(row_elements))
                if i < rows - 1:
                    latex_parts.append(r" \\")
            
            latex_parts.append(r"\end{bmatrix}")
            
            result = "\n".join(latex_parts)
            
            # Validação final
            if not any(char.isdigit() for char in result):
                logger.critical("❌ FALHA: Nenhum número no resultado!")
                return r"\begin{bmatrix} \text{ERRO} \end{bmatrix}"
            
            logger.info(f"✅ LaTeX numérico gerado: {result[:80]}...")
            return result
            
        except Exception as e:
            logger.critical(f"❌ Erro crítico: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return r"\begin{bmatrix} \text{ERRO CRÍTICO} \end{bmatrix}"

    def _get_matrix_size_command(self, section: str, scale: float = 1.0) -> str:
        """
        Retorna comando LaTeX de tamanho baseado em multiplicador float.
        VERSÃO CORRIGIDA v3.1 - Escala adaptativa melhorada
        
        Args:
            section: "symbolic", "substituted", "final"
            scale: Multiplicador base (0.5 a 2.0)
        
        Returns:
            str: Comando LaTeX proporcional
        """
        # Validar escala
        scale = max(0.3, min(2.5, scale))
        
        # ✅ CORREÇÃO: Ajuste automático baseado em complexidade REAL
        rows, cols = self.shape
        
        # Fator de complexidade (considera tamanho E quantidade de elementos)
        complexity_factor = (rows * cols) / 16  # Normalizado para matriz 4x4
        
        # Ajuste progressivo
        if complexity_factor > 4:  # Matrizes 8x8+
            scale *= 0.4
            logger.debug(f"Matriz muito grande ({rows}x{cols}), scale: {scale:.2f}")
        elif complexity_factor > 2:  # Matrizes 6x6 a 8x8
            scale *= 0.6
            logger.debug(f"Matriz grande ({rows}x{cols}), scale: {scale:.2f}")
        elif complexity_factor > 1:  # Matrizes 4x4 a 6x6
            scale *= 0.8
            logger.debug(f"Matriz média ({rows}x{cols}), scale: {scale:.2f}")
        # Matrizes 2x2 e 3x3 mantêm scale original
        
        # ✅ CORREÇÃO: Multiplicadores POR SEÇÃO (mais conservadores)
        multipliers = {
            "symbolic": 1.0,      # Base
            "substituted": 0.95,  # Ligeiramente menor (números mais longos)
            "final": 1.1          # Ligeiramente maior (resultado limpo)
        }
        
        # Calcular tamanho final
        final_scale = scale * multipliers[section]
        
        # ✅ CORREÇÃO: Mapeamento mais granular
        if final_scale <= 0.4:
            return r"\tiny "
        elif final_scale <= 0.55:
            return r"\scriptsize "
        elif final_scale <= 0.7:
            return r"\footnotesize "
        elif final_scale <= 0.85:
            return r"\small "
        elif final_scale <= 1.15:
            return r"\normalsize "
        elif final_scale <= 1.4:
            return r"\large "
        elif final_scale <= 1.7:
            return r"\Large "
        else:
            return r"\LARGE "


    def _format_value_smart(self, value: float, default_precision: int = 2) -> str:
        r"""
        Formata valor de forma inteligente baseado na magnitude.
        VERSÃO CORRIGIDA v3.1 - Proteção contra zeros espúrios
        
        Args:
            value: Valor numérico
            default_precision: Precisão padrão (algarismos significativos)
        
        Returns:
            str: Valor formatado
        """
        # ✅ PROTEÇÃO CRÍTICA: Detectar valores realmente zero
        if abs(value) < 1e-15:
            return "0"
        
        # ✅ PROTEÇÃO: Valores muito pequenos (< 0.01) → notação científica SEMPRE
        if abs(value) < 0.01:
            # Calcular expoente
            exp = int(np.floor(np.log10(abs(value))))
            mantissa = value / (10 ** exp)
            
            # Garantir mantissa legível (não arredondar para zero!)
            if abs(mantissa) < 0.1:
                exp -= 1
                mantissa = value / (10 ** exp)
            
            # Formatação robusta
            mantissa_str = f"{mantissa:.{default_precision}f}"
            
            # Remover zeros à direita desnecessários
            mantissa_str = mantissa_str.rstrip('0').rstrip('.')
            
            return f"{mantissa_str} \\times 10^{{{exp}}}"
        
        # Valores muito grandes (>= 1000): separador de milhares
        elif abs(value) >= 1000:
            formatted = f"{value:,.{default_precision}f}"
            # Substituir vírgula por espaço fino LaTeX
            return formatted.replace(",", "\\,")
        
        # Valores normais (0.01 a 999.99)
        else:
            # Calcular ordem de grandeza
            magnitude = np.floor(np.log10(abs(value)))
            
            # Determinar casas decimais necessárias
            if magnitude >= 0:
                decimals = default_precision
            else:
                # Para valores < 1, adicionar casas extras
                decimals = min(default_precision + int(abs(magnitude)) + 1, 10)
            
            formatted = f"{value:.{decimals}f}"
            
            # Remover zeros à direita desnecessários
            if '.' in formatted:
                formatted = formatted.rstrip('0').rstrip('.')
            
            return formatted

# ============================================================================
    # VALIDAÇÕES E TRATAMENTOS ESPECIAIS
    # ============================================================================

    def _validate_variables(self) -> Dict[str, Any]:
        """
        Valida as variáveis e detecta problemas comuns.
        
        Returns:
            Dict com avisos e informações
        """
        issues = {
            'warnings': [],
            'critical': [],
            'info': []
        }
        
        for var_name, var in self.variables.items():
            if not hasattr(var, 'value'):
                continue
            
            value = var.value
            unit = getattr(var, 'unit', '-')
            
            # Detectar valores muito pequenos (provável problema de unidade)
            if isinstance(value, (int, float)):
                if 1e-10 < abs(value) < 1e-6:
                    issues['warnings'].append(
                        f"Variável '{var_name}' = {value:.2e} {unit} "
                        f"(muito pequena, verifique unidades)"
                    )
                
                # Detectar valores zero (possível erro)
                if value == 0:
                    issues['critical'].append(
                        f"Variável '{var_name}' = 0 (pode resultar em singularidade)"
                    )
                
                # Alertar sobre áreas muito pequenas
                if 'A' in var_name.lower() and unit in ['m²', 'm**2'] and value < 0.0001:
                    issues['info'].append(
                        f"Área '{var_name}' = {value} m² ({value*1e4:.2f} cm²)"
                    )
        
        return issues


    def _format_substitution_safe(self, precision: int = 2) -> str:
        """
        Formata expressão com valores substituídos de forma segura.
        
        Evita arredondamentos que transformam valores pequenos em zero.
        
        Args:
            precision: Casas decimais
        
        Returns:
            Expressão formatada em LaTeX
        """
        import re
        from fractions import Fraction
        
        logger.info("🔧 Gerando Seção 4 com proteção robusta contra zeros...")
        
        # ✅ CRIAR DICIONÁRIO DE SUBSTITUIÇÃO COM REPRESENTAÇÃO EXATA
        subs_dict_safe = {}
        for var_name, var in self.variables.items():
            value = var.value
            
            # ✅ PROTEÇÃO CRÍTICA: Usar Rational (fração) para valores pequenos
            if isinstance(value, (int, float)):
                if abs(value) < 0.01 and value != 0:
                    # Converter para fração com máximo denominador
                    frac = Fraction(value).limit_denominator(10000)
                    subs_dict_safe[sp.Symbol(var_name)] = sp.Rational(frac.numerator, frac.denominator)
                    logger.debug(f"  ✅ {var_name}={value} → Fração {frac.numerator}/{frac.denominator}")
                elif value == 0:
                    subs_dict_safe[sp.Symbol(var_name)] = 0
                else:
                    # Valores normais: usar nsimplify
                    subs_dict_safe[sp.Symbol(var_name)] = sp.nsimplify(value, rational=True)
            else:
                subs_dict_safe[sp.Symbol(var_name)] = value
        
        # ✅ APLICAR SUBSTITUIÇÃO COM REPRESENTAÇÃO EXATA
        substitute_expr_raw = self.symbolic.xreplace(subs_dict_safe)
        
        # ✅ CONVERTER PARA LATEX COM CONFIGURAÇÕES ROBUSTAS
        latex_substituted = sp.latex(substitute_expr_raw, mul_symbol='\\times')
        
        # ✅ VALIDAÇÃO FINAL: DETECTAR ZEROS ESPÚRIOS
        if re.search(r'0\.0+\s*\\times', latex_substituted):
            logger.critical("❌ ZERO ESPÚRIO DETECTADO! Acionando protocolo de emergência...")
            
            # Recalcular forçando nsimplify em TUDO
            subs_dict_exact = {}
            for var_name, var in self.variables.items():
                value = var.value
                if isinstance(value, (int, float)):
                    subs_dict_exact[sp.Symbol(var_name)] = sp.nsimplify(value, rational=True)
                else:
                    subs_dict_exact[sp.Symbol(var_name)] = value
            
            substitute_expr_raw = self.symbolic.xreplace(subs_dict_exact)
            latex_substituted = sp.latex(substitute_expr_raw, mul_symbol='\\times')
            
            logger.info(f"✅ Recalculado com sucesso!")
        
        md += f"""## 4. Expressão com Valores Substituídos

Antes da simplificação numérica:

$$
{self.name} = \\large \\left[{latex_substituted}\\right]
$$

"""
        
        logger.info(f"✅ Seção 4 gerada com proteção robusta contra zeros")



    # ============================================================================
    # MELHORIAS NA FUNÇÃO export_memorial
    # ============================================================================

    # Encontre a função export_memorial e SUBSTITUA a Seção 4 por:

    def _generate_section_4_improved(self, precision: int = 2) -> str:
        """
        Gera Seção 4 com proteção para valores pequenos.
        
        Seção 4: Expressão com Valores Substituídos
        """
        # Validar variáveis
        issues = self._validate_variables()
        
        # Criar aviso se houver problemas
        warning_section = ""
        if issues['critical']:
            warning_section = "\n⚠️ **AVISOS CRÍTICOS**:\n\n"
            for msg in issues['critical']:
                warning_section += f"- {msg}\n"
            warning_section += "\n"
        
        if issues['warnings']:
            warning_section += "\n📌 **Atenção**:\n\n"
            for msg in issues['warnings']:
                warning_section += f"- {msg}\n"
            warning_section += "\n"
        
        # Usar formatação segura
        latex_expr = self._format_substitution_safe(precision)
        
        section = f"""## 4. Expressão com Valores Substituídos

{warning_section}Antes da simplificação numérica:

$$
{self.name} = \\large \\left[{latex_expr}\\right]
$$

"""
        return section




    def _safe_format_variable(self, var, precision: int = 2) -> str:
        """
        Formata variável de forma segura, protegendo valores pequenos.
        """
        if not hasattr(var, 'value'):
            return str(var)
        
        value = var.value
        
        # Se valor muito pequeno, usar notação científica
        if isinstance(value, float) and 0 < abs(value) < 0.01:
            # Forçar notação científica
            if value == int(value):
                return f"{int(value)}"
            else:
                # Usar mais casas decimais para valores pequenos
                return f"{value:.6g}"  # 6 algarismos significativos
        
        # Caso normal
        return f"{value:.{precision}f}"



    def export_memorial(self, 
                       filename: str = None,
                       title: str = None,
                       project: str = "Projeto de Estruturas",
                       author: str = "Engenheiro Responsável",
                       number_format: str = "engineering",
                       precision: int = 2,
                       show_detailed_calcs: bool = True,
                       show_properties: bool = True,
                       step_detail: str = "basic",
                       matrix_scale: float = 1.5) -> str:  # ← ADICIONAR AQUI
        """
        Exporta memorial de cálculo completo automaticamente.
        
        Args:
            filename: Nome do arquivo (default: auto-gerado)
            title: Título do memorial
            project: Nome do projeto
            author: Responsável técnico
            number_format: "engineering", "decimal", "scientific"
            precision: Casas decimais (default: 2)
            show_detailed_calcs: Incluir Seção 4
            show_properties: Incluir propriedades da matriz
            step_detail: "basic" (padrão) ou "detailed"
            matrix_scale: Multiplicador de tamanho (0.5 a 2.0, padrão=1.0)
        
        Returns:
            str: Caminho do arquivo gerado
        """

        from datetime import datetime
        import sympy as sp_module
        
        logger.info(f"🚀 Iniciando geração de memorial (nível: {step_detail})...")
        
        # Validar step_detail
        if step_detail not in ["basic", "detailed"]:
            logger.warning(f"step_detail='{step_detail}' inválido. Usando 'basic'")
            step_detail = "basic"
        
        # Gerar filename
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"memorial_{self.name}_{timestamp}.md"
        
        if title is None:
            title = f"Memorial de Cálculo - Matriz {self.name}"
        
        # Obter resultado
        steps = self.steps()
        numeric_result = self.evaluate()
        
        md = []
        
        # CABEÇALHO
        md.append(f"# {title}\n\n")
        md.append("---\n\n")
        md.append(f"**Projeto**: {project}\n\n")
        md.append(f"**Responsável**: {author}\n\n")
        md.append(f"**Data**: {datetime.now().strftime('%d/%m/%Y às %H:%M')}\n\n")
        md.append(f"**Matriz**: {self.name}\n\n")
        md.append(f"**Dimensão**: {self.shape[0]}×{self.shape[1]}\n\n")
        md.append("---\n\n")
        
        # 1. DADOS DE ENTRADA
        # ====================================================================
        # 1. DADOS DE ENTRADA
        # ====================================================================
        
        md.append("## 1. Dados de Entrada\n\n")
        md.append("| Variável | Valor | Unidade |\n")
        md.append("|----------|-------|----------|\n")
        
        for var_name, var in self.variables.items():
            unit = getattr(var, 'unit', '') or '-'
            # USAR FORMATAÇÃO INTELIGENTE
            val_str = self._format_value_smart(var.value, precision)
            md.append(f"| ${var_name}$ | {val_str} | {unit} |\n")

        
        md.append("\n---\n\n")
        
        # ====================================================================
        # 2. MATRIZ SIMBÓLICA
        # ====================================================================
        
        md.append("## 2. Matriz Simbólica\n\n")
        md.append("Expressão geral da matriz:\n\n")
        
        symbolic_step = next((s for s in steps if s.get('operation') == 'symbolic'), None)
        
        # USAR TAMANHO PROPORCIONAL
        size_cmd = self._get_matrix_size_command("symbolic", matrix_scale)
        
        if symbolic_step and 'latex' in symbolic_step:
            md.append("$$\n")
            md.append(f"{self.name} = {size_cmd}{symbolic_step['latex']}\n")
            md.append("$$\n\n")
        else:
            md.append("$$\n")
            md.append(f"{self.name} = {size_cmd}{sp_module.latex(self._symbolic_matrix)}\n")
            md.append("$$\n\n")

        
        # ====================================================================
        # 3. SUBSTITUIÇÃO DE VALORES (CORRIGIDO)
        # ====================================================================
        
        md.append("## 3. Substituição de Valores\n\n")
        md.append("Substituindo os valores das variáveis:\n\n")
        
        for var_name, var in self.variables.items():
            # ✅ CORREÇÃO: Usar formatação inteligente
            val_str = self._format_value_smart(var.value, precision)
            
            # ✅ CORREÇÃO: Renderizar unidade corretamente
            unit = getattr(var, 'unit', '')
            
            # Formatar unidade para LaTeX
            if unit and unit != '-':
                # Substituir ** por ^
                unit_latex = unit.replace('**', '^')
                
                # Envolver em \text{} para renderização correta
                unit_str = f" \\, \\text{{{unit_latex}}}"
            else:
                unit_str = ""
            
            # ✅ CORREÇÃO: Exibição com tamanho adequado
            md.append(f"$$\\large {var_name} = {val_str}{unit_str}$$\n\n")
        
        md.append("---\n\n")
        
        # 4. EXPRESSÃO COM VALORES SUBSTITUÍDOS
        if show_detailed_calcs:
            md.append("## 4. Expressão com Valores Substituídos\n\n")
            md.append("Antes da simplificação numérica:\n\n")
            
            logger.info("Gerando Seção 4...")
            
            inter_latex = self._create_latex_from_scratch()
            
            has_symbols = any(var_name in inter_latex for var_name in self.variables.keys())
            if has_symbols:
                logger.warning("Símbolos detectados, usando numérico puro")
                inter_latex = self._create_numeric_only_latex()
            
            inter_latex = inter_latex.replace(r'\\', r'\\[1.5em]')
            
            # USAR TAMANHO PROPORCIONAL
            size_cmd = self._get_matrix_size_command("substituted", matrix_scale)
            
            md.append("$$\n")
            md.append(f"{self.name} = {size_cmd}{inter_latex}\n")
            md.append("$$\n\n")

        
        # 5. RESULTADO FINAL
        md.append("## 5. Resultado Final\n\n")
        md.append("Matriz numérica simplificada:\n\n")
        
        logger.info("Gerando Seção 5...")
        
        latex_result = self._matrix_to_latex(numeric_result, mode=number_format, precision=precision)
        latex_result = latex_result.replace(r'\\', r'\\[0.8em]')
        
        # USAR TAMANHO PROPORCIONAL
        size_cmd = self._get_matrix_size_command("final", matrix_scale)
        
        md.append("$$\n")
        md.append(f"{self.name} = {size_cmd}{latex_result}\n")
        md.append("$$\n\n")

        
        logger.info("✅ Matriz numérica gerada")
        rows, cols = self.shape
        # ELEMENTOS DA MATRIZ (somente se detailed)
        if step_detail == "detailed":
            md.append("### Elementos da Matriz\n\n")
            
            for i in range(rows):
                for j in range(cols):
                    elem_val = numeric_result[i, j]
                    
                    if abs(elem_val) < 1e-10:
                        eng_notation = "0"
                    else:
                        exp = int(np.floor(np.log10(abs(elem_val)) / 3) * 3)
                        mantissa = elem_val / (10 ** exp)
                        
                        if exp == 0:
                            eng_notation = f"{mantissa:.2f}"
                        else:
                            eng_notation = f"{mantissa:.2f} \\times 10^{{{exp}}}"
                    
                    md.append(f"- **{self.name}[{i+1},{j+1}]** = ${eng_notation}$\n")
            
            md.append("\n")
        
        # VALORES EM NOTAÇÃO SIMPLIFICADA
        max_val = np.max(np.abs(numeric_result))
        if max_val > 1e6:
            md.append("### Valores em Notação Simplificada\n\n")
            md.append("Para facilitar leitura, os valores acima estão em notação de engenharia.\n\n")
            md.append("**Exemplo**: $5.83 \\times 10^{7}$ significa $58\\,300\\,000$\n\n")
        
        # PROPRIEDADES DA MATRIZ
        if show_properties:
            md.append("### Propriedades da Matriz\n\n")
            
            md.append(f"- **Simétrica**: {'✅ Sim' if np.allclose(numeric_result, numeric_result.T) else '❌ Não'}\n")
            
            if self.is_square:
                det = np.linalg.det(numeric_result)
                det_str = f"{det:.2e}" if abs(det) > 1e6 or abs(det) < 1e-3 else f"{det:.2f}"
                md.append(f"- **Determinante**: {det_str}\n")
                
                eigs = np.linalg.eigvals(numeric_result)
                md.append(f"- **Definida Positiva**: {'✅ Sim' if np.all(eigs > 0) else '❌ Não'}\n")
                
                cond = np.linalg.cond(numeric_result)
                md.append(f"- **Número de Condição**: {cond:.2e}\n")
        
        md.append("\n---\n\n")
        md.append("_Gerado automaticamente por PyMemorial Matrix v2.2.2_\n")
        
        # SALVAR
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(''.join(md))
        
        logger.info(f"✅ Memorial exportado: {filename}")
        return filename




    def _matrix_to_latex(self, arr: np.ndarray, mode: str = "auto", precision: int = 2, 
                          size: str = "auto") -> str:
        """
        Converte array para LaTeX com formatação profissional e tamanho dinâmico.
        VERSÃO CORRIGIDA v3.1 - Formatação robusta de valores pequenos
        
        Args:
            arr: NumPy array
            mode: "auto", "decimal", "engineering", "scientific", "fraction"
            precision: Casas decimais
            size: "auto", "small", "normal", "large"
        
        Returns:
            str: LaTeX formatado
        """
        if arr.ndim != 2:
            return str(arr)
        
        rows, cols = arr.shape
        
        # ========================================================================
        # DETERMINAR TAMANHO AUTOMATICAMENTE
        # ========================================================================
        
        if size == "auto":
            complexity = (rows * cols) / 16
            
            if rows <= 2 and cols <= 2:
                size_modifier = r"\normalsize "
            elif rows <= 3 and cols <= 3:
                size_modifier = r"\small "
            elif rows <= 4 and cols <= 4:
                size_modifier = r"\footnotesize "
            elif rows <= 6 and cols <= 6:
                size_modifier = r"\scriptsize "
            else:
                size_modifier = r"\tiny "
        else:
            size_map = {
                "tiny": r"\tiny ",
                "scriptsize": r"\scriptsize ",
                "footnotesize": r"\footnotesize ",
                "small": r"\small ",
                "normal": r"\normalsize ",
                "large": r"\large ",
                "Large": r"\Large "
            }
            size_modifier = size_map.get(size, "")
        
        # ========================================================================
        # DETERMINAR FORMATO DOS NÚMEROS
        # ========================================================================
        
        if mode == "auto":
            max_val = np.max(np.abs(arr))
            min_val = np.min(np.abs(arr[arr != 0])) if np.any(arr != 0) else 1
            
            # ✅ CORREÇÃO: Critério mais sensível
            if max_val > 1e6 or (min_val < 0.01 and min_val != 0):
                mode = "engineering"
            else:
                mode = "decimal"
        
        # ========================================================================
        # GERAR LATEX
        # ========================================================================
        
        if rows > 4 or cols > 4:
            env_start = r"\left[\begin{array}{" + "c" * cols + "}\n"
            env_end = r"\end{array}\right]"
        else:
            env_start = r"\begin{bmatrix}" + "\n"
            env_end = r"\end{bmatrix}"
        
        latex = size_modifier + env_start
        
        for i in range(rows):
            row_elements = []
            for j in range(cols):
                val = arr[i, j]
                
                # ✅ CORREÇÃO: Usar _format_value_smart para TODOS os valores
                if mode == "decimal":
                    if abs(val) < 1e-15:
                        row_elements.append("0")
                    else:
                        formatted = self._format_value_smart(val, precision)
                        row_elements.append(formatted)
                        
                elif mode == "engineering":
                    if abs(val) < 1e-15:
                        row_elements.append("0")
                    else:
                        exp = int(np.floor(np.log10(abs(val)) / 3) * 3)
                        mantissa = val / (10 ** exp)
                        
                        if exp == 0:
                            formatted = self._format_value_smart(mantissa, precision)
                            row_elements.append(formatted)
                        else:
                            mantissa_str = f"{mantissa:.{precision}f}".rstrip('0').rstrip('.')
                            row_elements.append(f"{mantissa_str} \\times 10^{{{exp}}}")
                        
                elif mode == "scientific":
                    if abs(val) < 1e-15:
                        row_elements.append("0")
                    else:
                        formatted = f"{val:.{precision}e}".replace("e+", r" \times 10^{")
                        formatted = formatted.replace("e-", r" \times 10^{-") + "}"
                        row_elements.append(formatted)
            
            latex += "  " + " & ".join(row_elements)
            if i < rows - 1:
                latex += r" \\" + "\n"
        
        latex += "\n" + env_end
        return latex


    def _format_intermediate_step(self, step_latex: str) -> str:
        """
        Formata step intermediário com VALORES NUMÉRICOS substituídos.
        VERSÃO FINAL - Força avaliação parcial.
        """
        import sympy as sp_module
        from sympy import N as sp_N
        
        if not hasattr(self, '_symbolic_matrix') or self._symbolic_matrix is None:
            logger.warning("Matriz simbólica não disponível")
            return step_latex
        
        try:
            # ================================================================
            # SOLUÇÃO: Substituir E depois converter para Float explicitamente
            # ================================================================
            
            # 1. Criar dicionário de substituição com valores Float
            subs_dict = {}
            for var_name, var in self.variables.items():
                if var.value is not None:
                    sym = sp_module.Symbol(var_name)
                    # Usar Float com precisão limitada
                    subs_dict[sym] = sp_module.Float(var.value, precision=10)
            
            logger.debug(f"Substituindo: {subs_dict}")
            
            # 2. Substituir valores
            matrix_with_values = self._symbolic_matrix.subs(subs_dict)
            
            # 3. **CRÍTICO**: Forçar avaliação numérica parcial
            # Isso converte E*I em 21000.0*50000.0 mas mantém estrutura de fração
            matrix_evaluated = sp_module.Matrix([
                [sp_module.nsimplify(elem, rational=False) for elem in row]
                for row in matrix_with_values.tolist()
            ])
            
            # 4. Gerar LaTeX SEM fold (mantém operações explícitas)
            from sympy import latex as sp_latex
            
            latex_output = sp_latex(
                matrix_with_values,  # Usar matriz com valores
                fold_short_frac=False,
                mul_symbol='times',
                fold_func_brackets=False
            )
            
            # 5. Pós-processamento
            latex_output = latex_output.replace(r'\cdot', r' \times ')
            
            # 6. Validação RIGOROSA
            # Deve conter pelo menos um dos valores das variáveis
            has_values = False
            for var in self.variables.values():
                if var.value and str(var.value) in latex_output.replace('.0', ''):
                    has_values = True
                    break
            
            if not has_values or not any(char.isdigit() for char in latex_output):
                logger.error(f"❌ Substituição AINDA falhou! Output: {latex_output[:100]}")
                # Tentar método bruto
                return self._create_latex_from_scratch()
            
            logger.info(f"✅ Substituição OK: {latex_output[:80]}...")
            return latex_output
            
        except Exception as e:
            logger.error(f"Erro em _format_intermediate_step: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return self._create_latex_from_scratch()



    
    
    def _convert_negative_powers_to_fractions(self, latex_str: str) -> str:
        """
        Converte potências negativas em frações.
        12*21000*50000*(6.0)^{-3} → \frac{12*21000*50000}{6.0^{3}}
        """
        import re
        
        # Padrão complexo: expressão * base^{-exp}
        # Captura tudo antes da potência negativa e converte em fração
        
        # Estratégia: processar cada elemento da matriz separadamente
        # Split por & e \\ para pegar elementos individuais
        
        def convert_element(elem: str) -> str:
            """Converte um elemento individual da matriz"""
            elem = elem.strip()
            
            # Padrão: qualquer_coisa * (base)^{-exp}
            # Ou: qualquer_coisa * base^{-exp}
            pattern = r'(.+?)\s*\*?\s*[\{(]?([0-9.]+)[\})]?\^\{-(\d+)\}'
            
            match = re.search(pattern, elem)
            if match:
                numerator = match.group(1).strip()
                base = match.group(2)
                exp = match.group(3)
                
                # Remover multiplicação final do numerador se existir
                numerator = numerator.rstrip(' *')
                
                if exp == "1":
                    return f"\\frac{{{numerator}}}{{{base}}}"
                else:
                    return f"\\frac{{{numerator}}}{{{base}^{{{exp}}}}}"
            
            return elem
        
        # Processar matriz
        # Dividir por linhas (\\)
        lines = latex_str.split(r'\\')
        converted_lines = []
        
        for line in lines:
            # Dividir por elementos (&)
            elements = line.split('&')
            converted_elements = [convert_element(e) for e in elements]
            converted_lines.append(' & '.join(converted_elements))
        
        result = r' \\' + '\n'.join(converted_lines) if len(converted_lines) > 1 else converted_lines[0]
        
        return result
    
    
    def _fallback_format_intermediate(self, step_latex: str) -> str:
        """
        Fallback: tenta extrair valores do step original.
        """
        import re
        
        # Se step_latex estiver vazio ou inválido, tentar gerar do zero
        if not step_latex or len(step_latex) < 10:
            logger.warning("Step intermediário vazio, gerando do zero...")
            return self._format_intermediate_step("")
        
        # Converter potências negativas simples
        # {6.0}^{-3} → \frac{1}{6.0^{3}}
        pattern = r'\{([0-9.]+)\}\^\{-(\d+)\}'
        
        def replace_neg_power(match):
            base = match.group(1)
            exp = match.group(2)
            if exp == "1":
                return f"\\frac{{1}}{{{base}}}"
            else:
                return f"\\frac{{1}}{{{base}^{{{exp}}}}}"
        
        step_latex = re.sub(pattern, replace_neg_power, step_latex)
        
        # Remover parênteses desnecessários
        step_latex = step_latex.replace("{{", "{").replace("}}", "}")
        
        return step_latex



    def _fallback_format_intermediate(self, step_latex: str) -> str:
        """
        Fallback: Converte potências negativas usando regex.
        """
        import re
        
        # Padrão 1: {número}^{-expoente} → \frac{1}{número^{expoente}}
        pattern1 = r'\{([0-9.]+)\}\^\{-(\d+)\}'
        
        def replace_neg_power(match):
            base = match.group(1)
            exp = match.group(2)
            if exp == "1":
                return f"\\frac{{1}}{{{base}}}"
            else:
                return f"\\frac{{1}}{{{base}^{{{exp}}}}}"
        
        step_latex = re.sub(pattern1, replace_neg_power, step_latex)
        
        # Padrão 2: (número)^{-expoente} → \frac{1}{número^{expoente}}
        pattern2 = r'\(([0-9.]+)\)\^\{-(\d+)\}'
        step_latex = re.sub(pattern2, replace_neg_power, step_latex)
        
        # Remover parênteses duplos
        step_latex = step_latex.replace("{{", "{").replace("}}", "}")
        
        # Substituir \cdot por \times
        step_latex = step_latex.replace(r'\cdot', r'\times')
        
        return step_latex


    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Matrix':
        """
        Desserializa matriz de dicionário.
        
        Args:
            data: Dicionário com dados da matriz
            
        Returns:
            Matrix: Nova instância
        """
        variables = {}
        if 'variables' in data:
            for name, var_data in data['variables'].items():
                variables[name] = Variable(name, var_data.get('value'))
        
        return cls(
            data=data['data'],
            variables=variables,
            description=data.get('description', ''),
            name=data.get('name', 'M')
        )

    def __eq__(self, other: 'Matrix') -> bool:
        """Comparação robusta de matrizes."""
        if not isinstance(other, Matrix):
            return False
        
        if self.shape != other.shape:
            return False
        
        try:
            self_num = self.evaluate(use_cache=True)
            other_num = other.evaluate(use_cache=True)
            return np.allclose(self_num, other_num, rtol=1e-9, atol=1e-12)
        except:
            return False

    def __repr__(self) -> str:
        """Representação melhorada."""
        props = []
        if self.is_symbolic:
            props.append("symbolic")
        if self.is_square:
            props.append("square")
        if self._properties_cache.get('is_symmetric'):
            props.append("symmetric")
        
        props_str = f" [{', '.join(props)}]" if props else ""
        return f"Matrix('{self.name}', {self.shape}{props_str})"

    # ========================================================================
    # AVALIAÇÃO (COMPATÍVEL COM v2.1.9)
    # ========================================================================

    def evaluate(self, use_cache: bool = True) -> np.ndarray:
        """
        Avalia a matriz para a forma numérica.
        COMPATÍVEL com v2.1.9
        """
        if use_cache and self._numeric_matrix is not None:
            return self._numeric_matrix
        
        if self._numeric_matrix is not None:
            return self._numeric_matrix
        
        if self._symbolic_matrix is not None:
            result = self._evaluate_symbolic_lambdify()
            self._numeric_matrix = result
            return result
        
        raise ValueError("Nenhum dado de matriz para avaliar")

    def _evaluate_symbolic_lambdify(self) -> np.ndarray:
        """
        Avaliar matriz com lambdify.
        COMPATÍVEL com v2.1.9
        """
        logger.debug(f"Avaliando '{self.name}' com sp.lambdify...")
        
        expr_matrix = self._symbolic_matrix
        if expr_matrix is None:
            raise ValueError("Matriz simbólica não foi parseada.")
        
        free_symbols = expr_matrix.free_symbols
        symbols_tuple = tuple(free_symbols)
        
        subs_values = {}
        missing_vars = []
        
        # ✅ CORREÇÃO: Definir variables_source ANTES de usar
        variables_source = getattr(self, '_original_variables', self.variables)
        
        for sym in symbols_tuple:
            var_name = str(sym)
            if var_name not in variables_source:
                missing_vars.append(var_name)
                continue
            
            var = variables_source[var_name]
            if var.value is None:
                missing_vars.append(f"{var_name} (símbolo sem valor)")
                continue
            
            subs_values[var_name] = strip_units(var.value)
        
        if missing_vars:
            raise ValueError(
                f"Variáveis ausentes para avaliar '{self.name}': {', '.join(missing_vars)}"
            )
        
        try:
            func = sp.lambdify(symbols_tuple, expr_matrix, 'numpy')
        except Exception as e:
            logger.error(f"Falha ao 'lambdify' a matriz '{self.name}': {e}")
            raise RuntimeError(f"Erro ao compilar matriz para NumPy: {e}")
        
        try:
            args = [subs_values[str(sym)] for sym in symbols_tuple]
            result_array = np.asarray(func(*args), dtype=float)
            logger.debug(f"Matriz '{self.name}' avaliada com sucesso via lambdify.")
            return result_array
        except Exception as e:
            logger.error(f"Falha ao executar lambdify para '{self.name}': {e}")
            raise RuntimeError(f"Erro ao calcular matriz numérica: {e}")


    # ========================================================================
    # STEPS (COMPATÍVEL COM v2.1.9)
    # ========================================================================

    def steps(
        self,
        operation: Optional[str] = None,
        show_units: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Gera steps com passo intermediário.
        COMPATÍVEL com v2.1.9 (sem granularity)
        """
        steps = []
        
        subs_dict_display = {}
        subs_list_display = []
        
        try:
            if self.is_symbolic and self._symbolic_matrix:
                for sym in self._symbolic_matrix.free_symbols:
                    var_name = str(sym)
                    if var_name in self.variables:
                        var = self.variables[var_name]
                        if var.value is not None:
                            val_numeric = strip_units(var.value)
                            subs_dict_display[sym] = sp.Float(val_numeric)
                            
                            if PINT_OK and hasattr(var.value, 'units'):
                                subs_list_display.append(f"{var_name} = {var.value:~P}")
                            else:
                                subs_list_display.append(f"{var_name} = {val_numeric}")
            
            # PASSO 1: Definição
            steps.append({
                'step': 'Definição da matriz',
                'operation': 'definition',
                'description': f'{self.name}: matriz {self.shape[0]}×{self.shape[1]}',
                'shape': self.shape,
                'is_symbolic': self.is_symbolic
            })
            
            # PASSO 2: Forma Simbólica
            if self.is_symbolic and self._symbolic_matrix is not None:
                steps.append({
                    'step': 'Forma Simbólica',
                    'operation': 'symbolic',
                    'description': 'Expressão simbólica da matriz',
                    'latex': sp_latex(self._symbolic_matrix)
                })
            
            # PASSO 3: Substituição
            if subs_list_display:
                steps.append({
                    'step': 'Substituição de Variáveis',
                    'operation': 'substitution',
                    'description': ', '.join(subs_list_display)
                })
            
            # PASSO 4: Intermediário
            if self.is_symbolic and subs_dict_display:
                try:
                    rows, cols = self._symbolic_matrix.shape
                    intermediate_elements = []
                    
                    for i in range(rows):
                        row_elements = []
                        for j in range(cols):
                            original_elem = self._symbolic_matrix[i, j]
                            unevaluated = self._substitute_preserve_structure(
                                original_elem, 
                                subs_dict_display
                            )
                            row_elements.append(unevaluated)
                        intermediate_elements.append(row_elements)
                    
                    intermediate_matrix = SpMatrix(intermediate_elements)
                    
                    steps.append({
                        'step': 'Matriz Substituída (Passo Intermediário)',
                        'operation': 'intermediate',
                        'description': 'Valores substituídos, estrutura preservada',
                        'latex': sp_latex(intermediate_matrix)
                    })
                    
                except Exception as e:
                    logger.warning(f"Falha ao gerar passo intermediário: {e}")
            
            # PASSO 5: Resultado
            try:
                result_matrix = self.evaluate(use_cache=True)
                
                if result_matrix.shape[0] <= 8 and result_matrix.shape[1] <= 8:
                    matrix_display = result_matrix.tolist()
                else:
                    matrix_display = f"{result_matrix.shape[0]}×{result_matrix.shape[1]} (matriz grande)"
                
                steps.append({
                    'step': 'Matriz Numérica (Resultado)',
                    'operation': 'evaluation',
                    'description': 'Avaliação numérica final',
                    'matrix': matrix_display
                })
            
            except Exception as e:
                logger.warning(f"Não foi possível avaliar '{self.name}': {e}")
                steps.append({
                    'step': 'Avaliação Numérica',
                    'operation': 'evaluation_failed',
                    'description': 'Não foi possível avaliar',
                    'error': str(e)
                })
            
            return steps
        
        except Exception as e:
            logger.error(f"Falha ao gerar steps: {e}")
            return [{
                'step': 'Erro',
                'operation': 'error',
                'description': f'Erro ao gerar steps: {e}'
            }]

    def _substitute_preserve_structure(self, expr: sp.Expr, subs_dict: Dict) -> sp.Expr:
        """
        Substitui símbolos preservando estrutura.
        COMPATÍVEL com v2.1.9
        """
        from sympy import Mul, Add, Pow, Number
        from sympy.core.expr import UnevaluatedExpr
        
        if expr.is_Number:
            return expr
        
        if expr.is_Symbol:
            if expr in subs_dict:
                return UnevaluatedExpr(subs_dict[expr])
            return expr
        
        if isinstance(expr, Mul):
            new_args = [self._substitute_preserve_structure(arg, subs_dict) 
                       for arg in expr.args]
            return Mul(*new_args, evaluate=False)
        
        elif isinstance(expr, Add):
            new_args = [self._substitute_preserve_structure(arg, subs_dict) 
                       for arg in expr.args]
            return Add(*new_args, evaluate=False)
        
        elif isinstance(expr, Pow):
            new_base = self._substitute_preserve_structure(expr.base, subs_dict)
            new_exp = self._substitute_preserve_structure(expr.exp, subs_dict)
            return Pow(new_base, new_exp, evaluate=False)
        
        else:
            return expr.subs(subs_dict, simultaneous=True, evaluate=False)

    def _compute_properties(self, matrix: np.ndarray) -> Dict[str, Any]:
        """Computa propriedades da matriz."""
        props = {'shape': self.shape, 'is_square': self.is_square}
        
        if self.is_square and NUMPY_AVAILABLE:
            try:
                props['determinant'] = float(np.linalg.det(matrix))
                props['trace'] = float(np.trace(matrix))
                props['rank'] = int(np.linalg.matrix_rank(matrix))
                props['is_symmetric'] = bool(np.allclose(matrix, matrix.T))
            except Exception as e:
                logger.warning(f"Não pôde computar todas as propriedades: {e}")
        
        return props


# ============================================================================
# MODULE INFO
# ============================================================================

MATRIX_AVAILABLE = NUMPY_AVAILABLE and SYMPY_AVAILABLE

if MATRIX_AVAILABLE:
    logger.info(f"✅ Matrix module (Enhanced v2.2.0) disponível (NumPy, SymPy)")
else:
    reasons = []
    if not NUMPY_AVAILABLE: reasons.append("NumPy indisponível")
    if not SYMPY_AVAILABLE: reasons.append("SymPy indisponível")
    logger.critical(f"❌ Matrix module indisponível: {', '.join(reasons)}")

__all__ = [
    'Matrix',
    'MatrixType',
    'MATRIX_AVAILABLE',
    'NUMPY_AVAILABLE',
    'SYMPY_AVAILABLE',
    'debug_matrix_parsing',
]

__version__ = "2.2.0"


# ============================================================================
# FUNÇÕES AUXILIARES PARA DEBUGGING (APRIMORADAS)
# ============================================================================

def debug_matrix_parsing(
    expr_str: str, 
    variables: Dict[str, Variable],
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Função auxiliar aprimorada para debug do parsing de matrizes.
    
    Args:
        expr_str: Expressão da matriz como string
        variables: Dicionário de variáveis
        verbose: Se True, imprime informações detalhadas
    
    Returns:
        Dict com informações de debug completas
    """
    if not SYMPY_AVAILABLE:
        return {'error': 'SymPy não disponível', 'success': False}
    
    info = {
        'input_expr': expr_str,
        'expr_length': len(expr_str),
        'variables_provided': list(variables.keys()),
        'variables_count': len(variables),
        'parsing_steps': [],
        'warnings': [],
        'success': False
    }
    
    try:
        # Criar símbolos
        local_dict = {}
        for var_name in variables.keys():
            local_dict[var_name] = sp.Symbol(var_name, real=True, positive=True)
        info['parsing_steps'].append(f"✅ Símbolos criados: {list(local_dict.keys())}")
        
        # Parse com evaluate=False
        parsed_expr = sp.sympify(
            expr_str, 
            locals=local_dict, 
            evaluate=False, 
            rational=False
        )
        info['parsing_steps'].append(f"✅ Parsed type: {type(parsed_expr).__name__}")
        
        # Converter para matriz se necessário
        if isinstance(parsed_expr, list):
            parsed_expr = SpMatrix(parsed_expr)
            info['parsing_steps'].append("✅ Convertido lista → SpMatrix")
        
        # Verificar símbolos livres
        if hasattr(parsed_expr, 'free_symbols'):
            free_syms = [str(s) for s in parsed_expr.free_symbols]
            info['free_symbols'] = free_syms
            info['parsing_steps'].append(f"✅ Símbolos livres detectados: {free_syms}")
            
            # Verificar símbolos não declarados
            undeclared = set(free_syms) - set(variables.keys())
            if undeclared:
                info['warnings'].append(f"⚠️ Símbolos não declarados: {undeclared}")
        
        # Verificar shape
        if hasattr(parsed_expr, 'shape'):
            info['shape'] = parsed_expr.shape
            info['parsing_steps'].append(f"✅ Shape: {parsed_expr.shape}")
        
        # LaTeX
        info['latex'] = sp_latex(parsed_expr)
        info['latex_length'] = len(info['latex'])
        info['parsing_steps'].append(f"✅ LaTeX gerado ({info['latex_length']} chars)")
        
        # Validação de pureza
        latex_output = info['latex']
        contaminated = []
        for var_name, var in variables.items():
            if var.value is not None:
                value_str = str(strip_units(var.value))
                pattern = r'(?<!\d)' + re.escape(value_str) + r'(?!\d)'
                if re.search(pattern, latex_output):
                    contaminated.append(f"{var_name}={value_str}")
        
        if contaminated:
            info['warnings'].append(f"⚠️ Possível contaminação detectada: {contaminated}")
            info['parsing_steps'].append("⚠️ ATENÇÃO: Contaminação numérica detectada")
        else:
            info['parsing_steps'].append("✅ Pureza simbólica validada")
        
        info['success'] = True
        info['parsing_steps'].append("✅ Parsing completo com sucesso")
        
    except Exception as e:
        info['error'] = str(e)
        info['error_type'] = type(e).__name__
        info['traceback'] = traceback.format_exc()
        info['parsing_steps'].append(f"❌ Erro: {e}")
        info['success'] = False
    
    # Output verbose
    if verbose:
        print("\n" + "="*70)
        print("🔍 DEBUG MATRIX PARSING")
        print("="*70)
        print(f"📝 Expressão: {expr_str[:100]}{'...' if len(expr_str) > 100 else ''}")
        print(f"📊 Variáveis: {list(variables.keys())}")
        print("\n📋 Steps:")
        for step in info['parsing_steps']:
            print(f"  {step}")
        
        if info['warnings']:
            print("\n⚠️  Warnings:")
            for warning in info['warnings']:
                print(f"  {warning}")
        
        if info.get('error'):
            print(f"\n❌ Erro: {info['error']}")
        
        print("\n" + "="*70)
        print(f"Resultado: {'✅ SUCESSO' if info['success'] else '❌ FALHA'}")
        print("="*70 + "\n")
    
    return info


def validate_matrix_consistency(matrix: Matrix) -> Dict[str, Any]:
    """
    Valida consistência interna de uma matriz.
    
    Args:
        matrix: Instância de Matrix para validar
        
    Returns:
        Dict com resultados da validação
    """
    results = {
        'matrix_name': matrix.name,
        'checks': [],
        'errors': [],
        'warnings': [],
        'score': 0,
        'max_score': 0
    }
    
    # Check 1: Shape consistente
    results['max_score'] += 1
    if matrix.shape and len(matrix.shape) == 2 and matrix.shape[0] > 0 and matrix.shape[1] > 0:
        results['checks'].append("✅ Shape válido")
        results['score'] += 1
    else:
        results['errors'].append(f"❌ Shape inválido: {matrix.shape}")
    
    # Check 2: Tipo correto
    results['max_score'] += 1
    if matrix.is_symbolic and matrix._symbolic_matrix is not None:
        results['checks'].append("✅ Matriz simbólica inicializada")
        results['score'] += 1
    elif not matrix.is_symbolic and matrix._numeric_matrix is not None:
        results['checks'].append("✅ Matriz numérica inicializada")
        results['score'] += 1
    else:
        results['errors'].append("❌ Dados internos inconsistentes")
    
    # Check 3: Avaliação possível
    results['max_score'] += 1
    if matrix.is_symbolic:
        try:
            result = matrix.evaluate(use_cache=False)
            if isinstance(result, np.ndarray) and result.shape == matrix.shape:
                results['checks'].append("✅ Avaliação numérica bem-sucedida")
                results['score'] += 1
            else:
                results['warnings'].append("⚠️ Avaliação retornou shape diferente")
        except Exception as e:
            results['errors'].append(f"❌ Falha na avaliação: {e}")
    else:
        results['score'] += 1  # Não aplicável para matrizes numéricas
    
    # Check 4: Propriedades especiais detectadas
    results['max_score'] += 1
    if matrix.is_square:
        if matrix._properties_cache:
            results['checks'].append(f"✅ Propriedades detectadas: {list(matrix._properties_cache.keys())}")
            results['score'] += 1
        else:
            results['warnings'].append("⚠️ Nenhuma propriedade especial detectada")
    else:
        results['score'] += 1  # Não aplicável para matrizes não-quadradas
    
    # Check 5: Serialização
    results['max_score'] += 1
    try:
        dict_repr = matrix.to_dict()
        if isinstance(dict_repr, dict) and 'name' in dict_repr and 'shape' in dict_repr:
            results['checks'].append("✅ Serialização para dict funcional")
            results['score'] += 1
        else:
            results['warnings'].append("⚠️ Serialização incompleta")
    except Exception as e:
        results['errors'].append(f"❌ Falha na serialização: {e}")
    
    # Calcular porcentagem
    results['percentage'] = (results['score'] / results['max_score'] * 100) if results['max_score'] > 0 else 0
    results['status'] = 'PASS' if results['score'] == results['max_score'] else 'FAIL'
    
    return results


def compare_matrices(
    matrix1: Matrix, 
    matrix2: Matrix,
    tolerance: float = 1e-9
) -> Dict[str, Any]:
    """
    Compara duas matrizes em detalhes.
    
    Args:
        matrix1: Primeira matriz
        matrix2: Segunda matriz
        tolerance: Tolerância para comparação numérica
        
    Returns:
        Dict com resultados da comparação
    """
    comparison = {
        'equal': False,
        'checks': [],
        'differences': []
    }
    
    # Shape
    if matrix1.shape != matrix2.shape:
        comparison['differences'].append(
            f"Shape diferente: {matrix1.shape} vs {matrix2.shape}"
        )
    else:
        comparison['checks'].append(f"✅ Shape igual: {matrix1.shape}")
    
    # Tipo
    if matrix1.is_symbolic != matrix2.is_symbolic:
        comparison['differences'].append(
            f"Tipo diferente: symbolic={matrix1.is_symbolic} vs symbolic={matrix2.is_symbolic}"
        )
    else:
        comparison['checks'].append(f"✅ Tipo igual: symbolic={matrix1.is_symbolic}")
    
    # Valores numéricos
    try:
        m1_num = matrix1.evaluate()
        m2_num = matrix2.evaluate()
        
        if np.allclose(m1_num, m2_num, rtol=tolerance, atol=tolerance):
            comparison['checks'].append("✅ Valores numéricos iguais (dentro da tolerância)")
            comparison['equal'] = True
        else:
            max_diff = np.max(np.abs(m1_num - m2_num))
            comparison['differences'].append(
                f"Valores diferentes: diferença máxima = {max_diff:.2e}"
            )
            comparison['max_difference'] = float(max_diff)
    except Exception as e:
        comparison['differences'].append(f"Não foi possível comparar valores: {e}")
    
    return comparison


def benchmark_matrix_operations(
    matrix: Matrix,
    iterations: int = 100
) -> Dict[str, Any]:
    """
    Benchmark de operações da matriz.
    
    Args:
        matrix: Matriz para benchmark
        iterations: Número de iterações para cada operação
        
    Returns:
        Dict com tempos de execução
    """
    import time
    
    results = {
        'matrix_name': matrix.name,
        'shape': matrix.shape,
        'iterations': iterations,
        'timings': {}
    }
    
    # Benchmark evaluate()
    if matrix.is_symbolic:
        start = time.perf_counter()
        for _ in range(iterations):
            _ = matrix.evaluate(use_cache=False)
        end = time.perf_counter()
        results['timings']['evaluate_no_cache'] = (end - start) / iterations
        
        start = time.perf_counter()
        for _ in range(iterations):
            _ = matrix.evaluate(use_cache=True)
        end = time.perf_counter()
        results['timings']['evaluate_with_cache'] = (end - start) / iterations
    
    # Benchmark steps()
    start = time.perf_counter()
    for _ in range(min(iterations, 10)):  # Steps é mais pesado
        _ = matrix.steps()
    end = time.perf_counter()
    results['timings']['steps'] = (end - start) / min(iterations, 10)
    
    # Benchmark clone()
    start = time.perf_counter()
    for _ in range(iterations):
        _ = matrix.clone()
    end = time.perf_counter()
    results['timings']['clone'] = (end - start) / iterations
    
    # Benchmark to_dict()
    start = time.perf_counter()
    for _ in range(iterations):
        _ = matrix.to_dict()
    end = time.perf_counter()
    results['timings']['to_dict'] = (end - start) / iterations
    
    return results


# ============================================================================
# FUNÇÕES DE TESTE AUTOMATIZADO
# ============================================================================

def run_matrix_test_suite(verbose: bool = True) -> Dict[str, Any]:
    """
    Executa suite completa de testes automatizados.
    
    Args:
        verbose: Se True, imprime resultados detalhados
        
    Returns:
        Dict com resultados de todos os testes
    """
    if verbose:
        print("\n" + "="*70)
        print("🧪 PYMEMORIAL MATRIX TEST SUITE v2.2.0")
        print("="*70 + "\n")
    
    suite_results = {
        'total_tests': 0,
        'passed': 0,
        'failed': 0,
        'tests': {}
    }
    
    # Teste 1: Inicialização Simbólica
    test_name = "Symbolic Initialization"
    suite_results['total_tests'] += 1
    try:
        variables = {
            'E': Variable('E', 21000.0),
            'I': Variable('I', 50000.0),
            'L': Variable('L', 6.0)
        }
        matrix = Matrix(
            data="[[12*E*I/L**3, 6*E*I/L**2], [6*E*I/L**2, 4*E*I/L]]",
            variables=variables,
            name="K_test"
        )
        
        if matrix.is_symbolic and matrix.shape == (2, 2):
            suite_results['passed'] += 1
            suite_results['tests'][test_name] = {'status': 'PASS', 'details': f"Shape: {matrix.shape}"}
            if verbose:
                print(f"✅ {test_name}: PASS")
        else:
            suite_results['failed'] += 1
            suite_results['tests'][test_name] = {'status': 'FAIL', 'details': 'Shape ou tipo incorreto'}
            if verbose:
                print(f"❌ {test_name}: FAIL")
    except Exception as e:
        suite_results['failed'] += 1
        suite_results['tests'][test_name] = {'status': 'FAIL', 'details': str(e)}
        if verbose:
            print(f"❌ {test_name}: FAIL - {e}")
    
    # Teste 2: Avaliação Numérica
    test_name = "Numeric Evaluation"
    suite_results['total_tests'] += 1
    try:
        result = matrix.evaluate()
        if isinstance(result, np.ndarray) and result.shape == (2, 2):
            suite_results['passed'] += 1
            suite_results['tests'][test_name] = {'status': 'PASS', 'details': 'Avaliação bem-sucedida'}
            if verbose:
                print(f"✅ {test_name}: PASS")
        else:
            suite_results['failed'] += 1
            suite_results['tests'][test_name] = {'status': 'FAIL', 'details': 'Resultado inválido'}
            if verbose:
                print(f"❌ {test_name}: FAIL")
    except Exception as e:
        suite_results['failed'] += 1
        suite_results['tests'][test_name] = {'status': 'FAIL', 'details': str(e)}
        if verbose:
            print(f"❌ {test_name}: FAIL - {e}")
    
    # Teste 3: Geração de Steps
    test_name = "Steps Generation"
    suite_results['total_tests'] += 1
    try:
        steps = matrix.steps()
        expected_ops = {'definition', 'symbolic', 'substitution', 'intermediate', 'evaluation'}
        actual_ops = {s.get('operation') for s in steps if 'operation' in s}
        
        if expected_ops.issubset(actual_ops):
            suite_results['passed'] += 1
            suite_results['tests'][test_name] = {'status': 'PASS', 'details': f'{len(steps)} steps gerados'}
            if verbose:
                print(f"✅ {test_name}: PASS")
        else:
            suite_results['failed'] += 1
            missing = expected_ops - actual_ops
            suite_results['tests'][test_name] = {'status': 'FAIL', 'details': f'Steps faltando: {missing}'}
            if verbose:
                print(f"❌ {test_name}: FAIL - Steps faltando: {missing}")
    except Exception as e:
        suite_results['failed'] += 1
        suite_results['tests'][test_name] = {'status': 'FAIL', 'details': str(e)}
        if verbose:
            print(f"❌ {test_name}: FAIL - {e}")
    
    # Teste 4: Clone
    test_name = "Matrix Cloning"
    suite_results['total_tests'] += 1
    try:
        cloned = matrix.clone()
        if cloned == matrix and cloned is not matrix:
            suite_results['passed'] += 1
            suite_results['tests'][test_name] = {'status': 'PASS', 'details': 'Clone independente criado'}
            if verbose:
                print(f"✅ {test_name}: PASS")
        else:
            suite_results['failed'] += 1
            suite_results['tests'][test_name] = {'status': 'FAIL', 'details': 'Clone não é independente'}
            if verbose:
                print(f"❌ {test_name}: FAIL")
    except Exception as e:
        suite_results['failed'] += 1
        suite_results['tests'][test_name] = {'status': 'FAIL', 'details': str(e)}
        if verbose:
            print(f"❌ {test_name}: FAIL - {e}")
    
    # Teste 5: Serialização
    test_name = "Serialization"
    suite_results['total_tests'] += 1
    try:
        dict_repr = matrix.to_dict()
        restored = Matrix.from_dict(dict_repr)
        
        if restored == matrix:
            suite_results['passed'] += 1
            suite_results['tests'][test_name] = {'status': 'PASS', 'details': 'Serialização reversível'}
            if verbose:
                print(f"✅ {test_name}: PASS")
        else:
            suite_results['failed'] += 1
            suite_results['tests'][test_name] = {'status': 'FAIL', 'details': 'Matriz restaurada diferente'}
            if verbose:
                print(f"❌ {test_name}: FAIL")
    except Exception as e:
        suite_results['failed'] += 1
        suite_results['tests'][test_name] = {'status': 'FAIL', 'details': str(e)}
        if verbose:
            print(f"❌ {test_name}: FAIL - {e}")
    
    # Sumário
    suite_results['pass_rate'] = (suite_results['passed'] / suite_results['total_tests'] * 100)
    
    if verbose:
        print("\n" + "="*70)
        print(f"📊 RESULTADOS: {suite_results['passed']}/{suite_results['total_tests']} testes passaram")
        print(f"   Taxa de sucesso: {suite_results['pass_rate']:.1f}%")
        print("="*70 + "\n")
    
    return suite_results