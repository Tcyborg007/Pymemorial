# src/pymemorial/core/units.py

"""
Sistema de unidades inteligente PyMemorial v2.0

FILOSOFIA:
- Wrapper robusto sobre Pint
- Fallback gracioso se Pint não instalado
- Zero pré-configurações de norma
- Integração com config.py

DESENVOLVIMENTO INCREMENTAL:
- PARTE 1: UnitRegistry Singleton ✅ ATUAL
- PARTE 2: UnitParser (aliases PT-BR)
- PARTE 3: UnitValidator (dimensional)
- PARTE 4: UnitFormatter (LaTeX)
- PARTE 5: Integration + Backward Compat
"""

from __future__ import annotations

import logging
from typing import Optional, Union, Any
from threading import Lock

logger = logging.getLogger(__name__)

__all__ = [
    'UnitRegistry',
    'get_unit_registry',
    'reset_unit_registry',
    'UnitError',
    'PINT_AVAILABLE'
]


# =============================================================================
# PINT IMPORT E FEATURE FLAG
# =============================================================================

try:
    import pint
    PINT_AVAILABLE = True
    logger.debug("Pint library loaded successfully")
except ImportError:
    pint = None
    PINT_AVAILABLE = False
    logger.warning(
        "Pint library not installed. Unit functionality limited. "
        "Install with: poetry add pint"
    )
try:
    import sympy as sp
    SYMPY_AVAILABLE = True
except ImportError:
    sp = None
    SYMPY_AVAILABLE = False

# =============================================================================
# EXCEÇÕES
# =============================================================================

class UnitError(Exception):
    """Erro de operação com unidades."""
    pass


# =============================================================================
# UNITREGISTRY - SINGLETON WRAPPER PINT (PARTE 1)
# =============================================================================

class UnitRegistry:
    """
    Wrapper inteligente sobre Pint UnitRegistry.
    
    PADRÃO: Singleton thread-safe
    
    CARACTERÍSTICAS:
    - Lazy loading (só carrega Pint quando necessário)
    - Custom definitions (tonne_force, percent)
    - Fallback gracioso se Pint não disponível
    - Thread-safe para ambientes concorrentes
    
    Attributes:
        ureg: Pint UnitRegistry (None se Pint não disponível)
    
    Examples:
        >>> reg = get_unit_registry()
        >>> q = reg.parse('10 m')
        >>> q.magnitude
        10.0
        
        >>> # Custom unit
        >>> tf = reg.parse('1 tf')  # tonne_force
        >>> reg.convert(tf, 'kN')
        <Quantity(9.80665, 'kilonewton')>
    """
    
    def __init__(self):
        """
        Inicializa UnitRegistry.
        
        IMPORTANTE: Use get_unit_registry() ao invés de instanciar diretamente!
        """
        self.ureg = None
        self._initialized = False
        
        if PINT_AVAILABLE:
            self._initialize_pint()
    
    def _initialize_pint(self):
        """
        Inicializa Pint UnitRegistry com custom definitions.
        """
        if self._initialized or not PINT_AVAILABLE:
            return
        
        try:
            # Criar UnitRegistry Pint
            self.ureg = pint.UnitRegistry()
            
            # Configuração padrão
            self.ureg.default_format = "~P"  # Pretty print
            
            # Definições customizadas
            self._add_custom_definitions()
            
            self._initialized = True
            logger.debug("UnitRegistry initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Pint: {e}")
            self.ureg = None
            self._initialized = False
    
    def _add_custom_definitions(self):
        """
        Adiciona definições customizadas de unidades.
        
        CUSTOMIZAÇÕES ENGENHARIA CIVIL BR:
        - tonne_force (tf): 1 tf = 1000 kgf
        - percent (%): Pint já tem, verificar disponibilidade
        
        IMPORTANTE: Definições são adicionadas de forma defensiva
        para evitar conflitos com definições padrão do Pint.
        """
        if not self.ureg:
            return
        
        try:
            # =====================================================================
            # tonne_force (tf) = 1000 kgf
            # =====================================================================
            # 1 kgf = 9.80665 N → 1 tf = 9806.65 N
            try:
                # Testar se 'tf' já existe
                test_tf = self.ureg('1 tf')
                logger.debug("Unit 'tonne_force' (tf) already available")
            except:
                # Não existe, definir
                self.ureg.define('tonne_force = 1000 * kilogram_force = tf')
                logger.debug("Custom unit 'tonne_force' (tf) added")
            
            # =====================================================================
            # percent (%) - Pint já tem nativamente
            # =====================================================================
            # Verificar se % funciona (Pint v0.20+ tem suporte nativo)
            try:
                test_percent = self.ureg('50%')
                # Se chegou aqui, % já funciona
                logger.debug("Unit 'percent' (%) already available")
            except:
                # Fallback: tentar definir se não funcionar
                try:
                    self.ureg.define('percent = 0.01 = %')
                    logger.debug("Custom unit 'percent' (%) added as fallback")
                except:
                    logger.warning("Failed to define 'percent' unit")
            
        except Exception as e:
            logger.warning(f"Failed to add custom definitions: {e}")



    def calculate_resultant_unit(
        self,
        expr: Any,
        unit_context: Dict[str, Optional[str]]
    ) -> Optional[str]:
        """
        Calcula a unidade resultante de uma expressão SymPy usando Pint.
        
        Usa análise dimensional pura (sem valores numéricos).
        
        Args:
            expr: Expressão SymPy
            unit_context: Dicionário {nome_variavel: unidade_str}
            
        Returns:
            String da unidade resultante simplificada, ou None se adimensional
            
        Raises:
            DimensionalError: Se houver inconsistência dimensional
            UnitError: Erro inesperado
        """
        if not self.ureg or not SYMPY_AVAILABLE:
            logger.debug("Cálculo de unidade resultante pulado (Pint ou SymPy indisponível).")
            return None
        
        try:
            # Construir dicionário de unidades Pint
            unit_dict_pint = {}
            expr_symbols = expr.free_symbols if hasattr(expr, 'free_symbols') else set()
            
            for sym in expr_symbols:
                name = str(sym)
                unit_str = unit_context.get(name)
                
                try:
                    if unit_str:
                        unit_obj = self.ureg(unit_str).units
                        unit_dict_pint[sym] = unit_obj
                    else:
                        unit_dict_pint[sym] = self.ureg.dimensionless
                except Exception as e:
                    logger.warning(
                        f"Unidade inválida '{unit_str}' para '{name}': {e}. "
                        f"Tratando como adimensional."
                    )
                    unit_dict_pint[sym] = self.ureg.dimensionless
            
            # Calcular dimensão resultante
            result_unit = self._evaluate_dimensional_expression(expr, unit_dict_pint)
            
            if result_unit is None or result_unit == self.ureg.dimensionless:
                return None
            
            # ===== SIMPLIFICAR UNIDADE =====
            try:
                # Criar quantity para manipular
                dummy_quantity = 1.0 * result_unit
                
                # ✅ TENTAR MANTER UNIDADES CONSISTENTES COM INPUT
                # Detectar qual sistema de unidades foi usado no input
                input_units_str = list(unit_context.values())
                
                # Se todas as unidades de comprimento são mm, manter resultado em mm
                if any('mm' in str(u) for u in input_units_str if u):
                    # Tentar converter para mm se for comprimento
                    try:
                        if dummy_quantity.dimensionality == self.ureg('mm').dimensionality:
                            result_in_preferred = dummy_quantity.to('mm')
                            return f"{result_in_preferred.units:~P}"
                    except Exception:
                        pass  # Se falhar, usar simplificação padrão
                
                # Se todas as unidades de comprimento são m, manter em m
                elif any(str(u) == 'm' for u in input_units_str if u):
                    try:
                        if dummy_quantity.dimensionality == self.ureg('m').dimensionality:
                            result_in_preferred = dummy_quantity.to('m')
                            return f"{result_in_preferred.units:~P}"
                    except Exception:
                        pass
                
                # ✅ SIMPLIFICAÇÃO PADRÃO: Usar unidades compactas (não base)
                # Tenta 'compact' ao invés de 'base_units'
                result_unit_str = f"{dummy_quantity.units:~C}"
                
                # Se ainda muito complexo, tentar simplificar
                if len(result_unit_str) > 20:
                    try:
                        simplified = dummy_quantity.to_base_units()
                        result_unit_str = f"{simplified.units:~P}"
                    except Exception:
                        pass
                
                return result_unit_str
                
            except Exception as e:
                logger.debug(f"Não foi possível simplificar unidade: {e}")
                return f"{result_unit:~P}"
        
        except pint.DimensionalityError as e:
            logger.error(f"Erro dimensional em calculate_resultant_unit para {expr}: {e}")
            raise DimensionalError(f"Inconsistência dimensional: {e}") from e
        
        except Exception as e:
            logger.error(f"Erro inesperado em calculate_resultant_unit para {expr}: {e}", exc_info=True)
            logger.warning("Falha ao calcular unidade via Pint, continuando sem unidade")
            return None

            
            # Simplificar para unidades base se possível
            try:
                # Criar uma quantity com a unidade para simplificar
                dummy_quantity = 1.0 * result_unit
                simplified = dummy_quantity.to_base_units()
                result_unit_str = f"{simplified.units:~P}"
                
                # Se simplificado é muito feio, tentar compact
                if len(result_unit_str) > 20:
                    result_unit_str = f"{simplified.units:~C}"
                
                return result_unit_str
                
            except Exception as e:
                logger.debug(f"Não foi possível simplificar unidade: {e}")
                return f"{result_unit:~P}"
        
        except pint.DimensionalityError as e:
            logger.error(f"Erro dimensional em calculate_resultant_unit para {expr}: {e}")
            raise DimensionalError(f"Inconsistência dimensional: {e}") from e
        
        except Exception as e:
            logger.error(f"Erro inesperado em calculate_resultant_unit para {expr}: {e}", exc_info=True)
            # Não falhar silenciosamente - logar mas retornar None
            logger.warning("Falha ao calcular unidade via Pint, continuando sem unidade")
            return None
    
    def _evaluate_dimensional_expression(
        self, 
        expr: Any, 
        unit_dict: Dict[Any, Any]
    ) -> Optional[Any]:
        """
        Avalia dimensão de uma expressão SymPy recursivamente.
        
        Args:
            expr: Expressão SymPy
            unit_dict: Dicionário {Symbol: Pint Unit}
            
        Returns:
            Pint Unit resultante
        """
        import sympy as sp
        
        # Caso base: símbolo
        if isinstance(expr, sp.Symbol):
            return unit_dict.get(expr, self.ureg.dimensionless)
        
        # Caso base: número
        if isinstance(expr, sp.Number):
            return self.ureg.dimensionless
        
        # Adição/Subtração: todas unidades devem ser iguais
        if isinstance(expr, sp.Add):
            units = [self._evaluate_dimensional_expression(arg, unit_dict) for arg in expr.args]
            # Verificar se todas são iguais
            first_unit = units[0]
            for u in units[1:]:
                if u != first_unit:
                    raise pint.DimensionalityError(
                        first_unit, u,
                        extra_msg="Não é possível somar/subtrair quantidades com unidades diferentes"
                    )
            return first_unit
        
        # Multiplicação: multiplicar unidades
        if isinstance(expr, sp.Mul):
            result_unit = self.ureg.dimensionless
            for arg in expr.args:
                arg_unit = self._evaluate_dimensional_expression(arg, unit_dict)
                result_unit = result_unit * arg_unit
            return result_unit
        
        # Potência: elevar unidade à potência
        if isinstance(expr, sp.Pow):
            base = expr.args[0]
            exponent = expr.args[1]
            
            base_unit = self._evaluate_dimensional_expression(base, unit_dict)
            
            # Expoente deve ser adimensional (número)
            if isinstance(exponent, sp.Number):
                exp_val = float(exponent)
                return base_unit ** exp_val
            else:
                # Expoente não é número puro
                exp_unit = self._evaluate_dimensional_expression(exponent, unit_dict)
                if exp_unit != self.ureg.dimensionless:
                    raise pint.DimensionalityError(
                        exp_unit, self.ureg.dimensionless,
                        extra_msg="Expoente deve ser adimensional"
                    )
                # Se chegou aqui, expoente é expressão que resulta em adimensional
                # Avaliar numericamente para obter valor
                exp_val = float(exponent.evalf())
                return base_unit ** exp_val
        
        # Funções (sin, cos, etc): argumento deve ser adimensional, resultado adimensional
        if isinstance(expr, sp.Function):
            # Verificar argumentos
            for arg in expr.args:
                arg_unit = self._evaluate_dimensional_expression(arg, unit_dict)
                if arg_unit != self.ureg.dimensionless:
                    logger.warning(
                        f"Argumento de função {expr.func} tem unidade {arg_unit}, "
                        f"esperado adimensional"
                    )
            return self.ureg.dimensionless
        
        # Fallback: tentar avaliar numericamente e assumir adimensional
        logger.debug(f"Tipo de expressão não reconhecido para análise dimensional: {type(expr)}")
        return self.ureg.dimensionless



# --- Adicionar DimensionalError às exceções em units.py se já não existir ---
class DimensionalError(UnitError):
    """Erro de incompatibilidade dimensional."""
    pass

    
    # =========================================================================
    # API PÚBLICA - PARSING
    # =========================================================================
    
    def parse(self, expression: Union[str, float, int]) -> Union[Any, float]:
        """
        Parse string de unidade → Quantity.
        
        Args:
            expression: String com unidade (ex: "10 m", "50 kN")
                        ou valor numérico (retorna como está)
        
        Returns:
            Pint Quantity se Pint disponível, senão float
        
        Raises:
            UnitError: Se parsing falhar
        
        Examples:
            >>> reg = get_unit_registry()
            >>> q = reg.parse('10 m')
            >>> q.magnitude
            10.0
            
            >>> # Sem Pint: retorna float
            >>> q = reg.parse('10 m')  # fallback
            >>> q
            10.0
        """
        # Se já é numérico, retornar como está
        if isinstance(expression, (int, float)):
            return expression
        
        # Se Pint disponível, usar Pint
        if PINT_AVAILABLE and self.ureg:
            try:
                return self.ureg(expression)
            except Exception as e:
                raise UnitError(f"Failed to parse '{expression}': {e}")
        
        # Fallback: extrair apenas número
        try:
            # Parsing simples: pegar primeiro número
            import re
            match = re.search(r'[-+]?[0-9]*\.?[0-9]+', str(expression))
            if match:
                return float(match.group())
            return 0.0
        except Exception as e:
            raise UnitError(f"Failed to parse '{expression}' (fallback): {e}")
    
    def convert(self, quantity: Any, target_unit: str) -> Any:
        """
        Converte Quantity para outra unidade.
        
        Args:
            quantity: Pint Quantity
            target_unit: Unidade alvo (ex: 'kN', 'mm')
        
        Returns:
            Quantity convertida
        
        Raises:
            UnitError: Se conversão impossível
        
        Examples:
            >>> reg = get_unit_registry()
            >>> q = reg.parse('1 m')
            >>> q_mm = reg.convert(q, 'mm')
            >>> q_mm.magnitude
            1000.0
        """
        if not PINT_AVAILABLE or not self.ureg:
            raise UnitError("Pint not available. Cannot convert units.")
        
        try:
            return quantity.to(target_unit)
        except Exception as e:
            raise UnitError(f"Failed to convert to '{target_unit}': {e}")


# =============================================================================
# SINGLETON GLOBAL
# =============================================================================

_global_unit_registry: Optional[UnitRegistry] = None
_registry_lock = Lock()


def get_unit_registry() -> UnitRegistry:
    """
    Retorna instância singleton global do UnitRegistry.
    
    Thread-safe.
    
    Returns:
        UnitRegistry singleton
    
    Examples:
        >>> reg = get_unit_registry()
        >>> q = reg.parse('10 m')
    """
    global _global_unit_registry
    
    if _global_unit_registry is None:
        with _registry_lock:
            if _global_unit_registry is None:
                _global_unit_registry = UnitRegistry()
    
    return _global_unit_registry


def reset_unit_registry():
    """
    Reseta singleton (útil para testes).
    
    Examples:
        >>> reset_unit_registry()
        >>> reg = get_unit_registry()  # Nova instância
    """
    global _global_unit_registry
    
    with _registry_lock:
        _global_unit_registry = None


# =============================================================================
# FUNÇÕES DE CONVENIÊNCIA (FUTURAS - PARTES 2-5)
# =============================================================================

# TODO PARTE 2: UnitParser (aliases PT-BR)
# TODO PARTE 3: UnitValidator (dimensional consistency)
# TODO PARTE 4: UnitFormatter (LaTeX)
# TODO PARTE 5: Integration + Backward Compatibility
# =============================================================================
# UNITPARSER - ALIASES PT-BR E NORMALIZAÇÃO (PARTE 2)
# =============================================================================

class UnitParser:
    """
    Parser inteligente com aliases PT-BR e normalização.
    
    FUNCIONALIDADES:
    - Aliases em português (metros → m, quilonewtons → kN)
    - Normalização de unidades (kN.m → kN*m, kN/m2 → kN/m**2)
    - Cache de conversões (performance)
    - Fallback gracioso (caso Pint não disponível)
    
    Examples:
        >>> reg = get_unit_registry()
        >>> parser = UnitParser(reg)
        >>> 
        >>> # Alias PT-BR
        >>> q = parser.parse_with_alias('10 metros')
        >>> q.magnitude
        10.0
        >>>
        >>> # Normalização
        >>> parser.normalize('kN.m')
        'kN*m'
    """
    
    # Aliases PT-BR → Unidades Pint
    PTBR_ALIASES = {
        # Comprimento
        'metros': 'm',
        'metro': 'm',
        'centimetros': 'cm',
        'centimetro': 'cm',
        'milimetros': 'mm',
        'milimetro': 'mm',
        'quilometros': 'km',
        'quilometro': 'km',
        
        # Força
        'newtons': 'N',
        'newton': 'N',
        'quilonewtons': 'kN',
        'quilonewton': 'kN',
        'meganewtons': 'MN',
        'meganewton': 'MN',
        
        # Pressão/Tensão
        'pascals': 'Pa',
        'pascal': 'Pa',
        'quilopascals': 'kPa',
        'quilopascal': 'kPa',
        'megapascals': 'MPa',
        'megapascal': 'MPa',
        'gigapascals': 'GPa',
        'gigapascal': 'GPa',
        
        # Momento
        'quilonewton-metro': 'kN*m',
        'quilonewton-metros': 'kN*m',
        
        # Massa
        'quilogramas': 'kg',
        'quilograma': 'kg',
        'toneladas': 't',
        'tonelada': 't',
        
        # Área
        'metros-quadrados': 'm**2',
        'metro-quadrado': 'm**2',
        'centimetros-quadrados': 'cm**2',
        'centimetro-quadrado': 'cm**2',
    }
    
    def __init__(self, registry: UnitRegistry):
        """
        Inicializa parser.
        
        Args:
            registry: UnitRegistry para usar
        """
        self.registry = registry
        self._cache = {}  # Cache de conversões
    
    def parse_with_alias(self, expression: str) -> Any:
        """
        Parse com suporte a aliases PT-BR.
        
        Args:
            expression: String com unidade (ex: "10 metros", "50 quilonewtons")
        
        Returns:
            Pint Quantity ou float (fallback)
        
        Examples:
            >>> parser = UnitParser(get_unit_registry())
            >>> q = parser.parse_with_alias('10 metros')
            >>> q.magnitude
            10.0
        """
        # Separar valor e unidade
        parts = expression.strip().split()
        if len(parts) < 2:
            # Sem unidade, tentar parse direto
            return self.registry.parse(expression)
        
        value_str = parts[0]
        unit_str = ' '.join(parts[1:]).lower()
        
        # Substituir aliases PT-BR
        # IMPORTANTE: Ordem importa! Mais específicos primeiro
        # Ex: 'quilonewtons' deve ser processado ANTES de 'newtons'
        
        # Ordenar por comprimento (descendente) para evitar substituições parciais
        sorted_aliases = sorted(
            self.PTBR_ALIASES.items(),
            key=lambda x: len(x[0]),
            reverse=True
        )
        
        # CORREÇÃO: Processar TODAS as palavras, não apenas primeira!
        for ptbr, pint_unit in sorted_aliases:
            if ptbr in unit_str:
                unit_str = unit_str.replace(ptbr, pint_unit)
                # NÃO BREAK! Continuar substituindo outras palavras
        
        # Reconstruir expressão
        expr_normalized = f"{value_str} {unit_str}"
        
        # Parse com registry
        return self.registry.parse(expr_normalized)


    
    def normalize(self, unit_str: str) -> str:
        """
        Normaliza string de unidade para formato Pint.
        
        NORMALIZAÇÕES:
        - '.' → '*' (kN.m → kN*m)
        - '/m2' → '/m**2'
        - Espaços desnecessários removidos
        
        Args:
            unit_str: String de unidade
        
        Returns:
            String normalizada
        
        Examples:
            >>> parser = UnitParser(get_unit_registry())
            >>> parser.normalize('kN.m')
            'kN*m'
            >>> parser.normalize('kN/m2')
            'kN/m**2'
        """
        # Cache lookup
        if unit_str in self._cache:
            return self._cache[unit_str]
        
        normalized = unit_str
        
        # Normalização 1: '.' → '*'
        # kN.m é ambíguo (produto ou abreviação?)
        # Assumir produto (momento) por convenção engenharia
        if '.' in normalized and '*' not in normalized:
            normalized = normalized.replace('.', '*')
        
        # Normalização 2: Expoentes
        # /m2 → /m**2, /m3 → /m**3
        import re
        normalized = re.sub(r'/([a-zA-Z]+)(\d)', r'/\1**\2', normalized)
        
        # Normalização 3: Espaços
        normalized = normalized.strip()
        
        # Cache store
        self._cache[unit_str] = normalized
        
        return normalized
    
    def parse(self, expression: str) -> Any:
        """
        Parse com normalização automática.
        
        Args:
            expression: String com unidade (ex: "10 kN.m")
        
        Returns:
            Pint Quantity ou float (fallback)
        
        Examples:
            >>> parser = UnitParser(get_unit_registry())
            >>> q = parser.parse('10 kN.m')  # Ambíguo
            >>> # Sistema normaliza para kN*m
        """
        # Separar valor e unidade
        parts = expression.strip().split()
        if len(parts) < 2:
            # Sem unidade, tentar parse direto
            return self.registry.parse(expression)
        
        value_str = parts[0]
        unit_str = ' '.join(parts[1:])
        
        # Normalizar unidade
        unit_normalized = self.normalize(unit_str)
        
        # Reconstruir expressão
        expr_normalized = f"{value_str} {unit_normalized}"
        
        # Parse
        return self.registry.parse(expr_normalized)


# =============================================================================
# UNITVALIDATOR - VALIDAÇÃO DIMENSIONAL (PARTE 3)
# =============================================================================

class UnitValidator:
    """
    Validador dimensional inteligente.
    
    FUNCIONALIDADES:
    - Verificação de consistência dimensional
    - Validação de operações (adição, multiplicação, etc.)
    - Mensagens de erro claras e úteis
    
    Examples:
        >>> reg = get_unit_registry()
        >>> validator = UnitValidator(reg)
        >>> 
        >>> # Verificar compatibilidade
        >>> q1 = reg.parse('10 m')
        >>> q2 = reg.parse('50 cm')
        >>> validator.are_compatible(q1, q2)
        True
        >>>
        >>> # Validar operação
        >>> q3 = reg.parse('5 kg')
        >>> validator.validate_operation(q1, q3, 'add')
        # Raises UnitError (incompatible dimensions)
    """
    
    def __init__(self, registry: UnitRegistry):
        """
        Inicializa validador.
        
        Args:
            registry: UnitRegistry para usar
        """
        self.registry = registry
    
    def are_compatible(self, quantity1: Any, quantity2: Any) -> bool:
        """
        Verifica se duas quantidades têm dimensões compatíveis.
        
        Args:
            quantity1: Primeira quantidade
            quantity2: Segunda quantidade
        
        Returns:
            True se compatíveis (mesma dimensão)
        
        Examples:
            >>> validator = UnitValidator(get_unit_registry())
            >>> q1 = validator.registry.parse('10 m')
            >>> q2 = validator.registry.parse('50 cm')
            >>> validator.are_compatible(q1, q2)
            True
        """
        if not PINT_AVAILABLE or not self.registry.ureg:
            # Fallback: sem Pint, não pode validar
            return True
        
        try:
            # Pint permite conversão se dimensões compatíveis
            quantity1.to(quantity2.units)
            return True
        except:
            return False
    
    def validate_operation(
        self, 
        quantity1: Any, 
        quantity2: Any, 
        operation: str
    ):
        """
        Valida se operação é dimensional válida.
        
        REGRAS:
        - add/sub: requer dimensões compatíveis
        - mul/div: sempre válidas (criam novas dimensões)
        
        Args:
            quantity1: Primeira quantidade
            quantity2: Segunda quantidade
            operation: Operação ('add', 'sub', 'mul', 'div')
        
        Raises:
            UnitError: Se operação inválida dimensionalmente
        
        Examples:
            >>> validator = UnitValidator(get_unit_registry())
            >>> q1 = validator.registry.parse('10 m')
            >>> q2 = validator.registry.parse('5 kg')
            >>> validator.validate_operation(q1, q2, 'add')
            # Raises UnitError
        """
        if not PINT_AVAILABLE or not self.registry.ureg:
            # Fallback: sem Pint, não pode validar
            return
        
        # Operações que requerem compatibilidade dimensional
        if operation in ('add', 'sub'):
            if not self.are_compatible(quantity1, quantity2):
                raise UnitError(
                    f"Cannot {operation} incompatible units: "
                    f"{quantity1.units} and {quantity2.units}. "
                    f"Dimensions: {quantity1.dimensionality} vs {quantity2.dimensionality}"
                )
        
        # Multiplicação e divisão sempre válidas
        # (criam novas dimensões compostas)


# =============================================================================
# UNITFORMATTER - FORMATAÇÃO LATEX (PARTE 4)
# =============================================================================

class UnitFormatter:
    """
    Formatador LaTeX para unidades.
    
    FUNCIONALIDADES:
    - Formatação LaTeX de quantidades com unidades
    - Precisão customizável
    - Notação científica opcional
    - Formatação de unidades compostas
    
    Examples:
        >>> reg = get_unit_registry()
        >>> formatter = UnitFormatter(reg)
        >>> 
        >>> q = reg.parse('10 m')
        >>> formatter.to_latex(q)
        '10 \\, \\mathrm{m}'
        >>>
        >>> # Com precisão
        >>> q2 = reg.parse('3.14159 m')
        >>> formatter.to_latex(q2, precision=2)
        '3.14 \\, \\mathrm{m}'
    """
    
    def __init__(self, registry: UnitRegistry):
        """
        Inicializa formatter.
        
        Args:
            registry: UnitRegistry para usar
        """
        self.registry = registry
    
    def to_latex(
        self,
        quantity: Any,
        precision: int = 3,
        scientific: bool = False
    ) -> str:
        """
        Formata quantidade com unidade para LaTeX.
        
        Args:
            quantity: Quantidade a formatar
            precision: Casas decimais (padrão: 3)
            scientific: Usar notação científica (padrão: False)
        
        Returns:
            String LaTeX formatada
        
        Examples:
            >>> formatter = UnitFormatter(get_unit_registry())
            >>> q = formatter.registry.parse('10 m')
            >>> formatter.to_latex(q)
            '10 \\, \\mathrm{m}'
        """
        if not PINT_AVAILABLE or not self.registry.ureg:
            # Fallback: sem Pint, formatação básica
            return str(quantity)
        
        try:
            magnitude = quantity.magnitude
            units = quantity.units
            
            # Formatar magnitude
            if scientific and (abs(magnitude) >= 1000 or (abs(magnitude) < 0.01 and magnitude != 0)):
                # Notação científica
                import math
                if magnitude == 0:
                    mag_str = f"{magnitude:.{precision}f}"
                else:
                    exp = int(math.floor(math.log10(abs(magnitude))))
                    mantissa = magnitude / (10 ** exp)
                    mag_str = f"{mantissa:.{precision}f} \\times 10^{{{exp}}}"
            else:
                # Notação decimal padrão
                mag_str = f"{magnitude:.{precision}f}"
            
            # Formatar unidades
            unit_str = str(units)
            # Substituir * por \cdot para LaTeX
            unit_str = unit_str.replace('*', r' \cdot ')
            # Envolver em \mathrm{}
            unit_latex = f"\\mathrm{{{unit_str}}}"
            
            # Combinar
            return f"{mag_str} \\, {unit_latex}"
            
        except Exception as e:
            logger.warning(f"Failed to format quantity to LaTeX: {e}")
            return str(quantity)


# =============================================================================
# ATUALIZAR __all__
# =============================================================================

# Re-export Quantity
if PINT_AVAILABLE:
    try:
        import pint
        Quantity = pint.Quantity
    except ImportError:
        Quantity = float
else:
    Quantity = float

__all__ = [
    'UnitRegistry',
    'UnitParser',
    'UnitValidator',
    'UnitFormatter',
    'get_unit_registry',
    'reset_unit_registry',
    'UnitError',
    'PINT_AVAILABLE',
    'Quantity',  # ✅ Re-export
]



# =============================================================================
# TODO FUTURAS MELHORIAS (FASE 2-3)
# =============================================================================

# TODO FASE 2:
# - UnitConverter: Conversão batch de múltiplas quantidades
# - UnitDatabase: Cache persistente de conversões frequentes
# - UnitAliasManager: Aliases customizáveis por usuário

# TODO FASE 3:
# - Protocol IUnitSystem: Abstração para trocar Pint por outra lib
# - Suporte a sistemas de unidades não-SI (imperial)
# - Auto-detecção de unidades em textos (NLP)
