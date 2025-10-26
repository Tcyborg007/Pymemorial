# src/pymemorial/core/variable.py

"""
Variable v2.0 - Variável inteligente PyMemorial

FILOSOFIA:
- Zero pré-definições de norma
- Liberdade total de unidades
- Auto-aprendizado via registry
- Integração com config, ast_parser

DESENVOLVIMENTO INCREMENTAL:
- PARTE 1: Core básico (name, value, unit) ✅ ATUAL
- PARTE 2: Operadores matemáticos
- PARTE 3: Conversão LaTeX
- PARTE 4: Auto-registro registry
- PARTE 5: Histórico e comparações
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Any, Union, List
from datetime import datetime

__all__ = [
    'Variable',
    'VariableError'
]


# =============================================================================
# EXCEÇÕES
# =============================================================================

class VariableError(Exception):
    """Erro de operação com variáveis."""
    pass


# =============================================================================
# VARIABLE CORE (PARTE 1)
# =============================================================================

@dataclass
class Variable:
    """
    Variável inteligente do PyMemorial v2.0
    
    CARACTERÍSTICAS:
    - Suporte a escalares, vetores, matrizes
    - Unidades opcionais (integração Pint)
    - Auto-registro no custom_registry
    - Conversão LaTeX automática
    - Histórico de valores
    
    Attributes:
        name: Nome da variável (ex: 'M_d', 'gamma_c')
        value: Valor numérico ou None
        unit: Unidade como string (ex: 'kN', 'MPa', 'm')
        description: Descrição opcional
        auto_register: Auto-registrar no global registry
    
    Examples:
        >>> v = Variable(name='M', value=150.0)
        >>> v.name
        'M'
        >>> v.value
        150.0
        
        >>> # Com unidade
        >>> v = Variable(name='sigma', value=25, unit='MPa')
        >>> str(v)
        'sigma = 25 MPa'
    """
    
    # Atributos principais
    name: str
    value: Optional[Union[float, int, List, Any]] = None
    unit: Optional[str] = None
    description: Optional[str] = None
    
    # Controle de comportamento
    auto_register: bool = False
    
    # Histórico (será implementado na PARTE 5)
    _history: List = field(default_factory=list, repr=False, init=False)
    _created_at: datetime = field(default_factory=datetime.now, repr=False, init=False)
    
    def __post_init__(self):
        """
        Inicialização pós-criação.
        """
        # Converter string para float se possível
        if isinstance(self.value, str):
            try:
                self.value = float(self.value)
            except ValueError:
                pass
        
        # PARTE 4: Auto-registro no registry
        if self.auto_register:
            self._register_symbol()
        
        # PARTE 5: Não adicionar valor inicial ao histórico
        # (será adicionado quando update_value() for chamado pela primeira vez)


        
        # TODO PARTE 4: Auto-registro no registry
        # if self.auto_register:
        #     self._register_symbol()
    
    # =========================================================================
    # REPRESENTAÇÃO (PARTE 1)
    # =========================================================================
    
    def __str__(self) -> str:
        """
        String legível para humanos.
        
        Returns:
            String formatada "name = value unit"
        
        Examples:
            >>> v = Variable(name='M', value=150)
            >>> str(v)
            'M = 150'
            
            >>> v = Variable(name='F', value=100, unit='kN')
            >>> str(v)
            'F = 100 kN'
        """
        parts = [f"{self.name}"]
        
        if self.value is not None:
            parts.append(f"= {self.value}")
        
        if self.unit:
            parts.append(self.unit)
        
        return " ".join(parts)
    
    def __repr__(self) -> str:
        """
        Representação técnica para debugging.
        
        Returns:
            String com todos os atributos
        """
        parts = [f"Variable(name='{self.name}'"]
        
        if self.value is not None:
            parts.append(f"value={self.value}")
        
        if self.unit:
            parts.append(f"unit='{self.unit}'")
        
        if self.description:
            parts.append(f"description='{self.description}'")
        
        return ", ".join(parts) + ")"
    
    # =========================================================================
    # PROPRIEDADES UTILITÁRIAS
    # =========================================================================
    
    @property
    def is_scalar(self) -> bool:
        """Verifica se valor é escalar."""
        return isinstance(self.value, (int, float))
    
    @property
    def is_vector(self) -> bool:
        """Verifica se valor é vetor (list/array)."""
        return isinstance(self.value, (list, tuple))
    
    @property
    def is_none(self) -> bool:
        """Verifica se valor é None."""
        return self.value is None


# =============================================================================
# FUNÇÕES DE CONVENIÊNCIA (FUTURAS)
# =============================================================================

# TODO PARTE 3: to_latex()
# TODO PARTE 4: _register_symbol()
# TODO PARTE 5: update_value(), get_history(), rollback()


    # =========================================================================
    # OPERADORES MATEMÁTICOS (PARTE 2)
    # =========================================================================
    
    def _check_value_exists(self, operation: str):
        """
        Verifica se valor existe antes de operação.
        
        Raises:
            VariableError: Se valor é None
        """
        if self.value is None:
            raise VariableError(
                f"Cannot perform {operation} on Variable '{self.name}' "
                f"with None value"
            )
    
    def __add__(self, other: Union[Variable, int, float]) -> Variable:
        """
        Soma de variáveis ou variável + escalar.
        
        Args:
            other: Outra variável ou valor numérico
        
        Returns:
            Nova variável com resultado
        
        Examples:
            >>> v1 = Variable(name='a', value=10)
            >>> v2 = Variable(name='b', value=5)
            >>> v3 = v1 + v2
            >>> v3.value
            15
            
            >>> v4 = v1 + 20
            >>> v4.value
            30
        """
        self._check_value_exists('addition')
        
        if isinstance(other, Variable):
            other._check_value_exists('addition')
            result_value = self.value + other.value
            result_name = f"result_{id(self)}"  # Nome único
        else:
            result_value = self.value + other
            result_name = f"result_{id(self)}"
        
        return Variable(
            name=result_name,
            value=result_value,
            unit=self.unit  # Manter unidade do primeiro operando
        )
    
    def __sub__(self, other: Union[Variable, int, float]) -> Variable:
        """Subtração."""
        self._check_value_exists('subtraction')
        
        if isinstance(other, Variable):
            other._check_value_exists('subtraction')
            result_value = self.value - other.value
        else:
            result_value = self.value - other
        
        return Variable(
            name=f"result_{id(self)}",
            value=result_value,
            unit=self.unit
        )
    
    def __mul__(self, other: Union[Variable, int, float]) -> Variable:
        """Multiplicação."""
        self._check_value_exists('multiplication')
        
        if isinstance(other, Variable):
            other._check_value_exists('multiplication')
            result_value = self.value * other.value
        else:
            result_value = self.value * other
        
        return Variable(
            name=f"result_{id(self)}",
            value=result_value,
            unit=self.unit  # TODO PARTE 3: Combinar unidades
        )
    
    def __truediv__(self, other: Union[Variable, int, float]) -> Variable:
        """Divisão."""
        self._check_value_exists('division')
        
        if isinstance(other, Variable):
            other._check_value_exists('division')
            if other.value == 0:
                raise VariableError("Division by zero")
            result_value = self.value / other.value
        else:
            if other == 0:
                raise VariableError("Division by zero")
            result_value = self.value / other
        
        return Variable(
            name=f"result_{id(self)}",
            value=result_value,
            unit=self.unit
        )
    
    def __pow__(self, exponent: Union[Variable, int, float]) -> Variable:
        """Potenciação."""
        self._check_value_exists('exponentiation')
        
        if isinstance(exponent, Variable):
            exponent._check_value_exists('exponentiation')
            result_value = self.value ** exponent.value
        else:
            result_value = self.value ** exponent
        
        return Variable(
            name=f"result_{id(self)}",
            value=result_value,
            unit=self.unit
        )
    
    def __neg__(self) -> Variable:
        """Negação unária (-v)."""
        self._check_value_exists('negation')
        
        return Variable(
            name=f"-{self.name}",
            value=-self.value,
            unit=self.unit
        )
    
    # Operadores reversos (quando variável está à direita)
    def __radd__(self, other: Union[int, float]) -> Variable:
        """Soma reversa (escalar + variável)."""
        return self.__add__(other)
    
    def __rmul__(self, other: Union[int, float]) -> Variable:
        """Multiplicação reversa (escalar * variável)."""
        return self.__mul__(other)
    
    def __rsub__(self, other: Union[int, float]) -> Variable:
        """Subtração reversa (escalar - variável)."""
        self._check_value_exists('subtraction')
        return Variable(
            name=f"result_{id(self)}",
            value=other - self.value,
            unit=self.unit
        )
    
    def __rtruediv__(self, other: Union[int, float]) -> Variable:
        """Divisão reversa (escalar / variável)."""
        self._check_value_exists('division')
        if self.value == 0:
            raise VariableError("Division by zero")
        return Variable(
            name=f"result_{id(self)}",
            value=other / self.value,
            unit=self.unit
        )
    # =========================================================================
    # CONVERSÃO LATEX (PARTE 3)
    # =========================================================================
    
    def to_latex(self, include_unit: bool = False) -> str:
        """
        Converte nome da variável para LaTeX.
        
        INTEGRAÇÃO:
        1. Verifica se existe no custom_registry (usa latex customizado)
        2. Se não, usa ast_parser para gerar automaticamente
        
        Args:
            include_unit: Incluir unidade na representação
        
        Returns:
            String LaTeX formatada
        
        Examples:
            >>> v = Variable(name='M_d', value=150)
            >>> v.to_latex()
            'M_{d}'
            
            >>> v = Variable(name='gamma_c', value=1.4)
            >>> v.to_latex()
            '\\\\gamma_{c}'
            
            >>> # Com registry customizado
            >>> from pymemorial.symbols import get_global_registry
            >>> registry = get_global_registry()
            >>> registry.define('M_y', latex=r'M_{y}^{custom}')
            >>> v = Variable(name='M_y', value=100)
            >>> v.to_latex()
            'M_{y}^{custom}'
        """
        from pymemorial.symbols import get_global_registry
        from pymemorial.recognition.ast_parser import PyMemorialASTParser
        
        # 1. Verificar se existe no registry
        registry = get_global_registry()
        symbol = registry.get(self.name)
        
        if symbol is not None:
            # Usar LaTeX customizado do registry
            latex = symbol.latex
        else:
            # 2. Gerar automaticamente com ast_parser
            parser = PyMemorialASTParser()
            latex = parser.to_latex(self.name)
        
        # Adicionar unidade se solicitado
        if include_unit and self.unit:
            latex = f"{latex}\\,\\text{{{self.unit}}}"
        
        return latex


# =============================================================================
# FUNÇÕES AUXILIARES PARA REGISTRO (PARTE 4)
# =============================================================================

    def _register_symbol(self):
        """
        Registra símbolo automaticamente no global registry.
        
        Usa to_latex() para gerar LaTeX automaticamente.
        """
        from pymemorial.symbols import get_global_registry
        
        registry = get_global_registry()
        
        # Só registrar se ainda não existe
        if not registry.has(self.name):
            latex = self.to_latex()
            registry.define(
                name=self.name,
                latex=latex,
                description=self.description,
                overwrite=False
            )


    # =========================================================================
    # HISTÓRICO DE VALORES (PARTE 5)
    # =========================================================================
    
    def update_value(self, new_value: Union[float, int]):
        """
        Atualiza valor e registra no histórico.
        
        Args:
            new_value: Novo valor numérico
        
        Examples:
            >>> v = Variable(name='x', value=10)
            >>> v.update_value(20)
            >>> v.value
            20
            >>> len(v.get_history())
            2
        """
        # Adicionar valor atual ao histórico antes de mudar
        if self.value is not None:
            self._history.append((datetime.now(), self.value))
        
        self.value = new_value
    
    def get_history(self) -> List[tuple]:
        """
        Retorna histórico de valores.
        
        Returns:
            Lista de tuplas (timestamp, valor)
        
        Examples:
            >>> v = Variable(name='x', value=10)
            >>> v.update_value(20)
            >>> history = v.get_history()
            >>> len(history)
            2
        """
        return self._history.copy()
    
    def rollback(self, steps: int = 1):
        """
        Volta para valor anterior no histórico.
        
        Args:
            steps: Número de passos para voltar
        
        Raises:
            VariableError: Se não há histórico suficiente
        
        Examples:
            >>> v = Variable(name='y', value=5)      # history: []
            >>> v.update_value(10)                    # history: [(t, 5)], value=10
            >>> v.update_value(15)                    # history: [(t, 5), (t, 10)], value=15
            >>> v.rollback(1)                         # value=10 (history[-1])
            >>> v.value
            10
        """
        if steps < 1:
            raise VariableError("steps must be >= 1")
        
        if len(self._history) < steps:
            raise VariableError(
                f"Cannot rollback {steps} steps. "
                f"History has only {len(self._history)} entries"
            )
        
        # Pegar valor de 'steps' posições atrás
        target_index = -steps
        _, target_value = self._history[target_index]
        
        # Atualizar valor
        self.value = target_value
        
        # Remover entradas após rollback point
        self._history = self._history[:target_index]


    
    # =========================================================================
    # COMPARAÇÕES (PARTE 5)
    # =========================================================================
    
    def __eq__(self, other: Union[Variable, int, float]) -> bool:
        """Igualdade (==)."""
        if isinstance(other, Variable):
            return self.value == other.value
        return self.value == other
    
    def __lt__(self, other: Union[Variable, int, float]) -> bool:
        """Menor que (<)."""
        if isinstance(other, Variable):
            return self.value < other.value
        return self.value < other
    
    def __le__(self, other: Union[Variable, int, float]) -> bool:
        """Menor ou igual (<=)."""
        if isinstance(other, Variable):
            return self.value <= other.value
        return self.value <= other
    
    def __gt__(self, other: Union[Variable, int, float]) -> bool:
        """Maior que (>)."""
        if isinstance(other, Variable):
            return self.value > other.value
        return self.value > other
    
    def __ge__(self, other: Union[Variable, int, float]) -> bool:
        """Maior ou igual (>=)."""
        if isinstance(other, Variable):
            return self.value >= other.value
        return self.value >= other
    
    def __ne__(self, other: Union[Variable, int, float]) -> bool:
        """Diferente (!=)."""
        return not self.__eq__(other)



