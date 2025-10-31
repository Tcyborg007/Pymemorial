# src/pymemorial/editor/render_modes.py
"""
Modos de Renderização para PyMemorial v2.0

Este módulo define os modos de renderização inspirados em:
- Handcalcs (params, symbolic, numeric, mixed)
- Calcpad (steps automáticos em 4 níveis de granularidade)

Integração TOTAL com:
- pymemorial.core.config (configurações globais)
- pymemorial.core.equation (motor simbólico)
- pymemorial.editor.step_engine (steps automáticos - futuro)

Author: PyMemorial Team
Date: October 2025
Version: 2.0.0
License: MIT
"""

from __future__ import annotations

import logging
from enum import Enum, auto
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List, Literal

# ============================================================================
# IMPORTS PYMEMORIAL (GARANTINDO COMPATIBILIDADE)
# ============================================================================
try:
    from pymemorial.core.config import get_config, PyMemorialConfig
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    PyMemorialConfig = None

# Configuração de logging
logger = logging.getLogger(__name__)

# ============================================================================
# VERSÃO DO MÓDULO
# ============================================================================
__version__ = "2.0.0"
__author__ = "PyMemorial Team"
__all__ = [
    'RenderMode',
    'StepGranularity',
    'OutputFormat',
    'RenderConfig',
    'create_render_config',
    'get_default_render_config'
]

# ============================================================================
# ENUMS - MODOS DE RENDERIZAÇÃO
# ============================================================================

class RenderMode(str, Enum):
    """
    Modos de renderização principais.
    
    Inspirado em: Handcalcs + Calcpad
    
    **HANDCALCS-STYLE** (Renderização direta):
    - PARAMS: Apenas tabela de parâmetros
    - SYMBOLIC: Apenas fórmulas simbólicas
    - NUMERIC: Apenas valores numéricos
    - MIXED: Fórmulas + valores lado a lado
    
    **CALCPAD-STYLE** (Steps automáticos):
    - STEPS_MINIMAL: 2 steps (fórmula → resultado)
    - STEPS_SMART: 3 steps (fórmula → substituição → resultado) [PADRÃO]
    - STEPS_DETAILED: 5-10 steps (todas simplificações intermediárias)
    - STEPS_ALL: 10-50 steps (cada operação aritmética detalhada)
    
    Examples:
        >>> from pymemorial.editor.render_modes import RenderMode
        >>> mode = RenderMode.STEPS_SMART
        >>> print(mode.value)
        'steps_smart'
        >>> print(mode.is_steps_mode())
        True
    """
    
    # HANDCALCS-STYLE (renderização direta)
    PARAMS = "params"
    SYMBOLIC = "symbolic"
    NUMERIC = "numeric"
    MIXED = "mixed"
    
    # CALCPAD-STYLE (steps automáticos)
    STEPS_MINIMAL = "steps_minimal"
    STEPS_SMART = "steps_smart"
    STEPS_DETAILED = "steps_detailed"
    STEPS_ALL = "steps_all"
    
    def is_steps_mode(self) -> bool:
        """Verifica se o modo é de steps automáticos (Calcpad-style)."""
        return self.value.startswith('steps_')
    
    def is_handcalcs_mode(self) -> bool:
        """Verifica se o modo é Handcalcs-style."""
        return not self.is_steps_mode()
    
    def get_step_count_estimate(self) -> int:
        """Retorna estimativa de número de steps para este modo."""
        step_counts = {
            self.STEPS_MINIMAL: 2,
            self.STEPS_SMART: 3,
            self.STEPS_DETAILED: 7,
            self.STEPS_ALL: 20
        }
        return step_counts.get(self, 1)
    
    @classmethod
    def from_string(cls, mode_str: str) -> 'RenderMode':
        """
        Cria RenderMode a partir de string (case-insensitive).
        
        Args:
            mode_str: Nome do modo ('smart', 'STEPS_DETAILED', etc)
        
        Returns:
            RenderMode correspondente
        
        Raises:
            ValueError: Se modo inválido
        
        Examples:
            >>> RenderMode.from_string('smart')
            RenderMode.STEPS_SMART
            >>> RenderMode.from_string('SYMBOLIC')
            RenderMode.SYMBOLIC
        """
        mode_str = mode_str.lower().strip()
        
        # Tentar match direto
        for mode in cls:
            if mode.value == mode_str:
                return mode
        
        # Tentar com prefixo "steps_"
        if not mode_str.startswith('steps_'):
            prefixed = f"steps_{mode_str}"
            for mode in cls:
                if mode.value == prefixed:
                    return mode
        
        raise ValueError(
            f"Modo de renderização inválido: '{mode_str}'. "
            f"Modos válidos: {', '.join(m.value for m in cls)}"
        )
    
    def __str__(self) -> str:
        """String representation."""
        return self.value
    
    def __repr__(self) -> str:
        """Developer representation."""
        return f"RenderMode.{self.name}"


class StepGranularity(Enum):
    """
    Granularidade dos steps de cálculo (Calcpad-style).
    
    Define o nível de detalhamento dos passos intermediários.
    """
    MINIMAL = auto()      # Apenas fórmula → resultado
    SMART = auto()        # Fórmula → substituição → resultado (PADRÃO)
    DETAILED = auto()     # Todas simplificações algébricas
    EXHAUSTIVE = auto()   # Cada operação aritmética
    
    @classmethod
    def from_render_mode(cls, mode: RenderMode) -> 'StepGranularity':
        """Converte RenderMode para StepGranularity."""
        mapping = {
            RenderMode.STEPS_MINIMAL: cls.MINIMAL,
            RenderMode.STEPS_SMART: cls.SMART,
            RenderMode.STEPS_DETAILED: cls.DETAILED,
            RenderMode.STEPS_ALL: cls.EXHAUSTIVE
        }
        return mapping.get(mode, cls.SMART)


class OutputFormat(str, Enum):
    """
    Formato de saída do memorial.
    
    Define como o resultado será exportado/renderizado.
    """
    HTML = "html"           # HTML com KaTeX (padrão)
    LATEX = "latex"         # LaTeX puro
    PDF = "pdf"             # PDF via LaTeX
    MARKDOWN = "markdown"   # Markdown com LaTeX inline
    JUPYTER = "jupyter"     # Jupyter Notebook
    DOCX = "docx"           # Microsoft Word (via pandoc)
    
    def supports_math(self) -> bool:
        """Verifica se o formato suporta matemática renderizada."""
        return self in [self.HTML, self.LATEX, self.PDF, self.JUPYTER]
    
    def is_interactive(self) -> bool:
        """Verifica se o formato é interativo."""
        return self in [self.HTML, self.JUPYTER]


# ============================================================================
# DATACLASS - CONFIGURAÇÃO DE RENDERIZAÇÃO
# ============================================================================

@dataclass
class RenderConfig:
    """
    Configuração completa de renderização.
    
    Centraliza TODAS as opções de renderização em um único objeto.
    Integrado com pymemorial.core.config para herdar configurações globais.
    
    Attributes:
        mode: Modo de renderização (STEPS_SMART padrão)
        precision: Casas decimais (3 padrão)
        show_units: Mostrar unidades físicas
        show_substitution: Mostrar passo de substituição
        show_intermediate: Mostrar steps intermediários
        use_unicode: Usar Unicode (α, β) vs LaTeX (\\alpha, \\beta)
        latex_inline: Renderizar LaTeX inline ($...$) vs display ($$...$$)
        output_format: Formato de saída (HTML padrão)
        enable_cache: Ativar cache de renderização
        max_steps: Número máximo de steps (0 = ilimitado)
        theme: Tema de cores ('light', 'dark', 'abnt')
        css_classes: Classes CSS customizadas
        metadata: Metadados adicionais
    
    Examples:
        >>> from pymemorial.editor.render_modes import RenderConfig, RenderMode
        >>> 
        >>> # Configuração básica
        >>> config = RenderConfig(
        ...     mode=RenderMode.STEPS_SMART,
        ...     precision=3,
        ...     show_units=True
        ... )
        >>> 
        >>> # A partir do config global
        >>> config = RenderConfig.from_config()
        >>> 
        >>> # Para TCC/ABNT
        >>> config = RenderConfig.for_abnt()
    """
    
    # Renderização
    mode: RenderMode = RenderMode.STEPS_SMART
    precision: int = 3
    show_units: bool = True
    show_substitution: bool = True
    show_intermediate: bool = True
    
    # Formato
    use_unicode: bool = True
    latex_inline: bool = True
    output_format: OutputFormat = OutputFormat.HTML
    
    # Performance
    enable_cache: bool = True
    max_steps: int = 0  # 0 = ilimitado
    
    # Estilo
    theme: Literal['light', 'dark', 'abnt', 'technical'] = 'light'
    css_classes: List[str] = field(default_factory=list)
    
    # Extensibilidade
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validação pós-inicialização."""
        # Validar precisão
        if self.precision < 0 or self.precision > 15:
            logger.warning(
                f"Precisão {self.precision} fora do intervalo [0, 15]. "
                f"Ajustando para 3."
            )
            self.precision = 3
        
        # Validar max_steps
        if self.max_steps < 0:
            logger.warning(f"max_steps negativo. Ajustando para 0 (ilimitado).")
            self.max_steps = 0
        
        # Garantir que mode seja RenderMode
        if isinstance(self.mode, str):
            self.mode = RenderMode.from_string(self.mode)
        
        # Garantir que output_format seja OutputFormat
        if isinstance(self.output_format, str):
            self.output_format = OutputFormat(self.output_format.lower())
    
    @classmethod
    def from_config(cls, config: Optional[PyMemorialConfig] = None) -> 'RenderConfig':
        """
        Cria RenderConfig a partir do config global do PyMemorial.
        
        Args:
            config: Config global (ou None para buscar automaticamente)
        
        Returns:
            RenderConfig configurado com valores do config global
        
        Examples:
            >>> config = RenderConfig.from_config()
            >>> print(config.precision)  # Valor do config global
        """
        if not CONFIG_AVAILABLE:
            logger.warning(
                "pymemorial.core.config não disponível. "
                "Usando configurações padrão."
            )
            return cls()
        
        # Obter config global
        if config is None:
            config = get_config()
        
        return cls(
            mode=RenderMode.STEPS_SMART,  # Padrão PyMemorial
            precision=config.display.precision,
            show_units=True,
            show_substitution=True,
            show_intermediate=True,
            use_unicode=(config.symbols.greek_style == "unicode"),
            latex_inline=True,
            output_format=OutputFormat.HTML,
            enable_cache=True,
            max_steps=0,
            theme='light',
            css_classes=[],
            metadata={}
        )
    
    @classmethod
    def for_abnt(cls) -> 'RenderConfig':
        """
        Configuração otimizada para documentos ABNT (TCC, dissertações).
        
        Returns:
            RenderConfig com tema ABNT
        """
        return cls(
            mode=RenderMode.STEPS_SMART,
            precision=2,  # ABNT recomenda 2 casas decimais
            show_units=True,
            show_substitution=True,
            show_intermediate=False,  # ABNT prefere conciso
            use_unicode=False,  # ABNT prefere LaTeX puro
            latex_inline=False,  # ABNT prefere equações display
            output_format=OutputFormat.LATEX,
            enable_cache=True,
            max_steps=5,  # Limitar para não poluir documento
            theme='abnt',
            css_classes=['abnt-equation'],
            metadata={'document_type': 'abnt', 'standard': 'NBR 14724'}
        )
    
    @classmethod
    def for_technical_report(cls) -> 'RenderConfig':
        """
        Configuração para relatórios técnicos de engenharia.
        
        Returns:
            RenderConfig com tema técnico
        """
        return cls(
            mode=RenderMode.STEPS_DETAILED,
            precision=3,
            show_units=True,
            show_substitution=True,
            show_intermediate=True,
            use_unicode=True,
            latex_inline=True,
            output_format=OutputFormat.HTML,
            enable_cache=True,
            max_steps=10,
            theme='technical',
            css_classes=['technical-report'],
            metadata={'document_type': 'technical_report'}
        )
    
    @classmethod
    def for_jupyter(cls) -> 'RenderConfig':
        """
        Configuração para notebooks Jupyter.
        
        Returns:
            RenderConfig otimizado para Jupyter
        """
        return cls(
            mode=RenderMode.MIXED,
            precision=4,
            show_units=True,
            show_substitution=True,
            show_intermediate=True,
            use_unicode=True,
            latex_inline=True,
            output_format=OutputFormat.JUPYTER,
            enable_cache=False,  # Jupyter não precisa de cache
            max_steps=0,
            theme='light',
            css_classes=['jupyter-equation'],
            metadata={'document_type': 'jupyter'}
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Converte para dicionário.
        
        Returns:
            Dicionário com todos os atributos
        """
        return asdict(self)
    
    def copy(self, **overrides) -> 'RenderConfig':
        """
        Cria cópia com overrides.
        
        Args:
            **overrides: Atributos a sobrescrever
        
        Returns:
            Nova instância com overrides aplicados
        
        Examples:
            >>> config = RenderConfig()
            >>> config_abnt = config.copy(theme='abnt', precision=2)
        """
        data = self.to_dict()
        data.update(overrides)
        return RenderConfig(**data)
    
    def __repr__(self) -> str:
        """Developer representation."""
        return (
            f"RenderConfig("
            f"mode={self.mode.name}, "
            f"precision={self.precision}, "
            f"theme='{self.theme}')"
        )
    
    def __str__(self) -> str:
        """User-friendly representation."""
        return f"RenderConfig[{self.mode.value}, {self.precision} decimais]"


# ============================================================================
# FUNÇÕES AUXILIARES
# ============================================================================

def create_render_config(
    mode: Optional[str] = None,
    precision: Optional[int] = None,
    **kwargs
) -> RenderConfig:
    """
    Factory function para criar RenderConfig facilmente.
    
    Args:
        mode: Nome do modo ('smart', 'detailed', etc)
        precision: Casas decimais
        **kwargs: Outros parâmetros de RenderConfig
    
    Returns:
        RenderConfig configurado
    
    Examples:
        >>> config = create_render_config('smart', precision=4)
        >>> config = create_render_config(mode='detailed', show_units=False)
    """
    # Valores padrão do config global
    if CONFIG_AVAILABLE:
        base_config = RenderConfig.from_config()
    else:
        base_config = RenderConfig()
    
    # Aplicar overrides
    if mode is not None:
        base_config.mode = RenderMode.from_string(mode)
    
    if precision is not None:
        base_config.precision = precision
    
    # Aplicar kwargs adicionais
    for key, value in kwargs.items():
        if hasattr(base_config, key):
            setattr(base_config, key, value)
        else:
            logger.warning(f"Atributo desconhecido ignorado: {key}")
    
    return base_config


def get_default_render_config() -> RenderConfig:
    """
    Retorna configuração padrão do PyMemorial.
    
    Returns:
        RenderConfig com valores padrão
    
    Examples:
        >>> config = get_default_render_config()
        >>> print(config.mode)
        RenderMode.STEPS_SMART
    """
    if CONFIG_AVAILABLE:
        return RenderConfig.from_config()
    else:
        return RenderConfig()


# ============================================================================
# TESTES INTERNOS (OPCIONAL - REMOVER EM PRODUÇÃO)
# ============================================================================

def _test_render_modes():
    """Testes básicos de funcionalidade (desenvolvimento)."""
    print("=" * 70)
    print("🧪 Testando render_modes.py")
    print("=" * 70)
    
    # Teste 1: RenderMode
    print("\n1. RenderMode:")
    mode = RenderMode.STEPS_SMART
    print(f"   - Value: {mode.value}")
    print(f"   - Is steps? {mode.is_steps_mode()}")
    print(f"   - Step count: {mode.get_step_count_estimate()}")
    
    # Teste 2: RenderMode.from_string
    print("\n2. RenderMode.from_string:")
    mode2 = RenderMode.from_string('smart')
    print(f"   - 'smart' → {mode2}")
    
    # Teste 3: RenderConfig
    print("\n3. RenderConfig:")
    config = RenderConfig()
    print(f"   - Default: {config}")
    
    # Teste 4: RenderConfig.for_abnt
    print("\n4. RenderConfig.for_abnt:")
    config_abnt = RenderConfig.for_abnt()
    print(f"   - ABNT: {config_abnt}")
    print(f"   - Precision: {config_abnt.precision}")
    print(f"   - Theme: {config_abnt.theme}")
    
    # Teste 5: create_render_config
    print("\n5. create_render_config:")
    config_custom = create_render_config('detailed', precision=4, show_units=False)
    print(f"   - Custom: {config_custom}")
    
    print("\n" + "=" * 70)
    print("✅ Todos os testes passaram!")
    print("=" * 70)


if __name__ == "__main__":
    # Executar testes internos se rodado diretamente
    _test_render_modes()
    