"""
Factory aprimorada para criação de backends de análise estrutural.

Suporta detecção automática de dependências, auto-seleção inteligente de backend,
e configurações por norma técnica (ex: NBR 6118, AISC). Mantém compatibilidade
total com APIs existentes.

Exemplo de uso:
    factory = BackendFactory()
    backend = factory.create('pynite', norm='nbr')  # Auto-configura load cases
    available = factory.available_backends()  # ['pynite']
"""

import logging
from typing import Dict, List, Any, Optional, Literal, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

# Configurar logger (nível INFO por default; ajuste via logging.basicConfig)
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

# Importações condicionais para dependências (evita erros em ambientes limpos)
PYNITE_AVAILABLE: bool = False
OPENSEES_AVAILABLE: bool = False
ANASTRUCT_AVAILABLE: bool = False

try:
    import Pynite  # Biblioteca oficial com P maiúsculo
    PYNITE_AVAILABLE = True
    _logger.debug("PyNite detectado e disponível.")
except ImportError:
    _logger.warning("PyNite não instalado. Instale via: pip install Pynite")

try:
    import openseespy.opensees as ops
    OPENSEES_AVAILABLE = True
    _logger.debug("OpenSeesPy detectado e disponível.")
except ImportError:
    _logger.warning("OpenSeesPy não instalado. Instale via: pip install openseespy")

try:
    import anastruct
    ANASTRUCT_AVAILABLE = True
    _logger.debug("Anastruct detectado e disponível.")
except ImportError:
    _logger.warning("Anastruct não instalado. Instale via: pip install anastruct")

# Imports locais (assumindo que existem; registry trata falhas)
try:
    from .pynite import PyniteBackend
    from .opensees import OpenSeesBackend
    from .anastruct import AnaStructBackend  # Nome corrigido conforme seu arquivo
except ImportError as e:
    _logger.warning(f"Erro ao importar backends locais: {e}. Usando registry fallback.")
    PyniteBackend = OpenSeesBackend = AnaStructBackend = None  # Placeholder para registry


class NormType(Enum):
    """Normas técnicas suportadas para auto-configuração de load cases."""
    NBR = "nbr"      # NBR 6118/6123 (Brasil)
    AISC = "aisc"    # AISC 360 (EUA)
    EUROCODE = "ec"  # Eurocode 2/3 (Europa)
    CUSTOM = "custom"


@dataclass
class BackendConfig:
    """Configurações globais para backends (ex: load cases, unidades)."""
    norm: NormType = NormType.NBR
    units_system: str = "SI"  # 'SI' ou 'Imperial'
    load_cases: Dict[str, Dict[str, Any]] = None  # Ex: {'DEAD': {'factor': 1.4}}
    max_nodes: Optional[int] = None  # Limite para otimização (ex: Anastruct para <100 nós)

    def __post_init__(self):
        if self.load_cases is None:
            self.load_cases = self._default_load_cases()

    def _default_load_cases(self) -> Dict[str, Dict[str, Any]]:
        """Load cases padrão por norma."""
        defaults = {
            NormType.NBR: {
                'PERMANENTE': {'factor': 1.4, 'description': 'Cargas permanentes'},
                'ACIDENTAL': {'factor': 1.6, 'description': 'Cargas acidentais'},
                'COMBINACAO': {'factor': 1.0, 'description': 'Combinação ULS'},
            },
            NormType.AISC: {
                'DEAD': {'factor': 1.2, 'description': 'Dead load'},
                'LIVE': {'factor': 1.6, 'description': 'Live load'},
                'WIND': {'factor': 0.6, 'description': 'Wind load'},
            },
            NormType.EUROCODE: {
                'G': {'factor': 1.35, 'description': 'Permanent actions'},
                'Q': {'factor': 1.5, 'description': 'Variable actions'},
            },
            NormType.CUSTOM: {},
        }
        return defaults[self.norm.value]


class BackendStatus(Enum):
    """Status de um backend para relatórios."""
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable (install required)"
    PLANNED = "planned (future release)"
    ERROR = "error (import failed)"


class BackendRegistry(ABC):
    """Registro abstrato para backends (facilita extensões)."""
    @abstractmethod
    def create_instance(self, **kwargs) -> 'StructuralBackend':
        pass

    @abstractmethod
    def get_status(self) -> BackendStatus:
        pass

    @abstractmethod
    def get_features(self) -> List[str]:
        """Features do backend (ex: ['2D', 'linear'])."""
        pass


# Registry concreto para cada backend (padrão Strategy)
class PyniteRegistry(BackendRegistry):
    def create_instance(self, **kwargs) -> 'StructuralBackend':
        if PyniteBackend is None:
            raise ImportError("PyniteBackend não disponível.")
        config = BackendConfig(**kwargs.get('config', {}))
        return PyniteBackend(config=config, **{k: v for k, v in kwargs.items() if k != 'config'})

    def get_status(self) -> BackendStatus:
        return BackendStatus.AVAILABLE if PYNITE_AVAILABLE else BackendStatus.UNAVAILABLE

    def get_features(self) -> List[str]:
        return ['3D', 'non-linear', 'steel', 'concrete', 'dynamic']


class OpenSeesRegistry(BackendRegistry):
    def create_instance(self, **kwargs) -> 'StructuralBackend':
        if OpenSeesBackend is None:
            raise ImportError("OpenSeesBackend não disponível.")
        config = BackendConfig(**kwargs.get('config', {}))
        return OpenSeesBackend(config=config, **{k: v for k, v in kwargs.items() if k != 'config'})

    def get_status(self) -> BackendStatus:
        return BackendStatus.AVAILABLE if OPENSEES_AVAILABLE else BackendStatus.UNAVAILABLE

    def get_features(self) -> List[str]:
        return ['2D/3D', 'non-linear', 'earthquake', 'fiber-sections']


class AnastructRegistry(BackendRegistry):
    def create_instance(self, **kwargs) -> 'StructuralBackend':
        if AnaStructBackend is None:
            raise NotImplementedError(
                "AnaStructBackend planejado para PHASE 8. Use 'pynite' para análises 2D por enquanto."
            )
        config = BackendConfig(**kwargs.get('config', {}))
        return AnaStructBackend(config=config, **{k: v for k, v in kwargs.items() if k != 'config'})

    def get_status(self) -> BackendStatus:
        if AnaStructBackend is None:
            return BackendStatus.PLANNED
        return BackendStatus.AVAILABLE if ANASTRUCT_AVAILABLE else BackendStatus.UNAVAILABLE

    def get_features(self) -> List[str]:
        return ['2D', 'linear', 'trusses', 'frames', 'fast']


class BackendFactory:
    """
    Factory aprimorada para backends estruturais com registry dinâmico.
    
    Mantém compatibilidade: create() e available_backends() funcionam como antes.
    Novidades: auto_select(), all_backends(), suporte a config/norm.
    """

    # Registry central (fácil de estender: adicione novos registries aqui)
    _REGISTRY: Dict[str, BackendRegistry] = {
        'pynite': PyniteRegistry(),
        'opensees': OpenSeesRegistry(),
        'anastruct': AnastructRegistry(),
    }

    @staticmethod
    def create(backend_name: str, **kwargs) -> 'StructuralBackend':
        """
        Cria instância de backend (compatível com versão anterior).
        
        Args:
            backend_name: Nome do backend ('pynite', 'opensees', 'anastruct').
            **kwargs: Argumentos (ex: config={'norm': 'nbr'}).
        
        Returns:
            Instância de StructuralBackend.
        
        Raises:
            ValueError: Backend desconhecido.
            ImportError/NotImplementedError: Dependência ausente ou planejado.
        
        Exemplo:
            backend = BackendFactory.create('pynite', config={'norm': 'aisc'})
        """
        backend_name = backend_name.lower().strip()
        _logger.info(f"Tentando criar backend: {backend_name}")

        if backend_name not in BackendFactory._REGISTRY:
            raise ValueError(
                f"Backend '{backend_name}' não suportado. "
                f"Opções: {', '.join(BackendFactory._REGISTRY.keys())}"
            )

        registry = BackendFactory._REGISTRY[backend_name]
        status = registry.get_status()
        if status == BackendStatus.UNAVAILABLE:
            raise ImportError(f"Dependência de '{backend_name}' não instalada. Veja logs.")
        elif status == BackendStatus.PLANNED:
            raise NotImplementedError(f"'{backend_name}' planejado para futuro. Use alternativa.")

        try:
            instance = registry.create_instance(**kwargs)
            _logger.info(f"Backend '{backend_name}' criado com sucesso (features: {registry.get_features()}).")
            return instance
        except Exception as e:
            _logger.error(f"Erro ao criar '{backend_name}': {e}")
            raise

    @staticmethod
    def auto_select(analysis_type: str, **kwargs) -> 'StructuralBackend':
        """
        Auto-seleção inteligente de backend baseada no tipo de análise.
        
        Args:
            analysis_type: Tipo ('2D-linear', '3D-nonlinear', 'truss', etc.).
            **kwargs: Como em create().
        
        Returns:
            Backend otimizado para o tipo.
        
        Exemplo:
            backend = BackendFactory.auto_select('2D-linear')  # → Anastruct
        """
        _logger.info(f"Auto-selecionando backend para: {analysis_type}")
        mappings = {
            '2d-linear': 'anastruct',
            'truss': 'anastruct',
            '3d': 'pynite',
            'nonlinear': 'opensees',
            'earthquake': 'opensees',
            'default': 'pynite',  # Fallback
        }
        backend_name = mappings.get(analysis_type.lower().replace('-', ''), mappings['default'])
        return BackendFactory.create(backend_name, **kwargs)

    @staticmethod
    def available_backends() -> List[str]:
        """
        Lista backends disponíveis (instalados e implementados) — compatível com anterior.
        
        Returns:
            Lista de nomes (ex: ['pynite', 'opensees']).
        """
        available = [
            name for name, registry in BackendFactory._REGISTRY.items()
            if registry.get_status() == BackendStatus.AVAILABLE
        ]
        _logger.debug(f"Backends disponíveis: {available}")
        return available

    @staticmethod
    def all_backends() -> Dict[str, Dict[str, Any]]:
        """
        Todos os backends com status e features (novo método prático).
        
        Returns:
            Dict: {'pynite': {'status': 'available', 'features': ['3D', ...]}}.
        """
        result = {}
        for name, registry in BackendFactory._REGISTRY.items():
            result[name] = {
                'status': registry.get_status().value,
                'features': registry.get_features(),
            }
        _logger.info(f"Relatório de backends: {result}")
        return result

    @staticmethod
    def get_recommended(analysis_type: str) -> str:
        """
        Recomendação rápida de backend por tipo de análise.
        
        Args:
            analysis_type: Como em auto_select().
        
        Returns:
            Nome do backend recomendado.
        """
        mappings = {
            '2d-linear': 'anastruct',
            '3d': 'pynite',
            'nonlinear': 'opensees',
        }
        return mappings.get(analysis_type.lower(), 'pynite')


# Exportações para __init__.py (compatível)
__all__ = [
    'BackendFactory',
    'NormType',
    'BackendConfig',
    'available_backends',  # Legacy
]