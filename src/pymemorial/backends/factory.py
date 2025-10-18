"""
Factory para criação de backends estruturais.
"""
from typing import Optional
from .base import StructuralBackend


class BackendFactory:
    """Factory para backends estruturais."""
    
    @staticmethod
    def create(backend_name: str, **kwargs) -> StructuralBackend:
        """
        Cria instância de backend.
        
        Args:
            backend_name: nome do backend ('pynite', 'opensees', 'anastruct')
            **kwargs: argumentos específicos do backend
        
        Returns:
            Instância do backend
        
        Raises:
            ValueError: se backend não for suportado
            ImportError: se dependência do backend não estiver instalada
        """
        backend_name = backend_name.lower()
        
        if backend_name == "pynite":
            from .pynite import PyniteBackend  # ✅ CORRIGIDO - minúsculo
            return PyniteBackend(**kwargs)
        
        elif backend_name == "opensees":
            from .opensees import OpenSeesBackend
            return OpenSeesBackend(**kwargs)
        
        elif backend_name == "anastruct":
            from .anastruct import AnaStructBackend
            return AnaStructBackend(**kwargs)
        
        else:
            raise ValueError(
                f"Backend '{backend_name}' não suportado. "
                f"Opções: 'pynite', 'opensees', 'anastruct'"
            )
    
    @staticmethod
    def available_backends() -> list[str]:
        """
        Lista backends disponíveis (instalados).
        
        Returns:
            Lista de nomes de backends
        """
        available = []
        
        # ✅ Verificar Pynite (biblioteca instalada com P maiúsculo)
        try:
            import Pynite  # ✅ Mantém P maiúsculo - nome da biblioteca
            available.append("pynite")
        except ImportError:
            pass
        
        # Verificar OpenSeesPy
        try:
            import openseespy.opensees
            available.append("opensees")
        except ImportError:
            pass
        
        # Verificar Anastruct
        try:
            import anastruct
            available.append("anastruct")
        except ImportError:
            pass
        
        return available
