"""
Interface abstrata para backends estruturais.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class Node:
    """Nó estrutural."""
    id: str
    x: float
    y: float
    z: float = 0.0


@dataclass
class Member:
    """Elemento estrutural (barra, viga, pilar)."""
    id: str
    node_i: str  # Nó inicial
    node_j: str  # Nó final
    section: str
    material: str


@dataclass
class Load:
    """Carregamento."""
    type: str  # 'nodal', 'distributed', 'point'
    target: str  # ID do nó ou elemento
    magnitude: float
    direction: str  # 'X', 'Y', 'Z', 'MX', 'MY', 'MZ'


@dataclass
class Support:
    """Apoio/restrição."""
    node_id: str
    dx: bool = False  # Restringir translação X
    dy: bool = False  # Restringir translação Y
    dz: bool = False  # Restringir translação Z
    rx: bool = False  # Restringir rotação X
    ry: bool = False  # Restringir rotação Y
    rz: bool = False  # Restringir rotação Z


class StructuralBackend(ABC):
    """
    Interface abstrata para backends de análise estrutural.
    
    Implementações concretas: PyNiteBackend, OpenSeesBackend, AnaStructBackend
    """
    
    def __init__(self, name: str = "Generic"):
        """
        Args:
            name: nome do backend
        """
        self.name = name
        self.nodes: Dict[str, Node] = {}
        self.members: Dict[str, Member] = {}
        self.loads: List[Load] = []
        self.supports: List[Support] = []
        self._is_analyzed = False
    
    @abstractmethod
    def add_node(self, node_id: str, x: float, y: float, z: float = 0.0):
        """Adiciona nó à estrutura."""
        pass
    
    @abstractmethod
    def add_member(
        self,
        member_id: str,
        node_i: str,
        node_j: str,
        section: str,
        material: str
    ):
        """Adiciona elemento estrutural."""
        pass
    
    @abstractmethod
    def add_support(self, node_id: str, **restrictions):
        """Adiciona apoio/restrição."""
        pass
    
    @abstractmethod
    def add_load(self, load_type: str, **kwargs):
        """Adiciona carregamento."""
        pass
    
    @abstractmethod
    def analyze(self) -> bool:
        """Executa análise estrutural."""
        pass
    
    @abstractmethod
    def get_displacements(self, node_id: str) -> Dict[str, float]:
        """Retorna deslocamentos de um nó."""
        pass
    
    @abstractmethod
    def get_member_forces(self, member_id: str) -> Dict[str, float]:
        """Retorna esforços internos de um elemento."""
        pass
    
    def is_analyzed(self) -> bool:
        """Verifica se análise já foi executada."""
        return self._is_analyzed
    
    def get_model_summary(self) -> Dict[str, int]:
        """Retorna resumo do modelo."""
        return {
            "nodes": len(self.nodes),
            "members": len(self.members),
            "loads": len(self.loads),
            "supports": len(self.supports),
        }
