"""Testes da interface base de backends."""
import pytest
from pymemorial.backends.base import StructuralBackend, Node, Member


class MockBackend(StructuralBackend):
    """Backend mock para testes."""
    
    def add_node(self, node_id: str, x: float, y: float, z: float = 0.0):
        self.nodes[node_id] = Node(id=node_id, x=x, y=y, z=z)
    
    def add_member(self, member_id: str, node_i: str, node_j: str, section: str, material: str):
        self.members[member_id] = Member(
            id=member_id, node_i=node_i, node_j=node_j, section=section, material=material
        )
    
    def add_support(self, node_id: str, **restrictions):
        pass
    
    def add_load(self, load_type: str, **kwargs):
        pass
    
    def analyze(self) -> bool:
        self._is_analyzed = True
        return True
    
    def get_displacements(self, node_id: str):
        return {"dx": 0.0, "dy": 0.0, "dz": 0.0}
    
    def get_member_forces(self, member_id: str):
        return {"N": 0.0, "V": 0.0, "M": 0.0}


def test_backend_creation():
    """Testa criação de backend."""
    backend = MockBackend(name="Test")
    assert backend.name == "Test"
    assert len(backend.nodes) == 0


def test_add_node():
    """Testa adição de nó."""
    backend = MockBackend()
    backend.add_node("N1", x=0.0, y=0.0, z=0.0)
    
    assert "N1" in backend.nodes
    assert backend.nodes["N1"].x == 0.0


def test_add_member():
    """Testa adição de elemento."""
    backend = MockBackend()
    backend.add_node("N1", 0, 0)
    backend.add_node("N2", 5, 0)
    backend.add_member("M1", "N1", "N2", "IPE200", "steel")
    
    assert "M1" in backend.members
    assert backend.members["M1"].node_i == "N1"


def test_is_analyzed():
    """Testa flag de análise."""
    backend = MockBackend()
    assert backend.is_analyzed() is False
    
    backend.analyze()
    assert backend.is_analyzed() is True


def test_get_model_summary():
    """Testa resumo do modelo."""
    backend = MockBackend()
    backend.add_node("N1", 0, 0)
    backend.add_node("N2", 5, 0)
    backend.add_member("M1", "N1", "N2", "sec", "mat")
    
    summary = backend.get_model_summary()
    assert summary["nodes"] == 2
    assert summary["members"] == 1
