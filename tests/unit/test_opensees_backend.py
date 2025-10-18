"""Testes do OpenSees backend."""
import pytest
from pymemorial.backends.factory import BackendFactory


@pytest.mark.skipif(
    "opensees" not in BackendFactory.available_backends(),
    reason="OpenSees não disponível"
)
def test_opensees_simple_beam():
    """Testa viga biapoiada com carga no meio."""
    backend = BackendFactory.create("opensees")
    
    # Viga com 3 nós - carga no meio
    backend.add_node("N1", 0, 0, 0)
    backend.add_node("N2", 3, 0, 0)
    backend.add_node("N3", 6, 0, 0)
    
    # Dois elementos
    backend.add_member("M1", "N1", "N2", "IPE200", "steel")
    backend.add_member("M2", "N2", "N3", "IPE200", "steel")
    
    # Apoios
    backend.add_support("N1", dx=True, dy=True, dz=True, rx=True, rz=True)
    backend.add_support("N3", dy=True, dz=True, rx=True, rz=True)
    
    # Carga no meio
    backend.add_load("nodal", node_id="N2", direction="Y", magnitude=-10000)
    
    # Analisar
    assert backend.analyze() is True
    
    # Verificar deslocamentos
    disp_n1 = backend.get_displacements("N1")
    disp_n2 = backend.get_displacements("N2")
    disp_n3 = backend.get_displacements("N3")
    
    # Apoios fixos
    assert abs(disp_n1["dy"]) < 1e-6
    assert abs(disp_n3["dy"]) < 1e-6
    
    # Meio do vão deflexionado
    assert disp_n2["dy"] < -1e-6, f"N2 deve defletir, obtido {disp_n2['dy']}"
    
    # Verificar esforços
    forces_m1 = backend.get_member_forces("M1")
    assert forces_m1["Mz"] > 0, "Deve ter momento fletor"


@pytest.mark.skipif(
    "opensees" not in BackendFactory.available_backends(),
    reason="OpenSees não disponível"
)
def test_opensees_cantilever():
    """Testa viga em balanço."""
    backend = BackendFactory.create("opensees")
    
    backend.add_node("N1", 0, 0, 0)
    backend.add_node("N2", 3, 0, 0)
    
    backend.add_member("M1", "N1", "N2")
    
    # Engaste completo
    backend.add_support("N1", dx=True, dy=True, dz=True, rx=True, ry=True, rz=True)
    
    # Carga na extremidade
    backend.add_load("nodal", node_id="N2", direction="Y", magnitude=-5000)
    
    assert backend.analyze() is True
    
    disp_n2 = backend.get_displacements("N2")
    assert disp_n2["dy"] < -1e-6


@pytest.mark.skipif(
    "opensees" not in BackendFactory.available_backends(),
    reason="OpenSees não disponível"
)
def test_opensees_initialization():
    """Testa inicialização."""
    backend = BackendFactory.create("opensees")
    
    assert backend.name == "OpenSees"
    assert len(backend.nodes) == 0
    assert len(backend.members) == 0
