"""Testes do Pynite backend."""
import pytest
from pymemorial.backends.factory import BackendFactory


@pytest.mark.skipif(
    "pynite" not in BackendFactory.available_backends(),
    reason="PyNite não disponível"
)
def test_pynite_simple_beam():
    """Testa viga biapoiada com carga no meio."""
    backend = BackendFactory.create("pynite")
    
    # ✅ CORREÇÃO: Viga com 3 nós - carga NO MEIO, não no apoio
    backend.add_node("N1", 0, 0, 0)      # Apoio esquerdo
    backend.add_node("N2", 3, 0, 0)      # Meio do vão (carga aqui)
    backend.add_node("N3", 6, 0, 0)      # Apoio direito
    
    # Dois elementos
    backend.add_member("M1", "N1", "N2", "IPE200", "steel")
    backend.add_member("M2", "N2", "N3", "IPE200", "steel")
    
    # Apoios SOMENTE nos extremos (N1 e N3)
    backend.add_support("N1", dx=True, dy=True, dz=True, rx=True, rz=True)
    backend.add_support("N3", dy=True, dz=True, rx=True, rz=True)
    
    # Carga no meio do vão (N2)
    backend.add_load("nodal", node_id="N2", direction="Y", magnitude=-10000)
    
    # Analisar
    assert backend.analyze() is True
    
    # Verificar deslocamentos
    disp_n1 = backend.get_displacements("N1")
    disp_n2 = backend.get_displacements("N2")
    disp_n3 = backend.get_displacements("N3")
    
    # Apoios devem estar fixos
    assert abs(disp_n1["dy"]) < 1e-6, "N1 deve ter deslocamento zero"
    assert abs(disp_n3["dy"]) < 1e-6, "N3 deve ter deslocamento zero"
    
    # Meio do vão deve ter deslocamento negativo
    assert disp_n2["dy"] < -1e-6, f"N2 deve defletir (dy < 0), obtido {disp_n2['dy']}"
    
    # Verificar esforços
    forces_m1 = backend.get_member_forces("M1")
    assert abs(forces_m1["M"]) > 0, "Deve ter momento fletor"


@pytest.mark.skipif(
    "pynite" not in BackendFactory.available_backends(),
    reason="PyNite não disponível"
)
def test_pynite_cantilever():
    """Testa viga em balanço (cantilever)."""
    backend = BackendFactory.create("pynite")
    
    # Viga engastada
    backend.add_node("N1", 0, 0, 0)
    backend.add_node("N2", 3, 0, 0)
    
    backend.add_member("M1", "N1", "N2")
    
    # Engaste completo em N1
    backend.add_support("N1", dx=True, dy=True, dz=True, rx=True, ry=True, rz=True)
    
    # Carga na extremidade livre (N2)
    backend.add_load("nodal", node_id="N2", direction="Y", magnitude=-5000)
    
    assert backend.analyze() is True
    
    # Verificar
    disp_n1 = backend.get_displacements("N1")
    disp_n2 = backend.get_displacements("N2")
    
    assert abs(disp_n1["dy"]) < 1e-6, "Engaste deve estar fixo"
    assert disp_n2["dy"] < -1e-6, f"Extremidade livre deve defletir, obtido {disp_n2['dy']}"
    
    forces = backend.get_member_forces("M1")
    # ✅ max_moment retorna valor absoluto, não considera sinal
    assert abs(forces["M"]) > 1e-3, f"Deve ter momento fletor significativo, obtido {forces['M']}"


@pytest.mark.skipif(
    "pynite" not in BackendFactory.available_backends(),
    reason="PyNite não disponível"
)
def test_pynite_initialization():
    """Testa inicialização básica."""
    backend = BackendFactory.create("pynite")
    
    assert backend.name == "Pynite"
    assert backend.model is not None
    assert len(backend.nodes) == 0
    assert len(backend.members) == 0


@pytest.mark.skipif(
    "pynite" not in BackendFactory.available_backends(),
    reason="PyNite não disponível"
)
def test_pynite_distributed_load():
    """Testa viga com carga distribuída (simulada com múltiplas cargas pontuais)."""
    backend = BackendFactory.create("pynite")
    
    # Viga de 10m com 5 nós
    backend.add_node("N1", 0, 0, 0)
    backend.add_node("N2", 2.5, 0, 0)
    backend.add_node("N3", 5.0, 0, 0)
    backend.add_node("N4", 7.5, 0, 0)
    backend.add_node("N5", 10, 0, 0)
    
    # Quatro elementos
    backend.add_member("M1", "N1", "N2", "IPE200", "steel")
    backend.add_member("M2", "N2", "N3", "IPE200", "steel")
    backend.add_member("M3", "N3", "N4", "IPE200", "steel")
    backend.add_member("M4", "N4", "N5", "IPE200", "steel")
    
    # Apoios nos extremos
    backend.add_support("N1", dx=True, dy=True, dz=True, rx=True, rz=True)
    backend.add_support("N5", dy=True, dz=True, rx=True, rz=True)
    
    # Cargas nos nós intermediários (simulando carga distribuída)
    for node_id in ["N2", "N3", "N4"]:
        backend.add_load("nodal", node_id=node_id, direction="Y", magnitude=-5000)
    
    assert backend.analyze() is True
    
    # Verificar que há deflexão
    disp_n3 = backend.get_displacements("N3")
    assert disp_n3["dy"] < -1e-6, "Meio do vão deve ter deflexão significativa"