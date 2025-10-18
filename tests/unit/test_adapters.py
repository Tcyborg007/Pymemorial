"""Testes dos adapters estruturais."""
import pytest
from pymemorial.backends.factory import BackendFactory
from pymemorial.backends.adapter import SimpleFrameAdapter


@pytest.mark.skipif(
    "pynite" not in BackendFactory.available_backends(),
    reason="Pynite não disponível"
)
def test_adapter_load_pynite():
    """Testa carregamento de modelo com Pynite."""
    backend = BackendFactory.create("pynite")
    adapter = SimpleFrameAdapter(backend)
    
    # Modelo como dicionário
    model = {
        'nodes': [
            {'id': 'N1', 'x': 0, 'y': 0, 'z': 0},
            {'id': 'N2', 'x': 3, 'y': 0, 'z': 0},
            {'id': 'N3', 'x': 6, 'y': 0, 'z': 0}
        ],
        'members': [
            {'id': 'M1', 'i': 'N1', 'j': 'N2', 'section': 'IPE200'},
            {'id': 'M2', 'i': 'N2', 'j': 'N3', 'section': 'IPE200'}
        ],
        'supports': [
            {'node': 'N1', 'dx': True, 'dy': True, 'dz': True, 'rx': True, 'rz': True},
            {'node': 'N3', 'dy': True, 'dz': True, 'rx': True, 'rz': True}
        ],
        'loads': [
            {'type': 'nodal', 'node': 'N2', 'direction': 'Y', 'magnitude': -10000}
        ]
    }
    
    # Carregar modelo
    assert adapter.load_from_dict(model) is True
    
    # Verificar que nós e elementos foram criados
    assert len(backend.nodes) == 3
    assert len(backend.members) == 2


@pytest.mark.skipif(
    "pynite" not in BackendFactory.available_backends(),
    reason="Pynite não disponível"
)
def test_adapter_analyze_pynite():
    """Testa análise completa com adapter Pynite."""
    backend = BackendFactory.create("pynite")
    adapter = SimpleFrameAdapter(backend)
    
    model = {
        'nodes': [
            {'id': 'N1', 'x': 0, 'y': 0, 'z': 0},
            {'id': 'N2', 'x': 3, 'y': 0, 'z': 0},
            {'id': 'N3', 'x': 6, 'y': 0, 'z': 0}
        ],
        'members': [
            {'id': 'M1', 'i': 'N1', 'j': 'N2'},
            {'id': 'M2', 'i': 'N2', 'j': 'N3'}
        ],
        'supports': [
            {'node': 'N1', 'dx': True, 'dy': True, 'dz': True, 'rx': True, 'rz': True},
            {'node': 'N3', 'dy': True, 'dz': True, 'rx': True, 'rz': True}
        ],
        'loads': [
            {'type': 'nodal', 'node': 'N2', 'direction': 'Y', 'magnitude': -10000}
        ]
    }
    
    adapter.load_from_dict(model)
    results = adapter.analyze_and_get_results()
    
    assert results is not None
    assert 'displacements' in results
    assert 'forces' in results
    assert results['displacements']['N2']['dy'] < 0


@pytest.mark.skipif(
    "opensees" not in BackendFactory.available_backends(),
    reason="OpenSees não disponível"
)
def test_adapter_load_opensees():
    """Testa carregamento de modelo com OpenSees."""
    backend = BackendFactory.create("opensees")
    adapter = SimpleFrameAdapter(backend)
    
    model = {
        'nodes': [
            {'id': 'N1', 'x': 0, 'y': 0, 'z': 0},
            {'id': 'N2', 'x': 5, 'y': 0, 'z': 0}
        ],
        'members': [
            {'id': 'M1', 'i': 'N1', 'j': 'N2'}
        ],
        'supports': [
            {'node': 'N1', 'dx': True, 'dy': True, 'dz': True, 'rx': True, 'ry': True, 'rz': True}
        ],
        'loads': [
            {'type': 'nodal', 'node': 'N2', 'direction': 'Y', 'magnitude': -5000}
        ]
    }
    
    assert adapter.load_from_dict(model) is True
    assert len(backend.nodes) == 2
    assert len(backend.members) == 1


@pytest.mark.skipif(
    "opensees" not in BackendFactory.available_backends(),
    reason="OpenSees não disponível"
)
def test_adapter_analyze_opensees():
    """Testa análise completa com adapter OpenSees."""
    backend = BackendFactory.create("opensees")
    adapter = SimpleFrameAdapter(backend)
    
    model = {
        'nodes': [
            {'id': 'N1', 'x': 0, 'y': 0, 'z': 0},
            {'id': 'N2', 'x': 3, 'y': 0, 'z': 0}
        ],
        'members': [
            {'id': 'M1', 'i': 'N1', 'j': 'N2'}
        ],
        'supports': [
            {'node': 'N1', 'dx': True, 'dy': True, 'dz': True, 'rx': True, 'ry': True, 'rz': True}
        ],
        'loads': [
            {'type': 'nodal', 'node': 'N2', 'direction': 'Y', 'magnitude': -5000}
        ]
    }
    
    adapter.load_from_dict(model)
    results = adapter.analyze_and_get_results()
    
    assert results is not None
    assert results['displacements']['N2']['dy'] < 0


def test_adapter_export():
    """Testa exportação de modelo."""
    if "pynite" in BackendFactory.available_backends():
        backend = BackendFactory.create("pynite")
        adapter = SimpleFrameAdapter(backend)
        
        model = {
            'nodes': [
                {'id': 'N1', 'x': 0, 'y': 0, 'z': 0},
                {'id': 'N2', 'x': 5, 'y': 0, 'z': 0}
            ],
            'members': [
                {'id': 'M1', 'i': 'N1', 'j': 'N2'}
            ],
            'supports': [],
            'loads': []
        }
        
        adapter.load_from_dict(model)
        exported = adapter.export_to_dict()
        
        assert len(exported['nodes']) == 2
        assert len(exported['members']) == 1
        assert exported['nodes'][0]['id'] == 'N1'
