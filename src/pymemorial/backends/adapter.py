"""
Adapters para conversão entre modelos estruturais e backends.
"""
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
from .base import StructuralBackend


class StructuralAdapter(ABC):
    """
    Adaptador abstrato para conversão de modelos estruturais.
    
    Converte representações de alto nível (dicionários, dataclasses)
    para comandos específicos de cada backend.
    """
    
    def __init__(self, backend: StructuralBackend):
        """
        Inicializa adapter.
        
        Args:
            backend: Backend estrutural a ser usado
        """
        self.backend = backend
    
    @abstractmethod
    def load_from_dict(self, model_dict: Dict[str, Any]) -> bool:
        """
        Carrega modelo a partir de dicionário.
        
        Args:
            model_dict: Dicionário com definição do modelo
                {
                    'nodes': [...],
                    'members': [...],
                    'supports': [...],
                    'loads': [...]
                }
        
        Returns:
            True se bem-sucedido
        """
        pass
    
    @abstractmethod
    def export_to_dict(self) -> Dict[str, Any]:
        """
        Exporta modelo atual para dicionário.
        
        Returns:
            Dicionário com modelo completo
        """
        pass


class SimpleFrameAdapter(StructuralAdapter):
    """
    Adapter para estruturas de pórtico simples (frames 2D/3D).
    
    Formato do dicionário:
    {
        'nodes': [
            {'id': 'N1', 'x': 0, 'y': 0, 'z': 0},
            {'id': 'N2', 'x': 5, 'y': 0, 'z': 0}
        ],
        'members': [
            {'id': 'M1', 'i': 'N1', 'j': 'N2', 'section': 'IPE200'}
        ],
        'supports': [
            {'node': 'N1', 'dx': True, 'dy': True, 'dz': True}
        ],
        'loads': [
            {'type': 'nodal', 'node': 'N2', 'direction': 'Y', 'magnitude': -10000}
        ]
    }
    """
    
    def load_from_dict(self, model_dict: Dict[str, Any]) -> bool:
        """Carrega modelo de pórtico a partir de dicionário."""
        try:
            # 1. Adicionar nós
            for node_data in model_dict.get('nodes', []):
                self.backend.add_node(
                    node_id=node_data['id'],
                    x=node_data['x'],
                    y=node_data['y'],
                    z=node_data.get('z', 0.0)
                )
            
            # 2. Adicionar elementos
            for member_data in model_dict.get('members', []):
                self.backend.add_member(
                    member_id=member_data['id'],
                    node_i=member_data['i'],
                    node_j=member_data['j'],
                    section=member_data.get('section', 'IPE200'),
                    material=member_data.get('material', 'steel')
                )
            
            # 3. Adicionar apoios
            for support_data in model_dict.get('supports', []):
                # Copiar dicionário para não modificar original
                support_copy = support_data.copy()
                node_id = support_copy.pop('node')
                self.backend.add_support(node_id, **support_copy)
            
            # 4. Adicionar cargas
            for load_data in model_dict.get('loads', []):
                # Copiar dicionário para não modificar original
                load_copy = load_data.copy()
                load_type = load_copy.pop('type')
                # Converter 'node' para 'node_id'
                if 'node' in load_copy:
                    load_copy['node_id'] = load_copy.pop('node')
                self.backend.add_load(load_type, **load_copy)
            
            return True
            
        except Exception as e:
            print(f"Erro ao carregar modelo: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def export_to_dict(self) -> Dict[str, Any]:
        """Exporta modelo atual para dicionário."""
        return {
            'nodes': [
                {
                    'id': node.id,
                    'x': node.x,
                    'y': node.y,
                    'z': node.z
                }
                for node in self.backend.nodes.values()
            ],
            'members': [
                {
                    'id': member.id,
                    'i': member.node_i,
                    'j': member.node_j,
                    'section': member.section,
                    'material': member.material
                }
                for member in self.backend.members.values()
            ]
        }
    
    def analyze_and_get_results(self) -> Optional[Dict[str, Any]]:
        """
        Executa análise e retorna resultados.
        
        Returns:
            Dicionário com resultados ou None se falhar
        """
        if not self.backend.analyze():
            return None
        
        results = {
            'displacements': {},
            'forces': {}
        }
        
        # Coletar deslocamentos de todos os nós
        for node_id in self.backend.nodes.keys():
            results['displacements'][node_id] = self.backend.get_displacements(node_id)
        
        # Coletar esforços de todos os elementos
        for member_id in self.backend.members.keys():
            results['forces'][member_id] = self.backend.get_member_forces(member_id)
        
        return results
