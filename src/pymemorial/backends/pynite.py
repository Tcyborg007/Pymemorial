"""
Backend para Pynite (análise linear de estruturas reticuladas).
"""
from typing import Dict
from .base import StructuralBackend, Node, Member

try:
    from Pynite import FEModel3D
    PYNITE_AVAILABLE = True
except ImportError:
    PYNITE_AVAILABLE = False
    FEModel3D = None


class PyniteBackend(StructuralBackend):
    """
    Backend para Pynite.
    
    Suporta:
    - Estruturas 2D e 3D
    - Análise linear estática
    - Elementos de barra/viga
    """
    
    def __init__(self):
        if not PYNITE_AVAILABLE:
            raise ImportError(
                "Pynite não está instalado. "
                "Instale com: pip install PyNiteFEA"
            )
        
        super().__init__(name="Pynite")
        self.model = FEModel3D()
        self._materials_defined = set()
        self._sections_defined = set()
        
        # Definir material e seção padrão
        self._define_default_material()
        self._define_default_section()
    
    def _define_default_material(self):
        """Define material padrão (aço)."""
        if "steel" not in self._materials_defined:
            self.model.add_material(
                name="steel",
                E=200e9,    # Pa (200 GPa)
                G=77e9,     # Pa (77 GPa)
                nu=0.3,     # Poisson
                rho=7850    # kg/m³
            )
            self._materials_defined.add("steel")
    
    def _define_default_section(self):
        """Define seção padrão (IPE200)."""
        if "IPE200" not in self._sections_defined:
            self.model.add_section(
                name="IPE200",
                A=28.5e-4,      # m² (área)
                Iy=1943e-8,     # m⁴ (inércia eixo y)
                Iz=142e-8,      # m⁴ (inércia eixo z)
                J=6.98e-8       # m⁴ (constante torsional)
            )
            self._sections_defined.add("IPE200")
    
    def add_node(self, node_id: str, x: float, y: float, z: float = 0.0):
        """Adiciona nó."""
        self.nodes[node_id] = Node(id=node_id, x=x, y=y, z=z)
        self.model.add_node(node_id, x, y, z)
    
    def add_member(
        self,
        member_id: str,
        node_i: str,
        node_j: str,
        section: str = "IPE200",
        material: str = "steel"
    ):
        """
        Adiciona elemento.
        
        Args:
            member_id: ID do elemento
            node_i: nó inicial
            node_j: nó final
            section: nome da seção (padrão: IPE200)
            material: nome do material (padrão: steel)
        """
        # Garantir que material e seção existem
        if material not in self._materials_defined:
            self._define_default_material()
        
        if section not in self._sections_defined:
            self._define_default_section()
        
        self.members[member_id] = Member(
            id=member_id, node_i=node_i, node_j=node_j,
            section=section, material=material
        )
        
        # ✅ Nova API do Pynite
        self.model.add_member(
            name=member_id,
            i_node=node_i,
            j_node=node_j,
            material_name=material,
            section_name=section
        )
    
    def add_support(self, node_id: str, **restrictions):
        """Adiciona apoio."""
        dx = restrictions.get("dx", False)
        dy = restrictions.get("dy", False)
        dz = restrictions.get("dz", False)
        rx = restrictions.get("rx", False)
        ry = restrictions.get("ry", False)
        rz = restrictions.get("rz", False)
        
        self.model.def_support(
            node_name=node_id,
            support_DX=dx, support_DY=dy, support_DZ=dz,
            support_RX=rx, support_RY=ry, support_RZ=rz
        )
    
    def add_load(self, load_type: str, **kwargs):
        """Adiciona carregamento."""
        if load_type == "nodal":
            node_id = kwargs["node_id"]
            direction = kwargs["direction"]
            magnitude = kwargs["magnitude"]
            
            if direction == "Y":
                self.model.add_node_load(node_id, "FY", magnitude)
            elif direction == "X":
                self.model.add_node_load(node_id, "FX", magnitude)
            elif direction == "Z":
                self.model.add_node_load(node_id, "FZ", magnitude)
    
    def analyze(self) -> bool:
        """Executa análise."""
        try:
            self.model.analyze()
            self._is_analyzed = True
            return True
        except Exception as e:
            print(f"Erro na análise Pynite: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_displacements(self, node_id: str) -> Dict[str, float]:
        """Retorna deslocamentos."""
        if not self._is_analyzed:
            raise RuntimeError("Execute analyze() primeiro")
        
        node = self.model.nodes[node_id]
        return {
            "dx": node.DX.get("Combo 1", 0.0),
            "dy": node.DY.get("Combo 1", 0.0),
            "dz": node.DZ.get("Combo 1", 0.0),
        }
    
    def get_member_forces(self, member_id: str) -> Dict[str, float]:
        """Retorna esforços."""
        if not self._is_analyzed:
            raise RuntimeError("Execute analyze() primeiro")
        
        member = self.model.members[member_id]
        return {
            "N": member.max_axial(),
            "V": member.max_shear("Fy"),
            "M": member.max_moment("Mz"),
        }
