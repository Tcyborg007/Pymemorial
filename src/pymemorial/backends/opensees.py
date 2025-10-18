"""
Backend para OpenSeesPy (análise não-linear de estruturas).
"""
from typing import Dict
from .base import StructuralBackend, Node, Member

try:
    import openseespy.opensees as ops
    OPENSEES_AVAILABLE = True
except ImportError:
    OPENSEES_AVAILABLE = False
    ops = None


class OpenSeesBackend(StructuralBackend):
    """
    Backend para OpenSeesPy.
    
    Suporta:
    - Estruturas 2D e 3D
    - Análise linear e não-linear
    - Elementos de frame (viga-coluna)
    - Análise P-Delta
    """
    
    def __init__(self):
        if not OPENSEES_AVAILABLE:
            raise ImportError(
                "OpenSeesPy não está instalado. "
                "Instale com: pip install openseespy"
            )
        
        super().__init__(name="OpenSees")
        
        # Inicializar modelo OpenSees
        ops.wipe()
        ops.model('basic', '-ndm', 3, '-ndf', 6)  # 3D, 6 graus de liberdade
        
        # Contadores
        self._next_material_tag = 1
        self._next_section_tag = 1
        self._next_transf_tag = 1
        
        # Mapeamentos
        self._material_tags = {}
        self._section_tags = {}
        self._transf_tags = {}
        
        # Definir material e seção padrão
        self._define_default_material()
        self._define_default_section()
        self._define_default_transformation()
    
    def _define_default_material(self):
        """Define material padrão (aço)."""
        if "steel" not in self._material_tags:
            mat_tag = self._next_material_tag
            self._next_material_tag += 1
            
            E = 200e9  # Pa (200 GPa)
            ops.uniaxialMaterial('Elastic', mat_tag, E)
            
            self._material_tags["steel"] = mat_tag
    
    def _define_default_section(self):
        """Define seção padrão (IPE200) com integração Lobatto."""
        if "IPE200" not in self._section_tags:
            sec_tag = self._next_section_tag
            self._next_section_tag += 1
            
            # Propriedades IPE200
            A = 28.5e-4      # m² (área)
            E = 200e9        # Pa
            G = 77e9         # Pa
            Jxx = 6.98e-8    # m⁴ (constante torsional)
            Iy = 1943e-8     # m⁴ (inércia eixo y - major axis)
            Iz = 142e-8      # m⁴ (inércia eixo z - minor axis)
            
            # Definir seção elástica
            ops.section('Elastic', sec_tag, E, A, Iz, Iy, G, Jxx)
            
            # Definir integração Gauss-Lobatto (5 pontos)
            # Lobatto coloca pontos nas extremidades (melhor para momentos)
            ops.beamIntegration('Lobatto', sec_tag, sec_tag, 5)
            
            self._section_tags["IPE200"] = sec_tag

    
    def _define_default_transformation(self):
        """Define transformação geométrica padrão (Linear)."""
        if "Linear" not in self._transf_tags:
            transf_tag = self._next_transf_tag
            self._next_transf_tag += 1
            
            # Vetor eixo local x (vertical)
            vecxz = [0, 0, 1]
            ops.geomTransf('Linear', transf_tag, *vecxz)
            
            self._transf_tags["Linear"] = transf_tag
    
    def add_node(self, node_id: str, x: float, y: float, z: float = 0.0):
        """Adiciona nó."""
        # OpenSees usa tags numéricos - extrair número do ID
        node_tag = int(node_id.replace("N", "").replace("n", ""))
        
        self.nodes[node_id] = Node(id=node_id, x=x, y=y, z=z)
        ops.node(node_tag, x, y, z)
    
    def add_member(
        self,
        member_id: str,
        node_i: str,
        node_j: str,
        section: str = "IPE200",
        material: str = "steel"
    ):
        """
        Adiciona elemento de frame.
        
        Args:
            member_id: ID do elemento
            node_i: nó inicial
            node_j: nó final
            section: nome da seção (padrão: IPE200)
            material: nome do material (não usado diretamente - seção já tem E)
        """
        # Garantir que seção existe
        if section not in self._section_tags:
            self._define_default_section()
        
        # Tags numéricos
        ele_tag = int(member_id.replace("M", "").replace("m", ""))
        inode_tag = int(node_i.replace("N", "").replace("n", ""))
        jnode_tag = int(node_j.replace("N", "").replace("n", ""))
        
        sec_tag = self._section_tags[section]
        transf_tag = self._transf_tags["Linear"]
        
        self.members[member_id] = Member(
            id=member_id, node_i=node_i, node_j=node_j,
            section=section, material=material
        )
        
        # Elemento dispBeamColumn (displacement-based beam-column)
        ops.element('dispBeamColumn', ele_tag, inode_tag, jnode_tag,
                    transf_tag, sec_tag, 5)  # 5 pontos de integração
    
    def add_support(self, node_id: str, **restrictions):
        """Adiciona apoio."""
        node_tag = int(node_id.replace("N", "").replace("n", ""))
        
        # OpenSees: 1 = fixo, 0 = livre
        dx = 1 if restrictions.get("dx", False) else 0
        dy = 1 if restrictions.get("dy", False) else 0
        dz = 1 if restrictions.get("dz", False) else 0
        rx = 1 if restrictions.get("rx", False) else 0
        ry = 1 if restrictions.get("ry", False) else 0
        rz = 1 if restrictions.get("rz", False) else 0
        
        ops.fix(node_tag, dx, dy, dz, rx, ry, rz)
    
    def add_load(self, load_type: str, **kwargs):
        """Adiciona carregamento."""
        if load_type == "nodal":
            node_id = kwargs["node_id"]
            node_tag = int(node_id.replace("N", "").replace("n", ""))
            direction = kwargs["direction"]
            magnitude = kwargs["magnitude"]
            
            # Criar pattern de carga se não existir
            if not hasattr(self, '_load_pattern_created'):
                ops.timeSeries('Linear', 1)
                ops.pattern('Plain', 1, 1)
                self._load_pattern_created = True
            
            # Aplicar carga
            if direction == "X":
                ops.load(node_tag, magnitude, 0.0, 0.0, 0.0, 0.0, 0.0)
            elif direction == "Y":
                ops.load(node_tag, 0.0, magnitude, 0.0, 0.0, 0.0, 0.0)
            elif direction == "Z":
                ops.load(node_tag, 0.0, 0.0, magnitude, 0.0, 0.0, 0.0)
    
    def analyze(self) -> bool:
        """Executa análise."""
        try:
            # Criar sistema de análise
            ops.system('BandSPD')
            ops.numberer('RCM')
            ops.constraints('Plain')
            ops.integrator('LoadControl', 1.0)
            ops.algorithm('Linear')
            ops.analysis('Static')
            
            # Executar análise
            ok = ops.analyze(1)
            
            if ok == 0:
                self._is_analyzed = True
                return True
            else:
                print(f"Análise OpenSees falhou com código: {ok}")
                return False
                
        except Exception as e:
            print(f"Erro na análise OpenSees: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_displacements(self, node_id: str) -> Dict[str, float]:
        """Retorna deslocamentos."""
        if not self._is_analyzed:
            raise RuntimeError("Execute analyze() primeiro")
        
        node_tag = int(node_id.replace("N", "").replace("n", ""))
        disp = ops.nodeDisp(node_tag)
        
        return {
            "dx": disp[0],
            "dy": disp[1],
            "dz": disp[2],
            "rx": disp[3],
            "ry": disp[4],
            "rz": disp[5],
        }
    
    def get_member_forces(self, member_id: str) -> Dict[str, float]:
        """Retorna esforços."""
        if not self._is_analyzed:
            raise RuntimeError("Execute analyze() primeiro")
        
        ele_tag = int(member_id.replace("M", "").replace("m", ""))
        forces = ops.eleForce(ele_tag)
        
        # Forças no nó i: [Fx, Fy, Fz, Mx, My, Mz]
        # Forças no nó j: [Fx, Fy, Fz, Mx, My, Mz]
        return {
            "N": abs(forces[0]),      # Força axial (nó i)
            "Vy": abs(forces[1]),     # Cortante Y
            "Vz": abs(forces[2]),     # Cortante Z
            "T": abs(forces[3]),      # Torção
            "My": abs(max(forces[4], forces[10])),  # Momento Y (max entre nós)
            "Mz": abs(max(forces[5], forces[11])),  # Momento Z (max entre nós)
        }
    
    def __del__(self):
        """Limpar modelo OpenSees ao destruir objeto."""
        try:
            ops.wipe()
        except:
            pass
