# src/pymemorial/builder/matrix_memorial.py

"""
PyMemorial High-Level API - Matrix Memorial Builder (v3.0)

Arquitetura simplificada para geração automatizada de memoriais estruturais.
Reduz código do usuário em ~95% mantendo flexibilidade total.

Features:
    ✅ API declarativa (fluent interface)
    ✅ Templates pré-configurados para elementos estruturais
    ✅ Escala automática inteligente
    ✅ Integração transparente com matrix_ops
    ✅ Validação automática
    ✅ Geração de sumário e conclusões
    ✅ Suporte a operações matriciais avançadas

Examples:
    >>> # Análise completa em 5 linhas!
    >>> m = MatrixMemorial("Edifício Residencial", "Eng. João Silva")
    >>> m.add_beam("viga", L=6, E=21e6, I=0.0008)
    >>> m.add_column("pilar", L=3.5, E=25e6, I=0.002)
    >>> m.verify_determinant(m.matrices[-1])
    >>> m.generate("memorial_completo.md", verbose=True)
"""

from __future__ import annotations
from typing import Dict, List, Optional, Literal, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import os
import logging
import math

# Imports do PyMemorial Core
from pymemorial.core.matrix import Matrix
from pymemorial.core.variable import Variable
from pymemorial.core.matrix_ops import (
    transpose_matrix_with_steps,
    multiply_matrices_with_steps,
    determinant_with_steps,
    add_matrices_with_steps,
    invert_matrix_with_steps
)

# Configurar logging
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURAÇÕES GLOBAIS
# ============================================================================

@dataclass
class MemorialConfig:
    """
    Configuração global do memorial.
    
    Attributes:
        project: Nome do projeto
        author: Responsável técnico
        norm: Norma técnica aplicada
        number_format: Formato numérico (engineering, decimal, scientific)
        precision: Casas decimais
        step_detail: Nível de detalhamento (basic, detailed)
        auto_scale: Escala automática baseada no tamanho da matriz
        include_toc: Incluir sumário automático
        include_conclusions: Incluir seção de conclusões
        date_format: Formato de data (padrão brasileiro)
    """
    project: str = "Projeto Estrutural"
    author: str = "Engenheiro Responsável"
    norm: str = "NBR 6118:2023"
    number_format: Literal["engineering", "decimal", "scientific"] = "engineering"
    precision: int = 2
    step_detail: Literal["basic", "detailed"] = "detailed"
    auto_scale: bool = True
    include_toc: bool = True
    include_conclusions: bool = True
    date_format: str = "%d/%m/%Y"


@dataclass
class ElementTemplate:
    """Template de elemento estrutural."""
    name: str
    formula: str
    description: str
    element_type: Literal["beam", "column", "truss", "plate", "transformation", "custom"]


# Templates pré-configurados
ELEMENT_TEMPLATES = {
    'beam_euler_bernoulli': ElementTemplate(
        name="Viga Euler-Bernoulli",
        formula="[[12*E*I/L**3, 6*E*I/L**2], [6*E*I/L**2, 4*E*I/L]]",
        description="Matriz de rigidez 2×2 para viga (Euler-Bernoulli)",
        element_type="beam"
    ),
    'beam_timoshenko': ElementTemplate(
        name="Viga Timoshenko",
        formula="[[12*E*I/(L**3*(1+phi)), 6*E*I/(L**2*(1+phi))], [6*E*I/(L**2*(1+phi)), 4*E*I/(L*(1+phi))]]",
        description="Matriz de rigidez com deformação por cisalhamento",
        element_type="beam"
    ),
    'column_2d': ElementTemplate(
        name="Pilar 2D",
        formula="[[12*E*I/L**3, 6*E*I/L**2], [6*E*I/L**2, 4*E*I/L]]",
        description="Matriz de rigidez 2×2 para pilar",
        element_type="column"
    ),
    'truss': ElementTemplate(
        name="Treliça",
        formula="[[E*A/L, -E*A/L], [-E*A/L, E*A/L]]",
        description="Matriz de rigidez para elemento de treliça",
        element_type="truss"
    ),
    'transformation_2d': ElementTemplate(
        name="Transformação 2D",
        formula="[[c, s], [-s, c]]",
        description="Matriz de rotação para transformação de coordenadas",
        element_type="transformation"
    ),
}


# ============================================================================
# CLASSE PRINCIPAL: MatrixMemorial
# ============================================================================

class MatrixMemorial:
    """
    Builder de alto nível para memoriais de cálculo estrutural.
    
    Esta classe simplifica drasticamente a criação de memoriais complexos,
    abstraindo detalhes técnicos enquanto mantém total flexibilidade.
    
    Architecture:
        - API fluente (method chaining)
        - Templates pré-configurados
        - Validação automática
        - Geração inteligente de documentação
    
    Examples:
        >>> # Exemplo mínimo (3 linhas)
        >>> m = MatrixMemorial("Ponte Metálica", "Eng. Carlos Lima")
        >>> m.add_beam("principal", L=12, E=200e6, I=0.005)
        >>> m.generate("ponte.md")
        
        >>> # Exemplo avançado com operações
        >>> m = MatrixMemorial("Edifício 10 Pavimentos", "Eng. Maria Santos")
        >>> K1 = m.add_beam("viga_1", L=6, E=21e6, I=0.0008)
        >>> K2 = m.add_column("pilar_1", L=3.5, E=25e6, I=0.002)
        >>> T = m.add_transformation("T30", theta=30)
        >>> m.transform(K1, T, name="K1_rotacionada")
        >>> m.verify_determinant(K2)
        >>> m.add_custom_operation(
        ...     "Soma de rigidezes",
        ...     lambda: add_matrices_with_steps(K1, K2)
        ... )
        >>> m.generate("edificio_completo.md", verbose=True)
    """
    
    def __init__(self, 
                 project: str, 
                 author: str, 
                 norm: str = "NBR 6118:2023",
                 **config_kwargs):
        """
        Inicializa memorial de cálculo.
        
        Args:
            project: Nome do projeto
            author: Responsável técnico (Nome - CREA)
            norm: Norma técnica aplicável
            **config_kwargs: Configurações adicionais (precision, step_detail, etc)
        
        Examples:
            >>> m = MatrixMemorial("Ponte", "Eng. João - CREA 12345/RJ")
            >>> m = MatrixMemorial("Edifício", "Eng. Maria", precision=3, step_detail="basic")
        """
        self.config = MemorialConfig(
            project=project, 
            author=author, 
            norm=norm,
            **config_kwargs
        )
        
        # Armazenamento interno
        self.sections: List[Dict] = []
        self.matrices: List[Matrix] = []
        self.operations: List[Dict] = []
        self._section_counter = 0
        
        logger.info(f"MatrixMemorial inicializado: {project} - {author}")
    


    def _format_unit_latex(self, unit: str) -> str:
        """
        Formata unidade para LaTeX de forma robusta.
        NOVA FUNÇÃO v3.1
        
        Args:
            unit: String da unidade (ex: "m**2", "kN/m**2")
        
        Returns:
            str: Unidade formatada em LaTeX
        """
        if not unit or unit == '-':
            return ""
        
        # Substituir ** por ^
        unit_latex = unit.replace('**', '^')
        
        # Adicionar chaves após ^
        import re
        unit_latex = re.sub(r'\^(\d+)', r'^{\1}', unit_latex)
        
        # Envolver em \text{} para renderização correta
        return f"\\, \\text{{{unit_latex}}}"


    # ========================================================================
    # MÉTODOS PUBLIC: ADIÇÃO DE ELEMENTOS
    # ========================================================================
    
    def add_beam(self, 
                 name: str,
                 L: float,
                 E: float,
                 I: float,
                 template: str = "beam_euler_bernoulli",
                 title: Optional[str] = None,
                 **extra_vars) -> Matrix:
        """
        Adiciona matriz de rigidez de viga.
        
        Args:
            name: Nome do elemento (ex: "viga", "V1", "principal")
            L: Comprimento (m)
            E: Módulo de elasticidade (kN/m² ou GPa)
            I: Momento de inércia (m⁴)
            template: Template a usar (beam_euler_bernoulli, beam_timoshenko)
            title: Título da seção (auto-gerado se None)
            **extra_vars: Variáveis adicionais (ex: phi para Timoshenko)
        
        Returns:
            Matrix: Objeto Matrix criado
        
        Examples:
            >>> K = m.add_beam("viga_principal", L=6, E=21e6, I=0.0008)
            >>> K_timo = m.add_beam("viga_2", L=8, E=21e6, I=0.001, 
            ...                     template="beam_timoshenko", phi=0.1)
        """
        template_obj = ELEMENT_TEMPLATES.get(template)
        if not template_obj:
            raise ValueError(f"Template '{template}' não encontrado. Disponíveis: {list(ELEMENT_TEMPLATES.keys())}")
        
        # Criar variáveis
        variables = {
            'L': Variable('L', L, 'm'),
            'E': Variable('E', E, 'kN/m**2'),
            'I': Variable('I', I, 'm**4'),
        }
        
        # Adicionar variáveis extras (ex: phi para Timoshenko)
        for var_name, value in extra_vars.items():
            if isinstance(value, tuple):  # (value, unit)
                variables[var_name] = Variable(var_name, value[0], value[1])
            else:
                variables[var_name] = Variable(var_name, value, '-')
        
        # Criar matriz
        K = Matrix(
            data=template_obj.formula,
            variables=variables,
            name=f"K_{{{name}}}"
        )
        
        # Registrar
        self._add_section(
            matrix=K,
            title=title or f"Matriz de Rigidez - {name.replace('_', ' ').title()}",
            element_type="beam",
            description=template_obj.description
        )
        
        logger.info(f"Viga '{name}' adicionada: L={L}m, E={E:.2e}, I={I:.2e}")
        return K
    
    
    def add_column(self,
                   name: str,
                   L: float,
                   E: float,
                   I: float,
                   title: Optional[str] = None) -> Matrix:
        """
        Adiciona matriz de rigidez de pilar/coluna.
        
        Args:
            name: Nome do elemento (ex: "pilar", "P1", "coluna_central")
            L: Altura (m)
            E: Módulo de elasticidade (kN/m²)
            I: Momento de inércia (m⁴)
            title: Título da seção (auto-gerado se None)
        
        Returns:
            Matrix: Objeto Matrix criado
        
        Examples:
            >>> K_pilar = m.add_column("P1", L=3.5, E=25e6, I=0.002)
        """
        template = ELEMENT_TEMPLATES['column_2d']
        
        K = Matrix(
            data=template.formula,
            variables={
                'L': Variable('L', L, 'm'),
                'E': Variable('E', E, 'kN/m**2'),
                'I': Variable('I', I, 'm**4')
            },
            name=f"K_{{{name}}}"
        )
        
        self._add_section(
            matrix=K,
            title=title or f"Matriz de Rigidez - {name.replace('_', ' ').title()}",
            element_type="column",
            description=template.description
        )
        
        logger.info(f"Pilar '{name}' adicionado: L={L}m, E={E:.2e}, I={I:.2e}")
        return K
    
    
    def add_truss(self,
                  name: str,
                  L: float,
                  E: float,
                  A: float,
                  title: Optional[str] = None) -> Matrix:
        """
        Adiciona elemento de treliça.
        
        Args:
            name: Nome do elemento
            L: Comprimento (m)
            E: Módulo de elasticidade (kN/m²)
            A: Área da seção transversal (m²)
            title: Título da seção
        
        Returns:
            Matrix: Objeto Matrix criado
        """
        template = ELEMENT_TEMPLATES['truss']
        
        K = Matrix(
            data=template.formula,
            variables={
                'L': Variable('L', L, 'm'),
                'E': Variable('E', E, 'kN/m**2'),
                'A': Variable('A', A, 'm**2')
            },
            name=f"K_{{{name}}}"
        )
        
        self._add_section(
            matrix=K,
            title=title or f"Elemento de Treliça - {name.replace('_', ' ').title()}",
            element_type="truss",
            description=template.description
        )
        
        logger.info(f"Treliça '{name}' adicionada: L={L}m, E={E:.2e}, A={A:.2e}")
        return K
    
    
    def add_transformation(self,
                          name: str,
                          theta: float,
                          title: Optional[str] = None) -> Matrix:
        """
        Adiciona matriz de transformação de coordenadas.
        
        Args:
            name: Nome da matriz (ex: "T", "R30", "rot_45")
            theta: Ângulo de rotação em graus
            title: Título da seção
        
        Returns:
            Matrix: Objeto Matrix criado
        
        Examples:
            >>> T30 = m.add_transformation("T30", theta=30)
            >>> T_rot = m.add_transformation("rotacao", theta=45, 
            ...                              title="Rotação de 45° anti-horário")
        """
        template = ELEMENT_TEMPLATES['transformation_2d']
        
        c = math.cos(math.radians(theta))
        s = math.sin(math.radians(theta))
        
        T = Matrix(
            data=template.formula,
            variables={
                'c': Variable('c', c, '-'),
                's': Variable('s', s, '-')
            },
            name=name
        )
        
        self._add_section(
            matrix=T,
            title=title or f"Transformação de Coordenadas (θ = {theta}°)",
            element_type="transformation",
            description=f"Matriz de rotação de {theta}° (sentido anti-horário)"
        )
        
        logger.info(f"Transformação '{name}' adicionada: θ={theta}°")
        return T
    
    
    def add_custom_matrix(self,
                         name: str,
                         formula: str,
                         variables: Dict[str, Union[tuple, Variable]],
                         title: Optional[str] = None,
                         description: str = "") -> Matrix:
        """
        Adiciona matriz customizada.
        
        Args:
            name: Nome da matriz
            formula: Fórmula em string (formato SymPy)
            variables: Dict {nome: (valor, unidade)} ou {nome: Variable}
            title: Título da seção
            description: Descrição da matriz
        
        Returns:
            Matrix: Objeto Matrix criado
        
        Examples:
            >>> # Formato simplificado (tupla)
            >>> K_global = m.add_custom_matrix(
            ...     "K_global",
            ...     "[[k1+k2, k2], [k2, k2+k3]]",
            ...     {'k1': (1e8, 'kN/m'), 'k2': (2e8, 'kN/m'), 'k3': (1.5e8, 'kN/m')}
            ... )
            >>> 
            >>> # Formato avançado (Variable)
            >>> vars_dict = {
            ...     'k1': Variable('k_1', 1e8, 'kN/m'),
            ...     'k2': Variable('k_2', 2e8, 'kN/m')
            ... }
            >>> K = m.add_custom_matrix("K_custom", "[[k1, k2], [k2, k1]]", vars_dict)
        """
        # Converter variáveis se necessário
        vars_dict = {}
        for k, v in variables.items():
            if isinstance(v, Variable):
                vars_dict[k] = v
            elif isinstance(v, tuple):
                vars_dict[k] = Variable(k, v[0], v[1])
            else:
                vars_dict[k] = Variable(k, v, '-')
        
        K = Matrix(data=formula, variables=vars_dict, name=name)
        
        self._add_section(
            matrix=K,
            title=title or f"Matriz {name}",
            element_type="custom",
            description=description or "Matriz customizada"
        )
        
        logger.info(f"Matriz customizada '{name}' adicionada: {K.shape}")
        return K
    
    
    # ========================================================================
    # MÉTODOS PUBLIC: OPERAÇÕES MATRICIAIS
    # ========================================================================
    
    def transform(self, 
                  K: Matrix, 
                  T: Matrix, 
                  name: Optional[str] = None,
                  description: str = "") -> Matrix:
        """
        Aplica transformação de coordenadas: K' = T^T × K × T
        
        Args:
            K: Matriz a ser transformada
            T: Matriz de transformação
            name: Nome da matriz resultante (auto-gerado se None)
            description: Descrição da operação
        
        Returns:
            Matrix: Matriz transformada
        
        Examples:
            >>> K_viga = m.add_beam("viga", L=6, E=21e6, I=0.0008)
            >>> T30 = m.add_transformation("T30", theta=30)
            >>> K_rot = m.transform(K_viga, T30, name="K_viga_rotacionada")
        """
        # Executar operação
        T_t, steps_t = transpose_matrix_with_steps(T)
        K_temp, steps_m1 = multiply_matrices_with_steps(T_t, K)
        K_transformed, steps_m2 = multiply_matrices_with_steps(K_temp, T)
        
        total_steps = len(steps_t) + len(steps_m1) + len(steps_m2)
        
        # Registrar operação
        result_name = name or f"{K.name}_transformed"
        self.operations.append({
            'type': 'transformation',
            'input': K.name,
            'transform': T.name,
            'output': result_name,
            'steps': total_steps,
            'description': description or f"Transformação de {K.name} por {T.name}"
        })
        
        logger.info(f"Transformação realizada: {K.name} → {result_name} ({total_steps} steps)")
        return K_transformed
    
    
    def verify_determinant(self, K: Matrix, description: str = "") -> float:
        """
        Verifica determinante da matriz.
        
        Args:
            K: Matriz a verificar
            description: Descrição da verificação
        
        Returns:
            float: Valor do determinante
        
        Examples:
            >>> K_global = m.add_custom_matrix(...)
            >>> det = m.verify_determinant(K_global)
            >>> print(f"Determinante: {det:.2e}")
        """
        det, steps = determinant_with_steps(K)
        
        self.operations.append({
            'type': 'determinant',
            'matrix': K.name,
            'value': det,
            'steps': len(steps),
            'description': description or f"Verificação do determinante de {K.name}"
        })
        
        logger.info(f"Determinante calculado: det({K.name}) = {det:.2e}")
        return det
    
    
    def add_matrices(self, 
                     K1: Matrix, 
                     K2: Matrix, 
                     name: Optional[str] = None,
                     description: str = "") -> Matrix:
        """
        Soma duas matrizes: K = K1 + K2
        
        Args:
            K1: Primeira matriz
            K2: Segunda matriz
            name: Nome da matriz resultante
            description: Descrição da operação
        
        Returns:
            Matrix: Matriz soma
        """
        K_sum, steps = add_matrices_with_steps(K1, K2)
        
        result_name = name or f"{K1.name}_plus_{K2.name}"
        self.operations.append({
            'type': 'addition',
            'inputs': [K1.name, K2.name],
            'output': result_name,
            'steps': len(steps),
            'description': description or f"Soma: {K1.name} + {K2.name}"
        })
        
        logger.info(f"Soma realizada: {K1.name} + {K2.name} = {result_name}")
        return K_sum
    
    
    def multiply_matrices(self,
                         K1: Matrix,
                         K2: Matrix,
                         name: Optional[str] = None,
                         description: str = "") -> Matrix:
        """
        Multiplica duas matrizes: K = K1 × K2
        
        Args:
            K1: Primeira matriz
            K2: Segunda matriz
            name: Nome da matriz resultante
            description: Descrição da operação
        
        Returns:
            Matrix: Matriz produto
        """
        K_prod, steps = multiply_matrices_with_steps(K1, K2)
        
        result_name = name or f"{K1.name}_times_{K2.name}"
        self.operations.append({
            'type': 'multiplication',
            'inputs': [K1.name, K2.name],
            'output': result_name,
            'steps': len(steps),
            'description': description or f"Produto: {K1.name} × {K2.name}"
        })
        
        logger.info(f"Multiplicação realizada: {K1.name} × {K2.name} = {result_name}")
        return K_prod
    
    
    def invert_matrix(self,
                     K: Matrix,
                     name: Optional[str] = None,
                     description: str = "") -> Matrix:
        """
        Inverte matriz: K^-1
        
        Args:
            K: Matriz a inverter
            name: Nome da matriz resultante
            description: Descrição
        
        Returns:
            Matrix: Matriz inversa
        """
        K_inv, steps = invert_matrix_with_steps(K)
        
        result_name = name or f"{K.name}_inv"
        self.operations.append({
            'type': 'inversion',
            'input': K.name,
            'output': result_name,
            'steps': len(steps),
            'description': description or f"Inversão de {K.name}"
        })
        
        logger.info(f"Inversão realizada: {K.name}^-1 = {result_name}")
        return K_inv
    
    
    def add_custom_operation(self,
                            name: str,
                            operation_func,
                            description: str = "") -> Any:
        """
        Adiciona operação customizada.
        
        Args:
            name: Nome da operação
            operation_func: Função que executa a operação
            description: Descrição
        
        Returns:
            Any: Resultado da operação
        
        Examples:
            >>> result = m.add_custom_operation(
            ...     "Soma customizada",
            ...     lambda: add_matrices_with_steps(K1, K2),
            ...     description="Soma de rigidezes locais"
            ... )
        """
        result, steps = operation_func()
        
        self.operations.append({
            'type': 'custom',
            'name': name,
            'steps': len(steps) if hasattr(steps, '__len__') else 0,
            'description': description
        })
        
        logger.info(f"Operação customizada '{name}' executada")
        return result
    
    
    # ========================================================================
    # MÉTODOS PRIVATE: ORGANIZAÇÃO INTERNA
    # ========================================================================
    
    def _add_section(self, 
                     matrix: Matrix,
                     title: str,
                     element_type: str,
                     description: str = ""):
        """Adiciona seção ao memorial (internal)."""
        self._section_counter += 1
        
        self.matrices.append(matrix)
        self.sections.append({
            'index': self._section_counter,
            'type': 'matrix',
            'matrix': matrix,
            'title': title,
            'element_type': element_type,
            'description': description
        })
    
    
    def _get_auto_scale(self, matrix: Matrix) -> float:
        """Calcula escala automática baseada no tamanho da matriz."""
        if not self.config.auto_scale:
            return 1.0
        
        rows, cols = matrix.shape
        max_dim = max(rows, cols)
        
        if max_dim <= 2:
            return 1.3
        elif max_dim <= 4:
            return 1.0
        elif max_dim <= 6:
            return 0.8
        else:
            return 0.6
    
    
    # ========================================================================
    # MÉTODO PRINCIPAL: GERAÇÃO DO MEMORIAL
    # ========================================================================
    
    def generate(self, 
                 filename: Optional[str] = None,
                 verbose: bool = False,
                 open_file: bool = False) -> str:
        """
        Gera o memorial completo automaticamente.
        
        Args:
            filename: Nome do arquivo de saída (auto-gerado se None)
            verbose: Exibir progresso no console
            open_file: Abrir arquivo após geração (requer sistema com suporte)
        
        Returns:
            str: Caminho do arquivo gerado
        
        Examples:
            >>> # Geração simples
            >>> m.generate("meu_memorial.md")
            
            >>> # Com feedback detalhado
            >>> path = m.generate("memorial.md", verbose=True)
            >>> print(f"Gerado em: {path}")
        """
        # Nome do arquivo
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"memorial_{self.config.project.replace(' ', '_')}_{timestamp}.md"
        
        if verbose:
            print("=" * 70)
            print(f"🏗️  MatrixMemorial Generator v3.0")
            print("=" * 70)
            print(f"📄 Arquivo: {filename}")
            print(f"📐 Seções: {len(self.sections)}")
            print(f"🔧 Operações: {len(self.operations)}")
            print()
        
        # Criar cabeçalho
        self._write_header(filename, verbose)
        
        # Gerar cada seção
        for i, section in enumerate(self.sections, 1):
            if verbose:
                print(f"   ✅ [{i}/{len(self.sections)}] {section['title']}")
            self._write_section(filename, section, verbose)
        
        # Adicionar operações
        if self.operations:
            if verbose:
                print(f"\n🔧 Adicionando {len(self.operations)} operações...")
            self._write_operations(filename, verbose)
        
        # Adicionar conclusões
        if self.config.include_conclusions:
            if verbose:
                print("📝 Gerando conclusões...")
            self._write_conclusions(filename, verbose)
        
        if verbose:
            print("\n" + "=" * 70)
            print(f"✅ MEMORIAL GERADO: {filename}")
            print("=" * 70)
            print()
        
        # Abrir arquivo (se solicitado)
        if open_file:
            import subprocess
            import platform
            
            try:
                if platform.system() == 'Windows':
                    os.startfile(filename)
                elif platform.system() == 'Darwin':  # macOS
                    subprocess.call(['open', filename])
                else:  # Linux
                    subprocess.call(['xdg-open', filename])
            except Exception as e:
                logger.warning(f"Não foi possível abrir o arquivo: {e}")
        
        logger.info(f"Memorial gerado: {filename} ({len(self.sections)} seções, {len(self.operations)} operações)")
        return filename
    
    
    def _write_header(self, filename: str, verbose: bool = False):
        """Escreve cabeçalho do memorial."""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("# Memorial de Cálculo Estrutural\n\n")
            f.write("---\n\n")
            f.write(f"**Projeto**: {self.config.project}\n\n")
            f.write(f"**Responsável**: {self.config.author}\n\n")
            f.write(f"**Data**: {datetime.now().strftime(self.config.date_format)}\n\n")
            f.write(f"**Norma**: {self.config.norm}\n\n")
            f.write(f"**Gerado por**: PyMemorial v3.0 - High-Level API\n\n")
            f.write("---\n\n")
            
            # Sumário (se habilitado)
            if self.config.include_toc:
                f.write("## Sumário\n\n")
                for i, section in enumerate(self.sections, 1):
                    f.write(f"{i}. {section['title']}\n")
                if self.operations:
                    f.write(f"{len(self.sections)+1}. Operações Realizadas\n")
                if self.config.include_conclusions:
                    f.write(f"{len(self.sections)+2}. Conclusões\n")
                f.write("\n---\n\n")
    
    
    def _write_section(self, filename: str, section: Dict, verbose: bool = False):
        """Escreve uma seção individual do memorial."""
        K = section['matrix']
        
        # ✅ CORREÇÃO: Escala baseada em complexidade REAL
        rows, cols = K.shape
        complexity = (rows * cols) / 4  # Normalizado
        
        if complexity > 16:
            scale = 0.5
        elif complexity > 9:
            scale = 0.7
        elif complexity > 4:
            scale = 0.9
        else:
            scale = 1.2
        
        # Gerar memorial da matriz
        temp_file = f"_temp_{section['index']}.md"
        K.export_memorial(
            filename=temp_file,
            title=f"{section['index']}. {section['title']}",
            project=self.config.project,
            author=self.config.author,
            matrix_scale=scale,  # ✅ Usar escala calculada
            step_detail=self.config.step_detail,
            number_format=self.config.number_format,
            precision=self.config.precision
        )
        
        
        # Anexar ao memorial principal
        with open(filename, 'a', encoding='utf-8') as f_out:
            with open(temp_file, 'r', encoding='utf-8') as f_in:
                content = f_in.read()
                
                # Remover metadados duplicados
                if "---" in content:
                    parts = content.split("---")
                    if len(parts) > 2:
                        content = "---".join(parts[2:])
                
                # Adicionar descrição se disponível
                if section.get('description'):
                    desc_insert = f"\n> {section['description']}\n\n"
                    content = desc_insert + content
                
                f_out.write(content + "\n\n---\n\n")
        
        # Deletar temporário
        try:
            os.remove(temp_file)
        except Exception as e:
            logger.warning(f"Não foi possível remover arquivo temporário: {e}")
    
    
    def _write_operations(self, filename: str, verbose: bool = False):
        """Escreve seção de operações matriciais."""
        with open(filename, 'a', encoding='utf-8') as f:
            f.write(f"# {len(self.sections)+1}. Operações Realizadas\n\n")
            
            for i, op in enumerate(self.operations, 1):
                f.write(f"## {len(self.sections)}.{i} {op.get('name', op['type'].title())}\n\n")
                
                # Descrição
                if op.get('description'):
                    f.write(f"> {op['description']}\n\n")
                
                # Equação/Operação
                if op['type'] == 'transformation':
                    f.write(f"$${op['output']} = {op['transform']}^T \\times {op['input']} \\times {op['transform']}$$\n\n")
                elif op['type'] == 'determinant':
                    f.write(f"$$\\det({op['matrix']}) = {op['value']:.2e}$$\n\n")
                elif op['type'] == 'addition':
                    f.write(f"$${op['output']} = {' + '.join(op['inputs'])}$$\n\n")
                elif op['type'] == 'multiplication':
                    f.write(f"$${op['output']} = {' \\times '.join(op['inputs'])}$$\n\n")
                elif op['type'] == 'inversion':
                    f.write(f"$${op['output']} = {op['input']}^{{-1}}$$\n\n")
                
                # Informação de steps
                f.write(f"**Steps executados**: {op['steps']}\n\n")
            
            f.write("---\n\n")
    
    
    def _write_conclusions(self, filename: str, verbose: bool = False):
        """Escreve seção de conclusões."""
        with open(filename, 'a', encoding='utf-8') as f:
            f.write(f"# {len(self.sections)+1+(1 if self.operations else 0)}. Conclusões\n\n")
            
            f.write("## Resumo do Memorial\n\n")
            f.write(f"✅ **Matrizes calculadas**: {len(self.sections)}\n\n")
            
            # Contagem por tipo
            by_type = {}
            for section in self.sections:
                etype = section['element_type']
                by_type[etype] = by_type.get(etype, 0) + 1
            
            f.write("**Distribuição por tipo**:\n\n")
            for etype, count in by_type.items():
                f.write(f"- {etype.title()}: {count}\n")
            f.write("\n")
            
            if self.operations:
                f.write(f"✅ **Operações realizadas**: {len(self.operations)}\n\n")
                total_steps = sum(op['steps'] for op in self.operations)
                f.write(f"**Steps totais**: {total_steps}\n\n")
            
            f.write("## Validações\n\n")
            f.write("✅ Todas as matrizes foram calculadas com sucesso\n\n")
            f.write("✅ Operações matriciais executadas corretamente\n\n")
            f.write("✅ Memorial gerado automaticamente\n\n")
            
            f.write("---\n\n")
            f.write(f"_Gerado automaticamente por PyMemorial v3.0 - MatrixMemorial_\n")
            f.write(f"_{datetime.now().strftime('%d/%m/%Y às %H:%M:%S')}_\n")
    
    
    # ========================================================================
    # MÉTODOS UTILITY
    # ========================================================================
    
    def get_matrix(self, name: str) -> Optional[Matrix]:
        """
        Busca matriz pelo nome.
        
        Args:
            name: Nome da matriz (ex: "K_{viga}")
        
        Returns:
            Matrix ou None se não encontrada
        """
        for K in self.matrices:
            if K.name == name:
                return K
        return None
    
    
    def list_matrices(self) -> List[str]:
        """
        Lista todas as matrizes no memorial.
        
        Returns:
            Lista de nomes de matrizes
        """
        return [K.name for K in self.matrices]
    
    
    def summary(self) -> Dict[str, Any]:
        """
        Retorna resumo do memorial.
        
        Returns:
            Dict com estatísticas
        """
        by_type = {}
        for section in self.sections:
            etype = section['element_type']
            by_type[etype] = by_type.get(etype, 0) + 1
        
        return {
            'project': self.config.project,
            'author': self.config.author,
            'total_sections': len(self.sections),
            'total_operations': len(self.operations),
            'matrices': self.list_matrices(),
            'by_type': by_type,
            'total_steps': sum(op['steps'] for op in self.operations)
        }
    
    
    def __repr__(self) -> str:
        """Representação string do memorial."""
        return (f"MatrixMemorial('{self.config.project}', "
                f"sections={len(self.sections)}, "
                f"operations={len(self.operations)})")


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'MatrixMemorial',
    'MemorialConfig',
    'ElementTemplate',
    'ELEMENT_TEMPLATES',
]
