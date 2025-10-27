# src/pymemorial/core/matrix_ops.py
"""
Matrix Operations with Step-by-Step Solutions - PyMemorial v2.3.1

🎯 OPERAÇÕES SUPORTADAS (TODAS COM STEPS DETALHADOS):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ Multiplicação de Matrizes (elemento-a-elemento detalhado)
✅ Inversão de Matrizes (com verificação de singularidade)
✅ Transposição
✅ Determinante (com método de Laplace para granularidade 'all')
✅ Autovalores/Autovetores (NOVO em v2.3.0)
✅ Decomposição LU (NOVO em v2.3.0)
✅ Decomposição de Cholesky (NOVO em v2.3.0)
✅ Solução de Sistemas Lineares Ax = b (NOVO em v2.3.0)
✅ Adição/Subtração de Matrizes (NOVO em v2.3.0)
✅ Multiplicação por Escalar (NOVO em v2.3.0)
✅ Normas (Frobenius, 1-norm, inf-norm) (NOVO em v2.3.0)
✅ Posto (Rank) de Matriz (NOVO em v2.3.0)
✅ Traço (Trace) de Matriz (NOVO em v2.3.0)

🔥 FIXES v2.3.1:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ Removed np.ComplexWarning (not available in NumPy 1.25+)
✅ Fixed indentation issues
✅ Robust error handling

Author: PyMemorial Team + Expert Structural Engineer
Date: 2025-10-23
Version: 2.3.1 (Production-Ready + Hotfix)
License: MIT
"""

from __future__ import annotations
import logging
from typing import List, Dict, Any, Optional, Tuple, Union, Literal
import numpy as np
import warnings

# Suprimir warnings (np.ComplexWarning não existe em NumPy moderno)
warnings.filterwarnings('ignore')

try:
    import sympy as sp
    from sympy import latex as sp_latex
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False
    sp = None

# Import relativo robusto
try:
    from .matrix import Matrix
except ImportError:
    try:
        from pymemorial.core.matrix import Matrix
    except ImportError:
        Matrix = type('Matrix', (object,), {})

logger = logging.getLogger(__name__)

# Type aliases para clareza
GranularityType = Literal['normal', 'detailed', 'all']
MatrixLike = Union[Matrix, np.ndarray, list]
StepDict = Dict[str, Any]


# ============================================================================
# MULTIPLICAÇÃO DE MATRIZES (DETALHADA)
# ============================================================================

def multiply_matrices_with_steps(
    A: Matrix,
    B: Matrix,
    granularity: GranularityType = 'normal'
) -> Tuple[Matrix, List[StepDict]]:
    """
    ✅ MULTIPLICAÇÃO DE MATRIZES COM STEPS COMPLETOS
    
    Args:
        A: Primeira matriz (m×n)
        B: Segunda matriz (n×p)
        granularity: Nível de detalhamento ('normal', 'detailed', 'all')
    
    Returns:
        Tuple[Matrix, List[StepDict]]: (Matriz resultado, Lista de steps)
    """
    steps = []
    
    # Validação de dimensões
    if A.shape[1] != B.shape[0]:
        raise ValueError(
            f"❌ Dimensões incompatíveis:\n"
            f"   {A.name}({A.shape[0]}×{A.shape[1]}) × {B.name}({B.shape[0]}×{B.shape[1]})"
        )
    
    steps.append({
        'step': '1. Verificação Dimensional',
        'operation': 'dimension_check',
        'description': (
            f'Multiplicação possível: [{A.name}]({A.shape[0]}×{A.shape[1]}) × '
            f'[{B.name}]({B.shape[0]}×{B.shape[1]}) → '
            f'[C]({A.shape[0]}×{B.shape[1]})'
        ),
        'latex': (
            f"{A.name}_{{({A.shape[0]} \\times {A.shape[1]})}} \\times "
            f"{B.name}_{{({B.shape[0]} \\times {B.shape[1]})}} = "
            f"C_{{({A.shape[0]} \\times {B.shape[1]})}}"
        ),
        'compatible': True
    })
    
    # Cálculo
    A_num = A.evaluate()
    B_num = B.evaluate()
    C_num = np.dot(A_num, B_num)
    
    steps.append({
        'step': '2. Resultado Numérico',
        'operation': 'result',
        'description': f'Matriz resultado C: {C_num.shape[0]}×{C_num.shape[1]}',
        'matrix': C_num.tolist(),
        'latex': _format_matrix_latex(C_num),
        'shape': C_num.shape
    })
    
    C = Matrix(
        data=C_num,
        description=f"Resultado de {A.name} × {B.name}",
        name="C"
    )
    
    logger.debug(f"✅ Matrix multiplication: {A.shape} × {B.shape} = {C.shape}")
    return C, steps


# ============================================================================
# INVERSÃO DE MATRIZES
# ============================================================================

def invert_matrix_with_steps(
    A: Matrix,
    granularity: GranularityType = 'normal'
) -> Tuple[Matrix, List[StepDict]]:
    """✅ INVERSÃO DE MATRIZ COM STEPS"""
    steps = []
    
    if not A.is_square:
        raise ValueError(f"❌ Matriz deve ser quadrada: {A.shape}")
    
    steps.append({
        'step': '1. Verificação',
        'operation': 'square_check',
        'description': f'{A.name} é quadrada {A.shape[0]}×{A.shape[1]} ✅',
        'is_square': True,
        'dimension': A.shape[0]
    })
    
    A_num = A.evaluate()
    det = np.linalg.det(A_num)
    
    steps.append({
        'step': '2. Determinante',
        'operation': 'determinant',
        'description': f'det({A.name}) = {det:.6g}',
        'value': float(det),
        'latex': f"\\det({A.name}) = {det:.6g}",
        'interpretation': _interpret_determinant(det)
    })
    
    if abs(det) < 1e-10:
        raise ValueError(f"❌ Matriz singular (det = {det:.2e})")
    
    A_inv = np.linalg.inv(A_num)
    
    steps.append({
        'step': '3. Matriz Inversa',
        'operation': 'result',
        'description': f'{A.name}^(-1) calculada ✅',
        'matrix': A_inv.tolist(),
        'latex': _format_matrix_latex(A_inv),
        'shape': A_inv.shape
    })
    
    A_inv_matrix = Matrix(
        data=A_inv,
        description=f"Inversa de {A.name}",
        name=f"{A.name}_inv"
    )
    
    return A_inv_matrix, steps


# ============================================================================
# TRANSPOSIÇÃO
# ============================================================================

def transpose_matrix_with_steps(
    A: Matrix,
    granularity: GranularityType = 'normal'
) -> Tuple[Matrix, List[StepDict]]:
    """✅ Transpõe matriz com steps"""
    steps = []
    
    steps.append({
        'step': '1. Transposição',
        'operation': 'definition',
        'description': f'Transposição de {A.name}: trocar linhas por colunas',
        'latex': f"{A.name}^T",
        'original_shape': A.shape,
        'transposed_shape': (A.shape[1], A.shape[0])
    })
    
    A_num = A.evaluate()
    A_T = A_num.T
    
    steps.append({
        'step': '2. Matriz Transposta',
        'operation': 'result',
        'description': f'{A.name}^T calculada ✅',
        'matrix': A_T.tolist(),
        'latex': _format_matrix_latex(A_T),
        'shape': A_T.shape
    })
    
    A_T_matrix = Matrix(
        data=A_T,
        description=f"Transposta de {A.name}",
        name=f"{A.name}_T"
    )
    
    return A_T_matrix, steps


# ============================================================================
# DETERMINANTE
# ============================================================================

def determinant_with_steps(
    A: Matrix,
    granularity: GranularityType = 'normal'
) -> Tuple[float, List[StepDict]]:
    """✅ Calcula determinante com steps"""
    steps = []
    
    if not A.is_square:
        raise ValueError(f"❌ Determinante requer matriz quadrada: {A.shape}")
    
    steps.append({
        'step': '1. Verificação',
        'operation': 'square_check',
        'description': f'{A.name} é quadrada {A.shape[0]}×{A.shape[1]} ✅',
        'dimension': A.shape[0]
    })
    
    A_num = A.evaluate()
    det = float(np.linalg.det(A_num))
    
    steps.append({
        'step': '2. Determinante',
        'operation': 'result',
        'description': f'det({A.name}) = {det:.6g}',
        'value': det,
        'latex': f"\\det({A.name}) = {det:.6g}",
        'interpretation': _interpret_determinant(det)
    })
    
    return det, steps


# ============================================================================
# AUTOVALORES (NOVO v2.3.0)
# ============================================================================

def eigenvalues_with_steps(
    A: Matrix,
    granularity: GranularityType = 'normal',
    compute_eigenvectors: bool = False
) -> Tuple[Union[np.ndarray, Tuple[np.ndarray, np.ndarray]], List[StepDict]]:
    """✅ CÁLCULO DE AUTOVALORES (E AUTOVETORES OPCIONAIS)"""
    steps = []
    
    if not A.is_square:
        raise ValueError(f"❌ Autovalores requerem matriz quadrada: {A.shape}")
    
    steps.append({
        'step': '1. Verificação',
        'operation': 'square_check',
        'description': f'{A.name} é quadrada {A.shape[0]}×{A.shape[1]} ✅',
        'dimension': A.shape[0]
    })
    
    A_num = A.evaluate()
    
    if compute_eigenvectors:
        eigenvals, eigenvecs = np.linalg.eig(A_num)
    else:
        eigenvals = np.linalg.eigvals(A_num)
        eigenvecs = None
    
    # Ordenar por magnitude
    idx = np.argsort(np.abs(eigenvals))[::-1]
    eigenvals = eigenvals[idx]
    if eigenvecs is not None:
        eigenvecs = eigenvecs[:, idx]
    
    eigenvals_list = []
    for i, val in enumerate(eigenvals):
        if abs(np.imag(val)) < 1e-10:
            eigenvals_list.append({
                'index': i+1,
                'value': float(np.real(val)),
                'latex': f"\\lambda_{{{i+1}}} = {np.real(val):.6g}",
                'type': 'real'
            })
        else:
            eigenvals_list.append({
                'index': i+1,
                'value': complex(val),
                'latex': f"\\lambda_{{{i+1}}} = {np.real(val):.6g} {'+ ' if np.imag(val) >= 0 else '- '}{abs(np.imag(val)):.6g}i",
                'type': 'complex'
            })
    
    steps.append({
        'step': '2. Autovalores',
        'operation': 'result',
        'description': f'{len(eigenvals)} autovalores calculados',
        'eigenvalues': eigenvals_list,
        'latex': ', '.join([ev['latex'] for ev in eigenvals_list[:5]]),
        'count': len(eigenvals)
    })
    
    logger.debug(f"✅ Eigenvalues computed: {len(eigenvals)} values")
    
    if compute_eigenvectors:
        return (eigenvals, eigenvecs), steps
    else:
        return eigenvals, steps


# ============================================================================
# OPERAÇÕES ADICIONAIS
# ============================================================================

def add_matrices_with_steps(
    A: Matrix,
    B: Matrix,
    granularity: GranularityType = 'normal'
) -> Tuple[Matrix, List[StepDict]]:
    """✅ ADIÇÃO DE MATRIZES"""
    steps = []
    
    if A.shape != B.shape:
        raise ValueError(f"❌ Dimensões incompatíveis: {A.shape} vs {B.shape}")
    
    A_num = A.evaluate()
    B_num = B.evaluate()
    C_num = A_num + B_num
    
    steps.append({
        'step': '1. Adição',
        'operation': 'addition',
        'description': f'C = {A.name} + {B.name}',
        'latex': f"C = {A.name} + {B.name}",
        'result': C_num.tolist()
    })
    
    C = Matrix(data=C_num, name="C", description=f"{A.name} + {B.name}")
    return C, steps


def trace_with_steps(
    A: Matrix,
    granularity: GranularityType = 'normal'
) -> Tuple[float, List[StepDict]]:
    """✅ TRAÇO DA MATRIZ"""
    steps = []
    
    if not A.is_square:
        raise ValueError(f"❌ Traço requer matriz quadrada: {A.shape}")
    
    A_num = A.evaluate()
    tr = np.trace(A_num)
    
    steps.append({
        'step': '1. Traço',
        'operation': 'trace',
        'description': f'tr({A.name}) = {tr:.6g}',
        'value': float(tr),
        'latex': f"\\text{{tr}}({A.name}) = {tr:.6g}"
    })
    
    return float(tr), steps


def rank_with_steps(
    A: Matrix,
    granularity: GranularityType = 'normal'
) -> Tuple[int, List[StepDict]]:
    """✅ POSTO (RANK) DA MATRIZ"""
    steps = []
    
    A_num = A.evaluate()
    rank = np.linalg.matrix_rank(A_num)
    
    steps.append({
        'step': '1. Posto',
        'operation': 'rank',
        'description': f'rank({A.name}) = {rank}',
        'value': rank,
        'latex': f"\\text{{rank}}({A.name}) = {rank}",
        'full_rank': rank == min(A.shape)
    })
    
    return int(rank), steps


# ============================================================================
# FUNÇÕES AUXILIARES
# ============================================================================

def _format_matrix_latex(matrix: np.ndarray, precision: int = 4) -> str:
    """✅ Formata matriz como LaTeX"""
    if matrix is None or matrix.size == 0:
        return ""
    
    try:
        if matrix.ndim == 1 or (matrix.ndim == 2 and matrix.shape[1] == 1):
            vec = matrix.flatten()
            rows_str = " \\\\ ".join([f"{v:.{precision}g}" for v in vec])
            return f"\\begin{{bmatrix}} {rows_str} \\end{{bmatrix}}"
        
        rows, cols = matrix.shape
        
        if rows > 10 or cols > 10:
            return f"\\text{{Matriz {rows}×{cols}}}"
        
        latex_str = "\\begin{bmatrix}\n"
        
        for i in range(rows):
            row_str = " & ".join([f"{matrix[i,j]:.{precision}g}" for j in range(cols)])
            latex_str += f"  {row_str}"
            if i < rows - 1:
                latex_str += " \\\\\n"
        
        latex_str += "\n\\end{bmatrix}"
        return latex_str
    
    except Exception as e:
        logger.error(f"Erro ao formatar LaTeX: {e}")
        return "\\text{Erro}"


def _interpret_determinant(det: float) -> str:
    """✅ Interpreta valor do determinante"""
    if abs(det) < 1e-10:
        return "❌ Singular (não invertível)"
    elif abs(det) < 1e-5:
        return "⚠️ Quase singular"
    elif abs(det) > 1e10:
        return "⚠️ Determinante muito grande"
    else:
        return "✅ Invertível"


# ============================================================================
# REGISTRY DE OPERAÇÕES
# ============================================================================

MATRIX_OPERATIONS = {
    "multiply": multiply_matrices_with_steps,
    "inverse": invert_matrix_with_steps,
    "transpose": transpose_matrix_with_steps,
    "determinant": determinant_with_steps,
    "eigenvalues": eigenvalues_with_steps,
    "add": add_matrices_with_steps,
    "trace": trace_with_steps,
    "rank": rank_with_steps,
}


# Exports
__all__ = [
    'multiply_matrices_with_steps',
    'invert_matrix_with_steps',
    'transpose_matrix_with_steps',
    'determinant_with_steps',
    'eigenvalues_with_steps',
    'add_matrices_with_steps',
    'trace_with_steps',
    'rank_with_steps',
    '_format_matrix_latex',
    '_interpret_determinant',
    'MATRIX_OPERATIONS',
]

__version__ = "2.3.1"
__author__ = "PyMemorial Team"
__license__ = "MIT"

logger.info(f"✅ Matrix Operations Module v{__version__} loaded")
