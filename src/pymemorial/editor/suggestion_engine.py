"""
Suggestion Engine - Sistema Inteligente de Sugestões

Analisa erros e fornece sugestões contextuais ao usuário.
"""

import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class SuggestionEngine:
    """Motor de sugestões contextuais para erros."""
    
    def __init__(self):
        # Banco de dados de erros conhecidos → soluções
        self.error_solutions = {
            'Operação não suportada: steps': {
                'context': '@matop',
                'problem': 'Steps não disponível em operações matriciais',
                'solutions': [
                    'Use @matop[multiply] para multiplicação de matrizes',
                    'Use @matrix[steps:detailed] para ver passos de uma matriz individual',
                    'Calcule a operação com @matop[multiply] e depois visualize com @matrix[steps]'
                ],
                'example': '@matop[multiply] K_global = T.T @ K_local @ T'
            },
            'Operação não suportada: eigenvalues': {
                'context': '@matop',
                'problem': 'Cálculo de autovalores não implementado em @matop',
                'solutions': [
                    'Use @eq com a função eigenvals() do SymPy',
                    'Calcule autovalores com: @eq lambda_vals = eigenvals(matriz)',
                    'Para análise modal completa, use biblioteca externa como scipy'
                ],
                'example': '@eq[steps:basic] lambdas = eigenvals(K_massa)'
            },
            'Matriz não encontrada': {
                'context': '@matop',
                'problem': 'Matriz referenciada não foi criada anteriormente',
                'solutions': [
                    'Verifique se a matriz foi definida com @matrix antes',
                    'Certifique-se que o nome está correto (case-sensitive)',
                    'Defina a matriz antes de usá-la em operações'
                ],
                'example': '@matrix K = [[1, 2], [3, 4]]'
            },
            'Variáveis ausentes': {
                'context': 'Avaliação numérica',
                'problem': 'Variáveis não foram definidas antes da matriz',
                'solutions': [
                    'Defina todas as variáveis antes de usar em matrizes',
                    'Use sintaxe: var = valor # unidade',
                    'Variáveis definidas com @eq ou @calc estão disponíveis automaticamente'
                ],
                'example': 'E = 21000 MPa\nI = 80000 cm4\n@matrix K = [[E*I, ...]]'
            }
        }
    
    def get_suggestion(self, error_message: str, context: str = '') -> Dict[str, any]:
        """
        Gera sugestão baseada na mensagem de erro.
        
        Args:
            error_message: Mensagem de erro original
            context: Contexto onde erro ocorreu (@matop, @matrix, etc)
        
        Returns:
            Dict com problema, soluções e exemplo
        """
        # Procurar padrão conhecido
        for pattern, info in self.error_solutions.items():
            if pattern.lower() in error_message.lower():
                return {
                    'found': True,
                    'context': info['context'],
                    'problem': info['problem'],
                    'solutions': info['solutions'],
                    'example': info['example']
                }
        
        # Sugestão genérica
        return {
            'found': False,
            'context': context,
            'problem': 'Erro não catalogado',
            'solutions': [
                'Consulte a documentação do PyMemorial',
                'Verifique a sintaxe da tag utilizada',
                'Teste com exemplo simplificado primeiro'
            ],
            'example': 'Consulte: https://pymemorial.readthedocs.io'
        }
    
    def format_suggestion_box(self, suggestion: Dict) -> str:
        """Formata sugestão como bloco Markdown."""
        if not suggestion['found']:
            return f"""
> **ℹ️ Informação**
> 
> {suggestion['problem']}
> 
> **Sugestões:**
> {chr(10).join(f'- {sol}' for sol in suggestion['solutions'])}
"""
        
        return f"""
> **💡 Como Resolver Este Problema**
> 
> **Contexto:** {suggestion['context']}
> 
> **Problema:** {suggestion['problem']}
> 
> **Soluções:**
> {chr(10).join(f'{i+1}. {sol}' for i, sol in enumerate(suggestion['solutions']))}
> 
> **Exemplo Correto:**
> ```
> {suggestion['example']}
> ```
"""


# Instância global
_suggestion_engine = SuggestionEngine()


def get_smart_suggestion(error: str, context: str = '') -> str:
    """Helper function para obter sugestão formatada."""
    suggestion = _suggestion_engine.get_suggestion(error, context)
    return _suggestion_engine.format_suggestion_box(suggestion)
