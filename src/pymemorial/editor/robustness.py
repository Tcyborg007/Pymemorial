"""
Robustness Module - Sistema de Robustez Total

Integra todas as camadas de defesa contra erros:
1. Validação inteligente
2. Fallback automático
3. Sugestões contextuais
4. Modo de compatibilidade
5. Ajuda inline
"""

__version__ = '1.0.0'
__all__ = ['SuggestionEngine', 'get_smart_suggestion']

from .suggestion_engine import SuggestionEngine, get_smart_suggestion
