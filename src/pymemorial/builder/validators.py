"""
Validadores para entrada de dados no builder.
"""
from typing import List, Set, Tuple
import re


class ValidationError(Exception):
    """Erro de validação."""
    pass


class MemorialValidator:
    """Validador de estrutura de memorial."""
    
    @staticmethod
    def validate_variable_name(name: str) -> bool:
        """
        Valida nome de variável.
        
        Args:
            name: nome para validar
        
        Returns:
            True se válido
        
        Raises:
            ValidationError se inválido
        """
        # Deve seguir regras Python: começa com letra/underscore, contém apenas alfanuméricos/_
        pattern = r'^[a-zA-Z_][a-zA-Z0-9_]*$'
        
        if not re.match(pattern, name):
            raise ValidationError(
                f"Nome de variável inválido: '{name}'. "
                "Deve começar com letra/underscore e conter apenas letras, números e underscores."
            )
        
        # Não pode ser palavra reservada Python
        reserved = {'and', 'or', 'not', 'in', 'is', 'for', 'if', 'else', 'def', 'class', 'return'}
        if name in reserved:
            raise ValidationError(f"Nome de variável '{name}' é palavra reservada Python.")
        
        return True
    
    @staticmethod
    def validate_section_level(level: int, parent_level: int = 0) -> bool:
        """
        Valida nível de seção.
        
        Args:
            level: nível da seção
            parent_level: nível da seção pai
        
        Returns:
            True se válido
        
        Raises:
            ValidationError se inválido
        """
        if level < 1:
            raise ValidationError("Nível de seção deve ser >= 1.")
        
        if level > parent_level + 1 and parent_level > 0:
            raise ValidationError(
                f"Salto de nível inválido: de {parent_level} para {level}. "
                "Use incrementos de 1."
            )
        
        return True
    
    @staticmethod
    def check_circular_references(
        variables: dict,
        equations: list
    ) -> Tuple[bool, List[str]]:
        """
        Verifica referências circulares em equações.
        
        Args:
            variables: dicionário de variáveis
            equations: lista de equações
        
        Returns:
            (tem_erro, lista_de_ciclos)
        """
        # Análise simplificada: verificar dependências diretas
        # Implementação completa requer análise de grafo
        cycles = []
        
        # TODO: implementar detecção de ciclos com algoritmo de Tarjan
        # Por enquanto, retorna sempre válido
        return False, cycles
    
    @staticmethod
    def check_undefined_variables(
        equations: list,
        defined_vars: Set[str]
    ) -> List[str]:
        """
        Verifica variáveis não definidas em equações.
        
        Args:
            equations: lista de equações
            defined_vars: conjunto de variáveis definidas
        
        Returns:
            Lista de variáveis não definidas
        """
        undefined = []
        
        for eq in equations:
            # Extrair símbolos da equação
            symbols = eq.expression.free_symbols
            for symbol in symbols:
                if str(symbol) not in defined_vars:
                    undefined.append(str(symbol))
        
        return list(set(undefined))


def validate_template(template: str) -> Tuple[bool, List[str]]:
    """
    Valida template de texto.
    
    Args:
        template: texto com placeholders {{var}}
    
    Returns:
        (é_válido, lista_de_variáveis_requeridas)
    """
    from ..recognition.patterns import PLACEHOLDER
    
    placeholders = PLACEHOLDER.findall(template)
    
    # Verificar chaves malformadas
    malformed = re.findall(r'(?<!\{)\{(?!\{)|(?<!\})\}(?!\})', template)
    
    is_valid = len(malformed) == 0
    return is_valid, list(set(placeholders))
