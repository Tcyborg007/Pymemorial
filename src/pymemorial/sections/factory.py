"""
Factory para criação de seções transversais.
"""
from typing import Optional
from .base import SectionAnalyzer


class SectionFactory:
    """
    Factory para criação de seções transversais.
    
    Detecta automaticamente qual biblioteca usar baseado no tipo.
    """
    
    @staticmethod
    def create(
        section_type: str,
        name: str,
        **kwargs
    ) -> SectionAnalyzer:
        """
        Cria seção baseada no tipo.
        
        Args:
            section_type: Tipo da seção ('steel', 'concrete', 'composite')
            name: Nome da seção
            **kwargs: Parâmetros específicos do tipo
        
        Returns:
            SectionAnalyzer apropriado
        
        Raises:
            ValueError: Se tipo não suportado
            ImportError: Se biblioteca necessária não instalada
        
        Examples:
            >>> # Seção de aço
            >>> steel = SectionFactory.create('steel', 'IPE200')
            >>> 
            >>> # Seção de concreto
            >>> concrete = SectionFactory.create('concrete', 'Viga-V1')
        """
        section_type_lower = section_type.lower()
        
        if section_type_lower in ['steel', 'steel_i', 'steel_rect', 'steel_tube']:
            from .steel import SteelSection
            return SteelSection(name=name, **kwargs)
        
        elif section_type_lower in ['concrete', 'concrete_rect', 'rc']:
            from .concrete import ConcreteSection
            return ConcreteSection(name=name, **kwargs)
        
        elif section_type_lower in ['composite', 'steel_concrete']:
            from .composite import CompositeSection
            return CompositeSection(name=name, **kwargs)
        
        else:
            raise ValueError(
                f"Tipo de seção '{section_type}' não suportado. "
                f"Tipos válidos: 'steel', 'concrete', 'composite'"
            )
    
    @staticmethod
    def available_analyzers() -> dict[str, bool]:
        """
        Lista analyzers disponíveis (bibliotecas instaladas).
        
        Returns:
            Dict com disponibilidade de cada analyzer
        
        Example:
            >>> SectionFactory.available_analyzers()
            {'steel': True, 'concrete': True, 'composite': False}
        """
        available = {}
        
        # Verificar SteelSection (sectionproperties)
        try:
            import sectionproperties
            available['steel'] = True
        except ImportError:
            available['steel'] = False
        
        # Verificar ConcreteSection (concreteproperties)
        try:
            import concreteproperties
            available['concrete'] = True
        except ImportError:
            available['concrete'] = False
        
        # Composite depende de ambas
        available['composite'] = available['steel'] and available['concrete']
        
        return available
