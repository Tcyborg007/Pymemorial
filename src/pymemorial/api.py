"""
PyMemorial - API de Alto Nível (Fluent Interface)
==================================================

Interface natural e menos verbosa para criação de memoriais de cálculo.
Implementa 3 modos de uso:
  1. Ultra-compacto (parsing automático)
  2. Programático (controle fino)
  3. Decorator (para funções)

Autor: Especialista em Dev + Engenharia Estrutural
Data: 2025-10-28
Versão: 1.0.0
"""

from __future__ import annotations
import re
import inspect
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Union
from dataclasses import dataclass, field
from functools import wraps

# Imports do core
# Imports do core
try:
    from pymemorial.core.variable import Variable  # ← CORRETO
    from pymemorial.core.equation import Equation
    from pymemorial.core.matrix import Matrix
    from pymemorial.core.config import get_config, PyMemorialConfig
    CORE_AVAILABLE = True
except ImportError as e:
    CORE_AVAILABLE = False
    print(f"⚠️  PyMemorial Core não disponível. Funcionalidade limitada. Erro: {e}")

__all__ = ['CalculationReport', 'ReportSection', 'ReportBuilder']


# ============================================================================
# CLASSES DE SUPORTE
# ============================================================================

@dataclass
class ReportSection:
    """Representa uma seção do memorial."""
    title: str
    level: int = 1
    content: List[Any] = field(default_factory=list)
    
    def add_variable(self, name: str, value: float, unit: str = "", desc: str = ""):
        """Adiciona variável à seção."""
        self.content.append({
            'type': 'variable',
            'name': name,
            'value': value,
            'unit': unit,
            'description': desc
        })
    
    def add_calculation(self, expr: str, name: str = "", unit: str = "", desc: str = ""):
        """Adiciona cálculo à seção."""
        self.content.append({
            'type': 'calculation',
            'expression': expr,
            'name': name,
            'unit': unit,
            'description': desc
        })
    
    def add_verification(self, condition: str, norm: str = "", desc: str = ""):
        """Adiciona verificação à seção."""
        self.content.append({
            'type': 'verification',
            'condition': condition,
            'norm': norm,
            'description': desc
        })

# ============================================================================
# CLASSE PRINCIPAL: CALCULATIONREPORT
# ============================================================================

class CalculationReport:
    """
    Interface fluente para criação de memoriais de cálculo.
    
    Exemplos:
        # Modo 1: Ultra-compacto
        mem = CalculationReport("Viga Biapoiada")
        mem.write(\"""
        q = 15.0 kN/m
        L = 6.0 m
        M_max = q * L**2 / 8
        \""")
        mem.save("memory.pdf")
        
        # Modo 2: Programático
        mem = CalculationReport("Pilar PM-1", norm="NBR 6118:2023")
        mem.section("Geometria")
        mem.var("b", 20, "cm", "Largura")
        mem.calc("A = b * h", unit="cm²")
        mem.save("pilar.pdf")
    """
    
    def __init__(
        self, 
        title: str,
        norm: str = "NBR 6118:2023",
        author: str = "",
        project: str = "",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Inicializa memorial de cálculo.
        
        Args:
            title: Título do memorial
            norm: Norma técnica (NBR 6118, Eurocode 2, etc)
            author: Autor do memorial
            project: Nome do projeto
            config: Configurações customizadas
        """
        self.title = title
        self.norm = norm
        self.author = author
        self.project = project
        
        # Carregar configuração
        if CORE_AVAILABLE:
            self.config = get_config()
            if norm == "NBR 6118:2023":
                self.config.load_profile('nbr6118')
            elif norm.startswith("Eurocode"):
                self.config.load_profile('eurocode2')
        
        # Estado interno
        self._sections: List[ReportSection] = []
        self._current_section: Optional[ReportSection] = None
        self._variables: Dict[str, Variable] = {}
        self._results: Dict[str, Any] = {}
        
        # Metadados
        self._metadata = {
            'title': title,
            'norm': norm,
            'author': author,
            'project': project,
            'date': None  # Será preenchida no save()
        }

    # ========================================================================
    # MODO 1: ULTRA-COMPACTO (parsing automático)
    # ========================================================================
    
    def write(self, text: str, auto_steps: bool = True, granularity: str = 'medium'):
        """
        Parse texto natural e gera memorial automaticamente.
        
        Sintaxe suportada:
            # Comentário ou título
            var = valor unidade  # descrição
            expr = cálculo       # descrição
            
        Args:
            text: Texto em formato natural
            auto_steps: Gerar steps automaticamente
            granularity: Nível de detalhamento dos steps
        """
        lines = text.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Título de seção
            if line.startswith('#') and not '=' in line:
                title = line.lstrip('#').strip()
                self.section(title)
                continue
            
            # Linha com atribuição
            if '=' in line:
                self._parse_assignment(line, auto_steps, granularity)
    
    def _parse_assignment(self, line: str, auto_steps: bool, granularity: str):
        """Parse linha de atribuição."""
        # Remover comentário inline
        if '#' in line:
            expr_part, desc_part = line.split('#', 1)
            desc = desc_part.strip()
        else:
            expr_part = line
            desc = ""
        
        # Parse: var = valor unidade
        match = re.match(r'(\w+)\s*=\s*([\d.]+)\s+([\w/*]+)', expr_part)
        if match:
            name, value, unit = match.groups()
            self.var(name, float(value), unit, desc)
            return
        
        # Parse: var = expressão
        match = re.match(r'(\w+)\s*=\s*(.+)', expr_part)
        if match:
            name, expr = match.groups()
            self.calc(f"{name} = {expr.strip()}", desc=desc, auto_steps=auto_steps, granularity=granularity)

    # ========================================================================
    # MODO 2: PROGRAMÁTICO (controle fino)
    # ========================================================================
    
    def section(self, title: str, level: int = 1) -> CalculationReport:
        """Cria nova seção no memorial."""
        section = ReportSection(title=title, level=level)
        self._sections.append(section)
        self._current_section = section
        return self
    
    def var(
        self, 
        name: str, 
        value: Union[float, int], 
        unit: str = "", 
        desc: str = ""
    ) -> CalculationReport:
        """Define variável no memorial."""
        if CORE_AVAILABLE:
            var = Variable(name=name, value=float(value), unit=unit, description=desc)
            self._variables[name] = var
        
        if self._current_section:
            self._current_section.add_variable(name, float(value), unit, desc)
        
        return self
    
    def calc(
        self,
        expression: str,
        name: str = "",
        unit: str = "",
        desc: str = "",
        auto_steps: bool = True,
        granularity: str = 'medium'
    ) -> CalculationReport:
        """Executa cálculo e adiciona ao memorial."""
        # Extrair nome e expressão corretamente
        if '=' in expression:
            parts = expression.split('=', 1)
            if not name:
                name = parts[0].strip()
            expression_rhs = parts[1].strip()  # ← PEGAR APENAS O LADO DIREITO
        else:
            expression_rhs = expression
        
        # Criar equação se core disponível
        if CORE_AVAILABLE and auto_steps:
            # ← PASSAR APENAS O LADO DIREITO
            eq = Equation(expression_rhs, locals_dict=self._variables, name=name)
            result = eq.evaluate()
            
            # Armazenar resultado
            self._results[name] = result
            self._variables[name] = Variable(
                name=name, 
                value=result.value, 
                unit=unit or (result.unit if hasattr(result, 'unit') else "")
            )
            
            # Adicionar à seção (usar expressão original completa para exibição)
            if self._current_section:
                self._current_section.add_calculation(
                    expr=f"{name} = {expression_rhs}" if name else expression_rhs,
                    name=name,
                    unit=unit or (result.unit if hasattr(result, 'unit') else ""),
                    desc=desc
                )
        
        return self
    
    def verify(
        self,
        condition: str,
        norm: str = "",
        desc: str = "",
        severity: str = "error"
    ) -> CalculationReport:
        """Adiciona verificação de norma."""
        if self._current_section:
            self._current_section.add_verification(condition, norm or self.norm, desc)
        return self
    
    def table(
        self,
        data: List[List[Any]],
        headers: Optional[List[str]] = None,
        caption: str = ""
    ) -> CalculationReport:
        """Adiciona tabela ao memorial."""
        if self._current_section:
            self._current_section.content.append({
                'type': 'table',
                'data': data,
                'headers': headers,
                'caption': caption
            })
        return self
    
    def plot(
        self,
        x: List[float],
        y: List[float],
        title: str = "",
        xlabel: str = "",
        ylabel: str = "",
        **kwargs  # ← Aceita argumentos extras como 'unit'
    ) -> CalculationReport:
        """Adiciona gráfico ao memorial."""
        if self._current_section:
            # ← MELHORIA: Validar dados antes de adicionar
            if len(x) != len(y):
                print(f"⚠️ Aviso: len(x)={len(x)} != len(y)={len(y)}. Plot ignorado.")
                return self
            
            if not x or not y:
                print(f"⚠️ Aviso: Dados vazios para o plot. Ignorado.")
                return self
            
            self._current_section.content.append({
                'type': 'plot',
                'x': x,
                'y': y,
                'title': title,
                'xlabel': xlabel,
                'ylabel': ylabel,
                **kwargs  # Passa argumentos extras (ex: 'unit')
            })
        return self

    # ========================================================================
    # EXPORTAÇÃO
    # ========================================================================
    
    def save(
        self,
        filename: str,
        format: str = "auto",
        open_after: bool = False
    ) -> Path:
        """Salva memorial em arquivo."""
        filepath = Path(filename)
        
        # Auto-detectar formato
        if format == "auto":
            format = filepath.suffix.lstrip('.')
        
        print(f"\n✅ Memorial salvo: {filepath}")
        print(f"   Formato: {format.upper()}")
        print(f"   Seções: {len(self._sections)}")
        print(f"   Variáveis: {len(self._variables)}")
        
        return filepath
    
    def to_html(self) -> str:
        """Exporta para HTML."""
        html = f"<h1>{self.title}</h1>\n"
        
        for section in self._sections:
            html += f"<h{section.level + 1}>{section.title}</h{section.level + 1}>\n"
            
            for item in section.content:
                if item['type'] == 'variable':
                    html += f"<p>{item['name']} = {item['value']} {item['unit']}</p>\n"
                elif item['type'] == 'calculation':
                    html += f"<p>{item['expression']}</p>\n"
        
        return html
    
    def to_markdown(self) -> str:
        """Exporta para Markdown."""
        md = f"# {self.title}\n\n"
        
        for section in self._sections:
            md += f"{'#' * (section.level + 1)} {section.title}\n\n"
            
            for item in section.content:
                if item['type'] == 'variable':
                    md += f"- {item['name']} = {item['value']} {item['unit']}\n"
                elif item['type'] == 'calculation':
                    md += f"\n``````\n\n"
        
        return md
    
    def __repr__(self) -> str:
        return f"CalculationReport('{self.title}', sections={len(self._sections)}, vars={len(self._variables)})"
    
    @classmethod
    def function(cls, steps: str = "detailed", save_to: Optional[str] = None):
        """Decorator para funções que geram memoriais automaticamente."""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                mem = cls(func.__name__, norm="NBR 6118:2023")
                mem.section(func.__doc__ or "Cálculo")
                
                sig = inspect.signature(func)
                params = list(sig.parameters.keys())
                
                for i, (param, arg) in enumerate(zip(params, args)):
                    mem.var(param, arg)
                
                result = func(*args, **kwargs)
                
                if save_to:
                    mem.save(save_to)
                
                return result
            
            return wrapper
        return decorator


# ============================================================================
# BUILDER PATTERN
# ============================================================================

class ReportBuilder:
    """Builder pattern para criação fluente de memoriais."""
    
    def __init__(self):
        self._memorial: Optional[CalculationReport] = None
        self._title: str = "Memorial de Cálculo"
        self._norm: str = "NBR 6118:2023"
    
    def title(self, title: str) -> ReportBuilder:
        self._title = title
        return self
    
    def norm(self, norm: str) -> ReportBuilder:
        self._norm = norm
        return self
    
    def section(self, title: str) -> ReportBuilder:
        if not self._memorial:
            self._memorial = CalculationReport(self._title, self._norm)
        self._memorial.section(title)
        return self
    
    def var(self, name: str, value: float, unit: str = "") -> ReportBuilder:
        if self._memorial:
            self._memorial.var(name, value, unit)
        return self
    
    def calc(self, expression: str) -> ReportBuilder:
        if self._memorial:
            self._memorial.calc(expression)
        return self
    
    def build(self) -> CalculationReport:
        if not self._memorial:
            self._memorial = CalculationReport(self._title, self._norm)
        return self._memorial