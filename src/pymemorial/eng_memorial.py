# src/pymemorial/eng_memorial.py

# CORREÇÃO: Movido para a primeira linha de código
from __future__ import annotations

# Agora o print de depuração pode vir depois
print("✅✅✅ PROVA DE VIDA: eng_memorial.py FOI CARREGADO CORRETAMENTE ✅✅✅")

"""
EngMemorial - Memorial de Cálculo Unificado (Engineering Memorial)
"""

import logging
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Literal, Callable
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps

# Imports do engine
try:
    from pymemorial.engine.context import MemorialContext, get_context, VariableScope
    from pymemorial.engine.processor import (
        UnifiedProcessor, ProcessingResult, GranularityLevel, ProcessingMode
    )
    ENGINE_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  ALERTA (eng_memorial): Falha ao importar 'engine'. {e}")
    ENGINE_AVAILABLE = False
    
    # CORREÇÃO: Definir TODOS os nomes importados como None no fallback
    MemorialContext = None
    get_context = None
    VariableScope = None
    UnifiedProcessor = None
    ProcessingResult = None
    GranularityLevel = None
    ProcessingMode = None

# Imports do core
try:
    from pymemorial.core import Variable, Equation, Calculator
    CORE_AVAILABLE = True
except ImportError:
    CORE_AVAILABLE = False

# Imports das sections
try:
    from pymemorial.sections import (
        SectionAnalyzer, ConcreteSection, SteelSection, CompositeSection
    )
    SECTIONS_AVAILABLE = True
except ImportError:
    SECTIONS_AVAILABLE = False

# Imports da visualization
try:
    from pymemorial.visualization.exporters import export_figure
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False


# ============================================================================
# LOGGING
# ============================================================================

logger = logging.getLogger(__name__)


# ============================================================================
# DATACLASSES - METADATA
# ============================================================================

@dataclass
class MemorialMetadata:
    """Metadados do memorial."""
    title: str
    author: str = ""
    company: str = ""
    project: str = ""
    date: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    revision: str = "1.0"
    norm: str = ""  # ex: "NBR 6118:2023"
    language: str = "pt-BR"
    
    def to_dict(self) -> Dict[str, str]:
        """Exporta para dicionário."""
        return {
            "title": self.title,
            "author": self.author,
            "company": self.company,
            "project": self.project,
            "date": self.date,
            "revision": self.revision,
            "norm": self.norm,
            "language": self.language
        }


@dataclass
class ContentBlock:
    """Bloco de conteúdo do memorial."""
    type: Literal["text", "calculation", "section", "verification", "figure", "table"]
    content: Any
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# ENG MEMORIAL - CLASSE PRINCIPAL
# ============================================================================

class EngMemorial:
    """
    Memorial de Cálculo Unificado - API Revolucionária.
    """
    
    def __init__(
        self,
        title: str,
        author: str = "",
        norm: str = "",
        granularity: GranularityLevel = GranularityLevel.MEDIUM,
        **kwargs
    ):
        """
        Inicializa memorial.
        """
        # Verificar dependências críticas
        if not ENGINE_AVAILABLE:
            raise RuntimeError(
                "engine não disponível. Instale com: pip install pymemorial[engine]"
            )
        
        # Metadados
        self.metadata = MemorialMetadata(
            title=title,
            author=author,
            norm=norm,
            **{k: v for k, v in kwargs.items() if k in MemorialMetadata.__annotations__}
        )
        
        # Context (singleton global)
        self._context = get_context()
        
        # Processor (unificado)
        self._processor = UnifiedProcessor(context=self._context)
        
        # Configuração
        self._granularity = granularity
        
        # Conteúdo do memorial (ordem de inserção preservada)
        self._content: List[ContentBlock] = []
        
        # Estado
        self._current_section: Optional[str] = None
    
    # ========================================================================
    # MODO 1: WRITE (TEXTO NATURAL) - API ULTRA-COMPACTA
    # ========================================================================
    
    def write(self, text: str, mode: ProcessingMode = ProcessingMode.AUTO) -> EngMemorial:
        """
        Escreve texto natural com detecção automática de código.
        """
        # Processar texto
        result = self._processor.process(
            text,
            mode=mode,
            granularity=self._granularity
        )
        
        # Adicionar ao conteúdo
        self._content.append(ContentBlock(
            type="text",
            content=result.output,
            metadata={
                "raw": text,
                "latex": result.output_latex,
                "success": result.success,
                "errors": result.errors,
                "section": self._current_section
            }
        ))
        
        # Log de erros
        if result.errors:
            for error in result.errors:
                logger.error(f"Erro no processamento: {error}")
        
        return self
    
    # ========================================================================
    # MODO 2: CALC (PROGRAMÁTICO) - API FLUENTE
    # ========================================================================
    
    def var(
        self,
        name: str,
        value: Union[float, int],
        unit: str = "",
        description: str = ""
    ) -> EngMemorial:
        """
        Define variável de entrada.
        """
        print(f"🔍 DEBUG var(): Adicionando variável '{name}' = {value} {unit}")
        
        # CORREÇÃO CRÍTICA: Garantir que a variável seja adicionada ao contexto
        try:
            # Tentar adicionar via contexto
            var_obj = self._context.set(name, value, unit, description)
            print(f"✅ DEBUG var(): Variável '{name}' adicionada via context.set()")
        except Exception as e:
            print(f"❌ DEBUG var(): Erro no context.set(): {e}")
            # Fallback: adicionar diretamente no escopo
            try:
                self._context._current_scope.variables[name] = value
                print(f"✅ DEBUG var(): Variável '{name}' adicionada diretamente no escopo")
            except Exception as e2:
                print(f"❌ DEBUG var(): Erro crítico: {e2}")
                return self
        
        # Verificar se foi adicionada
        ctx_vars = self._context.list_variables()
        print(f"🔍 DEBUG var(): Contexto após adicionar '{name}': {list(ctx_vars.keys())}")
        
        # Adicionar ao conteúdo do memorial
        self._content.append(ContentBlock(
            type="variable",
            content=f"{name} = {value} {unit}",
            metadata={
                "name": name,
                "value": value,
                "unit": unit,
                "description": description,
                "section": self._current_section
            }
        ))
        
        return self

    def calc(
        self,
        expression: str,
        unit: str = "",
        description: str = "",
        granularity: Optional[Union[str, Any]] = None
    ) -> EngMemorial:
        """
        Calcula expressão com steps automáticos.
        """
        # ESTE PRINT PROVARÁ QUE VOCÊ ESTÁ USANDO A VERSÃO CORRETA
        print(f"✅ DEBUG calc(): INICIANDO (VERSÃO CORRIGIDA): '{expression}'")
        
        # PASSO 1: Preparar contexto atual
        # Usar o list_variables (agora corrigido) para obter o contexto
        current_vars = {}
        try:
            # CORREÇÃO: Usar list_variables(include_parents=True)
            for name, var in self._context.list_variables(include_parents=True).items():
                val = var.value if hasattr(var, 'value') else var
                current_vars[name] = val
        except Exception as e:
            print(f"❌ DEBUG calc(): Erro ao montar current_vars: {e}")
        
        print(f"🔍 DEBUG calc(): Contexto para processador: {current_vars}")
        
        # PASSO 2: Processar cálculo
        gran = granularity if granularity is not None else self._granularity
        result = self._processor.process_calculation(
            expression=expression,
            context=current_vars,
            granularity=gran,
            unit=unit,
            description=description
        )
        
        print(f"🔍 DEBUG calc(): result.success = {result.success}")
        print(f"🔍 DEBUG calc(): result.variables = {result.variables}")
        
        # PASSO 3: Extrair nome da variável e valor calculado
        var_name = expression.split("=")[0].strip() if "=" in expression else "_result"
        calculated_value = None
        
        if result.success and result.variables:
            calculated_value = result.variables.get(var_name)
            if calculated_value is None:
                calculated_value = result.variables.get("_result")
        
        print(f"🔍 DEBUG calc(): var_name = '{var_name}'")
        print(f"🔍 DEBUG calc(): calculated_value = {calculated_value}")
        
        # PASSO 4: ADICIONAR AO CONTEXTO (ESTRATÉGIA ÚNICA E CORRETA)
        if calculated_value is not None:
            try:
                print(f"🎯 DEBUG calc(): Chamando self._context.set('{var_name}', {calculated_value})")
                
                # A ÚNICA ESTRATÉGIA CORRETA:
                self._context.set(var_name, calculated_value, unit, description)
                
                print(f"✅ DEBUG calc(): self._context.set() BEM-SUCEDIDO!")
            except Exception as e:
                print(f"❌ DEBUG calc(): FALHA ao chamar self._context.set(): {e}")
        else:
            print(f"❌ DEBUG calc(): calculated_value é None! Não pode adicionar ao contexto.")
        
        # Verificar se a variável foi realmente adicionada
        final_ctx_vars = self._context.list_variables()
        print(f"🔍 DEBUG calc(): Contexto FINAL: {list(final_ctx_vars.keys())}")
        
        # PASSO 5: Adicionar ao conteúdo do memorial (APENAS o bloco de cálculo)
        self._content.append(ContentBlock(
            type="calculation",
            content=result.output if result.success else f"Erro: {expression}",
            metadata={
                "expression": expression,
                "result": calculated_value,
                "unit": unit,
                "description": description,
                "latex": result.output_latex if result.success else "",
                "steps": result.steps if result.success else [],
                "section": self._current_section,
                "success": result.success
            }
        ))
        
        return self

    def _evaluate_expression_fallback(self, expression: str, context: Dict[str, Any]) -> Any:
        """Fallback para avaliar expressão quando o processor falha."""
        try:
            # Extrair RHS se tiver "="
            if "=" in expression:
                _, expr = expression.split("=", 1)
                expr = expr.strip()
            else:
                expr = expression.strip()
            
            # Criar namespace seguro
            safe_dict = {
                '__builtins__': {},
                'abs': abs, 'min': min, 'max': max, 'round': round,
                'pow': pow, 'sum': sum,
            }
            
            # Adicionar funções matemáticas
            try:
                import math
                safe_dict.update({
                    'sqrt': math.sqrt, 'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
                    'pi': math.pi, 'e': math.e, 'log': math.log, 'exp': math.exp,
                    'ceil': math.ceil, 'floor': math.floor,
                })
            except ImportError:
                pass
            
            # Adicionar contexto
            safe_dict.update(context)
            
            # Avaliar
            return eval(expr, safe_dict, {})
            
        except Exception as e:
            logger.error(f"Erro no fallback evaluation: {e}")
            return None
    
    def section(self, name: str, level: int = 1) -> EngMemorial:
        """
        Cria nova seção hierárquica.
        """
        # Atualizar seção atual
        self._current_section = name
        
        # Criar escopo no contexto
        self._context.push_scope(name)
        
        # Adicionar ao conteúdo
        header_prefix = "#" * level
        self._content.append(ContentBlock(
            type="section",
            content=f"{header_prefix} {name}",
            metadata={"name": name, "level": level}
        ))
        
        return self
    
    def verify(
        self,
        condition: str,
        norm: Optional[str] = None,
        desc: str = ""
    ) -> EngMemorial:
        """
        Adiciona verificação de norma.
        """
        # Avaliar condição
        context = {
            name: var.value
            for name, var in self._context.list_variables().items()
        }
        
        try:
            result = eval(condition, {"__builtins__": {}}, context)
            status = "✅ OK" if result else "❌ NÃO OK"
        except Exception as e:
            result = None
            status = f"⚠️ ERRO: {e}"
            logger.error(f"Erro ao avaliar condição '{condition}': {e}")
        
        # Formatar saída
        norm_str = norm or self.metadata.norm
        text = f"**Verificação ({norm_str}):** {desc}\n"
        text += f"  Condição: {condition}\n"
        text += f"  Resultado: {status}"
        
        self._content.append(ContentBlock(
            type="verification",
            content=text,
            metadata={
                "condition": condition,
                "result": result,
                "status": status,
                "norm": norm_str,
                "description": desc,
                "section": self._current_section
            }
        ))
        
        return self
    
    def figure(
        self,
        fig: Any,
        caption: str = "",
        label: str = ""
    ) -> EngMemorial:
        """
        Adiciona figura (matplotlib/plotly/pyvista).
        """
        self._content.append(ContentBlock(
            type="figure",
            content=fig,
            metadata={
                "caption": caption,
                "label": label,
                "section": self._current_section
            }
        ))
        return self
    
    def table(
        self,
        data: Union[List[List], Dict],
        caption: str = "",
        label: str = ""
    ) -> EngMemorial:
        """
        Adiciona tabela.
        """
        self._content.append(ContentBlock(
            type="table",
            content=data,
            metadata={
                "caption": caption,
                "label": label,
                "section": self._current_section
            }
        ))
        return self

    def render_pdf(self, output_path: str, **kwargs) -> Path:
        """
        Renderiza o memorial para PDF com LaTeX formatado via texto natural.
        """
        from pathlib import Path
        from jinja2 import Template
        
        try:
            from weasyprint import HTML
        except ImportError:
            raise ImportError("WeasyPrint necessário: pip install weasyprint")
        
        output_path = Path(output_path)
        
        # Template HTML com formatação matemática melhorada
        html_template = Template("""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{{ title }}</title>
    <style>
        @page { size: A4; margin: 2cm; }
        body { font-family: 'Arial', sans-serif; margin: 0; padding: 20px; line-height: 1.6; color: #2c3e50; }
        h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 12px; margin: 30px 0 20px 0; font-size: 24pt; }
        h2 { color: #34495e; border-bottom: 2px solid #95a5a6; padding-bottom: 8px; margin: 25px 0 15px 0; font-size: 18pt; }
        h3 { color: #7f8c8d; margin: 20px 0 10px 0; font-size: 14pt; }
        
        .metadata {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .metadata p { margin: 8px 0; font-size: 11pt; }
        .metadata strong { font-weight: 600; }
        
        .calculation {
            background: linear-gradient(to right, #f8f9fa, #ffffff);
            border-left: 5px solid #27ae60;
            padding: 20px;
            margin: 20px 0;
            border-radius: 4px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
            font-size: 11pt;
            line-height: 1.8;
            white-space: pre-wrap;
        }
        
        .variable {
            background: linear-gradient(to right, #fff3cd, #ffffff);
            border-left: 4px solid #f39c12;
            padding: 12px 16px;
            margin: 12px 0;
            border-radius: 4px;
            font-family: 'Consolas', monospace;
            font-size: 10pt;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }
        
        .section-title { page-break-before: auto; }
    </style>
</head>
<body>
    <h1>{{ title }}</h1>
    
    <div class="metadata">
        <p><strong>👤 Autor:</strong> {{ author }}</p>
        <p><strong>📋 Norma:</strong> {{ norm }}</p>
        <p><strong>📅 Data:</strong> {{ date }}</p>
    </div>
    
    {% for block in content_blocks %}
        {% if block.type == 'text' %}
            <p>{{ block.content }}</p>
        
        {% elif block.type == 'section' %}
            <h2 class="section-title">{{ block.content }}</h2>
        
        {% elif block.type == 'calculation' %}
            <div class="calculation">
            {% if block.metadata and block.metadata.get('natural') %}
                {# ✅ USAR SAÍDA NATURAL FORMATADA #}
                <pre style="margin: 0; font-family: 'Consolas', monospace;">{{ block.metadata.natural }}</pre>
            {% elif block.metadata and block.metadata.latex %}
                {# Fallback para LaTeX se natural não existir #}
                {% set latex_lines = block.metadata.latex.strip().split('\n') %}
                {% for line in latex_lines %}
                    {% if line.strip() %}
                        {% set clean_line = line.replace('$', '').replace('\\cdot', '×').replace('\\times', '×').strip() %}
                        {{ clean_line }}<br/>
                    {% endif %}
                {% endfor %}
            {% else %}
                {{ block.content }}
            {% endif %}
            </div>
        
        {% elif block.type == 'variable' %}
            {% if block.metadata %}
                <div class="variable">
                    <strong>{{ block.metadata.name }}</strong> = {{ block.metadata.value }} {{ block.metadata.unit }}
                    {% if block.metadata.description %}
                        <span style="color: #7f8c8d; margin-left: 10px;">({{ block.metadata.description }})</span>
                    {% endif %}
                </div>
            {% endif %}
        {% endif %}
    {% endfor %}
</body>
</html>
        """)
        
        # Renderizar HTML
        html_content = html_template.render(
            title=self.metadata.title,
            author=self.metadata.author,
            norm=self.metadata.norm,
            date=self.metadata.date,
            content_blocks=self._content
        )
        
        # Gerar PDF
        HTML(string=html_content).write_pdf(output_path)
        
        logger.info(f"Memorial renderizado: {output_path}")
        return output_path






    # ========================================================================
    # INTEGRAÇÃO COM SECTIONS
    # ========================================================================
    
    def add_section_analysis(
        self,
        section: SectionAnalyzer,
        include_figure: bool = True
    ) -> EngMemorial:
        """
        Adiciona análise de seção transversal.
        """
        if not SECTIONS_AVAILABLE:
            warnings.warn("pymemorial.sections não disponível")
            return self
        
        # Obter propriedades
        props = section.calculate_properties()
        
        # Adicionar propriedades como variáveis
        self.var("A", props.area, "m²", "Área da seção")
        self.var("I_x", props.ixx, "m⁴", "Momento de inércia em x")
        self.var("I_y", props.iyy, "m⁴", "Momento de inércia em y")
        
        # Adicionar figura se solicitado
        if include_figure:
            try:
                fig = section.plot()
                self.figure(fig, caption=f"Seção transversal - {section.name}")
            except Exception as e:
                logger.warning(f"Erro ao plotar seção: {e}")
        
        return self
    
    # ========================================================================
    # EXPORT - MÚLTIPLOS FORMATOS
    # ========================================================================
    
    def save(
        self,
        filepath: Union[str, Path],
        format: Optional[Literal["pdf", "html", "md", "latex", "json"]] = None
    ):
        """
        Salva memorial em arquivo.
        """
        filepath = Path(filepath)
        
        # Detectar formato
        if format is None:
            format = filepath.suffix[1:]  # Remove o ponto
        
        # Gerar conteúdo
        if format == "md":
            self._save_markdown(filepath)
        elif format == "html":
            self._save_html(filepath)
        elif format == "pdf":
            self._save_pdf(filepath)
        elif format == "latex":
            self._save_latex(filepath)
        elif format == "json":
            self._save_json(filepath)
        else:
            raise ValueError(f"Formato não suportado: {format}")
        
        logger.info(f"Memorial salvo em: {filepath}")
    
    def _save_markdown(self, filepath: Path):
        """Salva como Markdown."""
        lines = []
        
        # Metadados
        lines.append(f"# {self.metadata.title}")
        lines.append("")
        if self.metadata.author:
            lines.append(f"**Autor:** {self.metadata.author}")
        if self.metadata.norm:
            lines.append(f"**Norma:** {self.metadata.norm}")
        lines.append(f"**Data:** {self.metadata.date}")
        lines.append("")
        lines.append("---")
        lines.append("")
        
        # Conteúdo
        for block in self._content:
            # CORREÇÃO: Converter para string garantindo encoding
            content_str = str(block.content)
            lines.append(content_str)
            lines.append("")
        
        # CORREÇÃO: Escrever com UTF-8 explícito
        filepath.write_text("\n".join(lines), encoding="utf-8")

    
    def _save_html(self, filepath: Path):
        """Salva como HTML."""
        # Converter markdown para HTML (usar markdown library)
        try:
            import markdown
            md_content = "\n".join(block.content for block in self._content)
            html_body = markdown.markdown(md_content)
        except ImportError:
            # Fallback simples
            html_body = "<pre>\n" + "\n".join(block.content for block in self._content) + "\n</pre>"
        
        html = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <title>{self.metadata.title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; }}
        h1, h2, h3 {{ color: #333; }}
        code {{ background: #f5f5f5; padding: 2px 5px; }}
    </style>
</head>
<body>
    <h1>{self.metadata.title}</h1>
    <p><strong>Autor:</strong> {self.metadata.author}</p>
    <p><strong>Norma:</strong> {self.metadata.norm}</p>
    <p><strong>Data:</strong> {self.metadata.date}</p>
    <hr>
    {html_body}
</body>
</html>"""
        
        filepath.write_text(html, encoding="utf-8")
    
    def _save_pdf(self, filepath: Path):
        """Salva como PDF (via HTML + WeasyPrint ou pdfkit)."""
        # Primeiro gerar HTML
        html_path = filepath.with_suffix(".html")
        self._save_html(html_path)
        
        # Converter para PDF
        try:
            from weasyprint import HTML
            HTML(str(html_path)).write_pdf(str(filepath))
        except ImportError:
            warnings.warn("WeasyPrint não instalado. Salvando apenas HTML.")
            logger.info(f"Para gerar PDF, instale: pip install weasyprint")
    
    def _save_latex(self, filepath: Path):
        """Salva como LaTeX."""
        # TODO: Implementar geração LaTeX completa
        raise NotImplementedError("Export LaTeX em desenvolvimento")
    
    def _save_json(self, filepath: Path):
        """Salva metadados + conteúdo em JSON."""
        import json
        data = {
            "metadata": self.metadata.to_dict(),
            "content": [
                {
                    "type": block.type,
                    "content": str(block.content),
                    "metadata": block.metadata
                }
                for block in self._content
            ]
        }
        filepath.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    
    # ========================================================================
    # DECORATORS - MODO 3
    # ========================================================================
    
    @staticmethod
    def calculate(
        steps: Union[str, GranularityLevel] = "medium",
        save_to: Optional[Union[str, Path]] = None,
        title: Optional[str] = None
    ):
        """
        Decorator para gerar memorial automaticamente de funções.
        """
        # Converter string para enum
        if isinstance(steps, str):
            steps = GranularityLevel(steps)
        
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Criar memorial
                func_title = title or func.__name__.replace("_", " ").title()
                mem = EngMemorial(func_title, granularity=steps)
                
                # Adicionar docstring como descrição
                if func.__doc__:
                    mem.write(func.__doc__)
                
                # TODO: Capturar código da função e gerar steps
                # Por ora, apenas executa
                result = func(*args, **kwargs)
                
                # Salvar se solicitado
                if save_to:
                    mem.save(save_to)
                
                return result
            
            return wrapper
        return decorator


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "EngMemorial",
    "MemorialMetadata",
    "ContentBlock",
]