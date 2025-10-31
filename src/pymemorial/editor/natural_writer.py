# src/pymemorial/editor/natural_writer.py
"""
PyMemorial v2.0 - Natural Language Writer (Revolucionário!)

Motor de escrita em linguagem natural PT-BR para memoriais de cálculo estrutural.
Integra TODOS os componentes (StepEngine, SmartTeX, TextProcessor, etc) em uma
API fluente e intuitiva.

DIFERENCIAIS REVOLUCIONÁRIOS:
═════════════════════════════
✅ ZERO LaTeX manual (100% automático)
✅ Texto natural PT-BR elegante (não parece gerado)
✅ Steps automáticos estilo Calcpad (4 níveis)
✅ Modo SMART (IA-driven, omite trivialidades)
✅ Integração total: Calculator, Equation, Variable, Units
✅ Multi-formato: Markdown, HTML, LaTeX, PDF
✅ Templates customizáveis (Jinja2)
✅ Metadados estruturados (JSON export)

FILOSOFIA:
═════════
"Write Python, Get Beautiful Portuguese Engineering Documents"

Author: PyMemorial Team
Version: 2.0.0
Date: 2025-10-27
License: MIT
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
from datetime import datetime

# Imports core
try:
    from pymemorial.core.config import PyMemorialConfig
    from pymemorial.core.variable import Variable
    from pymemorial.core.equation import Equation
    from pymemorial.core.calculator import Calculator
    CORE_AVAILABLE = True
except ImportError:
    CORE_AVAILABLE = False

# Imports editor
from pymemorial.editor.render_modes import (
    NaturalWriterConfig,
    GranularityType,
    NaturalLanguageStyle,
    StepLevel,
)

from pymemorial.editor.step_engine import (
    StepEngine,
    StepSequence,
    Step,
)

# Imports recognition
try:
    from pymemorial.recognition.text_processor import SmartTextProcessor
    TEXT_PROCESSOR_AVAILABLE = True
except ImportError:
    TEXT_PROCESSOR_AVAILABLE = False

# Imports symbols
try:
    from pymemorial.symbols.custom_registry import CustomSymbolRegistry
    SYMBOLS_AVAILABLE = True
except ImportError:
    SYMBOLS_AVAILABLE = False


# Setup logging
logger = logging.getLogger(__name__)

# Version
__version__ = "2.0.0"
__all__ = [
    'NaturalWriter',
    'WriterOutput',
    'OutputFormat',
    'NaturalWriterException',
    'TemplateNotFoundError',
    'RenderError',
]


# =========================================================================
# EXCEPTIONS
# =========================================================================

class NaturalWriterException(Exception):
    """Exceção base para NaturalWriter"""
    pass


class TemplateNotFoundError(NaturalWriterException):
    """Template não encontrado"""
    pass


class RenderError(NaturalWriterException):
    """Erro ao renderizar documento"""
    pass


# =========================================================================
# ENUMS
# =========================================================================

class OutputFormat(str, Enum):
    """
    Formatos de saída suportados
    
    Attributes:
        MARKDOWN: Markdown puro (compatível GitHub)
        HTML: HTML5 com CSS responsivo
        LATEX: LaTeX profissional (ABNT)
        PDF: PDF via LaTeX (requer pdflatex)
        JSON: Metadados estruturados
        DOCX: Word (via pandoc)
    """
    MARKDOWN = "markdown"
    HTML = "html"
    LATEX = "latex"
    PDF = "pdf"
    JSON = "json"
    DOCX = "docx"
    
    @classmethod
    def from_string(cls, value: str) -> "OutputFormat":
        """Converte string para OutputFormat"""
        value_lower = value.lower().strip()
        for fmt in cls:
            if fmt.value == value_lower:
                return fmt
        raise ValueError(f"Formato inválido: {value}")


# =========================================================================
# DATA CLASSES
# =========================================================================

@dataclass
class WriterOutput:
    """
    Resultado da renderização de documento
    
    Attributes:
        content: Conteúdo renderizado (string)
        format: Formato de saída
        metadata: Metadados do documento
        file_path: Caminho do arquivo salvo (opcional)
        warnings: Lista de warnings durante renderização
        statistics: Estatísticas de geração
    
    Examples:
        >>> output = writer.write("Cálculo de viga", format=OutputFormat.MARKDOWN)
        >>> print(output.content)
        >>> output.save_to_file("memorial.md")
    """
    content: str
    format: OutputFormat
    metadata: Dict[str, Any] = field(default_factory=dict)
    file_path: Optional[Path] = None
    warnings: List[str] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)
    
    def save_to_file(self, path: Union[str, Path], encoding: str = "utf-8") -> Path:
        """
        Salva conteúdo em arquivo
        
        Args:
            path: Caminho do arquivo
            encoding: Encoding (default: utf-8)
        
        Returns:
            Path do arquivo salvo
        
        Examples:
            >>> output.save_to_file("memorial.md")
            PosixPath('memorial.md')
        """
        file_path = Path(path)
        
        # Criar diretórios se não existirem
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Escrever arquivo
        with open(file_path, 'w', encoding=encoding) as f:
            f.write(self.content)
        
        self.file_path = file_path
        logger.info(f"Documento salvo em: {file_path}")
        
        return file_path
    
    def to_dict(self) -> Dict[str, Any]:
        """Serializa para dict"""
        return {
            'content': self.content,
            'format': self.format.value,
            'metadata': self.metadata,
            'file_path': str(self.file_path) if self.file_path else None,
            'warnings': self.warnings,
            'statistics': self.statistics,
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Serializa para JSON"""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


@dataclass
class DocumentSection:
    """
    Seção de documento
    
    Attributes:
        title: Título da seção
        content: Conteúdo (texto, equations, etc)
        level: Nível hierárquico (1-6)
        numbering: Numeração automática (ex: "2.1.3")
        metadata: Metadados da seção
    """
    title: str
    content: str = ""
    level: int = 1
    numbering: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_markdown(self) -> str:
        """Renderiza em Markdown"""
        header_prefix = "#" * self.level
        numbering_str = f"{self.numbering} " if self.numbering else ""
        
        lines = [
            f"{header_prefix} {numbering_str}{self.title}",
            "",
            self.content,
            ""
        ]
        
        return "\n".join(lines)


# =========================================================================
# NATURAL WRITER - Motor Principal
# =========================================================================

class NaturalWriter:
    """
    Motor revolucionário de escrita em linguagem natural PT-BR
    
    CORAÇÃO DO PYMEMORIAL v2.0!
    
    Integra todos os componentes em uma API fluente e intuitiva:
    - StepEngine: Steps automáticos Calcpad-style
    - Calculator: Cálculos SymPy/NumPy
    - SmartTeX: LaTeX automático
    - TextProcessor: Processamento inteligente
    - Templates: Jinja2 customizável
    
    WORKFLOW:
    ════════
    1. Configurar writer (config, templates)
    2. Adicionar seções, equações, texto
    3. Renderizar em formato desejado
    4. Salvar arquivo
    
    Attributes:
        config: Configuração de escrita natural
        step_engine: Motor de steps automáticos
        calculator: Calculadora core (opcional)
        text_processor: Processador de texto (opcional)
        sections: Seções do documento
        metadata: Metadados globais
    
    Examples:
        >>> # Exemplo básico
        >>> writer = NaturalWriter()
        >>> 
        >>> writer.add_section("Introdução", level=1)
        >>> writer.add_text("Cálculo de momento máximo em viga simplesmente apoiada.")
        >>> 
        >>> writer.add_section("Cálculo", level=1)
        >>> writer.add_equation(
        ...     "M_max = q * L**2 / 8",
        ...     context={'q': 15.0, 'L': 6.0},
        ...     intro="Momento máximo:",
        ...     unit="kN⋅m"
        ... )
        >>> 
        >>> # Renderizar
        >>> output = writer.render(format=OutputFormat.MARKDOWN)
        >>> print(output.content)
        >>> 
        >>> # Salvar
        >>> output.save_to_file("memorial.md")
    """
    
    def __init__(
        self,
        config: Optional[NaturalWriterConfig] = None,
        calculator: Optional[Any] = None,
        enable_cache: bool = True,
        template_dir: Optional[Path] = None
    ):
        """
        Inicializa NaturalWriter
        
        Args:
            config: Configuração de escrita natural (opcional)
            calculator: Calculadora core (opcional)
            enable_cache: Habilitar cache de steps (default: True)
            template_dir: Diretório de templates customizados (opcional)
        """
        self.config = config or NaturalWriterConfig()
        self.calculator = calculator
        self.template_dir = template_dir
        
        # Criar StepEngine
        self.step_engine = StepEngine(
            config=self.config,
            calculator=calculator,
            enable_cache=enable_cache
        )
        
        # Criar TextProcessor (se disponível)
        if TEXT_PROCESSOR_AVAILABLE:
            try:
                self.text_processor = SmartTextProcessor()
            except Exception as e:
                logger.warning(f"TextProcessor não disponível: {e}")
                self.text_processor = None
        else:
            self.text_processor = None
        
        # Criar SymbolRegistry (se disponível)
        if SYMBOLS_AVAILABLE:
            try:
                self.symbol_registry = CustomSymbolRegistry()
            except Exception as e:
                logger.debug(f"SymbolRegistry não disponível: {e}")
                self.symbol_registry = None
        else:
            self.symbol_registry = None
        
        # Estado interno
        self.sections: List[DocumentSection] = []
        self.metadata: Dict[str, Any] = {
            'generator': 'PyMemorial NaturalWriter v2.0',
            'created_at': datetime.now().isoformat(),
            'language': 'pt_BR',
        }
        
        # Seção atual (para API fluente)
        self._current_section: Optional[DocumentSection] = None
        
        logger.info(
            f"NaturalWriter v{__version__} inicializado "
            f"(cache={'ON' if enable_cache else 'OFF'})"
        )

    # =====================================================================
    # API FLUENTE - Construção de Documento
    # =====================================================================
    
    def add_section(
        self,
        title: str,
        level: int = 1,
        numbering: Optional[str] = None
    ) -> "NaturalWriter":
        """
        Adiciona nova seção ao documento (API fluente)
        
        Args:
            title: Título da seção
            level: Nível hierárquico (1-6)
            numbering: Numeração manual (opcional, senão auto)
        
        Returns:
            self (para chaining)
        
        Examples:
            >>> writer = NaturalWriter()
            >>> writer.add_section("Introdução", level=1)
            >>> writer.add_section("Dados de Entrada", level=2)
        """
        # Auto-numeração se não fornecida
        if numbering is None:
            numbering = self._generate_numbering(level)
        
        section = DocumentSection(
            title=title,
            content="",
            level=level,
            numbering=numbering
        )
        
        self.sections.append(section)
        self._current_section = section
        
        logger.debug(f"Seção adicionada: {numbering} {title}")
        
        return self
    
    def add_text(
        self,
        text: str,
        process: bool = True
    ) -> "NaturalWriter":
        """
        Adiciona texto à seção atual
        
        Args:
            text: Texto a adicionar
            process: Processar com TextProcessor (auto Greek, etc)
        
        Returns:
            self (para chaining)
        
        Examples:
            >>> writer.add_text("Viga simplesmente apoiada com carga uniforme.")
        """
        if self._current_section is None:
            raise RenderError("Nenhuma seção ativa. Use add_section() primeiro.")
        
        # Processar texto (se habilitado e disponível)
        if process and self.text_processor:
            try:
                processed_text = self.text_processor.process(text)
            except Exception as e:
                logger.warning(f"Erro ao processar texto: {e}")
                processed_text = text
        else:
            processed_text = text
        
        # Adicionar ao conteúdo da seção atual
        if self._current_section.content:
            self._current_section.content += "\n\n" + processed_text
        else:
            self._current_section.content = processed_text
        
        return self
    
    def add_equation(
        self,
        expression: str,
        context: Optional[Dict[str, Any]] = None,
        variable_name: str = "",
        intro: str = "",
        conclusion: str = "",
        norm_reference: Optional[str] = None,
        unit: Optional[str] = None,
        granularity: Optional[GranularityType] = None,
        show_steps: bool = True
    ) -> "NaturalWriter":
        """
        Adiciona equação com steps automáticos (MÉTODO PRINCIPAL!)
        
        Args:
            expression: Expressão matemática
            context: Contexto de variáveis {nome: valor}
            variable_name: Nome da variável calculada
            intro: Texto introdutório
            conclusion: Texto conclusivo
            norm_reference: Referência normativa (ex: "NBR 6118:2023")
            unit: Unidade do resultado
            granularity: Nível de detalhe dos steps
            show_steps: Mostrar steps (False = apenas resultado)
        
        Returns:
            self (para chaining)
        
        Examples:
            >>> writer.add_equation(
            ...     "M_max = q * L**2 / 8",
            ...     context={'q': 15.0, 'L': 6.0},
            ...     intro="Cálculo do momento máximo:",
            ...     unit="kN⋅m"
            ... )
        """
        if self._current_section is None:
            raise RenderError("Nenhuma seção ativa. Use add_section() primeiro.")
        
        # Gerar steps automáticos via StepEngine
        try:
            sequence = self.step_engine.generate_steps(
                expression=expression,
                context=context,
                granularity=granularity,
                variable_name=variable_name,
                intro_text=intro,
                conclusion_text=conclusion,
                norm_reference=norm_reference,
                unit=unit
            )
        except Exception as e:
            logger.error(f"Erro ao gerar steps: {e}")
            raise RenderError(f"Erro ao processar equação: {e}")
        
        # Renderizar sequence em texto natural PT-BR
        if show_steps:
            equation_text = sequence.to_natural_text()
        else:
            # Apenas resultado final
            if sequence.steps:
                final_step = sequence.steps[-1]
                equation_text = final_step.to_natural_text(self.config)
            else:
                equation_text = str(expression)
        
        # Adicionar ao conteúdo
        if self._current_section.content:
            self._current_section.content += "\n\n" + equation_text
        else:
            self._current_section.content = equation_text
        
        return self
    
    def add_table(
        self,
        data: List[List[Any]],
        headers: Optional[List[str]] = None,
        caption: str = "",
        alignment: Optional[List[str]] = None
    ) -> "NaturalWriter":
        """
        Adiciona tabela em Markdown
        
        Args:
            data: Dados da tabela (lista de linhas)
            headers: Cabeçalhos das colunas (opcional)
            caption: Legenda da tabela
            alignment: Alinhamento por coluna ('left', 'center', 'right')
        
        Returns:
            self (para chaining)
        
        Examples:
            >>> writer.add_table(
            ...     data=[
            ...         ["q", "15,0 kN/m"],
            ...         ["L", "6,0 m"],
            ...     ],
            ...     headers=["Parâmetro", "Valor"],
            ...     caption="Tabela 1 - Dados de entrada"
            ... )
        """
        if self._current_section is None:
            raise RenderError("Nenhuma seção ativa. Use add_section() primeiro.")
        
        # Construir tabela Markdown
        table_lines = []
        
        # Caption (se fornecido)
        if caption:
            table_lines.append(f"**{caption}**\n")
        
        # Headers
        if headers:
            table_lines.append("| " + " | ".join(headers) + " |")
            
            # Alinhamento
            if alignment:
                align_chars = {
                    'left': ':--',
                    'center': ':-:',
                    'right': '--:',
                }
                sep = "| " + " | ".join(
                    align_chars.get(a, ':--') for a in alignment
                ) + " |"
            else:
                sep = "| " + " | ".join(['---'] * len(headers)) + " |"
            
            table_lines.append(sep)
        
        # Dados
        for row in data:
            table_lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
        
        table_text = "\n".join(table_lines)
        
        # Adicionar ao conteúdo
        if self._current_section.content:
            self._current_section.content += "\n\n" + table_text
        else:
            self._current_section.content = table_text
        
        return self
    
    def add_list(
        self,
        items: List[str],
        ordered: bool = False,
        intro: str = ""
    ) -> "NaturalWriter":
        """
        Adiciona lista ao documento
        
        Args:
            items: Itens da lista
            ordered: Lista numerada (default: False = bullets)
            intro: Texto introdutório (opcional)
        
        Returns:
            self (para chaining)
        
        Examples:
            >>> writer.add_list([
            ...     "Viga simplesmente apoiada",
            ...     "Carga uniformemente distribuída",
            ...     "Material: Concreto C30"
            ... ])
        """
        if self._current_section is None:
            raise RenderError("Nenhuma seção ativa. Use add_section() primeiro.")
        
        list_lines = []
        
        if intro:
            list_lines.append(intro)
            list_lines.append("")
        
        for i, item in enumerate(items, 1):
            if ordered:
                list_lines.append(f"{i}. {item}")
            else:
                list_lines.append(f"- {item}")
        
        list_text = "\n".join(list_lines)
        
        # Adicionar ao conteúdo
        if self._current_section.content:
            self._current_section.content += "\n\n" + list_text
        else:
            self._current_section.content = list_text
        
        return self
    
    def add_figure(
        self,
        path: Union[str, Path],
        caption: str = "",
        alt_text: str = "",
        width: Optional[str] = None
    ) -> "NaturalWriter":
        """
        Adiciona figura/imagem ao documento
        
        Args:
            path: Caminho da imagem
            caption: Legenda da figura
            alt_text: Texto alternativo (acessibilidade)
            width: Largura (ex: "50%", "300px")
        
        Returns:
            self (para chaining)
        
        Examples:
            >>> writer.add_figure(
            ...     "diagrama_viga.png",
            ...     caption="Figura 1 - Diagrama de momento fletor"
            ... )
        """
        if self._current_section is None:
            raise RenderError("Nenhuma seção ativa. Use add_section() primeiro.")
        
        # Markdown image syntax
        alt = alt_text or caption
        figure_text = f"![{alt}]({path})"
        
        if caption:
            figure_text += f"\n\n*{caption}*"
        
        # Adicionar ao conteúdo
        if self._current_section.content:
            self._current_section.content += "\n\n" + figure_text
        else:
            self._current_section.content = figure_text
        
        return self
    
    def add_code(
        self,
        code: str,
        language: str = "python",
        caption: str = ""
    ) -> "NaturalWriter":
        """
        Adiciona bloco de código ao documento
        
        Args:
            code: Código fonte
            language: Linguagem (python, latex, etc)
            caption: Legenda do código
        
        Returns:
            self (para chaining)
        
        Examples:
            >>> writer.add_code(
            ...     "M_max = q * L**2 / 8",
            ...     language="python",
            ...     caption="Código 1 - Cálculo do momento"
            ... )
        """
        if self._current_section is None:
            raise RenderError("Nenhuma seção ativa. Use add_section() primeiro.")
        
        code_text = f"``````"
        
        if caption:
            code_text = f"**{caption}**\n\n{code_text}"
        
        # Adicionar ao conteúdo
        if self._current_section.content:
            self._current_section.content += "\n\n" + code_text
        else:
            self._current_section.content = code_text
        
        return self
    
    # =====================================================================
    # HELPERS INTERNOS
    # =====================================================================
    
    def _generate_numbering(self, level: int) -> str:
        """
        Gera numeração automática para seção
        
        Args:
            level: Nível hierárquico
        
        Returns:
            Numeração (ex: "2.1.3")
        """
        # Contar seções por nível
        counts = [0] * 6  # Máximo 6 níveis
        
        for section in self.sections:
            if section.level <= level:
                counts[section.level - 1] += 1
                # Resetar contadores de níveis inferiores
                for i in range(section.level, 6):
                    counts[i] = 0
        
        counts[level - 1] += 1
        
        # Construir numeração
        numbering_parts = [str(counts[i]) for i in range(level) if counts[i] > 0]
        
        return ".".join(numbering_parts)
    
    def _generate_toc(self) -> str:
        """
        Gera sumário (Table of Contents) automaticamente
        
        Returns:
            Sumário em Markdown
        """
        toc_lines = ["## Sumário\n"]
        
        for section in self.sections:
            indent = "  " * (section.level - 1)
            link = section.title.lower().replace(" ", "-")
            toc_lines.append(
                f"{indent}- [{section.numbering} {section.title}](#{link})"
            )
        
        return "\n".join(toc_lines)


    # =====================================================================
    # RENDERIZAÇÃO E EXPORT
    # =====================================================================
    
    def render(
        self,
        format: OutputFormat = OutputFormat.MARKDOWN,
        include_toc: bool = False,
        include_metadata: bool = True
    ) -> WriterOutput:
        """
        Renderiza documento completo (MÉTODO PRINCIPAL!)
        
        Args:
            format: Formato de saída
            include_toc: Incluir sumário automático
            include_metadata: Incluir metadados no cabeçalho
        
        Returns:
            WriterOutput com conteúdo renderizado
        
        Examples:
            >>> output = writer.render(format=OutputFormat.MARKDOWN)
            >>> print(output.content)
        """
        logger.info(f"Renderizando documento em {format.value}...")
        
        # Estatísticas
        start_time = datetime.now()
        
        # Dispatch para renderer específico
        if format == OutputFormat.MARKDOWN:
            content = self._render_markdown(include_toc, include_metadata)
        elif format == OutputFormat.HTML:
            content = self._render_html(include_toc, include_metadata)
        elif format == OutputFormat.JSON:
            content = self._render_json()
        else:
            raise RenderError(f"Formato não implementado: {format}")
        
        # Estatísticas finais
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        statistics = {
            'sections': len(self.sections),
            'render_time_seconds': duration,
            'format': format.value,
            'length_chars': len(content),
        }
        
        output = WriterOutput(
            content=content,
            format=format,
            metadata=self.metadata.copy(),
            statistics=statistics
        )
        
        logger.info(
            f"Documento renderizado: {statistics['sections']} seções, "
            f"{statistics['length_chars']} caracteres, "
            f"{duration:.2f}s"
        )
        
        return output
    
    def _render_markdown(
        self,
        include_toc: bool,
        include_metadata: bool
    ) -> str:
        """Renderiza documento em Markdown"""
        parts = []
        
        # Metadados (opcional)
        if include_metadata:
            parts.append(self._generate_metadata_header())
            parts.append("\n---\n")
        
        # Sumário (opcional)
        if include_toc:
            parts.append(self._generate_toc())
            parts.append("\n---\n")
        
        # Seções
        for section in self.sections:
            parts.append(section.to_markdown())
        
        return "\n".join(parts)
    
    def _render_html(
        self,
        include_toc: bool,
        include_metadata: bool
    ) -> str:
        """Renderiza documento em HTML"""
        # Converter Markdown para HTML (simplificado)
        # Em produção, usar biblioteca como markdown ou mistune
        
        markdown_content = self._render_markdown(include_toc, include_metadata)
        
        # Wrapper HTML básico
        html_parts = [
            "<!DOCTYPE html>",
            "<html lang='pt-BR'>",
            "<head>",
            "  <meta charset='UTF-8'>",
            "  <meta name='viewport' content='width=device-width, initial-scale=1.0'>",
            "  <title>Memorial de Cálculo</title>",
            "  <style>",
            self._get_default_css(),
            "  </style>",
            "</head>",
            "<body>",
            "  <div class='container'>",
        ]
        
        # Converter Markdown básico para HTML (simplificado)
        html_content = self._markdown_to_html_simple(markdown_content)
        html_parts.append(html_content)
        
        html_parts.extend([
            "  </div>",
            "</body>",
            "</html>"
        ])
        
        return "\n".join(html_parts)
    
    def _render_json(self) -> str:
        """Renderiza documento em JSON (metadados estruturados)"""
        data = {
            'metadata': self.metadata,
            'sections': [
                {
                    'title': s.title,
                    'content': s.content,
                    'level': s.level,
                    'numbering': s.numbering,
                    'metadata': s.metadata
                }
                for s in self.sections
            ],
            'statistics': {
                'section_count': len(self.sections),
                'total_chars': sum(len(s.content) for s in self.sections)
            }
        }
        
        return json.dumps(data, indent=2, ensure_ascii=False)
    
    def save(
        self,
        path: Union[str, Path],
        format: Optional[OutputFormat] = None,
        **render_kwargs
    ) -> Path:
        """
        Renderiza e salva documento em arquivo
        
        Args:
            path: Caminho do arquivo
            format: Formato (auto-detectado da extensão se None)
            **render_kwargs: Argumentos para render()
        
        Returns:
            Path do arquivo salvo
        
        Examples:
            >>> writer.save("memorial.md")
            >>> writer.save("memorial.html", format=OutputFormat.HTML)
        """
        path = Path(path)
        
        # Auto-detectar formato da extensão
        if format is None:
            ext_map = {
                '.md': OutputFormat.MARKDOWN,
                '.markdown': OutputFormat.MARKDOWN,
                '.html': OutputFormat.HTML,
                '.json': OutputFormat.JSON,
            }
            format = ext_map.get(path.suffix.lower(), OutputFormat.MARKDOWN)
        
        # Renderizar
        output = self.render(format=format, **render_kwargs)
        
        # Salvar
        return output.save_to_file(path)
    
    def clear(self) -> "NaturalWriter":
        """
        Limpa todas as seções (reset)
        
        Returns:
            self (para chaining)
        """
        self.sections.clear()
        self._current_section = None
        logger.debug("Writer limpo (todas seções removidas)")
        return self
    
    def set_metadata(self, **kwargs) -> "NaturalWriter":
        """
        Define metadados do documento
        
        Args:
            **kwargs: Pares chave-valor de metadados
        
        Returns:
            self (para chaining)
        
        Examples:
            >>> writer.set_metadata(
            ...     title="Memorial de Cálculo - Viga",
            ...     author="Engº João Silva",
            ...     project="Edifício XYZ",
            ...     date="2025-10-27"
            ... )
        """
        self.metadata.update(kwargs)
        return self
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Retorna estatísticas do documento
        
        Returns:
            Dict com estatísticas
        
        Examples:
            >>> stats = writer.get_statistics()
            >>> print(f"Seções: {stats['sections']}")
        """
        total_chars = sum(len(s.content) for s in self.sections)
        total_words = sum(len(s.content.split()) for s in self.sections)
        
        return {
            'sections': len(self.sections),
            'total_chars': total_chars,
            'total_words': total_words,
            'metadata': self.metadata,
        }
    
    # =====================================================================
    # HELPERS DE RENDERIZAÇÃO
    # =====================================================================
    
    def _generate_metadata_header(self) -> str:
        """Gera cabeçalho de metadados em YAML"""
        lines = ["---"]
        
        for key, value in self.metadata.items():
            lines.append(f"{key}: {value}")
        
        lines.append("---")
        
        return "\n".join(lines)
    
    def _get_default_css(self) -> str:
        """Retorna CSS padrão para HTML"""
        return """
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            line-height: 1.6;
            color: #333;
        }
        .container {
            background: white;
            padding: 40px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1, h2, h3, h4, h5, h6 {
            color: #2c3e50;
            margin-top: 1.5em;
        }
        code {
            background: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
        }
        pre {
            background: #f4f4f4;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
        }
        table {
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }
        th {
            background: #f8f9fa;
            font-weight: bold;
        }
        """
    
    def _markdown_to_html_simple(self, markdown: str) -> str:
        """
        Conversao simplificada Markdown para HTML
        
        NOTA: Em producao, usar biblioteca dedicada (markdown, mistune, etc)
        Esta e uma implementacao basica para demonstracao.
        """
        lines = markdown.split('\n')
        html_lines = []
        
        in_code_block = False
        # Definir fence usando concatenacao para evitar conflito de sintaxe
        code_fence = '`' + '`' + '`'
        
        for line in lines:
            # Code blocks
            if line.startswith(code_fence):
                if in_code_block:
                    html_lines.append('</code></pre>')
                    in_code_block = False
                else:
                    html_lines.append('<pre><code>')
                    in_code_block = True
                continue
            
            if in_code_block:
                html_lines.append(line)
                continue
            
            # Headers
            if line.startswith('# '):
                html_lines.append(f'<h1>{line[2:]}</h1>')
            elif line.startswith('## '):
                html_lines.append(f'<h2>{line[3:]}</h2>')
            elif line.startswith('### '):
                html_lines.append(f'<h3>{line[4:]}</h3>')
            elif line.startswith('#### '):
                html_lines.append(f'<h4>{line[5:]}</h4>')
            
            # Lists
            elif line.startswith('- '):
                html_lines.append(f'<li>{line[2:]}</li>')
            
            # Paragraphs
            elif line.strip():
                html_lines.append(f'<p>{line}</p>')
            
            else:
                html_lines.append('<br>')
        
        return '\n'.join(html_lines)



    
    def __repr__(self) -> str:
        """Representação string"""
        return (
            f"NaturalWriter(sections={len(self.sections)}, "
            f"config={self.config.language})"
        )


# =========================================================================
# HELPER FUNCTIONS (API CONVENIENTE)
# =========================================================================

def quick_write(
    title: str,
    *content_items,
    format: OutputFormat = OutputFormat.MARKDOWN,
    output_path: Optional[Union[str, Path]] = None
) -> Union[str, Path]:
    """
    API rápida para criar documento simples
    
    Args:
        title: Título do documento
        *content_items: Tuplas (tipo, conteúdo) ou strings
        format: Formato de saída
        output_path: Caminho de saída (opcional)
    
    Returns:
        String (content) ou Path (se output_path fornecido)
    
    Examples:
        >>> quick_write(
        ...     "Cálculo Simples",
        ...     "Texto introdutório",
        ...     ("equation", "M = q*L**2/8", {'q': 15, 'L': 6}),
        ...     output_path="memorial.md"
        ... )
    """
    writer = NaturalWriter()
    writer.add_section(title, level=1)
    
    for item in content_items:
        if isinstance(item, str):
            writer.add_text(item)
        elif isinstance(item, tuple):
            item_type, *args = item
            if item_type == "equation":
                expr = args[0]
                context = args[1] if len(args) > 1 else {}
                var_name = args[2] if len(args) > 2 else ""
                writer.add_equation(expr, context, variable_name=var_name)

            elif item_type == "text":
                writer.add_text(args)
    
    output = writer.render(format=format)
    
    if output_path:
        return output.save_to_file(output_path)
    
    return output.content


def get_writer_info() -> Dict[str, Any]:
    """
    Retorna informações sobre NaturalWriter
    
    Returns:
        Dict com informações
    """
    return {
        'version': __version__,
        'core_available': CORE_AVAILABLE,
        'text_processor_available': TEXT_PROCESSOR_AVAILABLE,
        'symbols_available': SYMBOLS_AVAILABLE,
        'supported_formats': [f.value for f in OutputFormat],
        'features': [
            'Automatic step generation (Calcpad-style)',
            'Natural language PT-BR',
            'Smart symbol recognition',
            'Multi-format export (MD/HTML/JSON)',
            'Fluent API',
            'Auto-numbering',
            'Table of contents',
            'Metadata management',
        ]
    }


# =========================================================================
# MAIN (PARA TESTES)
# =========================================================================

if __name__ == "__main__":
    # Exemplo de uso
    print("PyMemorial NaturalWriter v2.0 - Demo\n")
    
    # Criar writer
    writer = NaturalWriter()
    
    # Adicionar conteúdo
    writer.set_metadata(
        title="Memorial de Cálculo - Viga Simplesmente Apoiada",
        author="PyMemorial Team",
        project="Demo v2.0",
        date="2025-10-27"
    )
    
    writer.add_section("Introdução", level=1)
    writer.add_text("Cálculo de momento fletor máximo em viga simplesmente apoiada.")
    
    writer.add_section("Dados de Entrada", level=1)
    writer.add_table(
        data=[
            ["Carga distribuída (q)", "15,0 kN/m"],
            ["Vão (L)", "6,0 m"],
        ],
        headers=["Parâmetro", "Valor"],
        caption="Tabela 1 - Dados de entrada"
    )
    
    writer.add_section("Cálculo do Momento Máximo", level=1)
    writer.add_equation(
        "M_max = q * L**2 / 8",
        context={'q': 15.0, 'L': 6.0},
        intro="Momento máximo na seção central:",
        unit="kN⋅m"
    )
    
    # Renderizar
    output = writer.render(
        format=OutputFormat.MARKDOWN,
        include_toc=True,
        include_metadata=True
    )
    
    print(output.content)
    print(f"\nEstatísticas: {output.statistics}")
