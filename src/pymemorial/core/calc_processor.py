# src/pymemorial/core/calc_processor.py
"""
Natural Language Calculation Processor - PyMemorial v2.0

Este módulo é o CORAÇÃO do PyMemorial para escrita natural.
Processa texto com comandos especiais (#eq, #calc, #steps, #for, #plot)
e integra com Variable, Calculator, Equation e StepEngine.

Filosofia:
- ZERO LaTeX manual do usuário
- Escrita 100% natural
- Detecção automática de variáveis
- Steps automáticos tipo Calcpad
- Integração total com core/

Author: PyMemorial Team
Date: October 2025
Version: 2.0.0
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path
import io
from contextlib import redirect_stdout

# Imports do core (existentes)
try:
    from pymemorial.core.variable import Variable
    from pymemorial.core.calculator import Calculator
    from pymemorial.core.equation import Equation
    CORE_AVAILABLE = True
except ImportError:
    CORE_AVAILABLE = False
    logging.warning("Core modules não disponíveis - modo fallback")

# Imports do editor (criados hoje)
try:
    from pymemorial.editor.step_engine import HybridStepEngine
    from pymemorial.editor.render_modes import RenderMode
    STEP_ENGINE_AVAILABLE = True
except ImportError:
    STEP_ENGINE_AVAILABLE = False
    logging.warning("StepEngine não disponível")

# Imports para plots
try:
    import matplotlib.pyplot as plt
    import numpy as np
    PLOT_AVAILABLE = True
except ImportError:
    PLOT_AVAILABLE = False
    logging.warning("Matplotlib não disponível - plots desabilitados")

# Logging
logger = logging.getLogger(__name__)

__version__ = "2.0.0"


# ============================================================================
# DATACLASSES
# ============================================================================

@dataclass
class ProcessingContext:
    """Contexto global de processamento."""
    
    variables: Dict[str, float] = field(default_factory=dict)
    """Variáveis calculadas (nome → valor)"""
    
    units: Dict[str, str] = field(default_factory=dict)
    """Unidades das variáveis (nome → unidade)"""
    
    equations: Dict[str, str] = field(default_factory=dict)
    """Equações definidas (nome → expressão)"""
    
    figures: List[str] = field(default_factory=list)
    """Lista de caminhos de figuras geradas"""
    
    tables: List[str] = field(default_factory=list)
    """Lista de tabelas HTML geradas"""
    
    equation_counter: int = 0
    """Contador de equações numeradas"""
    
    figure_counter: int = 0
    """Contador de figuras numeradas"""
    
    table_counter: int = 0
    """Contador de tabelas numeradas"""
    
    labels: Dict[str, str] = field(default_factory=dict)
    """Labels para referências cruzadas (label → número)"""


@dataclass
class ProcessedBlock:
    """Bloco processado."""
    
    original: str
    """Texto original"""
    
    processed: str
    """Texto processado (HTML/LaTeX)"""
    
    block_type: str
    """Tipo: 'text', 'eq', 'calc', 'steps', 'for', 'plot', 'table'"""
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    """Metadados adicionais"""


# ============================================================================
# CALC PROCESSOR (Classe Principal)
# ============================================================================

class CalcProcessor:
    """
    Processador de texto natural com comandos de cálculo.
    
    Suporta:
    - Detecção automática de variáveis (E = 200e9 # Pa)
    - #eq: para equações formatadas
    - #calc: para calcular e armazenar
    - #steps[mode]: para steps detalhados
    - #for ... #end: loops
    - #plot: ... #end: gráficos
    - {variavel} para substituição no texto
    - #ref:label para referências cruzadas
    
    Examples:
        >>> processor = CalcProcessor()
        >>> texto = '''
        ... E = 200e9  # Pa
        ... I = 8.33e-6  # m^4
        ... 
        ... #calc: w_max = 5*q*L**4/(384*E*I)
        ... 
        ... Resultado: {w_max*1000:.2f} mm
        ... '''
        >>> resultado = processor.process(texto)
    """
    
    def __init__(self):
        """Inicializa o processador."""
        self.context = ProcessingContext()
        
        # Inicializar componentes core
        if CORE_AVAILABLE:
            self.calculator = Calculator()
            logger.info("Calculator inicializado")
        else:
            self.calculator = None
            logger.warning("Calculator não disponível")
        
        # Inicializar step engine
        if STEP_ENGINE_AVAILABLE:
            self.step_engine = HybridStepEngine()
            logger.info("StepEngine inicializado")
        else:
            self.step_engine = None
            logger.warning("StepEngine não disponível")
        
        # Regex patterns
        self._compile_patterns()
        
        logger.info(f"CalcProcessor v{__version__} inicializado")
    
    def _compile_patterns(self):
        """Compila regex patterns."""
        
        # Variável: nome = valor  # unidade - comentário
        # ACEITA: espaços entre valor e #, comentário opcional, unidade opcional
        self.var_pattern = re.compile(
            r'^([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*([^#\n]+?)(?:\s*#\s*(.*))?$',
            re.MULTILINE
        )
        
        # Equação com label: #eq[label]: expr
        self.eq_pattern = re.compile(
            r'#eq(?:\[label=([^\]]+)\])?:\s*(.+)',
            re.IGNORECASE
        )
        
        # Cálculo inline: #calc: var = expr
        self.calc_pattern = re.compile(
            r'#calc:\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(.+)',
            re.IGNORECASE
        )
        
        # Steps: #steps[mode=...]: expr
        self.steps_pattern = re.compile(
            r'#steps(?:\[mode=([^\]]+)\])?:\s*(.+)',
            re.IGNORECASE
        )
        
        # Loop: #for ... : ... #end
        self.for_pattern = re.compile(
            r'#for\s+(.+?):\s*\n((?:.*\n)*?)#end',
            re.IGNORECASE | re.MULTILINE
        )
        
        # Plot: #plot: ... #end
        self.plot_pattern = re.compile(
            r'#plot:\s*\n((?:.*\n)*?)#end',
            re.IGNORECASE | re.MULTILINE
        )
        
        # Bloco #table[...]: ... #end
        self.table_pattern = re.compile(
            r'#table(?:\[(.+?)\])?:(.*?)#end',
            re.DOTALL
        )
        
        # Substituição {variavel}
        self.subst_pattern = re.compile(
            r'\{([a-zA-Z_][a-zA-Z0-9_]*(?:\*\d+)?(?:\.\d+)?(?::.+?)?)\}'
        )
        
        # Referência cruzada #ref:label
        # SOLUÇÃO: Adicionado ':' ao conjunto de caracteres permitidos
        self.ref_pattern = re.compile(
            r'#ref:([a-zA-Z0-9_:-]+)'
        )
    
    # ========================================================================
    # MÉTODO PRINCIPAL
    # ========================================================================
    
    def process(self, text: str) -> str:
        """
        Processa texto com comandos naturais.
        
        Args:
            text: Texto de entrada
        
        Returns:
            Texto processado (HTML/LaTeX)
        
        Examples:
            >>> proc = CalcProcessor()
            >>> result = proc.process("E = 200e9 # Pa\\n#calc: w = 1/E")
        """
        logger.info("Iniciando processamento de texto")
        
        # FASE 1: Detectar e armazenar variáveis
        text = self._process_variables(text)
        
        # FASE 2: Processar blocos especiais (ordem importante!)
        text = self._process_for_blocks(text)
        text = self._process_plot_blocks(text)
        text = self._process_table_blocks(text)
        
        # FASE 3: Processar comandos inline
        text = self._process_steps_commands(text)
        text = self._process_eq_commands(text)
        text = self._process_calc_commands(text)
        
        # FASE 4: Substituir {variavel} no texto
        text = self._substitute_variables(text)
        
        # FASE 5: Resolver referências cruzadas
        text = self._resolve_cross_references(text)
        
        logger.info("Processamento concluído")
        return text
    
    # ========================================================================
    # FASE 1: VARIÁVEIS
    # ========================================================================
    
    def _process_variables(self, text: str) -> str:
        """
        Detecta e armazena variáveis do tipo: nome = valor  # unidade - comentário
        """
        matches = list(self.var_pattern.finditer(text))
        
        for match in matches:
            var_name = match.group(1)
            var_expr = match.group(2).strip()
            comment_part = match.group(3).strip() if match.group(3) else ""
            
            # Parse do comment_part: "unidade - comentário" OU só "comentário"
            var_unit = ""
            if comment_part:
                # Tentar separar unidade do comentário pelo " - "
                if " - " in comment_part:
                    var_unit = comment_part.split(" - ")[0].strip()
                else:
                    # Sem " - ", tentar extrair só unidade (primeira palavra)
                    parts = comment_part.split()
                    if parts and re.match(r'^[a-zA-Z0-9/·^²³⁴]+$', parts[0]):
                        var_unit = parts[0]
            
            # Avaliar expressão
            try:
                value = self._evaluate_expression(var_expr)
                self.context.variables[var_name] = value
                if var_unit:
                    self.context.units[var_name] = var_unit
                
                logger.debug(f"Variável detectada: {var_name} = {value} {var_unit}")
            
            except Exception as e:
                logger.error(f"Erro ao avaliar variável {var_name}: {e}")
        
        return text
    
    # ========================================================================
    # FASE 2: BLOCOS #for, #plot, #table
    # ========================================================================
    
    def _process_for_blocks(self, text: str) -> str:
        """
        Processa blocos #for ... #end.
        
        Examples:
            #for h in [0.1, 0.2, 0.3]:
                w = 5*q*L**4/(384*E*(b*h**3/12))
                #row: | {h} | {w*1000:.2f} |
            #end
        """
        def replace_for(match):
            loop_var_expr = match.group(1).strip()
            loop_body = match.group(2)
            
            # Parse loop: "var in [values]" ou "var in range(...)"
            loop_match = re.match(r'(\w+)\s+in\s+(.+)', loop_var_expr)
            if not loop_match:
                logger.error(f"Sintaxe inválida no #for: {loop_var_expr}")
                return match.group(0)
            
            var_name = loop_match.group(1)
            iterable_expr = loop_match.group(2).strip()
            
            # CORREÇÃO CRÍTICA: Avaliar iterável com contexto GLOBAL
            try:
                # Criar namespace com todas as variáveis globais + builtins
                eval_namespace = dict(self.context.variables)
                eval_namespace['__builtins__'] = {}
                iterable = eval(iterable_expr, eval_namespace, {})
            except Exception as e:
                logger.error(f"Erro ao avaliar iterável '{iterable_expr}': {e}")
                return match.group(0)
            
            # Executar loop
            output_lines = []
            for value in iterable:
                # CORREÇÃO: Atualizar contexto GLOBAL com variável do loop
                self.context.variables[var_name] = value
                
                # Processar corpo do loop LINHA POR LINHA
                body_lines = loop_body.strip().split('\n')
                for line in body_lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Detectar e calcular variáveis dentro do loop
                    var_match = self.var_pattern.match(line)
                    if var_match:
                        local_var = var_match.group(1)
                        local_expr = var_match.group(2).strip()
                        try:
                            # CORREÇÃO: Avaliar com contexto GLOBAL completo
                            local_value = self._evaluate_expression(local_expr)
                            self.context.variables[local_var] = local_value
                            logger.debug(f"Loop var: {local_var} = {local_value}")
                        except Exception as e:
                            logger.error(f"Erro ao calcular {local_var}: {e}")
                    
                    # Processar #row:
                    elif line.startswith('#row:'):
                        row_content = line[5:].strip()
                        # Substituir variáveis com contexto GLOBAL
                        row_processed = self._substitute_variables(row_content)
                        output_lines.append(row_processed)
            
            return '\n'.join(output_lines)
        
        return self.for_pattern.sub(replace_for, text)
    
    def _process_plot_blocks(self, text: str) -> str:
        """Processa blocos #plot: ... #end."""
        if not PLOT_AVAILABLE:
            logger.warning("Matplotlib não disponível - plots ignorados")
            return self.plot_pattern.sub('[PLOT REQUER MATPLOTLIB]', text)
        
        def replace_plot(match):
            plot_body = match.group(1)
            
            # Criar nova figura
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # CORREÇÃO CRÍTICA: Preparar namespace com TODAS as variáveis globais
            plot_vars = dict(self.context.variables)
            plot_vars['np'] = np
            plot_vars['plt'] = plt
            plot_vars['ax'] = ax
            
            # Separar comandos de plot (#line, #xlabel...) do código Python
            python_code_lines = []
            plot_commands = []
            
            lines = plot_body.strip().split('\n')
            
            for line in lines:
                stripped = line.strip()
                
                if not stripped or stripped.startswith('#end'):
                    continue
                
                # Comandos de plot PyMemorial
                if stripped.startswith(('#line:', '#xlabel:', '#ylabel:', '#title:', '#legend:', '#grid:')):
                    plot_commands.append(stripped)
                
                # IGNORAR comandos #for e #end dentro do plot (são para loops PyMemorial, não Python)
                elif stripped.startswith('#for') or stripped == '#end':
                    continue
                
                # Código Python válido
                else:
                    # Remover indentação inicial (normalizar para 0)
                    python_code_lines.append(stripped)
            
            # PASSO 1: Executar TODO o código Python (incluindo loops Python nativos)
            if python_code_lines:
                # Reconstruir código com indentação correta
                python_code = self._reconstruct_python_code(python_code_lines)
                
                try:
                    # Executar código completo em um único exec
                    exec(python_code, plot_vars, plot_vars)
                    logger.debug(f"Código Python do plot executado com sucesso")
                except Exception as e:
                    logger.error(f"Erro ao executar código Python do plot:\n{python_code}\n{e}")
                    import traceback
                    traceback.print_exc()
            
            # PASSO 2: Executar comandos de plot (#line, #xlabel, etc)
            for cmd in plot_commands:
                try:
                    if cmd.startswith('#line:'):
                        args = cmd[6:].strip()
                        self._execute_plot_line(ax, args, plot_vars)
                    
                    elif cmd.startswith('#xlabel:'):
                        xlabel = cmd[8:].strip().strip('"\'')
                        ax.set_xlabel(xlabel)
                    
                    elif cmd.startswith('#ylabel:'):
                        ylabel = cmd[8:].strip().strip('"\'')
                        ax.set_ylabel(ylabel)
                    
                    elif cmd.startswith('#title:'):
                        title = cmd[7:].strip().strip('"\'')
                        ax.set_title(title)
                    
                    elif cmd.startswith('#legend:'):
                        loc = cmd[8:].strip() or 'best'
                        ax.legend(loc=loc)
                    
                    elif cmd.startswith('#grid:'):
                        ax.grid(True, alpha=0.3)
                
                except Exception as e:
                    logger.error(f"Erro ao executar comando de plot '{cmd}': {e}")
            
            # Salvar figura
            self.context.figure_counter += 1
            fig_path = f"figure_{self.context.figure_counter}.png"
            plt.tight_layout()
            plt.savefig(fig_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            self.context.figures.append(fig_path)
            
            return f'<figure>\n  <img src="{fig_path}" alt="Figura {self.context.figure_counter}">\n  <figcaption>Figura {self.context.figure_counter}</figcaption>\n</figure>'
        
        return self.plot_pattern.sub(replace_plot, text)
    
    def _reconstruct_python_code(self, lines: List[str]) -> str:
        """
        Reconstrói código Python com indentação correta.
        
        Detecta loops 'for' e adiciona indentação automática.
        """
        reconstructed = []
        indent_level = 0
        
        for line in lines:
            # Detectar início de bloco (for, if, while, def, class)
            if line.rstrip().endswith(':'):
                reconstructed.append('    ' * indent_level + line)
                indent_level += 1
            
            # Linha normal
            else:
                reconstructed.append('    ' * indent_level + line)
        
        return '\n'.join(reconstructed)
    
    def _execute_plot_line(self, ax, args: str, plot_vars: dict):
        """Executa comando #line no plot."""
        # Parse argumentos: x, y, label="...", color="..."
        # Simples: assumir x, y são os 2 primeiros
        parts = args.split(',')
        if len(parts) < 2:
            return
        
        x_name = parts[0].strip()
        y_name = parts[1].strip()
        
        x_data = plot_vars.get(x_name)
        y_data = plot_vars.get(y_name)
        
        if x_data is None or y_data is None:
            logger.error(f"Variáveis {x_name} ou {y_name} não encontradas no plot")
            return
        
        # Extrair kwargs (label, color, etc)
        kwargs = {}
        for part in parts[2:]:
            if '=' in part:
                key, value = part.split('=', 1)
                kwargs[key.strip()] = value.strip().strip('"\'')
        
        ax.plot(x_data, y_data, **kwargs)
    
    def _process_table_blocks(self, text: str) -> str:
        """
        Processa blocos #table[...]: ... #end.
        
        Examples:
            #table[caption="Resultados", label="tab:res"]:
            | Col1 | Col2 |
            |------|------|
            | 1    | 2    |
            #end
        """
        def replace_table(match):
            options = match.group(1) if match.group(1) else ""
            table_body = match.group(2).strip()
            
            # Parse opções (caption, label)
            caption = ""
            label = ""
            if options:
                caption_match = re.search(r'caption="([^"]+)"', options)
                label_match = re.search(r'label="([^"]+)"', options)
                if caption_match:
                    caption = caption_match.group(1)
                if label_match:
                    label = label_match.group(1)
            
            # Numerar tabela
            self.context.table_counter += 1
            table_num = self.context.table_counter
            
            # Armazenar label
            if label:
                self.context.labels[label] = f"Tabela {table_num}"
            
            # Converter Markdown table para HTML
            lines = table_body.strip().split('\n')
            html = '<table class="calculation-table">\n'
            
            if caption:
                html += f'  <caption>Tabela {table_num}: {caption}</caption>\n'
            
            for i, line in enumerate(lines):
                line = line.strip()
                if not line or line.startswith('|---'):
                    continue
                
                cols = [c.strip() for c in line.split('|')[1:-1]]
                
                if i == 0:
                    # Header
                    html += '  <thead>\n    <tr>\n'
                    for col in cols:
                        html += f'      <th>{col}</th>\n'
                    html += '    </tr>\n  </thead>\n  <tbody>\n'
                else:
                    # Dados
                    html += '    <tr>\n'
                    for col in cols:
                        html += f'      <td>{col}</td>\n'
                    html += '    </tr>\n'
            
            html += '  </tbody>\n</table>'
            
            self.context.tables.append(html)
            return html
        
        return self.table_pattern.sub(replace_table, text)
    
    # ========================================================================
    # FASE 3: COMANDOS #eq, #calc, #steps
    # ========================================================================
    
    def _process_eq_commands(self, text: str) -> str:
        """
        Processa comandos #eq: expressão.
        
        Gera equação numerada em LaTeX.
        """
        def replace_eq(match):
            label = match.group(1)
            expr = match.group(2).strip()
            
            # Numerar equação
            self.context.equation_counter += 1
            eq_num = self.context.equation_counter
            
            # Armazenar label se existir
            if label:
                self.context.labels[label] = f"Eq. ({eq_num})"
            
            # Converter para LaTeX
            latex_expr = self._to_latex(expr)
            
            # HTML com equação numerada
            html = f'<div class="equation" id="eq-{eq_num}">\n'
            html += f'  <div class="equation-content">\\[{latex_expr}\\]</div>\n'
            html += f'  <div class="equation-number">({eq_num})</div>\n'
            html += '</div>'
            
            return html
        
        return self.eq_pattern.sub(replace_eq, text)
    
    def _process_calc_commands(self, text: str) -> str:
        """
        Processa comandos #calc: expressão.
        
        Calcula e armazena resultado no contexto.
        """
        def replace_calc(match):
            var_name = match.group(1).strip()
            var_expr = match.group(2).strip()
            
            # Avaliar
            try:
                result = self._evaluate_expression(var_expr)
                self.context.variables[var_name] = result
                logger.debug(f"#calc: {var_name} = {result}")
                return f"<!-- {var_name} = {result} -->"
            
            except Exception as e:
                logger.error(f"Erro em #calc: {e}")
                return f"<!-- ERRO: {e} -->"
        
        return self.calc_pattern.sub(replace_calc, text)
    
    def _process_steps_commands(self, text: str) -> str:
        """
        Processa comandos #steps[mode]: expressão.
        
        Gera steps detalhados usando StepEngine.
        """
        if not STEP_ENGINE_AVAILABLE:
            logger.warning("StepEngine não disponível - #steps ignorado")
            return self.steps_pattern.sub('[STEPS REQUER STEP_ENGINE]', text)
        
        def replace_steps(match):
            mode = match.group(1) if match.group(1) else "smart"
            expr = match.group(2).strip()
            
            # Gerar steps
            try:
                steps = self.step_engine.generate_steps(
                    expression=expr,
                    context=self.context.variables,
                    units=self.context.units,
                    mode=mode
                )
                
                # Renderizar steps para HTML
                html = '<div class="calculation-steps">\n'
                for step in steps:
                    html += step.to_html() + '\n'
                html += '</div>'
                
                return html
            
            except Exception as e:
                logger.error(f"Erro ao gerar steps: {e}")
                return f'<div class="error">ERRO STEPS: {e}</div>'
        
        return self.steps_pattern.sub(replace_steps, text)
    
    # ========================================================================
    # FASE 4: SUBSTITUIÇÃO {variavel}
    # ========================================================================
    
    def _substitute_variables(self, text: str) -> str:
        """
        Substitui {variavel} e {variavel*1000:.2f} no texto.
        
        Suporta formatação tipo Python f-string.
        """
        def replace_var(match):
            expr = match.group(1)
            
            # Parse: variavel ou variavel*1000:.2f
            format_spec = ""
            if ':' in expr:
                expr, format_spec = expr.rsplit(':', 1)
            
            # Avaliar expressão
            try:
                result = self._evaluate_expression(expr)
                
                # Aplicar formatação
                if format_spec:
                    return f"{result:{format_spec}}"
                else:
                    return str(result)
            
            except Exception as e:
                logger.error(f"Erro ao substituir {{{expr}}}: {e}")
                return f"{{ERRO: {e}}}"
        
        return self.subst_pattern.sub(replace_var, text)
    
    # ========================================================================
    # FASE 5: REFERÊNCIAS CRUZADAS
    # ========================================================================
    
    def _resolve_cross_references(self, text: str) -> str:
        """
        Resolve referências #ref:label.
        
        Substitui por número correspondente (Tabela 3, Figura 2, etc).
        """
        def replace_ref(match):
            label = match.group(1)
            
            if label in self.context.labels:
                return self.context.labels[label]
            else:
                logger.warning(f"Label não encontrado: {label}")
                return f"??{label}??"
        
        return self.ref_pattern.sub(replace_ref, text)
    
    # ========================================================================
    # UTILIDADES
    # ========================================================================
    
    def _evaluate_expression(self, expr: str) -> float:
        """
        Avalia expressão usando Calculator.
        
        Args:
            expr: Expressão Python
        
        Returns:
            Resultado numérico
        """
        if self.calculator:
            try:
                return self.calculator.evaluate(expr, self.context.variables)
            except Exception as e:
                logger.error(f"Calculator falhou: {e}")
        
        # Fallback: eval Python nativo
        try:
            return eval(expr, {'__builtins__': {}}, self.context.variables)
        except Exception as e:
            raise RuntimeError(f"Erro ao avaliar '{expr}': {e}")
    
    def _to_latex(self, expr: str) -> str:
        """
        Converte expressão Python para LaTeX.
        
        Conversões básicas:
        - ** → ^
        - * → \cdot
        - / → \frac (quando apropriado)
        - _ → subscrito
        - Gregas (gamma → \gamma)
        """
        # Substituir operadores
        latex = expr
        latex = latex.replace('**', '^')
        latex = latex.replace('*', r' \cdot ')
        
        # Gregas - CORRIGIDO: usar replace simples ao invés de regex
        greek_map = {
            'alpha': r'\alpha', 'beta': r'\beta', 'gamma': r'\gamma',
            'delta': r'\delta', 'epsilon': r'\varepsilon', 'theta': r'\theta',
            'lambda': r'\lambda', 'mu': r'\mu', 'sigma': r'\sigma',
            'omega': r'\omega', 'phi': r'\phi', 'tau': r'\tau'
        }
        
        for greek, latex_greek in greek_map.items():
            # CORREÇÃO: usar replace ao invés de regex para evitar erro de escape
            latex = latex.replace(greek, latex_greek)
        
        # Subscrito: x_max → x_{max}
        latex = re.sub(r'([a-zA-Z0-9])_([a-zA-Z0-9]+)', r'\1_{\2}', latex)
        
        return latex


# ============================================================================
# API SIMPLIFICADA
# ============================================================================

def process_natural_text(text: str) -> str:
    """
    API simplificada para processar texto natural.
    
    Args:
        text: Texto com comandos naturais
    
    Returns:
        Texto processado (HTML)
    
    Examples:
        >>> from pymemorial.core.calc_processor import process_natural_text
        >>> result = process_natural_text('''
        ... E = 200e9  # Pa
        ... #calc: w = 1/E
        ... Resultado: {w*1e9:.2f} GPa
        ... ''')
    """
    processor = CalcProcessor()
    return processor.process(text)


# ============================================================================
# TESTES INTERNOS (DESENVOLVIMENTO)
# ============================================================================

def _test_calc_processor():
    """Testes básicos."""
    print("=" * 70)
    print("🧪 Testando CalcProcessor")
    print("=" * 70)
    
    processor = CalcProcessor()
    
    # Teste 1: Variáveis
    print("\n1. Teste de variáveis:")
    text1 = """
    E = 200e9  # Pa - Módulo de elasticidade
    I = 8.33e-6  # m^4 - Momento de inércia
    """
    processor.process(text1)
    print(f"   Variáveis: {processor.context.variables}")
    print(f"   Unidades: {processor.context.units}")
    
    # Teste 2: #calc
    print("\n2. Teste de #calc:")
    text2 = """
    L = 3.0  # m
    q = 10000  # N/m
    #calc: w_max = 5*q*L**4/(384*E*I)
    """
    result2 = processor.process(text2)
    print(f"   w_max = {processor.context.variables.get('w_max')}")
    
    # Teste 3: Substituição
    print("\n3. Teste de substituição:")
    text3 = "Deflexão máxima: {w_max*1000:.2f} mm"
    result3 = processor.process(text3)
    print(f"   Resultado: {result3}")
    
    print("\n" + "=" * 70)
    print("✅ Testes concluídos!")
    print("=" * 70)


if __name__ == "__main__":
    _test_calc_processor()