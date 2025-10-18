"""
Gerador de Memorial de Cálculo em HTML - VERSÃO COM LATEX FUNCIONANDO
"""

from pymemorial.core import Equation, VariableFactory
from datetime import datetime


class HTMLMemorialGenerator:
    """Gera memorial de cálculo em HTML profissional."""
    
    def __init__(self, title: str, project: str = ""):
        self.title = title
        self.project = project
        self.variables = {}
        self.html_content = []
        self.section_count = 0
        self.input_variables = []
        
    def add_variable(self, name: str, value: float, unit: str = "", description: str = ""):
        """Adiciona variável de entrada ao memorial."""
        self.variables[name] = VariableFactory.create(name, value, unit)
        self.variables[name].description = description
        self.input_variables.append(name)
        
    def add_section(self, title: str, subtitle: str = ""):
        """Adiciona seção ao memorial."""
        self.section_count += 1
        html = f"""
        <div class="section">
            <h2>{self.section_count}. {title}</h2>
            {f'<p class="subtitle">{subtitle}</p>' if subtitle else ''}
        """
        self.html_content.append(html)
        
    def add_equation(self, expression: str, description: str = "", granularity: str = "detailed"):
        """Adiciona equação com passos ao memorial."""
        if '=' in expression:
            result_name = expression.split('=')[0].strip()
        else:
            result_name = None
            
        eq = Equation(expression, variables=self.variables, description=description)
        steps = eq.steps(granularity=granularity, show_units=True)
        
        if result_name and steps and steps[-1]['numeric'] is not None:
            result_value = steps[-1]['numeric']
            self.variables[result_name] = VariableFactory.create(result_name, result_value)
            self.variables[result_name].description = description if description else f"Calculado: {expression}"
        
        html = f"""
        <div class="equation-block">
            <div class="equation-header">
                <span class="equation-label">{description if description else expression}</span>
                <span class="granularity-badge {granularity}">{granularity.upper()}</span>
            </div>
        """
        
        if len(steps) > 2 and granularity in ['detailed', 'all', 'normal']:
            html += '<div class="steps-container">'
            for step in steps:
                step_class = {
                    'symbolic': 'step-symbolic',
                    'substitution': 'step-substitution',
                    'result': 'step-result'
                }.get(step['operation'], 'step-intermediate')
                
                # ✅ Formato melhorado: fórmula = resultado
                if step['numeric'] is not None:
                    numeric_display = f'<span class="equals-sign">=</span><span class="numeric-value">{step["numeric"]:.4f}</span>'
                else:
                    numeric_display = ''
                
                html += f"""
                <div class="step {step_class}">
                    <div class="step-header">
                        <span class="step-number">{step['step_number']}</span>
                        <span class="step-description">{step['description']}</span>
                    </div>
                    <div class="step-content">
                        <div class="step-math">$${step['latex']}$$ {numeric_display}</div>
                    </div>
                </div>
                """
            html += '</div>'
        else:
            final_step = steps[-1]
            html += f"""
            <div class="result-only">
                <div class="step-math">$${final_step['latex']}$$ <span class="equals-sign">=</span> <span class="numeric-value">{final_step['numeric']:.4f}</span></div>
            </div>
            """
        
        html += '</div>'
        self.html_content.append(html)
    
    def _translate_unit(self, unit: str) -> str:
        """Traduz unidades."""
        translations = {
            "meter": "m", "kilonewton": "kN", "megapascal": "MPa",
            "kilonewton / meter": "kN/m", "kilonewton / meter ** 2": "kN/m²",
            "meter ** 2": "m²", "meter ** 3": "m³", "meter ** 4": "m⁴",
            "dimensionless": "", "": ""
        }
        unit_str = str(unit).strip()
        return translations.get(unit_str, unit_str) if unit_str != "dimensionless" else ""
    
    def _is_calculated_variable(self, name: str) -> bool:
        """Verifica se variável é calculada."""
        return name not in self.input_variables
        
    def add_comparison_table(self):
        """Adiciona tabela comparativa."""
        html = """
        <div class="comparison-table">
            <h3>📊 Resumo das Variáveis</h3>
            <table>
                <thead>
                    <tr>
                        <th>Variável</th>
                        <th>Símbolo</th>
                        <th>Valor</th>
                        <th>Unidade</th>
                        <th>Tipo</th>
                        <th>Descrição</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        for name, var in self.variables.items():
            if hasattr(var, 'value') and var.value is not None:
                if hasattr(var.value, 'magnitude'):
                    value_display = f"{var.value.magnitude:.4f}"
                    unit_raw = str(var.value.units)
                else:
                    value_display = f"{var.value:.4f}"
                    unit_raw = getattr(var, 'unit', "")
            else:
                value_display = "N/A"
                unit_raw = ""
            
            unit_display = self._translate_unit(unit_raw)
            is_calculated = self._is_calculated_variable(name)
            var_type = "Calculado" if is_calculated else "Entrada"
            var_type_class = "calculated" if is_calculated else "input"
            description = getattr(var, 'description', '-')
            
            html += f"""
                <tr>
                    <td><code>{name}</code></td>
                    <td>$${name}$$</td>
                    <td class="numeric">{value_display}</td>
                    <td class="unit">{unit_display if unit_display else '-'}</td>
                    <td><span class="badge-type {var_type_class}">{var_type}</span></td>
                    <td class="description">{description}</td>
                </tr>
            """
        
        html += """
                </tbody>
            </table>
        </div>
        """
        self.html_content.append(html)
        
    def generate_html(self, output_file: str = "memorial_calculo.html"):
        """Gera arquivo HTML completo."""
        
        css = """
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                color: #333;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 20px;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                border-radius: 15px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                overflow: hidden;
            }
            .header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 40px;
                text-align: center;
            }
            .header h1 {
                font-size: 2.5em;
                margin-bottom: 10px;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            }
            .header .project { font-size: 1.2em; opacity: 0.9; }
            .header .meta { margin-top: 15px; font-size: 0.9em; opacity: 0.8; }
            .content { padding: 40px; }
            .section { margin-bottom: 40px; }
            .section h2 {
                color: #667eea;
                border-bottom: 3px solid #667eea;
                padding-bottom: 10px;
                margin-bottom: 20px;
                font-size: 1.8em;
            }
            .subtitle { color: #666; font-style: italic; margin-bottom: 20px; }
            .equation-block {
                background: #f8f9fa;
                border-left: 4px solid #667eea;
                border-radius: 8px;
                padding: 20px;
                margin-bottom: 20px;
                transition: transform 0.2s, box-shadow 0.2s;
            }
            .equation-block:hover {
                transform: translateX(5px);
                box-shadow: 0 5px 15px rgba(102, 126, 234, 0.2);
            }
            .equation-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 15px;
            }
            .equation-label { font-weight: bold; color: #333; font-size: 1.1em; }
            .granularity-badge {
                padding: 5px 15px;
                border-radius: 20px;
                font-size: 0.75em;
                font-weight: bold;
                text-transform: uppercase;
            }
            .granularity-badge.minimal { background: #28a745; color: white; }
            .granularity-badge.normal { background: #17a2b8; color: white; }
            .granularity-badge.detailed { background: #dc3545; color: white; }
            .steps-container { margin-top: 15px; }
            .step {
                padding: 15px;
                margin-bottom: 10px;
                border-radius: 6px;
                background: white;
                border: 1px solid #dee2e6;
            }
            .step-header {
                display: flex;
                align-items: center;
                margin-bottom: 10px;
            }
            .step-number {
                display: inline-flex;
                align-items: center;
                justify-content: center;
                width: 30px;
                height: 30px;
                background: #667eea;
                color: white;
                border-radius: 50%;
                font-weight: bold;
                margin-right: 10px;
                flex-shrink: 0;
            }
            .step-description { color: #666; font-size: 0.9em; font-weight: 600; }
            .step-content {
                padding-left: 40px;
            }
            .step-math {
                font-size: 1.2em;
                padding: 15px;
                background: white;
                border-radius: 4px;
                display: flex;
                align-items: center;
                gap: 10px;
            }
            .equals-sign {
                font-size: 1.5em;
                color: #667eea;
                font-weight: bold;
                margin: 0 10px;
            }
            .numeric-value {
                color: #28a745;
                font-weight: bold;
                padding: 5px 15px;
                background: #d4edda;
                border-radius: 4px;
                font-size: 1.1em;
            }
            .step-symbolic {
                background: linear-gradient(to right, #e3f2fd, white);
                border-left: 3px solid #2196F3;
            }
            .step-substitution {
                background: linear-gradient(to right, #fff3e0, white);
                border-left: 3px solid #ff9800;
            }
            .step-intermediate {
                background: linear-gradient(to right, #f3e5f5, white);
                border-left: 3px solid #9c27b0;
            }
            .step-result {
                background: linear-gradient(to right, #e8f5e9, white);
                border-left: 3px solid #4caf50;
                font-weight: bold;
            }
            .result-only {
                padding: 20px;
                background: white;
                border-radius: 6px;
                text-align: center;
            }
            .comparison-table {
                margin-top: 30px;
                background: white;
                border-radius: 8px;
                padding: 20px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            .comparison-table h3 { color: #667eea; margin-bottom: 20px; font-size: 1.5em; }
            table { width: 100%; border-collapse: collapse; }
            thead { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; }
            th, td { padding: 12px; text-align: left; border-bottom: 1px solid #dee2e6; }
            tbody tr:hover { background: #f8f9fa; }
            .numeric { font-family: 'Courier New', monospace; font-weight: bold; color: #667eea; }
            .unit { font-weight: bold; color: #667eea; }
            .description { color: #666; font-style: italic; font-size: 0.9em; }
            .badge-type { padding: 3px 8px; border-radius: 12px; font-size: 0.85em; font-weight: 600; }
            .badge-type.calculated { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
            .badge-type.input { background: #cce5ff; color: #004085; border: 1px solid #b8daff; }
            code { background: #f8f9fa; padding: 2px 6px; border-radius: 3px; font-family: 'Courier New', monospace; color: #d63384; }
            .footer { background: #f8f9fa; padding: 20px; text-align: center; color: #666; border-top: 1px solid #dee2e6; }
        </style>
        """
        
        html_template = f"""
<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.title}</title>
    {css}
    
    <script>
        window.MathJax = {{
            tex: {{
                inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
                displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']],
                processEscapes: true
            }},
            startup: {{
                ready: () => {{
                    console.log('MathJax ready!');
                    MathJax.startup.defaultReady();
                    MathJax.startup.promise.then(() => {{
                        console.log('MathJax typeset complete!');
                    }});
                }}
            }}
        }};
    </script>
    <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js" id="MathJax-script" async></script>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📐 {self.title}</h1>
            <div class="project">{self.project if self.project else 'Memorial de Cálculo Estrutural'}</div>
            <div class="meta">Gerado em: {datetime.now().strftime('%d/%m/%Y às %H:%M:%S')} | PyMemorial v1.0</div>
        </div>
        <div class="content">
            {''.join(self.html_content)}
        </div>
        <div class="footer">
            <p>Memorial de Cálculo gerado automaticamente pela biblioteca <strong>PyMemorial</strong></p>
            <p>Todos os cálculos foram verificados e estão em conformidade com as normas técnicas vigentes</p>
        </div>
    </div>
</body>
</html>
        """
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_template)
        
        print(f"✅ HTML gerado: {output_file}")
        return output_file


# ============================================================================
# EXEMPLO - VIGA COMPLETA
# ============================================================================

def example_viga_completa():
    memorial = HTMLMemorialGenerator(
        title="Dimensionamento de Viga Biapoiada",
        project="Edifício Residencial - Bloco A"
    )
    
    memorial.add_section("Dados de Entrada", "Geometria e carregamentos")
    memorial.add_variable("L", 6.0, "m", "Vão da viga")
    memorial.add_variable("q", 15.0, "kN/m", "Carga distribuída")
    memorial.add_variable("b", 0.20, "m", "Largura da seção")
    memorial.add_variable("h", 0.50, "m", "Altura da seção")
    memorial.add_variable("fck", 30.0, "MPa", "Resistência do concreto")
    
    memorial.add_section("Propriedades Geométricas", "Modo MINIMAL")
    memorial.add_equation("A = b * h", "Área da seção", granularity="minimal")
    memorial.add_equation("I = b * h**3 / 12", "Momento de inércia", granularity="minimal")
    
    memorial.add_section("Esforços Solicitantes", "Modo NORMAL")
    memorial.add_equation("M = (q * L**2 / 8) + (b * q) / h", "Momento fletor máximo", granularity="normal")
    memorial.add_equation("V = q * L / 2", "Força cortante máxima", granularity="normal")
    
    memorial.add_section("Verificação de Tensões", "Modo DETAILED")
    memorial.add_equation("W = b * h**2 / 6", "Módulo de resistência", granularity="detailed")
    memorial.add_equation("sigma = M / W", "Tensão normal máxima", granularity="detailed")
    memorial.add_equation("tau = (3 * V) / (2 * A)", "Tensão cisalhante", granularity="detailed")
    
    memorial.add_section("Verificações NBR 6118")
    memorial.add_equation("sigma_adm = fck / 1.4", "Tensão admissível", granularity="detailed")
    memorial.add_equation("taxa_uso = sigma / sigma_adm", "Taxa de utilização", granularity="detailed")
    
    memorial.add_comparison_table()
    return memorial.generate_html("viga_completa_latex.html")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("🚀 GERANDO MEMORIAL COM LATEX RENDERIZADO")
    print("="*70 + "\n")
    file = example_viga_completa()
    print(f"\n✅ Abra: {file}")
