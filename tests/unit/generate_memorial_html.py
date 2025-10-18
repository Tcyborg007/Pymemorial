"""
Gerador de Memorial de Cálculo em HTML
Demonstra todos os modos de granularidade com visualização profissional
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
        
    def add_variable(self, name: str, value: float, unit: str = "", description: str = ""):
        """Adiciona variável ao memorial."""
        self.variables[name] = VariableFactory.create(name, value, unit)
        self.variables[name].description = description
        
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
        # Extrair nome da variável resultado
        if '=' in expression:
            result_name = expression.split('=')[0].strip()
        else:
            result_name = None
            
        # Criar equação
        eq = Equation(expression, variables=self.variables, description=description)
        
        # Gerar passos
        steps = eq.steps(granularity=granularity, show_units=True)
        
        # Armazenar resultado
        if result_name and steps and steps[-1]['numeric'] is not None:
            result_value = steps[-1]['numeric']
            self.variables[result_name] = VariableFactory.create(result_name, result_value)
        
        # Gerar HTML da equação
        html = f"""
        <div class="equation-block">
            <div class="equation-header">
                <span class="equation-label">{description if description else expression}</span>
                <span class="granularity-badge {granularity}">{granularity.upper()}</span>
            </div>
        """
        
        # Adicionar passos
        if len(steps) > 2 and granularity in ['detailed', 'all', 'normal']:
            html += '<div class="steps-container">'
            for step in steps:
                step_class = 'step-symbolic' if step['operation'] == 'symbolic' else \
                            'step-substitution' if step['operation'] == 'substitution' else \
                            'step-result' if step['operation'] == 'result' else 'step-intermediate'
                
                numeric_display = f'<span class="numeric-value">= {step["numeric"]:.4f}</span>' if step['numeric'] is not None else ''
                
                html += f"""
                <div class="step {step_class}">
                    <span class="step-number">{step['step_number']}</span>
                    <span class="step-description">{step['description']}</span>
                    <div class="step-math">\\[{step['latex']}\\]{numeric_display}</div>
                </div>
                """
            html += '</div>'
        else:
            # Modo minimal - apenas resultado
            final_step = steps[-1]
            html += f"""
            <div class="result-only">
                <div class="step-math">\\[{final_step['latex']} = {final_step['numeric']:.4f}\\]</div>
            </div>
            """
        
        html += '</div>'
        self.html_content.append(html)
        
    def add_comparison_table(self):
        """Adiciona tabela comparativa de variáveis."""
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
                    </tr>
                </thead>
                <tbody>
        """
        
        for name, var in self.variables.items():
            value_display = f"{var.value:.4f}" if hasattr(var, 'value') and var.value is not None else "N/A"
            unit_display = var.unit if hasattr(var, 'unit') and var.unit else "-"
            var_type = "Entrada" if not hasattr(var, 'calculated') else "Calculado"
            
            html += f"""
                <tr>
                    <td><code>{name}</code></td>
                    <td>\\({name}\\)</td>
                    <td class="numeric">{value_display}</td>
                    <td>{unit_display}</td>
                    <td><span class="badge-type">{var_type}</span></td>
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
            
            .header .project {
                font-size: 1.2em;
                opacity: 0.9;
            }
            
            .header .meta {
                margin-top: 15px;
                font-size: 0.9em;
                opacity: 0.8;
            }
            
            .content {
                padding: 40px;
            }
            
            .section {
                margin-bottom: 40px;
                animation: fadeIn 0.5s ease-in;
            }
            
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(20px); }
                to { opacity: 1; transform: translateY(0); }
            }
            
            .section h2 {
                color: #667eea;
                border-bottom: 3px solid #667eea;
                padding-bottom: 10px;
                margin-bottom: 20px;
                font-size: 1.8em;
            }
            
            .subtitle {
                color: #666;
                font-style: italic;
                margin-bottom: 20px;
            }
            
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
            
            .equation-label {
                font-weight: bold;
                color: #333;
                font-size: 1.1em;
            }
            
            .granularity-badge {
                padding: 5px 15px;
                border-radius: 20px;
                font-size: 0.75em;
                font-weight: bold;
                text-transform: uppercase;
                letter-spacing: 1px;
            }
            
            .granularity-badge.minimal {
                background: #28a745;
                color: white;
            }
            
            .granularity-badge.normal {
                background: #17a2b8;
                color: white;
            }
            
            .granularity-badge.detailed {
                background: #dc3545;
                color: white;
            }
            
            .steps-container {
                margin-top: 15px;
            }
            
            .step {
                padding: 15px;
                margin-bottom: 10px;
                border-radius: 6px;
                background: white;
                border: 1px solid #dee2e6;
                transition: all 0.2s;
            }
            
            .step:hover {
                box-shadow: 0 3px 10px rgba(0,0,0,0.1);
                border-color: #667eea;
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
            
            .step-number {
                display: inline-block;
                width: 30px;
                height: 30px;
                background: #667eea;
                color: white;
                border-radius: 50%;
                text-align: center;
                line-height: 30px;
                font-weight: bold;
                margin-right: 10px;
            }
            
            .step-description {
                color: #666;
                font-size: 0.9em;
                font-weight: 600;
            }
            
            .step-math {
                margin-top: 10px;
                font-size: 1.1em;
                padding: 10px;
                background: white;
                border-radius: 4px;
            }
            
            .numeric-value {
                color: #28a745;
                font-weight: bold;
                margin-left: 15px;
                padding: 3px 10px;
                background: #d4edda;
                border-radius: 4px;
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
            
            .comparison-table h3 {
                color: #667eea;
                margin-bottom: 20px;
                font-size: 1.5em;
            }
            
            table {
                width: 100%;
                border-collapse: collapse;
            }
            
            thead {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
            }
            
            th, td {
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #dee2e6;
            }
            
            tbody tr:hover {
                background: #f8f9fa;
            }
            
            .numeric {
                font-family: 'Courier New', monospace;
                font-weight: bold;
                color: #667eea;
            }
            
            .badge-type {
                padding: 3px 8px;
                border-radius: 12px;
                font-size: 0.85em;
                background: #e3f2fd;
                color: #1976d2;
            }
            
            code {
                background: #f8f9fa;
                padding: 2px 6px;
                border-radius: 3px;
                font-family: 'Courier New', monospace;
                color: #d63384;
            }
            
            .footer {
                background: #f8f9fa;
                padding: 20px;
                text-align: center;
                color: #666;
                border-top: 1px solid #dee2e6;
            }
            
            @media print {
                body { background: white; padding: 0; }
                .container { box-shadow: none; }
                .equation-block:hover { transform: none; }
            }
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
    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <script>
        window.MathJax = {{
            tex: {{
                inlineMath: [['\\(', '\\)']],
                displayMath: [['\\[', '\\]']]
            }}
        }};
    </script>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📐 {self.title}</h1>
            <div class="project">{self.project if self.project else 'Memorial de Cálculo Estrutural'}</div>
            <div class="meta">
                Gerado em: {datetime.now().strftime('%d/%m/%Y às %H:%M:%S')} | 
                PyMemorial v1.0
            </div>
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
        
        print(f"✅ HTML gerado com sucesso: {output_file}")
        return output_file


# ============================================================================
# EXEMPLO 1: VIGA BIAPOIADA - COMPARAÇÃO DE MODOS
# ============================================================================

def example_viga_completa():
    """Exemplo completo de viga biapoiada com todos os modos."""
    
    memorial = HTMLMemorialGenerator(
        title="Dimensionamento de Viga Biapoiada",
        project="Edifício Residencial - Bloco A"
    )
    
    # Dados de entrada
    memorial.add_section("Dados de Entrada", "Geometria e carregamentos")
    memorial.add_variable("L", 6.0, "m", "Vão da viga")
    memorial.add_variable("q", 15.0, "kN/m", "Carga distribuída")
    memorial.add_variable("b", 0.20, "m", "Largura da seção")
    memorial.add_variable("h", 0.50, "m", "Altura da seção")
    memorial.add_variable("fck", 30.0, "MPa", "Resistência do concreto")
    
    # Geometria - MODO MINIMAL
    memorial.add_section("Propriedades Geométricas", "Cálculos com modo MINIMAL")
    memorial.add_equation(
        "A = b * h",
        "Área da seção transversal",
        granularity="minimal"
    )
    memorial.add_equation(
        "I = b * h**3 / 12",
        "Momento de inércia",
        granularity="minimal"
    )
    memorial.add_equation(
        "W = b * h**2 / 6",
        "Módulo de resistência",
        granularity="minimal"
    )
    
    # Esforços - MODO SMART/NORMAL
    memorial.add_section("Esforços Solicitantes", "Cálculos com modo NORMAL")
    memorial.add_equation(
        "M = q * L**2 / 8",
        "Momento fletor máximo no centro do vão",
        granularity="normal"
    )
    memorial.add_equation(
        "V = q * L / 2",
        "Força cortante máxima nos apoios",
        granularity="normal"
    )
    
    # Tensões - MODO DETAILED
    memorial.add_section("Verificação de Tensões", "Cálculos com modo DETAILED")
    memorial.add_equation(
        "sigma = M / W",
        "Tensão normal máxima",
        granularity="detailed"
    )
    memorial.add_equation(
        "tau = (3 * V) / (2 * A)",
        "Tensão cisalhante máxima",
        granularity="detailed"
    )
    
    # Verificações - MODO DETAILED
    memorial.add_section("Verificações Normativas", "Comparação com limites da NBR 6118")
    memorial.add_equation(
        "sigma_adm = fck / 1.4",
        "Tensão admissível do concreto",
        granularity="detailed"
    )
    memorial.add_equation(
        "taxa_uso = sigma / sigma_adm",
        "Taxa de utilização da seção",
        granularity="detailed"
    )
    
    # Tabela resumo
    memorial.add_comparison_table()
    
    return memorial.generate_html("viga_biapoiada_completo.html")


# ============================================================================
# EXEMPLO 2: PILAR - COMPARAÇÃO LADO A LADO
# ============================================================================

def example_pilar_comparacao():
    """Exemplo de pilar com comparação de granularidades."""
    
    memorial = HTMLMemorialGenerator(
        title="Dimensionamento de Pilar",
        project="Análise Comparativa de Modos de Cálculo"
    )
    
    # Dados
    memorial.add_section("Dados do Pilar")
    memorial.add_variable("N", 1200.0, "kN", "Força normal de compressão")
    memorial.add_variable("b_pilar", 0.30, "m", "Largura do pilar")
    memorial.add_variable("h_pilar", 0.40, "m", "Altura do pilar")
    memorial.add_variable("fck_pilar", 25.0, "MPa", "fck do concreto")
    
    # Mesmo cálculo em 3 modos
    memorial.add_section("Modo 1: MINIMAL", "Apenas resultado final")
    memorial.add_equation(
        "A_pilar = b_pilar * h_pilar",
        "Área do pilar",
        granularity="minimal"
    )
    memorial.add_equation(
        "sigma_pilar = N / A_pilar",
        "Tensão de compressão",
        granularity="minimal"
    )
    
    memorial.add_section("Modo 2: NORMAL", "Com passos intermediários")
    memorial.add_equation(
        "A_pilar = b_pilar * h_pilar",
        "Área do pilar",
        granularity="normal"
    )
    memorial.add_equation(
        "sigma_pilar = N / A_pilar",
        "Tensão de compressão",
        granularity="normal"
    )
    
    memorial.add_section("Modo 3: DETAILED", "Todos os passos detalhados")
    memorial.add_equation(
        "A_pilar = b_pilar * h_pilar",
        "Área do pilar",
        granularity="detailed"
    )
    memorial.add_equation(
        "sigma_pilar = N / A_pilar",
        "Tensão de compressão",
        granularity="detailed"
    )
    
    memorial.add_comparison_table()
    
    return memorial.generate_html("pilar_comparacao_modos.html")


# ============================================================================
# EXECUTAR EXEMPLOS
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🚀 GERANDO MEMORIAIS HTML")
    print("="*70 + "\n")
    
    # Gerar exemplo 1
    print("📄 Exemplo 1: Viga Biapoiada Completa")
    file1 = example_viga_completa()
    
    # Gerar exemplo 2
    print("\n📄 Exemplo 2: Pilar - Comparação de Modos")
    file2 = example_pilar_comparacao()
    
    print("\n" + "="*70)
    print("✅ CONCLUÍDO!")
    print("="*70)
    print(f"\nArquivos gerados:")
    print(f"  1. {file1}")
    print(f"  2. {file2}")
    print("\nAbra os arquivos no navegador para visualizar!")
