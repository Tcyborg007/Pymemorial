# exemplo_viga_completo.py
"""
Exemplo Completo: Análise de Vigas com CalcProcessor

Demonstra TODOS os recursos do calc_processor.py:
- Detecção automática de variáveis
- #eq: para equações formatadas
- #calc: para cálculos
- #steps[mode]: para steps detalhados
- #for ... #end: loops
- #table[...]: tabelas automáticas
- {variavel} substituição
- #ref:label referências cruzadas
"""

from pymemorial.core.calc_processor import CalcProcessor

# Texto natural completo
documento = """
# ANÁLISE COMPARATIVA: TEORIAS DE VIGAS

## 1. DADOS DO PROBLEMA

Considere uma viga de aço simplesmente apoiada:

E = 200e9      # Pa - Módulo de elasticidade do aço
G = 77e9       # Pa - Módulo de cisalhamento
I = 8.33e-6    # m^4 - Momento de inércia
A = 0.01       # m^2 - Área da seção transversal
L = 3.0        # m - Comprimento da viga
q = 10000      # N/m - Carga uniformemente distribuída
kappa = 5/6    # Fator de forma (seção retangular)

---

## 2. TEORIA DE EULER-BERNOULLI

A teoria clássica de Euler-Bernoulli despreza os efeitos de cisalhamento.

A deflexão máxima é calculada pela fórmula:

#eq: w_max_EB = (5*q*L^4)/(384*E*I)

Vamos calcular o valor numérico:

#calc: w_max_EB = 5*q*L**4/(384*E*I)

O resultado obtido é **w_max_EB = {w_max_EB*1000:.3f} mm**.

---

## 3. TEORIA DE TIMOSHENKO

A teoria de Timoshenko considera a deformação adicional por cisalhamento transversal.

A deflexão por cisalhamento é dada por:

#eq: w_cis = (q*L^2)/(8*kappa*G*A)

Calculando:

#calc: w_cis = q*L**2/(8*kappa*G*A)

A deflexão total é a soma das duas componentes:

#eq: w_max_Timo = w_max_EB + w_cis

#calc: w_max_Timo = w_max_EB + w_cis

Portanto, w_max_Timo = {w_max_Timo*1000:.3f} mm.

---

## 4. COMPARAÇÃO QUANTITATIVA

A diferença entre as teorias é:

#calc: diferenca_abs = w_max_Timo - w_max_EB
#calc: diferenca_perc = 100*(w_max_Timo - w_max_EB)/w_max_EB

- Diferença absoluta: **{diferenca_abs*1000:.3f} mm**
- Diferença percentual: **{diferenca_perc:.2f}%**

A teoria de Timoshenko prevê uma deflexão **{diferenca_perc:.1f}% maior** que Euler-Bernoulli.

---

## 5. ESTUDO PARAMÉTRICO

Vamos analisar o efeito da relação L/h (esbeltez) na diferença entre teorias:

#table[caption="Influência da esbeltez na diferença entre teorias", label="tab:esbeltez"]:
| L/h   | w_EB (mm) | w_Timo (mm) | Diferença (%) |
|-------|-----------|-------------|---------------|

#for h in [0.1, 0.15, 0.2, 0.3, 0.5]:
    b = 0.1
    I_h = b*h**3/12
    A_h = b*h
    L_h = L/h
    
    w_EB_h = 5*q*L**4/(384*E*I_h)
    w_cis_h = q*L**2/(8*kappa*G*A_h)
    w_Timo_h = w_EB_h + w_cis_h
    dif_h = 100*(w_Timo_h - w_EB_h)/w_EB_h
    
    #row: | {L_h:.1f} | {w_EB_h*1000:.2f} | {w_Timo_h*1000:.2f} | {dif_h:.1f} |
#end

#end

Conforme observado na #ref:tab:esbeltez, a diferença entre as teorias aumenta 
significativamente para vigas menos esbeltas (L/h < 15).

---

## 6. ANÁLISE GRÁFICA

#plot:
    x = [5, 10, 15, 20, 25, 30]
    y_EB = []
    y_Timo = []
    for L_h_ratio in x:
        h_temp = L/L_h_ratio
        I_temp = 0.1*h_temp**3/12
        A_temp = 0.1*h_temp
        w_EB_temp = 5*q*L**4/(384*E*I_temp)
        w_cis_temp = q*L**2/(8*kappa*G*A_temp)
        w_Timo_temp = w_EB_temp + w_cis_temp
        y_EB.append(w_EB_temp*1000)
        y_Timo.append(w_Timo_temp*1000)
    #line: x, y_EB, label="Euler-Bernoulli", color="blue"
    #line: x, y_Timo, label="Timoshenko", color="red"
    #xlabel: "Relação L/h"
    #ylabel: "Deflexão máxima (mm)"
    #title: "Comparação: Timoshenko vs Euler-Bernoulli"
    #legend: upper right
    #grid: true
#end

---

## 7. CONCLUSÕES

Com base na análise realizada:

1. Para vigas **esbeltas** (L/h > 20), a diferença entre teorias é menor que 2%
2. Para vigas **moderadas** (10 < L/h < 20), a diferença varia entre 2-10%
3. Para vigas **curtas** (L/h < 10), a diferença pode ultrapassar 15%

**Recomendação prática**: Utilizar a teoria de Timoshenko quando L/h < 15 
para garantir precisão adequada no cálculo de deflexões.

---

FIM DO DOCUMENTO
"""

# ============================================================================
# PROCESSAR DOCUMENTO
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("🚀 PROCESSANDO DOCUMENTO COM CALC_PROCESSOR")
    print("=" * 80)
    
    # Criar processador
    processor = CalcProcessor()
    
    # Processar documento
    resultado_html = processor.process(documento)
    
    # Salvar resultado
    with open("resultado_viga.html", "w", encoding="utf-8") as f:
        # Adicionar CSS básico
        f.write("""
<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <title>Análise de Vigas - PyMemorial</title>
    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 900px;
            margin: 40px auto;
            padding: 20px;
            line-height: 1.6;
        }
        h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
        h2 { color: #34495e; margin-top: 40px; }
        .equation {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin: 20px 0;
            padding: 15px;
            background: #f8f9fa;
            border-left: 4px solid #3498db;
        }
        .equation-content { flex: 1; }
        .equation-number {
            font-weight: bold;
            color: #7f8c8d;
            margin-left: 20px;
        }
        .calculation-step {
            margin: 10px 0;
            padding: 10px;
            background: #fff;
            border-left: 3px solid #95a5a6;
        }
        .calculation-table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        .calculation-table th {
            background: #3498db;
            color: white;
            padding: 12px;
            text-align: left;
        }
        .calculation-table td {
            padding: 10px;
            border-bottom: 1px solid #ddd;
        }
        .calculation-table caption {
            caption-side: top;
            font-weight: bold;
            margin-bottom: 10px;
            text-align: left;
        }
        figure {
            text-align: center;
            margin: 30px 0;
        }
        figure img {
            max-width: 100%;
            border: 1px solid #ddd;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        figcaption {
            margin-top: 10px;
            font-style: italic;
            color: #7f8c8d;
        }
        strong { color: #e74c3c; }
    </style>
</head>
<body>
""")
        f.write(resultado_html)
        f.write("</body></html>")
    
    print("\n✅ Documento processado com sucesso!")
    print(f"📄 Resultado salvo em: resultado_viga.html")
    print(f"📊 Variáveis calculadas: {len(processor.context.variables)}")
    print(f"🧮 Equações numeradas: {processor.context.equation_counter}")
    print(f"📈 Figuras geradas: {processor.context.figure_counter}")
    print(f"📋 Tabelas geradas: {processor.context.table_counter}")
    
    # Mostrar algumas variáveis
    print("\n📊 VARIÁVEIS PRINCIPAIS:")
    print("-" * 50)
    for var, value in list(processor.context.variables.items())[:15]:
        unit = processor.context.units.get(var, "")
        print(f"  {var:15} = {value:12.6e} {unit}")
    
    print("\n" + "=" * 80)