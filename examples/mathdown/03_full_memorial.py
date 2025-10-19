"""
Example 3: Advanced MathDown - Complete Memorial

Demonstrates ALL features:
- Headings (Markdown)
- Math blocks with descriptions
- Variable persistence
- Multi-line equations
- Comparisons with auto-evaluation
- Code blocks
- Figure directives (TODO)
"""

from pymemorial.mathdown import MathDownParser
from pathlib import Path

mathdown_text = """
# Memorial de Cálculo Estrutural

**Projeto:** Ponte Rodoviária BR-101  
**Autor:** Eng. Maria Santos, CREA-RJ 98765  
**Revisão:** 2.0  
**Data:** 18/10/2025

---

## 1. Introdução

Este memorial apresenta os cálculos estruturais para dimensionamento 
da longarina principal da ponte sobre o Rio Paraíba.

## 2. Características Geométricas

### 2.1 Vão e Carregamento

Geometria da estrutura:

#$ L = 25 m $#

#$ b = 2.5 m $#

Carga permanente:

#$ g = 12 kN/m $#

Carga acidental:

#$ q = 15 kN/m $#

Carga total:

#$ p = g + q $#

#$ p = 27 kN/m $#

### 2.2 Propriedades da Seção

Seção transversal composta:

#$ h = 1.20 m $#

#$ b_f = 0.60 m $#

#$ A = 0.45 m² $#

#$ I = 0.065 m⁴ $#

## 3. Análise Estrutural

### 3.1 Momento Máximo

Para viga biapoiada sob carga uniforme:

#$ sabemos que o momento máximo ocorre no centro do vão 
   => M_max = p*L²/8 $#

Cálculo passo a passo:

#$ M_max = 27*25²/8 $#
#$ M_max = 27*625/8 $#
#$ M_max = 16875/8 $#
#$ M_max = 2109.4 kN·m $#

### 3.2 Cortante Máximo

#$ sabemos que o cortante máximo ocorre nos apoios 
   => V_max = p*L/2 $#

Calculando:

#$ V_max = 27*25/2 $#

#$ V_max = 337.5 kN $#

### 3.3 Flecha Máxima

#$ sabemos que a flecha no centro para carga uniforme é 
   => δ_max = 5*p*L⁴/(384*E*I) $#

Considerando concreto C40:

#$ E_c = 35000 MPa $#

#$ E_c = 35000000 kN/m² $#

Calculando:

#$ δ_max = 5*27*25**4/(384*35000000*0.065) $#

#$ δ_max = 0.0234 m $#

#$ δ_max = 2.34 cm $#

## 4. Verificação de Flechas

Flecha admissível (NBR 6118:2023):

#$ δ_adm = L/250 $#

#$ δ_adm = 25/250 $#

#$ δ_adm = 0.10 m $#

#$ δ_adm = 10 cm $#

Comparando:

#$ δ_max < δ_adm $#

**Resultado:** APROVADO ✓

## 5. Tensões Atuantes

### 5.1 Tensão Normal Máxima

#$ sabemos que a tensão normal é => σ = M_max*y_max/I $#

onde:

#$ y_max = h/2 $#

#$ y_max = 0.60 m $#

Calculando:

#$ σ = 2109.4*0.60/0.065 $#

#$ σ = 19478.8 kN/m² $#

#$ σ = 19.5 MPa $#

### 5.2 Tensão de Cisalhamento

#$ sabemos que a tensão de cisalhamento é => τ = V_max*Q/(I*b_w) $#

Para seção retangular, aproximadamente:

#$ τ = 1.5*V_max/A $#

#$ τ = 1.5*337.5/0.45 $#

#$ τ = 1125 kN/m² $#

#$ τ = 1.12 MPa $#

## 6. Verificação de Resistência

Resistência do concreto C40:

#$ f_ck = 40 MPa $#

#$ f_cd = f_ck/1.4 $#

#$ f_cd = 28.6 MPa $#

Tensão de tração (limite):

#$ f_ctd = 0.3*f_ck**(2/3)/1.4 $#

#$ f_ctd = 2.0 MPa $#

Comparações:

#$ σ < f_cd $# ✓ **COMPRESSÃO OK**

#$ τ < 0.25*f_cd $# ✓ **CISALHAMENTO OK**

## 7. Conclusão

### 7.1 Resumo das Verificações

Todas as verificações foram atendidas:

1. #$ δ_max < δ_adm $# → **2.34 cm < 10 cm** ✓
2. #$ σ < f_cd $# → **19.5 MPa < 28.6 MPa** ✓
3. #$ τ < 0.25*f_cd $# → **1.12 MPa < 7.15 MPa** ✓

### 7.2 Fatores de Segurança

Fator de segurança à flexão:

#$ FS_M = f_cd/σ $#

#$ FS_M = 28.6/19.5 $#

#$ FS_M = 1.47 $#

Fator de segurança ao cisalhamento:

#$ FS_V = (0.25*f_cd)/τ $#

#$ FS_V = 7.15/1.12 $#

#$ FS_V = 6.38 $#

**Conclusão Final:**

A longarina está adequadamente dimensionada com margens de segurança 
apropriadas conforme NBR 6118:2023.

---

*Elaborado por: Eng. Maria Santos, CREA-RJ 98765*  
*Data: 18/10/2025*
"""

# ============================================================================
# PARSE AND GENERATE COMPLETE REPORT
# ============================================================================

def main():
    # Create output directory
    output_dir = Path("outputs/mathdown")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse MathDown
    print("=" * 80)
    print("PARSING COMPLETE MEMORIAL...")
    print("=" * 80)
    
    parser = MathDownParser()
    result = parser.parse(mathdown_text)
    
    # Print statistics
    print("\n📊 PARSING STATISTICS:")
    print(f"   Tokens parsed: {len(result.tokens)}")
    print(f"   Math expressions: {len(result.expressions)}")
    print(f"   Variables detected: {len(result.context.variables)}")
    print(f"   Equations: {len(result.get_equations())}")
    print(f"   Assignments: {len(result.get_assignments())}")
    
    # Print variable table
    print("\n📋 VARIABLE TABLE:")
    print("-" * 80)
    print(f"{'Variable':<15} {'Value':>15} {'Unit':<10} {'LaTeX':<20}")
    print("-" * 80)
    
    for name in sorted(result.context.variables.keys()):
        var = result.context.variables[name]
        if var.value is not None:
            print(f"{name:<15} {var.value:>15.4f} {var.unit or '':<10} {var.latex:<20}")
    
    print("-" * 80)
    
    # Export to HTML
    html_file = output_dir / "03_full_memorial.html"
    result.to_html(str(html_file))
    
    print(f"\n✓ HTML report generated: {html_file}")
    print(f"✓ Open in browser to see the complete memorial!")
    
    # Export context to JSON
    import json
    context_file = output_dir / "03_context.json"
    context_file.write_text(
        json.dumps(result.context.to_dict(), indent=2),
        encoding='utf-8'
    )
    
    print(f"✓ Context exported: {context_file}")
    
    print("\n" + "=" * 80)
    print("COMPLETE!")
    print("=" * 80)

if __name__ == "__main__":
    main()
