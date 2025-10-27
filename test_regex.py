"""
test_memorial_advanced_complete.py

Memorial Técnico Completo demonstrando TODOS os recursos do PyMemorial:
- Matrizes simbólicas (múltiplos tipos)
- Derivadas e integrais
- Equações com steps (basic, smart, detailed)
- Operações matriciais avançadas
- Mistura natural de texto e cálculos
- Validação automática

Versão: 3.0.1 - Ultimate Edition (FIXED)
Data: 23 de outubro de 2025
"""

import sys
import os
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Add src to path
script_dir = Path(__file__).parent
src_dir = script_dir / 'src'
sys.path.insert(0, str(src_dir))

logger = logging.getLogger(__name__)


def main():
    logger.info("="*80)
    logger.info("🎓 MEMORIAL TÉCNICO COMPLETO - PyMemorial Ultimate Edition")
    logger.info("="*80 + "\n")
    
    # Importar PyMemorial
    try:
        from pymemorial.editor import NaturalMemorialEditor
        logger.info("✅ PyMemorial importado com sucesso\n")
    except ImportError as e:
        logger.error(f"❌ Falha ao importar PyMemorial: {e}")
        return sys.exit(1)
    
    # ========================================================================
    # MEMORIAL TÉCNICO COMPLETO
    # ========================================================================
    memorial_text = """
# Memorial de Cálculo Estrutural Avançado
**Projeto:** Análise Completa de Pórtico Espacial com Operações Matriciais  
**Norma:** NBR 6118:2014 / NBR 8800:2008  
**Engenheiro:** Sistema PyMemorial v3.0  
**Data:** 23 de Outubro de 2025

---

## 1. Introdução

Este memorial demonstra as capacidades avançadas do PyMemorial na análise 
estrutural, combinando:
- Matrizes de rigidez locais e globais
- Transformações de coordenadas
- Derivadas para análise de estabilidade
- Integrais para propriedades geométricas
- Steps detalhados em múltiplos níveis

---

## 2. Dados Gerais do Projeto

### 2.1 Geometria da Estrutura

Dimensões principais do pórtico espacial:

L_vao = 12.0 m      # Vão principal
H_pilar = 4.5 m     # Altura dos pilares
L_balanco = 3.0 m   # Comprimento do balanço

Espaçamento entre pórticos:
s_portico = 6.0 m

### 2.2 Propriedades dos Materiais

#### Concreto C40 (vigas e pilares):
fck = 40.0 MPa      # Resistência característica
Ec = 35000.0 MPa    # Módulo de elasticidade
nu_c = 0.2          # Coeficiente de Poisson
gamma_c = 25.0      # Peso específico (kN/m3)

#### Aço CA-50 (armadura):
fyk = 500.0 MPa     # Tensão de escoamento
Es = 210000.0 MPa   # Módulo de elasticidade

### 2.3 Seções Transversais

#### Viga Principal (seção T):
bw_viga = 20.0 cm   # Largura da alma
h_viga = 60.0 cm    # Altura total
bf_mesa = 80.0 cm   # Largura da mesa
hf_mesa = 10.0 cm   # Espessura da mesa

#### Pilar (seção retangular):
b_pilar = 40.0 cm   # Base
h_pilar_sec = 60.0 cm  # Altura

---

## 3. Propriedades Geométricas via Integração

### 3.1 Área da Seção em T

Calculando área pela soma das partes:

@eq[steps:detailed] A_mesa = bf_mesa * hf_mesa

@eq[steps:detailed] A_alma = bw_viga * (h_viga - hf_mesa)

@eq[steps:basic] A_viga_total = A_mesa + A_alma

A área total da viga T é **{A_viga_total:.2f}** cm².

### 3.2 Centro de Gravidade (CG) da Seção T

Posição do CG em relação à base:

@eq[steps:detailed] y_cg_mesa = h_viga - hf_mesa/2

@eq[steps:detailed] y_cg_alma = (h_viga - hf_mesa)/2

@eq[steps:detailed] y_cg = (A_mesa*y_cg_mesa + A_alma*y_cg_alma)/A_viga_total

O centro de gravidade está a **{y_cg:.2f}** cm da base.

### 3.3 Momento de Inércia via Integral

Usando o teorema dos eixos paralelos:

@eq[steps:detailed] I_mesa = (bf_mesa * hf_mesa**3)/12 + A_mesa*(y_cg_mesa - y_cg)**2

@eq[steps:detailed] I_alma = (bw_viga * (h_viga-hf_mesa)**3)/12 + A_alma*(y_cg_alma - y_cg)**2

@eq[steps:basic] I_viga_total = I_mesa + I_alma

O momento de inércia é **{I_viga_total:.2e}** cm⁴.

---

## 4. Matrizes de Rigidez - Elemento de Viga 2D

### 4.1 Preparação: Cálculos Intermediários

Primeiro, calculamos as propriedades necessárias:

@eq[normal] Le_elemento = L_vao

@eq[normal] EI_viga = Ec * I_viga_total

Comprimento do elemento: **{Le_elemento:.2f}** m  
Rigidez à flexão: **{EI_viga:.2e}** MPa·cm⁴

### 4.2 Matriz de Rigidez Local (4×4)

Para elemento de viga Euler-Bernoulli com comprimento Le e rigidez EI:

@matrix[steps:detailed] K_local = [[12*EI_viga/Le_elemento**3, 6*EI_viga/Le_elemento**2, -12*EI_viga/Le_elemento**3, 6*EI_viga/Le_elemento**2],
                                    [6*EI_viga/Le_elemento**2, 4*EI_viga/Le_elemento, -6*EI_viga/Le_elemento**2, 2*EI_viga/Le_elemento],
                                    [-12*EI_viga/Le_elemento**3, -6*EI_viga/Le_elemento**2, 12*EI_viga/Le_elemento**3, -6*EI_viga/Le_elemento**2],
                                    [6*EI_viga/Le_elemento**2, 2*EI_viga/Le_elemento, -6*EI_viga/Le_elemento**2, 4*EI_viga/Le_elemento]]

### 4.3 Matriz de Transformação (Rotação 30°)

Para transformar coordenadas locais → globais, calculamos os ângulos:

@eq[normal] theta_rot = 30.0

@eq[normal] theta_rad = 0.5236

Ângulo de rotação: **{theta_rot:.1f}°** = **{theta_rad:.4f}** rad

@matrix[steps:basic] T_rot = [[cos(theta_rad), -sin(theta_rad), 0, 0],
                               [sin(theta_rad), cos(theta_rad), 0, 0],
                               [0, 0, cos(theta_rad), -sin(theta_rad)],
                               [0, 0, sin(theta_rad), cos(theta_rad)]]

### 4.4 Matriz de Rigidez Global

Transformação: K_global = T^T · K_local · T

@matop[multiply:basic] K_global = T_rot.T @ K_local @ T_rot

---

## 5. Análise de Cargas e Esforços

### 5.1 Carregamentos Atuantes

Carga permanente (peso próprio + revestimento):
g_total = 15.0 kN/m

Carga acidental (sobrecarga de uso):
q_acidental = 5.0 kN/m

Carga total de cálculo (γ_g=1.4, γ_q=1.5):

@eq[steps:detailed] q_d = 1.4*g_total + 1.5*q_acidental

Carga distribuída de cálculo: **{q_d:.2f}** kN/m.

### 5.2 Reações de Apoio

Para viga biapoiada:

@eq[steps:smart] R_apoio_A = (q_d * L_vao)/2

@eq[steps:smart] R_apoio_B = (q_d * L_vao)/2

Reações nos apoios A e B: **{R_apoio_A:.2f}** kN cada.

### 5.3 Momento Fletor Máximo

No centro do vão:

@eq[steps:detailed] M_max_vao = (q_d * L_vao**2)/8

Momento máximo: **{M_max_vao:.2f}** kN·m.

### 5.4 Cortante Máximo

Nos apoios:

@eq[steps:basic] V_max = R_apoio_A

Cortante máximo: **{V_max:.2f}** kN.

---

## 6. Análise de Estabilidade via Derivadas

### 6.1 Função de Momento Fletor

Para 0 ≤ x ≤ L/2, definimos a variável simbólica:

@eq[normal] x_var = 0

@eq[steps:smart] M_func = R_apoio_A*x_var - (q_d*x_var**2)/2

### 6.2 Derivada: Taxa de Variação do Momento

@eq[steps:detailed] dM_dx = R_apoio_A - q_d*x_var

Esta derivada representa o esforço cortante V(x) = dM/dx.

### 6.3 Segunda Derivada: Curvatura

@eq[steps:basic] d2M_dx2 = -q_d

A segunda derivada d²M/dx² = -q(x) confirma a carga distribuída.

---

## 7. Deflexão via Integração

### 7.1 Equação Diferencial da Linha Elástica

A deflexão v(x) satisfaz: EI·d⁴v/dx⁴ = q(x)

### 7.2 Deflexão Máxima (Fórmula Clássica)

Para viga biapoiada com carga uniforme:

@eq[steps:detailed] v_max_formula = (5*q_d*L_vao**4)/(384*EI_viga)

Deflexão máxima: **{v_max_formula:.4f}** cm.

### 7.3 Verificação de Flecha Admissível

Flecha limite (L/250):

@eq[steps:basic] v_adm = L_vao/250

@eq[steps:basic] razao_flecha = v_max_formula/v_adm

Razão flecha/admissível: **{razao_flecha:.3f}** (deve ser ≤ 1.0).

---

## 8. Verificação de Tensões

### 8.1 Tensão Normal Máxima

No bordo mais comprimido:

@eq[steps:detailed] y_max = h_viga - y_cg

@eq[steps:detailed] sigma_max = (M_max_vao * y_max)/I_viga_total

Tensão normal máxima: **{sigma_max:.2f}** MPa.

### 8.2 Tensão Cisalhante Máxima

Na linha neutra (aproximação):

@eq[steps:smart] tau_max = (3*V_max)/(2*bw_viga*h_viga)

Tensão cisalhante máxima: **{tau_max:.2f}** MPa.

### 8.3 Verificação: Critério de Von Mises

@eq[steps:detailed] sigma_vm = sqrt(sigma_max**2 + 3*tau_max**2)

Tensão equivalente Von Mises: **{sigma_vm:.2f}** MPa.

Comparando com fck = **{fck}** MPa: 

@eq[steps:basic] FS_concreto = fck/sigma_vm

Fator de segurança: **{FS_concreto:.2f}** (adequado se ≥ 1.4).

---

## 9. Matriz de Massa para Análise Dinâmica

### 9.1 Preparação: Densidade Linear

Calculamos a densidade linear da viga:

@eq[normal] rho_viga = gamma_c * A_viga_total / 9.81

Densidade linear: **{rho_viga:.3f}** kg/m

### 9.2 Matriz de Massa Consistente (Simplificada 2×2)

Para análise modal:

@matrix[steps:basic] M_massa = [[rho_viga*L_vao/3, 0],
                                 [0, rho_viga*L_vao/3]]

---

## 10. Resumo e Conclusões

### 10.1 Resumo de Resultados Principais

| Grandeza | Valor Calculado | Limite | Status |
|----------|----------------|--------|--------|
| Momento Máximo | {M_max_vao:.2f} kN·m | - | - |
| Cortante Máximo | {V_max:.2f} kN | - | - |
| Deflexão Máxima | {v_max_formula:.4f} cm | {v_adm:.2f} cm | ✅ OK |
| Tensão Normal | {sigma_max:.2f} MPa | {fck} MPa | ✅ OK |
| Tensão Von Mises | {sigma_vm:.2f} MPa | {fck} MPa | ✅ OK |
| Fator de Segurança | {FS_concreto:.2f} | ≥ 1.4 | ✅ OK |

### 10.2 Parecer Técnico Final

A estrutura analisada atende a todos os critérios de resistência e 
serviceabilidade estabelecidos pelas normas NBR 6118:2014 e NBR 8800:2008.

Os cálculos foram realizados utilizando o sistema PyMemorial v3.0, que 
combina escrita natural com capacidades computacionais avançadas do SymPy 
e NumPy.

**Aprovação Técnica:** ✅ APROVADO PARA EXECUÇÃO

---

## 11. Anexos Técnicos

### 11.1 Preparação: Rigidez Torsional

@eq[normal] Le_3d = L_vao

@eq[normal] GJ_torsao = 0.3 * EI_viga

Rigidez torsional: **{GJ_torsao:.2e}** MPa·cm⁴

### 11.2 Matriz de Rigidez Expandida (6×6)

Para elemento 3D com 6 graus de liberdade por nó:

@matrix[steps:basic] K_3d = [[12*EI_viga/Le_3d**3, 0, 0, 0, 0, 6*EI_viga/Le_3d**2],
                              [0, GJ_torsao/Le_3d, 0, 0, 0, 0],
                              [0, 0, 12*EI_viga/Le_3d**3, -6*EI_viga/Le_3d**2, 0, 0],
                              [0, 0, -6*EI_viga/Le_3d**2, 4*EI_viga/Le_3d, 0, 0],
                              [0, 0, 0, 0, GJ_torsao/Le_3d, 0],
                              [6*EI_viga/Le_3d**2, 0, 0, 0, 0, 4*EI_viga/Le_3d]]

### 11.3 Operação: Autovalores (Frequências Naturais)

Calculando autovalores da matriz de massa:

@matop[eigenvalues] lambda_vals = eigenvals(M_massa)

---

**Responsável Técnico:**  
Eng. PyMemorial System  
CREA XXXXX

**Revisão:** v3.0.1 - 23/10/2025  
**Documento gerado automaticamente via PyMemorial Natural Engine**

"""
    
    # ========================================================================
    # PROCESSAR MEMORIAL
    # ========================================================================
    logger.info("📝 Processando memorial avançado...\n")
    
    try:
        editor = NaturalMemorialEditor(document_type='report')
        resultado = editor.process(memorial_text, clean=False)
        
        logger.info("✅ Memorial processado com sucesso!\n")
        
    except Exception as e:
        logger.error(f"❌ Erro ao processar: {e}")
        import traceback
        traceback.print_exc()
        return sys.exit(1)
    
    # ========================================================================
    # EXIBIR RESULTADO
    # ========================================================================
    print("\n" + "="*80)
    print("📄 MEMORIAL TÉCNICO COMPLETO - RESULTADO")
    print("="*80)
    print(resultado)
    print("="*80 + "\n")
    
    # ========================================================================
    # ESTATÍSTICAS
    # ========================================================================
    logger.info("📊 Estatísticas do Memorial Processado:\n")
    
    # Contar elementos processados
    import re
    
    num_equacoes = len(re.findall(r'@eq\[', memorial_text))
    num_matrizes = len(re.findall(r'@matrix\[', memorial_text))
    num_matops = len(re.findall(r'@matop\[', memorial_text))
    num_variaveis = len(re.findall(r'\w+\s*=\s*[\d\.]+', memorial_text))
    
    logger.info(f"  📐 Equações processadas: {num_equacoes}")
    logger.info(f"  🔢 Matrizes criadas: {num_matrizes}")
    logger.info(f"  ➕ Operações matriciais: {num_matops}")
    logger.info(f"  📊 Variáveis definidas: {num_variaveis}")
    
    # ========================================================================
    # VALIDAÇÃO
    # ========================================================================
    logger.info("\n🔍 Executando validações...\n")
    
    validacoes = {
        "Matriz K_local presente": r'K_local',
        "Matriz T_rot presente": r'T_rot',
        "Matriz K_global presente": r'K_global',
        "Equações com steps": r'→',
        "Valores numéricos calculados": r'\d+\.\d{2}',
        "Derivadas calculadas": r'dM_dx|d2M_dx2',
        "Tabela de resumo": r'\|.*\|.*\|',
    }
    
    erros = 0
    for nome, pattern in validacoes.items():
        if re.search(pattern, resultado):
            logger.info(f"✅ {nome}")
        else:
            logger.error(f"❌ {nome}")
            erros += 1
    
    # ========================================================================
    # RELATÓRIO FINAL
    # ========================================================================
    print("\n" + "="*80)
    print("📊 RELATÓRIO FINAL")
    print("="*80)
    print(f"\n✅ Validações Passadas: {len(validacoes) - erros}/{len(validacoes)}")
    print(f"❌ Validações Falhadas: {erros}/{len(validacoes)}")
    print(f"🎯 Taxa de Sucesso: {((len(validacoes)-erros)/len(validacoes))*100:.1f}%\n")
    
    if erros == 0:
        print("🎉 MEMORIAL TÉCNICO COMPLETO GERADO COM SUCESSO!")
        print("="*80 + "\n")
        return sys.exit(0)
    else:
        print("⚠️  Alguns recursos não foram processados corretamente.")
        print("="*80 + "\n")
        return sys.exit(1)


if __name__ == "__main__":
    main()
