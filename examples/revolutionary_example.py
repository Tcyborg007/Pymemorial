# examples/revolutionary_example.py
"""
Teste de Desafio v4.1 - Geração de Artigo Técnico Complexo (Escrita 100% Natural)

Este teste valida as capacidades máximas do NaturalMemorialEditor v4.1:
✅ Funções complexas do SymPy (integrate, sqrt, pi) via @calc
✅ Parser de duas passadas (detecta 'var = num' e 'var = expr_simples')
✅ Placeholders com formatação (ex: {var:.4f})
✅ Cadeia de cálculo longa (10+ dependências)
✅ Geração de documento técnico extenso e complexo SEM LaTeX explícito do usuário.
"""

import logging
import math # Import math for pi constant access if needed via sympify

# Habilitar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)-8s | %(name)-30s | %(message)s'
)

from pymemorial.editor import NaturalMemorialEditor

print("\n" + "=" * 80)
print("GERANDO ARTIGO TÉCNICO COM PYMEMORIAL (ESCRITA 100% NATURAL)")
print("=" * 80)

editor = NaturalMemorialEditor(document_type='memorial')

# ========== CONTEÚDO DO ARTIGO EM SINTAXE PYMEMORIAL (SEM LATEX EXPLÍCITO) ==========
# ✅ Use r""" (raw string) para evitar problemas com barras invertidas (\)
article_text = r"""
# Discretização de Vigas: Comparativo entre as Teorias de Euler-Bernoulli e Timoshenko no Método dos Elementos Finitos

**Autor:** Gerado por PyMemorial NaturalMemorialEditor v4.1
**Data:** 22 de Outubro de 2025
**Instituição:** HQ Serviços de Engenharia, Maricá-RJ

---
## Resumo
Este artigo explora as diferenças fundamentais entre as teorias de viga de Euler-Bernoulli e Timoshenko, com foco em sua aplicação no Método dos Elementos Finitos (MEF). Abordamos as hipóteses cinemáticas, as equações governantes e, crucialmente, a formulação das matrizes de rigidez elementares para ambas as teorias. Discutimos o fenômeno de "shear locking" associado à teoria de Timoshenko em elementos finos e apresentamos um exemplo numérico comparativo para ilustrar as diferenças práticas nos resultados de deslocamento e rotação. O objetivo é fornecer uma compreensão clara das vantagens e limitações de cada teoria no contexto da análise estrutural computacional.

---
## 1. Introdução
A análise de vigas é um pilar fundamental da engenharia de estruturas. Duas teorias predominam na modelagem do comportamento flexional de vigas: a teoria clássica de Euler-Bernoulli e a teoria mais refinada de Timoshenko. A escolha entre elas depende das características geométricas da viga e da precisão desejada.
A teoria de **Euler-Bernoulli**, desenvolvida no século XVIII, é adequada para vigas longas e esbeltas, onde os efeitos da deformação por cisalhamento podem ser desprezados. Suas hipóteses simplificadoras levam a equações diferenciais relativamente simples.
A teoria de **Timoshenko**, proposta no início do século XX, incorpora a deformação por cisalhamento, tornando-a mais precisa para vigas curtas e espessas ("vigas-parede") ou para materiais compósitos com baixa rigidez ao cisalhamento. No entanto, sua formulação é mais complexa.
Com o advento do Método dos Elementos Finitos (MEF), a discretização dessas teorias tornou-se essencial para a análise de estruturas complexas. Este artigo foca nas implicações da escolha da teoria na formulação da matriz de rigidez do elemento finito de viga e nas consequências práticas dessa escolha.

---
## 2. Teoria de Viga de Euler-Bernoulli
### 2.1 Hipóteses Cinemáticas
A teoria de Euler-Bernoulli baseia-se nas seguintes hipóteses simplificadoras:
1.  **Seções planas permanecem planas:** Uma seção transversal, inicialmente plana e perpendicular ao eixo da viga, permanece plana após a deformação.
2.  **Seções planas permanecem normais ao eixo deformado:** A seção transversal permanece perpendicular ao eixo longitudinal da viga após a deformação. Isso implica que a distorção por cisalhamento (gamma_xz) é nula.
3.  **Deslocamentos pequenos:** A teoria é válida para pequenas deformações e pequenos deslocamentos.
4.  **Material elástico linear e homogêneo:** O material segue a Lei de Hooke.

### 2.2 Equações Governantes
A partir das hipóteses, derivam-se as relações fundamentais:
* **Relação Momento-Curvatura:** O momento fletor (M) é proporcional à curvatura (kappa), que é a segunda derivada do deslocamento vertical (v(x)).
    M(x) = E * I * kappa(x)
    onde kappa(x) = d^2v/dx^2.

    # Definição de Variáveis Simbólicas (ignoradas pelo parser)
    E = 0
    I = 0
    v = 0
    x = 0
    M = 0

    *Nota: PyMemorial calcula valores. A equação momento-curvatura é: M = E*I * d^2v/dx^2.*

* **Rotação da Seção (theta):** A rotação é a primeira derivada do deslocamento.
    theta(x) = dv/dx

* **Relação Esforço Cortante-Momento:**
    V(x) = dM/dx = E * I * d^3v/dx^3

* **Relação Carga-Esforço Cortante:**
    q(x) = dV/dx = E * I * d^4v/dx^4

    A equação governante da viga de Euler-Bernoulli sob uma carga distribuída q(x) é (representada textualmente):
    E * I * d^4v/dx^4 = q(x)

---
## 3. Teoria de Viga de Timoshenko
### 3.1 Hipóteses Cinemáticas
A teoria de Timoshenko relaxa uma das hipóteses de Euler-Bernoulli:
1.  **Seções planas permanecem planas:** Mantida.
2.  **Seções planas NÃO permanecem normais ao eixo deformado:** A seção transversal pode girar independentemente da rotação do eixo da viga devido à deformação por cisalhamento (gamma_xz).
3.  **Deslocamentos pequenos:** Mantida.
4.  **Material elástico linear e homogêneo:** Mantida.

### 3.2 Equações Governantes
A rotação total da seção (theta(x)) tem componentes de flexão (psi(x)) e cisalhamento (gamma_xz).
* **Deslocamento Vertical:** v(x).
* **Rotação da Seção (Independente):** theta(x).
* **Relação Momento-Curvatura:** M(x) = E * I * d(theta)/dx
* **Relação Esforço Cortante-Distorção:** V(x) = k_s * G * A * gamma_xz
    onde gamma_xz = theta(x) - dv/dx

    # Variáveis Simbólicas (ignoradas)
    G = 0
    A = 0
    k_s = 0

    A equação para o esforço cortante se torna (textualmente):
    V(x) = k_s * G * A * ( theta(x) - dv/dx )

* **Equilíbrio de Momentos e Forças:** As equações de equilíbrio levam ao sistema acoplado (representado textualmente):

    d/dx( E*I * d(theta)/dx ) + k_s*G*A * ( dv/dx - theta ) = 0

    d/dx( k_s*G*A * ( theta - dv/dx ) ) = -q(x)

Comparada à equação única de 4ª ordem de Euler-Bernoulli, Timoshenko resulta em um sistema de duas equações diferenciais acopladas de 2ª ordem para v(x) e theta(x).

---
## 4. Discretização no Método dos Elementos Finitos (MEF)
O MEF discretiza a viga em elementos menores, conectados por nós.

### 4.1 Graus de Liberdade (GDL)
* **Euler-Bernoulli:** 2 nós, 2 GDL/nó (v, theta = dv/dx). Total 4 GDL. {d} = [v1, theta1, v2, theta2]^T
* **Timoshenko:** 2 nós, 2 GDL/nó (v, theta independente). Total 4 GDL. {d} = [v1, theta1, v2, theta2]^T

### 4.2 Funções de Forma
* **Euler-Bernoulli:** Continuidade C¹ (Hermite Cúbico).
    v(x) = N1(x)*v1 + N2(x)*theta1 + N3(x)*v2 + N4(x)*theta2
    theta(x) = dv/dx = sum( dNi/dx * di )
* **Timoshenko:** Continuidade C⁰ (Lagrange Linear/Quadrático).
    v(x) = Nv1(x)*v1 + Nv2(x)*v2
    theta(x) = Ntheta1(x)*theta1 + Ntheta2(x)*theta2

### 4.3 Matriz de Rigidez Elementar [k_e]
[k_e] = [k_b] + [k_s]
* **Euler-Bernoulli:** Apenas [k_b]. A matriz de rigidez [k_EB] é proporcional a (E*I)/L^3 e contém termos envolvendo 12, 6*L, 4*L^2, etc., relacionando forças/momentos nodais com deslocamentos/rotações nodais.

* **Timoshenko:** Inclui [k_s]. A matriz de rigidez [k_T] depende também de G, A, k_s e do parâmetro adimensional Phi.
    Phi = (12 * E * I) / (k_s * G * A * L**2)

    A matriz [k_T] é proporcional a (E*I)/(L^3 * (1+Phi)) e seus termos são modificados em relação à [k_EB] pela inclusão de Phi (ex: o termo (2,2) torna-se (4+Phi)*L^2).

    # Variável simbólica (ignorada)
    L_elem = 0

    **Observações:** Se Phi -> 0 (cisalhamento desprezível), [k_T] -> [k_EB].

### 4.4 Shear Locking (Travamento por Cisalhamento)
Ocorre em elementos de Timoshenko finos com interpolações simples (lineares), tornando-os artificialmente rígidos. Causado pela incapacidade de representar gamma_xz = theta - dv/dx = 0.
**Soluções:** Integração Reduzida/Seletiva, Interpolações Mistas, Elementos Baseados em Deformações, Elementos Livres de Travamento.

---
## 5. Comparativo das Teorias

| Característica         | Euler-Bernoulli                     | Timoshenko                             |
| :--------------------- | :---------------------------------- | :------------------------------------- |
| **Hipótese Chave** | Seções normais (gamma_xz=0)     | Seções não normais (gamma_xz != 0) |
| **Deform. Cisalhamento**| Desprezada                          | Incluída                               |
| **Variáveis Primárias** | v(x)                              | v(x), theta(x) (independentes)     |
| **Equação Governante** | 1 Eq. Dif. 4ª Ordem                | 2 Eqs. Dif. Acopladas 2ª Ordem        |
| **Aplicabilidade** | Vigas esbeltas (L/h > 15-20)        | Vigas curtas/espessas, compósitos      |
| **Precisão** | Menor                               | Maior (se cisalhamento relevante)      |
| **MEF GDL Nodal** | v, theta (=dv/dx)                 | v, theta (independente)            |
| **MEF Continuidade** | C¹ (Hermite)                      | C⁰ (Lagrange)                        |
| **MEF Risco** | -                                   | Shear Locking                          |

---
## 6. Exemplo Numérico
Viga biapoiada, seção retangular, carga concentrada P no meio.

### 6.1 Dados da Viga e Carregamento
# Propriedades Geométricas
L_viga = 500 cm
b_viga = 20 cm
h_viga = 50 cm

# Propriedades do Material (Aço)
E_mat = 20000 kN/cm2
nu_mat = 0.3 # Coeficiente de Poisson

# Carregamento
P_carga = 100 kN # Carga concentrada no meio

### 6.2 Cálculos Preliminares
# Área
@calc[numeric] A_sec = b_viga * h_viga

# Momento de Inércia
@calc[numeric] I_sec = (b_viga * h_viga**3) / 12

# Módulo de Cisalhamento
@calc[numeric] G_mat = E_mat / (2 * (1 + nu_mat))

# Fator de Correção de Cisalhamento (Retangular)
# Parser de duas passadas detecta e calcula 5/6
k_s_fator = 5/6

### 6.3 Resultados Teóricos (Fórmulas Analíticas)
# --- Euler-Bernoulli ---
@calc[steps:smart] v_max_EB = (P_carga * L_viga**3) / (48 * E_mat * I_sec)
@calc[steps:smart] theta_max_EB = (P_carga * L_viga**2) / (16 * E_mat * I_sec)

# --- Timoshenko ---
@calc[numeric] v_shear_T = (P_carga * L_viga) / (4 * k_s_fator * G_mat * A_sec)
@calc[steps:smart] v_max_T = v_max_EB + v_shear_T

### 6.4 Comparação dos Resultados
@calc[numeric] diff_v = ((v_max_T - v_max_EB) / v_max_EB) * 100

print("\n--- COMPARAÇÃO DOS RESULTADOS ANALÍTICOS ---")
# Usa placeholders com formatação
print(f"Deslocamento Máximo (Euler-Bernoulli): {v_max_EB:.4f} cm")
print(f"Deslocamento Máximo (Timoshenko):      {v_max_T:.4f} cm")
print(f"Diferença Percentual (Deslocamento): {diff_v:.2f}%")

print(f"\nRotação Máxima (Euler-Bernoulli): {theta_max_EB:.6f} rad")
print("(Rotação da linha elástica em Timoshenko é igual à de Bernoulli neste caso simplificado)")


### 6.5 Discussão do Exemplo
# Calcula L/h ratio e usa placeholder formatado
@calc[numeric] L_h_ratio = L_viga / h_viga
O exemplo mostra que o deslocamento calculado pela teoria de Timoshenko (**{v_max_T:.4f}** cm) é maior do que o previsto por Euler-Bernoulli (**{v_max_EB:.4f}** cm). A diferença, **{diff_v:.2f}**%, deve-se à inclusão da parcela de deformação por cisalhamento (**{v_shear_T:.4f}** cm).

Para esta viga com relação L/h = **{L_h_ratio:.1f}** (considerada esbelta), a diferença é relativamente pequena. Se a viga fosse mais curta e espessa (ex: L/h = 5), a contribuição do cisalhamento seria significativamente maior, e a teoria de Euler-Bernoulli subestimaria consideravelmente o deslocamento real.
No contexto do MEF, usar um elemento de Timoshenko (sem shear locking) capturaria essa diferença, enquanto um elemento de Euler-Bernoulli não.

---
## 7. Conclusão
A escolha entre as teorias de Euler-Bernoulli e Timoshenko para a discretização de vigas no MEF tem implicações diretas na precisão e aplicabilidade da análise.
* **Euler-Bernoulli:** Simples, eficiente para vigas esbeltas. Formulação MEF C¹ robusta.
* **Timoshenko:** Mais geral e precisa para vigas curtas/espessas, mas requer cuidado na formulação MEF (shear locking). Permite interpolações C⁰.
A compreensão das hipóteses é crucial para o engenheiro ao selecionar a modelagem apropriada. O `NaturalMemorialEditor` do PyMemorial facilita a exploração e documentação desses cálculos.

---
## 8. Referências
* Timoshenko, S. P., & Gere, J. M. (1972). *Mechanics of Materials*. Van Nostrand Reinhold.
* Cook, R. D., et al. (2001). *Concepts and Applications of Finite Element Analysis*. Wiley.
* Bathe, K. J. (1996). *Finite Element Procedures*. Prentice Hall.
* NBR 6118:2023 - Projeto de estruturas de concreto - Procedimento. ABNT.
* NBR 8800:2008 - Projeto de estruturas de aço e de estruturas mistas de aço e concreto de edifícios - Procedimento. ABNT.

"""

# ========== PROCESSAMENTO COM PYMEMORIAL ==========
try:
    result = editor.process(article_text)
    print(result)

    summary = editor.get_summary()
    print("\n" + "-" * 80)
    print("RESUMO DO ARTIGO GERADO:")
    print(f"  Variáveis Totais Detectadas e Calculadas: {summary['variable_count']}")
    print(f"  Equações Processadas (@calc): {summary['equation_count']}")
    # print(f"  Lista de Variáveis: {summary['variables']}") # Pode ser longa
    print(f"  Lista de Equações: {summary['equations']}")
    print("-" * 80)
    print("\n🚀 Artigo Técnico (Escrita Natural) Gerado com Sucesso! 🚀")

except Exception as e:
    print("\n" + "=" * 80)
    print("ERRO DURANTE O PROCESSAMENTO DO ARTIGO:")
    print(e)
    print("=" * 80)
    logging.exception("Erro detalhado:")

print("\n" + "=" * 80)