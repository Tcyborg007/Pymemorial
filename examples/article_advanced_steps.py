# examples/article_advanced_steps.py
"""
Artigo Técnico Avançado: Demonstrando PyMemorial Core Steps via Natural Editor

Objetivo: Showcase da integração natural_engine + core.Equation.steps com
          diferentes níveis de granularidade, integrais e simulação de matrizes.
"""

import logging
import math # Necessário se usar funções math diretamente (ex: pi)

# Habilitar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)-8s | %(name)-30s | %(message)s'
)

# Importa o editor que usa o Core por baixo
from pymemorial.editor import NaturalMemorialEditor

print("\n" + "=" * 80)
print("GERANDO ARTIGO TÉCNICO AVANÇADO COM PYMEMORIAL STEPS")
print("=" * 80)

# Inicializa o editor (que agora usa o Core)
editor = NaturalMemorialEditor(document_type='memorial') # Modo memorial mostra mais detalhes

# ========== CONTEÚDO DO ARTIGO ==========
# Usando r""" para raw string (evita problemas com \)
article_text = r"""
# Análise Avançada de Viga com PyMemorial: Detalhamento via Steps

**Autor:** PyMemorial Engine v5.1
**Data:** 22 de Outubro de 2025

---
## 1. Introdução

Este documento demonstra as capacidades avançadas do PyMemorial na análise estrutural, combinando escrita natural com o poder computacional do Python e SymPy, acessado através do PyMemorial Core. Focaremos na análise de uma viga simples, mas utilizaremos a funcionalidade `steps` com diferentes níveis de granularidade (`basic`, `smart`, `detailed`) para ilustrar o processo de cálculo, incluindo o cálculo de propriedades de seção via integração e a representação de termos de matrizes de rigidez. A filosofia é clara: escreva naturalmente, deixe o PyMemorial fazer o trabalho pesado e fornecer o nível de detalhe desejado.

---
## 2. Definição do Problema

Analisaremos uma viga biapoiada de seção retangular ("viga_ret") sujeita a uma carga uniformemente distribuída.

### 2.1 Dados Geométricos
Comprimento da viga:
L_viga = 6 m

Seção transversal retangular:
b_ret = 25 cm
h_ret = 60 cm

# "Truque" para variável de integração
y = 0

### 2.2 Propriedades do Material (Concreto C30)
Módulo de Elasticidade:
E_c = 31 GPa

### 2.3 Carregamento
Carga distribuída (peso próprio + carga acidental):
q_dist = 25 kN/m

### 2.4 Constantes
pi = 3.14159265 # Definido para clareza, poderia usar sympy.pi

---
## 3. Cálculo das Propriedades da Seção

Calcularemos a área e o momento de inércia da seção retangular. A inércia será calculada via integração para demonstrar a capacidade do SymPy.

### 3.1 Área da Seção
# Usando cálculo implícito (Parser 2-pass)
@calc[steps:smart] A_ret = b_ret * h_ret

### 3.2 Momento de Inércia (via Integral)
Utilizamos a definição integral $I_x = \int_A y^2 dA$, onde $dA = b dy$.

@calc[steps:smart] I_ret_integral = integrate(b_ret * y**2, (y, -h_ret/2, h_ret/2))

*Observação: O modo `basic` deve mostrar a fórmula simbólica, a substituição e o resultado final.*

---
## 4. Análise da Viga (Euler-Bernoulli)

Calcularemos as reações de apoio, o momento fletor máximo e a flecha máxima, demonstrando diferentes níveis de detalhamento dos passos.

### 4.1 Reações de Apoio
Para uma viga biapoiada com carga distribuída 'q', as reações são iguais.
@calc[steps:smart] R_apoio = (q_dist * L_viga) / 2

### 4.2 Momento Fletor Máximo (M_max = q * L^2 / 8)

**Modo Smart:** O PyMemorial Core decide o nível de detalhe.
@calc[steps:smart] M_max_smart = (q_dist * L_viga**2) / 8

**Modo Detailed:** Mostra mais passos intermediários (se implementado no Core).
@calc[steps:detailed] M_max_detailed = (q_dist * L_viga**2) / 8

*Comparando `{M_max_smart:~P}` com `{M_max_detailed:~P}`, vemos que o resultado final é o mesmo, mas a granularidade `detailed` pode (dependendo da implementação no core.Equation) expor mais etapas algébricas ou de avaliação.*

### 4.3 Flecha Máxima (v_max = 5 * q * L^4 / (384 * E * I))

**Modo Basic:** Apenas os passos essenciais.
@calc[steps:smart] v_max_basic = (5 * q_dist * L_viga**4) / (384 * E_c * I_ret_integral)

**Modo Detailed:** Máximo detalhamento dos passos.
@calc[steps:detailed] v_max_detailed_arrow = (5 * q_dist * L_viga**4) / (384 * E_c * I_ret_integral)

*A flecha máxima calculada é `{v_max_basic:.3f~P}` (modo basic) ou `{v_max_detailed_arrow:.3f~P}` (modo detailed). A diferença está na visualização do processo de cálculo.*

---
## 5. Simulação de Termos da Matriz de Rigidez (Euler-Bernoulli)

Podemos usar o `@calc` para computar termos individuais da matriz de rigidez $[k_e]$ de um elemento de viga de Euler-Bernoulli. Embora o PyMemorial não renderize a matriz completa, ele calcula seus componentes.

Elemento de comprimento L, rigidez EI:
L_elem = L_viga # Usando o comprimento total como exemplo de um elemento
@calc[numeric] E_I = E_c * I_ret_integral

Termos selecionados da matriz $[k_{EB}] = \frac{EI}{L^3} [\dots]$:

Termo (1,1) - Rigidez à translação vertical:
@calc[steps:smart] k11 = (12 * E_I) / L_elem**3

Termo (1,2) - Acoplamento translação-rotação:
@calc[steps:smart] k12 = (6 * E_I) / L_elem**2

Termo (2,2) - Rigidez à rotação:
@calc[steps:smart] k22 = (4 * E_I) / L_elem

*Estes valores (`**{k11:.3e~P}**`, `**{k12:.3e~P}**`, `**{k22:.3e~P}**`) representam os coeficientes que relacionam forças/momentos nodais a deslocamentos/rotações nodais no elemento finito.*
---
## 6. Conclusão

O PyMemorial, através da integração do `NaturalMemorialEditor` com o `core`, permite uma escrita fluida e natural para documentos de engenharia, enquanto realiza cálculos complexos nos bastidores. A capacidade de controlar a granularidade dos passos de cálculo via `@calc[steps:granularity]` oferece flexibilidade para o engenheiro documentar seu trabalho com o nível de detalhe apropriado, desde um resumo (`basic`) até uma depuração detalhada (`detailed` ou `all`). A integração com SymPy permite o uso de funções matemáticas avançadas como `integrate`, e a estrutura suporta a representação de conceitos como matrizes através do cálculo de seus termos.

---
## 7. Referências (Placeholder)
* Hibbeler, R. C. (2012). *Resistência dos Materiais*. Pearson.
* Beer, F. P., Johnston, E. R., Jr., DeWolf, J. T., & Mazurek, D. F. (2011). *Mecânica dos Materiais*. McGraw-Hill.

"""

# ========== PROCESSAMENTO COM PYMEMORIAL ==========
try:
    result = editor.process(article_text)
    print(result) # Imprime o artigo processado

    summary = editor.get_summary()
    print("\n" + "-" * 80)
    print("RESUMO DO ARTIGO AVANÇADO GERADO:")
    print(f"  Variáveis Totais: {summary['total_variables']}")
    print(f"  Equações Totais: {summary['total_equations']}")
    print(f"  Lista de Equações Processadas: {summary['equations_list']}")

    print("-" * 80)
    print("\n🚀 Artigo Técnico Avançado (com Steps) Gerado com Sucesso! 🚀")

except Exception as e:
    print("\n" + "=" * 80)
    print("ERRO DURANTE O PROCESSAMENTO DO ARTIGO AVANÇADO:")
    print(e)
    print("=" * 80)
    logging.exception("Erro detalhado:")

print("\n" + "=" * 80)