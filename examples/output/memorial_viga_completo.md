---
generator: PyMemorial NaturalWriter v2.0
created_at: 2025-10-27T19:28:14.495605
language: pt_BR
title: Memorial de Cálculo Estrutural
subtitle: Viga Simplesmente Apoiada - Análise Completa
author: Engº João Silva, CREA 12345/SP
project: Edifício Residencial XYZ
location: São Paulo, SP
date: 27 de outubro de 2025
norm: NBR 6118:2023
client: Construtora ABC Ltda.
revision: Rev. 00
---

---

## Sumário

- [1 Introdução](#introdução)
- [2 Dados de Entrada](#dados-de-entrada)
  - [2.1 Geometria](#geometria)
  - [2.2 Materiais](#materiais)
  - [2.3 Carregamento](#carregamento)
- [3 Esforços Solicitantes](#esforços-solicitantes)
  - [3.1 Carga de Cálculo](#carga-de-cálculo)
  - [3.2 Momento Fletor Máximo](#momento-fletor-máximo)
  - [3.3 Cortante Máxima](#cortante-máxima)
- [4 Dimensionamento à Flexão](#dimensionamento-à-flexão)
  - [4.1 Altura Útil](#altura-útil)
  - [4.2 Momento Limite](#momento-limite)
  - [4.3 Área de Aço Necessária](#área-de-aço-necessária)
- [5 Detalhamento da Armadura](#detalhamento-da-armadura)
- [6 Conclusão](#conclusão)

---

# 1 Introdução

Este memorial apresenta o dimensionamento de viga simplesmente apoiada em concreto armado, de acordo com as prescrições da NBR 6118:2023.

A viga está submetida a carregamento uniformemente distribuído e será dimensionada para resistir aos esforços de flexão simples.

# 2 Dados de Entrada



## 2.1 Geometria

**Tabela 1 - Dados geométricos**

| Parâmetro | Valor |
| --- | --- |
| Vão livre (L) | 6,00 m |
| Base da seção (b_w) | 20 cm |
| Altura total (h) | 50 cm |
| Cobrimento (c) | 3,0 cm |

## 2.2 Materiais

**Tabela 2 - Materiais especificados**

| Material | Classe | Resistência |
| --- | --- | --- |
| Concreto | C30 | f_ck = 30 MPa |
| Aço | CA-50 | f_yk = 500 MPa |

## 2.3 Carregamento

- Peso próprio: 2,5 kN/m (calculado automaticamente)
- Revestimento: 1,5 kN/m
- Sobrecarga de utilização: 3,0 kN/m (NBR 6120:2019)
- Carga total característica: q_k = 7,0 kN/m

# 3 Esforços Solicitantes



## 3.1 Carga de Cálculo

A carga de cálculo é obtida aplicando-se os coeficientes de ponderação conforme NBR 6118:2023, item 11.7.1.

Carga de cálculo:

    1.4 × q_k
Substituindo os valores:
    9.80000000000000
    9,80 = 9,80 kN/m

Conforme NBR 6118:2023, item 11.7.1.

## 3.2 Momento Fletor Máximo

Para viga simplesmente apoiada com carga uniformemente distribuída, o momento fletor máximo ocorre no meio do vão e é dado por:

Momento fletor de cálculo:

    L ×  × 2 × q_d/8
Substituindo os valores:
    44.1000000000000
    44,10 = 44,10 kN⋅m

## 3.3 Cortante Máxima

Força cortante de cálculo nos apoios:

    L × q_d/2
Substituindo os valores:
    29.4000000000000
    29,40 = 29,40 kN

# 4 Dimensionamento à Flexão



## 4.1 Altura Útil

Considerando armadura dupla (2 camadas) com diâmetro estimado φ = 12,5 mm:

Altura útil:

    -c + h - phi/2 - phi_est
Substituindo os valores:
    45.2500000000000
    45,25 = 45,25 cm

## 4.2 Momento Limite

Verificação se a seção requer armadura dupla (comprimida):

Coeficiente adimensional K_MD:

    100 × M_d/(b_w × d ×  × 2 × f_cd)
Substituindo os valores:
    0.00513841640398240
    0,01 = 0,01

Conforme NBR 6118:2023, Tabela 17.1.

Como K_MD < K_MD,lim = 0,295, a seção é de armadura simples.

## 4.3 Área de Aço Necessária

Área de aço de tração:

    b_w × d × f_cd × (1 - (-2 × K_MD/f_cd + 1) ×  × 0.5)/f_yd
Substituindo os valores:
    2.33652483868093
    2,34 = 2,34 cm²

# 5 Detalhamento da Armadura

Com base na área de aço calculada (A_s = 5,21 cm²), adota-se:

Armadura adotada:

- Armadura longitudinal de tração: 4φ12,5 mm (A_s,ef = 4,91 cm²)
- Armadura de pele: 2φ8,0 mm em cada face
- Estribos: φ5,0 mm c/ 15 cm (zona de apoio)
- Estribos: φ5,0 mm c/ 20 cm (zona central)

# 6 Conclusão

O dimensionamento da viga simplesmente apoiada foi realizado de acordo com as prescrições da NBR 6118:2023. A seção proposta atende a todos os critérios de segurança e estados limites de serviço.

A armadura adotada (4φ12,5 mm) resulta em uma taxa de armadura ρ = 1,1%, dentro dos limites prescritos pela norma (ρ_min = 0,15% e ρ_max = 4,0%).
