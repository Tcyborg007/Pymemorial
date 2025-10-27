# MEMORIAL DE CÁLCULO - VIGA T (V1)

**Projeto:** Edifício Residencial Alpha  
**Norma:** NBR 6118:2023  
**Data:** Outubro 2025  
**Engenheiro:** João Silva - CREA 12345/SP  

---

## 1. GEOMETRIA DA SEÇÃO

b_w = 12 cm
b_f = 60 cm
h_f = 10 cm
h = 50 cm
d = 45 cm

## 2. PROPRIEDADES DOS MATERIAIS

### 2.1 Concreto

f_ck = 30 MPa
gamma_c = 1.4

**[ERRO: f_ck / gamma_c]**

### 2.2 Aço

f_yk = 500 MPa
gamma_s = 1.15

**[ERRO: f_yk / gamma_s]**

## 3. SOLICITAÇÕES

M_k = 180 kN.m
gamma_f = 1.4

**[ERRO: gamma_f * M_k]**

## 4. VERIFICAÇÃO DE LIMITES

### 4.1 Momento Limite da Mesa

Verificando se a linha neutra está na mesa (simplificado):

b_eff = 60 cm
x_lim = 14 cm

### 4.2 Momento Resistente Aproximado

K_md = 0.295
K_z = 0.882

**[ERRO: K_md * b_eff * d * d * f_cd]**

## 5. RESUMO E CONCLUSÃO

### Valores de Cálculo Obtidos:

| Parâmetro | Fórmula | Valor |
|-----------|---------|-------|
| f_cd      | {{f_cd}} | {f_cd} |
| f_yd      | {{f_yd}} | {f_yd} |
| M_d       | {{M_d}}  | {M_d}  |

### Verificação Final:

- Momento solicitante: M_d = {M_d}
- Momento resistente: M_Rd ≈ {M_Rd_aprox}
- Verificação: M_d < M_Rd → **✓ APROVADO**

### Observações:

O cálculo completo de M_d foi: {{{M_d}}}

---

**✓ MEMORIAL APROVADO**  
**Viga V1 dimensionada conforme NBR 6118:2023**