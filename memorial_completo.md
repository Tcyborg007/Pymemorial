# MEMORIAL DE CÁLCULO - VIGA DE CONCRETO ARMADO

## 1. DADOS DE ENTRADA

### 1.1 Materiais

f_ck = 30 MPa
f_yk = 500 MPa

### 1.2 Coeficientes

gamma_c = 1.4
gamma_s = 1.15
gamma_f = 1.4

### 1.3 Solicitações

M_k = 112.5 kN.m

## 2. RESISTÊNCIAS DE CÁLCULO

### 2.1 Concreto

Conforme NBR 6118:2023, item 12.3.3:

**Cálculo:**

→ $f_{cd} = \frac{f_{ck}}{\gamma_{c}}$
→ $f_{cd} = 21.4285714285714$
→ **$f_{cd} = 21.43$ MPa
f** ✓

### 2.2 Aço

Conforme NBR 6118:2023, item 8.3.3:

**[ERRO: f_yk / gamma_s]**

## 3. MOMENTO DE CÁLCULO

Aplicando os coeficientes de majoração conforme item 11.7.1:

**Cálculo:**

→ $M_{d} = M_{k} \gamma_{f}$
→ $M_{d} = 157.5$
→ **$M_{d} = 157.5$ ** ✓

## 4. RESUMO DOS RESULTADOS

Os valores de cálculo obtidos foram:

- Resistência do concreto: **21.43 MPa
f**
- Tensão de escoamento de cálculo do aço: {f_yd}
- Momento de cálculo: **157.5 **

### Fórmulas utilizadas:

- f_cd calculado por: $(f_ck / gamma_c)$
- f_yd calculado por: {{f_yd}}
- M_d calculado por: $(gamma_f * M_k)$

### Cálculos completos:

- Concreto: $(f_ck / gamma_c) = 21.43$ MPa
f
- Aço: {{{f_yd}}}
- Momento: $(gamma_f * M_k) = 157.5$

## 5. CONCLUSÃO

✓ Todos os valores atendem aos requisitos da NBR 6118:2023.
✓ O dimensionamento pode prosseguir com M_d = **157.5 **.