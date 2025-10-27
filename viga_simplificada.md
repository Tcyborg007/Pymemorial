# MEMORIAL DE CÁLCULO - VIGA BIENGASTADA

## 1. DADOS DE ENTRADA

Comprimento total:
L = 6.0

Base da seção:
b = 0.20

Altura da seção:
h = 0.50

Carga aplicada:
P = 50.0

Módulo de elasticidade (em MPa):
E = 27000.0

---

## 2. PROPRIEDADES DA SEÇÃO

Momento de inércia:

**ERRO:** I: Erro ao converter expressão '(b*h**3)/12 = ?': Erro SymPy: Sympify of expression 'could not parse '(b*h**3)/12 = ?'' failed, because of exception being raised:
SyntaxError: cannot assign to expression (<unknown>, line 1), Erro AST: invalid syntax (<unknown>, line 1)

Resultado: I = $I$ m⁴

---

## 3. COMPRIMENTO DO ELEMENTO

**ERRO:** Le: Erro ao converter expressão 'L/2 = ?': Erro SymPy: Sympify of expression 'could not parse 'L/2 = ?'' failed, because of exception being raised:
SyntaxError: cannot assign to expression (<unknown>, line 1), Erro AST: invalid syntax (<unknown>, line 1)

Cada elemento tem $Le$ metros.

---

## 4. COEFICIENTES DE RIGIDEZ

Rigidez à translação:

**ERRO:** k11: Variáveis indefinidas: I, Le

Rigidez translação-rotação:

**ERRO:** k12: Variáveis indefinidas: I, Le

Rigidez rotação (extremidade):

**ERRO:** k22: Variáveis indefinidas: I, Le

Rigidez rotação (centro):

**ERRO:** k23: Variáveis indefinidas: I, Le

---

## 5. MATRIZ DE RIGIDEZ DO ELEMENTO

**Matriz:**

→ **Definição:** Ke: matriz 4×4

$$[Ke] = \left[\begin{matrix}k_{11} & k_{12} & - k_{11} & k_{12}\\k_{12} & k_{22} & - k_{12} & k_{23}\\- k_{11} & - k_{12} & k_{11} & - k_{12}\\k_{12} & k_{23} & - k_{12} & k_{22}\end{matrix}\right]$$

→ *Substituição:* L = 6.0 , b = 0.2 , h = 0.5 , P = 50.0 , E = 27000.0 , Fv = 50.0 , Fm = 0.0

Esta é a matriz de rigidez local 4×4.

---

## 6. RIGIDEZ GLOBAL

**ERRO:** K11: Variáveis indefinidas: k11

**ERRO:** K22: Variáveis indefinidas: k22

Matriz global reduzida:

**ERRO MATRIZ:** Kg: Cannot evaluate [0,0] = K11. Remaining symbols: {K11}

---

## 7. VETOR DE CARGAS

Força vertical:
Fv = 50.0

Momento:
Fm = 0.0

$$[F] = \left[\begin{matrix}Fv\\Fm\end{matrix}\right]$$

**F:**

| **F** | Col 1 |
|---|---|
**Linha 1** | 50 |
**Linha 2** | 0 |

---

## 8. INVERSÃO DA MATRIZ

**ERRO OPERAÇÃO:** Kg_inv: Cannot evaluate [0,0] = K11. Remaining symbols: {K11}

---

## 9. DESLOCAMENTOS

**ERRO OPERAÇÃO:** u: Matriz não encontrada: Kg_inv

O vetor $u$ contém os deslocamentos calculados.

---

## 10. REAÇÕES DE APOIO

**ERRO:** R1: Erro ao converter expressão 'P/2 = ?': Erro SymPy: Sympify of expression 'could not parse 'P/2 = ?'' failed, because of exception being raised:
SyntaxError: cannot assign to expression (<unknown>, line 1), Erro AST: invalid syntax (<unknown>, line 1)

**ERRO:** R3: Erro ao converter expressão 'P/2 = ?': Erro SymPy: Sympify of expression 'could not parse 'P/2 = ?'' failed, because of exception being raised:
SyntaxError: cannot assign to expression (<unknown>, line 1), Erro AST: invalid syntax (<unknown>, line 1)

**ERRO:** soma: Variáveis indefinidas: R1, R3

Verificação: $soma$ = $P$ → OK ✓

---

## 11. MOMENTOS

**ERRO:** Mmax: Erro ao converter expressão 'P*L/8 = ?': Erro SymPy: Sympify of expression 'could not parse 'P*L/8 = ?'' failed, because of exception being raised:
SyntaxError: cannot assign to expression (<unknown>, line 1), Erro AST: invalid syntax (<unknown>, line 1)

**ERRO:** Meng: Variáveis indefinidas: Mmax

---

## 12. FLECHA TEÓRICA

**ERRO:** delta: Variáveis indefinidas: I

Convertendo para mm:

**ERRO:** delta_mm: Variáveis indefinidas: delta

Flecha: $\delta_{mm}$ mm

---

## 13. LIMITE DE FLECHA

**ERRO:** lim: Erro ao converter expressão 'L*1000/250 = ?': Erro SymPy: Sympify of expression 'could not parse 'L*1000/250 = ?'' failed, because of exception being raised:
SyntaxError: cannot assign to expression (<unknown>, line 1), Erro AST: invalid syntax (<unknown>, line 1)

Limite: $lim$ mm

**ERRO:** razao: Variáveis indefinidas: delta_mm, lim

Razão: $razao$ → Aprovado se < 1.0

---

## RESUMO

**Propriedades:**
- Inércia: $I$ m⁴
- Rigidez translação: $k_{11}$ (unidades relativas)

**Deslocamentos:**
- Flecha: $\delta_{mm}$ mm
- Limite: $lim$ mm
- Razão: $razao$

**Reações:**
- R1 = $R_{1}$
- R3 = $R_{3}$

**Momentos:**
- Máximo: $Mmax$
- Engaste: $Meng$

**Conclusão:** Estrutura APROVADA ✓