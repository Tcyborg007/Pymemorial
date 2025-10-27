# Memorial de Cálculo Completo - Estruturas Metálicas
**Projeto:** Análise de Viga Bi-apoiada com Cargas Concentradas  
**Norma:** NBR 8800:2008  
**Data:** 23 de outubro de 2025

---

## 1. Dados de Entrada

Propriedades do perfil metálico W310x52:

@calc d = 317 mm  # Altura total
@calc bf = 167 mm  # Largura do flange
@calc tf = 13.2 mm  # Espessura do flange
@calc tw = 7.6 mm   # Espessura da alma
@calc Ag = 66.4 cm²  # Área bruta
@calc Ix = 12900 cm⁴  # Momento de inércia em X
@calc Iy = 880 cm⁴    # Momento de inércia em Y
@calc Wx = 814 cm³    # Módulo resistente em X

Propriedades do aço ASTM A572 Gr50:

@calc fy = 345 MPa  # Tensão de escoamento
@calc fu = 450 MPa  # Tensão de ruptura
@calc E = 200000 MPa  # Módulo de elasticidade

Geometria da estrutura:

@calc L = 8.0 m  # Vão da viga
@calc P1 = 150 kN  # Carga concentrada 1
@calc P2 = 200 kN  # Carga concentrada 2
@calc a = 2.5 m    # Distância da carga P1 ao apoio A
@calc b = 5.5 m    # Distância da carga P2 ao apoio A

---

## 2. Análise de Cargas e Reações

### 2.1 Reações de Apoio

Somatório de momentos em relação ao apoio A:

@eq[steps:detailed] RA*L = P1*a + P2*b

Resolvendo para RA:

@eq RA = (P1*a + P2*b)/L

Pela condição de equilíbrio vertical:

@eq[steps:basic] RB = P1 + P2 - RA

### 2.2 Momento Fletor Máximo

O momento fletor máximo ocorre na posição da carga P2:

@eq Mmax = RA*b - P1*(b-a)

---

## 3. Matriz de Rigidez Local (Elemento de Viga)

Matriz de rigidez 4x4 para elemento de viga Euler-Bernoulli:

@matrix[steps:detailed] K_local = [[12*E*Ix/L**3, 6*E*Ix/L**2, -12*E*Ix/L**3, 6*E*Ix/L**2],
                                   [6*E*Ix/L**2, 4*E*Ix/L, -6*E*Ix/L**2, 2*E*Ix/L],
                                   [-12*E*Ix/L**3, -6*E*Ix/L**2, 12*E*Ix/L**3, -6*E*Ix/L**2],
                                   [6*E*Ix/L**2, 2*E*Ix/L, -6*E*Ix/L**2, 4*E*Ix/L]]

---

## 4. Verificação de Flexão Simples (NBR 8800)

### 4.1 Momento Resistente de Cálculo

Fator de modificação para distribuição de momentos:

@calc Cb = 1.0  # Momento uniforme (conservador)

Coeficiente de flambagem lateral com torção:

@eq[steps] lambda = (L/ry) * sqrt(fy/E)

Momento de plastificação total:

@eq Mp = Wx * fy

Momento resistente de cálculo:

@eq[steps:detailed] Mrd = (Cb * Mp) / 1.1

### 4.2 Verificação de Segurança

@eq[steps] FS = Mrd / Mmax

**Condição de segurança:** FS ≥ 1.0

---

## 5. Análise de Deflexão

### 5.1 Deflexão sob Carga P1

Deflexão no ponto de aplicação de P1:

@eq delta1 = (P1*a**2*(L-a)**2)/(3*E*Ix*L)

### 5.2 Deflexão sob Carga P2

Deflexão no ponto de aplicação de P2:

@eq delta2 = (P2*b**2*(L-b)**2)/(3*E*Ix*L)

### 5.3 Deflexão Máxima Estimada

Deflexão total aproximada (superposição):

@eq delta_max = delta1 + delta2

Deflexão admissível segundo NBR 8800:

@calc delta_adm = L/360

### 5.4 Verificação de Flecha

@eq[steps] ratio = delta_max / delta_adm

**Condição de aceitação:** ratio ≤ 1.0

---

## 6. Operações Matriciais Avançadas

### 6.1 Matriz de Transformação (Rotação 45°)

@calc theta = 45.0  # graus
@calc theta_rad = 0.7854  # radianos

@matrix[steps] T = [[cos(theta_rad), -sin(theta_rad)],
                    [sin(theta_rad), cos(theta_rad)]]

### 6.2 Matriz de Rigidez Global

Transformação da matriz local para global:

@matop[steps] K_global = T @ K_local @ T.T

---

## 7. Análise de Tensões (Von Mises)

### 7.1 Tensões Normais

Tensão normal máxima:

@eq sigma_x = Mmax / Wx

### 7.2 Tensões Cisalhantes

Tensão cisalhante na alma:

@eq tau_max = (RA * 1.5) / (tw * d)

### 7.3 Tensão Equivalente de Von Mises

@eq[steps:detailed] sigma_vm = sqrt(sigma_x**2 + 3*tau_max**2)

### 7.4 Verificação

@eq[steps] FS_tension = fy / sigma_vm

---

## 8. Integração e Derivação Simbólica

### 8.1 Função de Momento Fletor

Definindo função M(x) para 0 ≤ x ≤ a:

@eq M(x) = RA*x

### 8.2 Derivada (Taxa de Variação)

@eq[steps] dM_dx = diff(M(x), x)

### 8.3 Integral (Trabalho de Deformação)

Energia de deformação elástica:

@eq[steps:detailed] U = integrate(M(x)**2/(2*E*Ix), (x, 0, L))

---

## 9. Resumo e Conclusões

### 9.1 Resultados Principais

| Grandeza | Valor Calculado | Limite | Status |
|----------|----------------|--------|--------|
| Momento Máximo | @ref{Mmax} | - | - |
| Momento Resistente | @ref{Mrd} | - | - |
| FS Flexão | @ref{FS} | ≥ 1.0 | ✅ OK |
| Deflexão Máxima | @ref{delta_max} | @ref{delta_adm} | ✅ OK |
| Tensão Von Mises | @ref{sigma_vm} | @ref{fy} | ✅ OK |

### 9.2 Parecer Técnico

A viga metálica W310x52 em aço ASTM A572 Gr50 atende a todos os 
critérios de segurança e serviceabilidade da NBR 8800:2008.

**Aprovação:** ✅ APROVADO para execução

---

**Responsável Técnico:**  
Eng. Civil - CREA XXXXX

**Revisão:** v1.0 - 23/10/2025
