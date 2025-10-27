# Análise Estrutural: Matriz de Rigidez Global

## 1. Dados do Problema

Elemento de viga com:
E = 210 GPa
I = 5e-5 m^4
L = 4 m

## 2. Matriz de Rigidez Local (Simbólica)

Definimos a matriz de rigidez local de um elemento de viga de Euler-Bernoulli:

*A matriz $K_{local}$ representa a rigidez do elemento no sistema de coordenadas local.*

## 3. Matriz de Transformação

Para um elemento inclinado a 45°:

theta = 45  # graus
c = 0.7071  # cos(45°)
s = 0.7071  # sin(45°)

## 4. Matriz de Rigidez Global

Calculamos a matriz no sistema global através da transformação:

Multiplicação final:

*A matriz $K_{global}$ é utilizada na montagem do sistema global de equações.*

## 5. Verificação: Inversão da Matriz de Rigidez

Para verificar a matriz, calculamos sua inversa (matriz de flexibilidade):

## 6. Propriedades da Matriz

Determinante:

O determinante é $det_{K}$, confirmando que a matriz é não-singular.

## 7. Matriz Reduzida (2x2)

Para análise simplificada, consideramos apenas os graus de liberdade verticais:

Inversa da matriz reduzida:

## 8. Conclusão

As matrizes foram calculadas com sucesso:
- Matriz local: $[K_{local}]$ (4×4)
- Matriz de transformação: $[T]$ (4×4)
- Matriz global: $[K_{global}]$ (4×4)
- Matriz de flexibilidade: $[K_{inv}]$ (4×4)

Determinante: $det_{K}$