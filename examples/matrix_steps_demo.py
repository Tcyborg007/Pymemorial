#!/usr/bin/env python3
"""
Demonstração de Matrizes com Steps no PyMemorial

🎯 Features demonstradas:
- Definição de matrizes simbólicas
- Definição de matrizes numéricas
- Multiplicação com steps
- Inversão com steps
- Transposição
- Determinante
"""

from pymemorial.editor import NaturalMemorialEditor

# Texto do memorial com matrizes
memorial_text = """
# Análise Estrutural: Matriz de Rigidez Global

## 1. Dados do Problema

Elemento de viga com:
E = 210 GPa
I = 5e-5 m^4
L = 4 m

## 2. Matriz de Rigidez Local (Simbólica)

Definimos a matriz de rigidez local de um elemento de viga de Euler-Bernoulli:

@matrix[steps:detailed] K_local = [[12*E*I/L**3, 6*E*I/L**2, -12*E*I/L**3, 6*E*I/L**2],
                                     [6*E*I/L**2, 4*E*I/L, -6*E*I/L**2, 2*E*I/L],
                                     [-12*E*I/L**3, -6*E*I/L**2, 12*E*I/L**3, -6*E*I/L**2],
                                     [6*E*I/L**2, 2*E*I/L, -6*E*I/L**2, 4*E*I/L]]

*A matriz $K_{local}$ representa a rigidez do elemento no sistema de coordenadas local.*

## 3. Matriz de Transformação

Para um elemento inclinado a 45°:

theta = 45  # graus
c = 0.7071  # cos(45°)
s = 0.7071  # sin(45°)

@matrix[steps:normal] T = [[c, s, 0, 0],
                           [-s, c, 0, 0],
                           [0, 0, c, s],
                           [0, 0, -s, c]]

## 4. Matriz de Rigidez Global

Calculamos a matriz no sistema global através da transformação:

@matop[multiply:detailed] K_temp = T.T * K_local

Multiplicação final:

@matop[multiply:normal] K_global = K_temp * T

*A matriz $K_{global}$ é utilizada na montagem do sistema global de equações.*

## 5. Verificação: Inversão da Matriz de Rigidez

Para verificar a matriz, calculamos sua inversa (matriz de flexibilidade):

@matop[inverse:detailed] K_inv = inv(K_global)

## 6. Propriedades da Matriz

Determinante:
@matop[determinant] det_K = det(K_global)

O determinante é $det_K$, confirmando que a matriz é não-singular.

## 7. Matriz Reduzida (2x2)

Para análise simplificada, consideramos apenas os graus de liberdade verticais:

@matrix[steps:basic] K_red = [[1968.75, -984.375],
                               [-984.375, 1968.75]]

Inversa da matriz reduzida:

@matop[inverse:normal] K_red_inv = inv(K_red)

## 8. Conclusão

As matrizes foram calculadas com sucesso:
- Matriz local: $[K_{local}]$ (4×4)
- Matriz de transformação: $[T]$ (4×4)  
- Matriz global: $[K_{global}]$ (4×4)
- Matriz de flexibilidade: $[K_{inv}]$ (4×4)

Determinante: $det_K$
"""

# Processar o memorial
editor = NaturalMemorialEditor(document_type='memorial')
resultado = editor.process(memorial_text)

# Salvar resultado
with open('output_matrix_steps.md', 'w', encoding='utf-8') as f:
    f.write(resultado)

print("✅ Memorial com matrizes gerado com sucesso!")
print(f"📊 Estatísticas:")
print(f"   - Matrizes: {len(editor.matrices)}")
print(f"   - Variáveis: {len(editor.variables)}")
print(f"   - Equações: {len(editor.equations)}")
print("\n📄 Arquivo salvo: output_matrix_steps.md")
