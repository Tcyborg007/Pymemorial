# examples/smart_memorial_example.py
"""Exemplo com TextProcessingBridge"""

# ✅ CORRETO (sem underscore)
from pymemorial.document.internal.text_processing.integration import create_text_bridge

def main():
    # Criar bridge
    bridge = create_text_bridge(document_type='memorial', render_mode='full')
    
    # Definir variáveis
    bridge.define_variables({
        'M_k': (112.5, 'kN.m', 'Momento característico'),
        'gamma_f': (1.4, '', 'Coeficiente de majoração'),
    })
    
    # Texto natural
    text = """
## Cálculo do Momento

O momento característico M_k = {M_k} é majorado por gamma_f = {gamma_f}.

[eq:M_d = gamma_f * M_k]

Resultado: M_d = {M_d}
"""
    
    # Processar
    result = bridge.process(text)
    print(result)

if __name__ == "__main__":
    main()
