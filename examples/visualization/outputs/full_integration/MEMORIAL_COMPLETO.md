# Memorial de Cálculo - Pilar Metálico W310x107

**Projeto:** Edifício Comercial São Paulo  
**Responsável:** Eng. João Silva, CREA-SP 12345  
**Norma:** NBR 8800:2024  
**Revisão:** 1.0  
**Data:** 2025-10-18

---

{'metadata': {'title': 'Memorial de Cálculo - Pilar Metálico W310x107', 'author': 'Eng. João Silva, CREA-SP 12345', 'date': '2025-10-18', 'project': 'Edifício Comercial São Paulo', 'revision': '1.0', 'norm': 'NBR 8800:2024'}, 'variables': {}, 'equations': [], 'sections': [{'title': '1. INTRODUÇÃO', 'level': 1, 'numbered': True, 'content': [{'type': 'text', 'content': 'Este memorial apresenta a verificação estrutural do pilar W310x107 conforme NBR 8800:2024.'}], 'subsections': []}, {'title': '2. PROPRIEDADES DO MATERIAL', 'level': 1, 'numbered': True, 'content': [{'type': 'text', 'content': '**Material:** ASTM A572 Gr. 50'}, {'type': 'text', 'content': '- Tensão de escoamento: f_y = 345 MPa'}, {'type': 'text', 'content': '- Módulo de elasticidade: E = 200 GPa'}], 'subsections': []}, {'title': '3. PROPRIEDADES DA SEÇÃO', 'level': 1, 'numbered': True, 'content': [{'type': 'text', 'content': '**Perfil:** W310x107'}, {'type': 'text', 'content': '- Área bruta: A_g = 137.32 cm²'}, {'type': 'text', 'content': '- Raio de giração: r_x = 13.61 cm'}, {'type': 'text', 'content': '- Módulo plástico: Z_x = 1635 cm³'}], 'subsections': []}, {'title': '4. SOLICITAÇÕES', 'level': 1, 'numbered': True, 'content': [{'type': 'text', 'content': '- Força axial de cálculo: N_d = 1200 kN'}, {'type': 'text', 'content': '- Momento fletor: M_d = 150 kN·m'}], 'subsections': []}, {'title': '5. CÁLCULOS DE CAPACIDADE', 'level': 1, 'numbered': True, 'content': [{'type': 'text', 'content': '### 5.1 Índice de Esbeltez'}, {'type': 'text', 'content': 'λ = L / r_x = 4.00 / 0.136 = 29.40'}, {'type': 'text', 'content': '**Referência:** NBR 8800:2024 - Item 5.3.2'}, {'type': 'text', 'content': '### 5.2 Fator de Redução'}, {'type': 'text', 'content': 'χ = 0.877 (para λ = 29.40 < 100)'}, {'type': 'text', 'content': '### 5.3 Capacidade Axial'}, {'type': 'text', 'content': 'N_Rd = χ × A_g × f_y / γ_a = 3777 kN'}, {'type': 'text', 'content': '**Referência:** NBR 8800:2024 - Item 5.3.3'}, {'type': 'text', 'content': '### 5.4 Capacidade Flexional'}, {'type': 'text', 'content': 'M_Rd = f_y × Z_x / γ_a = 513 kN·m'}, {'type': 'text', 'content': '**Referência:** NBR 8800:2024 - Item 5.4.2'}], 'subsections': []}, {'title': '6. VERIFICAÇÃO', 'level': 1, 'numbered': True, 'content': [{'type': 'text', 'content': '**Razão de utilização:** η = 43.18%'}, {'type': 'text', 'content': '**Status:** APROVADO ✓'}], 'subsections': []}]}

---

## ANEXOS

### Diagrama de Interação P-M
![P-M Diagram](01_pm_interaction.pdf)

### Diagrama Momento-Curvatura
![M-κ Diagram](02_moment_curvature.pdf)

---

## CONCLUSÃO

O pilar W310x107 **ATENDE** 
aos requisitos da NBR 8800:2024 para as solicitações especificadas.

- Razão de utilização: η = 43.2%
- Ductilidade: μ = 10.00 (high)

---

*Elaborado por Eng. João Silva, CREA-SP 12345*  
*Data: 2025-10-18*
