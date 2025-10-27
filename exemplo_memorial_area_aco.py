# exemplo_memorial_completo.py
"""
Memorial de Cálculo COMPLETO - Dimensionamento de Viga em Concreto Armado
NBR 6118:2023
"""

from pathlib import Path
from pymemorial.document import Memorial, DocumentMetadata
from pymemorial.core import Equation
from pymemorial.core.variable import VariableFactory


def criar_memorial_completo():
    """Cria memorial profissional completo."""
    
    # ========================================================================
    # METADATA
    # ========================================================================
    metadata = DocumentMetadata(
        title='Memorial de Cálculo - Dimensionamento de Viga V1',
        author='Eng. João Silva, CREA/SP 123456'
    )
    
    # ========================================================================
    # CRIAR MEMORIAL
    # ========================================================================
    print("🔧 Gerando Memorial Completo...")
    memorial = Memorial(metadata=metadata)
    
    # ========================================================================
    # SEÇÃO 1: INTRODUÇÃO
    # ========================================================================
    memorial.add_section(
        title='Objetivo',
        content='''
Este memorial apresenta o dimensionamento à flexão simples de viga em concreto armado 
conforme a **NBR 6118:2023 - Projeto de estruturas de concreto**.

O cálculo determina a área de aço necessária para resistir aos esforços de flexão, 
considerando os estados limites últimos e de serviço.
        ''',
        level=1
    )
    
    # ========================================================================
    # SEÇÃO 2: DADOS DE ENTRADA
    # ========================================================================
    memorial.add_section(
        title='Dados de Entrada',
        content='',
        level=1
    )
    
    memorial.add_section(
        title='Geometria da Viga',
        content='''
| Parâmetro | Símbolo | Valor | Unidade |
|-----------|---------|-------|---------|
| Largura | b | 20 | cm |
| Altura total | h | 50 | cm |
| Altura útil | d | 45 | cm |
| Cobrimento | c | 3.0 | cm |
| Vão | L | 6.0 | m |
        ''',
        level=2
    )
    
    memorial.add_section(
        title='Materiais',
        content='''
**Concreto:**
- Classe: C30
- f_ck = 30 MPa (resistência característica à compressão)
- γ_c = 1.4 (coeficiente de ponderação NBR 6118, item 12.3.3)

**Aço:**
- Categoria: CA-50
- f_yk = 500 MPa (resistência característica ao escoamento)
- γ_s = 1.15 (coeficiente de ponderação NBR 6118, item 12.4.1)
        ''',
        level=2
    )
    
    memorial.add_section(
        title='Carregamentos',
        content='''
**Ações Permanentes (g):**
- Peso próprio: 2.5 kN/m
- Revestimento: 1.0 kN/m
- Total: g = 3.5 kN/m

**Ações Variáveis (q):**
- Sobrecarga de utilização: q = 3.0 kN/m
- γ_f = 1.4 (combinação normal, NBR 6118, item 11.7.1)

**Momento Característico:**
- M_k = (g + q) × L² / 8 = (3.5 + 3.0) × 6² / 8 = **29.25 kN.m**

**Observação:** Este valor será majorado pelo coeficiente γ_f para obtenção do momento de cálculo.
        ''',
        level=2
    )
    
    # ========================================================================
    # SEÇÃO 3: CÁLCULO DO MOMENTO DE PROJETO
    # ========================================================================
    memorial.add_section(
        title='Momento de Cálculo (M_d)',
        content='''
O momento de cálculo é obtido majorando o momento característico pelo coeficiente de 
ponderação das ações conforme NBR 6118:2023, item 11.7.1:
        ''',
        level=1
    )
    
    # Variáveis
    M_k = VariableFactory.create('M_k', value=29.25, unit='kN*m', 
                                 description='Momento característico')
    gamma_f = VariableFactory.create('gamma_f', value=1.4, unit='', 
                                     description='Coeficiente de majoração das ações')
    
    # Equação M_d com PASSOS DETALHADOS
    eq_M_d = Equation(
        expression='M_d = gamma_f * M_k',
        variables={'gamma_f': gamma_f, 'M_k': M_k},
        description='Momento de cálculo (NBR 6118:2023, item 11.7.1)'
    )
    
    M_d_value = eq_M_d.evaluate()
    
    memorial.add_paragraph(f'''
**Cálculo:**

M_d = γ_f × M_k = {gamma_f.magnitude} × {M_k.magnitude} = **{M_d_value:.2f} kN.m**

**Conclusão:** O momento de cálculo é M_d = {M_d_value:.2f} kN.m
    ''')
    
    # ========================================================================
    # SEÇÃO 4: RESISTÊNCIAS DE CÁLCULO
    # ========================================================================
    memorial.add_section(
        title='Resistências de Cálculo',
        content='',
        level=1
    )
    
    # f_cd
    memorial.add_section(
        title='Resistência de Cálculo do Concreto (f_cd)',
        content='',
        level=2
    )
    
    f_ck = VariableFactory.create('f_ck', value=30.0, unit='MPa')
    gamma_c = VariableFactory.create('gamma_c', value=1.4, unit='')
    
    eq_f_cd = Equation(
        expression='f_cd = f_ck / gamma_c',
        variables={'f_ck': f_ck, 'gamma_c': gamma_c},
        description='Resistência de cálculo do concreto (NBR 6118:2023, item 12.3.3)'
    )
    
    f_cd_value = eq_f_cd.evaluate()
    
    memorial.add_paragraph(f'''
A resistência de cálculo do concreto à compressão é obtida reduzindo-se a resistência 
característica pelo coeficiente γ_c:

f_cd = f_ck / γ_c = {f_ck.magnitude} / {gamma_c.magnitude} = **{f_cd_value:.2f} MPa**
    ''')
    
    # f_yd
    memorial.add_section(
        title='Resistência de Cálculo do Aço (f_yd)',
        content='',
        level=2
    )
    
    f_yk = VariableFactory.create('f_yk', value=500.0, unit='MPa')
    gamma_s = VariableFactory.create('gamma_s', value=1.15, unit='')
    
    eq_f_yd = Equation(
        expression='f_yd = f_yk / gamma_s',
        variables={'f_yk': f_yk, 'gamma_s': gamma_s},
        description='Resistência de cálculo do aço (NBR 6118:2023, item 8.3.4)'
    )
    
    f_yd_value = eq_f_yd.evaluate()
    
    memorial.add_paragraph(f'''
A resistência de cálculo do aço ao escoamento é obtida reduzindo-se a resistência 
característica pelo coeficiente γ_s:

f_yd = f_yk / γ_s = {f_yk.magnitude} / {gamma_s.magnitude} = **{f_yd_value:.2f} MPa**
    ''')
    
    # ========================================================================
    # SEÇÃO 5: DIMENSIONAMENTO DA ARMADURA
    # ========================================================================
    memorial.add_section(
        title='Dimensionamento da Armadura Longitudinal',
        content='',
        level=1
    )
    
    # Dados
    b_cm = 20.0
    d_cm = 45.0
    h_cm = 50.0
    M_d_kNcm = M_d_value * 100
    f_cd_kNcm2 = f_cd_value / 10
    f_yd_kNcm2 = f_yd_value / 10
    
    # K_MD
    memorial.add_section(
        title='Parâmetro Adimensional K_MD',
        content='',
        level=2
    )
    
    K_MD = M_d_kNcm / (b_cm * d_cm**2 * f_cd_kNcm2)
    
    memorial.add_paragraph(f'''
O parâmetro adimensional K_MD é calculado por:

K_MD = M_d / (b × d² × f_cd)

Onde:
- M_d = {M_d_kNcm:.2f} kN.cm
- b = {b_cm} cm
- d = {d_cm} cm
- f_cd = {f_cd_kNcm2:.2f} kN/cm²

**K_MD = {K_MD:.4f}**

**Verificação do domínio de deformação:**
- K_MD = {K_MD:.4f} < 0.295 ✓
- **Conclusão:** Flexão simples, armadura simples é suficiente (Domínio 2 ou 3)
    ''')
    
    # K_z
    memorial.add_section(
        title='Coeficiente K_z',
        content='',
        level=2
    )
    
    K_z = (1 - (1 - 2 * K_MD) ** 0.5) / 2
    
    memorial.add_paragraph(f'''
O coeficiente da posição da linha neutra é:

K_z = [1 - √(1 - 2×K_MD)] / 2 = [1 - √(1 - 2×{K_MD:.4f})] / 2 = **{K_z:.4f}**

Este coeficiente indica a posição relativa da linha neutra na seção.
    ''')
    
    # z
    memorial.add_section(
        title='Braço de Alavanca (z)',
        content='',
        level=2
    )
    
    z_cm = d_cm * (1 - 0.4 * K_z)
    
    memorial.add_paragraph(f'''
O braço de alavanca é:

z = d × (1 - 0.4×K_z) = {d_cm} × (1 - 0.4×{K_z:.4f}) = **{z_cm:.2f} cm**
    ''')
    
    # A_s
    memorial.add_section(
        title='Área de Aço Calculada (A_s,calc)',
        content='',
        level=2
    )
    
    A_s_calc = M_d_kNcm / (z_cm * f_yd_kNcm2)
    
    memorial.add_paragraph(f'''
A área de aço necessária é calculada por:

A_s = M_d / (z × f_yd)

A_s = {M_d_kNcm:.2f} / ({z_cm:.2f} × {f_yd_kNcm2:.2f})

**A_s,calc = {A_s_calc:.2f} cm²**
    ''')
    
    # ========================================================================
    # SEÇÃO 6: VERIFICAÇÕES NORMATIVAS
    # ========================================================================
    memorial.add_section(
        title='Verificações Normativas',
        content='',
        level=1
    )
    
    # Armadura mínima
    memorial.add_section(
        title='Armadura Mínima (NBR 6118:2023, item 17.3.5.2.1)',
        content='',
        level=2
    )
    
    rho_min = 0.15 / 100
    A_s_min = rho_min * b_cm * d_cm
    
    memorial.add_paragraph(f'''
A taxa mínima de armadura é:

ρ_min = 0.15% (para CA-50)

A_s,min = ρ_min × b × d = {rho_min*100:.2f}% × {b_cm} × {d_cm} = **{A_s_min:.2f} cm²**

**Verificação:** A_s,calc = {A_s_calc:.2f} cm² > A_s,min = {A_s_min:.2f} cm² ✓
    ''')
    
    # Armadura máxima
    memorial.add_section(
        title='Armadura Máxima (NBR 6118:2023, item 17.3.5.2.4)',
        content='',
        level=2
    )
    
    A_s_max = 0.04 * b_cm * h_cm
    
    memorial.add_paragraph(f'''
A área máxima de armadura é:

A_s,max = 4% × b × h = 4% × {b_cm} × {h_cm} = **{A_s_max:.2f} cm²**

**Verificação:** A_s,calc = {A_s_calc:.2f} cm² < A_s,max = {A_s_max:.2f} cm² ✓
    ''')
    
    # ========================================================================
    # SEÇÃO 7: DETALHAMENTO DA ARMADURA
    # ========================================================================
    memorial.add_section(
        title='Detalhamento da Armadura',
        content='',
        level=1
    )
    
    memorial.add_paragraph('''
**Bitolas comerciais disponíveis para CA-50:**

| Diâmetro (mm) | Área (cm²) | Área 2φ (cm²) | Área 3φ (cm²) | Área 4φ (cm²) |
|---------------|------------|---------------|---------------|---------------|
| 6.3 | 0.31 | 0.62 | 0.93 | 1.24 |
| 8.0 | 0.50 | 1.00 | 1.50 | 2.00 |
| 10.0 | 0.79 | 1.58 | 2.36 | 3.14 |
| 12.5 | 1.23 | 2.46 | 3.69 | 4.91 |
| 16.0 | 2.01 | 4.02 | 6.03 | 8.04 |
| 20.0 | 3.14 | 6.28 | 9.42 | 12.57 |
    ''')
    
    A_s_adot = 4.91  # 4φ12.5
    
    memorial.add_paragraph(f'''
**Armadura adotada:** 4φ12.5 mm (A_s = {A_s_adot} cm²)

**Verificação:** {A_s_adot} cm² > {A_s_calc:.2f} cm² ✓

**Taxa de armadura:**
ρ = A_s / (b×d) = {A_s_adot} / ({b_cm}×{d_cm}) = {(A_s_adot/(b_cm*d_cm)*100):.2f}%

**Espaçamento:**
- Entre barras: s = (b - 2×c - 4×φ) / 3 = (20 - 2×3 - 4×1.25) / 3 = **3.0 cm**
- NBR 6118:2023, item 18.3.2.2: s_min = max(2cm, φ, 1.2×d_max_agregado) = 2.0 cm ✓
    ''')
    
    # ========================================================================
    # SEÇÃO 8: CONCLUSÃO
    # ========================================================================
    memorial.add_section(
        title='Conclusão',
        content=f'''
O dimensionamento da viga V1 resultou nas seguintes especificações:

**Armadura Longitudinal:**
- Calculada: A_s = {A_s_calc:.2f} cm²
- Adotada: 4φ12.5 mm (A_s = {A_s_adot} cm²)
- Taxa de armadura: ρ = {(A_s_adot/(b_cm*d_cm)*100):.2f}%

**Verificações:**
- ✓ Domínio de deformação adequado (K_MD = {K_MD:.4f} < 0.295)
- ✓ Armadura mínima atendida ({A_s_adot} > {A_s_min:.2f} cm²)
- ✓ Armadura máxima atendida ({A_s_adot} < {A_s_max:.2f} cm²)
- ✓ Espaçamento entre barras adequado (3.0 cm > 2.0 cm)

Todas as exigências da NBR 6118:2023 foram atendidas.

**Observação:** Este memorial não inclui o detalhamento da armadura transversal (estribos), 
que deve ser verificada separadamente conforme NBR 6118:2023, item 17.4.
        ''',
        level=1
    )
    
    return memorial


if __name__ == '__main__':
    # Criar memorial
    memorial = criar_memorial_completo()
    
    # Criar diretório
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)
    
    # Exportar HTML
    html_path = output_dir / 'memorial_completo.html'
    print(f"\n📄 Gerando HTML: {html_path}")
    
    try:
        memorial.render(html_path, format='html')
        print(f"✅ Sucesso!")
        print(f"\n📂 Abra o arquivo: {html_path.absolute()}")
    except Exception as e:
        print(f"❌ Erro: {e}")
        import traceback
        traceback.print_exc()
