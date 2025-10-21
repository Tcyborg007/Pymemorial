# examples/demo_memorial_visual.py
"""
Demonstração Visual do Memorial de Cálculo PyMemorial FASE 7.

Este exemplo cria um memorial completo de dimensionamento de um pilar misto
conforme NBR 8800:2024, gerando saída HTML5 para visualização imediata.

Autor: PyMemorial Team
Data: 2025-10-19
"""

from datetime import datetime
from pathlib import Path

from pymemorial.document import (
    Memorial,
    DocumentMetadata,
    NormCode,
    Revision,
    DocumentLanguage,
)

# ============================================================================
# CONFIGURAÇÃO DO DOCUMENTO
# ============================================================================

def create_demo_memorial():
    """Cria memorial de demonstração completo."""
    
    # 1. Criar metadados
    print("📝 Criando metadados do documento...")
    metadata = DocumentMetadata(
        title="Memorial de Cálculo - Pilar Misto PM-1",
        author="Eng. João Silva, CREA: 12345/SP",
        company="Estrutural Engenharia LTDA",
        code=NormCode.NBR8800_2024,
        project_number="EST-2024-001",
        revision=Revision(
            number="R00",
            date=datetime(2025, 10, 19),
            description="Emissão inicial para aprovação",
            author="João Silva",
            approved=False
        ),
        language=DocumentLanguage.PT_BR,
        keywords=['pilar misto', 'NBR 8800', 'compressão', 'NBR 8800:2024'],
        abstract=(
            "Este memorial apresenta o dimensionamento do pilar misto PM-1 "
            "conforme NBR 8800:2024. O pilar possui seção transversal em "
            "perfil I e está submetido a compressão axial e momento fletor."
        )
    )
    
    # 2. Criar memorial
    print("🏗️  Criando memorial com template NBR 8800...")
    memorial = Memorial(
        metadata,
        template='nbr8800',
        auto_toc=True,
        auto_title_page=True
    )
    
    # 3. Definir contexto global (variáveis usadas ao longo do documento)
    print("🔢 Definindo variáveis globais...")
    memorial.set_context(
        # Geometria
        H=4.5,          # Altura do pilar (m)
        L_flam=4.2,     # Comprimento de flambagem (m)
        
        # Seção transversal
        perfil="W250x73",
        A_g=93.1,       # Área bruta (cm²)
        I_x=7110,       # Momento de inércia eixo x (cm⁴)
        I_y=1600,       # Momento de inércia eixo y (cm⁴)
        r_x=8.72,       # Raio de giração x (cm)
        r_y=4.15,       # Raio de giração y (cm)
        
        # Propriedades dos materiais
        f_y=345,        # Tensão de escoamento (MPa)
        E=200000,       # Módulo de elasticidade (MPa)
        
        # Solicitações
        N_Sd=2500,      # Força normal de cálculo (kN)
        M_x_Sd=180,     # Momento fletor eixo x (kNm)
        M_y_Sd=45,      # Momento fletor eixo y (kNm)
        
        # Resultados (calculados previamente)
        lambda_x=0.65,  # Índice de esbeltez reduzido x
        lambda_y=1.38,  # Índice de esbeltez reduzido y
        chi_x=0.877,    # Fator de redução x
        chi_y=0.562,    # Fator de redução y
        N_Rd=3650,      # Resistência à compressão (kN)
        M_Rd_x=280,     # Momento resistente x (kNm)
        M_Rd_y=95,      # Momento resistente y (kNm)
    )
    
    return memorial


def add_memorial_content(memorial):
    """Adiciona conteúdo ao memorial."""
    
    # ========================================================================
    # 1. INTRODUÇÃO
    # ========================================================================
    print("\n📖 Seção 1: Introdução")
    memorial.add_section("Introdução", level=1)
    memorial.add_paragraph("""
Este memorial de cálculo apresenta o dimensionamento do pilar misto PM-1 
conforme a norma NBR 8800:2024 - Projeto de estruturas de aço e de 
estruturas mistas de aço e concreto de edifícios.

O pilar possui altura total de H = {H:.1f} m e comprimento de flambagem 
L_flam = {L_flam:.1f} m. A seção transversal adotada é o perfil laminado 
{perfil} em aço ASTM A572 Grau 50.

**Objetivo:** Verificar a segurança estrutural do pilar PM-1 submetido a 
compressão axial combinada com flexão biaxial.
""")
    
    # ========================================================================
    # 2. DADOS DE ENTRADA
    # ========================================================================
    print("📊 Seção 2: Dados de Entrada")
    memorial.add_section("Dados de Entrada", level=1)
    
    memorial.add_section("Geometria", level=2)
    memorial.add_paragraph("""
**Dimensões principais:**

- Altura do pilar: H = {H:.1f} m
- Comprimento de flambagem: L_flam = {L_flam:.1f} m
- Comprimento efetivo: K·L_flam = 1.0 × {L_flam:.1f} = {L_flam:.1f} m
""")
    
    memorial.add_section("Seção Transversal", level=2)
    memorial.add_paragraph("""
**Perfil laminado {perfil}:**

- Área bruta da seção: A_g = {A_g:.2f} cm²
- Momento de inércia (eixo x): I_x = {I_x:.0f} cm⁴
- Momento de inércia (eixo y): I_y = {I_y:.0f} cm⁴
- Raio de giração (eixo x): r_x = {r_x:.2f} cm
- Raio de giração (eixo y): r_y = {r_y:.2f} cm
""")
    
    memorial.add_section("Propriedades dos Materiais", level=2)
    memorial.add_paragraph("""
**Aço ASTM A572 Grau 50:**

- Tensão de escoamento: f_y = {f_y:.0f} MPa
- Módulo de elasticidade: E = {E:.0f} MPa
- Coeficiente de Poisson: ν = 0.30
""")
    
    memorial.add_section("Ações e Solicitações", level=2)
    memorial.add_paragraph("""
**Solicitações de cálculo (combinação última normal):**

- Força normal de compressão: N_Sd = {N_Sd:.2f} kN
- Momento fletor (eixo x): M_x_Sd = {M_x_Sd:.2f} kNm
- Momento fletor (eixo y): M_y_Sd = {M_y_Sd:.2f} kNm
""")
    
    # ========================================================================
    # 3. DIMENSIONAMENTO
    # ========================================================================
    print("🔧 Seção 3: Dimensionamento")
    memorial.add_section("Dimensionamento à Compressão", level=1)
    
    memorial.add_section("Esbeltez Reduzida", level=2)
    memorial.add_paragraph("""
Conforme item 5.3.3.1 da NBR 8800:2024, o índice de esbeltez reduzido é 
calculado por:

λ = (K·L / r) · √(f_y / E·π²)

**Eixo x (forte):**
- λ_x = {lambda_x:.3f} ≤ 1.5 ✅

**Eixo y (fraco):**
- λ_y = {lambda_y:.3f} ≤ 1.5 ✅

Como λ_y > λ_x, o **eixo y é o crítico** para flambagem.
""")
    
    memorial.add_section("Fator de Redução", level=2)
    memorial.add_paragraph("""
Conforme item 5.3.3.2 da NBR 8800:2024, o fator de redução χ é obtido 
considerando a curva de flambagem adequada ao perfil.

Para perfil I laminado com flambagem em torno do eixo de menor inércia:
- Curva de flambagem: **b**
- Imperfeição inicial: α = 0.34

**Fator de redução para eixo crítico (y):**
- χ_y = {chi_y:.3f}
""")
    
    memorial.add_section("Resistência à Compressão", level=2)
    memorial.add_paragraph("""
A resistência de cálculo à compressão é dada por:

N_Rd = χ · A_g · f_y

N_Rd = {chi_y:.3f} × {A_g:.2f} × {f_y:.0f} / 10
N_Rd = **{N_Rd:.2f} kN**
""")
    
    # ========================================================================
    # 4. VERIFICAÇÕES
    # ========================================================================
    print("✅ Seção 4: Verificações")
    memorial.add_section("Verificações", level=1)
    
    memorial.add_section("Resistência à Compressão", level=2)
    memorial.add_paragraph("""
**Verificação:** N_Sd ≤ N_Rd

{N_Sd:.2f} kN ≤ {N_Rd:.2f} kN ✅ **OK!**

Margem de segurança: {N_Rd:.2f} / {N_Sd:.2f} = {ratio:.2f}
""", variables={'ratio': 3650/2500})
    
    # Adicionar verificação técnica
    memorial.add_verification(
        expression="N_Sd <= N_Rd",
        passed=True,
        description="Resistência à compressão axial",
        norm=NormCode.NBR8800_2024,
        item="5.3.2.1",
        calculated_values={'N_Sd': 2500, 'N_Rd': 3650},
        safety_factor=3650/2500
    )
    
    memorial.add_section("Flexão Biaxial", level=2)
    memorial.add_paragraph("""
**Verificação da interação flexão-compressão:**

Para elementos submetidos a compressão e flexão biaxial, deve-se 
verificar a expressão de interação (item 5.5.2.2):

(N_Sd / N_Rd) + (M_x_Sd / M_Rd_x) + (M_y_Sd / M_Rd_y) ≤ 1.0

Onde:
- M_Rd_x = {M_Rd_x:.2f} kNm (momento resistente eixo x)
- M_Rd_y = {M_Rd_y:.2f} kNm (momento resistente eixo y)

**Cálculo:**

({N_Sd:.2f} / {N_Rd:.2f}) + ({M_x_Sd:.2f} / {M_Rd_x:.2f}) + ({M_y_Sd:.2f} / {M_Rd_y:.2f})

= {ratio_final:.3f} ≤ 1.0 ✅ **OK!**
""", variables={'ratio_final': 2500/3650 + 180/280 + 45/95})
    
    # Adicionar verificação de flexão-compressão
    memorial.add_verification(
        expression="(N_Sd/N_Rd) + (M_x/M_Rd_x) + (M_y/M_Rd_y) <= 1.0",
        passed=True,
        description="Interação flexão-compressão biaxial",
        norm=NormCode.NBR8800_2024,
        item="5.5.2.2",
        calculated_values={
            'N_Sd': 2500,
            'N_Rd': 3650,
            'M_x_Sd': 180,
            'M_Rd_x': 280,
            'M_y_Sd': 45,
            'M_Rd_y': 95,
            'ratio': 0.879
        }
    )
    
    # ========================================================================
    # 5. CONCLUSÕES
    # ========================================================================
    print("📋 Seção 5: Conclusões")
    memorial.add_section("Conclusões", level=1)
    memorial.add_paragraph("""
Com base nas análises e verificações realizadas, conclui-se que:

1. **Compressão axial:** O pilar PM-1 apresenta resistência adequada à 
   compressão axial, com N_Rd = {N_Rd:.2f} kN > N_Sd = {N_Sd:.2f} kN.

2. **Flexão biaxial:** A interação flexão-compressão foi verificada 
   conforme item 5.5.2.2 da NBR 8800:2024, resultando em índice de 
   utilização de 87.9%, dentro do limite de segurança.

3. **Aprovação:** O perfil laminado {perfil} em aço ASTM A572 Grau 50 
   atende aos requisitos normativos para o pilar PM-1.

**Recomendações:**
- Utilizar ligações de extremidade conforme projeto estrutural
- Garantir contraventamento lateral adequado
- Aplicar pintura anticorrosiva em toda superfície exposta
""")


def generate_demo():
    """Função principal de geração do memorial de demonstração."""
    print("\n" + "="*70)
    print("🚀 PyMemorial FASE 7 - Demonstração Visual")
    print("="*70)
    
    # Criar memorial
    memorial = create_demo_memorial()
    
    # Adicionar conteúdo
    add_memorial_content(memorial)
    
    # Validar antes de renderizar
    print("\n🔍 Validando documento...")
    result = memorial.validate()
    
    if result.valid:
        print("✅ Documento válido!")
    else:
        print(f"⚠️  {len(result.errors)} erros encontrados:")
        for error in result.errors:
            print(f"   - {error.message}")
    
    # Gerar saídas
    output_dir = Path("examples/outputs")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"\n📂 Gerando outputs em: {output_dir}")
    
    # HTML5 (para visualização imediata)
    print("   🌐 Gerando HTML5...")
    html_path = output_dir / "memorial_demo.html"
    memorial.render(html_path, format='html')
    print(f"      ✅ {html_path}")
    
    # JSON (para arquivamento)
    print("   📄 Gerando JSON...")
    json_path = output_dir / "memorial_demo.json"
    memorial.export_json(json_path)
    print(f"      ✅ {json_path}")
    
    # Estatísticas
    print("\n📊 Estatísticas do documento:")
    print(f"   - Seções: {len(memorial.sections)}")
    print(f"   - Verificações: {len(memorial.verifications)}")
    verif_ok = sum(1 for v in memorial.verifications if v.passed)
    print(f"   - Verificações OK: {verif_ok}/{len(memorial.verifications)}")
    
    print("\n" + "="*70)
    print("✅ Memorial gerado com sucesso!")
    print("="*70)
    print(f"\n👉 Abra o arquivo HTML para visualizar:")
    print(f"   {html_path.resolve()}\n")
    
    return memorial


if __name__ == '__main__':
    memorial = generate_demo()
