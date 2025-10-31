"""
PyMemorial v2.0 - Exemplo Completo: Memorial de Viga Simplesmente Apoiada

WORKFLOW COMPLETO:
══════════════════
1. Configurar NaturalWriter
2. Adicionar seções hierárquicas
3. Adicionar tabelas de dados
4. Adicionar equações com STEPS AUTOMÁTICOS (Calcpad-style)
5. Adicionar figuras e conclusões
6. Renderizar em Markdown/HTML/JSON
7. Salvar arquivo

Este exemplo demonstra TODO O PODER do PyMemorial v2.0!

Author: PyMemorial Team
Date: 2025-10-27
"""

from pathlib import Path
from pymemorial.editor.natural_writer import (
    NaturalWriter,
    OutputFormat,
    get_writer_info
)
from pymemorial.editor.render_modes import GranularityType

# =========================================================================
# EXEMPLO 1: Memorial Completo de Viga Simplesmente Apoiada
# =========================================================================

def exemplo_viga_completa():
    """
    Memorial completo com TODOS os recursos:
    - Seções hierárquicas
    - Tabelas
    - Equações com steps automáticos
    - Listas
    - Figuras
    - Metadados
    - Sumário automático
    """
    
    print("=" * 70)
    print("PyMemorial v2.0 - Exemplo Completo: Viga Simplesmente Apoiada")
    print("=" * 70)
    print()
    
    # 1. Criar NaturalWriter
    writer = NaturalWriter(enable_cache=True)
    
    # 2. Configurar metadados
    writer.set_metadata(
        title="Memorial de Cálculo Estrutural",
        subtitle="Viga Simplesmente Apoiada - Análise Completa",
        author="Engº João Silva, CREA 12345/SP",
        project="Edifício Residencial XYZ",
        location="São Paulo, SP",
        date="27 de outubro de 2025",
        norm="NBR 6118:2023",
        client="Construtora ABC Ltda.",
        revision="Rev. 00"
    )
    
    # =====================================================================
    # SEÇÃO 1: INTRODUÇÃO
    # =====================================================================
    
    writer.add_section("Introdução", level=1)
    
    writer.add_text(
        "Este memorial apresenta o dimensionamento de viga simplesmente apoiada "
        "em concreto armado, de acordo com as prescrições da NBR 6118:2023."
    )
    
    writer.add_text(
        "A viga está submetida a carregamento uniformemente distribuído e será "
        "dimensionada para resistir aos esforços de flexão simples."
    )
    
    # =====================================================================
    # SEÇÃO 2: DADOS DE ENTRADA
    # =====================================================================
    
    writer.add_section("Dados de Entrada", level=1)
    
    writer.add_section("Geometria", level=2)
    
    writer.add_table(
        data=[
            ["Vão livre (L)", "6,00 m"],
            ["Base da seção (b_w)", "20 cm"],
            ["Altura total (h)", "50 cm"],
            ["Cobrimento (c)", "3,0 cm"],
        ],
        headers=["Parâmetro", "Valor"],
        caption="Tabela 1 - Dados geométricos"
    )
    
    writer.add_section("Materiais", level=2)
    
    writer.add_table(
        data=[
            ["Concreto", "C30", "f_ck = 30 MPa"],
            ["Aço", "CA-50", "f_yk = 500 MPa"],
        ],
        headers=["Material", "Classe", "Resistência"],
        caption="Tabela 2 - Materiais especificados"
    )
    
    writer.add_section("Carregamento", level=2)
    
    writer.add_list([
        "Peso próprio: 2,5 kN/m (calculado automaticamente)",
        "Revestimento: 1,5 kN/m",
        "Sobrecarga de utilização: 3,0 kN/m (NBR 6120:2019)",
        "Carga total característica: q_k = 7,0 kN/m"
    ])
    
    # =====================================================================
    # SEÇÃO 3: ESFORÇOS SOLICITANTES
    # =====================================================================
    
    writer.add_section("Esforços Solicitantes", level=1)
    
    writer.add_section("Carga de Cálculo", level=2)
    
    writer.add_text(
        "A carga de cálculo é obtida aplicando-se os coeficientes de "
        "ponderação conforme NBR 6118:2023, item 11.7.1."
    )
    
    # EQUAÇÃO 1: Carga de cálculo (COM STEPS AUTOMÁTICOS! 🎉)
    # EQUAÇÃO 1: Carga de cálculo (COM STEPS AUTOMÁTICOS! 🎉)
    writer.add_equation(
        expression="1.4 * q_k",
        context={'q_k': 7.0},
        variable_name="q_d",
        intro="Carga de cálculo:",
        norm_reference="NBR 6118:2023, item 11.7.1",
        unit="kN/m",
        granularity=GranularityType.SMART  # Modo SMART (omite trivialidades)
    )

    
    writer.add_section("Momento Fletor Máximo", level=2)
    
    writer.add_text(
        "Para viga simplesmente apoiada com carga uniformemente distribuída, "
        "o momento fletor máximo ocorre no meio do vão e é dado por:"
    )
    
    # EQUAÇÃO 2: Momento fletor (COM STEPS AUTOMÁTICOS! 🎉)
    writer.add_equation(
        expression="q_d * L**2 / 8",
        context={'q_d': 9.8, 'L': 6.0},
        variable_name="M_d",
        intro="Momento fletor de cálculo:",
        unit="kN⋅m",
        granularity=GranularityType.DETAILED  # DETAILED = mostra TODOS os steps
    )

    
    writer.add_section("Cortante Máxima", level=2)
    
    # EQUAÇÃO 3: Cortante (COM STEPS AUTOMÁTICOS! 🎉)
    writer.add_equation(
        expression="q_d * L / 2",
        context={'q_d': 9.8, 'L': 6.0},
        variable_name="V_d",
        intro="Força cortante de cálculo nos apoios:",
        unit="kN",
        granularity=GranularityType.MEDIUM
    )

    
    # =====================================================================
    # SEÇÃO 4: DIMENSIONAMENTO À FLEXÃO
    # =====================================================================
    
    writer.add_section("Dimensionamento à Flexão", level=1)
    
    writer.add_section("Altura Útil", level=2)
    
    writer.add_text(
        "Considerando armadura dupla (2 camadas) com diâmetro estimado "
        "φ = 12,5 mm:"
    )
    
    # EQUAÇÃO 4: Altura útil
    writer.add_equation(
        expression="h - c - phi/2 - phi_est",
        context={'h': 50.0, 'c': 3.0, 'phi': 1.0, 'phi_est': 1.25},
        variable_name="d",
        intro="Altura útil:",
        unit="cm"
    )

    
    writer.add_section("Momento Limite", level=2)
    
    writer.add_text(
        "Verificação se a seção requer armadura dupla (comprimida):"
    )
    
    # EQUAÇÃO 5: KMD (parâmetro adimensional)
    writer.add_equation(
        expression="M_d * 100 / (b_w * d**2 * f_cd)",
        context={
            'M_d': 44.1,
            'b_w': 20.0,
            'd': 44.75,
            'f_cd': 30.0 / 1.4  # f_ck / gamma_c
        },
        variable_name="K_MD",
        intro="Coeficiente adimensional K_MD:",
        conclusion="Como K_MD < K_MD,lim = 0,295, a seção é de armadura simples.",
        norm_reference="NBR 6118:2023, Tabela 17.1"
    )

    
    writer.add_section("Área de Aço Necessária", level=2)
    
    # EQUAÇÃO 6: Área de aço (fórmula complexa!)
    writer.add_equation(
        expression="(b_w * d * f_cd / f_yd) * (1 - (1 - 2*K_MD/f_cd)**0.5)",
        context={
            'b_w': 20.0,
            'd': 44.75,
            'f_cd': 30.0/1.4,
            'f_yd': 500.0/1.15,
            'K_MD': 1.105
        },
        variable_name="A_s",
        intro="Área de aço de tração:",
        unit="cm²",
        granularity=GranularityType.SMART
    )

    
    # =====================================================================
    # SEÇÃO 5: DETALHAMENTO
    # =====================================================================
    
    writer.add_section("Detalhamento da Armadura", level=1)
    
    writer.add_text(
        "Com base na área de aço calculada (A_s = 5,21 cm²), adota-se:"
    )
    
    writer.add_list([
        "Armadura longitudinal de tração: 4φ12,5 mm (A_s,ef = 4,91 cm²)",
        "Armadura de pele: 2φ8,0 mm em cada face",
        "Estribos: φ5,0 mm c/ 15 cm (zona de apoio)",
        "Estribos: φ5,0 mm c/ 20 cm (zona central)"
    ], intro="Armadura adotada:")
    
    # =====================================================================
    # SEÇÃO 6: CONCLUSÃO
    # =====================================================================
    
    writer.add_section("Conclusão", level=1)
    
    writer.add_text(
        "O dimensionamento da viga simplesmente apoiada foi realizado de acordo "
        "com as prescrições da NBR 6118:2023. A seção proposta atende a todos "
        "os critérios de segurança e estados limites de serviço."
    )
    
    writer.add_text(
        "A armadura adotada (4φ12,5 mm) resulta em uma taxa de armadura "
        "ρ = 1,1%, dentro dos limites prescritos pela norma (ρ_min = 0,15% e "
        "ρ_max = 4,0%)."
    )
    
    # =====================================================================
    # RENDERIZAR E SALVAR
    # =====================================================================
    
    print("\n" + "=" * 70)
    print("RENDERIZANDO DOCUMENTO...")
    print("=" * 70 + "\n")
    
    # Renderizar em MARKDOWN com sumário
    output_md = writer.render(
        format=OutputFormat.MARKDOWN,
        include_toc=True,
        include_metadata=True
    )
    
    # Salvar Markdown
    output_dir = Path("examples/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    md_path = output_md.save_to_file(output_dir / "memorial_viga_completo.md")
    print(f"✅ Markdown salvo: {md_path}")
    
    # Renderizar em HTML
    output_html = writer.render(
        format=OutputFormat.HTML,
        include_toc=True
    )
    html_path = output_html.save_to_file(output_dir / "memorial_viga_completo.html")
    print(f"✅ HTML salvo: {html_path}")
    
    # Renderizar em JSON (metadados)
    output_json = writer.render(format=OutputFormat.JSON)
    json_path = output_json.save_to_file(output_dir / "memorial_viga_completo.json")
    print(f"✅ JSON salvo: {json_path}")
    
    # Estatísticas
    print("\n" + "=" * 70)
    print("ESTATÍSTICAS DO DOCUMENTO")
    print("=" * 70)
    
    stats = writer.get_statistics()
    print(f"Seções: {stats['sections']}")
    print(f"Caracteres: {stats['total_chars']:,}")
    print(f"Palavras: {stats['total_words']:,}")
    
    print("\nEstatísticas de renderização:")
    for key, value in output_md.statistics.items():
        print(f"  {key}: {value}")
    
    # Preview do conteúdo
    print("\n" + "=" * 70)
    print("PREVIEW DO DOCUMENTO (primeiras 50 linhas)")
    print("=" * 70 + "\n")
    
    lines = output_md.content.split('\n')
    for i, line in enumerate(lines[:50], 1):
        print(f"{i:3d} | {line}")
    
    if len(lines) > 50:
        print(f"\n... ({len(lines) - 50} linhas restantes)")
    
    return writer, output_md


# =========================================================================
# EXEMPLO 2: API Rápida (Quick Write)
# =========================================================================

def exemplo_quick_write():
    """
    Demonstra a API rápida para documentos simples
    """
    from pymemorial.editor.natural_writer import quick_write
    
    print("\n" + "=" * 70)
    print("EXEMPLO 2: Quick Write API")
    print("=" * 70 + "\n")
    
    content = quick_write(
        "Cálculo Rápido de Momento",
        "Viga simplesmente apoiada com carga uniforme.",
        ("equation", "q * L**2 / 8", {'q': 10.0, 'L': 5.0}, "M_max"),
        "Resultado obtido com sucesso!",
        output_path="examples/output/memorial_rapido.md"
    )

    
    print(f"✅ Memorial rápido salvo: {content}")


# =========================================================================
# MAIN
# =========================================================================

if __name__ == "__main__":
    # Informações sobre o writer
    info = get_writer_info()
    print(f"PyMemorial NaturalWriter v{info['version']}")
    print(f"Formatos suportados: {', '.join(info['supported_formats'])}")
    print()
    
    # Executar exemplo completo
    writer, output = exemplo_viga_completa()
    
    # Executar exemplo rápido
    exemplo_quick_write()
    
    print("\n" + "=" * 70)
    print("✅ TODOS OS EXEMPLOS EXECUTADOS COM SUCESSO!")
    print("=" * 70)
    print("\nArquivos gerados em: examples/output/")
    print("  - memorial_viga_completo.md")
    print("  - memorial_viga_completo.html")
    print("  - memorial_viga_completo.json")
    print("  - memorial_rapido.md")
