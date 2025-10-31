"""
Teste Simples da API Memorial
==============================
"""

# Importar a API
from pymemorial.api import Memorial, MemorialBuilder

print("="*80)
print("TESTE 1: MODO ULTRA-COMPACTO")
print("="*80)

# Criar memorial com parsing automático
mem = Memorial("Viga Biapoiada - Exemplo Simples")

mem.write("""
# Dados de Entrada
q = 15.0 kN/m  # Carga distribuída
L = 6.0 m      # Vão da viga

# Cálculo do Momento Máximo
M_max = q * L**2 / 8  # Momento fletor máximo
""")

print("\n✅ Memorial criado!")
print(f"   Título: {mem.title}")
print(f"   Seções: {len(mem._sections)}")
print(f"   Variáveis definidas: {list(mem._variables.keys())}")

# Exibir estrutura
print("\n📋 Estrutura do memorial:")
for i, section in enumerate(mem._sections, 1):
    print(f"   {i}. {section.title}")
    for item in section.content:
        if item['type'] == 'variable':
            print(f"      • {item['name']} = {item['value']} {item['unit']}")
        elif item['type'] == 'calculation':
            print(f"      • {item['expression']}")

# Exportar para markdown
print("\n📄 Exportando para Markdown:")
md = mem.to_markdown()
print(md)

# Salvar
mem.save("output/viga_simples.pdf")

print("\n" + "="*80)
print("TESTE 2: MODO PROGRAMÁTICO")
print("="*80)

# Criar memorial programático
mem2 = Memorial("Pilar PM-1", norm="NBR 6118:2023")

mem2.section("1. Geometria")
mem2.var("b", 20, "cm", "Largura da seção")
mem2.var("h", 50, "cm", "Altura da seção")
mem2.calc("A = b * h", unit="cm²", desc="Área da seção")

mem2.section("2. Verificações")
mem2.verify("A >= 360", norm="NBR 6118 item 17.3.5.3.1", desc="Área mínima")

print("\n✅ Memorial programático criado!")
print(f"   Seções: {len(mem2._sections)}")

# Exportar HTML
print("\n📄 HTML:")
html = mem2.to_html()
print(html)

mem2.save("output/pilar_pm1.pdf")

print("\n" + "="*80)
print("TESTE 3: BUILDER PATTERN")
print("="*80)

# Usar builder pattern
mem3 = (MemorialBuilder()
    .title("Laje L1")
    .norm("NBR 6118:2023")
    .section("Cargas")
    .var("g", 5.0, "kN/m²")
    .var("q", 3.0, "kN/m²")
    .calc("p = g + q")
    .build()
)

print("\n✅ Memorial via Builder criado!")
print(f"   {mem3}")

mem3.save("output/laje_l1.pdf")

print("\n🎉 Todos os testes passaram!")
