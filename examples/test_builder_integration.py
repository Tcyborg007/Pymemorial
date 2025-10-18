from pymemorial.builder import MemorialBuilder

# Criar memorial
memorial = MemorialBuilder("Viga Biapoiada", author="Eng. Teste")

# Adicionar variáveis
memorial.add_variable("L", value=6.0, unit="m", description="Vão")
memorial.add_variable("q", value=15.0, unit="kN/m", description="Carga")

# Adicionar seção
memorial.add_section("Dados", level=1)
memorial.add_text("Vão: {{L}}, Carga: {{q}}")

# Adicionar cálculo
memorial.add_section("Cálculo", level=1)
memorial.add_equation("M = q * L**2 / 8")

# Computar
results = memorial.compute()
print(f"Momento: {list(results.values())[0]:.2f} kN.m")

# Exportar
data = memorial.build()
print("Memorial construído com sucesso!")
print(f"Título: {data['metadata']['title']}")
print(f"Variáveis: {list(data['variables'].keys())}")
print(f"Seções: {len(data['sections'])}")
