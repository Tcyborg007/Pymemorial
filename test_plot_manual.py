"""
Teste manual de plotagem - Execute este arquivo diretamente.
"""
from pymemorial.sections import CompositeSection, CompositeType, SectionFactory

# Teste 1: Viga mista
print("Criando viga mista...")
viga = CompositeSection("VM-1", code="NBR8800:2024")
viga.build_composite_beam(
    steel_profile="W310x52",
    slab_width=2.5,
    slab_height=0.20
)
viga.add_shear_connectors(diameter=19, spacing=150)

print("Salvando viga_mista.png...")
viga.plot_geometry(filename="viga_mista.png")
print("✅ Imagem salva!")

# Ver propriedades da viga
props_viga = viga.get_properties()
print(f"\n📊 Propriedades da Viga Mista:")
print(f"  Área: {props_viga.area:.4f} m²")
print(f"  Ixx: {props_viga.ixx:.6e} m⁴")
print(f"  Centro Y: {props_viga.cy:.3f} m")
print(f"  Razão modular n₀: {viga.n0:.2f}")

# Capacidade dos conectores
if viga.shear_connectors:
    for i, c in enumerate(viga.shear_connectors):
        cap = viga.calculate_shear_connector_capacity(c)
        print(f"  Conector {i+1} (Ø{c.diameter}mm): QRd = {cap:.2f} kN")

# Teste 2: Pilar preenchido
print("\n" + "="*60)
print("Criando pilar preenchido...")
pilar = CompositeSection(
    "PM-1",
    code="NBR8800:2024",
    composite_type=CompositeType.FILLED_COLUMN  # ← CORRETO
)
pilar.build_filled_column(outer_diameter=0.40, wall_thickness=0.012)

print("Salvando pilar_preenchido.png...")
pilar.plot_geometry(filename="pilar_preenchido.png")
print("✅ Imagem salva!")

# Classificação NBR 8800:2024
classificacao = pilar.classify_section_nbr8800()
info = pilar.get_nbr8800_info()

print(f"\n📊 Classificação NBR 8800:2024:")
print(f"  Seção: {classificacao.upper()}")
print(f"  D = {info['D_mm']:.1f} mm")
print(f"  t = {info['t_mm']:.1f} mm")
print(f"  D/t = {info['D/t_ratio']:.2f}")
print(f"  Redução de rigidez: {info['stiffness_reduction']} (64%)")

# Propriedades do pilar
props_pilar = pilar.get_properties()
print(f"\n📊 Propriedades do Pilar:")
print(f"  Área: {props_pilar.area:.6f} m²")
print(f"  Ixx: {props_pilar.ixx:.6e} m⁴")
print(f"  Iyy: {props_pilar.iyy:.6e} m⁴")
print(f"  Razão Ixx/Iyy: {props_pilar.ixx/props_pilar.iyy if props_pilar.iyy > 0 else 0:.3f} (deve ser ~1.0 para circular)")

print("\n" + "="*60)
print("✅ Testes manuais concluídos!")
print("\nArquivos gerados:")
print("  📁 viga_mista.png")
print("  📁 pilar_preenchido.png")
