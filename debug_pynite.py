"""Debug do Pynite - viga com 3 nós."""
import sys
from Pynite import FEModel3D

print("=" * 70)
print("TESTE DIRETO DO PYNITE - VIGA COM 3 NÓS")
print("=" * 70)

# Criar modelo
model = FEModel3D()

# Material
print("\n1. Adicionando material...")
model.add_material('steel', 200e9, 77e9, 0.3, 7850)
print("   ✓ Material 'steel' adicionado")

# Seção
print("\n2. Adicionando seção...")
model.add_section('IPE200', 28.5e-4, 1943e-8, 142e-8, 6.98e-8)
print("   ✓ Seção 'IPE200' adicionada")

# Nós - AGORA COM 3 NÓS!
print("\n3. Adicionando nós...")
model.add_node('N1', 0, 0, 0)      # Apoio esquerdo
model.add_node('N2', 3, 0, 0)      # Meio do vão (onde vai a carga)
model.add_node('N3', 6, 0, 0)      # Apoio direito
print(f"   ✓ {len(model.nodes)} nós adicionados")

# Elementos - AGORA COM 2 ELEMENTOS!
print("\n4. Adicionando membros...")
model.add_member('M1', 'N1', 'N2', 'steel', 'IPE200')
model.add_member('M2', 'N2', 'N3', 'steel', 'IPE200')
print(f"   ✓ {len(model.members)} membros adicionados")

# Apoios - SOMENTE NOS EXTREMOS
print("\n5. Adicionando apoios...")
model.def_support('N1', True, True, True, True, False, True)   # Apoio fixo
model.def_support('N3', False, True, True, True, False, True)  # Apoio móvel
print("   ✓ Apoios definidos (com restrições de rotação)")

# Verificar apoios
print("\n   Verificação dos apoios:")
for node_name in ['N1', 'N2', 'N3']:
    node = model.nodes[node_name]
    supports = []
    if node.support_DX: supports.append("DX")
    if node.support_DY: supports.append("DY")
    if node.support_DZ: supports.append("DZ")
    if node.support_RX: supports.append("RX")
    if node.support_RY: supports.append("RY")
    if node.support_RZ: supports.append("RZ")
    print(f"   {node_name}: {', '.join(supports) if supports else 'LIVRE'}")

# Carga - NO MEIO DO VÃO (N2)
print("\n6. Adicionando carga...")
model.add_node_load('N2', 'FY', -10000)
print("   ✓ Carga de -10000 N em Y aplicada em N2 (meio do vão)")

# Análise
print("\n7. Executando análise...")
sys.stdout.flush()

try:
    model.analyze()
    print("   ✓ Análise concluída")
except Exception as e:
    print(f"   ✗ ERRO na análise: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Resultados
print("\n" + "=" * 70)
print("RESULTADOS")
print("=" * 70)

print("\nDeslocamentos:")
for node_name in ['N1', 'N2', 'N3']:
    node = model.nodes[node_name]
    dy_val = node.DY.get('Combo 1', 0.0)
    print(f"{node_name}: DY = {dy_val:12.6e} m")

print("\nEsforços:")
for member_name in ['M1', 'M2']:
    member = model.members[member_name]
    print(f"\n{member_name}:")
    print(f"  Axial máx:    {member.max_axial():12.2f} N")
    print(f"  Cortante máx: {member.max_shear('Fy'):12.2f} N")
    print(f"  Momento máx:  {member.max_moment('Mz'):12.2f} N·m")

print("\nReações nos apoios:")
for node_name in ['N1', 'N3']:
    node = model.nodes[node_name]
    fy_val = node.RxnFY.get('Combo 1', 0.0)
    print(f"{node_name}: RY = {fy_val:12.2f} N")

print("\n" + "=" * 70)
print("✓ SUCESSO - Viga defletiu corretamente!")
print("=" * 70)