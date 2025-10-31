"""
TESTE FINAL DE RENDERIZAÇÃO - VALIDAÇÃO 100%
Valida TODA a cadeia: calc() -> metadata['latex'] -> render_pdf() -> PDF
"""

from pathlib import Path
from pymemorial import EngMemorial

print("="*80)
print("🧪 TESTE FINAL DE RENDERIZAÇÃO - VALIDAÇÃO COMPLETA")
print("="*80)

# TESTE 1: Criar memorial e fazer cálculo
print("\n[TESTE 1] Criando memorial com cálculo...")
mem = EngMemorial("Teste de Renderização")
mem.section("Cálculos de Teste")
mem.var("F_d", 100, "kN", "Força")
mem.var("d_y", 2.5, "m", "Distância")
mem.calc("M_k = F_d * d_y", "kN.m", "Momento fletor")

print(f"✅ {len(mem._content)} ContentBlocks criados")

# TESTE 2: Verificar metadata['latex'] no ContentBlock
print("\n[TESTE 2] Verificando metadata['latex'] no ContentBlock...")
calc_block = None
for block in mem._content:
    if block.type == "calculation":
        calc_block = block
        break

if calc_block:
    print(f"✅ ContentBlock de cálculo encontrado")
    print(f"   Type: {calc_block.type}")
    print(f"   Has metadata: {hasattr(calc_block, 'metadata')}")
    
    if hasattr(calc_block, 'metadata') and calc_block.metadata:
        print(f"   Metadata keys: {list(calc_block.metadata.keys())}")
        
        if 'latex' in calc_block.metadata:
            latex = calc_block.metadata['latex']
            print(f"   ✅✅✅ metadata['latex'] EXISTE!")
            print(f"   Length: {len(latex)} chars")
            print(f"   Content:\n{latex}")
        else:
            print(f"   ❌ 'latex' NÃO está em metadata!")
            exit(1)
    else:
        print(f"   ❌ Sem metadata!")
        exit(1)
else:
    print(f"❌ Nenhum ContentBlock de cálculo encontrado!")
    exit(1)

# TESTE 3: Verificar se render_pdf existe
print("\n[TESTE 3] Verificando método render_pdf()...")
if hasattr(mem, 'render_pdf'):
    print(f"✅ Método render_pdf() EXISTE")
else:
    print(f"❌ Método render_pdf() NÃO EXISTE!")
    print(f"   Métodos disponíveis: {[m for m in dir(mem) if 'render' in m.lower()]}")
    exit(1)

# TESTE 4: Gerar HTML de debug (SEM PDF)
print("\n[TESTE 4] Gerando HTML de debug...")
from jinja2 import Template

debug_template = Template("""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{{ title }}</title>
    <style>
        body { font-family: Arial; margin: 40px; }
        .block { border: 1px solid #ccc; padding: 10px; margin: 10px 0; }
        .success { background-color: #d4edda; }
        .error { background-color: #f8d7da; }
        .calculation { background: #f5f5f5; padding: 15px; font-family: monospace; white-space: pre-wrap; }
    </style>
</head>
<body>
    <h1>{{ title }}</h1>
    <h2>ContentBlocks:</h2>
    
    {% for block in content_blocks %}
    <div class="block {% if block.type == 'calculation' and block.metadata and block.metadata.latex %}success{% elif block.type == 'calculation' %}error{% endif %}">
        <p><strong>Type:</strong> {{ block.type }}</p>
        <p><strong>Has metadata:</strong> {{ block.metadata is not none }}</p>
        
        {% if block.metadata %}
            <p><strong>Metadata keys:</strong> {{ block.metadata.keys() | list }}</p>
            
            {% if block.metadata.latex %}
                <p style="color: green;"><strong>✅ HAS LATEX!</strong> ({{ block.metadata.latex|length }} chars)</p>
                <div class="calculation">{{ block.metadata.latex }}</div>
            {% else %}
                <p style="color: red;"><strong>❌ NO LATEX!</strong></p>
            {% endif %}
        {% else %}
            <p style="color: red;"><strong>❌ NO METADATA!</strong></p>
        {% endif %}
        
        <p><strong>Content (first 200 chars):</strong></p>
        <pre>{{ block.content[:200] }}</pre>
    </div>
    {% endfor %}
</body>
</html>
""")

html_debug = debug_template.render(
    title=mem.metadata.title,
    content_blocks=mem._content
)

debug_html_file = Path("debug_render_final.html")
with open(debug_html_file, 'w', encoding='utf-8') as f:
    f.write(html_debug)

print(f"✅ HTML de debug gerado: {debug_html_file}")
print(f"   Abra no navegador para verificar se metadata.latex está presente!")

# TESTE 5: Tentar renderizar PDF
print("\n[TESTE 5] Tentando renderizar PDF...")
try:
    pdf_file = Path("debug_render_final.pdf")
    mem.render_pdf(str(pdf_file))
    
    if pdf_file.exists():
        print(f"✅ PDF gerado com sucesso: {pdf_file}")
        print(f"   Tamanho: {pdf_file.stat().st_size} bytes")
        print(f"\n   ⚠️ AGORA ABRA O PDF E VERIFIQUE SE OS CÁLCULOS ESTÃO FORMATADOS!")
        print(f"   ⚠️ Procure por: M = F * d com valores substituídos")
    else:
        print(f"❌ PDF NÃO foi gerado!")
        exit(1)
        
except Exception as e:
    print(f"❌ ERRO ao gerar PDF: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "="*80)
print("✅ TESTE COMPLETO! Verifique os arquivos:")
print("   1. debug_render_final.html - Deve mostrar '✅ HAS LATEX!' em verde")
print("   2. debug_render_final.pdf - Deve mostrar cálculos FORMATADOS")
print("="*80)
