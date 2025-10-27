# examples/core/memorial_with_smarttext.py
"""Memorial com SmartText - Processamento Inteligente"""

from pathlib import Path
from datetime import datetime
import logging

logging.getLogger('pymemorial').setLevel(logging.WARNING)

from pymemorial.core import get_core_bundle
from pymemorial.document import Memorial
from pymemorial.document.base_document import DocumentMetadata, NormCode, DocumentLanguage
from pymemorial.document.generators import get_generator

print("🧠 Memorial com SmartText\n")

# METADATA
metadata = DocumentMetadata(
    title="Memorial de Cálculo - Viga V1 (SmartText)",
    author="Eng. João Silva",
    company="Construtora ABC Ltda.",
    date=datetime.now().strftime("%Y-%m-%d"),
    language=DocumentLanguage.PT_BR,
    norm_code=NormCode.NBR6118_2023,
    revision="1.0",
    keywords=["concreto armado", "viga"]
)

# CORE
bundle = get_core_bundle(enable_cache=True)

b = bundle['variable'].create('b', 20.0, unit='cm')
h = bundle['variable'].create('h', 50.0, unit='cm')
M_k = bundle['variable'].create('M_k', 112.5, unit='kN.m')
gamma_f = bundle['variable'].create('gamma_f', 1.4)
f_ck = bundle['variable'].create('f_ck', 30.0, unit='MPa')
gamma_c = bundle['variable'].create('gamma_c', 1.4)

calc = bundle['calculator']
eq1 = bundle['equation'].create('M_d = M_k * gamma_f', {'M_k': M_k, 'gamma_f': gamma_f})
eq2 = bundle['equation'].create('f_cd = f_ck / gamma_c', {'f_ck': f_ck, 'gamma_c': gamma_c})

calc.add_equation(eq1)
calc.add_equation(eq2)

results = calc.evaluate_all()
M_d = results[id(eq1)]
f_cd = results[id(eq2)]

# MEMORIAL
memorial = Memorial(
    metadata=metadata,
    template='nbr6118',
    auto_toc=False,
    auto_title_page=True
)

# SEÇÃO 1: INTRODUÇÃO (NORMAL)
memorial.add_section('1. INTRODUÇÃO', level=1)
memorial.add_paragraph(
    f"Memorial de cálculo conforme {metadata.norm_code}.",
    processing_mode='normal'
)

# SEÇÃO 2: GEOMETRIA (SMART)
memorial.add_section('2. GEOMETRIA', level=1)
memorial.add_paragraph(
    f"Base: b = {b.magnitude} cm",
    processing_mode='smart'  # ✅ SmartText formata com bold e símbolos
)
memorial.add_paragraph(
    f"Altura: h = {h.magnitude} cm",
    processing_mode='smart'
)

# SEÇÃO 3: CÁLCULOS (DETAILED)
memorial.add_section('3. CÁLCULOS', level=1)

memorial.add_section('3.1. Momento de Cálculo', level=2)
memorial.add_paragraph(
    "Cálculo do momento de cálculo conforme NBR 6118:2023",
    processing_mode='normal'
)
memorial.add_equation(eq1)
memorial.add_paragraph(
    f"Calcular M_d com M_k={M_k.magnitude} e gamma_f={gamma_f.magnitude}",
    processing_mode='detailed'  # ✅ Gera passos detalhados!
)
memorial.add_paragraph(
    f"Resultado: M_d = {M_d:.2f} kN.m",
    processing_mode='smart'
)

memorial.add_section('3.2. Resistência do Concreto', level=2)
memorial.add_equation(eq2)
memorial.add_paragraph(
    f"f_cd = {f_cd:.2f} MPa",
    processing_mode='smart'
)

# SEÇÃO 4: CONCLUSÃO
memorial.add_section('4. CONCLUSÃO', level=1)
memorial.add_paragraph(
    f"Momento de cálculo determinado: M_d = {M_d:.2f} kN.m",
    processing_mode='smart'
)

# GENERATE
output_dir = Path('output')
output_dir.mkdir(exist_ok=True)

pdf_path = output_dir / 'memorial_smarttext.pdf'
get_generator('weasyprint').generate(memorial, str(pdf_path), style='nbr')

print(f"✅ PDF: {pdf_path.absolute()}")
print(f"   {pdf_path.stat().st_size:,} bytes\n")
print("🎉 Memorial com SmartText gerado!")
