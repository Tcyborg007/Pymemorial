# tests/debug/test_pdf_generation_debug.py
"""
Debug script for PDF generation with WeasyPrint (PHASE 7.3).

Tests:
- WeasyPrintGenerator initialization
- PDF generation from memorial
- Page numbering
- Headers/Footers
- Table of contents
- Image embedding
- Error handling

Author: PyMemorial Team
Date: 2025-10-20
Phase: 7.3
"""

import logging

# Silence fontTools verbose logging
logging.getLogger('fontTools').setLevel(logging.WARNING)
logging.getLogger('fontTools.subset').setLevel(logging.WARNING)
logging.getLogger('fontTools.ttLib').setLevel(logging.WARNING)

# -*- coding: utf-8 -*-
import sys
import io

# Force UTF-8 encoding for Windows PowerShell
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from pathlib import Path
from datetime import datetime
import pandas as pd


from pymemorial.document import (
    Memorial,
    DocumentMetadata,
    NormCode,
    Revision,
    DocumentLanguage,
)
from pymemorial.document.generators import (
    WeasyPrintGenerator,
    GeneratorConfig,
    PageConfig,
    PDFMetadata,
    generate_pdf,
)

# ============================================================================
# SETUP
# ============================================================================

def create_test_memorial():
    """Create test memorial with content."""
    metadata = DocumentMetadata(
        title="Memorial de Cálculo - Teste PDF",
        author="Eng. João Silva",
        company="Engenharia Estrutural LTDA",
        code=NormCode.NBR8800_2024,
        project_number="PROJ-2025-001",
        revision=Revision(
            number="R01",
            date=datetime(2025, 10, 20),
            description="Teste de geração de PDF",
            author="João Silva",
            approved=True
        ),
        language=DocumentLanguage.PT_BR,
    )
    
    memorial = Memorial(metadata, template='nbr8800')
    print("✅ Memorial created")
    return memorial


def populate_memorial(memorial):
    """Add content to memorial."""
    print("\n" + "="*70)
    print("POPULATING MEMORIAL WITH CONTENT")
    print("="*70)
    
    # Section 1: Introduction
    memorial.add_section(
        title="Introdução",
        content="""
        Este memorial de cálculo apresenta a verificação estrutural de um pilar 
        metálico submetido à flexo-compressão, conforme os critérios estabelecidos 
        pela norma NBR 8800:2024.
        
        O pilar analisado faz parte da estrutura principal do edifício comercial, 
        localizado no eixo A-1 da edificação.
        """,
        level=1
    )
    
    # Section 2: Geometria
    memorial.add_section(
        title="Geometria e Propriedades da Seção",
        content="""
        A seção transversal adotada é um perfil W310x52, laminado a quente.
        """,
        level=1
    )
    
    # Add figure
    from PIL import Image
    img_dir = Path("outputs/debug/figures")
    img_dir.mkdir(parents=True, exist_ok=True)
    
    img = Image.new('RGB', (400, 300), color='lightblue')
    img_path = img_dir / "secao_transversal.png"
    img.save(img_path)
    
    memorial.add_figure(
        img_path,
        caption="Seção transversal do perfil W310x52",
        source="Catálogo Gerdau"
    )
    
    # Add table
    df = pd.DataFrame({
        'Propriedade': ['Área (A_g)', 'Momento de inércia (I_x)', 'Módulo resistente (W_x)', 'Raio de giração (r_x)'],
        'Valor': [66.6, 11200, 721, 13.0],
        'Unidade': ['cm²', 'cm⁴', 'cm³', 'cm']
    })
    
    memorial.add_table(
        df,
        caption="Propriedades geométricas do perfil W310x52",
        source="NBR 8800 (ABNT, 2024), Tabela A-1"
    )
    
    # Section 3: Carregamento
    memorial.add_section(
        title="Solicitações de Cálculo",
        content="""
        As solicitações de cálculo foram obtidas a partir da análise estrutural 
        global do edifício, considerando as combinações últimas de serviço.
        """,
        level=1
    )
    
    # Add equations
    memorial.add_equation(
        "N_Sd = 1.4 * N_g + 1.4 * N_q",
        description="Força normal de cálculo",
        variables={
            'N_Sd': 'Força normal de cálculo (kN)',
            'N_g': 'Força normal permanente (kN)',
            'N_q': 'Força normal acidental (kN)',
        }
    )
    
    memorial.add_equation(
        "M_Sd = 1.4 * M_g + 1.4 * M_q",
        description="Momento fletor de cálculo",
        result="125.5 kNm"
    )
    
    # Section 4: Verificação
    memorial.add_section(
        title="Verificação da Resistência",
        content="""
        A verificação da resistência do pilar é realizada considerando o critério 
        de interação flexo-compressão estabelecido pela NBR 8800:2024.
        """,
        level=1
    )
    
    memorial.add_equation(
        "N_Sd / N_Rd + M_Sd / M_Rd <= 1.0",
        description="Critério de interação flexo-compressão",
        result="0.85 ≤ 1.0 ✓ OK"
    )
    
    print(f"✅ Memorial populated:")
    print(f"   - Sections: {len(memorial.sections)}")
    print(f"   - Figures: {len(memorial.figures)}")
    print(f"   - Tables: {len(memorial.tables)}")
    print(f"   - Equations: {len(memorial.equations)}")


# ============================================================================
# TEST 1: Basic PDF generation
# ============================================================================

def test_basic_pdf_generation(memorial):
    """Test basic PDF generation."""
    print("\n" + "="*70)
    print("TEST 1: Basic PDF Generation")
    print("="*70)
    
    output_dir = Path("outputs/debug/pdfs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "test_basic.pdf"
    
    try:
        # Generate PDF using convenience function
        pdf_path = generate_pdf(memorial, output_path)
        
        print(f"✅ PDF generated: {pdf_path}")
        print(f"   Size: {pdf_path.stat().st_size / 1024:.1f} KB")
        print(f"   Exists: {pdf_path.exists()}")
        
    except Exception as e:
        print(f"❌ PDF generation FAILED: {e}")
        import traceback
        traceback.print_exc()


# ============================================================================
# TEST 2: PDF with custom configuration
# ============================================================================

def test_custom_config_pdf(memorial):
    """Test PDF with custom configuration."""
    print("\n" + "="*70)
    print("TEST 2: PDF with Custom Configuration")
    print("="*70)
    
    # Create custom config
    config = GeneratorConfig(
        page=PageConfig(
            size='A4',
            orientation='portrait',
            margin_top=40.0,
            margin_bottom=30.0,
            margin_left=35.0,
            margin_right=25.0,
        ),
        metadata=PDFMetadata(
            title="Memorial de Cálculo - Pilar PM-1",
            author="Eng. João Silva",
            subject="Verificação estrutural de pilar metálico",
            keywords=['estruturas metálicas', 'NBR 8800', 'pilar'],
        ),
        page_numbering=True,
        show_header=True,
        show_footer=True,
        header_text="Memorial de Cálculo - PROJ-2025-001",
        debug=True,
        save_intermediate_html=True,
    )
    
    output_dir = Path("outputs/debug/pdfs")
    output_path = output_dir / "test_custom_config.pdf"
    
    try:
        generator = WeasyPrintGenerator(config)
        pdf_path = generator.generate(memorial, output_path)
        
        print(f"✅ PDF with custom config generated: {pdf_path}")
        print(f"   Size: {pdf_path.stat().st_size / 1024:.1f} KB")
        
        # Check if intermediate HTML was saved
        html_path = output_path.with_suffix('.html')
        if html_path.exists():
            print(f"✅ Intermediate HTML saved: {html_path}")
        
    except Exception as e:
        print(f"❌ Custom config PDF FAILED: {e}")
        import traceback
        traceback.print_exc()


# ============================================================================
# TEST 3: PDF using memorial.to_pdf()
# ============================================================================

def test_memorial_to_pdf(memorial):
    """Test memorial.to_pdf() convenience method."""
    print("\n" + "="*70)
    print("TEST 3: memorial.to_pdf() Convenience Method")
    print("="*70)
    
    output_dir = Path("outputs/debug/pdfs")
    output_path = output_dir / "test_memorial_to_pdf.pdf"
    
    try:
        pdf_path = memorial.to_pdf(output_path)
        
        print(f"✅ PDF via to_pdf() generated: {pdf_path}")
        print(f"   Size: {pdf_path.stat().st_size / 1024:.1f} KB")
        
    except Exception as e:
        print(f"❌ memorial.to_pdf() FAILED: {e}")
        import traceback
        traceback.print_exc()


# ============================================================================
# TEST 4: PDF to bytes (in-memory)
# ============================================================================

def test_pdf_to_bytes(memorial):
    """Test PDF generation to bytes."""
    print("\n" + "="*70)
    print("TEST 4: PDF to Bytes (In-Memory)")
    print("="*70)
    
    try:
        generator = WeasyPrintGenerator()
        pdf_bytes = generator.generate_to_bytes(memorial)
        
        print(f"✅ PDF generated to bytes")
        print(f"   Size: {len(pdf_bytes) / 1024:.1f} KB")
        print(f"   Type: {type(pdf_bytes)}")
        
        # Save bytes to file
        output_dir = Path("outputs/debug/pdfs")
        output_path = output_dir / "test_from_bytes.pdf"
        
        output_path.write_bytes(pdf_bytes)
        print(f"✅ Bytes saved to: {output_path}")
        
    except Exception as e:
        print(f"❌ PDF to bytes FAILED: {e}")
        import traceback
        traceback.print_exc()


# ============================================================================
# TEST 5: Error handling
# ============================================================================

def test_error_handling():
    """Test error handling."""
    print("\n" + "="*70)
    print("TEST 5: Error Handling")
    print("="*70)
    
    # Test 5.1: Invalid generator
    try:
        metadata = DocumentMetadata(
            title="Test",
            author="Test",
            company="Test",
            code=NormCode.NBR8800_2024,
        )
        memorial = Memorial(metadata)
        
        memorial.to_pdf("test.pdf", generator="invalid_generator")
        print("❌ Should have raised ValueError")
    except ValueError as e:
        print(f"✅ Correctly raised ValueError: {str(e)[:50]}...")
    except Exception as e:
        print(f"❌ Wrong exception: {e}")


# ============================================================================
# MAIN DEBUG RUNNER
# ============================================================================

def main():
    """Run all debug tests."""
    print("="*70)
    print("PyMemorial PHASE 7.3 - PDF Generation Debug")
    print("="*70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python: {sys.version.split()[0]}")
    print()
    
    try:
        # Create and populate memorial
        memorial = create_test_memorial()
        populate_memorial(memorial)
        
        # Run tests
        test_basic_pdf_generation(memorial)
        test_custom_config_pdf(memorial)
        test_memorial_to_pdf(memorial)
        test_pdf_to_bytes(memorial)
        test_error_handling()
        
        # Summary
        print("\n" + "="*70)
        print("DEBUG SUMMARY")
        print("="*70)
        
        output_dir = Path("outputs/debug/pdfs")
        if output_dir.exists():
            pdf_files = list(output_dir.glob("*.pdf"))
            print(f"✅ PDFs generated: {len(pdf_files)}")
            for pdf in pdf_files:
                size_kb = pdf.stat().st_size / 1024
                print(f"   - {pdf.name} ({size_kb:.1f} KB)")
        
        print()
        print("🎉 Debug completed successfully!")
        print()
        print("📄 Open PDFs to verify:")
        print(f"   - outputs/debug/pdfs/test_basic.pdf")
        print(f"   - outputs/debug/pdfs/test_custom_config.pdf")
        print(f"   - outputs/debug/pdfs/test_memorial_to_pdf.pdf")
        
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
