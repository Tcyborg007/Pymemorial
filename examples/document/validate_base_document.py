# examples/document/validate_base_document.py
"""
Practical validation script for BaseDocument.

This script performs end-to-end testing with debug output to validate
all functionality of base_document.py.

Author: PyMemorial Team
Date: 2025-10-19
"""

import logging
import sys
from pathlib import Path
from datetime import datetime

# Setup logging (Windows-compatible, without emojis)
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('validate_base_document.log', encoding='utf-8')
    ]
)


logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from pymemorial.document.base_document import (
    BaseDocument,
    DocumentMetadata,
    NormCode,
    Revision,
    DocumentLanguage,
    ValidationResult
)

# ============================================================================
# CONCRETE IMPLEMENTATION FOR TESTING
# ============================================================================

class TestDocument(BaseDocument):
    """Concrete implementation of BaseDocument for testing."""
    
    def render(self, output_path: Path, format: str = 'pdf', **kwargs) -> Path:
        """Mock render method."""
        logger.info(f"📄 Rendering document to: {output_path}")
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(f"Mock {format.upper()} document")
        return output_path
    
    def validate(self) -> ValidationResult:
        """Mock validation method."""
        logger.info("✓ Validating document...")
        return ValidationResult(valid=True, errors=[], warnings=[])
    
    def to_dict(self):
        """Serialize to dict."""
        return {
            'metadata': {
                'title': self.metadata.title,
                'author': self.metadata.author,
                'company': self.metadata.company,
                'code': str(self.metadata.code),
                'revision': self.metadata.revision.number
            },
            'sections': [
                {'number': s.number, 'title': s.title}
                for s in self.sections
            ],
            'figures': [
                {'number': f.number, 'caption': f.caption}
                for f in self.figures
            ],
            'tables': [
                {'number': t.number, 'caption': t.caption}
                for t in self.tables
            ],
            'equations': [
                {'number': e.number, 'latex': e.latex}
                for e in self.equations
            ]
        }


# ============================================================================
# VALIDATION TESTS
# ============================================================================

def test_metadata_creation():
    """Test 1: Metadata creation and validation."""
    logger.info("\n" + "="*80)
    logger.info("TEST 1: METADATA CREATION")
    logger.info("="*80)
    
    try:
        metadata = DocumentMetadata(
            title="Memorial de Cálculo - Pilar PM-1",
            author="Eng. João Silva, CREA: 12345/SP",
            company="Estrutural Engenharia LTDA",
            code=NormCode.NBR8800_2024,
            project_number="EST-2024-001",
            revision=Revision(
                number="R00",
                date=datetime(2025, 10, 19),
                description="Emissão inicial",
                author="João Silva",
                approved=False
            ),
            language=DocumentLanguage.PT_BR,
            keywords=['pilar misto', 'NBR 8800', 'concreto preenchido'],
            abstract="Este memorial apresenta o dimensionamento do pilar misto PM-1."
        )
        
        logger.info("[PASS] Metadata created successfully")  # Removido emoji ✅
        logger.info(f"   Title: {metadata.title}")
        logger.info(f"   Author: {metadata.author}")
        logger.info(f"   Company: {metadata.company}")
        logger.info(f"   Code: {metadata.code}")
        logger.info(f"   Revision: {metadata.revision.number}")
        
        return True, metadata
        
    except Exception as e:
        logger.error(f"[FAIL] Metadata creation failed: {e}")  # Removido emoji ❌
        return False, None



def test_document_initialization(metadata):
    """Test 2: Document initialization."""
    logger.info("\n" + "="*80)
    logger.info("TEST 2: DOCUMENT INITIALIZATION")
    logger.info("="*80)
    
    try:
        doc = TestDocument(metadata)
        
        logger.info(f"✅ Document initialized: {doc}")
        logger.info(f"   Sections: {len(doc.sections)}")
        logger.info(f"   Figures: {len(doc.figures)}")
        logger.info(f"   Tables: {len(doc.tables)}")
        
        return True, doc
        
    except Exception as e:
        logger.error(f"❌ Document initialization failed: {e}")
        return False, None


def test_section_numbering(doc):
    """Test 3: Section hierarchical numbering."""
    logger.info("\n" + "="*80)
    logger.info("TEST 3: SECTION NUMBERING")
    logger.info("="*80)
    
    try:
        # Add hierarchical sections
        sec1 = doc.add_section("Introdução", "Este memorial...", level=1)
        logger.info(f"✅ Added section: {sec1.number} - {sec1.title}")
        
        sec2 = doc.add_section("Objetivo", "Dimensionar...", level=2)
        logger.info(f"✅ Added section: {sec2.number} - {sec2.title}")
        
        sec3 = doc.add_section("Escopo", "Este documento...", level=2)
        logger.info(f"✅ Added section: {sec3.number} - {sec3.title}")
        
        sec4 = doc.add_section("Metodologia", "A metodologia...", level=1)
        logger.info(f"✅ Added section: {sec4.number} - {sec4.title}")
        
        sec5 = doc.add_section("Normas Aplicáveis", "...", level=2)
        logger.info(f"✅ Added section: {sec5.number} - {sec5.title}")
        
        # Validate numbering
        expected = ["1", "1.1", "1.2", "2", "2.1"]
        actual = [s.number for s in doc.sections]
        
        assert actual == expected, f"Numbering mismatch: {actual} != {expected}"
        
        logger.info(f"✅ Section numbering validated: {actual}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Section numbering failed: {e}")
        return False


def test_figure_management(doc):
    """Test 4: Figure management."""
    logger.info("\n" + "="*80)
    logger.info("TEST 4: FIGURE MANAGEMENT")
    logger.info("="*80)
    
    try:
        # Create temp figure
        temp_dir = Path("outputs") / "test_figures"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        fig_path = temp_dir / "test_figure.png"
        fig_path.write_bytes(b'PNG fake data')
        
        # Add figure
        figure = doc.add_figure(
            fig_path,
            caption="Diagrama de interação P-M do pilar PM-1",
            width="80%"
        )
        
        logger.info(f"✅ Figure added: {figure.number} - {figure.caption}")
        logger.info(f"   Path: {figure.path}")
        logger.info(f"   Width: {figure.width}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Figure management failed: {e}")
        return False


def test_table_management(doc):
    """Test 5: Table management."""
    logger.info("\n" + "="*80)
    logger.info("TEST 5: TABLE MANAGEMENT")
    logger.info("="*80)
    
    try:
        # Add table from dict
        data = {
            'Propriedade': ['Área (cm²)', 'Ixx (cm⁴)', 'Iyy (cm⁴)'],
            'Valor': [100.5, 12500.0, 8900.0],
            'Unidade': ['cm²', 'cm⁴', 'cm⁴']
        }
        
        table = doc.add_table(
            data=data,
            caption="Propriedades geométricas da seção"
        )
        
        logger.info(f"✅ Table added: {table.number} - {table.caption}")
        logger.info(f"   Headers: {table.headers}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Table management failed: {e}")
        return False


def test_equation_management(doc):
    """Test 6: Equation management."""
    logger.info("\n" + "="*80)
    logger.info("TEST 6: EQUATION MANAGEMENT")
    logger.info("="*80)
    
    try:
        # Add LaTeX equation
        eq = doc.add_equation(
            r"N_{Rd} = \chi \cdot (A_s \cdot f_{yd} + A_c \cdot 0.85 \cdot f_{cd})",
            label="eq:resistance",
            description="Resistência à compressão do pilar misto"
        )
        
        logger.info(f"✅ Equation added: {eq.number} ({eq.label})")
        logger.info(f"   LaTeX: {eq.latex[:50]}...")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Equation management failed: {e}")
        return False


def test_toc_generation(doc):
    """Test 7: TOC generation."""
    logger.info("\n" + "="*80)
    logger.info("TEST 7: TABLE OF CONTENTS")
    logger.info("="*80)
    
    try:
        toc = doc.get_toc()
        
        logger.info(f"✅ TOC generated with {len(toc)} entries:")
        for entry in toc:
            indent = "  " * (entry['level'] - 1)
            logger.info(f"   {indent}{entry['number']} {entry['title']}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ TOC generation failed: {e}")
        return False


def test_json_export(doc):
    """Test 8: JSON export."""
    logger.info("\n" + "="*80)
    logger.info("TEST 8: JSON EXPORT")
    logger.info("="*80)
    
    try:
        output_path = Path("outputs") / "test_document.json"
        doc.export_json(output_path)
        
        logger.info(f"✅ Document exported to JSON: {output_path}")
        
        # Verify file
        import json
        with open(output_path) as f:
            data = json.load(f)
        
        logger.info(f"   Sections in JSON: {len(data['sections'])}")
        logger.info(f"   Figures in JSON: {len(data['figures'])}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ JSON export failed: {e}")
        return False


# ============================================================================
# MAIN VALIDATION RUNNER
# ============================================================================

def main():
    """Run all validation tests."""
    logger.info("\n" + "🚀 STARTING BASE_DOCUMENT VALIDATION")
    logger.info("="*80)
    
    results = []
    
    # Test 1: Metadata
    success, metadata = test_metadata_creation()
    results.append(('Metadata Creation', success))
    
    if not success:
        logger.error("❌ Cannot continue without metadata")
        return
    
    # Test 2: Initialization
    success, doc = test_document_initialization(metadata)
    results.append(('Document Initialization', success))
    
    if not success:
        logger.error("❌ Cannot continue without document")
        return
    
    # Test 3-8: Document operations
    results.append(('Section Numbering', test_section_numbering(doc)))
    results.append(('Figure Management', test_figure_management(doc)))
    results.append(('Table Management', test_table_management(doc)))
    results.append(('Equation Management', test_equation_management(doc)))
    results.append(('TOC Generation', test_toc_generation(doc)))
    results.append(('JSON Export', test_json_export(doc)))
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("VALIDATION SUMMARY")
    logger.info("="*80)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"{status} - {test_name}")
    
    logger.info("="*80)
    logger.info(f"Result: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        logger.info("🎉 ALL TESTS PASSED!")
        return 0
    else:
        logger.error(f"❌ {total - passed} tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
