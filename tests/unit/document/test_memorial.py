# tests/unit/document/test_memorial.py
"""
Unit tests for Memorial document class.

This test suite validates:
- Memorial creation and initialization
- Template selection
- Natural language processing
- Global context management
- Validation (metadata, structure, cross-refs)
- Rendering (PDF, HTML, DOCX, LaTeX)
- Serialization (to_dict, JSON, YAML)
- Error handling

Author: PyMemorial Team
Date: 2025-10-19
Phase: 7 (Document Generation)
"""

import json
import pytest
import tempfile
from datetime import datetime
from pathlib import Path

from pymemorial.document import (
    Memorial,
    DocumentMetadata,
    NormCode,
    Revision,
    DocumentLanguage,
    ValidationError,
    DocumentValidationError,
    RenderError,
    CrossReferenceError,  # ✅ ADICIONAR ESTA LINHA
    CrossReferenceType,
)



# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_metadata():
    """Sample metadata for testing."""
    return DocumentMetadata(
        title="Memorial de Teste - Pilar PM-1",
        author="Eng. Test User, CREA: 99999/XX",
        company="Test Company LTDA",
        code=NormCode.NBR8800_2024,
        project_number="TEST-001",
        revision=Revision(
            number="R00",
            date=datetime(2025, 10, 19),
            description="Initial test revision",
            author="Test User",
            approved=False
        ),
        language=DocumentLanguage.PT_BR,
        keywords=['teste', 'pilar', 'NBR 8800'],
        abstract="Memorial de teste para validação do PyMemorial."
    )


@pytest.fixture
def memorial(sample_metadata):
    """Create Memorial instance for testing."""
    return Memorial(sample_metadata)


@pytest.fixture
def temp_output_dir(tmp_path):
    """Temporary directory for output files."""
    output_dir = tmp_path / "outputs"
    output_dir.mkdir(exist_ok=True)
    return output_dir


# ============================================================================
# TEST: Initialization
# ============================================================================

class TestMemorialInitialization:
    """Test Memorial initialization."""
    
    def test_init_with_metadata(self, sample_metadata):
        """Test Memorial initialization with metadata."""
        memorial = Memorial(sample_metadata)
        
        assert memorial.metadata == sample_metadata
        assert memorial.template_name == 'nbr8800'  # Auto-selected
        assert memorial.strict_validation is False
        assert memorial.auto_toc is True
        assert memorial.auto_title_page is True
        assert len(memorial.sections) == 0
    
    def test_init_with_custom_template(self, sample_metadata):
        """Test Memorial with custom template."""
        memorial = Memorial(sample_metadata, template='modern')
        
        assert memorial.template_name == 'modern'
    
    def test_init_with_strict_validation(self, sample_metadata):
        """Test Memorial with strict validation enabled."""
        memorial = Memorial(sample_metadata, strict_validation=True)
        
        assert memorial.strict_validation is True
    
    def test_template_auto_selection(self):
        """Test automatic template selection based on norm."""
        templates = {
            NormCode.NBR6118_2023: 'nbr6118',
            NormCode.NBR8800_2024: 'nbr8800',
            NormCode.AISC360_22: 'aisc360',
            NormCode.EN1993_2005: 'eurocode',
            NormCode.ACI318_19: 'aci318',
        }
        
        for norm, expected_template in templates.items():
            metadata = DocumentMetadata(
                title="Test",
                author="Test",
                company="Test",
                code=norm
            )
            memorial = Memorial(metadata)
            assert memorial.template_name == expected_template


# ============================================================================
# TEST: Content Addition
# ============================================================================

class TestContentAddition:
    """Test adding content to memorial."""
    
    def test_add_section(self, memorial):
        """Test adding a section."""
        section = memorial.add_section("Introduction", "Test content", level=1)
        
        assert len(memorial.sections) == 1
        assert section.title == "Introduction"
        assert section.content == "Test content"
        assert section.level == 1
        assert section.number == "1"
    
    def test_add_multiple_sections(self, memorial):
        """Test adding multiple sections with hierarchy."""
        sec1 = memorial.add_section("Chapter 1", level=1)
        sec2 = memorial.add_section("Section 1.1", level=2)
        sec3 = memorial.add_section("Section 1.2", level=2)
        sec4 = memorial.add_section("Chapter 2", level=1)
        
        assert len(memorial.sections) == 4
        assert sec1.number == "1"
        assert sec2.number == "1.1"
        assert sec3.number == "1.2"
        assert sec4.number == "2"
    
    def test_add_paragraph_simple(self, memorial):
        """Test adding simple paragraph."""
        paragraph = memorial.add_paragraph("This is a test paragraph.")
        
        assert len(memorial.sections) == 1
        assert paragraph.content == "This is a test paragraph."
        assert paragraph.metadata.get('type') == 'paragraph'
    
    def test_add_paragraph_with_context(self, memorial):
        """Test adding paragraph with variable substitution."""
        memorial.set_context(N_Sd=2500, N_Rd=3650)
        
        paragraph = memorial.add_paragraph(
            "Force: N_Sd = {N_Sd:.2f} kN, Resistance: N_Rd = {N_Rd:.2f} kN"
        )
        
        assert "2500.00" in paragraph.content
        assert "3650.00" in paragraph.content
    
    def test_add_paragraph_with_local_variables(self, memorial):
        """Test paragraph with local variables overriding global."""
        memorial.set_context(value=100)
        
        paragraph = memorial.add_paragraph(
            "Value: {value:.0f}",
            variables={'value': 200}
        )
        
        # Local should override global
        assert "200" in paragraph.content
    
    def test_add_verification(self, memorial):
        """Test adding verification."""
        verify = memorial.add_verification(
            expression="N_Sd <= N_Rd",
            passed=True,
            description="Compression resistance",
            norm=NormCode.NBR8800_2024,
            item="5.3.2.1",
            calculated_values={'N_Sd': 2500, 'N_Rd': 3650}
        )
        
        assert len(memorial.verifications) == 1
        assert verify.passed is True
        assert verify.expression == "N_Sd <= N_Rd"


# ============================================================================
# TEST: Context Management
# ============================================================================

class TestContextManagement:
    """Test global context management."""
    
    def test_set_context(self, memorial):
        """Test setting global context."""
        memorial.set_context(N_Sd=2500, chi=0.877)
        
        context = memorial.get_context()
        assert context['N_Sd'] == 2500
        assert context['chi'] == 0.877
    
    def test_update_context(self, memorial):
        """Test updating global context."""
        memorial.set_context(N_Sd=2500)
        memorial.update_context(chi=0.877, f_y=345)
        
        context = memorial.get_context()
        assert context['N_Sd'] == 2500
        assert context['chi'] == 0.877
        assert context['f_y'] == 345
    
    def test_clear_context(self, memorial):
        """Test clearing global context."""
        memorial.set_context(N_Sd=2500, chi=0.877)
        memorial.clear_context()
        
        context = memorial.get_context()
        assert len(context) == 0


# ============================================================================
# TEST: Validation
# ============================================================================

class TestValidation:
    """Test document validation."""
    
    def test_validate_empty_document(self, memorial):
        """Test validating empty document."""
        result = memorial.validate()
        
        # Should have error for empty document
        assert not result.valid
        assert any(err.error_type == 'empty_document' for err in result.errors)
    
    def test_validate_complete_document(self, memorial):
        """Test validating complete document."""
        memorial.add_section("Introduction", "Content...", level=1)
        memorial.add_section("Calculations", "Calcs...", level=1)
        
        result = memorial.validate()
        
        # Should be valid
        assert result.valid
        assert len(result.errors) == 0
    
    def test_validate_cross_references(self, memorial):
        """Test cross-reference validation."""
        sec1 = memorial.add_section("Section 1", level=1)
        sec2 = memorial.add_section("Section 2", level=1)
        
        # Add valid cross-reference
        memorial.add_cross_reference(
            from_id=sec1.id,
            to_id=sec2.id,
            ref_type=CrossReferenceType.SECTION
        )
        
        # Validate should pass
        result = memorial.validate()
        assert result.valid
        
        # Test that invalid cross-reference raises exception
        with pytest.raises(CrossReferenceError):
            memorial.add_cross_reference(
                from_id=sec1.id,
                to_id="invalid-id",
                ref_type=CrossReferenceType.SECTION
            )

    
    def test_strict_validation_raises(self, sample_metadata):
        """Test strict validation raises exception."""
        memorial = Memorial(sample_metadata, strict_validation=True)
        
        # Empty document should fail validation
        with pytest.raises(DocumentValidationError):
            memorial.render("output.pdf")


# ============================================================================
# TEST: Rendering
# ============================================================================

class TestRendering:
    """Test document rendering."""
    
    def test_render_html(self, memorial, temp_output_dir):
        """Test rendering to HTML."""
        memorial.add_section("Test Section", "Test content", level=1)
        
        output_path = temp_output_dir / "test.html"
        result_path = memorial.render(output_path, format='html')
        
        assert result_path.exists()
        assert result_path.suffix == '.html'
        
        # Check HTML content
        content = result_path.read_text()
        assert '<html' in content
        assert 'Test Section' in content
    
    @pytest.mark.skipif(
        not pytest.importorskip("weasyprint", reason="WeasyPrint not installed"),
        reason="WeasyPrint required for PDF rendering"
    )
    def test_render_pdf(self, memorial, temp_output_dir):
        """Test rendering to PDF."""
        memorial.add_section("Test Section", "Test content", level=1)
        
        output_path = temp_output_dir / "test.pdf"
        result_path = memorial.render(output_path, format='pdf')
        
        assert result_path.exists()
        assert result_path.suffix == '.pdf'
    
    def test_render_latex(self, memorial, temp_output_dir):
        """Test rendering to LaTeX."""
        memorial.add_section("Test Section", "Test content", level=1)
        
        output_path = temp_output_dir / "test.tex"
        result_path = memorial.render(output_path, format='latex')
        
        assert result_path.exists()
        assert result_path.suffix == '.tex'
        
        # Check LaTeX content
        content = result_path.read_text()
        assert '\\documentclass' in content
    
    def test_render_unknown_format(self, memorial, temp_output_dir):
        """Test rendering with unknown format raises error."""
        memorial.add_section("Test", "Content", level=1)
        
        output_path = temp_output_dir / "test.xyz"
        
        with pytest.raises(RenderError, match="Unknown format"):
            memorial.render(output_path, format='xyz')

    
    def test_render_freezes_document(self, memorial, temp_output_dir):
        """Test that render() freezes document."""
        memorial.add_section("Test", "Content", level=1)
        
        output_path = temp_output_dir / "test.html"
        memorial.render(output_path, format='html')
        
        # Should not allow adding content after render
        with pytest.raises(RuntimeError, match="frozen"):
            memorial.add_section("Another", "Content", level=1)


# ============================================================================
# TEST: Serialization
# ============================================================================

class TestSerialization:
    """Test document serialization."""
    
    def test_to_dict(self, memorial):
        """Test converting memorial to dictionary."""
        memorial.add_section("Test Section", "Content", level=1)
        
        data = memorial.to_dict()
        
        assert data['type'] == 'Memorial'
        assert data['template'] == 'nbr8800'
        assert data['metadata']['title'] == memorial.metadata.title
        assert len(data['sections']) == 1
        assert data['statistics']['sections'] == 1
    
    def test_export_json(self, memorial, temp_output_dir):
        """Test exporting to JSON."""
        memorial.add_section("Test", "Content", level=1)
        
        json_path = temp_output_dir / "memorial.json"
        memorial.export_json(json_path)
        
        assert json_path.exists()
        
        # Verify JSON content
        with open(json_path) as f:
            data = json.load(f)
        
        assert data['type'] == 'Memorial'
        assert len(data['sections']) == 1
    
    def test_export_yaml(self, memorial, temp_output_dir):
        """Test exporting to YAML."""
        pytest.importorskip("yaml", reason="PyYAML not installed")
        
        memorial.add_section("Test", "Content", level=1)
        
        yaml_path = temp_output_dir / "memorial.yaml"
        memorial.export_yaml(yaml_path)
        
        assert yaml_path.exists()


# ============================================================================
# TEST: String Representations
# ============================================================================

class TestStringRepresentations:
    """Test string representations."""
    
    def test_repr(self, memorial):
        """Test __repr__."""
        repr_str = repr(memorial)
        
        assert 'Memorial' in repr_str
        assert 'nbr8800' in repr_str
    
    def test_str(self, memorial):
        """Test __str__."""
        memorial.add_verification(
            expression="test",
            passed=True,
            description="Test",
            norm=NormCode.NBR8800_2024,
            item="1"
        )
        
        str_repr = str(memorial)
        
        assert memorial.metadata.title in str_repr
        assert '1/1 passed' in str_repr


# ============================================================================
# TEST: Integration
# ============================================================================

class TestIntegration:
    """Integration tests with other phases."""
    
    def test_complete_workflow(self, sample_metadata, temp_output_dir):
        """Test complete memorial workflow."""
        # 1. Create memorial
        memorial = Memorial(sample_metadata)
        
        # 2. Set context
        memorial.set_context(N_Sd=2500, N_Rd=3650, chi=0.877)
        
        # 3. Add content
        memorial.add_section("Introdução", level=1)
        memorial.add_paragraph("""
        Este memorial apresenta o dimensionamento do pilar PM-1.
        Força: N_Sd = {N_Sd:.2f} kN
        Resistência: N_Rd = {N_Rd:.2f} kN
        Fator: chi = {chi:.3f}
        """)
        
        memorial.add_section("Verificação", level=1)
        memorial.add_verification(
            expression="N_Sd <= N_Rd",
            passed=True,
            description="Resistência à compressão",
            norm=NormCode.NBR8800_2024,
            item="5.3.2.1",
            calculated_values={'N_Sd': 2500, 'N_Rd': 3650}
        )
        
        # 4. Validate
        result = memorial.validate()
        assert result.valid
        
        # 5. Render
        html_path = temp_output_dir / "complete.html"
        memorial.render(html_path, format='html')
        
        assert html_path.exists()
        
        # 6. Export
        json_path = temp_output_dir / "complete.json"
        memorial.export_json(json_path)
        
        assert json_path.exists()


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
