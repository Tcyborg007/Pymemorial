# tests/unit/document/test_base_document.py
"""
Comprehensive unit tests for BaseDocument (PHASE 7).

Tests cover:
- Document initialization and metadata validation
- Section numbering (hierarchical)
- Figure/table/equation numbering
- Cross-references
- Validation logic
- Integration with PHASES 1-6

Author: PyMemorial Team
Date: 2025-10-19
Coverage Target: 100%
"""

import pytest
import tempfile
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch

# Import base_document components
from pymemorial.document.base_document import (
    BaseDocument,
    DocumentMetadata,
    NormCode,
    Section,
    Figure,
    Table,
    EquationDoc,
    Verification,
    NormReference,
    CrossReference,
    CrossReferenceType,
    ValidationError,
    ValidationResult,
    SectionNumbering,
    ElementNumbering,
    Revision,
    DocumentLanguage,
    TableStyle,
    DocumentError,
    DocumentValidationError,
    CrossReferenceError,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def valid_metadata():
    """Valid document metadata for testing."""
    return DocumentMetadata(
        title="Test Document",
        author="Test Author",
        company="Test Company",
        code=NormCode.NBR8800_2024,
        project_number="TEST-001",
        language=DocumentLanguage.PT_BR
    )


@pytest.fixture
def concrete_document(valid_metadata):
    """Concrete implementation of BaseDocument for testing."""
    class TestDocument(BaseDocument):
        """Concrete implementation for testing."""
        
        def render(self, output_path: Path, format: str = 'pdf', **kwargs) -> Path:
            # Mock render
            output_path = Path(output_path)
            output_path.touch()
            return output_path
        
        def validate(self) -> ValidationResult:
            # Mock validation
            return ValidationResult(valid=True, errors=[], warnings=[])
        
        def to_dict(self):
            # Mock serialization
            return {
                'metadata': {
                    'title': self.metadata.title,
                    'author': self.metadata.author
                },
                'sections': len(self.sections),
                'figures': len(self.figures)
            }
    
    return TestDocument(valid_metadata)


@pytest.fixture
def temp_figure():
    """Create temporary figure file for testing."""
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        f.write(b'fake image data')
        temp_path = Path(f.name)
    
    yield temp_path
    
    # Cleanup
    if temp_path.exists():
        temp_path.unlink()


# ============================================================================
# TEST SUITE 1: METADATA AND INITIALIZATION
# ============================================================================

class TestMetadata:
    """Test DocumentMetadata dataclass."""
    
    def test_valid_metadata_creation(self, valid_metadata):
        """Test creating valid metadata."""
        assert valid_metadata.title == "Test Document"
        assert valid_metadata.author == "Test Author"
        assert valid_metadata.company == "Test Company"
        assert valid_metadata.code == NormCode.NBR8800_2024
        assert valid_metadata.language == DocumentLanguage.PT_BR
    
    def test_metadata_with_default_revision(self):
        """Test metadata with default revision."""
        metadata = DocumentMetadata(
            title="Test",
            author="Author",
            company="Company",
            code=NormCode.NBR6118_2023
        )
        
        assert metadata.revision.number == "R00"
        assert metadata.revision.author == "Unknown"
        assert metadata.revision.approved is False
    
    def test_metadata_validation_empty_title(self):
        """Test that empty title raises error."""
        with pytest.raises(ValueError, match="title cannot be empty"):
            DocumentMetadata(
                title="",
                author="Author",
                company="Company",
                code=NormCode.NBR8800_2024
            )
    
    def test_metadata_validation_empty_author(self):
        """Test that empty author raises error."""
        with pytest.raises(ValueError, match="author cannot be empty"):
            DocumentMetadata(
                title="Title",
                author="",
                company="Company",
                code=NormCode.NBR8800_2024
            )
    
    def test_metadata_with_custom_revision(self):
        """Test metadata with custom revision."""
        revision = Revision(
            number="R02",
            date=datetime(2025, 10, 19),
            description="Updated calculations",
            author="Reviewer",
            approved=True,
            approved_by="Manager"
        )
        
        metadata = DocumentMetadata(
            title="Test",
            author="Author",
            company="Company",
            code=NormCode.NBR8800_2024,
            revision=revision
        )
        
        assert metadata.revision.number == "R02"
        assert metadata.revision.approved is True


class TestBaseDocumentInitialization:
    """Test BaseDocument initialization."""
    
    def test_cannot_instantiate_abstract_class(self, valid_metadata):
        """Test that BaseDocument cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            BaseDocument(valid_metadata)
    
    def test_concrete_document_initialization(self, concrete_document, valid_metadata):
        """Test concrete implementation initializes correctly."""
        assert concrete_document.metadata == valid_metadata
        assert len(concrete_document.sections) == 0
        assert len(concrete_document.figures) == 0
        assert len(concrete_document.tables) == 0
        assert len(concrete_document.equations) == 0
        assert len(concrete_document.verifications) == 0
    
    def test_invalid_metadata_type(self, concrete_document):
        """Test that invalid metadata type raises error."""
        with pytest.raises(TypeError):
            type(concrete_document)({"title": "Invalid"})


# ============================================================================
# TEST SUITE 2: SECTION NUMBERING
# ============================================================================

class TestSectionNumbering:
    """Test hierarchical section numbering."""
    
    def test_sequential_numbering_level_1(self):
        """Test sequential numbering at level 1."""
        numbering = SectionNumbering()
        
        assert numbering.get_next_number(1) == "1"
        assert numbering.get_next_number(1) == "2"
        assert numbering.get_next_number(1) == "3"
    
    def test_hierarchical_numbering_simple(self):
        """Test simple hierarchical numbering."""
        numbering = SectionNumbering()
        
        assert numbering.get_next_number(1) == "1"      # 1
        assert numbering.get_next_number(2) == "1.1"    # 1.1
        assert numbering.get_next_number(2) == "1.2"    # 1.2
        assert numbering.get_next_number(1) == "2"      # 2
        assert numbering.get_next_number(2) == "2.1"    # 2.1
    
    def test_hierarchical_numbering_deep(self):
        """Test deep hierarchical numbering (3 levels)."""
        numbering = SectionNumbering()
        
        assert numbering.get_next_number(1) == "1"
        assert numbering.get_next_number(2) == "1.1"
        assert numbering.get_next_number(3) == "1.1.1"
        assert numbering.get_next_number(3) == "1.1.2"
        assert numbering.get_next_number(2) == "1.2"
        assert numbering.get_next_number(3) == "1.2.1"
    
    def test_reset_deeper_levels(self):
        """Test that deeper levels reset when going to shallower level."""
        numbering = SectionNumbering()
        
        numbering.get_next_number(1)  # 1
        numbering.get_next_number(2)  # 1.1
        numbering.get_next_number(3)  # 1.1.1
        numbering.get_next_number(1)  # 2 (resets level 2 and 3)
        
        assert numbering.get_next_number(2) == "2.1"  # Should start at 1 again
    
    def test_invalid_level_too_low(self):
        """Test that level < 1 raises error."""
        numbering = SectionNumbering()
        
        with pytest.raises(ValueError, match="level must be 1-6"):
            numbering.get_next_number(0)
    
    def test_invalid_level_too_high(self):
        """Test that level > 6 raises error."""
        numbering = SectionNumbering()
        
        with pytest.raises(ValueError, match="level must be 1-6"):
            numbering.get_next_number(7)
    
    def test_reset_counters(self):
        """Test resetting all counters."""
        numbering = SectionNumbering()
        
        numbering.get_next_number(1)  # 1
        numbering.get_next_number(2)  # 1.1
        
        numbering.reset()
        
        assert numbering.get_next_number(1) == "1"  # Starts from 1 again


class TestDocumentSections:
    """Test document section management."""
    
    def test_add_section_level_1(self, concrete_document):
        """Test adding level 1 section."""
        section = concrete_document.add_section("Introduction", "Content", level=1)
        
        assert section.number == "1"
        assert section.title == "Introduction"
        assert section.content == "Content"
        assert section.level == 1
        assert len(concrete_document.sections) == 1
    
    def test_add_multiple_sections_hierarchical(self, concrete_document):
        """Test adding multiple hierarchical sections."""
        sec1 = concrete_document.add_section("Chapter 1", "", level=1)
        sec2 = concrete_document.add_section("Section 1.1", "", level=2)
        sec3 = concrete_document.add_section("Section 1.2", "", level=2)
        sec4 = concrete_document.add_section("Chapter 2", "", level=1)
        
        assert sec1.number == "1"
        assert sec2.number == "1.1"
        assert sec3.number == "1.2"
        assert sec4.number == "2"
    
    def test_add_section_with_parent_id(self, concrete_document):
        """Test adding section with parent reference."""
        parent = concrete_document.add_section("Parent", "", level=1)
        child = concrete_document.add_section("Child", "", level=2, parent_id=parent.id)
        
        assert child.parent_id == parent.id
    
    def test_add_section_invalid_parent_id(self, concrete_document):
        """Test that invalid parent_id raises error."""
        with pytest.raises(ValueError, match="parent_id .* not found"):
            concrete_document.add_section("Section", "", level=2, parent_id="invalid-id")
    
    def test_add_section_invalid_level(self, concrete_document):
        """Test that invalid level raises error."""
        with pytest.raises(ValueError, match="level must be 1-6"):
            concrete_document.add_section("Section", "", level=7)
    
    def test_section_ids_are_unique(self, concrete_document):
        """Test that section IDs are unique UUIDs."""
        sec1 = concrete_document.add_section("S1", "", level=1)
        sec2 = concrete_document.add_section("S2", "", level=1)
        
        assert sec1.id != sec2.id
        assert len(sec1.id) == 36  # UUID format


# ============================================================================
# TEST SUITE 3: FIGURE/TABLE/EQUATION NUMBERING
# ============================================================================

class TestElementNumbering:
    """Test simple sequential numbering."""
    
    def test_figure_numbering(self):
        """Test figure numbering."""
        numbering = ElementNumbering("Figure")
        
        assert numbering.get_next() == "Figure 1"
        assert numbering.get_next() == "Figure 2"
        assert numbering.get_next() == "Figure 3"
    
    def test_table_numbering_portuguese(self):
        """Test table numbering in Portuguese."""
        numbering = ElementNumbering("Tabela")
        
        assert numbering.get_next() == "Tabela 1"
        assert numbering.get_next() == "Tabela 2"
    
    def test_custom_start_number(self):
        """Test custom start number."""
        numbering = ElementNumbering("Equation", start=5)
        
        assert numbering.get_next() == "Equation 5"
        assert numbering.get_next() == "Equation 6"
    
    def test_reset_numbering(self):
        """Test resetting counter."""
        numbering = ElementNumbering("Figure")
        
        numbering.get_next()  # 1
        numbering.get_next()  # 2
        numbering.reset()
        
        assert numbering.get_next() == "Figure 1"


class TestFigureManagement:
    """Test document figure management."""
    
    def test_add_figure_from_path(self, concrete_document, temp_figure):
        """Test adding figure from file path."""
        figure = concrete_document.add_figure(
            temp_figure,
            caption="Test Figure"
        )
        
        assert figure.number == "Figura 1"  # Portuguese (default metadata)
        assert figure.caption == "Test Figure"
        assert figure.path == temp_figure
        assert len(concrete_document.figures) == 1
    
    def test_add_figure_nonexistent_file(self, concrete_document):
        """Test that nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            concrete_document.add_figure(
                Path("/nonexistent/file.png"),
                caption="Invalid"
            )
    
    def test_add_figure_with_custom_width(self, concrete_document, temp_figure):
        """Test adding figure with custom width."""
        figure = concrete_document.add_figure(
            temp_figure,
            caption="Wide Figure",
            width="800px"
        )
        
        assert figure.width == "800px"
    
    @patch('pymemorial.document.base_document.export_figure')  # PATCH CORRETO (no módulo onde é usado)
    def test_add_figure_from_matplotlib(self, mock_export, concrete_document, temp_figure):
        """Test adding matplotlib figure (PHASE 6 integration)."""
        # Mock export_figure to return temp path
        mock_export.return_value = temp_figure
        
        # CREATE PROPER MATPLOTLIB FIGURE (not Mock)
        import matplotlib
        matplotlib.use('Agg')  # Headless
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [4, 5, 6])
        ax.set_title("Test Plot")
        
        # Add figure to document
        figure = concrete_document.add_figure(
            fig,
            caption="Matplotlib Figure"
        )
        
        plt.close(fig)
        
        # Assertions
        assert figure.caption == "Matplotlib Figure"
        assert figure.path == temp_figure
        assert mock_export.called




class TestTableManagement:
    """Test document table management."""
    
    def test_add_table_from_dict(self, concrete_document):
        """Test adding table from dictionary."""
        data = {
            'Property': ['Area', 'Ixx', 'Iyy'],
            'Value': [100, 1000, 500],
            'Unit': ['cm²', 'cm⁴', 'cm⁴']
        }
        
        table = concrete_document.add_table(
            data=data,
            caption="Section Properties"
        )
        
        assert table.number == "Tabela 1"
        assert table.caption == "Section Properties"
        assert table.headers == ['Property', 'Value', 'Unit']
    
    def test_add_table_from_list(self, concrete_document):
        """Test adding table from list."""
        data = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ]
        headers = ['A', 'B', 'C']
        
        table = concrete_document.add_table(
            data=data,
            caption="Data Table",
            headers=headers
        )
        
        assert table.headers == headers
        assert table.data == data


class TestEquationManagement:
    """Test document equation management."""
    
    def test_add_equation_from_string(self, concrete_document):
        """Test adding equation from LaTeX string."""
        eq = concrete_document.add_equation(
            r"N_{Rd} = \chi \cdot A_g \cdot f_y",
            label="eq:resistance",
            description="Compression resistance"
        )
        
        assert eq.number == "Eq. 1"
        assert eq.label == "eq:resistance"
        assert "N_{Rd}" in eq.latex
    
    def test_add_multiple_equations(self, concrete_document):
        """Test adding multiple equations."""
        eq1 = concrete_document.add_equation(r"\sigma = \frac{P}{A}")
        eq2 = concrete_document.add_equation(r"\epsilon = \frac{\Delta L}{L}")
        
        assert eq1.number == "Eq. 1"
        assert eq2.number == "Eq. 2"


# ============================================================================
# TEST SUITE 4: CROSS-REFERENCES
# ============================================================================

class TestCrossReferences:
    """Test cross-reference management."""
    
    def test_add_valid_cross_reference(self, concrete_document, temp_figure):
        """Test adding valid cross-reference."""
        section = concrete_document.add_section("Introduction", "", level=1)
        figure = concrete_document.add_figure(temp_figure, "Test")
        
        concrete_document.add_cross_reference(
            from_id=section.id,
            to_id=figure.id,
            ref_type=CrossReferenceType.FIGURE
        )
        
        assert section.id in concrete_document.cross_refs
        assert len(concrete_document.cross_refs[section.id]) == 1
    
    def test_add_cross_reference_invalid_source(self, concrete_document):
        """Test that invalid source ID raises error."""
        with pytest.raises(CrossReferenceError, match="Source element not found"):
            concrete_document.add_cross_reference(
                from_id="invalid-id",
                to_id="also-invalid",
                ref_type=CrossReferenceType.SECTION
            )
    
    def test_add_cross_reference_invalid_target(self, concrete_document):
        """Test that invalid target ID raises error."""
        section = concrete_document.add_section("Section", "", level=1)
        
        with pytest.raises(CrossReferenceError, match="Target element not found"):
            concrete_document.add_cross_reference(
                from_id=section.id,
                to_id="invalid-target-id",
                ref_type=CrossReferenceType.FIGURE
            )


# ============================================================================
# TEST SUITE 5: UTILITY METHODS
# ============================================================================

class TestUtilityMethods:
    """Test utility methods."""
    
    def test_get_toc(self, concrete_document):
        """Test TOC generation."""
        concrete_document.add_section("Chapter 1", "", level=1)
        concrete_document.add_section("Section 1.1", "", level=2)
        concrete_document.add_section("Chapter 2", "", level=1)
        
        toc = concrete_document.get_toc()
        
        assert len(toc) == 3
        assert toc[0]['number'] == "1"
        assert toc[1]['number'] == "1.1"
        assert toc[2]['number'] == "2"
    
    def test_get_element_by_id(self, concrete_document):
        """Test getting element by ID."""
        section = concrete_document.add_section("Test", "", level=1)
        
        retrieved = concrete_document.get_element_by_id(section.id)
        
        assert retrieved == section
    
    def test_get_element_by_invalid_id(self, concrete_document):
        """Test getting element with invalid ID returns None."""
        retrieved = concrete_document.get_element_by_id("invalid-id")
        
        assert retrieved is None
    
    def test_export_json(self, concrete_document, tmp_path):
        """Test JSON export."""
        concrete_document.add_section("Test", "", level=1)
        
        json_path = tmp_path / "doc.json"
        concrete_document.export_json(json_path)
        
        assert json_path.exists()
        
        import json
        with open(json_path) as f:
            data = json.load(f)
        
        assert 'metadata' in data
        assert data['sections'] == 1


# ============================================================================
# TEST SUMMARY
# ============================================================================

def test_summary():
    """Print test summary."""
    print("\n" + "="*80)
    print("BASE_DOCUMENT.PY - TEST SUMMARY")
    print("="*80)
    print("✅ 40+ unit tests covering:")
    print("   - Metadata validation")
    print("   - Section numbering (hierarchical)")
    print("   - Figure/table/equation numbering")
    print("   - Cross-references")
    print("   - Utility methods")
    print("   - Error handling")
    print("="*80)
