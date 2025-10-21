# tests/debug/test_auto_lists_debug.py
"""
Debug script for automatic lists generation (PHASE 7.2).

Tests:
- get_list_of_figures() in base_document
- get_list_of_tables() in base_document
- get_list_of_equations() in base_document
- _generate_list_of_figures() in memorial
- _generate_list_of_tables() in memorial
- _generate_list_of_equations() in memorial
- auto_lists=True parameter

Author: PyMemorial Team
Date: 2025-10-20
Phase: 7.2 (Auto Lists)
"""

import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from pymemorial.document import (
    Memorial,
    DocumentMetadata,
    NormCode,
    Revision,
    DocumentLanguage,
)

# ============================================================================
# SETUP
# ============================================================================

def create_test_memorial(auto_lists: bool = False):
    """Create test memorial for debugging."""
    metadata = DocumentMetadata(
        title="Debug Test - Auto Lists",
        author="PyMemorial Debug",
        company="Test Company",
        code=NormCode.NBR8800_2024,
        project_number="DEBUG-002",
        revision=Revision(
            number="R00",
            date=datetime(2025, 10, 20),
            description="Debug test auto lists",
            author="Test",
            approved=False
        ),
        language=DocumentLanguage.PT_BR,
    )
    
    memorial = Memorial(
        metadata,
        template='nbr8800',
        auto_lists=auto_lists  # ✅ TEST THIS
    )
    print(f"✅ Memorial created (auto_lists={auto_lists})")
    return memorial


# ============================================================================
# TEST 1: Base methods - get_list_of_*()
# ============================================================================

def test_base_list_methods(memorial):
    """Test base_document list methods."""
    print("\n" + "="*70)
    print("TEST 1: base_document.get_list_of_*() methods")
    print("="*70)
    
    # Add test content
    output_dir = Path("outputs/debug/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Add figures
    from PIL import Image
    img = Image.new('RGB', (100, 100), color='red')
    
    for i in range(3):
        img_path = output_dir / f"test_fig_{i+1}.png"
        img.save(img_path)
        memorial.add_figure(
            img_path,
            caption=f"Test Figure {i+1}",
            source="Test source" if i == 0 else None
        )
    
    print(f"✅ Added {len(memorial.figures)} figures")
    
    # Add tables
    for i in range(2):
        df = pd.DataFrame({
            'Col1': [1, 2, 3],
            'Col2': [4, 5, 6],
        })
        memorial.add_table(
            df,
            caption=f"Test Table {i+1}",
            source="NBR 8800" if i == 0 else None
        )
    
    print(f"✅ Added {len(memorial.tables)} tables")
    
    # Add equations
    for i in range(2):
        memorial.add_equation(
            f"x_{i+1} = y * z",
            description=f"Test Equation {i+1}",
            reference="NBR 8800, item 5.1" if i == 0 else None
        )
    
    print(f"✅ Added {len(memorial.equations)} equations")
    
    # Test get_list_of_figures()
    try:
        figures_list = memorial.get_list_of_figures()
        print(f"\n✅ get_list_of_figures(): {len(figures_list)} items")
        for fig in figures_list:
            print(f"   - {fig['number']}: {fig['caption']}")
            if fig.get('source'):
                print(f"     Source: {fig['source']}")
    except Exception as e:
        print(f"❌ get_list_of_figures() FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    # Test get_list_of_tables()
    try:
        tables_list = memorial.get_list_of_tables()
        print(f"\n✅ get_list_of_tables(): {len(tables_list)} items")
        for tbl in tables_list:
            print(f"   - {tbl['number']}: {tbl['caption']}")
            print(f"     Dimensions: {tbl['rows']}×{tbl['cols']}")
            if tbl.get('source'):
                print(f"     Source: {tbl['source']}")
    except Exception as e:
        print(f"❌ get_list_of_tables() FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    # Test get_list_of_equations()
    try:
        equations_list = memorial.get_list_of_equations()
        print(f"\n✅ get_list_of_equations(): {len(equations_list)} items")
        for eq in equations_list:
            print(f"   - {eq['number']}: {eq['description']}")
            if eq.get('reference'):
                print(f"     Reference: {eq['reference']}")
    except Exception as e:
        print(f"❌ get_list_of_equations() FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    # Test get_summary_statistics()
    try:
        stats = memorial.get_summary_statistics()
        print(f"\n✅ get_summary_statistics():")
        print(f"   - Sections: {stats['sections']}")
        print(f"   - Figures: {stats['figures']}")
        print(f"   - Tables: {stats['tables']}")
        print(f"   - Equations: {stats['equations']}")
    except Exception as e:
        print(f"❌ get_summary_statistics() FAILED: {e}")


# ============================================================================
# TEST 2: Manual list generation
# ============================================================================

def test_manual_list_generation(memorial):
    """Test manual list generation methods."""
    print("\n" + "="*70)
    print("TEST 2: Memorial._generate_list_of_*() methods (manual)")
    print("="*70)
    
    # Test _generate_list_of_figures()
    try:
        memorial._generate_list_of_figures()
        print(f"✅ _generate_list_of_figures() executed")
        
        # Check if section was added
        for section in memorial.sections:
            if section.metadata.get('type') == 'list_of_figures':
                print(f"   ✅ List of Figures section created")
                print(f"   Title: {section.title}")
                print(f"   Content length: {len(section.content)} chars")
                break
        else:
            print(f"   ❌ List of Figures section NOT found")
    except Exception as e:
        print(f"❌ _generate_list_of_figures() FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    # Test _generate_list_of_tables()
    try:
        memorial._generate_list_of_tables()
        print(f"\n✅ _generate_list_of_tables() executed")
        
        # Check if section was added
        for section in memorial.sections:
            if section.metadata.get('type') == 'list_of_tables':
                print(f"   ✅ List of Tables section created")
                print(f"   Title: {section.title}")
                print(f"   Content length: {len(section.content)} chars")
                break
        else:
            print(f"   ❌ List of Tables section NOT found")
    except Exception as e:
        print(f"❌ _generate_list_of_tables() FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    # Test _generate_list_of_equations()
    try:
        memorial._generate_list_of_equations()
        print(f"\n✅ _generate_list_of_equations() executed")
        
        # Check if section was added
        for section in memorial.sections:
            if section.metadata.get('type') == 'list_of_equations':
                print(f"   ✅ List of Equations section created")
                print(f"   Title: {section.title}")
                print(f"   Content length: {len(section.content)} chars")
                break
        else:
            print(f"   ❌ List of Equations section NOT found")
    except Exception as e:
        print(f"❌ _generate_list_of_equations() FAILED: {e}")
        import traceback
        traceback.print_exc()


# ============================================================================
# TEST 3: Automatic lists (auto_lists=True)
# ============================================================================

def test_automatic_lists():
    """Test automatic lists generation on init."""
    print("\n" + "="*70)
    print("TEST 3: Automatic lists (auto_lists=True)")
    print("="*70)
    
    try:
        # Create memorial with auto_lists=True
        memorial = create_test_memorial(auto_lists=True)
        
        # Add content
        output_dir = Path("outputs/debug/figures")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        from PIL import Image
        img = Image.new('RGB', (100, 100), color='blue')
        
        # Add 2 figures
        for i in range(2):
            img_path = output_dir / f"auto_fig_{i+1}.png"
            img.save(img_path)
            memorial.add_figure(img_path, caption=f"Auto Figure {i+1}")
        
        # Add 2 tables
        for i in range(2):
            df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
            memorial.add_table(df, caption=f"Auto Table {i+1}")
        
        # Add 1 equation
        memorial.add_equation("x = y + z", description="Auto Equation 1")
        
        print(f"\n✅ Added content:")
        print(f"   - Figures: {len(memorial.figures)}")
        print(f"   - Tables: {len(memorial.tables)}")
        print(f"   - Equations: {len(memorial.equations)}")
        
        # Check sections
        print(f"\n✅ Checking sections:")
        print(f"   Total sections: {len(memorial.sections)}")
        
        list_sections = []
        for section in memorial.sections:
            section_type = section.metadata.get('type', '')
            if 'list_of' in section_type:
                list_sections.append(section)
                print(f"   ✅ Found: {section.title} (type: {section_type})")
        
        if not list_sections:
            print(f"   ⚠️  No list sections found (expected 3)")
            print(f"   This is NORMAL if lists are generated AFTER content is added")
            print(f"   Solution: Call _generate_list_of_*() manually after adding content")
        
    except Exception as e:
        print(f"❌ Automatic lists test FAILED: {e}")
        import traceback
        traceback.print_exc()


# ============================================================================
# TEST 4: Section ordering
# ============================================================================

def test_section_ordering(memorial):
    """Test section ordering (pre-textual before textual)."""
    print("\n" + "="*70)
    print("TEST 4: Section ordering")
    print("="*70)
    
    print("\nSection order:")
    for i, section in enumerate(memorial.sections):
        section_type = section.metadata.get('type', 'unknown')
        pre_textual = section.metadata.get('pre_textual', False)
        marker = "📄" if pre_textual else "📝"
        print(f"   {i+1}. {marker} {section.title} (type: {section_type})")
    
    # Expected order:
    # 1. Title page (pre-textual)
    # 2. TOC (pre-textual)
    # 3. List of Figures (pre-textual)
    # 4. List of Tables (pre-textual)
    # 5. List of Equations (pre-textual)
    # 6. ... textual content ...


# ============================================================================
# MAIN DEBUG RUNNER
# ============================================================================

def main():
    """Run all debug tests."""
    print("="*70)
    print("PyMemorial PHASE 7.2 - Auto Lists Debug")
    print("="*70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python: {sys.version.split()[0]}")
    print()
    
    try:
        # Test 1: Create memorial and test base methods
        memorial = create_test_memorial(auto_lists=False)
        test_base_list_methods(memorial)
        
        # Test 2: Manual list generation
        test_manual_list_generation(memorial)
        
        # Test 3: Automatic lists
        test_automatic_lists()
        
        # Test 4: Section ordering
        test_section_ordering(memorial)
        
        # Summary
        print("\n" + "="*70)
        print("DEBUG SUMMARY")
        print("="*70)
        print(f"✅ Figures:   {len(memorial.figures)}")
        print(f"✅ Tables:    {len(memorial.tables)}")
        print(f"✅ Equations: {len(memorial.equations)}")
        print(f"✅ Sections:  {len(memorial.sections)}")
        
        # Count list sections
        list_sections = sum(
            1 for s in memorial.sections
            if 'list_of' in s.metadata.get('type', '')
        )
        print(f"✅ List sections: {list_sections}")
        print()
        print("🎉 Debug completed successfully!")
        
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
