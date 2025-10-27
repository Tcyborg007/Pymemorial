# tests/debug/test_generators.py
"""Test generators module with all abstract methods implemented."""

import sys
from pathlib import Path

# ============================================================================
# TEST 1: Imports
# ============================================================================

try:
    from pymemorial.document.generators import (
        get_generator,
        WeasyPrintGenerator,
        HTMLGenerator,
        WEASYPRINT_AVAILABLE,
        HTML_AVAILABLE,
    )
    
    print("✅ Imports successful!")
    print(f"   WeasyPrint available: {WEASYPRINT_AVAILABLE}")
    print(f"   HTML available: {HTML_AVAILABLE}")
    
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# ============================================================================
# TEST 2: Create generators
# ============================================================================

try:
    if HTML_AVAILABLE:
        html_gen = get_generator('html')
        print(f"✅ HTML generator created: {html_gen}")
    
    if WEASYPRINT_AVAILABLE:
        pdf_gen = get_generator('weasyprint')
        print(f"✅ PDF generator created: {pdf_gen}")
    
except Exception as e:
    print(f"❌ Generator creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# TEST 3: Test generate_to_bytes() method
# ============================================================================

try:
    # Create a mock document
    class MockDocument:
        def __init__(self):
            self.title = "Test Memorial"
        
        def to_html(self, style='nbr'):
            return f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>{self.title}</title>
                <style>body {{ font-family: Arial; }}</style>
            </head>
            <body>
                <h1>{self.title}</h1>
                <p>This is a test document.</p>
            </body>
            </html>
            """
    
    doc = MockDocument()
    
    if HTML_AVAILABLE:
        html_gen = HTMLGenerator()
        
        # Test generate_to_bytes
        html_bytes = html_gen.generate_to_bytes(doc, style='nbr')
        print(f"✅ generate_to_bytes() works: {len(html_bytes)} bytes")
        
        # Test generate to file
        output_path = Path('output/test_memorial.html')
        output_path.parent.mkdir(exist_ok=True)
        html_gen.generate(doc, str(output_path), style='nbr')
        print(f"✅ HTML file generated: {output_path}")
        
        # Verify file exists
        if output_path.exists():
            print(f"✅ File verified: {output_path.stat().st_size} bytes")
        else:
            print(f"❌ File not found: {output_path}")
    
except Exception as e:
    print(f"❌ Generation test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# SUCCESS
# ============================================================================

print("\n🎉 All tests passed!")
print("\nGenerated files:")
if output_path.exists():
    print(f"   📄 {output_path.absolute()}")
