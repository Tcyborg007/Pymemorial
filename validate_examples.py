# tests/validate_examples.py
"""
Validation script for Phase 6 examples.

Runs all examples in a safe environment and reports results.
"""

import sys
import subprocess
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).parent.parent
EXAMPLES_DIR = PROJECT_ROOT / "examples" / "visualization"

# Ensure examples directory exists
EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("PyMemorial Phase 6 - Examples Validation")
print("=" * 80)

# ============================================================================
# DEFINE EXAMPLES TO VALIDATE
# ============================================================================

EXAMPLES = [
    {
        "file": "01_basic_pm_diagram.py",
        "name": "Example 1: Basic P-M Diagram",
        "timeout": 30,
        "required": True,
    },
    {
        "file": "02_moment_curvature_nbr8800.py",
        "name": "Example 2: Moment-Curvature (NBR 8800)",
        "timeout": 60,
        "required": True,
    },
    {
        "file": "03_advanced_3d_structure.py",
        "name": "Example 3: Advanced 3D Structure",
        "timeout": 120,
        "required": False,  # May fail if PyVista not installed
    },
    {
        "file": "04_complete_calculation_report.py",
        "name": "Example 4: Complete Calculation Memorial",
        "timeout": 180,
        "required": True,
    },
]


# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================


def check_example_exists(example: Dict) -> bool:
    """Check if example file exists."""
    filepath = EXAMPLES_DIR / example["file"]
    return filepath.exists()


def run_example(example: Dict) -> Dict[str, any]:
    """
    Run example script and capture output.

    Returns:
        Dict with keys: success, output, error, duration
    """
    import time

    filepath = EXAMPLES_DIR / example["file"]

    print(f"\n{'=' * 80}")
    print(f"Running: {example['name']}")
    print(f"File: {example['file']}")
    print(f"{'=' * 80}")

    start_time = time.time()

    try:
        # Run example as subprocess
        result = subprocess.run(
            [sys.executable, str(filepath)],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=example["timeout"],
        )

        duration = time.time() - start_time

        success = result.returncode == 0

        return {
            "success": success,
            "output": result.stdout,
            "error": result.stderr,
            "returncode": result.returncode,
            "duration": duration,
        }

    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        return {
            "success": False,
            "output": "",
            "error": f"TIMEOUT: Example exceeded {example['timeout']} seconds",
            "returncode": -1,
            "duration": duration,
        }

    except Exception as e:
        duration = time.time() - start_time
        return {
            "success": False,
            "output": "",
            "error": f"EXCEPTION: {str(e)}",
            "returncode": -2,
            "duration": duration,
        }


def print_result(example: Dict, result: Dict) -> None:
    """Print example execution result."""
    if result["success"]:
        symbol = "✓"
        status = "SUCCESS"
    else:
        symbol = "✗"
        status = "FAILED"

    print(f"\n{symbol} Status: {status} (Duration: {result['duration']:.2f}s)")

    if result["output"]:
        print("\n--- OUTPUT ---")
        print(result["output"][:500])  # First 500 chars
        if len(result["output"]) > 500:
            print(f"... ({len(result['output']) - 500} more characters)")

    if result["error"]:
        print("\n--- ERROR ---")
        print(result["error"][:500])
        if len(result["error"]) > 500:
            print(f"... ({len(result['error']) - 500} more characters)")


# ============================================================================
# MAIN VALIDATION LOOP
# ============================================================================


def main():
    """Run all examples and report results."""

    results = []
    total_examples = len(EXAMPLES)
    passed = 0
    failed = 0
    skipped = 0

    for i, example in enumerate(EXAMPLES, 1):
        print(f"\n[{i}/{total_examples}] Checking: {example['file']}")

        # Check if file exists
        if not check_example_exists(example):
            print(f"  ⊘ SKIPPED: File not found")
            skipped += 1
            results.append(
                {
                    "example": example,
                    "result": {"success": False, "error": "File not found"},
                }
            )
            continue

        # Run example
        result = run_example(example)
        print_result(example, result)

        results.append({"example": example, "result": result})

        if result["success"]:
            passed += 1
        else:
            failed += 1
            if example["required"]:
                print(f"  ⚠ WARNING: Required example failed!")

    # ============================================================================
    # SUMMARY REPORT
    # ============================================================================

    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)

    print(f"\n📊 Results:")
    print(f"  ✓ Passed:  {passed}/{total_examples}")
    print(f"  ✗ Failed:  {failed}/{total_examples}")
    print(f"  ⊘ Skipped: {skipped}/{total_examples}")

    # Check required examples
    required_failed = [
        r
        for r in results
        if r["example"]["required"] and not r["result"]["success"]
    ]

    if required_failed:
        print(f"\n⚠ {len(required_failed)} required example(s) failed:")
        for r in required_failed:
            print(f"  • {r['example']['name']}")
        print("\n✗ VALIDATION FAILED")
        return 1
    else:
        print("\n✓ ALL REQUIRED EXAMPLES PASSED")

        if failed > 0:
            print(f"  (Optional examples may have failed due to missing dependencies)")

        return 0


if __name__ == "__main__":
    sys.exit(main())
