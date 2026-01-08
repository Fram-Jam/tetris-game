"""
Validation script that Claude Code can run to check for errors.
Run with: python tests/validate_app.py
"""

import sys
import os
import traceback

# Add parent directory to path so we can import src modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def validate():
    errors = []

    # Test 1: All imports work
    print("Testing imports...")
    try:
        from src.data.normalizer import DailySummary, DataSource
        from src.data.synthetic.patient_generator import generate_demo_data
        from src.data.synthetic.lab_generator import generate_lab_history
        from src.insights.ai_coach import get_ai_insights
        print("  ✓ All imports successful")
    except Exception as e:
        errors.append(f"Import error: {e}\n{traceback.format_exc()}")
        print(f"  ✗ Import failed: {e}")

    # Test 2: Synthetic data generates without error
    print("Testing synthetic data generation...")
    try:
        patient, health_data = generate_demo_data(days=30)
        assert len(health_data) == 31, f"Expected 31 days, got {len(health_data)}"
        assert health_data[0]['sleep_duration'] > 0, "Sleep duration should be positive"
        assert health_data[0]['hrv'] > 0, "HRV should be positive"
        print(f"  ✓ Generated {len(health_data)} days of data for patient {patient.name}")
    except Exception as e:
        errors.append(f"Data generation error: {e}\n{traceback.format_exc()}")
        print(f"  ✗ Data generation failed: {e}")

    # Test 3: Lab data generates
    print("Testing lab data generation...")
    try:
        labs = generate_lab_history(
            patient_id="test",
            health_conditions=[],
            age=35,
            sex='M',
            num_panels=2
        )
        assert len(labs) == 2, f"Expected 2 panels, got {len(labs)}"
        # Check for any glucose-related marker (name varies due to title casing)
        has_glucose = any('glucose' in k.lower() or 'hba1c' in k.lower() for k in labs[0].results.keys())
        assert has_glucose, "Should have glucose markers"
        print(f"  ✓ Generated {len(labs)} lab panels with {len(labs[0].results)} biomarkers each")
    except Exception as e:
        errors.append(f"Lab generation error: {e}\n{traceback.format_exc()}")
        print(f"  ✗ Lab generation failed: {e}")

    # Test 4: Streamlit pages have no syntax errors
    print("Testing Streamlit page syntax...")
    import ast
    import os
    pages_dir = "pages"
    if os.path.exists(pages_dir):
        for filename in sorted(os.listdir(pages_dir)):
            if filename.endswith('.py'):
                filepath = os.path.join(pages_dir, filename)
                try:
                    with open(filepath, 'r') as f:
                        ast.parse(f.read())
                    print(f"  ✓ {filename} - syntax OK")
                except SyntaxError as e:
                    errors.append(f"Syntax error in {filename}: {e}")
                    print(f"  ✗ {filename} - syntax error: {e}")

    # Test 5: Main app has no syntax errors
    print("Testing main app syntax...")
    try:
        with open('app.py', 'r') as f:
            ast.parse(f.read())
        print("  ✓ app.py - syntax OK")
    except SyntaxError as e:
        errors.append(f"Syntax error in app.py: {e}")
        print(f"  ✗ app.py - syntax error: {e}")

    # Test 6: AI insights work (rule-based fallback)
    print("Testing AI insights...")
    try:
        from src.insights.ai_coach import _get_rule_based_insights, get_correlation_insights
        insights = _get_rule_based_insights(health_data)
        assert len(insights) > 0, "Should generate some insights"
        correlations = get_correlation_insights(health_data)
        print(f"  ✓ Generated insights ({len(insights)} chars) and {len(correlations)} correlations")
    except Exception as e:
        errors.append(f"AI insights error: {e}\n{traceback.format_exc()}")
        print(f"  ✗ AI insights failed: {e}")

    # Test 7: Check all required files exist
    print("Testing required files...")
    required_files = [
        'app.py',
        'requirements.txt',
        'README.md',
        '.streamlit/config.toml',
        'src/data/normalizer.py',
        'src/data/synthetic/patient_generator.py',
        'src/data/synthetic/lab_generator.py',
        'src/insights/ai_coach.py',
    ]
    for filepath in required_files:
        if os.path.exists(filepath):
            print(f"  ✓ {filepath} exists")
        else:
            errors.append(f"Missing file: {filepath}")
            print(f"  ✗ {filepath} MISSING")

    # Summary
    print("\n" + "="*50)
    if errors:
        print(f"VALIDATION FAILED - {len(errors)} error(s) found:\n")
        for i, err in enumerate(errors, 1):
            print(f"{i}. {err}\n")
        sys.exit(1)
    else:
        print("VALIDATION PASSED - All checks successful!")
        print("\nYou can now run: streamlit run app.py")
        sys.exit(0)

if __name__ == "__main__":
    validate()
