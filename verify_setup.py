"""
Verify environment setup and critical files existence.
"""
import sys
from pathlib import Path

FILES = [
    "run.py",
    "app.py",
    "api/main.py",
    "src/inference/predictor.py",
    "src/inference/predict_3days.py",
    "src/models/train_model.py",
    "src/features/feature_pipeline.py",
    "dashboard/components/gauge.py",
    "dashboard/components/forecast_cards.py",
    "dashboard/components/charts.py",
    "dashboard/utils/config.py",
    "dashboard/utils/data_loader.py",
    "config/config.yaml",
    ".github/workflows/feature_pipeline.yml",
    ".github/workflows/training_pipeline.yml",
    "requirements.txt",
]

def check_files():
    print("Checking file structure...")
    missing = []
    for f in FILES:
        if not Path(f).exists():
            missing.append(f)
            print(f"❌ Missing: {f}")
        else:
            print(f"✅ Found: {f}")

    if missing:
        print(f"\nCRITICAL: {len(missing)} files missing!")
        sys.exit(1)
    else:
        print("\n✅ All critical files present.")

def check_imports():
    print("\nChecking imports...")
    try:
        import plotly
        import streamlit
        import fastapi
        import hopsworks
        import lightgbm
        print("✅ Core dependencies importable.")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    check_files()
    check_imports()
