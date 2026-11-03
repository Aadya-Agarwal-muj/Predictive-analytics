"""
run_all.py - reproduce the whole project with one command:

    python run_all.py

Runs scripts 01 -> 06 in order with the same Python interpreter and stops at
the first error. Afterwards start the dashboard with:

    streamlit run dashboard/app.py
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS = [
    "01_inspect_archive.py",
    "02_extract_dataset.py",
    "03_preprocess.py",
    "04_eda.py",
    "05_modeling.py",
    "06_evaluation.py",
]

for name in SCRIPTS:
    print(f"\n{'#' * 70}\n# Running scripts/{name}\n{'#' * 70}", flush=True)
    # sys.executable = the python currently running this file (same venv).
    subprocess.run([sys.executable, str(ROOT / "scripts" / name)], cwd=ROOT, check=True)

print("\nAll stages completed. Outputs are in outputs/ ; data in data/processed/.")
print("Launch the dashboard with:  streamlit run dashboard/app.py")
