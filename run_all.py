
import subprocess
import sys
from pathlib import Path

SCRIPTS = [
    "01_data_eda.py",
    "02_preprocessing_imbalance.py",
    "03_train_lr.py",
    "04_train_xgb.py",
    "05_compare_models.py",
]

def main():
    here = Path(__file__).resolve().parent
    for s in SCRIPTS:
        print("\n=== Running", s, "===")
        subprocess.check_call([sys.executable, str(here / s)])

if __name__ == "__main__":
    main()
