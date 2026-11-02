"""Execute all notebooks in order, in place, then the check scripts. Needs the raw CSV in data/raw/.

Notebook 04 retrains and overwrites the saved models, which takes about half an hour.

Run from the project root: python -m scripts.run_notebooks
"""
import subprocess
import sys

from src.data import PROJECT_ROOT

SCRIPTS = ["lender_fields_sensitivity", "model_v2_experiment", "lifetime_check",
           "lifetime_temporal_check", "champion_feature_psi", "ablation_2016_fields", "ablation_flag", "reason_code_check",
           "realised_money", "training_regime", "prepayment_check"]  # confirm_2018 is a one-time check and is left out


def main():
    for path in sorted((PROJECT_ROOT / "notebooks").glob("[0-9][0-9]_*.ipynb")):
        print(path.name, flush=True)
        subprocess.run([sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute",
                        "--inplace", "--ExecutePreprocessor.timeout=7200", str(path)], check=True)
    for name in SCRIPTS:
        print(name, flush=True)
        subprocess.run([sys.executable, "-m", f"scripts.{name}"], cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
