"""Execute all notebooks in order, in place. Needs the raw Kaggle CSV in data/raw/.

Notebook 04 retrains and overwrites the saved models, which takes about half an hour.

Run from the project root: python -m scripts.run_notebooks
"""
import subprocess
import sys

from src.data import PROJECT_ROOT


def main():
    for path in sorted((PROJECT_ROOT / "notebooks").glob("[0-9][0-9]_*.ipynb")):
        print(path.name, flush=True)
        subprocess.run([sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute",
                        "--inplace", "--ExecutePreprocessor.timeout=7200", str(path)], check=True)


if __name__ == "__main__":
    main()
