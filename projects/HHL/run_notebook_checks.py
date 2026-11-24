from pathlib import Path

import nbformat
from nbclient import NotebookClient


NOTEBOOK = Path(__file__).with_name(
    "Quantum_Algorithms_for_Solving_Linear_Systems_HHL_and_Variational_Methods.ipynb"
)


def main() -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    NotebookClient(
        notebook,
        kernel_name="python3",
        timeout=900,
        resources={"metadata": {"path": str(NOTEBOOK.parent)}},
    ).execute()
    print(f"PASS {NOTEBOOK}")


if __name__ == "__main__":
    main()
