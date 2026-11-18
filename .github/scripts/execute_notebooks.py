import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[2]
SKIPPED_PROJECT_PREFIXES = (
    "Electicty load prediction forecasting",
    "Modular_QEML_admet_Pipeline",
)
RESOURCE_INTENSIVE_NOTEBOOKS = {
    "Molecular_Hamiltonian_Construction_Active_Space_Scaling.ipynb",
}


def main():
    os.environ["NOTEBOOK_CI"] = "1"
    project_root = ROOT / "projects"
    notebooks = sorted(project_root.rglob("*.ipynb"))
    runnable = [
        path
        for path in notebooks
        if path.name not in RESOURCE_INTENSIVE_NOTEBOOKS
        and not any(
            parent.startswith(prefix)
            for parent in path.relative_to(project_root).parts[:-1]
            for prefix in SKIPPED_PROJECT_PREFIXES
        )
    ]

    if not runnable:
        raise SystemExit("No offline Quantum-Computing notebooks were found to execute.")

    for path in notebooks:
        if path not in runnable:
            reason = (
                "resource-intensive active-space scaling"
                if path.name in RESOURCE_INTENSIVE_NOTEBOOKS
                else "external data, uploads, or credentials required"
            )
            print(f"SKIP {path.relative_to(ROOT)}: {reason}", flush=True)

    for path in runnable:
        print(f"RUN {path.relative_to(ROOT)}", flush=True)
        notebook = nbformat.read(path, as_version=4)
        client = NotebookClient(
            notebook,
            kernel_name="python3",
            timeout=900,
            resources={"metadata": {"path": str(path.parent)}},
        )
        client.execute()
        print(f"PASS {path.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
