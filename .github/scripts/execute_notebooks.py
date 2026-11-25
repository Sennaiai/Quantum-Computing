import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[2]
SKIPPED_PROJECT_PREFIXES = (
    "Electicty load prediction forecasting",
    "Modular_QEML_admet_Pipeline",
)

def skip_reason(path):
    relative = path.relative_to(ROOT / "projects")
    if any(part in {"artifacts", ".ipynb_checkpoints", "__pycache__"} for part in relative.parts):
        return "generated artifact, not a source notebook"
    if relative.as_posix() == "Quanutm Chemistry /Quantum_Simulation_of_the_Hydrogen_Molecule_H2_Hamiltonian.ipynb":
        return "validated by dedicated H2 scientific tests and notebook workflow"
    if relative.as_posix() == "Quanutm Chemistry /Chemistry_HCN_VQE_Study.ipynb":
        return "validated by dedicated HCN scientific tests and notebook workflow"
    if relative.parts[0] == "QFT" or relative.as_posix() == "Shor_Senay_Kahsay.ipynb":
        return "relocated coursework; execution refinement pending"
    if relative.as_posix() == "Quanutm Chemistry /Molecular_Hamiltonian_Construction_Active_Space_Scaling.ipynb":
        return "validated by dedicated scaling scientific tests and notebook workflow"
    if relative.parts[0] == "Quantum Molecular Learning for ADMET":
        return "validated by the dedicated ADMET scientific experiment workflow"
    if any(parent.startswith(prefix) for parent in relative.parts[:-1]
           for prefix in SKIPPED_PROJECT_PREFIXES):
        return "external data, uploads, or credentials required"
    return None


def main():
    os.environ["NOTEBOOK_CI"] = "1"
    project_root = ROOT / "projects"
    notebooks = sorted(project_root.rglob("*.ipynb"))
    runnable = [path for path in notebooks if skip_reason(path) is None]

    if not runnable:
        raise SystemExit("No offline Quantum-Computing notebooks were found to execute.")

    for path in notebooks:
        if path not in runnable:
            reason = skip_reason(path)
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
