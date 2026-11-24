import importlib.util
import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[2]
RESOURCE_INTENSIVE_NOTEBOOKS = {
    Path("Quanutm Chemistry ")
    / "Molecular_Hamiltonian_Construction_Active_Space_Scaling.ipynb",
    Path("Electicty load prediction forecasting ")
    / "Electric grid project"
    / "Kerr_CV_QRC.ipynb",
}
LOCAL_INPUTS = {
    Path("Electicty load prediction forecasting ") / "Classical baseline.ipynb": (
        Path("projects/Electicty load prediction forecasting ")
        / "QUAC 605 Project Data.xlsx",
    ),
    Path("Electicty load prediction forecasting ")
    / "Hybrid_Quantum_Classical_Reservoir_Computing_for_Electricity_Grid_Load_Forecasting.ipynb": (
        Path("projects/Electicty load prediction forecasting ")
        / "QUAC 605 Project Data.xlsx",
    ),
    Path("Electicty load prediction forecasting ")
    / "Electric grid project"
    / "605 Qubit QRC Code - Copy.ipynb": (
        Path("projects/Electicty load prediction forecasting ")
        / "QUAC 605 Project Data.xlsx",
    ),
    Path("Electicty load prediction forecasting ") / "cv_6_QRC.ipynb": (
        Path("projects/Electicty load prediction forecasting ")
        / "Electric grid project"
        / "cv_QRC"
        / "QUAC 605 Project Data_processed.xlsx - 3 month_processed.csv",
    ),
    Path("Electicty load prediction forecasting ")
    / "Hybrid_CV_Kerr_Quantum_Reservoir_Computing_for_Electricity_Grid_Load_Forecasting.ipynb": (
        Path("projects/Electicty load prediction forecasting ")
        / "Electric grid project"
        / "cv_QRC"
        / "QUAC 605 Project Data_processed.xlsx - 3 month_processed.csv",
    ),
    Path("Electicty load prediction forecasting ") / "Kerr_CV_QRC.ipynb": (
        Path("projects/Electicty load prediction forecasting ")
        / "Electric grid project"
        / "cv_QRC"
        / "QUAC 605 Project Data_processed.xlsx - 3 month_processed.csv",
    ),
}


def skip_reason(path: Path, source: str) -> str | None:
    relative_path = path.relative_to(ROOT / "projects")
    if relative_path in RESOURCE_INTENSIVE_NOTEBOOKS:
        return "active-space scaling exceeds the routine execution budget"
    if "ADMET_Triple_Alpha_End_to_End.ipynb" in path.name:
        if not (
            os.environ.get("TRIPLE_ALPHA_API_KEY")
            and os.environ.get("TRIPLE_ALPHA_RUN_URL")
        ):
            return "Triple Alpha credentials and service are not configured"
    if "google.colab" in source or "files.upload()" in source:
        return "requires the Google Colab upload runtime"
    if "strawberryfields.fock" in source and importlib.util.find_spec(
        "pennylane_sf"
    ) is None:
        return "requires the PennyLane Strawberry Fields plugin"
    for required_file in LOCAL_INPUTS.get(relative_path, ()):
        if not (ROOT / required_file).is_file():
            return f"local input is missing: {required_file}"
    return None


def main():
    os.environ["NOTEBOOK_CI"] = "1"
    project_root = ROOT / "projects"
    notebooks = sorted(project_root.rglob("*.ipynb"))

    runnable = []
    for path in notebooks:
        notebook = nbformat.read(path, as_version=4)
        source = "\n".join(
            cell.source for cell in notebook.cells if cell.cell_type == "code"
        )
        reason = skip_reason(path, source)
        if reason:
            print(f"SKIP {path.relative_to(ROOT)}: {reason}", flush=True)
        else:
            runnable.append(path)

    if not runnable:
        raise SystemExit("No runnable Quantum-Computing project notebooks were found.")

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
