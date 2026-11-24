"""Run scientific tests and all three notebooks offline in fresh Python kernels."""
from __future__ import annotations

import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parent
for thread_setting in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(thread_setting, "1")
NOTEBOOKS = ("pipeline/ADMET_future_curation_complete.ipynb",
             "pipeline/Counter-diabatic hamiltonian_complete.ipynb",
             "ADMET_Triple_Alpha_End_to_End.ipynb")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-notebooks", action="store_true", help="Tests only; not the full acceptance check")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "validation")
    args = parser.parse_args(argv)
    sys.path[:0] = [str(ROOT / "src"), str(ROOT / "python_classes")]
    from artifact_paths import validate_output_dir
    output = validate_output_dir(args.output_dir, project_root=ROOT)
    output.mkdir(parents=True, exist_ok=True)
    for name in ("MPLCONFIGDIR", "IPYTHONDIR", "JUPYTER_RUNTIME_DIR"):
        os.environ[name] = str(output / name.lower())
        Path(os.environ[name]).mkdir(exist_ok=True)
    os.environ["ADMET_OUTPUT_DIR"] = str(output / "runtime")
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    from admet_submission_entry import environment_versions, source_hashes
    initial_hashes = source_hashes()
    (output / "validation_summary.json").write_text(json.dumps({
        "status": "incomplete", "source_hashes": initial_hashes,
        "reason": "acceptance run has not completed"}, indent=2)+"\n")
    compatibility = subprocess.run([sys.executable, "-m", "pip", "check"], check=False)
    if compatibility.returncode:
        raise SystemExit(compatibility.returncode)
    from offline_guard import install
    install()
    started = time.monotonic()
    tests = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
    result = unittest.TextTestRunner(verbosity=2).run(tests)
    if not result.wasSuccessful():
        raise SystemExit(1)
    executed = []
    if not args.skip_notebooks:
        import nbformat
        from nbclient import NotebookClient
        from jupyter_client import KernelManager
        for relative in NOTEBOOKS:
            path = ROOT / relative
            notebook = nbformat.read(path, as_version=4)
            nbformat.validate(notebook)
            guard = nbformat.v4.new_code_cell(
                "import sys\n"
                f"sys.path.insert(0, {str(ROOT / 'src')!r})\n"
                "from offline_guard import install\ninstall()\n")
            notebook.cells.insert(0, guard)
            manager = KernelManager(kernel_name="python3")
            manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
            client = NotebookClient(notebook, km=manager, timeout=300,
                                    resources={"metadata": {"path": str(path.parent)}})
            print(f"RUN {relative}", flush=True)
            try:
                client.execute()
            finally:
                if manager.has_kernel:
                    manager.shutdown_kernel(now=True)
            notebook.cells.pop(0)
            destination = output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            nbformat.write(notebook, destination)
            executed.append(relative)
            print(f"PASS {relative}", flush=True)
    if source_hashes() != initial_hashes:
        raise RuntimeError("Project sources changed during verification; rerun against a stable source tree")
    report = {"status": "passed", "tests": result.testsRun, "failures": 0,
              "skipped_tests": len(result.skipped), "executed_notebooks": executed,
              "notebooks_skipped_explicitly": args.skip_notebooks,
              "network_policy": "denied in test process and notebook cells",
              "environment": environment_versions(), "source_hashes": source_hashes(),
              "elapsed_seconds": time.monotonic()-started,
              "native_compilation": "not performed", "live_api": "not performed",
              "real_admet_data_validation": (
                  "pinned BBB artifact executed in curation notebook" if os.environ.get("ADMET_BBB_FILE") and not args.skip_notebooks
                  else "published source-hashed BBB evidence checked; raw-data notebook execution not requested"),
              "transport_experiment": "executed in CD notebook" if not args.skip_notebooks else "notebook experiment skipped explicitly",
              "published_evidence": "checksums, source identity, and numerical acceptance checked by tests"}
    (output / "validation_summary.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({key: report[key] for key in ("status", "tests", "executed_notebooks")}, indent=2))


if __name__ == "__main__":
    main()
