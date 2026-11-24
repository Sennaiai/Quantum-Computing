"""Reproduce bounded quantum-control and optional measured-data experiments.

The only permitted network operation is explicit public BBB dataset acquisition.
All experiment code runs with Python socket connections denied. No Triple Alpha
client, credentials, native compilation, or paid API is involved.
"""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
for setting in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(setting, "1")
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "python_classes")]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "benchmarks")
    parser.add_argument("--bbb-file", type=Path, help="Pinned official BBB_Martins raw table; no network")
    parser.add_argument("--download-bbb", action="store_true", help="Explicitly fetch the pinned public dataset")
    args = parser.parse_args(argv)
    if args.bbb_file and args.download_bbb:
        parser.error("--bbb-file and --download-bbb are mutually exclusive")
    from artifact_paths import validate_output_dir
    output = validate_output_dir(args.output_dir, project_root=ROOT)
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".benchmark-cache" / "matplotlib"))
    from admet_submission_entry import environment_versions, source_hashes
    from runtime_input_protocol import atomic_write_json, canonical_json
    before = source_hashes()
    atomic_write_json(output / "benchmark_summary.json", {"status": "incomplete", "source_hashes": before})
    bbb_file = args.bbb_file
    if args.download_bbb:
        from data_benchmark import download_bbb
        bbb_file = download_bbb(output / ".benchmark-data" / "bbb_martins.tab")
    from offline_guard import install
    install()
    from admet_curation import FIXTURE_NAME, molecule_problem_input, run_complete_curation
    from transport_benchmark import run_transport_benchmark
    from benchmark_reporting import plot_transport, plot_information
    curation = run_complete_curation()
    adapter, theta, provenance = molecule_problem_input(curation, FIXTURE_NAME, 13)
    transport = run_transport_benchmark(adapter, FIXTURE_NAME, theta)
    transport["curation_provenance"] = provenance
    data = {"status": "not_run", "reason": "Supply --bbb-file or explicitly --download-bbb"}
    learning = {"status": "not_run", "reason": "Supply the pinned BBB artifact"}
    if bbb_file is not None:
        from data_benchmark import run_data_benchmark
        data = run_data_benchmark(bbb_file, output_dir=output / ".benchmark-work" / "real_data")
        if data["status"] != "passed":
            raise ValueError("No fixed real-data profile reached the complete runtime handoff; inspect the diagnostic report")
        import io
        import pandas as pd
        from data_benchmark import verify_bbb_file, BBB_SHA256
        from learning_workflow import prepare_learning_workflow
        raw = verify_bbb_file(bbb_file).read_bytes()
        if hashlib.sha256(raw).hexdigest() != BBB_SHA256:
            raise ValueError("BBB bytes changed before supervised preparation")
        _, learning = prepare_learning_workflow(pd.read_csv(io.BytesIO(raw),sep="\t"),
            "BBB_Martins",output/".benchmark-work"/"learning")
        learning["data_source_sha256"] = BBB_SHA256
    if source_hashes() != before:
        raise RuntimeError("Project sources changed during the experiment; rerun on a stable source tree")
    source_digest = hashlib.sha256(canonical_json(before).encode()).hexdigest()
    report = {
        "schema_version": 1, "status": "passed", "source_hashes": before,
        "source_tree_sha256": source_digest,
        "environment": {**environment_versions(), "matplotlib": version("matplotlib")},
        "units": "chosen dimensionless energy/time scale, hbar=1; no physical ADMET calibration",
        "encoding": "data-derived information/feature Ising model; not an electronic-structure Hamiltonian",
        "transport": transport, "real_data": data, "supervised_learning": learning,
        "network_policy": "explicit optional public data acquisition, then denied experiment connections",
        "native_compilation": "unverified", "triple_alpha_calls": 0,
        "interpretation": "bounded numerical experiments and measured classical held-out baselines; quantum readout awaits results, without universal CD improvement, hardware accuracy, or speedup claims",
    }
    plot_transport(transport, output / "transport.svg")
    files = ["benchmark_summary.json", "transport.svg"]
    if data["status"] == "passed":
        plot_information(data, output / "information.svg")
        files.append("information.svg")
    atomic_write_json(output / "benchmark_summary.json", report)
    manifest = {"schema_version": 1, "source_tree_sha256": source_digest,
                "files": {name: hashlib.sha256((output / name).read_bytes()).hexdigest() for name in files},
                "raw_dataset_committed": False, "triple_alpha_calls": 0}
    atomic_write_json(output / "manifest.json", manifest)
    print(json.dumps({"status": "passed", "output": str(output),
        "transport_runs": len(transport["rows"]), "real_data_status": data["status"],
        "source_tree_sha256": source_digest}, indent=2))


if __name__ == "__main__":
    main()
