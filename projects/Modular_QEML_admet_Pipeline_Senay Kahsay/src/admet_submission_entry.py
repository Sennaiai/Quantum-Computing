"""Orchestrate the validated ADMET counterdiabatic-to-QSVT pipeline.

The circuit is assumed to be compiled and deployed already. This module only
constructs and validates the ordered runtime input stream.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import json
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts"
VALIDATED_HCD_PATH = ARTIFACT_DIR / "HCD_hamiltonians.pkl"
CURATED_HAMILTONIANS_PATH = ARTIFACT_DIR / "curated_admet_hamiltonians.pkl"
PYTHON_CLASSES_DIR = ROOT / "python_classes"
PIPELINE_CLASSES_DIR = PYTHON_CLASSES_DIR
DEFAULT_OUTPUT_DIR = ROOT / "qbe_modules"
for candidate in (PYTHON_CLASSES_DIR, PIPELINE_CLASSES_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

# Notebook-derived, numerically validated implementations.
from hamiltonian_validation import HamiltonianValidation  # noqa: E402
from qsvt_helper import QSVTHelper  # noqa: E402
from classical_preprocessing import AGPSettings, recalculate_counterdiabatic_hamiltonian  # noqa: E402
from runtime_sync_validation import verify_runtime_synchronization  # noqa: E402


def load_validated_hamiltonians(path: Path = VALIDATED_HCD_PATH) -> dict[str, Any]:
    """Load the counterdiabatic Hamiltonians exported by the validated notebooks."""
    with path.open("rb") as handle:
        return pickle.load(handle)


def load_curated_hamiltonians(path: Path = CURATED_HAMILTONIANS_PATH) -> dict[str, Any]:
    """Load the Stage-6 Hamiltonian families used for AGP recalculation."""
    with path.open("rb") as handle:
        return pickle.load(handle)


def build_runtime_payload(
    hamiltonians: dict[str, Any],
    dataset_key: str = "BBB_Martins",
    sample_index: int = 0,
    term_limit: int | None = 10,
) -> dict[str, Any]:
    """Select one runtime Hamiltonian/sample without reconstructing its physics."""
    if dataset_key not in hamiltonians:
        raise KeyError(f"Unknown dataset {dataset_key!r}; choose from {sorted(hamiltonians)}")

    source = hamiltonians[dataset_key]
    terms = list(source["pauli_terms"])
    if term_limit is not None:
        if term_limit < 1:
            raise ValueError("term_limit must be positive or None")
        terms = terms[:term_limit]
    if not terms:
        raise ValueError("The selected Hamiltonian contains no Pauli terms")

    samples = source["X_selected_scaled"]
    if not 0 <= sample_index < len(samples):
        raise IndexError(f"sample_index must be in 0..{len(samples) - 1}")

    return {
        "dataset": dataset_key,
        "hamiltonian_type": source["hamiltonian_type"],
        "formula": source["formula"],
        "selected_features": list(source["selected_features"]),
        "pauli_terms": terms,
        "feature_vector": [float(value) for value in samples[sample_index]],
        "sample_index": sample_index,
    }


def build_validator(payload: dict[str, Any]) -> HamiltonianValidation:
    validator = HamiltonianValidation(payload)
    validator.validate()
    return validator


def prepare_qsvt_settings(
    validator: HamiltonianValidation,
    physical_time: float = 1.0,
    epsilon: float = 1e-6,
) -> dict[str, Any]:
    """Synthesize validated Low-Chuang phases from runtime λ, time, and error."""
    helper = QSVTHelper(validator, physical_time=physical_time, epsilon=epsilon)
    phases = helper.cosine_phases()
    return {
        "lambda_value": helper.lambda_value,
        "physical_time": helper.physical_time,
        "epsilon": helper.epsilon,
        "tau": helper.tau,
        "polynomial_degree": helper.cosine_degree,
        "phase_angles": [float(phi) for phi in phases],
    }


def prepare_runtime_inputs(
    payload: dict[str, Any],
    output_dir: Path | None = None,
    physical_time: float = 1.0,
    epsilon: float = 1e-6,
) -> tuple[Path, dict[str, Any]]:
    """Prepare and validate inputs for the fixed, already-compiled circuit."""
    output_dir = output_dir or DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    validator = build_validator(payload)
    qsvt_settings = prepare_qsvt_settings(validator, physical_time, epsilon)
    qsvt_settings["synchronization"] = verify_runtime_synchronization(
        payload, output_dir, qsvt_settings, validator
    )
    return output_dir, qsvt_settings


def update_compiled_job(
    runtime_inputs: list[float],
    api_key: str | None = None, run_url: str | None = None,
    job_id: str | None = None,
) -> dict[str, Any]:
    """Update a compiled Triple Alpha job using only its runtime input stream."""
    api_key = api_key or os.getenv("TRIPLE_ALPHA_API_KEY")
    run_url = run_url or os.getenv("TRIPLE_ALPHA_RUN_URL")
    job_id = job_id or os.getenv("TRIPLE_ALPHA_JOB_ID")
    if not api_key or not run_url or not job_id:
        raise ValueError("api key, run URL, and compiled job ID are required")

    update_url = run_url if run_url.rstrip("/").endswith("run/update_input") else run_url.rstrip("/") + "/run/update_input"
    request = urllib.request.Request(
        update_url,
        data=json.dumps({"job_id": job_id, "inputs": runtime_inputs}).encode("utf-8"),
        headers={"X-Api-Key": api_key, "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return {"status": "updated", "response": response.read().decode("utf-8")}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="BBB_Martins")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--term-limit", type=int, default=10)
    parser.add_argument("--recalculate-agp", action="store_true")
    parser.add_argument("--interpolation", type=float, default=0.5)
    parser.add_argument("--lambda-dot", type=float, default=1.0)
    parser.add_argument("--physical-time", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=1e-6)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--job-id", default=None)
    parser.add_argument("--submit", action="store_true", help="Update an existing compiled job")
    args = parser.parse_args(argv)

    validated = load_validated_hamiltonians()
    if args.recalculate_agp:
        source = validated[args.dataset]
        if not 0 <= args.sample_index < len(source["X_selected_scaled"]):
            raise IndexError("sample-index is outside the selected dataset")
        payload = recalculate_counterdiabatic_hamiltonian(
            load_curated_hamiltonians(),
            args.dataset,
            source["X_selected_scaled"][args.sample_index],
            AGPSettings(
                interpolation=args.interpolation,
                lambda_dot=args.lambda_dot,
                interaction_term_limit=args.term_limit,
            ),
        )
        payload["sample_index"] = args.sample_index
    else:
        payload = build_runtime_payload(
            validated, args.dataset, args.sample_index, args.term_limit
        )
    output_dir, qsvt = prepare_runtime_inputs(
        payload, args.output_dir, args.physical_time, args.epsilon,
    )
    runtime_document = qsvt["synchronization"]["runtime_document"]
    print(f"Runtime inputs validated against: {output_dir / 'main.qbe'}")
    print(
        f"terms={len(payload['pauli_terms'])}, lambda={qsvt['lambda_value']:.12g}, "
        f"index_qubits={build_validator(payload).num_index_qubits}, "
        f"qsvt_degree={qsvt['polynomial_degree']}"
    )
    print(f"runtime_inputs={runtime_document['input_count']}, sha256={runtime_document['inputs_sha256']}")
    if args.submit:
        print(update_compiled_job(runtime_document["inputs"], job_id=args.job_id))


if __name__ == "__main__":
    main()
