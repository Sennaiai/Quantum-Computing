"""Validate immutable Beryllium modules and per-molecule runtime inputs."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from runtime_input_protocol import build_runtime_input_document, build_runtime_input_values


def verify_runtime_synchronization(payload, output_dir: Path, qsvt_settings, validator) -> dict[str, Any]:
    """Validate inputs against the fixed, already-compiled main-module contract."""
    main = (output_dir / "main.qbe").read_text(encoding="utf-8")
    reads = (
        "input -> num num_coefs", "input -> num num_qubits",
        "input -> num num_ancillas", "input -> num num_phases",
        "input -> angle_value", "input -> coefficient_value",
        "input -> pauli_code", "input -> phase_value",
        "pauli_code -> pauli_strings.[flat_index]",
    )
    missing = [read for read in reads if read not in main]
    if missing:
        raise ValueError(f"main.qbe runtime input contract is incomplete: {missing}")

    classical_lambda = sum(abs(float(term["coefficient"])) for term in payload["pauli_terms"])
    if not math.isclose(classical_lambda, float(qsvt_settings["lambda_value"]), rel_tol=1e-12):
        raise ValueError("Hamiltonian and QSVT lambda differ")

    runtime_document = build_runtime_input_document(payload, qsvt_settings, validator)
    # Rebuild independently; this catches serializer/document drift.
    if runtime_document["inputs"] != build_runtime_input_values(payload, qsvt_settings, validator):
        raise ValueError("Serialized runtime inputs do not match preprocessing")

    report = {
        "status": "synchronized",
        "module_policy": "fixed_precompiled_circuit",
        "dataset": payload.get("dataset"),
        "sample_index": payload.get("sample_index"),
        "num_terms": validator.num_terms,
        "num_system_qubits": validator.num_system_qubits,
        "num_index_qubits": validator.num_index_qubits,
        "num_qsvt_phases": len(qsvt_settings["phase_angles"]),
        "lambda_value": float(qsvt_settings["lambda_value"]),
        "physical_time": float(qsvt_settings["physical_time"]),
        "epsilon": float(qsvt_settings["epsilon"]),
        "polynomial_degree": int(qsvt_settings["polynomial_degree"]),
        "runtime_input_count": runtime_document["input_count"],
        "runtime_inputs_sha256": runtime_document["inputs_sha256"],
        "agp_metadata": payload.get("agp_metadata"),
        "curation_provenance": payload.get("curation_provenance"),
        "runtime_document": runtime_document,
    }
    return report
