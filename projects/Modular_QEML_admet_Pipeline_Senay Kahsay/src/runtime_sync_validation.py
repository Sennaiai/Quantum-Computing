"""Local source-contract and independently decoded runtime validation.

No native compiler, deployed-program identity, or hardware behavior is certified
by these checks. Hashes identify the exact local source bundle inspected.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import re

from hamiltonian_validation import HamiltonianValidation
from runtime_input_protocol import (
    PROTOCOL_VERSION, build_runtime_input_document, canonical_json,
    validate_runtime_input_document, validate_qsvt_settings,
)

QBE_MODULES = ("main.qbe", "qsvt_mod.qbe", "walk_mod.qbe", "prep_mod.qbe",
               "block_encode.qbe", "sel_mod.qbe", "math.qbe", "reflect_mod.qbe")
EXPECTED_READS = ("num_coefs", "num_qubits", "num_ancillas", "num_cos_phases",
                  "num_sin_phases", "angle_value", "coefficient_value", "pauli_code",
                  "cos_phase_value", "sin_phase_value", "branch_angle")


def module_source_hashes(source_dir: Path, expected=None):
    source_dir = Path(source_dir).resolve(strict=True)
    hashes = {}
    for name in QBE_MODULES:
        path = source_dir / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Required local module must be a regular nonsymlink file: {name}")
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected is not None and hashes != expected:
        raise ValueError("Local QBE module hashes differ from the expected source bundle")
    return hashes


def verify_main_source_contract(main_source: str):
    """Check ordered read sites, not the semantics of a proprietary language."""
    source = "\n".join(line.split("#", 1)[0] for line in main_source.splitlines())
    reads = tuple(re.findall(r"\binput\s*->\s*(?:num\s+)?([A-Za-z_][A-Za-z_0-9]*)", source))
    if reads != EXPECTED_READS:
        raise ValueError("main.qbe ordered runtime read sites differ from protocol v2")
    if not re.search(r"pauli_code\s*->\s*pauli_strings\.\[flat_index\]", source):
        raise ValueError("main.qbe lacks the row-major Pauli storage site")
    if not re.search(r"ry\(\s*3\.141592653589793\s*\*\s*angle_value\s*\)", source):
        raise ValueError("main.qbe feature encoding differs from RY(pi*x)")
    return True


def verify_runtime_synchronization(payload, output_dir: Path, qsvt_settings, validator=None,
                                   *, expected_module_hashes=None):
    """Verify the local v2 source/serialization contract without compiling it."""
    source_dir = Path(output_dir)
    hashes = module_source_hashes(source_dir, expected_module_hashes)
    verify_main_source_contract((source_dir / "main.qbe").read_text(encoding="utf-8"))
    current = HamiltonianValidation(payload)
    branches = validate_qsvt_settings(qsvt_settings, current)
    document = build_runtime_input_document(payload, qsvt_settings, validator)
    decoded = validate_runtime_input_document(document)
    expected_fields = {
        "num_terms": current.num_terms,
        "num_system_qubits": current.num_system_qubits,
        "num_index_qubits": current.num_index_qubits,
        "feature_vector": current.feature_vector,
        "coefficients": current.coefficients,
        "pauli_rows": [list(row) for row in current.pauli_rows],
        "cosine_phases": branches["cosine"]["phase_angles"],
        "sine_phases": branches["sine"]["phase_angles"],
        "branch_rotation_angle": branches["branch_rotation_angle"],
    }
    for name, expected in expected_fields.items():
        if decoded[name] != expected:
            raise ValueError(f"Independent runtime decoding differs for {name}")
    if not math.isclose(decoded["lambda_value"], current.lambda_value, rel_tol=1e-12):
        raise ValueError("Decoded Hamiltonian norm differs from the source Hamiltonian")
    report = {
        "status": "local_contract_verified", "protocol_version": PROTOCOL_VERSION,
        "validation_scope": "local source read sites and independently decoded numeric stream",
        "module_policy": "fixed_source_bundle", "compiler_verified": False,
        "deployment_verified": False, "service_contract_verified": False,
        "phase_synthesis_verified_by_this_check": False,
        "dataset": payload.get("dataset"), "sample_index": payload.get("sample_index"),
        "num_terms": current.num_terms, "num_system_qubits": current.num_system_qubits,
        "num_index_qubits": current.num_index_qubits, "extra_ancillas": 2,
        "num_total_qubits": document["num_total_qubits"],
        "num_cosine_phases": decoded["num_cosine_phases"],
        "num_sine_phases": decoded["num_sine_phases"],
        "lambda_value": current.lambda_value,
        "physical_time": qsvt_settings["physical_time"], "epsilon": qsvt_settings["epsilon"],
        "qsvt_settings": {key: value for key, value in qsvt_settings.items() if key != "synchronization"},
        "runtime_input_count": document["input_count"],
        "runtime_inputs_sha256": document["inputs_sha256"], "module_hashes": hashes,
        "agp_metadata": payload.get("agp_metadata"),
        "curation_provenance": payload.get("curation_provenance"),
        "approximation_metadata": payload.get("approximation_metadata"),
        "runtime_document": document,
    }
    canonical_json(report)
    return report
