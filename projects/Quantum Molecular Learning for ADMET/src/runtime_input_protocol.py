"""Versioned, strict serialization for the local Beryllium input contract.

Version 2 carries both parity branches of a heralded cosine/sine LCU. This
module validates a source-level contract; it does not certify native execution.
"""
from __future__ import annotations

import hashlib
import json
import math
import numbers
import os
from pathlib import Path
import tempfile
from typing import Any

from hamiltonian_validation import (
    HamiltonianValidation, PAULI_CODE, finite_real, positive_integer,
    validate_feature_vector,
)

PROTOCOL_VERSION = 2
MAX_RUNTIME_INPUTS = 1_000_000
ALGORITHM = "heralded_cosine_sine_lcu_v2"
PHASE_CONVENTION = "projector_exp_i_phi_R"
LAYOUT = ["num_terms", "num_system_qubits", "num_index_qubits",
          "num_cosine_phases", "num_sine_phases",
          "features[num_system_qubits] encoded as RY(pi*x)",
          "for each term: signed_coefficient, pauli_codes[num_system_qubits]",
          "cosine_phases[num_cosine_phases]", "sine_phases[num_sine_phases]",
          "branch_rotation_angle"]


def canonical_json(value) -> str:
    def numeric_scalar(item):
        # NumPy scalar integers/reals occur naturally in scientific callers.
        # Preserve strict finite JSON without accepting arbitrary objects or
        # invoking object-provided serialization hooks.
        if isinstance(item, numbers.Integral) and not isinstance(item, bool):
            return int(item)
        if isinstance(item, numbers.Real) and not isinstance(item, bool):
            return finite_real(item)
        raise TypeError(f"Unsupported JSON value type: {type(item).__name__}")
    return json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"), default=numeric_scalar)


def inputs_sha256(values) -> str:
    return hashlib.sha256(canonical_json(values).encode("utf-8")).hexdigest()


def dense_pauli_rows(payload, num_qubits: int) -> list[list[int]]:
    validator = HamiltonianValidation(payload)
    if validator.num_system_qubits != num_qubits:
        raise ValueError("Pauli register size differs from selected_features")
    return [list(row) for row in validator.pauli_rows]


def validate_qsvt_settings(settings, validator) -> dict[str, Any]:
    if not isinstance(settings, dict) or settings.get("algorithm") != ALGORITHM:
        raise ValueError(f"Runtime protocol v2 requires algorithm={ALGORITHM!r}")
    if settings.get("phase_convention") != PHASE_CONVENTION:
        raise ValueError("Unsupported QSVT phase convention")
    lam = finite_real(settings.get("lambda_value"), "lambda_value")
    time = finite_real(settings.get("physical_time"), "physical_time")
    epsilon = finite_real(settings.get("epsilon"), "epsilon")
    tau = finite_real(settings.get("tau"), "tau")
    if lam <= 0 or time <= 0 or tau <= 0 or not 0 < epsilon < 1:
        raise ValueError("lambda, time and tau must be positive; epsilon must lie in (0, 1)")
    if not math.isclose(lam, validator.lambda_value, rel_tol=1e-12):
        raise ValueError("Hamiltonian and QSVT lambda differ")
    expected_tau = lam * time
    if not math.isfinite(expected_tau) or expected_tau <= 0:
        raise ValueError("lambda * physical_time overflowed or underflowed")
    if not math.isclose(tau, expected_tau, rel_tol=1e-12):
        raise ValueError("QSVT tau differs from lambda * physical_time")
    result, error_bounds = {}, []
    for name, parity in (("cosine", 0), ("sine", 1)):
        branch = settings.get(name)
        if not isinstance(branch, dict):
            raise ValueError(f"Missing {name} QSVT branch")
        raw_phases = branch.get("phase_angles")
        if not isinstance(raw_phases, (list, tuple)) or not raw_phases:
            raise ValueError(f"{name}.phase_angles must be a nonempty sequence")
        phases = [finite_real(value, f"{name}.phase_angles") for value in raw_phases]
        degree = positive_integer(branch.get("polynomial_degree"), f"{name}.polynomial_degree", allow_zero=True)
        if degree != len(phases) - 1 or degree % 2 != parity:
            raise ValueError(f"{name} polynomial degree, phase count, or parity is inconsistent")
        scale = finite_real(branch.get("scale"), f"{name}.scale")
        if not 0 < scale <= 1:
            raise ValueError(f"{name}.scale must lie in (0, 1]")
        for key in ("approximation_error_bound", "synthesis_error_bound"):
            error_bound = finite_real(branch.get(key), f"{name}.{key}")
            if error_bound < 0:
                raise ValueError(f"{name}.{key} must be nonnegative")
            error_bounds.append(error_bound)
        result[name] = {"phase_angles": phases, "scale": scale}
    beta = finite_real(settings.get("success_amplitude"), "success_amplitude")
    sc, ss = result["cosine"]["scale"], result["sine"]["scale"]
    # Avoid reciprocals of subnormal scales; fail if probabilities cannot be
    # represented rather than accepting an unheraldable zero-amplitude block.
    minimum_scale, maximum_scale = min(sc, ss), max(sc, ss)
    expected_beta = minimum_scale / (1 + minimum_scale / maximum_scale)
    if expected_beta <= 0 or expected_beta * expected_beta == 0:
        raise ValueError("Branch scales underflow the representable success probability")
    angle = finite_real(settings.get("branch_rotation_angle"), "branch_rotation_angle")
    expected_angle = 2 * math.acos(math.sqrt(expected_beta / result["cosine"]["scale"]))
    if beta <= 0 or not math.isclose(beta, expected_beta, rel_tol=1e-12):
        raise ValueError("Branch scales and success amplitude are inconsistent")
    if not math.isclose(angle, expected_angle, rel_tol=1e-12, abs_tol=1e-14):
        raise ValueError("Branch scales and rotation angle are inconsistent")
    recovered_error = finite_real(settings.get("recovered_evolution_error_bound"), "recovered_evolution_error_bound")
    if (recovered_error < 0 or recovered_error > epsilon
            or not math.isclose(recovered_error, math.fsum(error_bounds), rel_tol=1e-12, abs_tol=1e-16)):
        raise ValueError("Recovered evolution error budget is inconsistent or exceeds epsilon")
    for key, expected in (("ideal_success_probability", beta * beta),
                          ("success_probability_lower_bound", beta * beta * (1 - recovered_error) ** 2)):
        probability = finite_real(settings.get(key), key)
        if probability <= 0 or not math.isclose(probability, expected, rel_tol=1e-12):
            raise ValueError(f"{key} is inconsistent with the branch/error metadata")
    if positive_integer(settings.get("extra_ancillas"), "extra_ancillas") != 2:
        raise ValueError("Protocol v2 requires two additional quantum ancillas")
    result["branch_rotation_angle"] = angle
    return result


def _validated_payload(payload, validator=None):
    current = HamiltonianValidation(payload)
    if validator is not None and (
        current.num_system_qubits != validator.num_system_qubits
        or current.coefficients != validator.coefficients
        or current.pauli_strings != validator.pauli_strings
        or current.selected_features != validator.selected_features
    ):
        raise ValueError("Validator does not describe the supplied payload")
    if "feature_vector" not in payload:
        raise ValueError("Runtime payload requires feature_vector")
    current.feature_vector = validate_feature_vector(payload["feature_vector"], current.num_system_qubits)
    if "sample_index" in payload:
        positive_integer(payload["sample_index"], "sample_index", allow_zero=True)
    if "dataset" in payload and (not isinstance(payload["dataset"], str) or not payload["dataset"].strip()):
        raise ValueError("dataset must be a nonempty string")
    return current


def expected_input_count(num_terms: int, num_qubits: int, num_cosine_phases: int, num_sine_phases: int) -> int:
    nt = positive_integer(num_terms, "num_terms")
    nq = positive_integer(num_qubits, "num_qubits")
    nc = positive_integer(num_cosine_phases, "num_cosine_phases")
    ns = positive_integer(num_sine_phases, "num_sine_phases")
    count = 5 + nq + nt * (1 + nq) + nc + ns + 1
    if count > MAX_RUNTIME_INPUTS:
        raise ValueError("Runtime stream exceeds the bounded local input limit")
    return count


def build_runtime_input_values(payload, qsvt_settings, validator=None) -> list[float]:
    current = _validated_payload(payload, validator)
    branches = validate_qsvt_settings(qsvt_settings, current)
    cosine, sine = branches["cosine"]["phase_angles"], branches["sine"]["phase_angles"]
    expected_input_count(current.num_terms, current.num_system_qubits, len(cosine), len(sine))
    values = [float(current.num_terms), float(current.num_system_qubits),
              float(current.num_index_qubits), float(len(cosine)), float(len(sine))]
    values.extend(current.feature_vector)
    for coefficient, row in zip(current.coefficients, current.pauli_rows):
        values.append(coefficient)
        values.extend(float(code) for code in row)
    values.extend(cosine)
    values.extend(sine)
    values.append(branches["branch_rotation_angle"])
    return values


def decode_runtime_input_values(values) -> dict[str, Any]:
    """Independently consume the QBE stream, rejecting missing/trailing values.

    The decoder deliberately does not invoke the packer or trust its metadata.
    It bounds sizes before allocating rows and checks the complete read order.
    """
    if not isinstance(values, (list, tuple)) or not 5 <= len(values) <= MAX_RUNTIME_INPUTS:
        raise ValueError("Runtime inputs must be a bounded numeric sequence")
    stream = [finite_real(value, f"inputs[{index}]") for index, value in enumerate(values)]
    header = []
    for value in stream[:5]:
        if not value.is_integer() or value <= 0:
            raise ValueError("Runtime dimensions must be positive integers")
        header.append(int(value))
    nt, nq, ni, nc, ns = header
    if ni != max(1, (nt - 1).bit_length()):
        raise ValueError("Index register size is inconsistent with term count")
    if nc % 2 != 1 or ns % 2 != 0:
        raise ValueError("Cosine and sine phase counts have incorrect parity")
    if len(stream) != expected_input_count(nt, nq, nc, ns):
        raise ValueError("Runtime stream length does not match its header")
    offset = 5
    features = stream[offset:offset + nq]
    if any(value < 0 or value > 1 for value in features):
        raise ValueError("Encoded features must lie in [0, 1]")
    offset += nq
    coefficients, rows, seen = [], [], set()
    for _ in range(nt):
        coefficients.append(stream[offset])
        row = stream[offset + 1:offset + 1 + nq]
        if any(not code.is_integer() or code < 0 or code > 3 for code in row):
            raise ValueError("Runtime Pauli codes must be integers in [0, 3]")
        row = tuple(int(code) for code in row)
        if row in seen:
            raise ValueError("Duplicate canonical Pauli row in runtime stream")
        seen.add(row)
        rows.append(list(row))
        offset += 1 + nq
    try:
        lam = math.fsum(abs(value) for value in coefficients)
    except OverflowError as exc:
        raise ValueError("Runtime Hamiltonian norm is nonfinite") from exc
    if not math.isfinite(lam) or lam <= 0:
        raise ValueError("Runtime Hamiltonian norm must be positive and finite")
    cosine, sine = stream[offset:offset + nc], stream[offset + nc:offset + nc + ns]
    angle = stream[-1]
    if not 0 <= angle <= math.pi:
        raise ValueError("Branch rotation must lie in [0, pi]")
    return {"num_terms": nt, "num_system_qubits": nq, "num_index_qubits": ni,
            "num_cosine_phases": nc, "num_sine_phases": ns,
            "feature_vector": features, "coefficients": coefficients,
            "pauli_rows": rows, "cosine_phases": cosine, "sine_phases": sine,
            "branch_rotation_angle": angle, "lambda_value": lam}


def build_runtime_input_document(payload, qsvt_settings, validator=None) -> dict[str, Any]:
    values = build_runtime_input_values(payload, qsvt_settings, validator)
    decoded = decode_runtime_input_values(values)
    document = {"protocol_version": PROTOCOL_VERSION, "algorithm": ALGORITHM,
                "dataset": payload.get("dataset"), "sample_index": payload.get("sample_index"),
                "layout": list(LAYOUT), "input_count": len(values),
                "inputs_sha256": inputs_sha256(values), "inputs": values,
                "num_total_qubits": decoded["num_system_qubits"] + decoded["num_index_qubits"] + 2}
    canonical_json(document)
    return document


def validate_runtime_input_document(document):
    if (not isinstance(document, dict) or type(document.get("protocol_version")) is not int
            or document["protocol_version"] != PROTOCOL_VERSION):
        raise ValueError("Unsupported runtime document protocol")
    if document.get("algorithm") != ALGORITHM or document.get("layout") != LAYOUT:
        raise ValueError("Runtime document algorithm or layout mismatch")
    decoded = decode_runtime_input_values(document.get("inputs"))
    if (type(document.get("input_count")) is not int
            or document["input_count"] != len(document["inputs"])):
        raise ValueError("Runtime document count mismatch")
    if document.get("inputs_sha256") != inputs_sha256(document["inputs"]):
        raise ValueError("Runtime document checksum mismatch")
    if (type(document.get("num_total_qubits")) is not int
            or document["num_total_qubits"] != decoded["num_system_qubits"] + decoded["num_index_qubits"] + 2):
        raise ValueError("Runtime document qubit metadata mismatch")
    return decoded


def atomic_write_json(path: Path, document) -> None:
    """Replace a JSON file only after a complete, strict write has succeeded."""
    path = Path(path)
    serialized = canonical_json(document) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_runtime_input_payload(path: Path, payload, qsvt_settings, validator=None):
    document = build_runtime_input_document(payload, qsvt_settings, validator)
    atomic_write_json(path, document)
    return document
