"""Serialize molecule-specific ADMET/QSVT data for Beryllium ``input`` reads."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


PAULI_CODE = {"I": 0, "X": 1, "Y": 2, "Z": 3}
PROTOCOL_VERSION = 1


def dense_pauli_rows(payload: dict[str, Any], num_qubits: int) -> list[list[int]]:
    rows = []
    for term in payload["pauli_terms"]:
        row = [0] * num_qubits
        for token in term["pauli_string"].split():
            letter, qubit = token[0], int(token[1:])
            if letter not in PAULI_CODE or not 0 <= qubit < num_qubits:
                raise ValueError(f"Invalid runtime Pauli token {token!r}")
            row[qubit] = PAULI_CODE[letter]
        rows.append(row)
    return rows


def build_runtime_input_values(payload, qsvt_settings, validator) -> list[float]:
    """Pack values in exactly the order consumed by immutable ``main.qbe``."""
    phases = [float(value) for value in qsvt_settings["phase_angles"]]
    values: list[float] = [
        float(validator.num_terms),
        float(validator.num_system_qubits),
        float(validator.num_index_qubits),
        float(len(phases)),
    ]
    values.extend(float(value) for value in payload["feature_vector"])
    for term, row in zip(
        payload["pauli_terms"], dense_pauli_rows(payload, validator.num_system_qubits)
    ):
        values.append(float(term["coefficient"]))
        values.extend(float(code) for code in row)
    values.extend(phases)
    return values


def expected_input_count(num_terms: int, num_qubits: int, num_phases: int) -> int:
    return 4 + num_qubits + num_terms * (1 + num_qubits) + num_phases


def build_runtime_input_document(payload, qsvt_settings, validator) -> dict[str, Any]:
    """Build the API input document in memory; no circuit source is generated."""
    values = build_runtime_input_values(payload, qsvt_settings, validator)
    expected = expected_input_count(
        validator.num_terms, validator.num_system_qubits, len(qsvt_settings["phase_angles"])
    )
    if len(values) != expected:
        raise ValueError(f"Runtime input length {len(values)} != expected {expected}")
    document = {
        "protocol_version": PROTOCOL_VERSION,
        "dataset": payload.get("dataset"),
        "sample_index": payload.get("sample_index"),
        "layout": [
            "num_terms", "num_system_qubits", "num_index_qubits", "num_qsvt_phases",
            "initial_angles[num_system_qubits]",
            "for each term: signed_coefficient, pauli_codes[num_system_qubits]",
            "qsvt_phases[num_qsvt_phases]",
        ],
        "input_count": len(values),
        "inputs_sha256": hashlib.sha256(json.dumps(values, separators=(",", ":")).encode()).hexdigest(),
        "inputs": values,
    }
    return document


def write_runtime_input_payload(path: Path, payload, qsvt_settings, validator) -> dict[str, Any]:
    """Optionally persist a runtime-input document for inspection."""
    document = build_runtime_input_document(payload, qsvt_settings, validator)
    path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    return document
