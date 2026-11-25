"""Dependency-free validation of real, finite Pauli Hamiltonians.

Pauli indices are zero based. An empty string denotes the identity; explicit
identity factors are accepted and removed from the canonical representation.
"""

from __future__ import annotations

import math
import numbers
import re
from collections.abc import Mapping

PAULI_CODE = {"I": 0, "X": 1, "Y": 2, "Z": 3}
_TOKEN = re.compile(r"([IXYZ])(0|[1-9][0-9]*)\Z")


def finite_real(value, name="value") -> float:
    """Convert a real scalar without silently accepting strings or Booleans."""
    if isinstance(value, bool) or not isinstance(value, numbers.Number):
        raise ValueError(f"{name} must be a real numeric scalar")
    if isinstance(value, numbers.Complex) and not isinstance(value, numbers.Real):
        if not math.isfinite(float(value.imag)) or abs(value.imag) > 1e-12:
            raise ValueError(f"{name} must be real")
        value = value.real
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite and real") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def positive_integer(value, name="value", *, allow_zero=False) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < (0 if allow_zero else 1):
        raise ValueError(f"{name} is outside its permitted range")
    return result


def parse_pauli_string(pauli_string, num_qubits: int) -> tuple[int, ...]:
    """Return a dense canonical row, rejecting malformed or repeated wires."""
    num_qubits = positive_integer(num_qubits, "num_qubits")
    if not isinstance(pauli_string, str):
        raise ValueError("pauli_string must be a string")
    row = [0] * num_qubits
    used = set()
    for token in pauli_string.split():
        match = _TOKEN.fullmatch(token)
        if match is None:
            raise ValueError(f"Invalid Pauli token {token!r}")
        letter, index = match.groups()
        wire = int(index)
        if wire >= num_qubits:
            raise ValueError(f"Pauli wire {wire} is outside the system register")
        if wire in used:
            raise ValueError(f"Repeated Pauli wire {wire}")
        used.add(wire)
        row[wire] = PAULI_CODE[letter]
    return tuple(row)


def canonical_pauli_string(row) -> str:
    letters = "IXYZ"
    return " ".join(f"{letters[code]}{wire}" for wire, code in enumerate(row) if code)


def validate_feature_vector(values, num_qubits: int) -> list[float]:
    if isinstance(values, (str, bytes)):
        raise ValueError("feature_vector must be a numeric sequence")
    try:
        values = list(values)
    except TypeError as exc:
        raise ValueError("feature_vector must be a numeric sequence") from exc
    if len(values) != num_qubits:
        raise ValueError("feature_vector length must equal the system-register size")
    result = [finite_real(value, f"feature_vector[{index}]") for index, value in enumerate(values)]
    if any(value < 0.0 or value > 1.0 for value in result):
        raise ValueError("feature_vector values must lie in [0, 1] for RY(pi*x) encoding")
    return result


class HamiltonianValidation:
    """Validate a Hermitian Pauli sum and derive its LCU metadata.

    A feature vector is optional for Hamiltonian-only reference calculations.
    The runtime protocol requires it before producing an executable stream.
    """

    def __init__(self, hamiltonian):
        if not isinstance(hamiltonian, Mapping):
            raise ValueError("Hamiltonian must be a mapping")
        self.hamiltonian = hamiltonian
        self.name = hamiltonian.get("hamiltonian_type", "counterdiabatic")
        features = hamiltonian.get("selected_features")
        if isinstance(features, (str, bytes)) or features is None:
            raise ValueError("selected_features must be a nonempty sequence")
        try:
            self.selected_features = list(features)
            if any(isinstance(value, bool) or not (
                isinstance(value, numbers.Integral) and value >= 0
                or isinstance(value, str) and bool(value.strip())
            ) for value in self.selected_features):
                raise ValueError("selected_features require nonnegative integer or nonempty string identifiers")
            if len(set(self.selected_features)) != len(self.selected_features):
                raise ValueError("selected_features must be unique")
            self.selected_features = [int(value) if isinstance(value, numbers.Integral) else value
                                      for value in self.selected_features]
        except TypeError as exc:
            raise ValueError("selected_features require integer or string identifiers") from exc
        self.num_system_qubits = positive_integer(len(self.selected_features), "num_system_qubits")
        terms = hamiltonian.get("pauli_terms")
        if isinstance(terms, (str, bytes)) or terms is None:
            raise ValueError("pauli_terms must be a nonempty sequence")
        try:
            source_terms = list(terms)
        except TypeError as exc:
            raise ValueError("pauli_terms must be a nonempty sequence") from exc
        self.num_terms = positive_integer(len(source_terms), "num_terms")
        self.coefficients, self.pauli_strings, self.pauli_rows, self.pauli_terms = [], [], [], []
        seen = set()
        for index, term in enumerate(source_terms):
            if not isinstance(term, Mapping) or "coefficient" not in term or "pauli_string" not in term:
                raise ValueError(f"pauli_terms[{index}] requires coefficient and pauli_string")
            coefficient = finite_real(term["coefficient"], f"pauli_terms[{index}].coefficient")
            row = parse_pauli_string(term["pauli_string"], self.num_system_qubits)
            if row in seen:
                raise ValueError(f"Duplicate canonical Pauli term at index {index}")
            seen.add(row)
            word = canonical_pauli_string(row)
            self.coefficients.append(coefficient)
            self.pauli_strings.append(word)
            self.pauli_rows.append(row)
            self.pauli_terms.append({"coefficient": coefficient, "pauli_string": word})
        try:
            self.lambda_value = math.fsum(abs(value) for value in self.coefficients)
        except OverflowError as exc:
            raise ValueError("Hamiltonian L1 norm must be finite") from exc
        if not math.isfinite(self.lambda_value) or self.lambda_value <= 0:
            raise ValueError("Hamiltonian L1 norm must be positive and finite")
        self.num_index_qubits = max(1, (self.num_terms - 1).bit_length())
        self.weights = [abs(value) / self.lambda_value for value in self.coefficients]
        self.feature_vector = None
        if "feature_vector" in hamiltonian:
            self.feature_vector = validate_feature_vector(hamiltonian["feature_vector"], self.num_system_qubits)
        self.validate()

    def validate(self):
        if not math.isclose(math.fsum(self.weights), 1.0, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("LCU weights do not sum to one")
        if 2 ** self.num_index_qubits < self.num_terms:
            raise ValueError("Index register is too small")
        return True

    def report(self):
        return {"name": self.name, "num_terms": self.num_terms,
                "num_system_qubits": self.num_system_qubits,
                "num_index_qubits": self.num_index_qubits,
                "lambda_value": self.lambda_value,
                "weights_sum": math.fsum(self.weights), "weights": list(self.weights)}
