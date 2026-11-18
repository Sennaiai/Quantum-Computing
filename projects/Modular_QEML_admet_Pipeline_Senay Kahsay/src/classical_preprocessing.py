"""Submission-grade classical ADMET and counterdiabatic preprocessing.

The stages mirror ``pipeline/ADMET_future_curation_complete.ipynb`` and
``pipeline/Counter-diabatic hamiltonian_complete.ipynb`` while exposing a
bounded, reproducible per-dataset/per-molecule execution path.
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Iterable


Term = dict[str, Any]
_MUL = {
    ("I", "I"): (1, "I"), ("I", "X"): (1, "X"), ("I", "Y"): (1, "Y"), ("I", "Z"): (1, "Z"),
    ("X", "I"): (1, "X"), ("X", "X"): (1, "I"), ("X", "Y"): (1j, "Z"), ("X", "Z"): (-1j, "Y"),
    ("Y", "I"): (1, "Y"), ("Y", "X"): (-1j, "Z"), ("Y", "Y"): (1, "I"), ("Y", "Z"): (1j, "X"),
    ("Z", "I"): (1, "Z"), ("Z", "X"): (1j, "Y"), ("Z", "Y"): (-1j, "X"), ("Z", "Z"): (1, "I"),
}


@dataclass(frozen=True)
class AGPSettings:
    interpolation: float = 0.5
    lambda_dot: float = 1.0
    interaction_term_limit: int | None = None
    zero_tolerance: float = 1e-12

    def validate(self) -> None:
        if not 0.0 <= self.interpolation <= 1.0:
            raise ValueError("interpolation must lie in [0, 1]")
        if self.interaction_term_limit is not None and self.interaction_term_limit < 1:
            raise ValueError("interaction_term_limit must be positive or None")
        if self.zero_tolerance <= 0:
            raise ValueError("zero_tolerance must be positive")


def parse_pauli_string(pauli_string: str) -> dict[int, str]:
    result: dict[int, str] = {}
    for token in pauli_string.split():
        if len(token) < 2 or token[0] not in "IXYZ":
            raise ValueError(f"Invalid Pauli token {token!r}")
        qubit = int(token[1:])
        if qubit in result:
            raise ValueError(f"Duplicate qubit {qubit} in {pauli_string!r}")
        if token[0] != "I":
            result[qubit] = token[0]
    return result


def format_pauli_word(word: dict[int, str]) -> str:
    return " ".join(f"{word[q]}{q}" for q in sorted(word))


def multiply_pauli_words(left: dict[int, str], right: dict[int, str]) -> tuple[complex, dict[int, str]]:
    phase = 1 + 0j
    product: dict[int, str] = {}
    for qubit in sorted(set(left) | set(right)):
        local_phase, operator = _MUL[(left.get(qubit, "I"), right.get(qubit, "I"))]
        phase *= local_phase
        if operator != "I":
            product[qubit] = operator
    return phase, product


def _term(coefficient: complex, word: dict[int, str]) -> Term:
    return {
        "coefficient": coefficient,
        "pauli_string": format_pauli_word(word),
        "locality": len(word),
    }


def merge_terms(terms: Iterable[Term], tolerance: float = 1e-12) -> list[Term]:
    merged: defaultdict[str, complex] = defaultdict(complex)
    for term in terms:
        canonical = format_pauli_word(parse_pauli_string(term["pauli_string"]))
        merged[canonical] += complex(term["coefficient"])
    return [
        _term(coefficient, parse_pauli_string(pauli_string))
        for pauli_string, coefficient in sorted(merged.items())
        if abs(coefficient) > tolerance
    ]


def scale_terms(terms: Iterable[Term], scalar: complex) -> list[Term]:
    return [dict(term, coefficient=complex(term["coefficient"]) * scalar) for term in terms]


def commutator(left_terms: Iterable[Term], right_terms: Iterable[Term], tolerance: float = 1e-12) -> list[Term]:
    output: list[Term] = []
    for left in left_terms:
        left_word = parse_pauli_string(left["pauli_string"])
        for right in right_terms:
            right_word = parse_pauli_string(right["pauli_string"])
            phase_lr, word_lr = multiply_pauli_words(left_word, right_word)
            phase_rl, _ = multiply_pauli_words(right_word, left_word)
            coefficient = complex(left["coefficient"]) * complex(right["coefficient"]) * (phase_lr - phase_rl)
            if abs(coefficient) > tolerance:
                output.append(_term(coefficient, word_lr))
    return merge_terms(output, tolerance)


def build_problem_hamiltonian(
    curated: dict[str, Any], dataset_key: str, feature_vector: Iterable[float], interaction_term_limit: int | None
) -> dict[str, Any]:
    """Build H_problem(theta)=H_A+H_B(theta) following ADMET Stage 6."""
    interaction = curated["hamiltonian_A_interaction_ising"][dataset_key]
    selected_features = list(interaction["selected_features"])
    theta = [float(value) for value in feature_vector]
    if len(theta) != len(selected_features):
        raise ValueError("feature vector and selected-feature dimensions differ")

    interaction_terms = deepcopy(interaction["pauli_terms"])
    if interaction_term_limit is not None:
        interaction_terms = interaction_terms[:interaction_term_limit]
    local_terms = [
        {"coefficient": value, "pauli_string": f"Z{qubit}", "locality": 1, "parameter_index": qubit}
        for qubit, value in enumerate(theta)
    ]
    terms = merge_terms(interaction_terms + local_terms)
    return {
        "hamiltonian_type": "problem_hamiltonian",
        "formula": "H_problem(theta) = H_A + H_B(theta)",
        "selected_features": selected_features,
        "feature_vector": theta,
        "pauli_terms": terms,
    }


def build_initial_hamiltonian(num_qubits: int) -> list[Term]:
    """Transverse-field driver used by the counterdiabatic notebook."""
    return [{"coefficient": -1.0, "pauli_string": f"X{q}", "locality": 1} for q in range(num_qubits)]


def _real_terms(terms: Iterable[Term], tolerance: float) -> list[Term]:
    result: list[Term] = []
    for term in merge_terms(terms, tolerance):
        coefficient = complex(term["coefficient"])
        if abs(coefficient.imag) > tolerance:
            raise ValueError(f"Non-Hermitian coefficient {coefficient} for {term['pauli_string']}")
        result.append(dict(term, coefficient=float(coefficient.real)))
    return result


def recalculate_counterdiabatic_hamiltonian(
    curated: dict[str, Any], dataset_key: str, feature_vector: Iterable[float], settings: AGPSettings = AGPSettings()
) -> dict[str, Any]:
    """Recalculate O0/O1/O2, alpha1, AGP, and H_CD for one molecule."""
    settings.validate()
    problem = build_problem_hamiltonian(
        curated, dataset_key, feature_vector, settings.interaction_term_limit
    )
    num_qubits = len(problem["selected_features"])
    initial_terms = build_initial_hamiltonian(num_qubits)
    problem_terms = problem["pauli_terms"]

    interpolation = settings.interpolation
    had_terms = merge_terms(
        scale_terms(initial_terms, 1.0 - interpolation)
        + scale_terms(problem_terms, interpolation),
        settings.zero_tolerance,
    )
    o0_terms = merge_terms(
        scale_terms(problem_terms, 1.0) + scale_terms(initial_terms, -1.0),
        settings.zero_tolerance,
    )
    o1_terms = commutator(had_terms, o0_terms, settings.zero_tolerance)
    o2_terms = commutator(had_terms, o1_terms, settings.zero_tolerance)
    o1_norm_sq = sum(abs(complex(term["coefficient"])) ** 2 for term in o1_terms)
    o2_norm_sq = sum(abs(complex(term["coefficient"])) ** 2 for term in o2_terms)
    if o2_norm_sq <= settings.zero_tolerance:
        raise ValueError("Cannot compute alpha1: ||O2||^2 is zero")
    alpha1 = -(o1_norm_sq / o2_norm_sq)

    agp_terms = merge_terms(scale_terms(o1_terms, 1j * alpha1), settings.zero_tolerance)
    cd_terms = _real_terms(
        had_terms + scale_terms(agp_terms, settings.lambda_dot), settings.zero_tolerance
    )
    payload = {
        "dataset": dataset_key,
        "hamiltonian_type": "counterdiabatic",
        "formula": "H_CD = H_ad + lambda_dot * A_lambda",
        "selected_features": problem["selected_features"],
        "feature_vector": problem["feature_vector"],
        "pauli_terms": cd_terms,
        "agp_metadata": {
            "interpolation": interpolation,
            "lambda_dot": settings.lambda_dot,
            "o0_terms": len(o0_terms),
            "o1_terms": len(o1_terms),
            "o2_terms": len(o2_terms),
            "o1_norm_sq": o1_norm_sq,
            "o2_norm_sq": o2_norm_sq,
            "alpha1": alpha1,
        },
    }
    validate_counterdiabatic_payload(payload)
    return payload


def validate_counterdiabatic_payload(payload: dict[str, Any]) -> None:
    if not payload["pauli_terms"]:
        raise ValueError("Counterdiabatic Hamiltonian is empty")
    num_qubits = len(payload["selected_features"])
    if len(payload["feature_vector"]) != num_qubits:
        raise ValueError("Feature-vector dimension mismatch")
    seen: set[str] = set()
    for term in payload["pauli_terms"]:
        pauli_string = format_pauli_word(parse_pauli_string(term["pauli_string"]))
        if pauli_string in seen:
            raise ValueError(f"Duplicate Pauli operator {pauli_string}")
        seen.add(pauli_string)
        for qubit in parse_pauli_string(pauli_string):
            if not 0 <= qubit < num_qubits:
                raise ValueError(f"Qubit {qubit} outside 0..{num_qubits - 1}")
        if not isinstance(term["coefficient"], float):
            raise TypeError("Final H_CD coefficients must be real floats")
