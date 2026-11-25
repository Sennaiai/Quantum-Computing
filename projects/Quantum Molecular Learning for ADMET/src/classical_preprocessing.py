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
import math
import re
import numbers


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
    # Historical API name: this bounds residual imaginary coefficients only.
    # It must never prune coefficients before the nonlinear AGP norm ratio.
    zero_tolerance: float = 1e-12
    max_commutator_products: int = 1000000

    def validate(self) -> None:
        for name in ("interpolation", "lambda_dot", "zero_tolerance"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite real number")
        if not 0.0 <= self.interpolation <= 1.0:
            raise ValueError("interpolation must lie in [0, 1]")
        if not math.isfinite(self.lambda_dot):
            raise ValueError("lambda_dot must be finite")
        if self.interaction_term_limit is not None and (
            isinstance(self.interaction_term_limit, bool)
            or not isinstance(self.interaction_term_limit, int)
            or self.interaction_term_limit < 1
        ):
            raise ValueError("interaction_term_limit must be a positive integer or None")
        if not math.isfinite(self.zero_tolerance) or self.zero_tolerance <= 0:
            raise ValueError("zero_tolerance must be finite and positive")
        if (isinstance(self.max_commutator_products, bool)
                or not isinstance(self.max_commutator_products, int) or self.max_commutator_products < 1):
            raise ValueError("max_commutator_products must be a positive integer")


def parse_pauli_string(pauli_string: str) -> dict[int, str]:
    if not isinstance(pauli_string, str):
        raise TypeError("Pauli word must be a string")
    result: dict[int, str] = {}
    seen: set[int] = set()
    for token in pauli_string.split():
        if re.fullmatch(r"[IXYZ](0|[1-9][0-9]*)", token) is None:
            raise ValueError(f"Invalid Pauli token {token!r}")
        qubit = int(token[1:])
        if qubit in seen:
            raise ValueError(f"Duplicate qubit {qubit} in {pauli_string!r}")
        seen.add(qubit)
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


def merge_terms(terms: Iterable[Term], tolerance: float = 0.0) -> list[Term]:
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("merge tolerance must be finite and nonnegative")
    merged: defaultdict[str, complex] = defaultdict(complex)
    for term in terms:
        canonical = format_pauli_word(parse_pauli_string(term["pauli_string"]))
        value = term["coefficient"]
        if isinstance(value, bool) or not isinstance(value, numbers.Number):
            raise ValueError("Pauli coefficients must be numeric scalars, not strings or Booleans")
        coefficient = complex(value)
        if not math.isfinite(coefficient.real) or not math.isfinite(coefficient.imag):
            raise ValueError("Pauli coefficients must be finite")
        merged[canonical] += coefficient
        if not math.isfinite(merged[canonical].real) or not math.isfinite(merged[canonical].imag):
            raise ValueError("Pauli coefficient accumulation overflowed")
    return [
        _term(coefficient, parse_pauli_string(pauli_string))
        for pauli_string, coefficient in sorted(merged.items())
        if abs(coefficient) > tolerance
    ]


def scale_terms(terms: Iterable[Term], scalar: complex) -> list[Term]:
    if isinstance(scalar, bool) or not isinstance(scalar, numbers.Number):
        raise ValueError("Term scale must be a numeric scalar")
    scalar = complex(scalar)
    if not math.isfinite(scalar.real) or not math.isfinite(scalar.imag):
        raise ValueError("Term scale must be finite")
    result = []
    for term in terms:
        value = term["coefficient"]
        if isinstance(value, bool) or not isinstance(value, numbers.Number):
            raise ValueError("Pauli coefficients must be numeric scalars")
        coefficient = complex(value)
        product = coefficient*scalar
        if not all(math.isfinite(x) for x in (coefficient.real, coefficient.imag, product.real, product.imag)):
            raise ValueError("Pauli scaling overflowed or contains nonfinite values")
        if coefficient != 0 and scalar != 0 and product == 0:
            raise ValueError("Nonzero Pauli scaling underflowed float64 precision")
        result.append(dict(term, coefficient=product))
    return result


def commutator(left_terms: Iterable[Term], right_terms: Iterable[Term], tolerance: float = 0.0,
               *, max_term_products: int = 1000000) -> list[Term]:
    if isinstance(max_term_products, bool) or not isinstance(max_term_products, int) or max_term_products < 1:
        raise ValueError("max_term_products must be a positive integer")
    output: list[Term] = []
    left_terms = merge_terms(left_terms, tolerance)
    right_terms = merge_terms(right_terms, tolerance)
    if len(left_terms)*len(right_terms) > max_term_products:
        raise ValueError("Commutator product budget exceeded before expansion; no terms were truncated")
    for left in left_terms:
        left_word = parse_pauli_string(left["pauli_string"])
        for right in right_terms:
            right_word = parse_pauli_string(right["pauli_string"])
            phase_lr, word_lr = multiply_pauli_words(left_word, right_word)
            phase_rl, _ = multiply_pauli_words(right_word, left_word)
            phase_difference = phase_lr-phase_rl
            if phase_difference == 0:
                continue
            coefficient = complex(left["coefficient"]) * complex(right["coefficient"]) * phase_difference
            if not math.isfinite(coefficient.real) or not math.isfinite(coefficient.imag):
                raise ValueError("Noncommuting Pauli product overflowed float64 precision")
            if coefficient == 0:
                raise ValueError("Noncommuting Pauli product underflowed: insufficient float64 precision")
            if abs(coefficient) > tolerance:
                output.append(_term(coefficient, word_lr))
    return merge_terms(output, tolerance)


def build_problem_hamiltonian(
    curated: dict[str, Any], dataset_key: str, feature_vector: Iterable[float], interaction_term_limit: int | None,
    *, hermiticity_tolerance: float = 1e-12,
) -> dict[str, Any]:
    """Build H_problem(theta)=H_A+H_B(theta) following ADMET Stage 6."""
    interaction = curated["hamiltonian_A_interaction_ising"][dataset_key]
    selected_features = list(interaction["selected_features"])
    raw_theta = list(feature_vector)
    if any(isinstance(x, bool) or not isinstance(x, numbers.Real) for x in raw_theta):
        raise ValueError("Feature values must be real numeric scalars, not strings or Booleans")
    theta = [float(value) for value in raw_theta]
    if not selected_features or len(set(selected_features)) != len(selected_features):
        raise ValueError("selected features must be nonempty and unique")
    if len(theta) != len(selected_features):
        raise ValueError("feature vector and selected-feature dimensions differ")
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in theta):
        raise ValueError("feature values must be finite and lie in [0, 1]")
    if interaction_term_limit is not None:
        AGPSettings(interaction_term_limit=interaction_term_limit).validate()

    interaction_terms = deepcopy(interaction["pauli_terms"])
    for term in interaction_terms:
        parse_pauli_string(term["pauli_string"])
        if any(int(token[1:]) >= len(selected_features) for token in term["pauli_string"].split()):
            raise ValueError("Interaction Pauli wire exceeds selected-feature dimension")
    dropped = []
    if interaction_term_limit is not None:
        dropped = interaction_terms[interaction_term_limit:]
        interaction_terms = interaction_terms[:interaction_term_limit]
    local_terms = [
        {"coefficient": value, "pauli_string": f"Z{qubit}", "locality": 1, "parameter_index": qubit}
        for qubit, value in enumerate(theta)
    ]
    terms = _real_terms(interaction_terms + local_terms, hermiticity_tolerance)
    for term in terms:
        if any(q >= len(selected_features) for q in parse_pauli_string(term["pauli_string"])):
            raise ValueError("Interaction Pauli wire exceeds selected-feature dimension")
    return {
        "hamiltonian_type": "problem_hamiltonian",
        "formula": "H_problem(theta) = H_A + H_B(theta)",
        "selected_features": selected_features,
        "feature_vector": theta,
        "pauli_terms": terms,
        "interaction_truncation": {
            "dropped_count": len(dropped),
            "dropped_l1_operator_bound": sum(abs(complex(t["coefficient"])) for t in dropped),
            "scope": "omitted H_A interactions only; not an AGP or H_CD error bound",
        },
    }


def build_initial_hamiltonian(num_qubits: int) -> list[Term]:
    """Transverse-field driver used by the counterdiabatic notebook."""
    if isinstance(num_qubits, bool) or not isinstance(num_qubits, int) or num_qubits < 1:
        raise ValueError("num_qubits must be a positive integer")
    return [{"coefficient": -1.0, "pauli_string": f"X{q}", "locality": 1} for q in range(num_qubits)]


def _real_terms(terms: Iterable[Term], tolerance: float) -> list[Term]:
    """Retain every nonzero real coefficient; bound imaginary residuals only."""
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Hermiticity residual tolerance must be finite and positive")
    result: list[Term] = []
    for term in merge_terms(terms, 0.0):
        coefficient = complex(term["coefficient"])
        if abs(coefficient.imag) > tolerance:
            raise ValueError(f"Non-Hermitian coefficient {coefficient} for {term['pauli_string']}")
        if coefficient.real != 0.0:
            result.append(dict(term, coefficient=float(coefficient.real)))
    return result


def recalculate_counterdiabatic_hamiltonian(
    curated: dict[str, Any], dataset_key: str, feature_vector: Iterable[float], settings: AGPSettings = AGPSettings(),
    *, include_components: bool = False,
) -> dict[str, Any]:
    """Recalculate O0/O1/O2, alpha1, AGP, and H_CD for one molecule."""
    settings.validate()
    problem = build_problem_hamiltonian(
        curated, dataset_key, feature_vector, settings.interaction_term_limit,
        hermiticity_tolerance=settings.zero_tolerance,
    )
    num_qubits = len(problem["selected_features"])
    initial_terms = build_initial_hamiltonian(num_qubits)
    problem_terms = problem["pauli_terms"]

    interpolation = settings.interpolation
    had_terms = merge_terms(
        scale_terms(initial_terms, 1.0 - interpolation)
        + scale_terms(problem_terms, interpolation),
        0.0,
    )
    o0_terms = merge_terms(
        scale_terms(problem_terms, 1.0) + scale_terms(initial_terms, -1.0),
        0.0,
    )
    o1_terms = commutator(had_terms, o0_terms, 0.0,
                          max_term_products=settings.max_commutator_products)
    o2_terms = commutator(had_terms, o1_terms, 0.0,
                          max_term_products=settings.max_commutator_products)
    try:
        o1_norm_sq = math.fsum(abs(complex(term["coefficient"])) ** 2 for term in o1_terms)
        o2_norm_sq = math.fsum(abs(complex(term["coefficient"])) ** 2 for term in o2_terms)
    except OverflowError as exc:
        raise ValueError("Nested-commutator norms overflowed float64 precision") from exc
    if not math.isfinite(o1_norm_sq) or not math.isfinite(o2_norm_sq):
        raise ValueError("Nested-commutator norms overflowed")
    if not o1_terms:
        # Exact algebraic cancellation, not a coefficient-size threshold,
        # defines a commuting interpolation and its zero-gauge choice.
        alpha1 = 0.0
    elif o1_norm_sq == 0.0 or o2_norm_sq == 0.0:
        raise ValueError("Nonzero O1 with an underflowed/zero norm: insufficient float64 precision for AGP")
    else:
        alpha1 = -(o1_norm_sq / o2_norm_sq)
    if not math.isfinite(alpha1):
        raise ValueError("First-order AGP coefficient is not finite")

    agp_terms = merge_terms(scale_terms(o1_terms, 1j * alpha1), 0.0)
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
            "ansatz_order": 1,
            "norm_convention": "normalized Hilbert-Schmidt: Tr(O_dagger O)/2**n",
            "approximation": "variational first-order nested-commutator AGP, not an exact AGP in general",
            "hermiticity_imaginary_tolerance": settings.zero_tolerance,
            "coefficient_pruning": "none; exact zero cancellations only",
            "max_commutator_products": settings.max_commutator_products,
            "interaction_term_limit": settings.interaction_term_limit,
            "commuting_path": not o1_terms,
            "interaction_truncation": problem["interaction_truncation"],
        },
    }
    validate_counterdiabatic_payload(payload)
    if include_components:
        # Complex O1 coefficients are intentionally retained for scientific
        # inspection. Production JSON payloads use the default False.
        payload["agp_components"] = {
            "initial": initial_terms, "problem": problem_terms,
            "adiabatic": had_terms, "o0": o0_terms,
            "o1": o1_terms, "o2": o2_terms, "agp": agp_terms,
        }
    return payload


def validate_counterdiabatic_payload(payload: dict[str, Any]) -> None:
    if not payload["pauli_terms"]:
        raise ValueError("Counterdiabatic Hamiltonian is empty")
    num_qubits = len(payload["selected_features"])
    if num_qubits < 1 or len(set(payload["selected_features"])) != num_qubits:
        raise ValueError("Selected features must be nonempty and unique")
    if len(payload["feature_vector"]) != num_qubits:
        raise ValueError("Feature-vector dimension mismatch")
    if any(not math.isfinite(float(x)) or not 0 <= float(x) <= 1 for x in payload["feature_vector"]):
        raise ValueError("Feature vector must be finite and lie in [0, 1]")
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
        if not math.isfinite(term["coefficient"]):
            raise ValueError("Final H_CD coefficients must be finite")
        if term.get("locality", len(parse_pauli_string(pauli_string))) != len(parse_pauli_string(pauli_string)):
            raise ValueError("Pauli locality is inconsistent")
