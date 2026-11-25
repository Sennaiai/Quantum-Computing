"""Synthetic public fixtures; no ADMET records or serialized private data."""
from pathlib import Path
import math
import sys

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "python_classes"):
    sys.path.insert(0, str(directory))


def payload():
    return {"dataset": "synthetic", "sample_index": 0, "selected_features": [4, 8],
            "feature_vector": [0.0, 1.0], "hamiltonian_type": "synthetic_reference",
            "pauli_terms": [{"coefficient": -0.25, "pauli_string": ""},
                            {"coefficient": 0.5, "pauli_string": "X1"},
                            {"coefficient": 0.75, "pauli_string": "Y0 Z1"},
                            {"coefficient": -0.125, "pauli_string": "Z0 X1"}]}


def settings():
    sc, ss = 0.8, 0.6
    beta = 1 / (1 / sc + 1 / ss)
    return {"algorithm": "heralded_cosine_sine_lcu_v2",
            "phase_convention": "projector_exp_i_phi_R", "lambda_value": 1.625,
            "physical_time": 0.5, "tau": 0.8125, "epsilon": 0.001,
            "cosine": {"phase_angles": [0.1, 0.2, 0.3], "polynomial_degree": 2,
                       "scale": sc, "approximation_error_bound": 0.00001,
                       "synthesis_error_bound": 0.00001},
            "sine": {"phase_angles": [0.4, 0.5, 0.6, 0.7], "polynomial_degree": 3,
                     "scale": ss, "approximation_error_bound": 0.00001,
                     "synthesis_error_bound": 0.00001},
            "success_amplitude": beta,
            "recovered_evolution_error_bound": 0.00004,
            "ideal_success_probability": beta * beta,
            "success_probability_lower_bound": beta * beta * (1 - 0.00004) ** 2,
            "branch_rotation_angle": 2 * math.acos(math.sqrt(beta / sc)),
            "extra_ancillas": 2, "native_validation": {"status": "unverified", "compiler_verified": False}}
