"""Bounded dense reference for CD/LCU/qubitization/QSVT mathematics.

This is an independently assembled NumPy/SciPy circuit model, not a Beryllium
interpreter or compiler test. Basis order: branch, real-extraction, index (MSB
first), system (wire 0 MSB). All extra and index ancillas must be zero to herald
successful evolution. Dense verification is deliberately resource bounded.
"""
from __future__ import annotations

import math
import numpy as np
from scipy.linalg import block_diag, expm
from classical_preprocessing import parse_pauli_string

PAULI = {"I": np.eye(2), "X": np.array([[0, 1], [1, 0]]),
         "Y": np.array([[0, -1j], [1j, 0]]), "Z": np.diag([1, -1])}
HADAMARD = np.array([[1, 1], [1, -1]]) / np.sqrt(2)


def pauli_matrix(pauli_string, num_qubits):
    word = parse_pauli_string(pauli_string)
    if any(q >= num_qubits for q in word):
        raise ValueError("Pauli wire exceeds system dimension")
    result = np.ones((1, 1), dtype=complex)
    for q in range(num_qubits):
        result = np.kron(result, PAULI[word.get(q, "I")])
    return result


def hamiltonian_matrix(terms, num_qubits):
    return sum((complex(t["coefficient"])*pauli_matrix(t["pauli_string"], num_qubits)
                for t in terms), np.zeros((2**num_qubits, 2**num_qubits), complex))


def _rotation(theta):
    return np.array([[np.cos(theta/2), -np.sin(theta/2)],
                     [np.sin(theta/2), np.cos(theta/2)]])


def prepare_matrix(amplitudes, num_index_qubits):
    """Multiplexed Ry tree, assembled gate by gate (not a Householder oracle)."""
    amplitudes = np.asarray(amplitudes, dtype=float)
    dimension = 2**num_index_qubits
    if (dimension < len(amplitudes) or np.any(amplitudes < 0)
            or not np.all(np.isfinite(amplitudes))
            or not np.isclose(amplitudes@amplitudes, 1, rtol=1e-12, atol=1e-14)):
        raise ValueError("PREPARE amplitudes must be finite, nonnegative and normalized")
    padded = np.pad(amplitudes, (0, dimension-len(amplitudes)))
    result = np.eye(dimension)
    for level in range(num_index_qubits):
        remaining = num_index_qubits-level-1
        for prefix in range(2**level):
            offset = prefix * 2**(remaining+1)
            half = 2**remaining
            left = float(np.sum(padded[offset:offset+half]**2))
            right = float(np.sum(padded[offset+half:offset+2*half]**2))
            theta = 0 if left+right == 0 else 2*np.arctan2(np.sqrt(right), np.sqrt(left))
            rotation = _rotation(theta)
            gate = np.eye(dimension)
            for suffix in range(half):
                rows = [offset+suffix, offset+half+suffix]
                gate[np.ix_(rows, rows)] = rotation
            result = gate @ result
    return result


def select_matrix(terms, num_qubits, num_index_qubits):
    """Signed Pauli multiplexor; unused padded index states apply identity."""
    blocks = []
    for index in range(2**num_index_qubits):
        if index < len(terms):
            term = terms[index]
            coefficient = float(term["coefficient"])
            blocks.append((-1 if coefficient < 0 else 1)*pauli_matrix(term["pauli_string"], num_qubits))
        else:
            blocks.append(np.eye(2**num_qubits))
    return block_diag(*blocks)


def lcu_operators(payload, *, max_dimension=512):
    terms = payload["pauli_terms"]
    nq = len(payload["selected_features"])
    ni = max(1, (len(terms)-1).bit_length())
    dim = 2**(nq+ni)
    if dim > max_dimension:
        raise ValueError(f"Dense reference dimension {dim} exceeds budget {max_dimension}")
    coefficients = np.asarray([float(t["coefficient"]) for t in terms])
    lam = float(np.sum(np.abs(coefficients)))
    if lam <= 0 or not np.all(np.isfinite(coefficients)):
        raise ValueError("Reference requires a finite nonzero Hermitian Hamiltonian")
    amplitudes = np.sqrt(np.abs(coefficients)/lam)
    prep = prepare_matrix(amplitudes, ni)
    full_prep = np.kron(prep, np.eye(2**nq))
    select = select_matrix(terms, nq, ni)
    block = full_prep.conj().T @ select @ full_prep
    reflection = -np.eye(dim)
    reflection[:2**nq, :2**nq] = np.eye(2**nq)
    walk = reflection @ block
    return {"prepare": prep, "select": select, "block": block,
            "reflection": reflection, "walk": walk,
            "walk_inverse": block.conj().T @ reflection,
            "lambda_value": lam, "num_index_qubits": ni,
            "num_system_qubits": nq, "amplitudes": amplitudes}


def qsvt_matrix(block, num_system_qubits, phase_angles):
    """D0 B D1 ... B Dd with d queries for d+1 phase angles."""
    sign = -np.ones(len(block))
    sign[:2**num_system_qubits] = 1
    phases = np.asarray(phase_angles, dtype=float)
    if not len(phases) or not np.all(np.isfinite(phases)):
        raise ValueError("QSVT phases must be nonempty and finite")
    result = np.diag(np.exp(1j*phases[0]*sign))
    for phi in phases[1:]:
        # Apply a diagonal on the right without a dense matrix multiply.
        result = (result @ block)*np.exp(1j*phi*sign)[None, :]
    return result


def real_extraction_unitary(unitary):
    """Coherent U/U† multiplexing, with exact phases retained."""
    hadamard = np.kron(HADAMARD, np.eye(len(unitary)))
    return hadamard @ block_diag(unitary, unitary.conj().T) @ hadamard


def heralded_evolution_unitary(operators, settings, *, max_dimension=1024):
    """Actual branch + real-extraction unitary, including both unpreparations."""
    block = operators["block"]
    if 4*len(block) > max_dimension:
        raise ValueError("Full heralded unitary exceeds dense reference budget")
    nq = operators["num_system_qubits"]
    cosine = qsvt_matrix(block, nq, settings["cosine"]["phase_angles"])
    sine = qsvt_matrix(block, nq, settings["sine"]["phase_angles"])
    cosine_real = real_extraction_unitary(cosine)
    sine_real = real_extraction_unitary(sine)
    prep = np.kron(_rotation(settings["branch_rotation_angle"]), np.eye(2*len(block)))
    return prep.conj().T @ block_diag(cosine_real, -1j*sine_real) @ prep


def verify_quantum_reference(payload, settings, *, max_dimension=256):
    """Return measured numerical errors against dense H and scipy.linalg.expm.

    max_dimension bounds the index+system matrix; the full coherent unitary
    has four times that dimension. No empirical hardware claim follows.
    """
    ops = lcu_operators(payload, max_dimension=max_dimension)
    nq = ops["num_system_qubits"]
    ds = 2**nq
    block, select, walk = (ops[k] for k in ("block", "select", "walk"))
    identity = np.eye(len(block))
    H = hamiltonian_matrix(payload["pauli_terms"], nq)
    unitary = heralded_evolution_unitary(ops, settings, max_dimension=4*max_dimension)
    success_block = unitary[:ds, :ds]
    beta = settings["success_amplitude"]
    target = expm(-1j*settings["physical_time"]*H)
    recovered_error = float(np.linalg.norm(success_block/beta-target, 2))
    prep_target = np.pad(ops["amplitudes"], (0, len(ops["prepare"])-len(ops["amplitudes"])))
    report = {
        "status": "verified_mathematical_reference",
        "native_compiler_verified": False,
        "basis_order": "branch,real,index_MSB,system_wire0_MSB",
        "prepare_amplitude_error": float(np.max(np.abs(ops["prepare"][:, 0]-prep_target))),
        "prepare_inverse_error": float(np.linalg.norm(ops["prepare"].T@ops["prepare"]-np.eye(len(ops["prepare"])), 2)),
        "select_unitarity_error": float(np.linalg.norm(select.conj().T@select-identity, 2)),
        "select_hermiticity_error": float(np.linalg.norm(select-select.conj().T, 2)),
        "block_encoding_error": float(np.linalg.norm(block[:ds, :ds]-H/ops["lambda_value"], 2)),
        "block_hermiticity_error": float(np.linalg.norm(block-block.conj().T, 2)),
        "walk_inverse_error": float(np.linalg.norm(ops["walk_inverse"]@walk-identity, 2)),
        "signal_from_walk_error": float(np.linalg.norm(ops["reflection"]@walk-block, 2)),
        "heralded_unitarity_error": float(np.linalg.norm(unitary.conj().T@unitary-np.eye(len(unitary)), 2)),
        "recovered_evolution_error": recovered_error,
        "certified_recovered_evolution_error_bound": settings["recovered_evolution_error_bound"],
        "success_amplitude": beta,
        "minimum_success_probability": float(np.linalg.svd(success_block, compute_uv=False)[-1]**2),
        "frozen_interpolation": payload.get("agp_metadata", {}).get("interpolation"),
        "interpretation": "heralded exp(-i t H_CD) for a frozen Hamiltonian, not time-ordered adiabatic state preparation",
    }
    primitive_keys = [key for key in report if key.endswith("error") and key != "recovered_evolution_error"]
    if any(report[key] > 1e-10 for key in primitive_keys):
        raise ValueError(f"Quantum reference primitive failed: {report}")
    if recovered_error > settings["recovered_evolution_error_bound"]+1e-11:
        raise ValueError("Dense evolution exceeds certified error bound")
    return report


def verify_spectral_reference(payload, settings, *, max_system_qubits=10):
    """Check the phase response on every eigenvalue against direct expm.

    This avoids allocating the much larger index/branch circuit unitary.
    It verifies the polynomial/evolution target, not native gate execution.
    Full gate-by-gate ancilla references are exercised separately on small
    systems in the scientific tests.
    """
    nq = len(payload["selected_features"])
    if nq > max_system_qubits:
        raise ValueError("System exceeds the bounded spectral-reference budget")
    H = hamiltonian_matrix(payload["pauli_terms"], nq)
    if np.linalg.norm(H-H.conj().T, 2) > 1e-12:
        raise ValueError("Spectral reference requires a Hermitian Hamiltonian")
    eigenvalues, vectors = np.linalg.eigh(H)
    normalized = eigenvalues/settings["lambda_value"]
    if np.max(np.abs(normalized)) > 1+1e-12:
        raise ValueError("Hamiltonian eigenvalue exceeds the LCU norm")
    responses = []
    for name in ("cosine", "sine"):
        branch = settings[name]
        values = []
        for x in np.clip(normalized, -1, 1):
            signal = np.array([[x, np.sqrt(max(0., 1-x*x))],
                               [np.sqrt(max(0., 1-x*x)), -x]])
            values.append(qsvt_matrix(signal, 0, branch["phase_angles"])[0, 0].real/branch["scale"])
        responses.append(np.asarray(values))
    recovered = (vectors*(responses[0]-1j*responses[1])[None, :])@vectors.conj().T
    target = expm(-1j*settings["physical_time"]*H)
    error = float(np.linalg.norm(recovered-target, 2))
    if error > settings["recovered_evolution_error_bound"]+1e-11:
        raise ValueError("Spectral reference exceeds the numerical error certificate")
    probability = settings["success_amplitude"]**2*np.abs(responses[0]-1j*responses[1])**2
    return {"status": "verified_spectral_reference", "num_system_qubits": nq,
            "native_compiler_verified": False, "recovered_evolution_error": error,
            "certified_recovered_evolution_error_bound": settings["recovered_evolution_error_bound"],
            "minimum_success_probability": float(np.min(probability)),
            "maximum_success_probability": float(np.max(probability)),
            "interpretation": "all Hamiltonian eigenvalues and direct expm; full ancilla gate model tested separately on bounded fixtures"}
