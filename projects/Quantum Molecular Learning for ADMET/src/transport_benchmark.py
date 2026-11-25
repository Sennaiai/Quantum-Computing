"""Bounded time-dependent counterdiabatic transport experiment.

The production first-order Pauli AGP is compared with bare annealing and an
independent, state-specific exact spectral reference. The reference transports
the isolated ground-state projector; it does not claim a full-spectrum AGP.
These small classical simulations establish experiment consistency, not quantum
advantage or ADMET prediction accuracy. Reference: Sels & Polkovnikov,
arXiv:1607.05687; nested-commutator ansatz: Claeys et al., arXiv:1904.03209.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import numbers

import numpy as np
from scipy.integrate import simpson, solve_ivp
from scipy.linalg import expm

from classical_preprocessing import (
    AGPSettings, build_initial_hamiltonian, build_problem_hamiltonian,
    recalculate_counterdiabatic_hamiltonian,
)
from quantum_reference import hamiltonian_matrix, verify_quantum_reference, verify_spectral_reference

METHODS = ("bare", "first_order_cd", "exact_ground_cd")


def _real(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite real number")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive")
    return float(value)


def _integer(value, name, low, high):
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    return value


@dataclass(frozen=True)
class TransportConfig:
    durations: tuple[float, ...] = (0.2, 0.5, 1.0, 2.0, 5.0)
    grid_points: int = 25
    midpoint_steps: tuple[int, ...] = (32, 64, 128)
    max_system_qubits: int = 6
    rtol: float = 1e-9
    atol: float = 1e-11
    gap_tolerance: float = 1e-10
    max_rhs_evaluations: int = 40000
    norm_tolerance: float = 1e-7
    exact_infidelity_tolerance: float = 1e-7
    midpoint_state_tolerance: float = 2e-3
    qsvt_snapshots: bool = True
    snapshot_physical_time: float = 0.1
    snapshot_epsilon: float = 1e-6

    def validate(self):
        _integer(self.grid_points, "grid_points", 5, 501)
        _integer(self.max_system_qubits, "max_system_qubits", 1, 6)
        _integer(self.max_rhs_evaluations, "max_rhs_evaluations", 32, 1000000)
        if not isinstance(self.durations, (tuple, list)) or not 1 <= len(self.durations) <= 16:
            raise ValueError("durations must contain between 1 and 16 positive times")
        durations = [_real(x, "duration", positive=True) for x in self.durations]
        if durations != sorted(set(durations)):
            raise ValueError("durations must be unique and increasing")
        if (not isinstance(self.midpoint_steps, (tuple, list)) or not 2 <= len(self.midpoint_steps) <= 5):
            raise ValueError("midpoint_steps must contain 2 to 5 increasing step counts")
        steps = [_integer(n, "midpoint step count", 4, 2048) for n in self.midpoint_steps]
        if steps != sorted(set(steps)):
            raise ValueError("midpoint_steps must be unique and increasing")
        for name in ("rtol", "atol", "gap_tolerance", "norm_tolerance", "exact_infidelity_tolerance",
                     "midpoint_state_tolerance", "snapshot_physical_time", "snapshot_epsilon"):
            _real(getattr(self, name), name, positive=True)
        if not 1e-13 <= self.rtol <= 1e-4 or not 1e-15 <= self.atol <= 1e-5:
            raise ValueError("Integrator tolerances exceed the supported float64 range")
        if not 1e-10 <= self.snapshot_epsilon < 1:
            raise ValueError("snapshot_epsilon must lie in [1e-10, 1)")
        if self.norm_tolerance > 1e-4 or self.exact_infidelity_tolerance > 1e-4 or self.midpoint_state_tolerance > .1:
            raise ValueError("Experiment acceptance tolerances are too loose")
        if not isinstance(self.qsvt_snapshots, bool):
            raise ValueError("qsvt_snapshots must be Boolean")


def smooth_schedule(u):
    """Cubic interpolation and derivative with respect to u=t/T."""
    u = _real(u, "schedule fraction")
    if not 0 <= u <= 1:
        raise ValueError("schedule fraction must lie in [0, 1]")
    return u*u*(3-2*u), 6*u*(1-u)


def exact_ground_projector_agp(H, derivative, *, gap_tolerance=1e-10):
    """A_g=i[dP_g/ds,P_g] from an independent dense spectral decomposition.

    Only ground↔excited couplings are needed to transport an isolated ground
    state. Excited degeneracies cause no singular denominators. The threshold
    is relative to max(1, ||H||_2); a closing ground gap fails explicitly.
    """
    H, derivative = np.asarray(H, complex), np.asarray(derivative, complex)
    if (H.ndim != 2 or H.shape != derivative.shape or H.shape[0] != H.shape[1]
            or H.shape[0] < 2 or not np.isfinite(H).all() or not np.isfinite(derivative).all()):
        raise ValueError("Spectral AGP requires matching finite square matrices")
    if H.shape[0] > 64:
        raise ValueError("Spectral AGP exceeds the 64-dimensional reference budget")
    if np.linalg.norm(H-H.conj().T, 2) > 1e-12 or np.linalg.norm(derivative-derivative.conj().T, 2) > 1e-12:
        raise ValueError("Spectral AGP requires Hermitian matrices")
    tolerance = _real(gap_tolerance, "gap_tolerance", positive=True)
    values, vectors = np.linalg.eigh(H)
    scale = max(1.0, float(np.max(np.abs(values))))
    gap = float(values[1]-values[0])
    if gap <= tolerance*scale:
        raise ValueError("Ground-state gap is unresolved; exact-projector transport is undefined at this tolerance")
    eigen_derivative = vectors.conj().T @ derivative @ vectors
    amplitudes = eigen_derivative[1:, 0]/(values[0]-values[1:])
    connection = np.zeros_like(H)
    connection[1:, 0] = 1j*amplitudes
    connection[0, 1:] = -1j*amplitudes.conj()
    gauge = vectors @ connection @ vectors.conj().T
    return gauge, values, vectors


def state_distance(left, right):
    """Euclidean state distance after removing the physically irrelevant phase."""
    left, right = np.asarray(left, complex), np.asarray(right, complex)
    if left.shape != right.shape or left.ndim != 1 or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("State distance requires matching finite vectors")
    norms = np.linalg.norm(left), np.linalg.norm(right)
    if not all(math.isfinite(value) for value in norms) or min(norms) <= 0:
        raise ValueError("State vectors must be nonzero")
    left, right = left/norms[0], right/norms[1]
    overlap = np.vdot(left, right)
    return float(np.linalg.norm(left-right*np.exp(-1j*np.angle(overlap))))


class _TransportProblem:
    def __init__(self, adapter, dataset, theta, config):
        self.adapter, self.dataset = adapter, dataset
        problem = build_problem_hamiltonian(adapter, dataset, theta, None)
        self.problem_payload = {"selected_features": [int(x) if isinstance(x, numbers.Integral) else x
                                                      for x in problem["selected_features"]],
                                "pauli_terms": problem["pauli_terms"]}
        self.theta = problem["feature_vector"]
        self.num_qubits = len(self.theta)
        if self.num_qubits > config.max_system_qubits:
            raise ValueError("Transport benchmark exceeds the bounded system-qubit budget")
        self.dimension = 2**self.num_qubits
        self.H0 = hamiltonian_matrix(build_initial_hamiltonian(self.num_qubits), self.num_qubits)
        self.Hp = hamiltonian_matrix(problem["pauli_terms"], self.num_qubits)
        self.derivative = self.Hp-self.H0
        self.config = config
        self.initial = np.full(self.dimension, 1/np.sqrt(self.dimension), complex)
        if np.linalg.norm(self.H0@self.initial+self.num_qubits*self.initial) > 1e-12:
            raise ValueError("Transverse-driver initial state is inconsistent")
        self.min_observed_gap = math.inf
        self.target_values, self.target_vectors = np.linalg.eigh(self.Hp)
        # Establish endpoint uniqueness explicitly, before an expensive solve.
        self.exact(0.)
        self.exact(1.)

    def H(self, s):
        return (1-s)*self.H0+s*self.Hp

    def exact(self, s):
        gauge, values, vectors = exact_ground_projector_agp(
            self.H(s), self.derivative, gap_tolerance=self.config.gap_tolerance)
        self.min_observed_gap = min(self.min_observed_gap, float(values[1]-values[0]))
        return gauge, values, vectors

    def first(self, s):
        payload = recalculate_counterdiabatic_hamiltonian(
            self.adapter, self.dataset, self.theta,
            AGPSettings(interpolation=float(s), lambda_dot=1.), include_components=True)
        gauge = hamiltonian_matrix(payload["agp_components"]["agp"], self.num_qubits)
        return gauge, payload

    def gauge(self, s, method):
        if method == "bare":
            return np.zeros_like(self.H0)
        if method == "first_order_cd":
            return self.first(s)[0]
        if method == "exact_ground_cd":
            return self.exact(s)[0]
        raise ValueError("Unknown transport method")

    def generator(self, u, duration, method):
        s, ds_du = smooth_schedule(float(u))
        return duration*self.H(s)+ds_du*self.gauge(s, method)


def _observables(problem, state):
    norm_sq = float(np.vdot(state, state).real)
    if not math.isfinite(norm_sq) or norm_sq <= 0:
        raise ValueError("Integrator returned an invalid state norm")
    normalized = state/np.sqrt(norm_sq)
    fidelity = float(abs(np.vdot(problem.target_vectors[:, 0], normalized))**2)
    energy = float(np.vdot(normalized, problem.Hp@normalized).real-problem.target_values[0])
    if energy < -1e-10 or not -1e-12 <= fidelity <= 1+1e-12:
        raise ValueError("Final quantum observables violate their physical bounds")
    return min(1., max(0., fidelity)), max(0., energy), abs(norm_sq-1)


def _integrate(problem, duration, method, u_grid):
    calls = 0
    def rhs(u, state):
        nonlocal calls
        calls += 1
        if calls > problem.config.max_rhs_evaluations:
            raise ValueError("Transport RHS evaluation budget exceeded")
        return -1j*problem.generator(u, duration, method)@state
    solution = solve_ivp(rhs, (0., 1.), problem.initial, method="DOP853", t_eval=u_grid,
                         rtol=problem.config.rtol, atol=problem.config.atol)
    if not solution.success or solution.y.shape != (problem.dimension, len(u_grid)):
        raise ValueError("Time-dependent Schrödinger integration did not complete")
    norms = np.sum(np.abs(solution.y)**2, axis=0)
    max_norm_error = float(np.max(np.abs(norms-1)))
    if not np.isfinite(solution.y).all() or max_norm_error > problem.config.norm_tolerance:
        raise ValueError("Transport integration violated the norm acceptance tolerance")
    fidelity, energy, _ = _observables(problem, solution.y[:, -1])
    trajectory = []
    for u, state, norm_sq in zip(u_grid, solution.y.T, norms):
        s, _ = smooth_schedule(float(u))
        _, _, vectors = problem.exact(s)
        probability = float(abs(np.vdot(vectors[:, 0], state))**2/norm_sq)
        trajectory.append(min(1., max(0., probability)))
    if method == "exact_ground_cd" and 1-min(trajectory) > problem.config.exact_infidelity_tolerance:
        raise ValueError("Exact spectral ground-projector transport failed its fidelity reference")
    return solution.y[:, -1], {"duration": duration, "method": method,
        "ground_fidelity": fidelity, "ground_infidelity": 1-fidelity,
        "excitation_energy": energy, "max_norm_error": max_norm_error,
        "rhs_evaluations": int(solution.nfev), "ground_fidelity_trajectory": trajectory}


def _residual_grid(problem, u_grid):
    rows, control = [], {method: {"spectral": [], "hs_squared": []} for method in METHODS}
    d = problem.dimension
    for u in u_grid:
        s, ds_du = smooth_schedule(float(u))
        H = problem.H(s)
        exact, values, vectors = problem.exact(s)
        first, payload = problem.first(s)
        O1 = H@problem.derivative-problem.derivative@H
        O2 = H@O1-O1@H
        n1, n2 = np.vdot(O1, O1).real, np.vdot(O2, O2).real
        alpha = 0. if n1 == 0 else -float(n1/n2)
        independent_first = 1j*alpha*O1
        agreement = float(np.linalg.norm(first-independent_first, 2))
        if agreement > 1e-10:
            raise ValueError("Production Pauli AGP disagrees with the independent matrix ansatz")
        row = {"u": float(u), "interpolation": s, "ds_du": ds_du,
               "ground_gap": float(values[1]-values[0]), "first_order_alpha": payload["agp_metadata"]["alpha1"],
               "dense_alpha": alpha, "production_dense_agp_error": agreement}
        actions = {}
        for method, A in (("bare", np.zeros_like(H)), ("first_order_cd", first), ("exact_ground_cd", exact)):
            G = problem.derivative+1j*(A@H-H@A)
            action = float(np.vdot(G, G).real/d)
            eigen_residual = vectors.conj().T@G@vectors
            ground_transition = float(np.linalg.norm(eigen_residual[1:, 0]))
            row[f"{method}_hs_action"] = action
            row[f"{method}_ground_transition_residual"] = ground_transition
            actions[method] = action
            control[method]["spectral"].append(float(np.linalg.norm(A, 2)))
            control[method]["hs_squared"].append(float(np.vdot(A, A).real/d))
        if actions["first_order_cd"] > actions["bare"]+1e-10*max(1., actions["bare"]):
            raise ValueError("Variational first-order residual is larger than the zero-gauge residual")
        if row["exact_ground_cd_ground_transition_residual"] > 1e-9:
            raise ValueError("Exact-projector AGP failed ground-transition cancellation")
        rows.append(row)
    return rows, control


def _midpoint_convergence(problem, duration, reference_states):
    results = []
    for method in METHODS:
        errors = []
        for steps in problem.config.midpoint_steps:
            state = problem.initial.copy()
            for step in range(steps):
                u = (step+.5)/steps
                state = expm(-1j*problem.generator(u, duration, method)/steps)@state
            distance = state_distance(reference_states[(duration, method)], state)
            fidelity, energy, norm_error = _observables(problem, state)
            if norm_error > 1e-10:
                raise ValueError("Independent midpoint-expm propagation violated unitarity")
            errors.append(distance)
            results.append({"duration": duration, "method": method, "midpoint_steps": steps,
                            "phase_aligned_state_distance": distance, "ground_fidelity": fidelity,
                            "excitation_energy": energy, "norm_error": norm_error})
        numerical_floor = max(1e-10, 100*problem.config.rtol)
        if errors[-1] > problem.config.midpoint_state_tolerance or (errors[-1] >= errors[0] and errors[0] > numerical_floor):
            raise ValueError("Independent midpoint-expm refinement did not confirm the ODE state")
    return results


def _qsvt_snapshots(problem, duration):
    from hamiltonian_validation import HamiltonianValidation
    from qsvt_helper import QSVTHelper
    rows = []
    for u in (.25, .5, .75):
        s, ds_du = smooth_schedule(u)
        payload = recalculate_counterdiabatic_hamiltonian(problem.adapter, problem.dataset, problem.theta,
            AGPSettings(interpolation=s, lambda_dot=ds_du/duration))
        validator = HamiltonianValidation(payload)
        settings = QSVTHelper(validator, physical_time=problem.config.snapshot_physical_time,
                              epsilon=problem.config.snapshot_epsilon).to_settings()
        dimension = 2**(validator.num_system_qubits+validator.num_index_qubits)
        if dimension <= 256:
            reference = verify_quantum_reference(payload, settings, max_dimension=256)
        else:
            reference = verify_spectral_reference(payload, settings, max_system_qubits=problem.config.max_system_qubits)
        rows.append({"schedule_fraction": u, "interpolation": s, "lambda_dot": ds_du/duration,
                     "snapshot_time": problem.config.snapshot_physical_time,
                     "epsilon": problem.config.snapshot_epsilon, "num_terms": validator.num_terms,
                     "lambda_value": validator.lambda_value,
                     "cosine_degree": settings["cosine"]["polynomial_degree"],
                     "sine_degree": settings["sine"]["polynomial_degree"],
                     "success_amplitude": settings["success_amplitude"],
                     "success_probability_lower_bound": settings["success_probability_lower_bound"],
                     "reference": reference})
    return rows


def run_transport_benchmark(curated_adapter, dataset, feature_vector, config=None):
    """Return JSON-ready scientific evidence for one bounded molecular instance.

    No fitting, downloads, API operations, or files are involved. Production
    Pauli AGPs drive the first-order ODE and midpoint propagator. The exact
    ground-projector gauge and matrix-residual checks are independent spectral
    references. A duration sweep is a deterministic experiment, not a
    statistical sample or a claim of general CD superiority.
    """
    config = TransportConfig() if config is None else config
    if not isinstance(dataset, str) or not dataset:
        raise ValueError("dataset must be a nonempty string")
    if not isinstance(config, TransportConfig):
        raise ValueError("config must be a TransportConfig")
    config.validate()
    problem = _TransportProblem(curated_adapter, dataset, feature_vector, config)
    u_grid = np.linspace(0., 1., config.grid_points)
    residual_grid, controls = _residual_grid(problem, u_grid)
    rows, reference_states = [], {}
    derivative = np.asarray([smooth_schedule(float(u))[1] for u in u_grid])
    for duration in map(float, config.durations):
        for method in METHODS:
            state, row = _integrate(problem, duration, method, u_grid)
            reference_states[(duration, method)] = state
            row["sampled_peak_control_norm"] = float(np.max(derivative*np.asarray(controls[method]["spectral"])/duration))
            row["integrated_control_hs_squared_quadrature"] = float(simpson(
                derivative**2*np.asarray(controls[method]["hs_squared"])/duration, x=u_grid))
            rows.append(row)
    convergence_duration = min(map(float, config.durations), key=lambda T: abs(T-1.))
    convergence = _midpoint_convergence(problem, convergence_duration, reference_states)
    snapshots = _qsvt_snapshots(problem, convergence_duration) if config.qsvt_snapshots else []
    initial_spectrum = np.linalg.eigvalsh(problem.H0)
    excited_degenerate_pairs = sum(abs(initial_spectrum[i]-initial_spectrum[j]) <= config.gap_tolerance*max(1., problem.num_qubits)
                                   for i in range(1, problem.dimension) for j in range(i+1, problem.dimension))
    canonical_problem = json.dumps(problem.problem_payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return {"schema_version": 1, "status": "passed", "dataset": dataset,
        "problem_hamiltonian": problem.problem_payload,
        "problem_hamiltonian_sha256": hashlib.sha256(canonical_problem.encode()).hexdigest(),
        "endpoint_ground_multiplicity": 1, "driver_excited_degenerate_pairs": int(excited_degenerate_pairs),
        "excited_degeneracy_policy": "allowed: the exact reference couples only the isolated ground projector to its complement",
        "num_system_qubits": problem.num_qubits, "feature_vector": list(problem.theta),
        "config": asdict(config), "u_grid": u_grid.tolist(),
        "schedule": "s(u)=3u^2-2u^3; u=t/T; endpoint schedule derivatives vanish",
        "initial_state": "unique H0=-sum(X) ground state |+>^n; distinct from API feature RY initialization",
        "exact_reference": "A_g=i[dP_g/ds,P_g], isolated ground-projector spectral AGP; not full-spectrum exact AGP",
        "observed_minimum_ground_gap": problem.min_observed_gap,
        "gap_policy": "every spectral evaluation requires E1-E0 > gap_tolerance*max(1,||H||2); sampled gap is not a rigorous global bound",
        "rows": rows, "residual_grid": residual_grid, "convergence": convergence,
        "qsvt_snapshots": snapshots,
        "control_cost_convention": "sampled spectral norm and Simpson quadrature of normalized Hilbert-Schmidt control squared; energy units are those of the constructed Hamiltonian",
        "scope": "bounded classical quantum-state simulation and mathematical QSVT snapshots; no native compiler, hardware, time-ordered QSVT circuit, predictive ADMET performance, or speedup claim"}
