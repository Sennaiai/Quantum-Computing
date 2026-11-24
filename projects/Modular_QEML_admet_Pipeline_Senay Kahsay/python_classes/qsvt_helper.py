"""Bounded, certified phase synthesis for heralded Hamiltonian simulation.

The two real QSVT polynomials encode scaled cosine and sine. A coherent
real-part extraction and cosine/sine LCU are required; a single cosine sequence
is not a time-evolution circuit. Native Beryllium compilation remains a separate
validation gate.
"""
from __future__ import annotations

import contextlib
import io
import math
import numpy as np
from scipy.fft import dct
from scipy.special import gammaln, jv
from pyqsp.angle_sequence import QuantumSignalProcessingPhases
from hamiltonian_validation import finite_real


PHASE_CONVENTION = "projector_exp_i_phi_R"
ALGORITHM = "heralded_cosine_sine_lcu_v2"


def reflection_response(points, phases):
    """Independent 2x2 reflection-QSP matrix response, not pyqsp.response.

    R(x)=[[x,sqrt(1-x*x)],[sqrt(1-x*x),-x]],
    D(phi)=diag(exp(i phi),exp(-i phi)); U=D0 R D1 ... R Dd.
    The desired real polynomial is Re(U[0,0]), coherently extracted using
    (U+U†)/2 in the circuit.
    """
    values = []
    for x in np.asarray(points, dtype=float):
        block = np.array([[x, np.sqrt(max(0.0, 1-x*x))],
                          [np.sqrt(max(0.0, 1-x*x)), -x]], dtype=complex)
        unitary = np.diag(np.exp(1j * phases[0] * np.array([1, -1])))
        for phi in phases[1:]:
            unitary = unitary @ block @ np.diag(np.exp(1j * phi * np.array([1, -1])))
        values.append(unitary[0, 0])
    return np.asarray(values)


def _tail_bound(tau, degree):
    # DLMF 10.14.4: https://dlmf.nist.gov/10.14.E4
    # For real tau, |J_k(tau)| <= (|tau|/2)**k/k!. Bound the
    # complete remaining Chebyshev tail by a geometric majorant. This
    # deliberately includes both parities, so it bounds either branch.
    ratio = abs(tau) / (2 * (degree + 2))
    if ratio >= 1:
        return math.inf
    log_first = (degree + 1)*math.log(abs(tau)/2) - gammaln(degree + 2)
    if log_first > 700:
        return math.inf
    return float(2 * math.exp(log_first) / (1-ratio))


class QSVTHelper:
    """Synthesize both parity branches and certify recovered evolution error.

    Certification combines a Jacobi-Anger tail majorant with an L1 norm of
    independently reconstructed Chebyshev synthesis-error coefficients.
    Floating-point reconstruction includes an explicit numerical allowance;
    this is numerical verification, not interval-arithmetic proof.
    """
    def __init__(self, validator, physical_time=1.0, epsilon=1e-6, *, max_degree=256):
        self.lambda_value = finite_real(validator.lambda_value, "lambda_value")
        self.physical_time = finite_real(physical_time, "physical_time")
        self.epsilon = finite_real(epsilon, "epsilon")
        if not all(math.isfinite(x) for x in (self.lambda_value, self.physical_time, self.epsilon)):
            raise ValueError("lambda, physical_time, and epsilon must be finite")
        if self.lambda_value <= 0 or self.physical_time <= 0:
            raise ValueError("lambda and physical_time must be positive")
        if not 1e-10 <= self.epsilon < 1:
            raise ValueError("epsilon must lie in [1e-10, 1); smaller targets need higher-precision certification")
        if isinstance(max_degree, bool) or not isinstance(max_degree, int) or max_degree < 3:
            raise ValueError("max_degree must be an integer >= 3")
        self.tau = self.lambda_value * self.physical_time
        if not math.isfinite(self.tau) or self.tau <= 0:
            raise ValueError("lambda * physical_time overflowed or underflowed")
        degree = 2
        while degree <= max_degree and _tail_bound(self.tau, degree) > self.epsilon/16:
            degree += 1
        if degree > max_degree:
            raise ValueError(f"Requested evolution exceeds phase-synthesis degree budget {max_degree}")
        even_degree = degree + degree % 2
        odd_degree = degree + (1-degree % 2)
        if max(even_degree, odd_degree) > max_degree:
            raise ValueError("Parity-rounded polynomial exceeds max_degree")
        self._branches = {}
        self.suc = 1 - min(self.epsilon/64, 1e-4)
        for name, last_degree, parity in (("cosine", even_degree, 0), ("sine", odd_degree, 1)):
            coefficients = np.zeros(last_degree+1)
            if parity == 0:
                coefficients[0] = jv(0, self.tau)
                for k in range(2, last_degree+1, 2):
                    coefficients[k] = 2*(-1)**(k//2)*jv(k, self.tau)
            else:
                for k in range(1, last_degree+1, 2):
                    coefficients[k] = 2*(-1)**((k-1)//2)*jv(k, self.tau)
            bounded = .5 * coefficients
            # pyqsp rescales by suc and adds eps/2 to the highest
            # Chebyshev coefficient. Both effects are included below.
            with contextlib.redirect_stdout(io.StringIO()):
                wx = np.asarray(QuantumSignalProcessingPhases(
                    bounded, signal_operator="Wx", measurement="x", method="laurent",
                    # This is pyqsp's internal sampled reconstruction
                    # threshold, not our final scientific acceptance bound.
                    eps=self.epsilon/64, suc=self.suc,
                    tolerance=max(1e-9, self.epsilon/16),
                ), dtype=float)
            phases = self._wx_to_low_chuang(wx)
            if len(phases) != last_degree+1 or not np.all(np.isfinite(phases)):
                raise ValueError("Phase synthesis returned invalid degree or nonfinite angles")
            nodes = np.cos(np.pi*(np.arange(last_degree+1)+.5)/(last_degree+1))
            response = reflection_response(nodes, phases).real
            reconstructed = dct(response, type=2)/(last_degree+1)
            reconstructed[0] /= 2
            scale = .5*self.suc
            rounding = 256*np.finfo(float).eps*(last_degree+1)**2
            synthesis_error = float(np.sum(np.abs(reconstructed/scale-coefficients)) + rounding/scale)
            approximation_error = _tail_bound(self.tau, last_degree)
            # A second set of points detects numerical reconstruction or
            # convention mistakes independently of the interpolation nodes.
            grid = np.linspace(-1, 1, max(513, 8*last_degree+1))
            grid_response = reflection_response(grid, phases).real/scale
            if np.max(np.abs(grid_response-np.polynomial.chebyshev.chebval(grid, coefficients))) > synthesis_error:
                raise ValueError("Independent phase-response reconstruction failed")
            self._branches[name] = {
                "phase_angles": phases.tolist(), "polynomial_degree": last_degree,
                "scale": scale, "approximation_error_bound": approximation_error,
                "synthesis_error_bound": synthesis_error,
                "unscaled_chebyshev_coefficients": coefficients.tolist(),
            }
            setattr(self, "cos_coeffs" if parity == 0 else "sin_coeffs", bounded)
            setattr(self, "cos_wx_phases" if parity == 0 else "sin_wx_phases", wx)
            setattr(self, "cos_lc_phases" if parity == 0 else "sin_lc_phases", phases)
            setattr(self, "cos_scale" if parity == 0 else "sin_scale", scale)
        self.recovered_evolution_error_bound = sum(
            b["approximation_error_bound"] + b["synthesis_error_bound"]
            for b in self._branches.values()
        )
        if self.recovered_evolution_error_bound > self.epsilon:
            raise ValueError(f"Certified recovered-evolution error {self.recovered_evolution_error_bound:.3g} exceeds epsilon {self.epsilon:.3g}")

    @staticmethod
    def _wx_to_low_chuang(wx_phases):
        """Wx -> reflection convention, Appendix A.2 of arXiv:2105.02859.

        Equivalent modulo global 2*pi to PennyLane transform_angles for
        QSP -> QSVT. Source: https://docs.pennylane.ai/en/stable/_modules/
        pennylane/templates/subroutines/qsvt.html#transform_angles
        """
        phases = np.asarray(wx_phases, dtype=float).copy()
        degree = len(phases)-1
        phases[0] += (2*degree-1)*np.pi/4
        phases[1:-1] -= np.pi/2
        phases[-1] -= np.pi/4
        return phases

    def cosine_phases(self):
        return self.cos_lc_phases.copy()

    def sine_phases(self):
        return self.sin_lc_phases.copy()

    @property
    def cosine_degree(self):
        return self._branches["cosine"]["polynomial_degree"]

    @property
    def sine_degree(self):
        return self._branches["sine"]["polynomial_degree"]

    def to_settings(self):
        import copy
        branches = copy.deepcopy(self._branches)
        sc, ss = branches["cosine"]["scale"], branches["sine"]["scale"]
        beta = 1/(1/sc+1/ss)
        return {
            "algorithm": ALGORITHM, "phase_convention": PHASE_CONVENTION,
            "lambda_value": self.lambda_value, "physical_time": self.physical_time,
            "epsilon": self.epsilon, "tau": self.tau,
            **branches, "success_amplitude": beta,
            "ideal_success_probability": beta*beta,
            "success_probability_lower_bound": beta*beta*(1-self.recovered_evolution_error_bound)**2,
            "branch_rotation_angle": 2*math.acos(math.sqrt((1/sc)/(1/sc+1/ss))),
            "extra_ancillas": 2,
            "logical_resources": {
                "cosine_block_queries": branches["cosine"]["polynomial_degree"],
                "sine_block_queries": branches["sine"]["polynomial_degree"],
                "total_controlled_block_queries": 2*(branches["cosine"]["polynomial_degree"]+branches["sine"]["polynomial_degree"]),
                "extra_branch_and_extraction_qubits": 2,
                "interpretation": "logical query counts only; no native depth/noise/advantage claim",
            },
            "recovered_evolution_error_bound": self.recovered_evolution_error_bound,
            "certificate": {
                "target": "heralded zero-ancilla block / beta approximates exp(-i H physical_time)",
                "norm": "spectral/operator norm for Hermitian H with ||H|| <= lambda",
                "method": "Jacobi-Anger tail majorant plus Chebyshev L1 phase-reconstruction error and floating-point allowance",
                "arithmetic": "float64 numerical certificate; not interval arithmetic",
                "postselected_state_distance_bound": 2*self.recovered_evolution_error_bound/(1-self.recovered_evolution_error_bound),
                "scope": "frozen H_CD at the chosen interpolation; not time-ordered annealing",
            },
            "native_validation": {
                "status": "unverified", "compiler_verified": False,
                "gate_convention": "Ry(theta)=exp(-i theta Y/2); globalphase(theta)=exp(i theta)",
                "live_api_execution_allowed": False,
            },
        }

    def report(self):
        s = self.to_settings()
        print({key: s[key] for key in ("lambda_value", "physical_time", "epsilon", "success_amplitude", "recovered_evolution_error_bound")})
