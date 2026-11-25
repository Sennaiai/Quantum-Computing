"""Window-reset CV and analogue-spin reservoirs; exact ideal expectations.

The NumPy batch kernels execute state evolution. Independent PennyLane,
Qiskit and analytic checks live in test_physics.py; they are not training code.
"""
from __future__ import annotations

from itertools import combinations
import numpy as np
from scipy.linalg import expm


def finite_windows(windows, features=7):
    x = np.asarray(windows, dtype=float)
    if x.ndim != 3 or x.shape[0] == 0 or not 1 <= x.shape[1] <= 48 or x.shape[2] != features:
        raise ValueError('Expected nonempty [samples, time<=48, seven grid features]')
    if not np.isfinite(x).all():
        raise ValueError('Reservoir input must be finite')
    return x


class CVReservoir:
    """Two modes, hbar=2, D(alpha), R(phi), B(theta,phase), K(kappa).

    K=exp(+i*kappa*n**2). Cutoff is per mode; truncated generators are
    unitary, so norm conservation alone cannot diagnose Fock convergence.
    """
    def __init__(self, seed, kerr=.2, cutoff=10):
        if isinstance(cutoff, bool) or not isinstance(cutoff, int) or not 4 <= cutoff <= 20:
            raise ValueError('CV cutoff must be an integer in [4,20]')
        if not np.isfinite(kerr) or abs(kerr) > 1:
            raise ValueError('Invalid Kerr strength')
        self.cutoff = cutoff; self.kerr = float(kerr)
        rng = np.random.default_rng(seed)
        self.projection = rng.normal(0, 1/np.sqrt(7), (2, 7))
        self.phi = rng.uniform(-1., 1., 2)
        self.theta = float(rng.uniform(.2, .6)); self.phase = float(rng.uniform(-1., 1.))
        self.kappas = self.kerr*np.array([1., -.8])
        a = np.diag(np.sqrt(np.arange(1, cutoff)), 1).astype(complex)
        self.disp_eig, self.disp_vec = np.linalg.eigh(1j*(a.conj().T-a))
        eye = np.eye(cutoff); n = np.arange(cutoff)
        bs = expm(self.theta*(np.exp(1j*self.phase)*np.kron(a, a.conj().T)
                             - np.exp(-1j*self.phase)*np.kron(a.conj().T, a)))
        rotation = np.exp(1j*(self.phi[0]*n[:, None] + self.phi[1]*n[None, :])).ravel()
        k = np.exp(1j*(self.kappas[0]*n[:, None]**2+self.kappas[1]*n[None, :]**2)).ravel()
        self.unitary = k[:, None]*bs*rotation[None, :]

    def encode(self, windows):
        return .08*np.tanh(finite_windows(windows) @ self.projection.T)

    def evolve(self, windows):
        z = self.encode(windows); d = self.cutoff
        state = np.zeros((len(z), d, d), dtype=complex); state[:, 0, 0] = 1
        boundary = 0.
        v = self.disp_vec
        for t in range(z.shape[1]):
            s = np.einsum('ij,bjk->bik', v.conj().T, state, optimize=True)
            s *= np.exp(-1j*z[:, t, 0, None]*self.disp_eig)[..., None]
            state = np.einsum('ij,bjk->bik', v, s, optimize=True)
            s = state @ v.conj()
            s *= np.exp(-1j*z[:, t, 1, None]*self.disp_eig)[:, None, :]
            state = s @ v.T
            state = (state.reshape(len(z), -1) @ self.unitary.T).reshape(-1, d, d)
            p = abs(state)**2
            boundary = max(boundary, float(np.max(p[:, -1, :].sum(1)+p[:, :, -1].sum(1)-p[:, -1, -1])))
        if np.max(abs(np.sum(abs(state)**2, axis=(1, 2))-1)) > 1e-9:
            raise ArithmeticError('CV state normalization failed')
        return state, boundary

    def features(self, windows):
        s, boundary = self.evolve(windows); d = self.cutoff
        a0 = np.sum(s[:, :-1, :].conj()*s[:, 1:, :]*np.sqrt(np.arange(1, d))[None, :, None], axis=(1, 2))
        a1 = np.sum(s[:, :, :-1].conj()*s[:, :, 1:]*np.sqrt(np.arange(1, d))[None, None, :], axis=(1, 2))
        p = abs(s)**2
        f = np.column_stack([2*a0.real, 2*a0.imag, 2*a1.real, 2*a1.imag,
                             np.sum(p*np.arange(d)[None, :, None], axis=(1, 2)),
                             np.sum(p*np.arange(d)[None, None, :], axis=(1, 2))])
        return f, {'max_cutoff_boundary_probability': boundary,
                   'max_norm_error': float(np.max(abs(p.sum((1, 2))-1)))}

    def gaussian_reference(self, windows):
        """Exact classical coherent amplitudes when kappa=0; no Fock cutoff."""
        if self.kerr != 0:
            raise ValueError('Gaussian reference requires kerr=0')
        z = self.encode(windows); alpha = np.zeros((len(z), 2), complex)
        c, s = np.cos(self.theta), np.sin(self.theta)
        mix = np.array([[c, -np.exp(-1j*self.phase)*s], [np.exp(1j*self.phase)*s, c]])
        for t in range(z.shape[1]):
            alpha = ((alpha+z[:, t])*np.exp(1j*self.phi)) @ mix.T
        return np.column_stack([2*alpha[:, 0].real, 2*alpha[:, 0].imag,
                                2*alpha[:, 1].real, 2*alpha[:, 1].imag,
                                abs(alpha[:, 0])**2, abs(alpha[:, 1])**2])


def spin_terms(n, seed, coupled=True):
    rng = np.random.default_rng(seed)
    hx = rng.uniform(.5, 1., n); hz = rng.uniform(-.6, .6, n)
    coupling = rng.uniform(.2, .7, n-1) if coupled else np.zeros(n-1)
    return hx, hz, coupling


def spin_hamiltonian(n, hx, hz, coupling):
    """H=sum hx X + hz Z + nearest-neighbour J ZZ, little-endian basis."""
    size = 2**n; indices = np.arange(size)
    signs = [1-2*((indices >> q) & 1) for q in range(n)]
    diag = sum(hz[q]*signs[q] for q in range(n))
    diag += sum(coupling[q]*signs[q]*signs[q+1] for q in range(n-1))
    h = np.diag(diag).astype(complex)
    for q in range(n):
        h[indices ^ (1 << q), indices] += hx[q]
    return h


class SpinReservoir:
    """Exact exp(-i tau H), tau=.2, between input RY and quadratic RZ drives.

    Six qubits by default. This simulates analogue spin dynamics with coherent
    control; it is neither a bosonic Kerr device nor an analogue hardware run.
    """
    def __init__(self, seed, n=6, coupled=True):
        if isinstance(n, bool) or not isinstance(n, int) or not 2 <= n <= 9:
            raise ValueError('Spin register must have 2..9 qubits')
        self.n = n; self.seed = seed; self.tau = .2
        self.hx, self.hz, self.coupling = spin_terms(n, seed, coupled)
        self.hamiltonian = spin_hamiltonian(n, self.hx, self.hz, self.coupling)
        self.unitary = expm(-1j*self.tau*self.hamiltonian)
        self.projection = np.random.default_rng(seed+1000).normal(0, 1/np.sqrt(7), (n, 7))
        self.pairs = list(combinations(range(n), 2))

    def encode(self, windows):
        return .35*np.tanh(finite_windows(windows) @ self.projection.T)

    def evolve(self, windows):
        z = self.encode(windows); indices = np.arange(2**self.n)
        state = np.zeros((len(z), len(indices)), complex); state[:, 0] = 1
        for t in range(z.shape[1]):
            for q in range(self.n):
                low = indices[(indices & (1 << q)) == 0]; high = low | (1 << q)
                c, s = np.cos(z[:, t, q]/2)[:, None], np.sin(z[:, t, q]/2)[:, None]
                a, b = state[:, low].copy(), state[:, high].copy()
                state[:, low] = c*a-s*b; state[:, high] = s*a+c*b
                angle = .1*z[:, t, q]**2
                state *= np.exp(-.5j*angle[:, None]*(1-2*((indices >> q) & 1)))
            state = state @ self.unitary.T
        if np.max(abs(np.sum(abs(state)**2, axis=1)-1)) > 1e-9:
            raise ArithmeticError('Spin state normalization failed')
        return state

    def features(self, windows):
        state = self.evolve(windows); indices = np.arange(2**self.n)
        prob = abs(state)**2; signs = [1-2*((indices >> q) & 1) for q in range(self.n)]
        features = []
        for q in range(self.n):
            low = indices[(indices & (1 << q)) == 0]; high = low | (1 << q)
            cross = np.sum(state[:, low].conj()*state[:, high], axis=1)
            features.extend([2*cross.real, 2*cross.imag, prob @ signs[q]])
        features.extend(prob @ (signs[a]*signs[b]) for a, b in self.pairs)
        return np.column_stack(features), {'max_norm_error': float(np.max(abs(prob.sum(1)-1)))}

    def qiskit_circuit(self, window, slices=None):
        """Exact-unitary reference or second-order product-formula resource proxy."""
        from qiskit import QuantumCircuit
        z = self.encode(np.asarray(window)[None])[0]
        if slices is not None and (isinstance(slices, bool) or not isinstance(slices, int) or slices < 1):
            raise ValueError('Product-formula slices must be positive integers')
        circuit = QuantumCircuit(self.n)
        for row in z:
            for q in range(self.n):
                circuit.ry(float(row[q]), q); circuit.rz(float(.1*row[q]**2), q)
            if slices is None:
                circuit.unitary(self.unitary, list(range(self.n)), label='exp(-i tau H)')
            else:
                dt = self.tau/slices
                for _ in range(slices):
                    for q in range(self.n): circuit.rz(float(dt*self.hz[q]), q)
                    for q in range(self.n-1): circuit.rzz(float(dt*self.coupling[q]), q, q+1)
                    for q in range(self.n): circuit.rx(float(2*dt*self.hx[q]), q)
                    for q in range(self.n-1): circuit.rzz(float(dt*self.coupling[q]), q, q+1)
                    for q in range(self.n): circuit.rz(float(dt*self.hz[q]), q)
        return circuit


def echo_features(windows, seed, width=33):
    """Classical nonlinear fixed-weight window reservoir, matched spin dimension."""
    x = finite_windows(windows); rng = np.random.default_rng(seed)
    w = rng.normal(0, .5/np.sqrt(7), (7, width))
    recurrent = rng.normal(0, 1/np.sqrt(width), (width, width))
    recurrent *= .8/max(abs(np.linalg.eigvals(recurrent)))
    h = np.zeros((len(x), width))
    for t in range(x.shape[1]): h = np.tanh(x[:, t]@w+h@recurrent)
    return h
