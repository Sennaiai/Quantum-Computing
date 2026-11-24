"""Small, explicit H2 benchmarks; all reported energies include nuclear repulsion."""
from __future__ import annotations

import platform
import hashlib
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
import pennylane as qml
from pyscf import dft, fci, gto, lib, scf
from scipy.optimize import minimize_scalar

lib.num_threads(1)
WIRE_ORDER = [3, 2, 1, 0]  # statevector bits are |q3 q2 q1 q0>
BOHR_ANGSTROM = 0.529177210903  # CODATA 2018; pinned PL 0.36 accepts bohr
COEFFICIENTS = [-0.138754, -0.152989, 0.164190, 0.144579,
                0.111373, 0.146726, 0.169348, -0.035353, 0.035353]


def require(condition, message):
    """Scientific acceptance checks stay enabled under python -O."""
    if not condition:
        raise ValueError(message)


def coordinates_bohr(distance):
    """Two hydrogen nuclei on z; public bond-length input is in angstrom."""
    if not np.isfinite(distance) or distance <= 0:
        raise ValueError('H-H distance must be finite and positive, in angstrom')
    return np.array([[0., 0., 0.], [0., 0., distance]]) / BOHR_ANGSTROM


def historical_operator():
    """Historical rounded coefficients, with no claimed molecular provenance."""
    c = COEFFICIENTS
    terms = [(c[0], {}), (c[1], {2:'Z'}), (c[1], {3:'Z'}),
             (c[2], {0:'Z'}), (c[2], {1:'Z'}), (c[3], {3:'Z',2:'Z'}),
             (c[4], {3:'Z',1:'Z'}), (c[4], {2:'Z',0:'Z'}),
             (c[5], {2:'Z',1:'Z'}), (c[5], {3:'Z',0:'Z'}),
             (c[6], {1:'Z',0:'Z'}),
             (c[7], {3:'Y',2:'Y',1:'X',0:'X'}),
             (c[7], {3:'X',2:'X',1:'Y',0:'Y'}),
             (c[8], {3:'X',2:'Y',1:'Y',0:'X'}),
             (c[8], {3:'Y',2:'X',1:'X',0:'Y'})]
    constructors = {'X':qml.PauliX, 'Y':qml.PauliY, 'Z':qml.PauliZ}
    return qml.Hamiltonian([t[0] for t in terms], [
        qml.prod(*(constructors[p](w) for w,p in word.items()))
        if word else qml.Identity(0) for _,word in terms])


def historical_analysis():
    matrix = qml.matrix(historical_operator(), wire_order=WIRE_ORDER)
    block = matrix[np.ix_([3,12], [3,12])].real
    ea, eb, v = block[0,0], block[1,1], block[0,1]
    center, delta = (ea+eb)/2, (ea-eb)/2
    # cos(theta)|0011> - sin(theta)|1100>
    theta = (np.arctan2(-v, delta) + np.pi)/2
    state = np.zeros(16); state[3],state[12] = np.cos(theta),-np.sin(theta)
    return dict(matrix=matrix, block=block, coupling=float(v), theta=float(theta),
                analytic_min=float(center-np.hypot(delta,v)),
                expectation=float(np.real(state @ matrix @ state)))


def classical_point(distance, basis='sto-3g', grid_level=3):
    coordinates_bohr(distance)  # validate before expensive external calls
    require(isinstance(grid_level, int) and 0 <= grid_level <= 9,
            'DFT grid level must be an integer from 0 to 9')
    mol = gto.M(atom=[('H',(0,0,0)),('H',(0,0,float(distance)))],
                unit='Angstrom', basis=basis, charge=0, spin=0, verbose=0)
    rhf = scf.RHF(mol); rhf.conv_tol = 1e-11; rhf.max_cycle = 100
    rhf.kernel()
    if not rhf.converged:
        raise RuntimeError(f'RHF failed at {distance} A')
    exact = fci.FCI(rhf); exact.conv_tol = 1e-12
    exact_energy, ci_vector = exact.kernel()
    if not exact.converged:
        raise RuntimeError(f'FCI failed at {distance} A')
    spin_squared, _ = exact.spin_square(ci_vector, mol.nao, mol.nelec)
    occupations = np.linalg.eigvalsh(exact.make_rdm1(ci_vector, mol.nao, mol.nelec))[::-1]
    rks = dft.RKS(mol); rks.xc = 'pbe'; rks.grids.level = grid_level
    rks.conv_tol = 1e-11; rks.max_cycle = 100; rks.kernel()
    if not rks.converged:
        raise RuntimeError(f'RKS-PBE failed at {distance} A')
    require(np.all(np.isfinite([rhf.e_tot, exact_energy, rks.e_tot, spin_squared])),
            'Nonfinite classical reference result')
    require(abs(spin_squared) < 1e-10, 'FCI reference is not a singlet')
    return dict(distance_angstrom=float(distance), basis=basis,
                rhf_hartree=float(rhf.e_tot), fci_hartree=float(exact_energy),
                rks_pbe_hartree=float(rks.e_tot), grid_level=grid_level,
                nuclear_repulsion_hartree=float(mol.energy_nuc()),
                rhf_converged=bool(rhf.converged), fci_converged=bool(exact.converged),
                rks_converged=bool(rks.converged), fci_spin_squared=float(spin_squared),
                fci_natural_occupations=occupations.tolist())


def quantum_point(distance):
    # PennyLane's differentiable HF integrals provide a separate implementation
    # from the PySCF FCI calculation, rather than reusing its mapped matrix.
    geometry = coordinates_bohr(distance)
    operator, n = qml.qchem.molecular_hamiltonian(
        ['H','H'], qml.numpy.array(geometry, requires_grad=False),
        charge=0, mult=1, basis='sto-3g', method='dhf', mapping='jordan_wigner')
    if n != 4:
        raise AssertionError('H2/STO-3G must have four spin orbitals')
    matrix = np.asarray(qml.matrix(operator, wire_order=range(n)))
    np.testing.assert_allclose(matrix, matrix.conj().T, atol=1e-12, rtol=0)
    # Interleaved alpha/beta orbitals; select Nalpha=Nbeta=1.
    sector = [i for i in range(16)
              if sum(int(b) for b in f'{i:04b}'[::2]) == 1
              and sum(int(b) for b in f'{i:04b}'[1::2]) == 1]
    outside = sorted(set(range(16))-set(sector))
    np.testing.assert_allclose(matrix[np.ix_(outside,sector)], 0, atol=1e-12, rtol=0)
    target = float(np.linalg.eigvalsh(matrix[np.ix_(sector,sector)])[0])
    number = np.diag([i.bit_count() for i in range(16)])
    np.testing.assert_allclose(matrix @ number, number @ matrix, atol=1e-12, rtol=0)
    spin2 = qml.matrix(qml.qchem.spin2(2, 4), wire_order=range(4))
    dev = qml.device('default.qubit', wires=4)

    @qml.qnode(dev)
    def state(theta):
        qml.BasisState(np.array([1,1,0,0]), wires=range(4))
        qml.DoubleExcitation(theta, wires=range(4))
        return qml.state()

    @qml.qnode(dev)
    def circuit_energy(theta):
        qml.BasisState(np.array([1,1,0,0]), wires=range(4))
        qml.DoubleExcitation(theta, wires=range(4))
        return qml.expval(operator)

    def energy(theta):
        return float(circuit_energy(theta))

    # One double excitation spans the gerade singlet block for minimal-basis H2.
    result = minimize_scalar(energy, bounds=(-np.pi,np.pi), method='bounded',
                             options={'xatol':1e-12, 'maxiter':200})
    if not result.success:
        raise RuntimeError(result.message)
    psi = np.asarray(state(result.x)); e = energy(result.x)
    np.testing.assert_allclose(e, np.real(np.vdot(psi, matrix @ psi)), atol=1e-12, rtol=0)
    residual = matrix @ psi - e*psi
    return dict(sector_exact_hartree=target, vqe_hartree=e,
                vqe_gap_hartree=e-target, theta=float(result.x),
                variance_hartree2=float(np.real(np.vdot(residual,residual))),
                sector_leakage=float(np.sum(abs(psi[outside])**2)),
                vqe_spin_squared=float(np.real(np.vdot(psi, spin2 @ psi))),
                vqe_particle_number=float(np.real(np.vdot(psi, number @ psi))),
                hf_qubit_hartree=float(np.real(matrix[12,12])),
                optimizer_evaluations=int(result.nfev), qubits=n,
                pauli_terms=len(operator.terms()[0]))


def validate_row(row):
    """Reject failed references or nonphysical results before reporting success."""
    for key in ['rhf_hartree', 'fci_hartree', 'rks_pbe_hartree',
                'sector_exact_hartree', 'vqe_hartree', 'vqe_gap_hartree',
                'sector_leakage', 'variance_hartree2', 'vqe_spin_squared',
                'fci_spin_squared', 'vqe_particle_number', 'hf_qubit_hartree']:
        require(np.isfinite(row[key]), f'Nonfinite {key}')
    for key in ['rhf_converged', 'fci_converged', 'rks_converged']:
        require(row[key], f'Failed convergence: {key}')
    np.testing.assert_allclose(row['sector_exact_hartree'], row['fci_hartree'], atol=2e-7, rtol=0)
    np.testing.assert_allclose(row['hf_qubit_hartree'], row['rhf_hartree'], atol=2e-7, rtol=0)
    np.testing.assert_allclose(row['vqe_hartree']-row['sector_exact_hartree'],
                               row['vqe_gap_hartree'], atol=1e-12, rtol=0)
    require(-1e-10 <= row['vqe_gap_hartree'] < 1e-8, 'VQE gap failed')
    require(0 <= row['sector_leakage'] < 1e-12, 'Electron/spin-projection sector leakage')
    require(0 <= row['variance_hartree2'] < 1e-10, 'Energy variance failed')
    require(abs(row['vqe_spin_squared']) < 1e-10, 'VQE is not a singlet')
    require(abs(row['fci_spin_squared']) < 1e-10, 'FCI is not a singlet')
    require(abs(row['vqe_particle_number']-2) < 1e-12, 'Incorrect electron number')
    require(row['rhf_hartree'] >= row['fci_hartree']-1e-9, 'RHF variational bound failed')
    occupations = np.asarray(row['fci_natural_occupations'])
    require(np.all(np.isfinite(occupations)) and np.all(occupations >= -1e-10)
            and np.all(occupations <= 2+1e-10), 'Invalid natural occupations')
    require(abs(float(occupations.sum())-2) < 1e-10, 'Density matrix trace is not two')


def run_scan(distances):
    require(len(distances) > 0, 'At least one bond length is required')
    rows = []
    for distance in distances:
        start = perf_counter()
        row = classical_point(distance); row.update(quantum_point(distance))
        validate_row(row)
        row['elapsed_seconds'] = perf_counter()-start
        rows.append(row)
    return rows


def provenance():
    return dict(python=platform.python_version(), packages={n:version(n) for n in
                ['numpy','scipy','pennylane','autoray','pennylane-lightning','pyscf',
                 'h5py','matplotlib','nbclient','nbformat','ipykernel']},
                energy_convention='Born-Oppenheimer total energy including nuclear repulsion',
                geometry_unit='angstrom', charge=0, multiplicity=1,
                dft='restricted PBE', requested_pyscf_threads=1,
                scf_energy_tolerance_hartree=1e-11, fci_tolerance=1e-12,
                optimizer='bounded scalar, theta in [-pi, pi], xatol=1e-12, maxiter=200',
                basis='sto-3g', frozen_orbitals=0, spin_orbital_order='alpha,beta interleaved',
                measurement='exact analytic expectation, no shots', bohr_angstrom=BOHR_ANGSTROM,
                benchmark_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
