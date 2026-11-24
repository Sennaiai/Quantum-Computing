"""Fixed-orbital HCN CAS(4,4) benchmark with explicit integral and qubit stages.

All energies are nonrelativistic Born–Oppenheimer totals including nuclear
repulsion. PySCF supplies converged orbitals/integrals and the CASCI solver;
qubit diagonalization checks mapping/solver consistency, not an independent
integral engine. Dense matrices and analytic expectations are used on 8 qubits.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
import platform
from time import perf_counter

from autograd import value_and_grad
import numpy as np
import pennylane as qml
from pyscf import ao2mo, dft, gto, lib, mcscf, scf, symm
from scipy.optimize import minimize


DEFAULT_CN_DISTANCES = (1.067, 1.167, 1.267)
DEFAULT_SEEDS = (0, 1, 2)
ACTIVE_ELECTRONS = 4
ACTIVE_ORBITALS = 4
WIRES = tuple(range(8))
REFERENCE_TOLERANCE = 1e-8


def require(condition, message):
    """Keep scientific checks active under optimized Python."""
    if not condition:
        raise ValueError(message)


def require_close(actual, expected, tolerance, message):
    require(np.all(np.isfinite(actual)) and np.all(np.isfinite(expected)),
            f"Nonfinite values: {message}")
    require(np.allclose(actual, expected, atol=tolerance, rtol=0), message)


def geometry_angstrom(r_cn=1.167, r_hc=1.065):
    """Return linear H-C-N coordinates with public bond lengths in angstrom."""
    require(np.isfinite(r_cn) and r_cn > 0, "C-N length must be finite and positive")
    require(np.isfinite(r_hc) and r_hc > 0, "H-C length must be finite and positive")
    return np.array([[0., 0., 0.], [0., 0., r_hc], [0., 0., r_hc + r_cn]])


def sector_indices():
    """Nalpha=Nbeta=2 with MSB wire order 0..7, alpha/beta interleaved."""
    return np.array([
        i for i in range(256)
        if sum(map(int, f"{i:08b}"[::2])) == 2
        and sum(map(int, f"{i:08b}"[1::2])) == 2
    ], dtype=int)


@dataclass
class HCNProblem:
    """Intermediate construction artifacts are retained for notebook inspection."""
    coordinates: np.ndarray
    one_body: np.ndarray
    two_body_chemist: np.ndarray
    two_body_physicist: np.ndarray
    core_energy: float
    fermionic: object
    hamiltonian: object
    matrix: np.ndarray
    sector: np.ndarray
    number_matrix: np.ndarray
    spin2_matrix: np.ndarray
    reference: dict


def map_active_integrals(core_energy, one_body, two_body_chemist):
    """Map PySCF spatial-orbital integrals through fermions to Jordan–Wigner.

    PySCF's restored ERIs use chemist axes. PennyLane 0.36's
    fermionic_observable expects tensor[p,q,r,s] = chemist[p,s,r,q],
    multiplying a_p^dagger a_q^dagger a_r a_s with its internal 1/2 factor.
    The scalar core energy already includes nuclear repulsion and all frozen
    occupied contributions; add it exactly once.
    """
    one = np.asarray(one_body)
    chemist = np.asarray(two_body_chemist)
    require(one.shape == (4, 4) and chemist.shape == (4, 4, 4, 4),
            "CAS(4,4) requires 4x4 and 4x4x4x4 spatial tensors")
    require(np.isfinite(core_energy) and np.all(np.isfinite(one))
            and np.all(np.isfinite(chemist)), "Nonfinite active integrals")
    require_close(one, one.T.conj(), 1e-10, "One-body tensor is not Hermitian")
    physicist = chemist.swapaxes(1, 3)
    fermionic = qml.qchem.fermionic_observable(np.array([core_energy]), one, physicist)
    hamiltonian = qml.qchem.qubit_observable(fermionic)
    matrix = np.asarray(qml.matrix(hamiltonian, wire_order=WIRES))
    require_close(matrix, matrix.T.conj(), 1e-10, "Mapped Hamiltonian is not Hermitian")
    return physicist, fermionic, hamiltonian, matrix


def build_problem(r_cn=1.167, r_hc=1.065, grid_level=5):
    """Construct and validate converged classical references and the active model."""
    coordinates = geometry_angstrom(r_cn, r_hc)
    require(isinstance(grid_level, int) and 0 <= grid_level <= 9,
            "DFT grid level must be an integer from 0 to 9")
    start = perf_counter()
    lib.num_threads(1)
    molecule = gto.M(
        atom=list(zip(["H", "C", "N"], coordinates)), unit="Angstrom",
        basis="sto-3g", charge=0, spin=0, symmetry="C2v", verbose=0,
    )
    mean_field = scf.RHF(molecule)
    mean_field.conv_tol = 1e-11
    mean_field.max_cycle = 100
    mean_field.kernel()
    if not mean_field.converged:
        raise RuntimeError(f"RHF did not converge at C-N={r_cn} angstrom")
    require(molecule.nelectron == 14 and molecule.nao == 11,
            "Unexpected HCN/STO-3G electron or spatial-orbital count")
    labels = list(symm.label_orb_symm(
        molecule, molecule.irrep_name, molecule.symm_orb, mean_field.mo_coeff,
    ))
    require(set(labels[5:7]) == {"B1", "B2"}
            and set(labels[7:9]) == {"B1", "B2"},
            "Active orbital ordering changed: expected occupied and virtual pi pairs")
    require_close(mean_field.mo_occ[5:9], [2, 2, 0, 0], 1e-12,
                  "Active orbital occupations changed")

    casci = mcscf.CASCI(mean_field, ACTIVE_ORBITALS, (2, 2))
    casci.fcisolver.conv_tol = 1e-12
    casci.kernel()
    if not casci.converged:
        raise RuntimeError(f"CASCI did not converge at C-N={r_cn} angstrom")
    require(casci.ncore == 5, "Expected five frozen doubly occupied spatial orbitals")
    casci_spin2 = float(casci.fcisolver.spin_square(casci.ci, 4, (2, 2))[0])
    require(abs(casci_spin2) < 1e-8, "CASCI reference is not a singlet")

    one_body, core_energy = casci.get_h1eff()
    two_chemist = ao2mo.restore(1, casci.get_h2eff(), ACTIVE_ORBITALS)
    two_physicist, fermionic, hamiltonian, matrix = map_active_integrals(
        float(core_energy), one_body, two_chemist,
    )
    sector = sector_indices()
    outside = np.setdiff1d(np.arange(256), sector)
    require_close(matrix[np.ix_(outside, sector)], 0, 1e-10,
                  "Hamiltonian leaves Nalpha=Nbeta=2 sector")
    eigenvalues, eigenvectors = np.linalg.eigh(matrix[np.ix_(sector, sector)])
    sector_exact = float(eigenvalues[0])
    number = np.diag([i.bit_count() for i in range(256)])
    spin2 = np.asarray(qml.matrix(qml.qchem.spin2(4, 8), wire_order=WIRES))
    require_close(matrix @ number, number @ matrix, 1e-10,
                  "Hamiltonian does not conserve particle number")
    require_close(matrix @ spin2, spin2 @ matrix, 1e-9,
                  "Hamiltonian does not conserve total spin")
    exact_state = np.zeros(256, dtype=complex)
    exact_state[sector] = eigenvectors[:, 0]
    exact_spin2 = float(np.vdot(exact_state, spin2 @ exact_state).real)
    require(abs(exact_spin2) < 1e-8, "Sector ground state is not a singlet")
    require_close(sector_exact, casci.e_tot, REFERENCE_TOLERANCE,
                  "Qubit-sector energy disagrees with matched CASCI")
    hf_energy = float(matrix[240, 240].real)  # |11110000>, wires0..7
    require_close(hf_energy, mean_field.e_tot, REFERENCE_TOLERANCE,
                  "HF determinant energy disagrees with full-basis RHF")

    pbe = dft.RKS(molecule)
    pbe.xc = "pbe"
    pbe.grids.level = grid_level
    pbe.conv_tol = 1e-11
    pbe.max_cycle = 100
    pbe.kernel()
    if not pbe.converged:
        raise RuntimeError(f"RKS-PBE did not converge at C-N={r_cn} angstrom")
    require(np.all(np.isfinite([mean_field.e_tot, casci.e_tot, pbe.e_tot])),
            "Nonfinite classical reference energy")
    occupations = np.linalg.eigvalsh(
        casci.fcisolver.make_rdm1(casci.ci, 4, (2, 2)),
    )[::-1]
    require_close(occupations.sum(), 4, 1e-10, "Active density-matrix trace is not four")
    reference = dict(
        r_cn_angstrom=float(r_cn), r_hc_angstrom=float(r_hc), basis="sto-3g",
        total_electrons=14, spatial_orbitals=11, full_space_qubits=22,
        active_electrons=4, active_orbitals=4, qubits=8,
        core_indices=list(range(5)), active_indices=list(range(5, 9)),
        excluded_virtual_indices=[9, 10], mo_symmetry_labels=labels,
        mo_energies_hartree=mean_field.mo_energy.tolist(),
        mo_occupations=mean_field.mo_occ.tolist(),
        rhf_hartree=float(mean_field.e_tot), casci_hartree=float(casci.e_tot),
        rks_pbe_hartree=float(pbe.e_tot), pbe_grid_level=grid_level,
        rhf_converged=bool(mean_field.converged), casci_converged=bool(casci.converged),
        rks_converged=bool(pbe.converged), core_energy_hartree=float(core_energy),
        nuclear_repulsion_hartree=float(molecule.energy_nuc()),
        sector_exact_hartree=sector_exact, hf_qubit_hartree=hf_energy,
        sector_dimension=len(sector), casci_spin_squared=casci_spin2,
        sector_exact_spin_squared=exact_spin2,
        active_natural_occupations=occupations.tolist(),
        pauli_terms=len(hamiltonian.terms()[0]), construction_seconds=perf_counter()-start,
    )
    return HCNProblem(
        coordinates, np.asarray(one_body), np.asarray(two_chemist), two_physicist,
        float(core_energy), fermionic, hamiltonian, matrix, sector, number, spin2,
        reference,
    )


def make_circuits(problem, layers=6):
    """GateFabric preserves the HF reference's particle number and total spin.

    With include_pi=True, fixed orbital rotations permute the HF occupations;
    zero trainable weights do not generally produce the original HF state.
    """
    require(isinstance(layers, int) and layers > 0, "Layers must be a positive integer")
    shape = qml.GateFabric.shape(layers, 8)
    device = qml.device("default.qubit", wires=WIRES)
    hf = qml.qchem.hf_state(4, 8)

    @qml.qnode(device, interface="autograd", diff_method="adjoint")
    def energy(weights):
        qml.GateFabric(weights, wires=WIRES, init_state=hf, include_pi=True)
        return qml.expval(qml.Hermitian(problem.matrix, wires=WIRES))

    @qml.qnode(device)
    def state(weights):
        qml.GateFabric(weights, wires=WIRES, init_state=hf, include_pi=True)
        return qml.state()

    return shape, energy, state


def state_diagnostics(problem, state):
    """Independent matrix residual and symmetry diagnostics for a saved state."""
    state = np.asarray(state)
    require(state.shape == (256,) and np.all(np.isfinite(state)), "Invalid statevector")
    require_close(np.vdot(state, state).real, 1, 1e-10, "Statevector is not normalized")
    energy = float(np.vdot(state, problem.matrix @ state).real)
    residual = problem.matrix @ state - energy * state
    outside = np.setdiff1d(np.arange(256), problem.sector)
    return dict(
        best_energy_hartree=energy,
        gap_hartree=energy-problem.reference["sector_exact_hartree"],
        energy_variance_hartree2=float(np.vdot(residual, residual).real),
        particle_number=float(np.vdot(state, problem.number_matrix @ state).real),
        spin_squared=float(np.vdot(state, problem.spin2_matrix @ state).real),
        sector_leakage=float(np.sum(np.abs(state[outside])**2)),
    )


def validate_run(result, require_accuracy=True):
    """Strict results require optimizer convergence and independent accuracy checks.

    Smoke runs may exhaust their budget; their status remains explicit, and
    basic finiteness, variational and symmetry checks are never disabled.
    """
    keys = ["best_energy_hartree", "reference_energy_hartree", "gap_hartree", "energy_variance_hartree2",
            "particle_number", "spin_squared", "sector_leakage"]
    require(np.all(np.isfinite([result[key] for key in keys])), "Nonfinite VQE result")
    require_close(result["best_energy_hartree"]-result["reference_energy_hartree"],
                  result["gap_hartree"], 1e-10, "Reported VQE gap is inconsistent")
    trace = result["energy_trace"]
    require(all(np.isfinite(row["energy_hartree"]) for row in trace), "Nonfinite energy trace")
    require(np.all(np.isfinite(result["best_parameters"])), "Nonfinite saved parameters")
    index = result["best_evaluation"]
    require(isinstance(index, int) and 0 <= index < len(trace), "Invalid checkpoint index")
    require_close(trace[index]["energy_hartree"], result["best_energy_hartree"], 1e-10,
                  "Checkpoint trace energy is inconsistent")
    require_close(min(row["energy_hartree"] for row in trace),
                  result["best_energy_hartree"], 1e-10, "Checkpoint is not the best evaluated state")
    require(result["gap_hartree"] >= -1e-9, "VQE violates the matched-sector bound")
    require(abs(result["particle_number"]-4) < 1e-10, "VQE electron number changed")
    require(abs(result["spin_squared"]) < 1e-8, "VQE total spin changed")
    require(0 <= result["sector_leakage"] < 1e-12, "VQE leaked from the target sector")
    require(result["energy_variance_hartree2"] >= 0, "Negative energy variance")
    if require_accuracy:
        if not result["optimizer_success"]:
            raise RuntimeError(f"Optimizer did not converge: {result['optimizer_message']}")
        require(result["gap_hartree"] < 1e-6, "VQE gap exceeds the solver target")
        require(result["energy_variance_hartree2"] < 1e-7,
                "VQE variance exceeds the solver target")


def optimize_seed(problem, seed=0, layers=6, maxiter=300, initial_scale=.1):
    """Return every optimizer status and a reproducible best-state checkpoint.

    Accuracy validation is separate so callers can export all seeds before
    applying strict acceptance. A nonconverged optimizer never becomes a success
    merely because one low energy was observed.
    """
    require(isinstance(maxiter, int) and maxiter > 0, "maxiter must be positive")
    require(np.isfinite(initial_scale) and initial_scale >= 0, "Invalid initialization scale")
    start = perf_counter()
    shape, energy, state = make_circuits(problem, layers)
    initial = np.random.default_rng(seed).normal(scale=initial_scale, size=shape)
    trace = []
    best = {"energy": float("inf"), "weights": None, "evaluation": None}

    def record(weights, value, stage):
        require(np.isfinite(value) and np.all(np.isfinite(weights)),
                "Optimizer produced nonfinite parameters or energy")
        trace.append(dict(evaluation=len(trace), stage=stage, energy_hartree=float(value)))
        if value < best["energy"]:
            best.update(energy=float(value), weights=np.array(weights, copy=True),
                        evaluation=len(trace)-1)

    initial_energy = float(energy(qml.numpy.array(initial, requires_grad=True)))
    record(initial, initial_energy, "initial")
    value_gradient = value_and_grad(energy)

    def objective(flat):
        weights = qml.numpy.array(np.asarray(flat).reshape(shape), requires_grad=True)
        value, gradient = value_gradient(weights)
        record(weights, value, "optimizer")
        require(np.all(np.isfinite(gradient)), "Optimizer gradient became nonfinite")
        return float(value), np.asarray(gradient).ravel()

    optimized = minimize(
        objective, initial.ravel(), jac=True, method="L-BFGS-B",
        options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-7},
    )
    final_weights = np.asarray(optimized.x).reshape(shape)
    final_energy = float(energy(qml.numpy.array(final_weights, requires_grad=True)))
    record(final_weights, final_energy, "final")
    diagnostics = state_diagnostics(problem, state(best["weights"]))
    require_close(diagnostics["best_energy_hartree"], best["energy"], 1e-10,
                  "Saved best-state parameters do not reproduce their energy")
    result = dict(
        seed=int(seed), layers=layers, parameter_count=int(np.prod(shape)),
        maxiter=maxiter, initial_scale=float(initial_scale), include_pi=True,
        optimizer_success=bool(optimized.success), optimizer_message=str(optimized.message),
        optimizer_iterations=int(optimized.nit), optimizer_evaluations=int(optimized.nfev),
        status="optimizer_converged" if optimized.success else "optimizer_not_converged",
        initial_energy_hartree=initial_energy, final_energy_hartree=final_energy,
        best_parameters=best["weights"].tolist(), best_evaluation=best["evaluation"],
        energy_trace=trace, reference_energy_hartree=problem.reference["sector_exact_hartree"],
        elapsed_seconds=perf_counter()-start, **diagnostics,
    )
    validate_run(result, require_accuracy=False)
    result["accuracy_target_met"] = bool(
        -1e-9 <= result["gap_hartree"] < 1e-6 and result["energy_variance_hartree2"] < 1e-7,
    )
    return result


def run_study(distances=DEFAULT_CN_DISTANCES, seeds=DEFAULT_SEEDS,
              layers=6, maxiter=300):
    """Retain all reference/seed outcomes; callers apply strict acceptance afterward."""
    require(len(distances) > 0 and len(seeds) > 0, "Study requires geometries and seeds")
    rows = []
    for distance in distances:
        problem = build_problem(distance)
        runs = [optimize_seed(problem, seed, layers, maxiter) for seed in seeds]
        # Checkpoint parameters are tied to these orbital phases/order. Retain
        # the small spatial tensors so replay does not depend on a fresh SCF's
        # choice of basis inside degenerate pi-orbital pairs.
        rows.append(dict(
            reference=problem.reference,
            active_integrals=dict(
                core_energy_hartree=problem.core_energy,
                one_body_hartree=problem.one_body.tolist(),
                two_body_chemist_hartree=problem.two_body_chemist.tolist(),
                orbital_indices=list(range(5, 9)),
                tensor_convention="real spatial chemist ERIs (pq|rs)",
            ),
            runs=runs,
        ))
    return rows


def provenance():
    """Record the implementation and numerical settings behind an exported study."""
    return dict(
        python=platform.python_version(), packages={name: version(name) for name in
            ["numpy", "scipy", "pennylane", "autoray", "autograd", "pennylane-lightning",
             "pyscf", "h5py", "matplotlib", "nbclient", "nbformat", "ipykernel"]},
        benchmark_source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        geometry_unit="angstrom", charge=0, multiplicity=1, point_group="C2v",
        basis="sto-3g", active_space="4 electrons in 4 spatial orbitals",
        orbital_order="canonical RHF, core0..4, active5..8, alpha/beta interleaved",
        energy_convention="total Born-Oppenheimer energy including nuclear repulsion",
        reference_scope="same PySCF integrals; independent CASCI and qubit-sector solvers",
        classical_baseline_scope="RHF and RKS-PBE use the full 11-orbital basis",
        dft="restricted PBE, grid level5", scf_energy_tolerance_hartree=1e-11,
        casci_tolerance=1e-12, requested_pyscf_threads=1,
        ansatz="GateFabric, 6layers default, include_pi=True",
        optimizer="L-BFGS-B, exact adjoint gradient, ftol1e-12, gtol1e-7, maxiter300",
        simulation="dense 8-qubit observable, exact statevector, no measurement shots",
    )


def full_fci_reference(problem=None):
    """Reference-geometry full-basis FCI in the A1, Nalpha=Nbeta=7 sector.

    The 330x330 CI coefficient array is not a dense Hamiltonian matrix. This
    direct FCI calculation retains all 11 spatial orbitals and provides an
    active-space truncation control at one geometry, not a dissociation curve.
    """
    from pyscf import fci

    start = perf_counter()
    if problem is None:
        problem = build_problem()
    require_close(problem.reference["r_cn_angstrom"], 1.167, 1e-12,
                  "Full FCI control is restricted to reference C-N geometry")
    require_close(problem.reference["r_hc_angstrom"], 1.065, 1e-12,
                  "Full FCI control is restricted to reference H-C geometry")
    lib.num_threads(1)
    mol = gto.M(
        atom=list(zip(["H", "C", "N"], problem.coordinates)), unit="Angstrom",
        basis="sto-3g", charge=0, spin=0, symmetry="C2v", verbose=0, max_memory=1000,
    )
    hf = scf.RHF(mol)
    hf.conv_tol = 1e-11
    hf.max_cycle = 100
    hf.max_memory = 1000
    hf.kernel()
    if not hf.converged:
        raise RuntimeError("Full-basis control RHF did not converge")
    solver = fci.FCI(hf)
    solver.wfnsym = "A1"
    solver.conv_tol = 1e-10
    solver.max_cycle = 100
    solver.max_memory = 1000
    solver.max_space = 12
    energy, vector = solver.kernel()
    if not solver.converged:
        raise RuntimeError("Full-basis A1 FCI control did not converge")
    spin_squared, multiplicity = solver.spin_square(vector, 11, (7, 7))
    require(np.isfinite(energy) and np.isfinite(spin_squared), "Nonfinite full FCI result")
    require(abs(spin_squared) < 1e-8, "Full FCI A1 state is not a singlet")
    truncation = problem.reference["casci_hartree"]-energy
    require(truncation >= -1e-8, "CASCI is below the full-basis FCI reference")
    return dict(
        r_cn_angstrom=1.167, r_hc_angstrom=1.065, basis="sto-3g",
        full_fci_hartree=float(energy), full_fci_converged=bool(solver.converged),
        full_fci_spin_squared=float(spin_squared), full_fci_multiplicity=float(multiplicity),
        sector="A1, Nalpha=Nbeta=7; singlet verified", ci_vector_shape=list(vector.shape),
        casci_hartree=problem.reference["casci_hartree"],
        casci_minus_full_fci_hartree=float(truncation),
        rhf_hartree=problem.reference["rhf_hartree"],
        rks_pbe_hartree=problem.reference["rks_pbe_hartree"],
        elapsed_seconds=perf_counter()-start,
        resource_limits=dict(requested_memory_mb=1000, max_cycle=100, max_space=12),
    )


def pbe_grid_control():
    """PBE integration-grid check at the reference geometry only."""
    grid3 = build_problem(grid_level=3).reference
    grid5 = build_problem(grid_level=5).reference
    grid7 = build_problem(grid_level=7).reference
    delta = grid7["rks_pbe_hartree"]-grid5["rks_pbe_hartree"]
    require(abs(delta) < 1e-6, "Reference PBE grid5-to7 control exceeds 1e-6 Ha")
    return dict(r_cn_angstrom=1.167, r_hc_angstrom=1.065,
                grid3_hartree=grid3["rks_pbe_hartree"],
                grid5_hartree=grid5["rks_pbe_hartree"],
                grid7_hartree=grid7["rks_pbe_hartree"],
                grid3_to5_hartree=grid5["rks_pbe_hartree"]-grid3["rks_pbe_hartree"],
                accepted_levels=[5, 7], delta_hartree=float(delta))


def run_scan(profile="default"):
    """Bounded notebook profiles; smoke status is not optimization certification."""
    require(profile in {"smoke", "default"}, "Unknown HCN profile")
    smoke = profile == "smoke"
    rows = run_study(
        distances=(1.167,) if smoke else DEFAULT_CN_DISTANCES,
        seeds=(0,) if smoke else DEFAULT_SEEDS,
        maxiter=30 if smoke else 300,
    )
    return dict(profile=profile, provenance=provenance(), scan=rows)
