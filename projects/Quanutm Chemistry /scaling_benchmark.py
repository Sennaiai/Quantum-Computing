"""Active-space accuracy versus Pauli resources for three fixed molecular models.

No full qubit Hamiltonian matrix is allocated. Default exact qubit checks use
only the Nalpha=Nbeta sector (at most 400 determinants). Larger reference
calculations use direct classical CASCI; optional large operators are not solved.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from importlib.metadata import version
from itertools import combinations
from math import comb
from pathlib import Path
import json
import platform
from time import perf_counter

import numpy as np
import pennylane as qml
from pyscf import ao2mo, dft, gto, lib, mcscf, scf
from scipy.sparse import coo_matrix

GEOMETRIES = Path(__file__).with_name('data') / 'scaling_geometries.json'
DEFAULT_SPACES = ((2, 2), (4, 4), (6, 6))
MAX_QUBITS = 12
MAX_SECTOR = 400
MAX_ACTION_ENTRIES = 2_000_000
TOLERANCE = 1e-8


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, tolerance=TOLERANCE, message='Reference mismatch'):
    require(np.all(np.isfinite(actual)) and np.all(np.isfinite(expected)), message)
    require(np.allclose(actual, expected, atol=tolerance, rtol=0), message)


def load_models():
    data = json.loads(GEOMETRIES.read_text())
    require(data['unit'] == 'angstrom', 'Geometry units must be explicit')
    return data['molecules']


def disable_checkpoints(solver):
    """Calculations retain orbitals in memory; close PySCF's unused temp file."""
    checkpoint = getattr(solver, '_chkfile', None)
    solver.chkfile = None
    if checkpoint is not None:
        checkpoint.close()


def validate_space(electrons, orbitals, active_electrons, active_orbitals):
    values = (electrons, orbitals, active_electrons, active_orbitals)
    require(all(isinstance(v, int) and not isinstance(v, bool) for v in values),
            'Electron and orbital counts must be integers')
    require(0 < electrons <= 2*orbitals and electrons % 2 == 0,
            'Only even-electron closed-shell models are supported')
    require(0 < active_electrons <= electrons and active_electrons % 2 == 0,
            'Active electrons must form a positive even subset')
    require(active_electrons <= 2*active_orbitals, 'Active space lacks electron capacity')
    core = (electrons-active_electrons)//2
    require(core+active_orbitals <= orbitals, 'Frozen and active orbitals exceed the basis')
    return list(range(core)), list(range(core, core+active_orbitals))


@dataclass
class MolecularModel:
    name: str
    molecule: object
    rhf: object
    baseline: dict


def build_model(name):
    models = load_models()
    require(name in models, 'Unknown molecular model')
    spec = models[name]
    coords = np.asarray(spec['coordinates_angstrom'], dtype=float)
    require(coords.shape == (len(spec['symbols']), 3) and np.all(np.isfinite(coords)),
            'Invalid geometry array')
    distances = np.linalg.norm(coords[:, None]-coords[None, :], axis=2)
    require(np.all(distances[np.triu_indices(len(coords), 1)] > .1), 'Overlapping nuclei')
    lib.num_threads(1)
    mol = gto.M(atom=list(zip(spec['symbols'], coords)), unit='Angstrom',
                charge=spec['charge'], spin=spec['multiplicity']-1,
                basis='sto-3g', verbose=0)
    require((mol.nelectron, mol.nao) ==
            (spec['expected_electrons'], spec['expected_spatial_orbitals']),
            'Molecule charge/basis/electron count mismatch')
    hf = scf.RHF(mol)
    disable_checkpoints(hf)
    hf.conv_tol = 1e-11
    hf.max_cycle = 100
    hf.kernel()
    if not hf.converged:
        raise RuntimeError(f'{name}: RHF did not converge')
    pbe = dft.RKS(mol)
    disable_checkpoints(pbe)
    pbe.xc = 'pbe'
    pbe.grids.level = 5
    pbe.conv_tol = 1e-11
    pbe.max_cycle = 100
    pbe.kernel()
    if not pbe.converged:
        raise RuntimeError(f'{name}: PBE did not converge')
    require(np.all(np.isfinite([hf.e_tot, pbe.e_tot])), 'Nonfinite SCF reference')
    baseline = dict(name=name, electrons=int(mol.nelectron), spatial_orbitals=int(mol.nao),
                    full_space_qubits=2*int(mol.nao), charge=spec['charge'], multiplicity=1,
                    coordinates_angstrom=coords.tolist(), symbols=spec['symbols'],
                    geometry_provenance=spec['provenance'], rhf_hartree=float(hf.e_tot),
                    rks_pbe_hartree=float(pbe.e_tot), pbe_grid_level=5,
                    nuclear_repulsion_hartree=float(mol.energy_nuc()),
                    mo_energies_hartree=hf.mo_energy.tolist())
    return MolecularModel(name, mol, hf, baseline)


def pbe_grid_control(model):
    fine = dft.RKS(model.molecule)
    disable_checkpoints(fine)
    fine.xc = 'pbe'
    fine.grids.level = 7
    fine.conv_tol = 1e-11
    fine.max_cycle = 100
    fine.kernel()
    if not fine.converged:
        raise RuntimeError(f'{model.name}: fine-grid PBE did not converge')
    difference = float(fine.e_tot-model.baseline['rks_pbe_hartree'])
    require(np.isfinite(difference) and abs(difference) < 1e-6,
            'PBE grid5-to7 control failed')
    return dict(levels=[5, 7], difference_hartree=difference, tolerance_hartree=1e-6)


def cas_reference(model, active_electrons, active_orbitals):
    core, active = validate_space(int(model.molecule.nelectron), int(model.molecule.nao),
                                  active_electrons, active_orbitals)
    cas = mcscf.CASCI(model.rhf, active_orbitals, (active_electrons//2,)*2)
    cas.fcisolver.conv_tol = 1e-12
    cas.fcisolver.max_cycle = 100
    cas.kernel()
    if not cas.converged:
        raise RuntimeError(f'{model.name}: CASCI({active_electrons},{active_orbitals}) failed')
    spin = float(cas.fcisolver.spin_square(cas.ci, active_orbitals,
                                         (active_electrons//2,)*2)[0])
    require(np.isfinite(cas.e_tot) and abs(spin) < 1e-8, 'CASCI reference is not a finite singlet')
    return cas, dict(active_electrons=active_electrons, active_orbitals=active_orbitals,
                     core_indices=core, active_indices=active, casci_hartree=float(cas.e_tot),
                     casci_spin_squared=spin, casci_converged=bool(cas.converged))


def map_integrals(core, one, two_chemist):
    one, two = np.asarray(one), np.asarray(two_chemist)
    n = one.shape[0]
    require(one.shape == (n, n) and two.shape == (n, n, n, n), 'Invalid integral tensor shapes')
    require(np.isfinite(core) and np.all(np.isfinite(one)) and np.all(np.isfinite(two)),
            'Nonfinite integrals')
    close(one, one.T.conj(), message='One-electron tensor is not Hermitian')
    fermionic = qml.qchem.fermionic_observable(np.array([core]), one, two.swapaxes(1, 3), cutoff=1e-12)
    operator = qml.qchem.qubit_observable(fermionic, cutoff=1e-12)
    return fermionic, operator


def pauli_records(operator, qubits):
    records = []
    for word, coefficient in operator.pauli_rep.items():
        coefficient = complex(coefficient)
        require(np.isfinite(coefficient) and abs(coefficient.imag) < 1e-10,
                'Hamiltonian coefficients must be finite and real')
        if abs(coefficient.real) <= 1e-12:
            continue
        labels = ['I']*qubits
        for wire, label in word.items():
            require(isinstance(wire, int) and 0 <= wire < qubits, 'Unexpected wire index')
            labels[wire] = label
        records.append((''.join(labels), float(coefficient.real)))
    return sorted(records)


def qwc_groups(words):
    """Deterministic first-fit qubit-wise commuting measurement settings.

    This counts compatible Pauli bases, not optimal groups, shots, runtime, or
    hardware gates. Sort by weight then lexicographically for reproducibility.
    """
    groups = []
    for word in sorted(words, key=lambda w: (-sum(p != 'I' for p in w), w)):
        for group in groups:
            if all(a == 'I' or b == 'I' or a == b for a, b in zip(group['basis'], word)):
                group['basis'] = ''.join(b if a == 'I' else a for a, b in zip(group['basis'], word))
                group['words'].append(word)
                break
        else:
            groups.append(dict(basis=word, words=[word]))
    return groups


def resource_metrics(operator, qubits):
    records = pauli_records(operator, qubits)
    nonidentity = [(word, c) for word, c in records if set(word) != {'I'}]
    groups = qwc_groups([word for word, _ in nonidentity])
    weights = [sum(p != 'I' for p in word) for word, _ in nonidentity]
    l1 = sum(abs(c) for _, c in nonidentity)
    return dict(pauli_terms=len(records), nonidentity_terms=len(nonidentity),
                identity_coefficient_hartree=sum(c for word, c in records if set(word) == {'I'}),
                nonidentity_l1_hartree=float(l1), max_pauli_weight=max(weights, default=0),
                weight_histogram={str(w): weights.count(w) for w in sorted(set(weights))},
                greedy_qwc_settings=len(groups))


def determinant_expectation(operator, occupied_bits):
    """Computational-basis expectation without a statevector or dense matrix."""
    bits = list(occupied_bits)
    require(all(bit in (0, 1) for bit in bits), 'Occupations must be binary')
    energy = 0.
    for word, coefficient in pauli_records(operator, len(bits)):
        if any(label in 'XY' for label in word):
            continue
        energy += coefficient*np.prod([1-2*bit for label, bit in zip(word, bits) if label == 'Z'])
    return float(energy)


def sector_basis(electrons, orbitals):
    require(2*orbitals <= MAX_QUBITS, 'Exact qubit checks are limited to 12 qubits')
    validate_space(electrons, orbitals, electrons, orbitals)
    require(comb(orbitals, electrons//2)**2 <= MAX_SECTOR, 'Sector determinant budget exceeded')
    alpha = [sum(1 << (2*orbitals-1-2*i) for i in inds)
             for inds in combinations(range(orbitals), electrons//2)]
    beta = [sum(1 << (2*orbitals-2-2*i) for i in inds)
            for inds in combinations(range(orbitals), electrons//2)]
    return np.array(sorted(a | b for a in alpha for b in beta), dtype=int)


def sector_matrix(operator, electrons, orbitals):
    """Apply Pauli strings to selected determinants; never allocate a full H.

    Sparse action also tests cancellation of couplings out of the physical sector.
    Guard the basis and sparse-entry budget BEFORE allocating any large arrays.
    """
    basis = sector_basis(electrons, orbitals)
    n = 2*orbitals
    records = pauli_records(operator, n)
    require(len(records)*len(basis) <= MAX_ACTION_ENTRIES, 'Sparse-action budget exceeded')
    rows, cols, values = [], [], []
    for word, coefficient in records:
        destination = basis.copy()
        phase = np.full(len(basis), complex(coefficient))
        for wire, label in enumerate(word):
            bit = (basis >> (n-1-wire)) & 1
            if label in 'XY':
                destination ^= 1 << (n-1-wire)
            if label == 'Y':
                phase *= 1j*(1-2*bit)
            if label == 'Z':
                phase *= 1-2*bit
        rows.append(destination)
        cols.append(np.arange(len(basis)))
        values.append(phase)
    action = coo_matrix((np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
                        shape=(2**n, len(basis))).tocsr()
    outside = np.ones(2**n, dtype=bool)
    outside[basis] = False
    leaked = action[outside].data
    require(np.max(np.abs(leaked), initial=0) < 1e-9, 'Hamiltonian leaves Nalpha=Nbeta sector')
    block = action[basis].toarray()
    close(block, block.T.conj(), message='Projected matrix is not Hermitian')
    return basis, block


def build_case(model, electrons, orbitals, allow_large=False):
    validate_space(int(model.molecule.nelectron), int(model.molecule.nao), electrons, orbitals)
    require(orbitals <= 6 or (allow_large and orbitals <= 8),
            'Large operators require explicit opt-in; maximum 8 spatial orbitals')
    start = perf_counter()
    cas, info = cas_reference(model, electrons, orbitals)
    one, core = cas.get_h1eff()
    two = ao2mo.restore(1, cas.get_h2eff(), orbitals)
    t = perf_counter()
    fermionic, operator = map_integrals(float(core), one, two)
    construction = perf_counter()-t
    row = dict(**info, qubits=2*orbitals, fermionic_terms=len(fermionic),
               core_constant_hartree=float(core), **resource_metrics(operator, 2*orbitals),
               integral_to_pauli_seconds=construction)
    hf_bits = [1]*electrons+[0]*(2*orbitals-electrons)
    hf_energy = determinant_expectation(operator, hf_bits)
    close(hf_energy, model.rhf.e_tot, message='Mapped HF determinant differs from RHF')
    row['hf_qubit_hartree'] = hf_energy
    if orbitals <= 6:
        basis, block = sector_matrix(operator, electrons, orbitals)
        eigenvalues, eigenvectors = np.linalg.eigh(block)
        energy = float(eigenvalues[0])
        close(energy, cas.e_tot, message='Pauli-sector exact energy differs from CASCI')
        hf_index = int('1'*electrons+'0'*(2*orbitals-electrons), 2)
        hf_position = int(np.searchsorted(basis, hf_index))
        close(block[hf_position, hf_position].real, model.rhf.e_tot,
              message='Mapped HF determinant differs from RHF')
        spin = qml.qchem.spin2(electrons, 2*orbitals)
        _, spin_block = sector_matrix(spin, electrons, orbitals)
        spin_squared = float(np.vdot(eigenvectors[:, 0], spin_block @ eigenvectors[:, 0]).real)
        require(abs(spin_squared) < 1e-8, 'Qubit-sector ground state is not a singlet')
        row.update(sector_dimension=len(basis), sector_exact_hartree=energy,
                   sector_exact_spin_squared=spin_squared,
                   hf_qubit_hartree=float(block[hf_position, hf_position].real))
    else:
        row.update(qubit_diagonalization='not performed: exceeds exact-check budget')
    row['active_integrals'] = dict(core_energy_hartree=float(core), one_body_hartree=one.tolist(),
                                   two_body_chemist_hartree=two.tolist())
    row['elapsed_seconds'] = perf_counter()-start
    return row, operator


def reference_space(model):
    return (10, 8) if model.name == 'ammonia' else (8, 8)


def validate_hierarchy(cases, reference):
    previous = None
    for row in cases:
        require(row['casci_hartree'] >= reference['casci_hartree']-TOLERANCE,
                'Active-space energy below the containing reference')
        if previous is not None:
            require(set(previous['active_indices']).issubset(row['active_indices']),
                    'Spaces are not nested in the same fixed orbital basis')
            require(row['casci_hartree'] <= previous['casci_hartree']+TOLERANCE,
                    'Nested variational hierarchy increased in energy')
        require(set(row['active_indices']).issubset(reference['active_indices']),
                'Classical reference does not contain the tested active space')
        previous = row


def run_study(profile='default'):
    require(profile in {'default', 'smoke'}, 'Unknown scaling profile')
    results = []
    for name in load_models():
        model = build_model(name)
        _, reference = cas_reference(model, *reference_space(model))
        reference['scope'] = ('full STO-3G FCI' if name == 'ammonia'
                              else 'larger CAS(8,8), not full-basis FCI')
        cases = []
        for electrons, orbitals in DEFAULT_SPACES if profile == 'default' else DEFAULT_SPACES[:2]:
            row, _ = build_case(model, electrons, orbitals)
            row['error_vs_larger_reference_hartree'] = row['casci_hartree']-reference['casci_hartree']
            cases.append(row)
        validate_hierarchy(cases, reference)
        results.append(dict(baseline=model.baseline, reference=reference, cases=cases,
                            pbe_grid_control=pbe_grid_control(model)))
    return dict(profile=profile, provenance=provenance(), molecules=results)


def provenance():
    return dict(python=platform.python_version(), packages={n: version(n) for n in
                ['numpy', 'scipy', 'pennylane', 'autoray', 'pyscf', 'matplotlib', 'nbclient', 'nbformat']},
                source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
                geometries_sha256=sha256(GEOMETRIES.read_bytes()).hexdigest(),
                geometry_unit='angstrom', basis='sto-3g', orbital_basis='fixed canonical RHF per molecule',
                energy_convention='total Born-Oppenheimer energy including nuclear repulsion',
                integral_backend='PySCF; CASCI and qubit solvers share integral tensors',
                integral_cutoff=1e-12, pauli_cutoff=1e-12,
                grouping='deterministic weight-first, lexicographic first-fit QWC; not optimal/shots',
                max_exact_qubits=MAX_QUBITS, max_sector_determinants=MAX_SECTOR,
                max_sparse_action_entries=MAX_ACTION_ENTRIES,
                scf_tolerance=1e-11, casci_tolerance=1e-12,
                requested_pyscf_threads=1, dft='restricted PBE, grid level5')
