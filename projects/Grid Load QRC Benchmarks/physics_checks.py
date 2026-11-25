"""Physics convergence and logical resource estimates; no accuracy tuning."""
from math import ceil, log
from time import perf_counter
import numpy as np
from qiskit import transpile
from qiskit.quantum_info import Operator, Statevector
from reservoirs import CVReservoir, SpinReservoir
from data_contract import SEEDS, WINDOW


def gaussian_pennylane(reservoir, window):
    import pennylane as qml
    dev = qml.device('default.gaussian', wires=2, shots=None, hbar=2)
    z = reservoir.encode(np.asarray(window)[None])[0]
    @qml.qnode(dev)
    def circuit(index):
        for row in z:
            for q in range(2):
                qml.Displacement(float(row[q]), 0., wires=q)
                qml.Rotation(float(reservoir.phi[q]), wires=q)
            qml.Beamsplitter(reservoir.theta, reservoir.phase, wires=[0, 1])
        if index < 4:
            observable = qml.QuadX(index//2) if index % 2 == 0 else qml.QuadP(index//2)
        else:
            observable = qml.NumberOperator(index-4)
        return qml.expval(observable)
    # This legacy Gaussian device accepts one observable per execution.
    return np.array([circuit(i) for i in range(6)], float)


def resource_report(windows):
    # Fixed evenly spaced inputs inspect physical convergence, never choose a
    # forecasting hyperparameter. Include all seeds and all data partitions.
    probes = windows[np.linspace(0, len(windows)-1, 24, dtype=int)]
    cv, spin = [], []
    for seed in SEEDS:
        for kerr in (0., .2):
            base = CVReservoir(seed, kerr=kerr, cutoff=10)
            high = CVReservoir(seed, kerr=kerr, cutoff=14)
            f, diag = base.features(probes); reference, _ = high.features(probes)
            error = float(np.max(abs(f-reference)))
            if error > 1e-6: raise ArithmeticError('Fock cutoff convergence failed')
            row = {'seed': seed, 'kerr': kerr, **diag,
                   'max_feature_error_cutoff10_vs14': error}
            if kerr == 0:
                row['pennylane_feature_error'] = float(np.max(abs(f[0]-gaussian_pennylane(base, probes[0]))))
                if row['pennylane_feature_error'] > 1e-8:
                    raise ArithmeticError('PennyLane Gaussian reference mismatch')
            cv.append(row)
        reservoir = SpinReservoir(seed)
        exact = reservoir.evolve(probes[:1])[0]
        qiskit = Statevector.from_instruction(reservoir.qiskit_circuit(probes[0])).data
        error = float(np.linalg.norm(exact-qiskit))
        if error > 1e-10: raise ArithmeticError('Qiskit exact dynamics mismatch')
        convergence = []
        for slices in (1, 2, 4, 8):
            circuit = reservoir.qiskit_circuit(probes[0], slices=slices)
            state = Statevector.from_instruction(circuit).data
            compiled = transpile(circuit, basis_gates=['u', 'cx'], optimization_level=0, seed_transpiler=19)
            convergence.append({'slices_per_input': slices, 'state_error_l2': float(np.linalg.norm(state-exact)),
                                'qubits': compiled.num_qubits, 'depth': compiled.depth(),
                                'u': int(compiled.count_ops().get('u', 0)),
                                'cx': int(compiled.count_ops().get('cx', 0))})
        if convergence[-1]['state_error_l2'] >= convergence[0]['state_error_l2']:
            raise ArithmeticError('Product-formula convergence failed')
        spin.append({'seed': seed, 'exact_qiskit_state_error_l2': error,
                     'hamiltonian_hermiticity_error': float(np.linalg.norm(reservoir.hamiltonian-reservoir.hamiltonian.conj().T)),
                     'strang_product_formula': convergence})
    feature_count = 33; epsilon = .02; delta = .05
    shots = ceil(2*log(2*feature_count/delta)/epsilon**2)
    return {'cv_validation': cv, 'spin_validation': spin,
            'cv': {'modes': 2, 'cutoff': 10, 'hilbert_dimension': 100, 'features': 6,
                   'complex128_state_bytes': 16*100, 'dense_fixed_unitary_bytes': 16*100**2,
                   'gates_per_window_gaussian': WINDOW*5, 'gates_per_window_kerr': WINDOW*7,
                   'measurement_settings': 'X, P and number: three idealized settings; no certified CV shot bound'},
            'spin': {'qubits': 6, 'hilbert_dimension': 64, 'features': feature_count,
                     'complex128_state_bytes': 16*64, 'dense_fixed_unitary_bytes': 16*64**2,
                     'exact_evolution_time_per_window_dimensionless': .2*WINDOW,
                     'measurement_settings': 3, 'epsilon_per_pauli': epsilon, 'joint_failure_probability': delta,
                     'hoeffding_shots_per_setting_per_forecast': shots,
                     'total_reprepared_windows_per_forecast': 3*shots,
                     'bound_assumptions': 'Independent ideal +/-1 Pauli outcomes; union bound across all 33 features. No noise, latency, routing or calibration model.'},
            'scaling': {'fock_state_bytes_cutoff10_modes1_to6': [16*10**m for m in range(1, 7)],
                        'spin_state_bytes_qubits6_9_16_24': {str(n): 16*2**n for n in (6, 9, 16, 24)},
                        'warning': 'State payload lower bounds only; dense operators scale quadratically in Hilbert dimension. Batch arrays and ML memory are additional.'}}
