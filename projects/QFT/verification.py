"""Independent semantic permutation references and bounded coherent circuit probes."""
import numpy as np
from qiskit import QuantumCircuit,transpile
from qiskit_aer import AerSimulator
from qft_core import MAX_CIRCUIT_QUBITS,integer


def execute_state(circuit, initial=None):
    if circuit.num_qubits>MAX_CIRCUIT_QUBITS or circuit.num_clbits or circuit.parameters:
        raise ValueError('State probe exceeds its fully bound unitary budget')
    backend=AerSimulator(method='statevector',max_parallel_threads=1)
    prepared=QuantumCircuit(circuit.num_qubits)
    if initial is not None:
        vector=np.asarray(initial,dtype=complex)
        if vector.shape!=(2**circuit.num_qubits,) or not np.isfinite(vector).all() or not np.isclose(np.linalg.norm(vector),1,atol=1e-12,rtol=0):
            raise ValueError('Initial state must be a normalized vector of the full register size')
        prepared.set_statevector(vector)
    prepared.compose(circuit,inplace=True)
    compiled=transpile(prepared,backend,optimization_level=1,seed_transpiler=2718)
    compiled.save_statevector()
    return np.asarray(backend.run(compiled,seed_simulator=2718).result().get_statevector())


def coherent_probe(circuit,mapping,*,seed=1234,tolerance=1e-9):
    """Probe all declared basis inputs with distinct complex amplitudes.

    mapping is a separately constructed semantic input->output permutation on
    the promised subspace; zero workspace is explicit in the output indices.
    """
    if not mapping or len(set(mapping.values()))!=len(mapping):raise ValueError('Reference mapping must be injective')
    dim=2**circuit.num_qubits
    if circuit.num_qubits>MAX_CIRCUIT_QUBITS:raise ValueError('Coherent probe exceeds memory budget')
    indices=sorted(mapping)
    if min(indices)<0 or max(max(indices),max(mapping.values()))>=dim:raise ValueError('Reference index out of range')
    rng=np.random.default_rng(seed)
    amplitudes=rng.normal(size=len(indices))+1j*rng.normal(size=len(indices));amplitudes/=np.linalg.norm(amplitudes)
    initial=np.zeros(dim,complex);expected=np.zeros(dim,complex)
    initial[indices]=amplitudes
    for source,amp in zip(indices,amplitudes):expected[mapping[source]]=amp
    actual=execute_state(circuit,initial)
    error=float(np.linalg.norm(actual-expected))
    output=set(mapping.values())
    mask=np.ones(dim,dtype=bool);mask[list(output)]=False
    leakage=float(np.sum(np.abs(actual[mask])**2))
    if error>tolerance:raise ValueError(f'Coherent arithmetic reference error {error}')
    return {'valid_basis_inputs':len(indices),'full_register_qubits':circuit.num_qubits,
        'state_error_l2':error,'outside_promised_output_probability':max(0.,leakage)}
