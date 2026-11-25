"""Bounded, little-endian Qiskit circuits with independent amplitude references."""
from __future__ import annotations
from itertools import combinations, product
import math
import numbers
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

MAX_SYSTEM_QUBITS = 15
MAX_TOTAL_QUBITS = 16
MAX_ITERATIONS = 128


def integer(value, name, lower=0, upper=MAX_ITERATIONS):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or not lower <= value <= upper:
        raise ValueError(f'{name} must be an integer in [{lower}, {upper}]')
    return int(value)


def targets_checked(n, targets):
    integer(n, 'system qubits', 1, MAX_SYSTEM_QUBITS)
    if not isinstance(targets, (list, tuple)):
        raise ValueError('targets must be an explicit list or tuple')
    if any(not isinstance(t, str) or len(t) != n or set(t)-{'0','1'} for t in targets):
        raise ValueError('targets must be binary strings of exactly n bits')
    if len(set(targets)) != len(targets):
        raise ValueError('Duplicate targets would cancel phase flips; use a set of distinct targets')
    if len(targets) > 256:
        raise ValueError('Lookup oracle exceeds its explicit target budget')
    return tuple(sorted(targets))


def phase_flip(qc, wires, bitstring):
    """Flip one basis phase; strings list most significant bit first."""
    for wire, bit in zip(wires, reversed(bitstring)):
        if bit == '0': qc.x(wire)
    if len(wires) == 1: qc.z(wires[0])
    else:
        qc.h(wires[-1]); qc.mcx(wires[:-1], wires[-1]); qc.h(wires[-1])
    for wire, bit in zip(wires, reversed(bitstring)):
        if bit == '0': qc.x(wire)


def oracle_for_targets(n, targets):
    targets = targets_checked(n, targets)
    qc = QuantumCircuit(n, name='Lookup phase oracle')
    for target in targets: phase_flip(qc, list(range(n)), target)
    return qc


def uniform_preparation(n):
    integer(n, 'system qubits', 1, MAX_SYSTEM_QUBITS)
    qc = QuantumCircuit(n, name='A'); qc.h(range(n))
    return qc


def validate_preparation(preparation):
    if not isinstance(preparation, QuantumCircuit) or not 1 <= preparation.num_qubits <= MAX_SYSTEM_QUBITS:
        raise ValueError('A must be a bounded QuantumCircuit')
    if preparation.num_clbits or preparation.parameters:
        raise ValueError('A must be a fully bound unitary circuit without classical bits')
    if any(i.operation.name in {'measure','reset','initialize','if_else','while_loop'} or i.operation.num_clbits for i in preparation.data):
        raise ValueError('A must be reversible and measurement free')
    try: preparation.to_gate().inverse()
    except Exception as exc: raise ValueError('A must have a unitary inverse') from exc


def diffusion(preparation):
    """Exactly D_A=2|psi><psi|-I, including its global sign."""
    validate_preparation(preparation)
    n = preparation.num_qubits
    qc = QuantumCircuit(n, name='D_A')
    qc.compose(preparation.inverse(), inplace=True)
    phase_flip(qc, list(range(n)), '0'*n)
    qc.compose(preparation, inplace=True)
    qc.global_phase += math.pi
    return qc


def build_amplification(preparation, oracle, iterations):
    validate_preparation(preparation)
    integer(iterations, 'iterations')
    n = preparation.num_qubits
    if not isinstance(oracle, QuantumCircuit) or not n <= oracle.num_qubits <= MAX_TOTAL_QUBITS or oracle.num_clbits or oracle.parameters:
        raise ValueError('Oracle must be a bound unitary on system plus bounded workspace')
    try: oracle.to_gate().inverse()
    except Exception as exc: raise ValueError('Oracle must be reversible') from exc
    qc = QuantumCircuit(oracle.num_qubits, name='Amplitude amplification')
    qc.compose(preparation, qubits=range(n), inplace=True)
    reflection = diffusion(preparation).to_gate()
    phase_oracle = oracle.to_gate()
    for _ in range(iterations):
        qc.append(phase_oracle, range(oracle.num_qubits))
        qc.append(reflection, range(n))
    return qc


def analytic_probability(p, iterations):
    integer(iterations, 'iterations', upper=10**7)
    if isinstance(p, bool) or not isinstance(p, numbers.Real) or not math.isfinite(p) or not 0 <= p <= 1:
        raise ValueError('Initial success probability must be finite and in [0,1]')
    if p in (0,1): return float(p)
    return float(math.sin((2*iterations+1)*math.asin(math.sqrt(p)))**2)


def first_peak_iterations(p):
    """Best integer adjacent to the first continuous peak; ties use fewer queries."""
    analytic_probability(p,0)
    if p in (0,1): return 0
    peak = max(0., math.pi/(4*math.asin(math.sqrt(p)))-.5)
    if not math.isfinite(peak) or peak > 10**7-1:
        raise ValueError('First peak exceeds the supported query budget')
    lo = math.floor(peak)
    candidates = (lo,lo+1)
    values = [analytic_probability(p,k) for k in candidates]
    return lo if values[0] >= values[1]-1e-14 else lo+1


def dense_reference(psi, targets, iterations):
    """Vector reflection reference: no Qiskit oracle or diffuser is reused."""
    vector = np.asarray(psi,dtype=complex)
    if vector.ndim!=1 or len(vector)<2 or len(vector)&(len(vector)-1) or len(vector)>2**MAX_SYSTEM_QUBITS:
        raise ValueError('Reference state must have a bounded power-of-two dimension')
    if not np.isfinite(vector).all() or not np.isclose(np.linalg.norm(vector),1,atol=1e-12,rtol=0):
        raise ValueError('Reference state must be finite and normalized')
    integer(iterations,'iterations')
    n = len(vector).bit_length()-1
    indices = [int(t,2) for t in targets_checked(n,targets)]
    result = vector.copy()
    for _ in range(iterations):
        result[indices] *= -1
        result = 2*vector*np.vdot(vector,result)-result
    return result


def simulate(preparation, oracle, targets, iterations, *, shots=4096, seed=2718):
    validate_preparation(preparation)
    targets = targets_checked(preparation.num_qubits, targets)
    integer(shots,'shots',1,1_000_000); integer(seed,'seed',0,2**32-1)
    qc = build_amplification(preparation,oracle,iterations)
    n = preparation.num_qubits
    backend = AerSimulator(method='statevector',max_parallel_threads=1)
    compiled = transpile(qc,backend,optimization_level=1,seed_transpiler=seed)
    compiled.save_statevector()
    vector = np.asarray(backend.run(compiled,seed_simulator=seed).result().get_statevector())
    by_workspace = vector.reshape(-1,2**n)
    leakage = float(np.sum(np.abs(by_workspace[1:])**2))
    if leakage > 1e-12: raise ValueError('Oracle workspace did not return to zero')
    initial = Statevector.from_instruction(preparation).data
    reference = dense_reference(initial,targets,iterations)
    error = float(np.linalg.norm(by_workspace[0]-reference))
    if error > 1e-9: raise ValueError('Circuit differs from the independent vector reference')
    indices = [int(t,2) for t in targets]
    p0 = min(1., max(0., float(np.sum(np.abs(initial[indices])**2))))  # float64 endpoint roundoff
    exact = float(np.sum(np.abs(by_workspace[0,indices])**2))
    expected = analytic_probability(p0,iterations)
    if abs(exact-expected)>1e-10: raise ValueError('Circuit violates the two-dimensional rotation law')
    measured = qc.copy(); measured.add_register(__import__('qiskit').ClassicalRegister(n))
    measured.measure(range(n),range(n))
    counts = backend.run(transpile(measured,backend,optimization_level=1,seed_transpiler=seed),shots=shots,seed_simulator=seed).result().get_counts()
    good = sum(counts.get(t,0) for t in targets)
    half_width = math.sqrt(math.log(2/.001)/(2*shots))
    if abs(good/shots-exact)>half_width: raise ValueError('Sample exceeds the declared Hoeffding tolerance')
    return {'iterations':iterations,'initial_success':p0,'analytic_success':expected,'exact_success':exact,
        'sampled_success':good/shots,'shots':shots,'seed':seed,'counts':dict(sorted(counts.items())),
        'hoeffding_delta':.001,'hoeffding_half_width':half_width,'state_error_l2':error,
        'workspace_leakage':leakage,'system_qubits':n,'workspace_qubits':qc.num_qubits-n,
        'phase_oracle_calls':iterations,'A_calls':iterations+1,'A_inverse_calls':iterations}, qc


def valid_board(cols):
    if not isinstance(cols,(list,tuple)) or not cols:
        return False
    if any(isinstance(c,bool) or not isinstance(c,numbers.Integral) or not 0 <= c < len(cols) for c in cols):
        return False
    return all(cols[i]!=cols[j] and abs(cols[i]-cols[j])!=j-i for i,j in combinations(range(len(cols)),2))


def queens_reference(size):
    integer(size,'board size',1,5)
    boards = [cols for cols in product(range(size),repeat=size) if valid_board(cols)]
    bits = max(1,(size-1).bit_length())
    targets = [''.join(format(c,f'0{bits}b') for c in cols) for cols in boards]
    nodes, attempts, backtracking = 0,0,[]
    def visit(prefix):
        nonlocal nodes, attempts
        nodes+=1
        if len(prefix)==size: backtracking.append(tuple(prefix)); return
        for col in range(size):
            attempts+=1
            if all(col!=old and abs(col-old)!=len(prefix)-r for r,old in enumerate(prefix)):
                visit(prefix+[col])
    visit([])
    if boards!=backtracking: raise ValueError('Independent classical solvers disagree')
    n = size*bits; p = len(targets)/2**n
    k = first_peak_iterations(p)
    return {'size':size,'bits_per_row':bits,'system_qubits':n,'binary_search_space':2**n,
        'in_range_assignments':size**size,'marked_count':len(targets),'targets':targets,
        'boards':[list(b) for b in boards],'initial_success':p,'first_peak_iterations':k,
        'analytic_peak_success':analytic_probability(p,k),'backtracking_nodes':nodes,
        'backtracking_column_attempts':attempts,'classical_enumeration_candidates':size**size}


def queens4_constraint_oracle():
    """Compute six pair-conflict flags, phase valid boards, uncompute all flags.

    No complete solutions are enumerated. Each pair predicate has four input
    bits and flips its flag for equal-column or diagonal conflicts.
    """
    qc = QuantumCircuit(14,name='4-Queens constraint oracle')
    for a,(i,j) in enumerate(combinations(range(4),2)):
        wires = [2*(3-i),2*(3-i)+1,2*(3-j),2*(3-j)+1]
        for ci,cj in product(range(4),repeat=2):
            if ci==cj or abs(ci-cj)==j-i:
                bits = format(ci,'02b')[::-1]+format(cj,'02b')[::-1]
                for wire,bit in zip(wires,bits):
                    if bit=='0':qc.x(wire)
                qc.mcx(wires,8+a)
                for wire,bit in zip(wires,bits):
                    if bit=='0':qc.x(wire)
    compute = qc.copy()
    phase_flip(qc,list(range(8,14)),'000000')
    qc.compose(compute.inverse(),inplace=True)
    return qc


def resource_counts(circuit):
    if circuit.num_qubits>MAX_TOTAL_QUBITS: raise ValueError('Resource compilation budget exceeded')
    native = transpile(circuit,basis_gates=['u','cx'],optimization_level=0,seed_transpiler=2718)
    return {'qubits':circuit.num_qubits,'u_count':int(native.count_ops().get('u',0)),
        'cx_count':int(native.count_ops().get('cx',0)),'depth':native.depth(),
        'basis':['u','cx'],'optimization_level':0,'routing':'none; all-to-all logical connectivity'}


def classical_success(p, queries):
    analytic_probability(p,0);integer(queries,'classical queries',0,10**7)
    return -math.expm1(queries*math.log1p(-p)) if 0<p<1 else float(p if queries else 0)
