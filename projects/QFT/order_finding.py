"""Bounded Shor order finding with explicit oracle provenance and FFT reference."""
from fractions import Fraction
from math import gcd,lcm,isqrt
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit_aer import AerSimulator
from scipy.stats import binomtest
from qft_core import integer,qft
from arithmetic import modular_exponentiation
from verification import execute_state


def order_parameters(a,N,t=None):
    N=integer(N,'modulus',3,31);a=integer(a,'base',2,N-1)
    if N%2==0:raise ValueError('Even composites are handled classically')
    if gcd(a,N)!=1:raise ValueError('Non-coprime bases reveal a classical gcd; no order circuit is needed')
    n=(N-1).bit_length();t=(N*N-1).bit_length() if t is None else integer(t,'counting width',1,9)
    if t>9:raise ValueError('Counting register exceeds the bounded demonstration')
    return a,N,n,t


def multiplier_permutation(a,N):
    """Full unitary extension: multiply residues modulo N, fix every y>=N.

    This is a compiled small-instance comparison, not a scalable arithmetic
    oracle. The native-QFT route is independently implemented in arithmetic.py.
    """
    N=integer(N,'modulus',3,31);a=integer(a,'multiplier',1,N-1)
    if gcd(a,N)!=1:raise ValueError('Multiplier must be a unit modulo N')
    n=(N-1).bit_length()
    permutation=[(a*y)%N if y<N else y for y in range(2**n)]
    if len(set(permutation))!=2**n:raise ValueError('Multiplier must be a permutation')
    return permutation


def compiled_controlled_multiplier(a,N):
    mapping=multiplier_permutation(a,N);dim=len(mapping)
    # control is the first/least-significant wire in the gate's local layout.
    matrix=np.zeros((2*dim,2*dim),complex)
    for y,out in enumerate(mapping):
        matrix[2*y,2*y]=1;matrix[2*out+1,2*y+1]=1
    return UnitaryGate(matrix,label=f'compiled *{a} mod {N}')


def order_circuit(a,N,*,t=None,oracle='qft_arithmetic',cutoff=None):
    a,N,n,t=order_parameters(a,N,t)
    if oracle not in {'qft_arithmetic','compiled_permutation'}:raise ValueError('Unknown order oracle')
    width=t+2*n+2 if oracle=='qft_arithmetic' else t+n
    if width>21:raise ValueError('Order circuit exceeds its 21-qubit budget')
    qc=QuantumCircuit(width,name=f'Order({a},{N})')
    qc.h(range(t))
    if oracle=='qft_arithmetic':
        qc.compose(modular_exponentiation(n,N,a,t),inplace=True)
    else:
        qc.x(t)
        for j in range(t):
            k=pow(a,2**j,N)
            if k!=1:qc.append(compiled_controlled_multiplier(k,N),[j]+list(range(t,t+n)))
    qc.compose(qft(t,inverse=True,cutoff=cutoff),qubits=range(t),inplace=True)
    return qc


def fft_order_reference(a,N,t):
    """Build modular powers directly, then FFT the count axis; no circuit QFT."""
    a,N,n,t=order_parameters(a,N,t);Q=2**t
    state=np.zeros((2**n,Q),complex)
    value=1
    for x in range(Q):
        state[value,x]=1/np.sqrt(Q);value=value*a%N
    return np.fft.fft(state,axis=1)/np.sqrt(Q)


def _reduce_verified_order(a,N,candidate):
    # Remove prime factors from an already verified exponent; do not scan
    # unobserved order guesses. These small integer bounds are explicit.
    result=candidate;factor=2
    while factor*factor<=candidate:
        if candidate%factor==0:
            while candidate%factor==0:candidate//=factor
            while result%factor==0 and pow(a,result//factor,N)==1:result//=factor
        factor+=1
    if candidate>1:
        while result%candidate==0 and pow(a,result//candidate,N)==1:result//=candidate
    return result


def recover_order(counts,a,N,t):
    a,N,n,t=order_parameters(a,N,t)
    if not isinstance(counts,dict) or not counts:raise ValueError('Counts must be a nonempty mapping')
    total=0
    for bits,count in counts.items():
        if not isinstance(bits,str) or len(bits)!=t or set(bits)-{'0','1'}:raise ValueError('Count key has wrong bit order/width')
        total+=integer(count,'count',1,1_000_000)
    if total>1_000_000:raise ValueError('Shot budget exceeded')
    denominators=[];candidates=set()
    # Only observed nonzero phases propose denominators. There is no fallback
    # that determines the order by a blind classical search.
    for bits,count in sorted(counts.items(),key=lambda item:(-item[1],item[0]))[:8]:
        m=int(bits,2)
        if m==0:continue
        frac=Fraction(m,2**t).limit_denominator(N-1)
        d=frac.denominator
        if d>1:
            denominators.append(d);candidates.add(d)
    for left in denominators:
        for right in denominators:
            if 1<lcm(left,right)<N:candidates.add(lcm(left,right))
    valid=[_reduce_verified_order(a,N,r) for r in candidates if 1<r<N and pow(a,r,N)==1]
    r=min(valid) if valid else None
    return {'order':r,'observed_denominators':denominators,'verified_candidates':sorted(set(valid)),
        'rule':'nonzero observed phases -> rational denominators/LCMs -> modular verification; no blind order search'}


def factors_from_order(a,N,r):
    a,N,n,t=order_parameters(a,N,1)
    if r is None:return {'status':'no_verified_order','factors':[]}
    r=integer(r,'order',1,N-1)
    if pow(a,r,N)!=1:raise ValueError('Supplied order is not a verified exponent')
    if r%2:return {'status':'odd_order','factors':[]}
    root=pow(a,r//2,N)
    if root in (1,N-1):return {'status':'trivial_square_root','factors':[]}
    divisors=sorted({gcd(root-1,N),gcd(root+1,N)}-{1,N})
    if len(divisors)==2 and divisors[0]*divisors[1]==N:
        return {'status':'factored','factors':divisors}
    return {'status':'no_factor_pair','factors':divisors}


def run_order_finding(a,N,*,oracle='qft_arithmetic',t=None,cutoff=None,shots=4096,seed=2718):
    a,N,n,t=order_parameters(a,N,t)
    shots=integer(shots,'shots',1,1_000_000);seed=integer(seed,'seed',0,2**32-1)
    circuit=order_circuit(a,N,t=t,oracle=oracle,cutoff=cutoff)
    actual=execute_state(circuit)
    norm_error=float(abs(np.vdot(actual,actual).real-1))
    if norm_error>1e-9:raise ValueError('Order evolution failed norm conservation')
    expected=fft_order_reference(a,N,t).reshape(-1)
    slices=actual.reshape(-1,len(expected))
    leakage=float(np.sum(np.abs(slices[1:])**2))
    error=float(np.linalg.norm(slices[0]-expected))
    if leakage>1e-10:raise ValueError('Arithmetic workspace not clean after order finding')
    if cutoff is None and error>1e-9:raise ValueError(f'Order circuit disagrees with independent modular-power/FFT reference: {error}')
    probabilities=np.sum(np.abs(slices[0].reshape(2**n,2**t))**2,axis=0)
    expected_p=np.sum(np.abs(expected.reshape(2**n,2**t))**2,axis=0)
    # Ideal sampling from the verified final state avoids repeating evolution.
    sample=QuantumCircuit(t+n,t);sample.set_statevector(slices[0]);sample.measure(range(t),range(t))
    counts=AerSimulator(method='statevector',max_parallel_threads=1).run(sample,shots=shots,seed_simulator=seed).result().get_counts()
    counts=dict(sorted(counts.items()))
    observed=np.zeros(2**t)
    for bits,count in counts.items():observed[int(bits,2)]=count/shots
    bound=float(np.sqrt(np.log(2*(2**t)/.001)/(2*shots)))
    if np.max(np.abs(observed-probabilities))>bound:raise ValueError('Shot frequencies exceed the simultaneous per-bin sampling tolerance')
    recovered=recover_order(counts,a,N,t)
    factors=factors_from_order(a,N,recovered['order'])
    return {'a':a,'N':N,'counting_qubits':t,'work_register_qubits':n,'total_qubits':circuit.num_qubits,
        'oracle_kind':oracle,'precision_condition_Q_ge_N_squared':2**t>=N*N,'qft_cutoff':cutoff,
        'state_error_l2_vs_exact_fft':error,'workspace_leakage':leakage,'norm_error':norm_error,
        'phase_probabilities':probabilities.tolist(),'phase_tv_vs_exact_fft':float(.5*np.sum(np.abs(probabilities-expected_p))),
        'shots':shots,'seed':seed,'counts':counts,'recovery':recovered,'factorization':factors,
        'sampling_mode':'Aer readout of validated final state; no hardware run',
        'simultaneous_per_bin_hoeffding_delta':.001,'per_bin_half_width':bound,
        'controlled_power_multipliers':[pow(a,2**j,N) for j in range(t)]}
