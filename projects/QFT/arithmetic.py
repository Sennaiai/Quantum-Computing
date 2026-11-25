"""Reversible QFT arithmetic with clean-workspace contracts on valid residues.

Register order is little-endian within each listed register. Modular adders use
one overflow and one flag qubit, both initially zero. They promise clean output
for 0 <= target < N; behavior outside that subspace is not residue arithmetic.
All circuits remain unitary on the full Hilbert space.
"""
from math import gcd
from qiskit import QuantumCircuit
from qft_core import integer,qft,phase_sum

MAX_ARITHMETIC_WIDTH=5

def parameters(n,N):
    n=integer(n,'data width',1,MAX_ARITHMETIC_WIDTH)
    N=integer(N,'modulus',2,2**n)
    return n,N


def modular_constant_adder(n,N,k):
    """|b>|00> -> |b+k mod N>|00>, 0<=b<N; final two wires are work."""
    n,N=parameters(n,N);k=integer(k,'constant',-(2**31),2**31-1)%N
    qc=QuantumCircuit(n+2,name=f'+{k} mod {N}')
    if k==0:return qc
    data=list(range(n+1));sign=n;flag=n+1
    F=qft(n+1);Finv=F.inverse()
    qc.compose(F,qubits=data,inplace=True)
    phase_sum(qc,data,k);phase_sum(qc,data,-N)
    qc.compose(Finv,qubits=data,inplace=True);qc.cx(sign,flag)
    qc.compose(F,qubits=data,inplace=True);phase_sum(qc,data,N,control=flag)
    phase_sum(qc,data,-k)
    qc.compose(Finv,qubits=data,inplace=True)
    qc.x(sign);qc.cx(sign,flag);qc.x(sign)
    qc.compose(F,qubits=data,inplace=True);phase_sum(qc,data,k)
    qc.compose(Finv,qubits=data,inplace=True)
    return qc


def modular_quantum_adder(n,N):
    """|a>|b>|00> -> |a>|a+b mod N>|00>, with b<N (a arbitrary n-bit)."""
    n,N=parameters(n,N);qc=QuantumCircuit(2*n+2,name='modular quantum add')
    for i in range(n):
        gate=modular_constant_adder(n,N,2**i).to_gate().control(1)
        qc.append(gate,[i]+list(range(n,2*n+2)))
    return qc


def outplace_constant_adder(n,N,k):
    """|a>|0>|00> -> |a>|a+k mod N>|00>, a<N; copy is basis-value CNOT."""
    n,N=parameters(n,N);qc=QuantumCircuit(2*n+2,name='outplace constant add')
    for i in range(n):qc.cx(i,n+i)
    qc.compose(modular_constant_adder(n,N,k),qubits=range(n,2*n+2),inplace=True)
    return qc


def outplace_quantum_adder(n,N):
    """Preserve a,b and write their sum in initially-zero result; b<N."""
    n,N=parameters(n,N);qc=QuantumCircuit(3*n+2,name='outplace quantum add')
    for i in range(n):qc.cx(n+i,2*n+i)
    qc.compose(modular_quantum_adder(n,N),qubits=list(range(n))+list(range(2*n,3*n+2)),inplace=True)
    return qc


def multiply_accumulate(n,N,k):
    """|a>|b>|00> -> |a>|b+k*a mod N>|00>, b<N. k need not be invertible."""
    n,N=parameters(n,N);k=integer(k,'constant',-(2**31),2**31-1)%N
    qc=QuantumCircuit(2*n+2,name=f'MAC {k} mod {N}')
    for i in range(n):
        qc.append(modular_constant_adder(n,N,k*2**i).to_gate().control(1),[i]+list(range(n,2*n+2)))
    return qc


def modular_multiplier(n,N,k):
    """|a>|0>|00> -> |k*a mod N>|0>|00>, a<N and gcd(k,N)=1."""
    n,N=parameters(n,N);k=integer(k,'multiplier',1,2**31-1)%N
    if gcd(k,N)!=1:raise ValueError('In-place multiplication requires a unit modulo N')
    inverse=pow(k,-1,N)  # valid for composite N; no Fermat-prime assumption
    qc=QuantumCircuit(2*n+2,name=f'*{k} mod {N}')
    if k==1:return qc
    qc.compose(multiply_accumulate(n,N,k),inplace=True)
    for i in range(n):qc.swap(i,n+i)
    qc.compose(multiply_accumulate(n,N,-inverse),inplace=True)
    return qc


def quantum_multiply_accumulate(n,N):
    """|a>|b>|r>|00> -> |a>|b>|r+a*b mod N>|00>, r<N."""
    n,N=parameters(n,N);qc=QuantumCircuit(3*n+2,name='quantum MAC')
    for j in range(n):
        gate=multiply_accumulate(n,N,2**j).to_gate().control(1)
        qc.append(gate,[n+j]+list(range(n))+list(range(2*n,3*n+2)))
    return qc


def modular_exponentiation(n,N,a,t):
    """|x>|0>|0>|00> -> |x>|a**x mod N>|0>|00>, gcd(a,N)=1.

    Low t wires hold x, next n hold result, next n support, last two are work.
    Only controlled modular multipliers are used, not exponent truth tables.
    """
    n,N=parameters(n,N);t=integer(t,'exponent width',1,9)
    a=integer(a,'base',2,N-1)
    if gcd(a,N)!=1:raise ValueError('Exponentiation base must be coprime to N')
    if t+2*n+2>21:raise ValueError('Exponentiation exceeds 21-qubit simulation budget')
    qc=QuantumCircuit(t+2*n+2,name='QFT modular exponentiation');qc.x(t)
    work=list(range(t,t+2*n+2))
    for j in range(t):
        k=pow(a,2**j,N)
        if k!=1:qc.append(modular_multiplier(n,N,k).to_gate().control(1),[j]+work)
    return qc
