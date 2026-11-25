"""Exact positive-sign QFT with an explicit little-endian register contract."""
from __future__ import annotations
import math
import numbers
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator

MAX_WIDTH=6
MAX_CIRCUIT_QUBITS=21

def integer(value,name,lower=0,upper=2**31-1):
    if isinstance(value,bool) or not isinstance(value,numbers.Integral) or not lower<=value<=upper:
        raise ValueError(f'{name} must be an integer in [{lower}, {upper}]')
    return int(value)


def qft(n, *, inverse=False, cutoff=None):
    """F[y,x]=exp(2πixy/2**n)/sqrt(2**n); wire 0 is LSB.

    Swaps are included. cutoff retains controlled phases with wire distance
    <= cutoff; None means exact. Inverting an approximate circuit is not a
    test of its agreement with the ideal Fourier transform.
    """
    n=integer(n,'QFT width',1,MAX_WIDTH+4)
    if not isinstance(inverse,bool):raise ValueError('inverse must be Boolean')
    if cutoff is not None:cutoff=integer(cutoff,'phase distance',0,n-1)
    qc=QuantumCircuit(n,name='F')
    for target in reversed(range(n)):
        qc.h(target)
        for control in reversed(range(target)):
            distance=target-control
            if cutoff is None or distance<=cutoff:
                qc.cp(math.pi/2**distance,control,target)
    for low in range(n//2):qc.swap(low,n-1-low)
    return qc.inverse() if inverse else qc


def fourier_matrix(n):
    n=integer(n,'dense reference width',1,MAX_WIDTH)
    x=np.arange(2**n)
    return np.exp(2j*np.pi*np.outer(x,x)/2**n)/np.sqrt(2**n)


def phase_sum(qc,wires,k,*,control=None):
    """Fourier-space integer shift using P, not RZ; coherent phases are exact."""
    m=len(wires)
    for j,wire in enumerate(wires):
        angle=2*math.pi*(k%(2**m))*2**j/2**m
        if control is None:qc.p(angle,wire)
        else:qc.cp(angle,control,wire)


def constant_adder(n,k):
    n=integer(n,'width',1,MAX_WIDTH);k=integer(k,'constant',-(2**31),2**31-1)
    qc=QuantumCircuit(n,name=f'+{k}')
    qc.compose(qft(n),inplace=True);phase_sum(qc,list(range(n)),k)
    qc.compose(qft(n,inverse=True),inplace=True)
    return qc


def quantum_adder(n):
    """|a>|b> -> |a>|a+b mod 2**n>; a is the low register."""
    n=integer(n,'width',1,MAX_WIDTH)
    qc=QuantumCircuit(2*n,name='QFT add')
    qc.compose(qft(n),qubits=range(n,2*n),inplace=True)
    for i in range(n):phase_sum(qc,list(range(n,2*n)),2**i,control=i)
    qc.compose(qft(n,inverse=True),qubits=range(n,2*n),inplace=True)
    return qc


def resource_counts(circuit):
    if not isinstance(circuit,QuantumCircuit) or circuit.num_qubits>MAX_CIRCUIT_QUBITS:
        raise ValueError('Resource circuit exceeds the bounded qubit budget')
    compiled=transpile(circuit,basis_gates=['u','cx'],optimization_level=0,seed_transpiler=2718)
    return {'qubits':circuit.num_qubits,'u_count':int(compiled.count_ops().get('u',0)),
        'cx_count':int(compiled.count_ops().get('cx',0)),'depth':compiled.depth(),
        'basis':['u','cx'],'optimization_level':0,'connectivity':'all-to-all; no routing'}


def phase_regression():
    good=constant_adder(3,1)
    bad=QuantumCircuit(3);bad.compose(qft(3),inplace=True)
    for j in range(3):bad.rz(2*math.pi*2**j/8,j)
    bad.compose(qft(3,inverse=True),inplace=True)
    G=Operator(good).data;B=Operator(bad).data
    return {'basis_probability_error':float(np.max(np.abs(np.abs(G)**2-np.abs(B)**2))),
        'controlled_operator_error':float(np.linalg.norm(Operator(good.to_gate().control()).data-Operator(bad.to_gate().control()).data,2)),
        'interpretation':'RZ substitution preserves isolated probabilities but changes the controlled branch phase'}
