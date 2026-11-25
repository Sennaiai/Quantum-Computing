"""Native PennyLane core operations; explicit wire-order normalization to Qiskit."""
import math
import numpy as np
import pennylane as qml
from qft_core import integer
from arithmetic import parameters


def fourier_ops(wires):
    # wires are LSB first; qml.matrix/state use wire_order reversed(wires).
    for target in reversed(range(len(wires))):
        qml.Hadamard(wires=wires[target])
        for control in reversed(range(target)):
            qml.ControlledPhaseShift(math.pi/2**(target-control),wires=[wires[control],wires[target]])
    for low in range(len(wires)//2):qml.SWAP(wires=[wires[low],wires[-1-low]])


def pl_qft_matrix(n, *, builtin=False):
    n=integer(n,'PennyLane matrix width',1,5)
    wires=list(range(n))
    if builtin:
        return np.asarray(qml.matrix(qml.QFT(wires=wires[::-1]),wire_order=wires[::-1]),dtype=complex)
    return np.asarray(qml.matrix(fourier_ops,wire_order=wires[::-1])(wires),dtype=complex)


def phase_ops(wires,k,control=None):
    for j,wire in enumerate(wires):
        angle=2*math.pi*(k%2**len(wires))*2**j/2**len(wires)
        if control is None:qml.PhaseShift(angle,wires=wire)
        else:qml.ControlledPhaseShift(angle,wires=[control,wire])


def modular_ops(n,N,k):
    n,N=parameters(n,N);k=integer(k,'constant',-(2**31),2**31-1)%N
    if k==0:return
    wires=list(range(n+1));sign=n;flag=n+1
    fourier_ops(wires);phase_ops(wires,k);phase_ops(wires,-N)
    qml.adjoint(fourier_ops)(wires);qml.CNOT(wires=[sign,flag])
    fourier_ops(wires);phase_ops(wires,N,flag);phase_ops(wires,-k)
    qml.adjoint(fourier_ops)(wires)
    qml.PauliX(wires=sign);qml.CNOT(wires=[sign,flag]);qml.PauliX(wires=sign)
    fourier_ops(wires);phase_ops(wires,k);qml.adjoint(fourier_ops)(wires)


def pl_modular_matrix(n,N,k):
    n,N=parameters(n,N)
    if n>3:raise ValueError('PennyLane dense arithmetic reference is bounded at three data qubits')
    return np.asarray(qml.matrix(modular_ops,wire_order=list(reversed(range(n+2))))(n,N,k),dtype=complex)


def pl_modular_state(n,N,k,initial):
    n,N=parameters(n,N)
    if n>3:raise ValueError('PennyLane state comparison is bounded at three data qubits')
    device=qml.device('default.qubit',wires=list(reversed(range(n+2))),shots=None)
    @qml.qnode(device)
    def execute():
        qml.StatePrep(initial,wires=list(reversed(range(n+2))))
        modular_ops(n,N,k)
        return qml.state()
    return np.asarray(execute(),dtype=complex)
