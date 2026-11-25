"""Replay actual Qiskit circuit blocks exactly in the invariant one-choice space.

Only complete PNR mixer blocks are restricted; their internal gates may leave
the subspace. This avoids allocating a 2**32 dense statevector without changing
the circuit. Small instances are additionally checked against full Qiskit.
"""
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator


def replay_circuit(model,params):
    qc=model.circuit(params);cursor=0;state=np.ones(model.shape,complex)
    def bits(item):return tuple(qc.find_bit(q).index for q in item.qubits)
    for axis,group in enumerate(model.groups):
        item=qc.data[cursor];cursor+=1
        if bits(item)!=group or item.operation.name!='state_preparation':raise ArithmeticError('Unexpected initialization circuit')
        prepared=Operator(item.operation).data[:,0];one=np.array([1<<q for q in range(len(group))])
        if abs(np.linalg.norm(prepared[one])-1)>1e-10:raise ArithmeticError('Initial state escapes one-choice space')
        shape=[1]*len(model.shape);shape[axis]=len(group);state*=prepared[one].reshape(shape)
    for axis,bit in enumerate(model.slack_bits,start=len(model.groups)):
        item=qc.data[cursor];cursor+=1
        if bits(item)!=(bit,) or item.operation.name!='h':raise ArithmeticError('Unexpected slack initialization')
        shape=[1]*len(model.shape);shape[axis]=2;state*=Operator(item.operation).data[:,0].reshape(shape)
    p=len(params)//2
    for layer in range(p):
        for _ in range(len(model.z)+len(model.zz)):
            item=qc.data[cursor];cursor+=1
            if item.operation.name not in ('rz','rzz'):raise ArithmeticError('Unexpected cost gate')
            local=sum(model.bits[:,bit].astype(np.uint32)<<j for j,bit in enumerate(bits(item)))
            diagonal=np.diag(Operator(item.operation).data)
            state*=diagonal[local].reshape(model.shape)
        for axis,group in enumerate(model.groups):
            local=QuantumCircuit(len(group));mapping={q:j for j,q in enumerate(group)}
            count=2*len(group)+3
            for item in qc.data[cursor:cursor+count]:
                qargs=bits(item)
                if not set(qargs)<=set(group):raise ArithmeticError('Mixer block spans PNRs')
                local.append(item.operation,[mapping[q] for q in qargs])
            cursor+=count;matrix=Operator(local).data;one=np.array([1<<j for j in range(len(group))])
            block=matrix[np.ix_(one,one)]
            if not np.allclose(block.conj().T@block,np.eye(len(group)),atol=1e-10):raise ArithmeticError('Circuit mixer leaks outside invariant space')
            state=np.moveaxis(np.tensordot(block,state,axes=(1,axis)),0,axis)
        for axis,bit in enumerate(model.slack_bits,start=len(model.groups)):
            item=qc.data[cursor];cursor+=1
            if bits(item)!=(bit,) or item.operation.name!='rx':raise ArithmeticError('Unexpected slack mixer')
            matrix=Operator(item.operation).data
            state=np.moveaxis(np.tensordot(matrix,state,axes=(1,axis)),0,axis)
    if cursor!=len(qc.data):raise ArithmeticError('Unverified circuit gates remain')
    return state.reshape(-1)


def verify(model,params):
    independent=replay_circuit(model,params);expected,_=model.evolve(params)
    fidelity=float(abs(np.vdot(independent,expected))**2)
    if abs(fidelity-1)>1e-8:raise ArithmeticError('Qiskit circuit-block replay disagrees with subspace evolution')
    return fidelity
