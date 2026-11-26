"""Exact invariant-subspace simulation of the existing one-choice QAOA circuit.

No amplitudes are truncated. One route/dummy per PNR is conserved exactly;
every slack bit remains a quantum degree of freedom in the unchanged QUBO.
"""
import math
import numpy as np
from qaoa_core import QUBOStatevector,optimize
from constraint_qaoa import OneHotQAOA
from data_contract import integer


class SubspaceQAOA:
    mixer_name=OneHotQAOA.mixer_name
    beta_period=OneHotQAOA.beta_period
    parameters=QUBOStatevector.parameters
    evolve=QUBOStatevector.evolve
    physical_energy=QUBOStatevector.physical_energy
    circuit=OneHotQAOA.circuit

    def __init__(self,compiled,max_qubits=36,max_states=262144):
        self.labels=tuple(compiled.bqm.variables);self.n=len(self.labels)
        if not 1<=self.n<=integer(max_qubits,'logical qubit cap',1,40):raise ValueError('Logical qubit budget exceeded')
        self.index={v:i for i,v in enumerate(self.labels)};self.bqm=compiled.bqm.copy()
        groups=[];seen=set()
        for coeff,constant in compiled.constraints[:len(compiled.dummy_variables)]:
            if constant!=-1 or any(c!=1 for c in coeff.values()):raise ValueError('Canonical one-choice groups required')
            group=tuple(self.index[v] for v in coeff)
            if not group or seen.intersection(group):raise ValueError('Groups must be nonempty and disjoint')
            groups.append(group);seen.update(group)
        expected={self.index[v] for v in (*compiled.option_variables,*compiled.dummy_variables)}
        if seen!=expected:raise ValueError('Incomplete one-choice mixer coverage')
        self.groups=tuple(groups);self.slack_bits=tuple(sorted(set(range(self.n))-seen))
        self.shape=tuple(len(g) for g in self.groups)+(2,)*len(self.slack_bits)
        self.dimension=math.prod(self.shape)
        if self.dimension>integer(max_states,'subspace budget',1,2**22):raise ValueError('Reachable subspace budget exceeded')
        grid=np.indices(self.shape,dtype=np.uint64).reshape(len(self.shape),-1)
        self.states=np.zeros(self.dimension,dtype=np.uint64)
        for axis,group in enumerate(self.groups):self.states |= np.array([1<<q for q in group],np.uint64)[grid[axis]]
        for axis,bit in enumerate(self.slack_bits,start=len(self.groups)):self.states |= grid[axis]<<bit
        self.bits=((self.states[:,None]>>np.arange(self.n,dtype=np.uint64))&1).astype(np.uint8)
        self.energy=np.asarray(self.bqm.energies((self.bits,self.labels)),dtype=float)
        h,j,offset=self.bqm.to_ising();self.offset=float(offset)
        self.z={self.index[v]:-float(c) for v,c in h.items() if c}
        self.zz={(self.index[u],self.index[v]):float(c) for (u,v),c in j.items() if c}
        self.scale=max(1.,sum(abs(c) for c in self.z.values())+sum(abs(c) for c in self.zz.values()))
        if not np.isfinite(self.energy).all() or not np.isfinite(self.scale):raise ValueError('Nonfinite subspace Hamiltonian')
        self.cost=(self.energy-self.offset)/self.scale

    def initial_state(self):return np.full(self.dimension,1/np.sqrt(self.dimension),complex)

    def mixer(self,state,beta):
        view=np.array(state,complex,copy=True).reshape(self.shape)
        for axis in range(len(self.groups)):
            view += (np.exp(-1j*beta)-1)*view.mean(axis=axis,keepdims=True)
        for axis in range(len(self.groups),len(self.shape)):
            view=np.cos(beta)*view-1j*np.sin(beta)*np.flip(view,axis=axis)
        return view.reshape(-1)

    def mixer_generator(self,state):
        view=state.reshape(self.shape);result=np.zeros(self.shape,complex)
        for axis in range(len(self.groups)):result+=view.mean(axis=axis,keepdims=True)
        for axis in range(len(self.groups),len(self.shape)):result+=np.flip(view,axis=axis)
        return result.reshape(-1)

    def draw(self,params,shots,seed):
        shots=integer(shots,'shots',1,10000000);seed=integer(seed,'seed',0,2**32-1)
        state,_=self.evolve(params);prob=abs(state)**2;prob/=prob.sum()
        counts=np.random.default_rng(seed).multinomial(shots,prob)
        return {int(self.states[i]):int(counts[i]) for i in np.flatnonzero(counts)}

    def basis_sample(self,basis):
        basis=integer(basis,'physical basis index',0,(1<<self.n)-1)
        return {v:(basis>>i)&1 for i,v in enumerate(self.labels)}
