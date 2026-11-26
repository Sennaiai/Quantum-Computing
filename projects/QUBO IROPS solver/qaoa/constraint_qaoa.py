"""Same QUBO, one-hot-preserving projector mixers and ordinary slack-X mixing.

Each route/dummy group starts in its one-excitation W state. Disjoint group
projectors commute with one another and with slack-X terms, giving an exact
mixer unitary and adjoint derivative. Flight capacities remain in the QUBO.
"""
import numpy as np
from qaoa_core import QUBOStatevector


class OneHotQAOA(QUBOStatevector):
    mixer_name='onehot_projector_plus_slack_X'
    beta_period=2*np.pi

    def __init__(self,compiled,max_qubits=16):
        super().__init__(compiled.bqm,max_qubits)
        groups=[];seen=set()
        for coeff,constant in compiled.constraints[:len(compiled.dummy_variables)]:
            if constant!=-1 or any(value!=1 for value in coeff.values()):raise ValueError('Mixer requires canonical one-hot groups')
            group=tuple(self.index[v] for v in coeff)
            if not group or seen & set(group):raise ValueError('Mixer groups must be nonempty and disjoint')
            seen.update(group);groups.append(group)
        expected={self.index[v] for v in (*compiled.option_variables,*compiled.dummy_variables)}
        if seen!=expected:raise ValueError('Mixer does not cover all route/unassignment variables')
        self.groups=tuple(groups);self.slack_bits=tuple(sorted(set(range(self.n))-seen));self.blocks=[]
        support=np.ones(len(self.states),bool)
        for group in self.groups:
            mask=sum(1 << bit for bit in group)
            outside=self.states[(self.states & mask)==0]
            block=outside[:,None] | np.array([1 << bit for bit in group],dtype=np.uint32)[None,:]
            self.blocks.append(block)
            support &= self.bits[:,group].sum(axis=1)==1
        self.support=support

    def initial_state(self):
        state=np.zeros(1 << self.n,complex);state[self.support]=1/np.sqrt(self.support.sum())
        return state

    def project(self,state,block):
        projected=np.zeros_like(state);projected[block]=state[block].mean(axis=1)[:,None]
        return projected

    def mixer(self,state,beta):
        state=np.array(state,complex,copy=True)
        for block in self.blocks:state += (np.exp(-1j*beta)-1)*self.project(state,block)
        c=np.cos(beta);s=-1j*np.sin(beta)
        for bit in self.slack_bits:
            view=state.reshape(-1,2,1 << bit);a=view[:,0,:].copy();b=view[:,1,:].copy()
            view[:,0,:]=c*a+s*b;view[:,1,:]=s*a+c*b
        return state

    def mixer_generator(self,state):
        result=np.zeros_like(state)
        for block in self.blocks:result += self.project(state,block)
        for bit in self.slack_bits:result += state[self.states ^ (1 << bit)]
        return result

    def circuit(self,params):
        from qiskit import QuantumCircuit
        from qiskit.circuit.library import StatePreparation
        params,p=self.parameters(params);qc=QuantumCircuit(self.n);preparations=[]
        for group in self.groups:
            w=np.zeros(1 << len(group),complex)
            for bit in range(len(group)):w[1 << bit]=1/np.sqrt(len(group))
            preparation=StatePreparation(w);preparations.append(preparation);qc.append(preparation,list(group))
        for bit in self.slack_bits:qc.h(bit)
        for gamma,beta in zip(params[:p],params[p:]):
            for bit,value in self.z.items():qc.rz(2*gamma*value/self.scale,bit)
            for (a,b),value in self.zz.items():qc.rzz(2*gamma*value/self.scale,a,b)
            for group,preparation in zip(self.groups,preparations):
                qc.append(preparation.inverse(),list(group));qc.x(list(group))
                if len(group)==1:qc.p(-beta,group[0])
                else:qc.mcp(-beta,list(group[:-1]),group[-1])
                qc.x(list(group));qc.append(preparation,list(group))
            for bit in self.slack_bits:qc.rx(2*beta,bit)
        return qc
