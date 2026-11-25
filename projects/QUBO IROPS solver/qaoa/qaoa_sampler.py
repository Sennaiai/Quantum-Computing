"""Reusable binary-BQM QAOA sampler; no exact optimum is passed to training."""
import numpy as np
import dimod
from qaoa_core import QUBOStatevector,optimize
from data_contract import integer


class QAOASampler:
    def __init__(self,max_qubits=16):
        self.max_qubits=integer(max_qubits,'sampler qubit cap',1,20)

    def sample(self,bqm,*,num_reads=1024,depth=3,seed=7,starts=5,maxiter=150,previous=None):
        model=QUBOStatevector(bqm,self.max_qubits)
        return self.sample_prepared(model,num_reads=num_reads,depth=depth,seed=seed,starts=starts,maxiter=maxiter,previous=previous)

    def sample_constrained(self,compiled,**kwargs):
        from constraint_qaoa import OneHotQAOA
        return self.sample_prepared(OneHotQAOA(compiled,self.max_qubits),**kwargs)

    def sample_prepared(self,model,*,num_reads=1024,depth=3,seed=7,starts=5,maxiter=150,previous=None):
        depth=integer(depth,'depth',1,8);seed=integer(seed,'seed',0,2**32-1)
        measurement_seed=(seed+1009*depth) % 2**32
        if model.n>self.max_qubits:raise ValueError('Prepared QUBO exceeds sampler budget')
        params,training=optimize(model,depth,seed,starts,maxiter,previous)
        counts,_=model.sample(params,num_reads,measurement_seed)
        indices=list(counts)
        return dimod.SampleSet.from_samples((np.array([model.bits[i] for i in indices],dtype=np.int8),model.labels),
            vartype=dimod.BINARY,energy=[model.energy[i] for i in indices],num_occurrences=[counts[i] for i in indices],
            info={'backend':'ideal QAOA statevector with finite final sampling','training':training,'parameters':params.tolist(),
                  'depth':depth,'optimizer_seed':seed,'measurement_seed':measurement_seed})


def basis_counts(samples,model):
    counts={}
    if set(samples.variables)!=set(model.labels):raise ValueError('Sampler variable labels differ from compiled QUBO')
    for row in samples.data(['sample','num_occurrences']):
        if any(row.sample[v] not in (0,1) for v in model.labels):raise ValueError('Nonbinary sampler output')
        index=sum(int(row.sample[v]) << q for q,v in enumerate(model.labels))
        counts[index]=counts.get(index,0)+int(row.num_occurrences)
    return counts
