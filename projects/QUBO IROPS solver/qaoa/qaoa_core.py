"""Diagonal-QUBO QAOA with exact adjoint gradients and independent circuits.

Ideal statevector optimization, finite-shot final sampling; no hardware claim.
The full 128-variable airline chunks are intentionally rejected by this backend.
"""
from time import perf_counter
import numpy as np
import dimod
from scipy.optimize import minimize
from bootstrap import PARENT
from data_contract import integer


class QUBOStatevector:
    mixer_name="transverse_X"
    beta_period=np.pi

    def __init__(self, bqm, max_qubits=20):
        max_qubits = integer(max_qubits, 'simulation qubit cap', 1, 20)
        if bqm.vartype is not dimod.BINARY: raise ValueError('QAOA requires a binary QUBO')
        self.labels = tuple(bqm.variables); self.n = len(self.labels)
        if not 1 <= self.n <= max_qubits: raise ValueError('QUBO exceeds explicit statevector qubit budget')
        self.index = {v:i for i,v in enumerate(self.labels)}; self.bqm = bqm.copy()
        self.states = np.arange(1 << self.n, dtype=np.uint32)
        self.bits = np.stack([(self.states >> i) & 1 for i in range(self.n)], axis=1).astype(np.uint8)
        energy = np.full(len(self.states), float(bqm.offset))
        for v,c in bqm.linear.items(): energy += float(c)*self.bits[:,self.index[v]]
        for (u,v),c in bqm.quadratic.items(): energy += float(c)*self.bits[:,self.index[u]]*self.bits[:,self.index[v]]
        h,j,offset = bqm.to_ising()
        if not np.isfinite(energy).all(): raise ValueError('Nonfinite QUBO spectrum')
        # dimod uses s=2x-1; computational Z has eigenvalue 1-2x.
        self.z = {self.index[v]:-float(c) for v,c in h.items() if c}
        self.zz = {(self.index[u],self.index[v]):float(c) for (u,v),c in j.items() if c}
        self.offset = float(offset)
        self.scale = max(1.,sum(abs(c) for c in self.z.values())+sum(abs(c) for c in self.zz.values()))
        if not np.isfinite(self.scale): raise ValueError('Rescale overflowing QUBO coefficients')
        self.energy = energy; self.cost = (energy-self.offset)/self.scale

    def parameters(self, values):
        values = np.asarray(values, dtype=float)
        if values.ndim != 1 or len(values) < 2 or len(values)%2 or not np.isfinite(values).all():
            raise ValueError('Expected finite gamma/beta vectors of equal nonzero length')
        return values, len(values)//2

    def mixer(self, state, beta):
        state = np.array(state, dtype=complex, copy=True)
        cosine = np.cos(beta); sine = -1j*np.sin(beta)
        for bit in range(self.n):
            view = state.reshape(-1,2,1 << bit)
            a = view[:,0,:].copy(); b = view[:,1,:].copy()
            view[:,0,:] = cosine*a+sine*b; view[:,1,:] = sine*a+cosine*b
        return state

    def mixer_generator(self, state):
        result = np.zeros_like(state)
        for bit in range(self.n): result += state[self.states ^ (1 << bit)]
        return result

    def initial_state(self):
        return np.full(1 << self.n,1/np.sqrt(1 << self.n),dtype=complex)

    def evolve(self, params, gradient=False):
        params,p = self.parameters(params); gammas = params[:p]; betas = params[p:]
        state = self.initial_state(); trajectory = []
        for gamma,beta in zip(gammas,betas):
            cost_state = np.exp(-1j*gamma*self.cost)*state
            state = self.mixer(cost_state,beta)
            if gradient: trajectory.append((cost_state,state))
        norm = float(np.vdot(state,state).real)
        if abs(norm-1) > 1e-10: raise ArithmeticError('QAOA unitary lost normalization')
        expectation = float(np.vdot(state,self.cost*state).real)
        if not gradient: return state,expectation
        adjoint = self.cost*state; grad = np.zeros(2*p)
        for layer in range(p-1,-1,-1):
            cost_state,mixed = trajectory[layer]
            grad[p+layer] = 2*np.vdot(adjoint,-1j*self.mixer_generator(mixed)).real
            adjoint = self.mixer(adjoint,-betas[layer])
            grad[layer] = 2*np.vdot(adjoint,-1j*self.cost*cost_state).real
            adjoint = np.exp(1j*gammas[layer]*self.cost)*adjoint
        if not np.isfinite(grad).all(): raise ArithmeticError('Nonfinite QAOA gradient')
        return expectation,grad

    def physical_energy(self, params):
        state,_ = self.evolve(params)
        return float(np.dot(np.abs(state)**2,self.energy))

    def sample(self, params, shots, seed):
        shots = integer(shots,'shots',1,10000000); seed = integer(seed,'sampling seed',0,2**32-1)
        state,_ = self.evolve(params); probabilities = np.abs(state)**2
        probabilities /= probabilities.sum()
        counts = np.random.default_rng(seed).multinomial(shots,probabilities)
        return {int(i):int(counts[i]) for i in np.flatnonzero(counts)},probabilities

    def sample_dict(self, state):
        state = integer(state,'computational basis index',0,(1 << self.n)-1)
        return {v:int(self.bits[state,j]) for j,v in enumerate(self.labels)}

    def hamiltonian(self):
        """Physical QUBO energy as sparse Pauli Z/ZZ, including the offset."""
        from qiskit.quantum_info import SparsePauliOp
        terms=[('I'*self.n,self.offset)]
        for bit,value in self.z.items():
            label=['I']*self.n;label[self.n-1-bit]='Z';terms.append((''.join(label),value))
        for (a,b),value in self.zz.items():
            label=['I']*self.n;label[self.n-1-a]=label[self.n-1-b]='Z';terms.append((''.join(label),value))
        return SparsePauliOp.from_list(terms)

    def circuit(self, params):
        from qiskit import QuantumCircuit
        params,p = self.parameters(params); circuit = QuantumCircuit(self.n)
        circuit.h(range(self.n))
        for gamma,beta in zip(params[:p],params[p:]):
            for bit,value in self.z.items(): circuit.rz(2*gamma*value/self.scale,bit)
            for (a,b),value in self.zz.items(): circuit.rzz(2*gamma*value/self.scale,a,b)
            for bit in range(self.n): circuit.rx(2*beta,bit)
        return circuit


def optimize(model, depth, seed, starts=3, maxiter=60, previous=None):
    depth = integer(depth,'QAOA depth',1,8); seed = integer(seed,'optimizer seed',0,2**32-1)
    starts = integer(starts,'optimizer starts',1,100); maxiter = integer(maxiter,'optimizer iterations',1,10000)
    tick = perf_counter(); rng = np.random.default_rng(seed); initial = []
    bounds = [(0,4*np.pi)]*depth+[(0,model.beta_period)]*depth
    if previous is not None:
        old,p = model.parameters(previous)
        if p != depth-1: raise ValueError('Continuation must come from the immediately preceding depth')
        if np.any(old[:p]<0) or np.any(old[:p]>4*np.pi) or np.any(old[p:]<0) or np.any(old[p:]>model.beta_period): raise ValueError('Continuation angles lie outside the declared search window')
        initial.append(np.r_[old[:p],0.,old[p:],0.])
    while len(initial) < starts:
        initial.append(np.r_[rng.uniform(0,4*np.pi,depth),rng.uniform(0,model.beta_period,depth)])
    best_value = np.inf; best = None; records = []; evaluations = 0
    def objective(x):
        nonlocal evaluations
        evaluations += 1
        return model.evolve(x,gradient=True)
    for x in initial:
        value,_ = model.evolve(x,gradient=True)
        if value < best_value: best_value,best = value,x.copy()
        result = minimize(objective,x,jac=True,method='L-BFGS-B',bounds=bounds,
                          options={'maxiter':maxiter,'ftol':1e-12,'gtol':1e-7,'maxls':30})
        if np.isfinite(result.fun) and result.fun < best_value: best_value,best = float(result.fun),result.x.copy()
        records.append({'success':bool(result.success),'status':int(result.status),'message':str(result.message),
                        'iterations':int(result.nit),'evaluations':int(result.nfev),'normalized_energy':float(result.fun)})
    return best,{'depth':depth,'parameters':best.tolist(),'expected_energy':model.physical_energy(best),
        'normalized_energy':best_value,'normalization_scale':model.scale,'optimization_evaluations':evaluations,
        'starts':records,'seconds':perf_counter()-tick,
        'training':'Exact statevector expectations and analytic adjoint gradients; no exact optimum used to initialize or terminate optimization',
        'mixer':model.mixer_name,'beta_period':float(model.beta_period),
        'angle_domain':'Gamma [0,4pi] is a declared search window, not a claimed period for real-valued costs; beta uses the declared mixer period'}
