"""Source-bound Fourier, reversible arithmetic, and Shor execution evidence."""
from pathlib import Path
import hashlib,json
from importlib.metadata import version
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from qiskit.synthesis import adder_ripple_c04
from qft_core import qft,fourier_matrix,constant_adder,quantum_adder,resource_counts,phase_regression
from arithmetic import (modular_constant_adder,modular_quantum_adder,outplace_constant_adder,
    outplace_quantum_adder,multiply_accumulate,quantum_multiply_accumulate,modular_multiplier,modular_exponentiation)
from cross_backend import pl_qft_matrix,pl_modular_matrix,pl_modular_state
from verification import coherent_probe
from order_finding import run_order_finding,order_circuit

ROOT=Path(__file__).resolve().parent
NOTEBOOKS=('QFT_Arithmetic_Validation.ipynb','Shor_Senay_Kahsay.ipynb')

def source_hashes():
    paths=[*ROOT.glob('*.py'),*(ROOT/name for name in NOTEBOOKS),ROOT/'requirements.txt']
    return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def environment():
    return {name:version(name) for name in ('numpy','scipy','qiskit','qiskit-aer','pennylane','matplotlib')}


def run_arithmetic_experiments():
    before=source_hashes();fourier=[]
    for n in range(1,6):
        F=fourier_matrix(n);Q=Operator(qft(n)).data
        fourier.append({'qubits':n,'qiskit_matrix_error_2':float(np.linalg.norm(Q-F,2)),
            'pennylane_matrix_error_2':float(np.linalg.norm(pl_qft_matrix(n)-F,2)),
            'pennylane_builtin_error_2':float(np.linalg.norm(pl_qft_matrix(n,builtin=True)-F,2))})
    if max(max(r[k] for k in r if k.endswith('error_2')) for r in fourier)>1e-11:
        raise ValueError('Cross-framework Fourier matrix acceptance failed')
    approximation=[];n=6;F=fourier_matrix(n)
    for cutoff in range(n):
        circuit=qft(n,cutoff=cutoff);matrix=Operator(circuit).data
        approximation.append({'qubits':n,'cutoff':cutoff,'retained_controlled_phases':int(circuit.count_ops().get('cp',0)),
            'operator_error_2':float(np.linalg.norm(matrix-F,2)),
            'roundtrip_error_2':float(np.linalg.norm(matrix.conj().T@matrix-np.eye(2**n),2)),
            'resources':resource_counts(circuit)})
    coherence=[]
    for n,N,k in [(2,3,2),(3,5,3),(3,7,5),(3,8,7)]:
        circuit=modular_constant_adder(n,N,k)
        run=coherent_probe(circuit,{b:(b+k)%N for b in range(N)})
        run.update(operation='modular_constant_add',data_width=n,N=N,k=k,
            pennylane_full_operator_error_2=float(np.linalg.norm(Operator(circuit).data-pl_modular_matrix(n,N,k),2)))
        coherence.append(run)
    n,N=3,7
    configurations=[
        ('outplace_constant_add',outplace_constant_adder(n,N,5),{a:a+(((a+5)%N)<<n) for a in range(N)}),
        ('quantum_add',modular_quantum_adder(n,N),{a+(b<<n):a+(((a+b)%N)<<n) for a in range(2**n) for b in range(N)}),
        ('outplace_quantum_add',outplace_quantum_adder(n,N),{a+(b<<n):a+(b<<n)+(((a+b)%N)<<(2*n)) for a in range(2**n) for b in range(N)}),
        ('constant_MAC',multiply_accumulate(n,N,3),{a+(b<<n):a+(((b+3*a)%N)<<n) for a in range(2**n) for b in range(N)}),
        ('inplace_multiplier',modular_multiplier(n,N,3),{a:(3*a)%N for a in range(N)}),
        ('quantum_MAC',quantum_multiply_accumulate(n,N),{a+(b<<n)+(r<<(2*n)):a+(b<<n)+(((r+a*b)%N)<<(2*n)) for a in range(2**n) for b in range(2**n) for r in range(N)}),
        ('modular_exponentiation',modular_exponentiation(n,N,4,3),{x:x+(pow(4,x,N)<<3) for x in range(8)})]
    for name,circuit,mapping in configurations:
        row=coherent_probe(circuit,mapping);row.update(operation=name,N=N,data_width=n);coherence.append(row)
    row=coherent_probe(modular_multiplier(4,15,2),{a:(2*a)%15 for a in range(15)})
    row.update(operation='composite_modulus_multiplier',N=15,data_width=4);coherence.append(row)
    # A composite modulus from the Shor notebook is checked on all valid residues.
    row=coherent_probe(modular_multiplier(5,21,2),{a:(2*a)%21 for a in range(21)})
    row.update(operation='composite_modulus_multiplier',N=21,data_width=5);coherence.append(row)
    adders=[]
    for n in range(2,6):
        q=quantum_adder(n);r=adder_ripple_c04(n,kind='fixed')
        mapping={a+(b<<n):a+(((a+b)%2**n)<<n) for a in range(2**n) for b in range(2**n)}
        qcheck=coherent_probe(q,mapping);rcheck=coherent_probe(r,mapping)
        adders.append({'data_width':n,'modulus':2**n,'qft':resource_counts(q),'ripple':resource_counts(r),
            'qft_coherent_error':qcheck['state_error_l2'],'ripple_coherent_error':rcheck['state_error_l2']})
    if source_hashes()!=before:raise RuntimeError('Sources changed during arithmetic execution')
    return {'schema_version':1,'status':'passed','source_hashes':before,'environment':environment(),
        'fourier':fourier,'approximation':approximation,'coherent_arithmetic':coherence,
        'controlled_phase_regression':phase_regression(),'adder_comparison':adders,
        'contracts':'valid residues and zero-initialized workspace; full complex amplitudes, not only basis output probabilities',
        'scope':'ideal simulation, bounded inputs; no general hardware or speedup claim'}


def run_shor_experiments():
    before=source_hashes()
    native=run_order_finding(2,15,oracle='qft_arithmetic')
    compiled15=run_order_finding(2,15,oracle='compiled_permutation')
    compiled21=run_order_finding(2,21,oracle='compiled_permutation')
    bad_base=run_order_finding(5,21,oracle='compiled_permutation')
    approximate=run_order_finding(2,21,oracle='compiled_permutation',cutoff=1)
    if native['factorization']['factors']!=[3,5] or compiled21['factorization']['factors']!=[3,7]:
        raise ValueError('Verified factor extraction failed')
    if bad_base['factorization']['status']!='trivial_square_root':raise ValueError('Bad-base negative control was lost')
    agreement=float(.5*np.sum(np.abs(np.asarray(native['phase_probabilities'])-compiled15['phase_probabilities'])))
    if agreement>1e-9:raise ValueError('Gate-built and compiled-oracle phase distributions differ')
    if source_hashes()!=before:raise RuntimeError('Sources changed during Shor execution')
    return {'schema_version':1,'status':'passed','source_hashes':before,'environment':environment(),
        'gate_arithmetic_N15':native,'compiled_N15':compiled15,'compiled_N21_base2':compiled21,
        'compiled_N21_base5_negative_control':bad_base,'approximate_IQFT_N21':approximate,
        'oracle_route_phase_TV_N15':agreement,
        'gate_arithmetic_N15_resources':resource_counts(order_circuit(2,15)),
        'exact_IQFT_resources':resource_counts(qft(9,inverse=True)),
        'approximate_IQFT_resources':resource_counts(qft(9,inverse=True,cutoff=1)),
        'oracle_scope':'N15 uses reversible QFT arithmetic; N21 uses full small-instance modular-permutation matrices with invalid codes fixed',
        'scope':'ideal simulation of small known composites; no scalable factoring implementation, hardware, or runtime advantage claim'}


def write_json(value,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')


def save_figure(figure,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    figure.savefig(path,metadata={'Date':None,'Creator':'QFT scientific validation'})
    if path.suffix=='.svg':path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')


def plot_arithmetic(report,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'svg.hashsalt':'qft-v1','axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(10,3.7),constrained_layout=True)
    rows=report['approximation']
    axes[0].semilogy([r['retained_controlled_phases'] for r in rows],[r['operator_error_2'] for r in rows],'o-',label='Error versus exact Fourier matrix')
    axes[0].semilogy([r['retained_controlled_phases'] for r in rows],[r['roundtrip_error_2'] for r in rows],'s--',label='Forward/inverse round trip')
    axes[0].set(title='A  Six-qubit QFT: accuracy versus truncation',xlabel='Retained controlled-phase gates',ylabel='Operator 2-norm error')
    axes[0].legend(fontsize=7);axes[0].grid(alpha=.2)
    rows=report['adder_comparison']
    for name,style in [('qft','o-'),('ripple','s--')]:axes[1].plot([r['data_width'] for r in rows],[r[name]['cx_count'] for r in rows],style,label=name)
    axes[1].set(title='B  Same modular sum, different implementations',xlabel='Bits per data register',ylabel='Logical CX gates')
    axes[1].set_xticks([r['data_width'] for r in rows]);axes[1].legend();axes[1].grid(alpha=.2)
    save_figure(fig,path);plt.close(fig)


def plot_shor(report,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'svg.hashsalt':'qft-v1','axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(9,3.7),constrained_layout=True)
    r=report['compiled_N21_base2'];p=np.asarray(r['phase_probabilities']);Q=len(p)
    ax.plot(np.arange(Q),p,label='Exact inverse QFT; circuit/FFT agreement')
    ax.plot(np.arange(Q),report['approximate_IQFT_N21']['phase_probabilities'],'--',alpha=.75,label='Approximate inverse QFT, distance cutoff 1')
    ax.axhline(1/Q,color='gray',linestyle=':',label='Counting marginal before inverse QFT')
    keys=sorted(r['counts'],key=r['counts'].get,reverse=True)[:12]
    ax.scatter([int(k,2) for k in keys],[r['counts'][k]/r['shots'] for k in keys],s=15,color='black',label='Seeded Aer readout')
    ax.set(title='Order finding: a=2, N=21, nine counting qubits',xlabel='Counting-register outcome (integer)',ylabel='Probability')
    ax.set_xticks([0,85,171,256,341,427,511]);ax.legend(fontsize=7);ax.grid(alpha=.15)
    save_figure(fig,path);plt.close(fig)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'artifacts/experiments')
    args=parser.parse_args();output=args.output_dir.resolve()
    if output==ROOT or (output.is_relative_to(ROOT) and not output.is_relative_to(ROOT/'artifacts')):
        parser.error('Generated evidence must be in artifacts/ or outside the source project')
    from run_checks import deny_connections
    deny_connections()
    arithmetic=run_arithmetic_experiments();shor=run_shor_experiments()
    write_json(arithmetic,output/'arithmetic.json');write_json(shor,output/'shor.json')
    plot_arithmetic(arithmetic,output/'arithmetic.svg');plot_shor(shor,output/'order_finding.svg')
    write_json({'schema_version':1,'files':{name:hashlib.sha256((output/name).read_bytes()).hexdigest()
        for name in ('arithmetic.json','shor.json','arithmetic.svg','order_finding.svg')}},output/'manifest.json')
    print(json.dumps({'status':'passed','output':str(output)},indent=2))
