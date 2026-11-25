"""Deterministic simulator experiments; no external backend or credentials."""
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import numpy as np
from qiskit import QuantumCircuit
from amplification import *

ROOT=Path(__file__).resolve().parent
NOTEBOOK='Grovers_Algorithm_for_Quantum_Search_and_Acceleration.ipynb'

def source_hashes():
    paths=[*ROOT.glob('*.py'),ROOT/NOTEBOOK,ROOT/'requirements.txt']
    return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def run_experiments():
    initial_sources=source_hashes()
    targets=['00011','10101'];a=uniform_preparation(5);o=oracle_for_targets(5,targets)
    uniform,circuit=simulate(a,o,targets,first_peak_iterations(2/32))
    arbitrary=QuantumCircuit(3,name='Nonuniform A')
    arbitrary.ry(.4,0);arbitrary.ry(.9,1);arbitrary.h(2);arbitrary.cx(2,1);arbitrary.rz(.6,0)
    good=['001','110'];psi=Statevector.from_instruction(arbitrary).data
    p=float(sum(abs(psi[int(t,2)])**2 for t in good))
    nonuniform,_=simulate(arbitrary,oracle_for_targets(3,good),good,first_peak_iterations(p))
    four=queens_reference(4);predicate=queens4_constraint_oracle()
    constraint,q4=simulate(uniform_preparation(8),predicate,four['targets'],four['first_peak_iterations'])
    lookup4=oracle_for_targets(8,four['targets'])
    predicate5=queens_constraint_oracle(5)  # constructed from constraints, before reference solutions
    five=queens_reference(5);lookup5=oracle_for_targets(15,five['targets'])
    execution5,_=simulate(uniform_preparation(15),lookup5,five['targets'],five['first_peak_iterations'])
    constraint5,_=simulate(uniform_preparation(15),predicate5,five['targets'],five['first_peak_iterations'])
    observed=sorted(bits for bits,count in constraint5['counts'].items() if count and
        valid_board([int(bits[i:i+3],2) for i in range(0,15,3)]))
    if observed!=five['targets']:
        raise ValueError('Seeded five-Queens run did not recover the complete reference solution set')
    resources={
        'uniform_complete_run':resource_counts(circuit),
        'queens4_constraint_oracle':resource_counts(predicate),
        'queens4_lookup_oracle':resource_counts(lookup4),
        'queens4_constraint_complete_run':resource_counts(q4),
        'queens5_lookup_oracle':resource_counts(lookup5),
        'queens5_constraint_oracle':resource_counts(predicate5),
        'queens5_diffuser':resource_counts(diffusion(uniform_preparation(15)))}
    resources['queens5_modular_run']={
        'qubits':19,'phase_oracle_calls':constraint5['iterations'],
        'cx_count':constraint5['iterations']*(resources['queens5_constraint_oracle']['cx_count']+resources['queens5_diffuser']['cx_count']),
        'scope':'CX sum for separately compiled oracle and diffuser modules; no cross-module optimization or device routing'}
    sweep=[{'iterations':k,'quantum_success':analytic_probability(2/32,k),
        'classical_success_k_plus_one_checks':classical_success(2/32,k+1)} for k in range(11)]
    # Empirical query-law check for one marked state. This is an analytic count
    # table, explicitly distinct from measured gate counts and CPU performance.
    query_scaling=[{'system_qubits':n,'N':2**n,'M':1,'iterations':first_peak_iterations(1/2**n),
        'analytic_success':analytic_probability(1/2**n,first_peak_iterations(1/2**n)),
        'classical_expected_checks_without_replacement':(2**n+1)/2} for n in range(2,15)]
    if source_hashes()!=initial_sources:raise RuntimeError('Sources changed during experiment')
    return {'schema_version':2,'status':'passed','source_hashes':initial_sources,
        'environment':{p:version(p) for p in ('qiskit','qiskit-aer','numpy','scipy','matplotlib')},
        'uniform':uniform,'nonuniform':nonuniform,'queens4':{'classical':four,'constraint_run':constraint},
        'queens5':{'classical':five,'lookup_run':execution5,'constraint_run':constraint5,
            'lookup_scope':'pre-enumerated solution lookup; comparison only',
            'constraint_scope':'15 board qubits plus four violation-counter qubits; no solution list in oracle construction',
            'reference_observed_marked_states':observed,
            'observed_valid_boards':[[int(bits[i:i+3],2) for i in range(0,15,3)] for bits in observed],
            'known_success_assumption':'M=10 from the independent classical reference sets k=44; unknown-M search is not implemented'},
        'resources':resources,'overshoot_sweep':sweep,'analytic_query_scaling':query_scaling,
        'classical_controls':{'uniform_with_replacement':'1-(1-p)^q; q=k+1 includes a final candidate verification',
            'without_replacement_expected_first_success':'(N+1)/(M+1), assuming M>0',
            'limitations':'phase-oracle calls, verifier calls, state preparation and gates have different costs; structured backtracking is not an unstructured query baseline'},
        'scope':'ideal Aer statevector and seeded shots; no noise, hardware, or wall-clock speedup claim',
        'five_queens_encoding':'15 uniform system bits represent 32768 assignments; column-range and pair constraints mark exactly 10 boards',
        'queens5_sweep':[{'iterations':k,'quantum_success':analytic_probability(10/32768,k),
            'classical_success_k_plus_one_checks':classical_success(10/32768,k+1)} for k in range(89)],
        'network_calls':0}


def write_report(report,destination):
    path=Path(destination);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')

def plot_report(report,destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'svg.hashsalt':'amplitude-v1','axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(10,3.7),constrained_layout=True)
    rows=report['queens5_sweep']
    axes[0].plot([r['iterations'] for r in rows],[r['quantum_success'] for r in rows],'o-',label='Analytic amplification')
    axes[0].plot([r['iterations'] for r in rows],[r['classical_success_k_plus_one_checks'] for r in rows],'s--',label='Classical sampling, q=k+1')
    run=report['queens5']['constraint_run']
    axes[0].errorbar(run['iterations'],run['sampled_success'],yerr=run['hoeffding_half_width'],fmt='x',color='black',label='Full-circuit Aer run; readout bound')
    axes[0].set(title='A  5-Queens: amplification and overshooting',xlabel='Quantum marking calls k',ylabel='Success probability',ylim=(-.04,1.04))
    axes[0].legend(fontsize=7);axes[0].grid(alpha=.2)
    names=['queens5_lookup_oracle','queens5_constraint_oracle']
    costs=[report['resources'][name]['cx_count'] for name in names]
    bars=axes[1].bar(['Lookup\n15 qubits; answers enumerated','Constraint\n15 + 4 counter qubits'],costs,color=['#526777','#007F86'])
    axes[1].bar_label(bars,padding=3)
    axes[1].set(title='B  5-Queens: one phase-oracle call',ylabel='Logical CX gates',ylim=(0,max(costs)*1.2))
    axes[1].text(.5,-.26,'u/CX basis; optimization 0; no routing',ha='center',transform=axes[1].transAxes,fontsize=8)
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(destination,metadata={'Date':None,'Creator':'Reproducible amplitude amplification demonstration'})
    if destination.suffix=='.svg':
        destination.write_text('\n'.join(line.rstrip() for line in destination.read_text().splitlines())+'\n')
    plt.close(fig)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts/benchmark_summary.json')
    args=parser.parse_args()
    destination=args.output.resolve()
    if destination==ROOT or (destination.is_relative_to(ROOT) and not destination.is_relative_to(ROOT/'artifacts')):
        parser.error('Generated output must be in artifacts/ or outside the source project')
    report=run_experiments();write_report(report,destination);plot_report(report,destination.with_suffix('.svg'))
    print(json.dumps({'status':report['status'],'output':str(destination),'cases':5},indent=2))
