# QAOA Extension for Airline-Recovery QUBOs

A circuit-based extension of the [classical recovery pipeline](../README.md), using its unchanged QUBO compiler and penalties. It generalizes the [HHL project’s](../../HHL/README.md) alternating-layer design with sparse Pauli costs, analytic adjoint gradients and independent Qiskit validation.

Four actual residual neighborhoods contain **8, 9, 12 and 13 qubits**, including dummy/slack variables. Selection uses structural size bands before algorithm outcomes; outside bookings stay fixed. These local results do not replace the parent’s full-population allocation.

## Reallocation outcome

The full classical hybrid pipeline reallocates **72.19% of 7,653 PNRs and 61.29% of 11,822 passengers**, averaged over five seeds. QAOA was tested only on four local neighborhoods: its audited patches added **zero PNRs and zero passengers** to the fixed incumbent. No full-population QAOA allocation rate was measured.

## Local candidate quality

Each method generates **20,480 candidate draws**: four neighborhoods × five seeds × 1,024 samples. Draws may repeat an assignment. A valid candidate respects one route or explicit unassignment per PNR and all seat capacities, before repair. Validity does not mean every PNR is served or the solution is optimal. Counts below are neither booking reallocation rates nor coverage of all possible feasible solutions.

| Sampler | Valid candidate draws / 20,480 | Runs finding a local S3 optimum / 20 |
|---|---:|---:|
| Uniform | 848 / 20,480 | 18 / 20 |
| Untrained one-choice | 10,027 / 20,480 | 20 / 20 |
| Classical simulated annealing | 20,336 / 20,480 | 20 / 20 |
| X-QAOA depth 1 | 2,920 / 20,480 | 20 / 20 |
| X-QAOA depth 5 | 7,464 / 20,480 | 19 / 20 |
| One-choice QAOA depth 1 | 14,663 / 20,480 | 20 / 20 |
| One-choice QAOA depth 5 | 16,949 / 20,480 | 20 / 20 |

The last column counts runs with at least one valid draw attaining the independently enumerated local S3 optimum, before repair. These local optima optimize the business objective, not necessarily PNR count alone.

Depths 1–5 use five optimizer starts. All methods share the same QUBOs and final sample counts. One-choice projector mixing plus slack-X preserves PNR choices; capacities remain penalized in the cost Hamiltonian. An untrained one-choice control separates initialization from optimization.

Constrained QAOA produces more valid candidates than X-QAOA and its untrained control; classical annealing remains stronger. Both annealing and the untrained one-choice control already find a local optimum in every run. **No allocation or speedup advantage is demonstrated.** All depths, raw counts, strict slack/ground-state diagnostics, repairs and global audits remain in [results](results/). Higher depth need not improve finite-sample outcomes.

**End-to-end: 76.1 seconds**, Apple M5/arm64, Python 3.11.18, one requested BLAS/OpenMP thread. Timing includes all references, samplers, optimization, circuit verification, admission audits and exports; installation, kernel startup and case generation are excluded. Training uses exact statevector expectations and final sampling is finite-shot; physical-shot/compute budgets are not equated between methods.

## Larger-neighborhood follow-up

The [expanded study](expanded/README.md) tests six actual 17–28-qubit neighborhoods with four to ten PNRs, depths 1–5 and exact local Pareto references. It adds no PNRs or passengers; exact enumeration confirms the fixed local catalogs have no allocation headroom. Paired S3 results distinguish occasional QAOA wins from the stronger overall annealing result.

## Reproduce

Python 3.11, from this folder:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_qaoa_checks.py
```

The [executed notebook](QAOA_IROPS_Neighbourhoods.ipynb) exposes the Hamiltonian and audits. The [one-choice example circuit](results/example_qaoa.qasm) round-trips through Qiskit and matches the statevector. Tests independently check signs, endian conventions, offsets, gradients, dense/circuit evolution, excitation conservation and all small-case binary energies.

This is ideal simulation with bounded local optimization. Gate resources assume all-to-all connectivity; no hardware, noise, speedup, industry-outperformance or globally optimal parameter claim is made. Classical repair is separate; independent patches cannot be summed.

Method context: [QAOA](https://arxiv.org/abs/1411.4028), [Grover-style projectors](https://arxiv.org/abs/2006.00354), and [local constraint-preserving mixers](https://arxiv.org/abs/2606.11530). The implementation uses disjoint one-choice projectors; it does not prepare the globally capacity-feasible subspace.
