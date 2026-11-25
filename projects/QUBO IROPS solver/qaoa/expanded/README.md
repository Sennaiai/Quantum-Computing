# Larger QAOA Neighborhoods: Allocation and Pareto Audit

Extend the [baseline](../README.md) with six actual neighborhoods: **4–10 PNRs, 17–28 logical qubits**, unchanged S3/capacity-slack QUBOs and fixed outside bookings. Structural selection retains three controls alongside broader route/blocker cases, without using solver outcomes.

## Reallocation outcome

**Added PNRs: 0. Added passengers: 0**, for both solvers. Exact enumeration confirms the incumbent reaches both local allocation ceilings in all six restricted catalogs. This certifies local saturation only. The full classical mean remains **72.19% of PNRs and 61.29% of passengers**; no full-population QAOA rate was measured.

## Matched local solution quality

Six cases × three seeds = **18 runs per setting**, each with 1,024 final draws. Below: optimum-discovery counts before repair, not booking allocation rates. Paired comparisons match case/seed; lower S3 is better.

| Method | Runs finding the exact local S3 optimum / 18 | Lower / equal / higher S3 than annealing |
|---|---:|---:|
| Untrained one-choice | 10 / 18 | — |
| Simulated annealing | 14 / 18 | — |
| QAOA depth 1 | 11 / 18 | 2 / 11 / 5 |
| QAOA depth 2 | 11 / 18 | 0 / 11 / 7 |
| QAOA depth 3 | 12 / 18 | 2 / 12 / 4 |
| QAOA depth 4 | 13 / 18 | 2 / 13 / 3 |
| QAOA depth 5 | 10 / 18 | 1 / 11 / 6 |

QAOA sometimes finds lower S3; annealing finds the exact optimum more often overall. All depths are reported, without a statistical or quantum-advantage claim. [Results](results/) retain draw counts, repairs, circuits and global audits.

## Pareto analysis and resources

Maximize PNRs/passengers and minimize S3. Enumeration finds **one nondominated metric point per case**, a degenerate frontier. Observed frontiers use feasible raw samples. Admission requires S3 improvement without reducing PNRs/passengers; independent patches cannot be summed.

Exact invariant-subspace simulation uses 2,592–32,768 amplitudes and retains every allowed choice/slack state. Qiskit block replay checks **90 optimized circuits**; full-state checks include a 22-qubit control. This exploits conserved structure; no hardware execution is claimed.

**End-to-end: 71.2 seconds**, Apple M5/arm64, Python 3.11.18, one requested BLAS/OpenMP thread. Includes loading, references, training, sampling, verification, audits and exports; excludes installation, kernel startup and case generation. QAOA: three starts, at most 100 iterations/start, exact expectations. Annealing: 500 sweeps. Final draw counts match; training compute and physical-shot budgets do not.

From the parent `qaoa/` folder, using its pinned environment:

```bash
python -m pip install -r requirements.txt
python -m expanded.run_checks
```

Acceptance runs the [notebook](Larger_IROPS_QAOA.ipynb) offline, scientific tests and replay of **144 sample batches and global patch audits**. Scientific CI stays within this QAOA project.
