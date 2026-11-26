# QAOA for Airline-Recovery QUBOs

Compare QAOA and annealing on the [parent solver’s](../README.md) unchanged QUBOs. The [main notebook](expanded/Larger_IROPS_QAOA.ipynb) tests **six actual neighborhoods: 4–10 PNRs, 17–28 logical qubits**, with outside bookings fixed.

## Findings

**Additional PNRs: 0. Additional passengers: 0.** Exact enumeration confirms local allocation saturation; full-airline optimality is unproven. No full-population QAOA rate was measured.

Six cases × three seeds = **18 runs/setting**, with 1,024 final draws each. Below: local optimum discovery and paired best feasible S3 before repair, **not booking allocation rates**. Lower S3 is better.

| Method | Optimum found: runs / 18 | Lower / equal / higher S3 than annealing |
|---|---:|---:|
| Untrained one-choice | 10 / 18 | — |
| Annealing | 14 / 18 | — |
| QAOA depth 1 | 11 / 18 | 2 / 11 / 5 |
| QAOA depth 2 | 11 / 18 | 0 / 11 / 7 |
| QAOA depth 3 | 12 / 18 | 2 / 12 / 4 |
| QAOA depth 4 | 13 / 18 | 2 / 13 / 3 |
| QAOA depth 5 | 10 / 18 | 1 / 11 / 6 |

Depth 4 has the highest QAOA optimum-discovery count; annealing remains stronger overall. Pareto analysis maximizing PNRs/passengers and minimizing S3 yields **one point per case**, without a tradeoff curve.

## Execution and verification

Exact one-choice/slack simulation retains all allowed amplitudes. Qiskit block replay verifies 90 optimized circuits; full-state references include a 22-qubit control. [Evidence](expanded/results/) separates raw samples, repair and admission; patches cannot be summed.

The [mixer ablation](QAOA_IROPS_Neighbourhoods.ipynb) and its [evidence](results/) compare X/one-choice mixing and untrained controls on 8–13-qubit cases; this is not a matched before/after comparison.

Python 3.11, from this folder:

```bash
python -m pip install -r requirements.txt
python run_qaoa_checks.py
python -m expanded.run_checks
```

Checks cover Hamiltonians, bit order, gradients, circuits and allocation audits in fresh offline notebooks. Final draw counts match; exact-expectation training compute does not. No hardware, speedup or statistical-advantage claim is made.

End-to-end: **71.2 s main study; 76.1 s ablation**, Apple M5/arm64, Python 3.11.18, one requested BLAS/OpenMP thread. Includes training, references, verification and exports; excludes installation, kernel startup and case generation. [Configuration](expanded/config.json) fixes budgets.
