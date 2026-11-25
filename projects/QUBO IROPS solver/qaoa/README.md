# QAOA Extension for Airline-Recovery QUBOs

A circuit-based extension of the [classical recovery pipeline](../README.md), using its unchanged QUBO compiler and penalties. It generalizes the [HHL project’s](../../HHL/README.md) alternating-layer design with sparse Pauli costs, analytic adjoint gradients and independent Qiskit validation.

Four actual residual neighborhoods contain **8, 9, 12 and 13 qubits**, including dummy/slack variables. Selection uses structural size bands before algorithm outcomes; outside bookings stay fixed. These local results do not replace the parent’s full-population allocation.

## Measured comparison

Five seeds, depths 1–5, five optimizer starts and 1,024 final samples per run. Exact enumeration, thermal SA and both untrained controls use the same QUBOs. Standard X mixing is compared with one-choice-preserving projector mixing plus slack-X mixing. The latter preserves PNR choices; flight capacities remain in the cost Hamiltonian.

| Sampler | Raw physical feasibility | Raw exact-ground samples | Runs sampling the physical S3 optimum |
|---|---:|---:|---:|
| Uniform | 4.14% | 0.16% | 18/20 |
| Untrained one-choice | 48.96% | 1.53% | 20/20 |
| Thermal SA | 99.30% | 30.35% | 20/20 |
| X-QAOA depth 1 | 14.26% | 0.85% | 20/20 |
| X-QAOA depth 5 | 36.45% | 4.29% | 19/20 |
| One-choice QAOA depth 1 | 71.60% | 4.78% | 20/20 |
| One-choice QAOA depth 5 | 82.76% | 22.27% | 20/20 |

On these cases, constrained QAOA improves sampling concentration over its untrained control and X-QAOA; thermal SA remains stronger. The untrained one-choice control already samples every physical optimum, so this study establishes no allocation or speedup advantage. All depths, raw counts, strict slack feasibility, repairs, circuit resources and independent global admission audits are in [results](results/). Higher depth does not guarantee better finite-sample solution quality. No global PNR-allocation increase was demonstrated in these neighborhoods.

**End-to-end: 76.1 seconds**, Apple M5/arm64, Python 3.11.18, one requested BLAS/OpenMP thread. Timing includes all references, samplers, optimization, circuit verification, admission audits and exports; installation, kernel startup and case generation are excluded. Training uses exact statevector expectations and final sampling is finite-shot; physical-shot/compute budgets are not equated between methods.

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
