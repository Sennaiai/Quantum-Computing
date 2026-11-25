# Experimental QUBO Solver for Airline Re-accommodation

The goal is to develop a QUBO IROPS solver, measure feasible PNR/passenger allocation, and assess it against the client’s own industry-standard solver. **The client performs that comparison independently; no competing solver is built here and no outperformance claim is made.**

The [executed notebook](QUBO_IROPS_Allocation_Experiment.ipynb) runs the supplied de-identified catalog: **7,653 PNRs, 57,963 options and 11,822 passengers**. All fixed-seed results, assignments, runtime and resource evidence are in [results](results/).

## Results and runtime

Across five fixed seeds, mean allocation was **54.98% of PNRs** and **46.21% of passengers**, with **zero final capacity violations**.

**End-to-end runtime: 107.5 seconds for all five runs**; mean solver time **19.74 seconds per seed**. Measured on Apple M5/arm64, Python 3.11.18, with BLAS/OpenMP threads set to 1. End-to-end timing includes loading, certificate verification, optimization, repair, final audits and result export; dependency installation and kernel startup are excluded.

The 70% target is infeasible under this catalog and remaining-seat inventory: exact-rational certificates bound allocation by **4,557 PNRs (59.55%)** and **5,930 passengers (50.16%)**, separately. [Certificate verification](capacity_bounds.py) checks every option independently of the optimizer.

## Reproduce

Python 3.11, from this folder:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_checks.py
```

Acceptance tests the formulation and executes the actual-data notebook in a fresh offline kernel. Controlled fixtures serve only as unit tests. Generated outputs stay in ignored `artifacts/`.

## Method and limits

Original S1–S4 scoring weights are preserved; S3 is the main experiment. Each PNR chooses one itinerary or explicit unassignment. Complete capacity inequalities use bounded integer slack variables. A penalty above the linear-objective range guarantees feasible global ground states. Finite simulated annealing, whole-PNR decomposition and reconciliation do not guarantee global optimality or maximum allocation.

The implementation uses **classical simulated annealing (`neal`)**, with shared residual inventory and independent final capacity audits. Raw sample and repaired outcomes are separate. Resource reports include variables, slack overhead, couplers, coefficient range and logical hardware lower bounds. No quantum hardware execution, embedding feasibility or speedup is claimed.

The authorized [catalog](data/provenance.json) preserves costs, capacities, party sizes and priority order. Identifiers are replaced, dates shifted and CVM normalized. Raw tables and reversible mappings remain local. Negative availability becomes zero remaining seats, with counts recorded. Flagged original-journey/ranking outliers remain an upstream limitation.

Allocation denominators cover the supplied candidate catalog, not every disrupted booking. Client comparisons require matched populations and business rules. Optional local processed-table execution uses `run_private.py`; its outputs remain private.

References: [penalties](https://docs.dwavequantum.com/en/latest/concepts/penalty.html), [QUBO/Ising offsets](https://docs.dwavequantum.com/en/latest/ocean/api_ref_dimod/generated/dimod.utilities.ising_to_qubo.html), [classical samplers](https://docs.dwavequantum.com/en/latest/ocean/api_ref_dimod/sampler_composites.html).
