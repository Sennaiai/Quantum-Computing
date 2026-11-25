# QUBO Airline Recovery: Capacity, Connections and Allocation

Develop an IROPS QUBO solver, measure feasible PNR/passenger allocation, and assess it against the client’s industry-standard solver. **The client performs that comparison independently, with matched populations and rules.**

The [executed notebook](QUBO_IROPS_Allocation_Experiment.ipynb) uses **7,653 PNRs and 11,822 passengers**. Both experiments share inventory, S3 weights/scales, recovery constraints and optimization budgets.

## Results and runtime

Five fixed seeds; means shown. Final solutions passed independent capacity and route-policy audits.

| Candidate catalog | Options | PNR allocation | Passenger allocation | Mean variant time |
|---|---:|---:|---:|---:|
| Original routes, policy filtered | 57,830 | 57.60% | 47.46% | 21.65 s |
| Expanded routes and cabin alternatives | 439,145 | **72.19%** | **61.29%** | 27.00 s |

**End-to-end: 265.1 seconds** for both catalogs across all five seeds. Apple M5/arm64, Python 3.11.18; BLAS/OpenMP threads set to 1. This includes loading, route generation, certificate verification, optimization, audits and exports; installation, kernel startup and precomputed certificate generation are excluded. [Results](results/) retain assignments, stage timings, variability and resources.

Mean journey time rises **7.61 → 9.13 hours**, and downgrades **4.09% → 8.55%** of allocated PNRs. These compare different served cohorts, not measured satisfaction.

Exact-rational certificates cap the expanded instance at **5,599 PNRs (73.16%)** and **7,437 passengers (62.91%)**, separately; attainability is unproven. Added routes enlarge the feasible set without adding seats. Original-catalog bounds apply only to that control.

## External airline benchmark

The airline will use its own Gurobi solver as the benchmark. Under identical data/rules, separate allocation optima lie between our best feasible results and certified ceilings: **PNRs 5,526–5,599 (about 72.21–73.16%); passengers 7,252–7,437 (about 61.34–62.91%)**. These bound allocation-maximization optima, not Gurobi runtime, S3 outcomes or joint attainability. The airline’s additional business rules may differ; Gurobi is not implemented or executed here.

The **72.19% result above is classical**. The optional [QAOA extension](qaoa/README.md) compares the same QUBO formulation on actual 8–13-qubit residual neighborhoods.

## Method and reproduction

Whole-PNR decomposition uses exact capacity-slack QUBOs, explicit unassignment and a sufficient penalty bound. **Classical `neal` simulated annealing** precedes separately audited route improvement, bounded rerouting and passenger-preserving two-for-one exchanges. Accepted moves improve S3. Exact size estimation avoids constructing rejected BQMs; accepted chunks retain full validation and a 128-variable budget. Search remains heuristic; resource counts imply no embedding, hardware execution or quantum advantage.

Expansion reconstructs original features on frozen scales, excludes 133 routes exceeding the 72-hour final-arrival window, and includes connecting routes and feasible cabin patterns. Negative inventory is conservatively floored at zero.

Python 3.11, from this folder:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_checks.py
```

Acceptance runs scientific tests and the full notebook in a fresh offline kernel. Raw inputs and reversible mappings remain local; [provenance](data/provenance.json) and anonymous network metadata support reproduction. Private processed-table execution supports `run_private.py --expand-connections`.

The prototype assumes 45-minute transfers, at most 18-hour layovers and two stops; airport-specific certification remains external. Original-journey/ranking anomalies and bookings outside the candidate population remain limitations. S3 is a business proxy, with no monetary-profit claim.
