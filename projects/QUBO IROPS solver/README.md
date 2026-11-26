# QUBO Airline Recovery

Develop and assess an airline reallocation solver against the client’s external Gurobi benchmark. The [executed study](QUBO_IROPS_Allocation_Experiment.ipynb) uses **7,653 bookings (PNRs) and 11,822 passengers** from an authorized, de-identified catalog.

## Results

Five-seed means; all final assignments passed capacity and route-policy audits.

| Candidate catalog | Options | PNRs reallocated | Passengers reallocated | Mean run |
|---|---:|---:|---:|---:|
| Original, policy filtered | 57,830 | 57.60% | 47.46% | 21.65 s |
| Expanded routes/cabins | 439,145 | **72.19%** | **61.29%** | 27.00 s |

Percentages use the full populations above. **End-to-end: 265.1 s** across both catalogs and all five seeds, Apple M5/arm64, Python 3.11.18, one BLAS/OpenMP thread. Includes loading, route generation, optimization, certificate verification, audits and exports; excludes installation, kernel startup and precomputed certificate generation.

Mean journey time increased **7.61→9.13 hours**; downgrades increased **4.09→8.55% of served PNRs**. Served cohorts differ; these are not satisfaction measurements.

Under identical rules, separate allocation optima are bounded by best feasible results and exact-rational certificates: **PNRs 72.21–73.16%; passengers 61.34–62.91%**. Joint attainability, Gurobi runtime and S3 outcomes are not established. The airline performs the benchmark independently.

## Method and reproduction

The classical hybrid uses whole-PNR capacity-slack QUBOs, explicit unassignment, sufficient penalties, `neal` simulated annealing and audited route improvement. Accepted moves improve S3; chunks have a 128-variable budget. The [QAOA comparison](qaoa/README.md) tests local quantum circuits separately and demonstrates no additional allocations.

Python 3.11, in this folder:

```bash
python -m pip install -r requirements.txt
python run_checks.py
```

Checks execute scientific tests and a fresh offline notebook. [Results](results/) retain assignments, variability, timings and resources; [provenance](data/provenance.json) identifies public inputs. Raw data/mappings remain local; private execution uses `run_private.py --expand-connections`.

Limits: 45-minute transfers, 18-hour layovers, at most two stops and a 72-hour recovery window; negative inventory is floored at zero. Airport-specific rules, source ranking anomalies and excluded bookings remain external limitations. S3 is a business proxy; search is heuristic. No monetary-profit, hardware or quantum-advantage claim is made.
