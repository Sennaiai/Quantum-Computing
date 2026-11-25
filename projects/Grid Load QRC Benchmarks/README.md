# Quantum Reservoir Benchmarks for Grid Load Forecasting

Two executed studies compare fixed quantum reservoirs with strong classical models on the same England–Wales demand targets. The focus is reproducible quantum dynamics, controlled prediction experiments and resource accounting.

| Study | Physics and controls |
| --- | --- |
| [CV reservoir](CV_Reservoir_Benchmark.ipynb) | Two bosonic modes; Gaussian/Kerr ablation, native PennyLane Gaussian reference, analytic coherent/Kerr checks and Fock-cutoff convergence |
| [Analogue spin reservoir](Analogue_Spin_QRC_Benchmark.ipynb) | Six interacting qubits; exact Hamiltonian evolution, uncoupled control, Qiskit state/readout references and product-formula resource estimates |

Both notebooks retain compact tables and figures. Machine-readable [results](results/) include all seeds, predictions, validation choices, error intervals and resource diagnostics.

## Measured outcome

The 2,823 heldout targets favour gradient boosting: **324.3 MW RMSE**, compared with **351.0 MW** for both Kerr-hybrid and interacting-spin-hybrid ensembles, and **806.4 MW** for persistence. Reservoir-only, Gaussian, uncoupled and classical-reservoir controls are also reported. These results demonstrate verified QRC execution and benchmarking; they do not establish quantum predictive advantage.

## Reproduce

Python 3.11, from this folder:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_checks.py
```

Acceptance checks data integrity, causal alignment, selection boundaries and physics, then executes both notebooks in fresh offline kernels. Generated evidence stays in ignored `artifacts/validation/`. CI repeats the complete run; source and dataset hashes accompany results.

## Benchmark contract

The licensed NESO snapshot spans January–October 2025. London settlement periods are converted to contiguous UTC intervals, including 46/50-period days. Forecasts use information through t−1 to predict t, 30 minutes ahead. This is a rolling historical hindcast: outturn latency and retrospective revisions prevent an operational forecasting claim. The undocumented temperature column is excluded.

The encoder fits initial training observations only. Three expanding validation folds select readout/model parameters before the final 20% test interval. All three predetermined reservoir seeds are retained; their equal-weight ensemble is fixed in advance. Persistence, daily/weekly seasonal forecasts, lagged ridge, random forest, gradient boosting and a classical nonlinear reservoir provide controls. Reservoir-only and hybrid readouts predict a residual from the same persistence anchor; hybrid inputs contain the identical classical features. Test labels do not select models, seeds or stopping times.

## Physics, costs and limits

CV uses D, R, beamsplitter and genuine K(κ)=exp(iκn²) gates with ℏ=2. State norm, Fock-edge occupancy and cutoff-10/14 agreement are checked separately. Gaussian coherent dynamics have an efficient classical reference. The spin model evolves exp(−iτH), H=Σ(hₓX+h_zZ)+ΣJZZ; its input-squared RZ drive is not a bosonic Kerr interaction. States reset per eight-step window; neither model demonstrates unbounded streaming memory or hardware execution.

Reports distinguish measured CPU times, state/operator memory, trained readout size, CV gate counts, Qiskit logical u/CX costs and an ideal Pauli shot bound. Timing depends on the machine; hardware routing, calibration, noise and CV shot certification are absent. Seed spread, train/test gaps and paired temporal-block intervals expose sensitivity; they cannot guarantee absence of overfitting. One geographic series and one seasonal holdout do not establish general superiority or quantum advantage.

Useful extensions: as-of data vintages, additional years/regions, longer horizons, finite-shot/noisy readout and hardware-calibrated resource models.

## Credit and sources

Based on the QUAC 605 collaboration by Dominic Kosnica, Sennai Kahssai, Dawinder Bansrow and Michael Liu; the present benchmark refines its execution and evaluation protocol. The original report and duplicate notebooks are omitted from this focused release; tracked history preserves the starting implementations.

Supported by National Energy SO Open Data. [Dataset](https://www.neso.energy/data-portal/historic-demand-data), [licence](https://www.neso.energy/data-portal/neso-open-licence), [snapshot provenance](data/provenance.json). Physics references: [Fujii–Nakajima QRC](https://arxiv.org/abs/1602.08159), [Kerr convention](https://docs.pennylane.ai/en/stable/code/api/pennylane.Kerr.html), [beamsplitter convention](https://strawberryfields.readthedocs.io/en/latest/code/api/strawberryfields.ops.BSgate.html), [settlement time](https://www.elexon.co.uk/bsc/settlement/).
