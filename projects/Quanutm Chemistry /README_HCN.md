# HCN active-space VQE benchmark

The [notebook](Chemistry_HCN_VQE_Study.ipynb) constructs a nitrile π/π* CAS(4,4)
Hamiltonian explicitly, maps it to eight qubits, prepares HF/GateFabric states,
and compares multi-seed VQE with matched CASCI. Full-basis RHF/PBE and one
full-FCI control distinguish solver and active-space effects.

From this directory with Python 3.11:

```bash
python -m venv /tmp/hcn-benchmark-env
source /tmp/hcn-benchmark-env/bin/activate
python -m pip install -r requirements-hcn.txt
python run_hcn_checks.py
```

Default execution uses three C–N distances, three seeds, and a 300-iteration
budget. `HCN_PROFILE=smoke` uses one distance/seed and 30 iterations; it reports
incomplete optimization honestly and does not certify accuracy. Generated
outputs go to `artifacts/hcn/` (override `HCN_OUTPUT_DIR`). `--save-review`
refreshes default outputs and [benchmarks/hcn](benchmarks/hcn/).

Standalone installation was verified on macOS ARM64; resolved versions are in
`requirements-hcn-macos-arm64.lock`. Saved integrals allow checkpoint replay
without assuming identical orbital phases across machines.

All energies include nuclear repulsion. The σ framework is frozen and the
basis is minimal; the short fixed-H–C scan does not establish complete
triple-bond dissociation, binding affinity, or experimental accuracy.
[HCN_VALIDATION.md](HCN_VALIDATION.md) records numerical checks, software
versions, rejected-backend diagnostics, and remaining limits.
