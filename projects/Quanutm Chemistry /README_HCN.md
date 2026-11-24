# HCN active-space VQE benchmark

The [notebook](Chemistry_HCN_VQE_Study.ipynb) constructs a nitrile π/π* CAS(4,4)
Hamiltonian, maps it to eight qubits, and compares spin-preserving, multi-seed
VQE with CASCI. Full-basis RHF/PBE and one full-FCI control distinguish solver
and active-space effects.

After the [shared setup](README.md):

```bash
python run_checks.py --study hcn
```

Default execution uses three C–N distances, three seeds and a 300-iteration
budget. `HCN_PROFILE=smoke` uses one distance/seed and 30 iterations; it reports
incomplete optimization without certifying accuracy. Generated outputs go to
`artifacts/hcn/` (override `HCN_OUTPUT_DIR`). Add `--save-review` to refresh
default outputs and [numerical evidence](benchmarks/hcn/).

Saved integral tensors permit checkpoint replay in the recorded orbital basis.
The frozen σ framework, minimal basis and fixed H–C distance limit interpretation.
See [validation and limits](VALIDATION.md#hcn).
