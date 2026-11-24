# H₂ Hamiltonian and variational benchmark

The [notebook](Quantum_Simulation_of_the_Hydrogen_Molecule_H2_Hamiltonian.ipynb)
shows geometry and MO integrals, fermionic assembly, Jordan–Wigner mapping,
Hartree–Fock preparation, variational optimization, and FCI/RHF/PBE comparisons.
Basis/grid controls and natural occupations separate solver and modeling effects.

From this directory, with Python 3.11:

```bash
python -m venv /tmp/h2-benchmark-env
source /tmp/h2-benchmark-env/bin/activate
python -m pip install -r requirements-h2.txt
python run_h2_checks.py
```

Default execution samples seven bond lengths. `H2_PROFILE=smoke` samples two;
`H2_PROFILE=extended` samples 22. Generated evidence goes to `artifacts/h2/`
(override `H2_OUTPUT_DIR`). `--save-review` refreshes the default notebook
outputs and [saved benchmarks](benchmarks/h2/).

The clean macOS ARM64 environment is recorded in
`requirements-h2-macos-arm64.lock`; cross-platform CI uses the direct pins.
[H2_VALIDATION.md](H2_VALIDATION.md) records commands, results and remaining
checks. Repository discovery is checked with
`python -m unittest -v test_notebook_discovery`.

All molecular energies include nuclear repulsion. FCI is exact within the
chosen finite basis; the noiseless four-qubit calculation does not establish
experimental accuracy or quantum advantage. Remote Linux CI is pending.
