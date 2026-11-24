# H₂ Hamiltonian and variational benchmark

The [notebook](Quantum_Simulation_of_the_Hydrogen_Molecule_H2_Hamiltonian.ipynb)
shows geometry and MO integrals, fermionic assembly, Jordan–Wigner mapping,
HF preparation, variational optimization, and FCI/RHF/PBE comparisons. Basis,
grid and natural-occupation controls separate solver and model effects.

After the [shared setup](README.md):

```bash
python run_checks.py --study h2
```

The default uses seven bond lengths; `H2_PROFILE=smoke` uses two and
`H2_PROFILE=extended` uses 22. Generated outputs go to `artifacts/h2/`
(override `H2_OUTPUT_DIR`). Add `--save-review` to refresh the default notebook
and [numerical evidence](benchmarks/h2/).

All molecular energies include nuclear repulsion. FCI is exact within the
chosen finite basis. The historical coefficient table is a separate algebra
exercise with unknown molecular provenance. See [validation and limits](VALIDATION.md#h2).
