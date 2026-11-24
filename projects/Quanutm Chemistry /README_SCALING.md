# Active-space resource and accuracy benchmark

The [notebook](Molecular_Hamiltonian_Construction_Active_Space_Scaling.ipynb)
shows geometry, effective integrals, fermionic assembly, mapping, physical-sector
checks, and resource/energy comparisons for NH₃, H₂CO₃ and HCO₃⁻.

After the [shared setup](README.md):

```bash
python run_checks.py --study scaling
```

Defaults use nested 4/8/12-qubit spaces. `SCALING_PROFILE=smoke` uses 4/8 qubits.
`SCALING_LARGE_OPERATORS=1` explicitly adds 16-qubit operators without qubit
diagonalization. Generated outputs go to `artifacts/scaling/`; override
`SCALING_OUTPUT_DIR` for alternate runs. Add `--save-review` to refresh default
outputs and [numerical evidence](benchmarks/scaling/).

Default execution reads fixed geometry JSON. Optional regeneration uses
`regenerate_scaling_geometries.py` with RDKit 2026.09.2.

References are finite-basis/active-space dependent. QWC counts are heuristic
measurement settings, not shot or hardware costs. The geometries are not
quantum-optimized; STO-3G lacks diffuse functions for the anion.
See [validation and limits](VALIDATION.md#scaling).
