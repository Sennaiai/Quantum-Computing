# Quantum chemistry benchmarks

Three complementary studies with explicit molecular models, Hamiltonian
construction, classical references, numerical assertions, and saved results.

| Study | Scientific question | Guide |
|---|---|---|
| [H₂](Quantum_Simulation_of_the_Hydrogen_Molecule_H2_Hamiltonian.ipynb) | Analytical/circuit consistency; solver, method and basis effects | [H₂](README_H2.md) |
| [HCN](Chemistry_HCN_VQE_Study.ipynb) | Spin-preserving VQE and π-active-space truncation | [HCN](README_HCN.md) |
| [Active-space scaling](Molecular_Hamiltonian_Construction_Active_Space_Scaling.ipynb) | Operator resources versus controlled correlation-energy recovery | [Scaling](README_SCALING.md) |

From this directory in a full repository checkout, with Python 3.11:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_checks.py
```

Use `--study h2`, `--study hcn`, or `--study scaling` to execute one study;
`--tests-only` runs tests without kernels. `--save-review` refreshes all selected
default notebook outputs and benchmark JSON only after successful execution.

[VALIDATION.md](VALIDATION.md) records the evidence, tolerances, commands,
CI status and limitations. Direct dependencies are pinned; the resolved macOS
ARM64 environment is in `requirements-macos-arm64.lock`. Scientific tests and
fresh-kernel execution are distinct from notebook structure checks.

These are research-oriented, noiseless benchmarks within specified finite
models. They demonstrate scientific reasoning and reproducible implementation;
they establish neither novel research nor experimental accuracy, hardware
performance or quantum advantage. No credentials or external datasets are needed.
