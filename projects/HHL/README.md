# HHL and Variational Linear Solvers

This project uses a small, \(4 \times 4\) positive-definite system to demonstrate HHL and variational approaches to solving \(Ax=b\). The executable walkthrough is [the HHL notebook](Quantum_Algorithms_for_Solving_Linear_Systems_HHL_and_Variational_Methods.ipynb).

The notebook:

- Checks the matrix eigensystem and classical reference solution.
- Simulates QPE, the HHL controlled rotation, and ancilla post-selection.
- Compares the HHL estimate and a VQE nullspace solution with the classical reference.
- Includes one-layer QAOA as a shallow variational baseline.
- Uses seeded simulations and numerical assertions for key results, including a comparison of the observed HHL post-selection rate with its analytic probability within a five-standard-error binomial tolerance.

## Run

From the repository root, create and activate the HHL environment, then launch the notebook:

```bash
conda env create -f projects/HHL/environment.yml
conda activate hhl-linear-solvers
jupyter lab projects/HHL/Quantum_Algorithms_for_Solving_Linear_Systems_HHL_and_Variational_Methods.ipynb
```

Run the same schema check and notebook execution used by HHL CI with:

```bash
python projects/HHL/run_notebook_checks.py
```

The dedicated GitHub Actions workflow runs these checks when HHL files change.

## Scope and limitations

This is an educational simulation, not a scalable solver or hardware result. HHL uses a classical eigendecomposition to construct controlled evolution and calibrate phase patterns, and initializes the input state directly. Its measured output is reconstructed from probabilities, which yields amplitudes' magnitudes only; the chosen reference solution is nonnegative, but general signed or complex solutions require phase-sensitive measurements. VQE and QAOA use exact statevectors and dense matrices. No practical quantum speedup is claimed.
