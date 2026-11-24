# Validation and scientific scope

Validated locally on 2026-11-24 with Python 3.11.15, macOS ARM64. The three
studies share pinned software, not notebook state. Numerical benchmark JSON
contains model inputs, solver settings, reference values, diagnostics, source
hashes and versions. HCN/scaling retain effective integral tensors so saved
results can be replayed without assuming identical orbital phases across hosts.
Plots are embedded in the notebooks; generated copies stay in ignored `artifacts/`.

## Reproduce

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
python run_checks.py --save-review
python -O run_checks.py --tests-only
```

The full check runs scientific, saved-evidence, runner-safety and repository
policy tests, then executes each selected notebook in a fresh kernel using the
same interpreter. Failed execution never replaces published review snapshots.
Non-default profiles cannot be published by `--save-review`. Generated output
cannot target source files or `benchmarks/`. Dependency pins and the resolved
platform-specific lock are version records, not a universal or hash lockfile.

Additional execution controls:

```bash
H2_PROFILE=extended H2_OUTPUT_DIR=artifacts/h2-extended python run_checks.py --study h2
HCN_PROFILE=smoke HCN_OUTPUT_DIR=artifacts/hcn-smoke python run_checks.py --study hcn
SCALING_PROFILE=smoke SCALING_LARGE_OPERATORS=1 SCALING_OUTPUT_DIR=artifacts/scaling-large-smoke python run_checks.py --study scaling
```

The HCN smoke budget deliberately permits recorded nonconvergence while retaining
finiteness and physical checks. It does not certify optimizer accuracy.

## H2

Neutral singlet H₂, explicit angstrom geometry, STO-3G, no frozen orbitals.
PennyLane molecular integrals and mapped-sector diagonalization are compared
with independently implemented PySCF RHF/FCI at every sampled geometry. A
single double-excitation circuit is optimized without using reference energies.
Checks cover the HF determinant, Hermiticity, particle sector, total spin,
energy variance, analytical two-state formulas and wire-order conventions.

- Seven default bond lengths and a 22-point extended scan were executed.
- Independent exact energies agree within 1.7e-9 Ha; the acceptance tolerance
  is 2e-7 Ha. VQE agrees with its exact reference to floating-point precision.
- At 0.74 Å, STO-3G FCI is -1.137283834489 Ha and cc-pVDZ FCI is
  -1.163374490319 Ha. This is basis sensitivity, not a complete-basis error.
- PBE integration-grid and FCI natural-occupation controls distinguish method
  behavior and stretched-bond correlation from solver error.
- The historical rounded Pauli table is tested as algebra only: molecular
  geometry/integral provenance is unknown. It is not a literature energy check.
- Hermitian Pauli coefficients are explicitly validated before real-dtype
  normalization; no complex contribution is silently discarded.

## HCN

Linear neutral singlet HCN, fixed H–C=1.065 Å; C–N samples 1.067/1.167/1.267 Å.
STO-3G has 14 electrons and 11 spatial orbitals. Five occupied orbitals are
frozen; occupied/virtual π pairs form CAS(4,4). Orbital symmetry and occupations
are checked. Canonical RHF orbitals are not CASSCF-optimized.

- Nine six-layer GateFabric/L-BFGS-B runs (three seeds at three geometries)
  converged. Maximum VQE gap is 3.35e-10 Ha; variance is below 2.83e-10 Ha².
- Acceptance requires optimizer success, gap below 1e-6 Ha, variance below
  1e-7 Ha², particle number four, singlet character and sector leakage below
  1e-12. Best parameters, statuses and every evaluated energy are retained.
- The mapped 36-determinant sector agrees with CASCI within 2.3e-13 Ha.
  These solvers share PySCF integrals; this is a mapping/solver cross-check.
- At the reference geometry, CASCI is -91.775084878382 Ha and full STO-3G FCI
  is -91.843927637703 Ha: active-space truncation is 68.842759 mHa. The full
  reference selects the A1 spatial sector and verifies a singlet; it does not
  search every irrep. Direct FCI uses a 330×330 coefficient array, not a dense
  22-qubit Hamiltonian.
- Full-basis RHF/PBE are separate method baselines. PBE grid 3→5 missed a
  1e-6 Ha control; grid 5→7 agreed within 1.94e-8 Ha, so grid 5 is used.
- Fixed π gates mean zero GateFabric weights do not reproduce HF. Raw HF is
  checked separately. Saved checkpoints replay in their recorded orbital basis.
- The pinned PennyLane differentiable-HF backend failed reference agreement
  for this geometry. `python diagnose_hcn_backend.py` reproduces the rejected
  result; its small diagnostic JSON is retained with the HCN evidence.

## Scaling

Fixed NH₃/H₂CO₃/HCO₃⁻ geometries, charges 0/0/−1, singlets, STO-3G. Full-space
counts are 16/44/42 qubits. These are metadata; full acid/anion operators are
not constructed. Fixed RHF orbitals define nested CAS(2,2)/(4,4)/(6,6) spaces.
Nuclear and frozen-occupied constants are counted once.

- All nine default cases agree with CASCI within 6.3e-13 Ha. Exact qubit checks
  use sparse Pauli action and at most a 400×400 physical-sector block.
- Guards precede allocation: maximum 12 exact-check qubits, 400 determinants
  and 2 million sparse contributions. Explicit opt-in adds 16-qubit operators
  with HF checks; their qubit spectra are not diagonalized. All three opt-ins ran.
- The 12-qubit cases contain 819–1259 Pauli terms and 239–358 deterministic
  greedy QWC settings. These depend on orbitals, mapping, cutoff and heuristic;
  they are neither optimal group counts nor shot/gate/runtime estimates.
- Differences from containing references remain 24.047258/6.991073/32.344915
  mHa for NH₃/acid/anion. NH₃ uses full STO-3G FCI; acid/anion use larger
  CAS(8,8) references, so their differences are not full-basis errors.
- Identity coefficients, core scalars and molecular energies are kept distinct.
  PBE grid 5→7 checks passed for all three geometries within 1e-6 Ha.
- Acid/anion geometries were generated with RDKit 2026.09.2, ETKDGv3 seed 1729
  and converged UFF; regeneration reproduced the stored coordinates exactly.
  `regenerate_scaling_geometries.py` is optional and needs that separate RDKit
  version; the default study reads the fixed JSON with no RDKit dependency.

## Verification status and limits

CI requires all three scientific jobs, shared notebook execution, and the
structure-only check. The current commit status is authoritative on
[PR #5](https://github.com/Sennaiai/Quantum-Computing/pull/5). Chemistry working
directories end in `/.` because GitHub Actions trims terminal whitespace;
the discovery regression protects the existing folder name.

Local full/default, H₂ extended, HCN smoke, scaling smoke/large-operator, dependency,
syntax, notebook schema, local-link and diff checks have passed. **55 tests passed** locally, including saved-evidence replay and runner safety,
both normally and under optimized Python. All three default notebooks were
re-executed in fresh kernels after public cleanup. Remote verification is attached to the current PR head, rather than inferred
from an earlier passing commit. A macOS PySCF wheel lacks
OpenMP; this affects thread-setting requests, not validated energies. Earlier
H₂/HCN dependency cleanup can emit nonfatal temporary-file ResourceWarnings.
Warning messages are retained; machine-specific environment prefixes are removed
from published notebook outputs.

All models use a minimal basis except the one H₂ basis control. Fixed geometries,
frozen σ orbitals in HCN, frontier active-space choices, and missing diffuse
functions for the anion limit chemical interpretation. Orbital symmetry guards
are not general reaction-path overlap tracking. No full dissociation curve,
thermochemistry, binding affinity, experimental validation, noisy measurements,
hardware performance, asymptotic scaling or practical quantum advantage was
established. Passing these checks supports the stated bounded computations;
it cannot establish absolute correctness or production readiness.
