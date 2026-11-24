# HCN validation record

Local validation: 2026-11-24, Python 3.11.15, macOS ARM64. This is a fixed-orbital
CAS(4,4) simulation with exact circuit expectations. HCN was committed as `3f6f826` after H2 commit `ad85a0c`.

## Reproduction and checks

A new, standalone HCN environment was installed from `requirements-hcn.txt`.
Installation and `pip check` passed. Its resolved macOS ARM64 package set is
saved in `requirements-hcn-macos-arm64.lock`; Linux reproduction remains pending.

```bash
/opt/homebrew/bin/python3.11 -m venv /private/tmp/hcn-review-env
/private/tmp/hcn-review-env/bin/python -m pip install -r requirements-hcn.txt
/private/tmp/hcn-review-env/bin/python -m pip check
MPLCONFIGDIR=/private/tmp/h2-matplotlib \
  /private/tmp/hcn-review-env/bin/python run_hcn_checks.py --save-review
MPLCONFIGDIR=/private/tmp/h2-matplotlib \
  /private/tmp/hcn-review-env/bin/python -O -m unittest -q \
  test_h2_benchmark test_hcn_benchmark test_notebook_discovery
```

The smoke profile was previously executed in the clean H2 environment with the
same numerical package versions, using `HCN_PROFILE=smoke` and
`HCN_OUTPUT_DIR=artifacts/hcn-smoke`. It remains an execution/symmetry check,
not an optimization certificate.

Fifteen HCN tests cover active-space character, matched references, integral
axes and constant counting, full-FCI control, invalid inputs, number/spin
conservation, fixed-gate initialization, best-state checkpoints, optimizer and
reference failures, corrupt results, adjoint gradients versus finite differences,
the PBE integration grid, saved-reference regression, and replay of all nine
saved best-parameter sets. Five discovery tests cover H2/HCN delegation,
relocated coursework, resource/data exclusions, and generated artifacts.

The default notebook completed all three C–N geometries and all three seeds
with strict checks enabled. The smoke profile has a 30-iteration budget and
permits explicitly recorded nonconvergence; it is not an optimization certificate.
The combined 36-test suite passed under optimized Python. It emitted a
nonfatal temporary-file `ResourceWarning`; this remains a library/resource-lifecycle
observation, not a failed numerical check. Schema checks,
Python compilation, workflow YAML/paths, and `git diff --check` are additional
engineering checks, separate from the numerical results.

## Model and observed results

Geometry is linear H–C–N, fixed H–C=1.065 Å. C–N samples are 1.067, 1.167, and
1.267 Å. Charge is zero, spin singlet, basis STO-3G. Fourteen electrons occupy
11 spatial orbitals. Five occupied orbitals are frozen; occupied π and virtual
π* pairs (indices 5–8) form CAS(4,4). Occupations and C2v B1/B2 labels are checked
at every geometry, so changed active-orbital ordering fails explicitly.

Effective integrals come from converged PySCF RHF/CASCI. The frozen-core scalar
includes nuclear repulsion and is added once. PySCF chemist ERIs are converted
to PennyLane axes before fermionic assembly and Jordan–Wigner mapping. The
36-determinant Nalpha=Nbeta=2 matrix is diagonalized and its singlet character
verified. CASCI and qubit diagonalization share integrals; agreement is a
mapping/solver cross-check, not independent integral-engine validation.

| Default result | Observed |
|---|---:|
| Optimizer successes | 9 / 9 |
| Maximum VQE–sector-exact gap | 3.342e-10 Ha |
| Maximum energy variance | 2.823e-10 Ha² |
| Maximum absolute active-electron count error | 6.218e-15 |
| Maximum absolute VQE spin expectation | 1.949e-18 |
| Sector leakage | 0 |
| Maximum mapped-sector/CASCI difference | 2.274e-13 Ha |

The six-layer GateFabric circuit has 36 parameters and `include_pi=True`.
Fixed π gates rearrange occupations: zero parameters do not give the HF state.
HF preparation is checked separately. L-BFGS-B uses adjoint gradients and a
300-iteration cap. Every energy evaluation, status, and matching best-parameter
checkpoint is saved, including initial and final evaluations. A small gap alone
never overrides a failed optimizer status in strict validation.

At C–N=1.167 Å:

| Method/control | Total energy / Ha |
|---|---:|
| Full-basis RHF | -91.67459259588719 |
| CAS(4,4) CASCI | -91.77508487838224 |
| Full STO-3G FCI, selected A1 sector | -91.84392763770316 |
| Full-basis RKS-PBE, grid 5 | -92.09749120250225 |

CASCI is **68.842759 mHa above full FCI**, separating active-space truncation
from the much smaller variational solver error. The full-FCI control uses a
direct solver and a 330×330 CI coefficient array, never a dense 22-qubit
Hamiltonian. It selects the RHF A1 irrep and verifies singlet character; it does
not search all spatial-symmetry sectors. Its memory setting is a requested
1000 MB budget, with at most 100 solver cycles and 12 Davidson-space vectors.

The PBE grid 3→5 shift is +1.125740e-6 Ha, which failed the initial 1e-6 Ha control.
Grid 5→7 shifts by -1.933347e-8 Ha and passes; grid 5 is the study baseline.
Full-basis PBE comparisons are signed method comparisons, not variational bounds
or measured experimental errors.

## Rejected backend and resolved failures

The PennyLane 0.36 default differentiable-HF path fails reference agreement for
this geometry. The optional reproducible diagnostic is:

```bash
MPLCONFIGDIR=/private/tmp/h2-matplotlib \
  /private/tmp/h2-review-env/bin/python diagnose_hcn_backend.py
```

The public `qchem.hf_energy` call returned -95.16203527167147 Ha, differing from
converged PySCF RHF by -3.48744267578428 Ha. Its finite return value is not
accepted as convergence evidence. The observed rejection is saved in
`benchmarks/hcn/dhf_diagnostic.json`. This finding is specific to this pinned
backend and model; the validated study uses converged PySCF integrals.

The initial PBE grid test failed and prompted the measured grid refinement
above. The macOS PySCF wheel lacks OpenMP; requested thread settings do not
establish actual parallel behavior. Notebook kernels require authorized local
socket access outside the filesystem sandbox.

## Evidence and remaining limits

The notebook contains executed outputs. `benchmarks/hcn/` stores all nine seed
outcomes, best parameters, evaluation traces, full-FCI/grid controls, plots,
software versions, and the benchmark source digest. Each geometry also stores
its active one-/two-electron tensors and core scalar. Checkpoint replay uses
these recorded tensors, avoiding assumptions about the signs or ordering of
fresh RHF orbitals inside degenerate π pairs. Fresh scalar reference energies
are checked separately against the saved results. Both plots were visually
checked for readable labels and agreement with the recorded values.

The σ framework and two external virtual orbitals are omitted from the active
model, RHF orbitals are not optimized as CASSCF orbitals, the basis is minimal,
and H–C is fixed. A three-point local scan does not locate a converged minimum
or demonstrate full triple-bond dissociation, binding affinity, thermochemistry,
or experimental accuracy. Orbital symmetry/occupation guards do not substitute
for general orbital-overlap tracking on long reaction paths. No noise, shots,
quantum hardware, measurement-cost scaling, or quantum advantage is assessed.
Grid convergence was tested at the reference geometry only. Remote Linux CI
and a second-machine reproduction remain pending. Active-space scaling remains
unchanged and awaits its own refinement after HCN review.

## H2-standard review

The final review retained the complete scientific chain while clarifying
spatial versus spin-orbital indices with explicit spin sums, stating that CASCI
orbitals are fixed rather than CASSCF-optimized, and distinguishing evaluated
statevector observables from hardware measurements. The concise Markdown now
answers the scientific question with the observed solver/truncation separation.
Saved-evidence regression and orbital-basis-aware parameter replay close the
remaining test-parity gap with H2. Model and solver choices were not changed.

## Remote CI follow-up

Draft PR #5 first exposed a workflow path failure before dependencies or
scientific tests could run: GitHub Actions trimmed the terminal space from the
project directory name. Both chemistry working directories now end in `/.`,
preserving that space as an internal path component. A regression test checks
that trimming the configured path still resolves to an existing directory.
Remote verification is pending the corrected run.
