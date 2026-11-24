# H2 refinement validation

Validated locally on 2026-11-24 on macOS ARM64. The final reproduction used a
clean Python 3.11.15 virtual environment installed from `requirements-h2.txt`.
HCN and active-space scaling remain pending their own scientific refinement.

## Commands and outcomes

From this project directory, the clean-environment setup was:

```bash
/opt/homebrew/bin/python3.11 -m venv /private/tmp/h2-review-env
/private/tmp/h2-review-env/bin/python -m pip install -r requirements-h2.txt
/private/tmp/h2-review-env/bin/python -m pip check
```

Installation passed; `pip check` reported no broken requirements. The direct
pins and complete resolved macOS ARM64 package set are saved in
`requirements-h2.txt` and `requirements-h2-macos-arm64.lock`. The lock records
this tested environment; it is not a universal cross-platform or hash lock.

Final execution commands:

```bash
MPLCONFIGDIR=/private/tmp/h2-matplotlib \
  /private/tmp/h2-review-env/bin/python run_h2_checks.py --save-review

MPLCONFIGDIR=/private/tmp/h2-matplotlib H2_PROFILE=extended \
  H2_OUTPUT_DIR=artifacts/h2-extended \
  /private/tmp/h2-review-env/bin/python run_h2_checks.py

/private/tmp/h2-review-env/bin/python -m unittest -v test_notebook_discovery

MPLCONFIGDIR=/private/tmp/h2-matplotlib \
  /private/tmp/h2-review-env/bin/python -O -m unittest -v \
  test_h2_benchmark test_notebook_discovery
```

- Sixteen scientific tests passed. Coverage includes the historical invariant
  block and coupling, computational-basis expectations, evolution formulas,
  geometry units, nuclear repulsion, invalid inputs, independent molecular
  references, natural occupations, saved evidence, and rejection of solver
  failures, inconsistent reporting, nonfinite energies, spin contamination,
  electron-number errors, and variational-bound violations.
- Notebook schema validation and fresh-kernel execution passed for both the
  default seven-point scan and the extended 22-point scan (0.45–3.0 angstrom).
- Four repository-discovery tests passed: generated artifacts are excluded,
  H2 delegates to dedicated CI, relocated coursework stays explicitly
  unvalidated, and existing resource/data policies are preserved.
- All 20 tests also passed under optimized Python (`-O`), confirming that the
  acceptance checks do not disappear with ordinary Python assertions disabled.
- All 24 current source notebooks, including the four relocated notebooks,
  passed `nbformat.validate` on their JSON. This is structural validation only.
- The relocated MOPSO notebook passed fresh-kernel execution in the clean
  environment using `/private/tmp/check_moved_mopso.py`, which supplied the
  notebook path and bound the kernel to the same interpreter. Its executed copy
  is `/private/tmp/Unit_Commitment_MOPSO-executed.ipynb`. Scientific correctness
  was not assessed; that notebook awaits its own refinement.
- Workflow YAML parsed with `yaml.BaseLoader`; its event keys and actual
  working-directory path were checked. Remote GitHub Actions execution remains
  pending. `git diff --check` and Python compilation checks passed.

## Scientific evidence

The reviewed default outputs are saved directly in the H2 notebook, and
`benchmarks/h2/` contains JSON, both plots, software versions, solver settings,
and the SHA-256 digest of `h2_benchmark.py`. Generated execution copies remain
in ignored `artifacts/h2/` and `artifacts/h2-extended/`.

| Diagnostic | Default: 7 points | Extended: 22 points |
|---|---:|---:|
| Maximum absolute PennyLane/PySCF exact-energy difference | 1.681e-9 Ha | 1.681e-9 Ha |
| Maximum absolute VQE/sector-exact energy difference | 4.441e-16 Ha | 6.662e-16 Ha |
| Maximum VQE energy variance | 9.765e-17 Ha² | 3.123e-16 Ha² |
| Maximum VQE spin expectation, absolute | 0 | 0 |
| Sector leakage | 0 | 0 |

Electron number is two within 1e-12 at every point. All RHF, RKS-PBE, and FCI
calculations reported convergence. HF determinant energies from the qubit
Hamiltonian also match PySCF RHF within the reference tolerance. VQE optimizes
a circuit Hamiltonian expectation and separately checks that expectation
against a dense matrix; exact reference energies do not drive the optimizer.

At 0.74 angstrom, PySCF STO-3G FCI is -1.137283834489 Ha and cc-pVDZ FCI is
-1.163374490319 Ha. Their difference is a basis-sensitivity control, not a
complete-basis error estimate. PBE grid-5 minus grid-3 energy is -4.798e-10 Ha
at this geometry. Natural occupations change from approximately (1.975, 0.025)
at 0.74 angstrom to (1.075, 0.925) at 3.0 angstrom, supporting the discussion
of increasing static correlation within this model.

Both plots were visually inspected for readable labels, units, and consistency
with the exported values. The notebook's LaTeX escape sequences and matrix row
separator were corrected during review. The final notebook has concise scientific
Markdown and an explicit worked example of MO integrals → fermionic assembly →
Jordan–Wigner mapping → HF/excitation circuit → optimization and reference
validation. Its added integral/mapping regression test passed. Run details are
in the dedicated, concise `README_H2.md`. Tiny VQE energy differences indicate
floating-point precision, not experimental accuracy.

## Failures encountered and resolved

- Sandboxed package downloads failed on DNS resolution; authorized network
  access succeeded.
- Latest PySCF 2.15.1 attempted a source build and failed on a missing libxcfun
  patch. The pinned 2.11.0 ARM64 binary wheel installed successfully.
- Initial molecular tests exposed older PennyLane API signatures; explicit
  angstrom-to-bohr conversion and the pinned API resolved them.
- Fresh-kernel execution initially failed because the sandbox blocks local
  socket binding; authorized execution outside the sandbox succeeded.
- A negative-gap rejection test initially conflated an inconsistent reported
  gap with an internally consistent variational-bound violation. Separate tests
  now cover both cases, and both pass.
- The PySCF wheel has no OpenMP support. Its thread-setting call has no effect;
  no parallel scaling or comparative wall-clock performance claim is made.

## Remaining limits and review status

H₂ is a standard example; the scope is a reproducible, bounded molecular
benchmark. Claims about larger molecules require their own validation.

Linux CI, a second machine, noisy measurements, and hardware execution have
not been checked. DFT controls use restricted PBE; functional sensitivity,
unrestricted symmetry breaking, grid convergence along the entire curve,
complete-basis extrapolation, relativistic corrections, and vibrational/thermal
corrections remain outside scope. The historical coefficient table has no
verified molecular provenance and is evaluated only as an algebra exercise.
No production-readiness, chemical realism, or quantum-advantage claim follows
from passing these checks.

## Combined commit scope

The user confirmed that unmatched assignment deletions are intentional. Include
these structural changes with H2 and its supporting files:

- Four byte-identical notebook moves: the PennyLane and Qiskit arithmetic
  notebooks into `projects/QFT/`, QUAC605 Assignment2 to
  `projects/Shor_Senay_Kahsay.ipynb`, and MOPSO into `projects/Optimization/`.
- Move `theory/{algorithms,basics,models}.md` to `resources/theory/`.
- Remove the old HCN/HHL assignment copies, QUAC-603, both Assignment-3 files,
  QPSO, and Untitled-1; remove the empty conferences/reading/tools placeholders.
- Update the root README and shared notebook discovery policy; add dedicated
  H2 CI with dependency checks, scientific tests, execution, and evidence upload.

Do not include local forecasting spreadsheets, reports, archives, ADMET
artifacts, optimization project documents, or Python caches. HCN and active-space
notebook contents are unchanged. The combined commit is authorized. Publishing a PR, merging, and releasing
remain separate steps; human review is required before merge or release.
