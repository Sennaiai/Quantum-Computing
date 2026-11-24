# Verification scope

The acceptance command is `python run_checks.py` in the pinned Python 3.11 environment. Reproduce the complete control/data experiments with `python run_benchmarks.py --download-bbb` or supply the pinned artifact using `--bbb-file`. `--skip-notebooks` runs tests only and is not the complete acceptance check. Executed notebooks and a source-hashed `validation_summary.json` are retained under `artifacts/validation/`; they are generated evidence, excluded from Git.

## Independent references

| Layer | Comparison and failure checks |
| --- | --- |
| Molecular representation | Hashed Morgan count multiplicity, atom-order invariance, invalid/empty SMILES, binary labels, original-row alignment, duplicate/conflicting identities |
| Information screening | Exact discrete MI reference, analytic Gaussian MI probe, redundant and XOR co-information signs, constant guards, deterministic seeds/ties, caller RNG restoration, budget rejection |
| Hamiltonian binding | Original fingerprint → selected position → compact qubit mapping; molecule counts rather than MI weights; source-table/fingerprint hashes and scaler provenance |
| AGP/CD | Pauli products/commutators against independent matrices; one-qubit exact AGP; two-qubit variational stationarity; commuting and ill-conditioned paths; complete retained interactions |
| LCU primitives | Multiplexed RY tree and inverse, zero padding, signed X/Y/Z/identity SELECT, block-encoding identity, Hermiticity, walk inverse |
| QSVT | Two parity branches; degree/query/phase counts; independent 2×2 response reconstruction; bounded full coherent ancilla unitary versus SciPy `expm`; recovered error and heralding probability |
| Default molecular evolution | Four system qubits and all 46 CD terms; every eigenvalue and direct `expm`, without allocating the full 4096-dimensional ancilla unitary |
| Transport experiment | Bare / production first-order / exact ground-projector CD; five durations; fidelity, excitation energy, norm defect, control cost, sampled gap; independent midpoint-expm refinement and three frozen QSVT snapshots |
| Real BBB data | Pinned official raw artifact; cleanup and identity diagnostics; fixed top-4/top-8 profiles; five jitter seeds; exact discrete-count CMI/co-information and descriptive label permutations; complete offline runtime handoff where interactions are retained |
| Adversarial inputs | Complex/text labels, portable feature IDs, tiny-norm normalization, underflow/overflow, nonfinite clocks/deadlines, strict scientific JSON scalars |
| Runtime | Hand-calculated golden stream, independent decoder, strict finite values/dimensions/codes, register capacity, missing/trailing values, checksums, source/read-order mutations |
| Persistence | Content identity, strict JSON, source hashes, payload/settings/stream consistency, atomic publication, corruption and partial-write rejection, source/output separation |
| API adapter | Injected transport/clock only: full lifecycle, terminal failures, timeout/deadline, invalid responses, endpoint variants, redirects, credential redaction, explicit opt-in and verification guards |
| Notebooks | Schema validation followed by fresh-kernel execution of all three, using the same interpreter and denied external connections |

The shared repository notebook check covers structure only. The project runner performs scientific checks and executes notebooks. Numerical tests are bounded examples and regressions; passing them cannot prove correctness for every input.

## Approximation and resource policy

- The first-order AGP minimizes a normalized Hilbert–Schmidt variational objective within its nested-commutator ansatz. It is generally approximate. A frozen CD Hamiltonian is not a time-ordered annealing experiment.
- Count-feature KSG estimates with tiny seeded jitter preserve the intended exploratory method, but continuous estimators on tied discrete data are approximations. Screens do not carry null calibration or multiple-testing control.
- Positive triad co-information measures redundant shared dependence under the chosen convention. Negative estimates remain diagnostics; the method does not claim to optimize synergy.
- Pair/triad estimator calls, fingerprint allocations, commutator products, polynomial degree, runtime stream sizes, and dense references have explicit limits. Exceeding a budget raises a diagnostic rather than silently changing the Hamiltonian.
- All nonzero retained coefficients participate in AGP algebra. Numerical imaginary residual checks are distinct from coefficient pruning. Ill-conditioned floating-point norms fail explicitly.
- QSVT certification combines a Jacobi–Anger tail majorant, Chebyshev coefficient L1 reconstruction, and a float64 allowance. The recovered operator target is divided by the recorded success amplitude β. No amplitude amplification, deterministic-success, native depth/noise, or speedup claim is made.
- Large instances can explicitly skip the exponential matrix reference; this omission is recorded. It does not change the analytic/numerical phase acceptance criteria or enable the API.

## Quantum-control experiment and scientific acceptance

The information/feature Ising encoding uses a chosen dimensionless energy scale with ℏ=1. Count and information weights are not calibrated electronic energies. The transport experiment starts in the unique transverse-driver ground state, separately from the runtime protocol's feature RY initialization.

The cubic schedule is s(u)=3u²−2u³, u=t/T. The Schrödinger generator in u is T H(s)+s′(u)A(s), with zero endpoint schedule velocity. Five durations (0.2, 0.5, 1, 2, 5) compare bare annealing, the unchanged first-order Pauli AGP, and the independent exact ground-projector reference i[∂P_g,P_g]. The latter requires an isolated ground state but permits degenerate excited states; it is state-specific and dense, not a full-spectrum or efficient implementation claim.

- Every integration must complete, remain finite, and conserve norm within 10⁻⁷. Exact-reference ground infidelity must remain below 10⁻⁷ on the sampled trajectory.
- The production AGP is compared with independently assembled matrix commutators. Its Hilbert–Schmidt action must not exceed the zero-gauge action. Ground-state transport fidelity is reported without assuming it improves at every duration.
- The ground gap is checked relative to max(1,‖H‖₂) at spectral evaluations. The reported minimum is observed/sampled, not a rigorous bound over the whole continuous path.
- A separate midpoint-exponential propagator is refined at 32, 64, and 128 steps for the duration nearest T=1. It must confirm the adaptive ODE state within the configured tolerance and improve overall under refinement. This refinement study covers one representative duration; norm and exact-reference checks cover the duration sweep.
- Report sampled peak control norm and quadrature of control Hilbert–Schmidt norm squared, including their 1/T cost scaling. They are numerical grid/quadrature measurements, not certified continuous maxima or hardware gate costs.
- Three frozen CD snapshots at schedule fractions 0.25, 0.5, and 0.75 must satisfy the independent direct-exponential error check and requested QSVT tolerance. They do not constitute a time-ordered QSVT circuit.

The measured first-order result is slightly worse than bare annealing at T=5. The negative ablation is retained in the report and figure. It motivates higher-order/ground-weighted research extensions rather than a universal superiority claim.

## Real-data identity and sensitivity

The official artifact is Harvard Dataverse file 4259566, downloaded from `https://dataverse.harvard.edu/api/access/datafile/4259566`. Its pinned SHA-256 is `eb32676c7bf6e2bcd98b7625186bbb2ba4e4644c1d1468dfa18b6be665bd5e98` (137,899 bytes). The raw table has 2,039 rows; the TDC documentation separately states 1,975 drugs. The benchmark records 2,039 valid rows, 1,975 canonical identities, 64 duplicate observations, and 10 conflicting-label identities, with no rejected SMILES. The unique count matches the documentation's stated number, but its processing definition is not inferred; all raw observations remain in the declared exploratory fit. Only derived results and attribution are committed, not the raw table.

Top-4 and top-8 are fixed bounded profiles, distinct from the original top-100 configuration. Seeds are estimator-jitter replicates, not independent molecular cohorts. Exact discrete-count information values provide a method-sensitivity baseline. Top-8 pair-support Jaccard spans 0.643–0.857 across jitter seeds, and 20/140 pair-screen decisions differ from the exact discrete baseline; this limitation is retained rather than described as statistically validated interactions. Label permutations are descriptive diagnostics, not calibrated p-values. All information fitting/scaling is exploratory on the supplied table; no held-out predictive performance is inferred.

`results/benchmark_summary.json` contains source/environment/data identity and complete measured records. `results/manifest.json` binds the published JSON and figures by checksum. Published-evidence checks reject missing files, stale source hashes, policy overclaims, or inconsistent numerical acceptance. CI independently regenerates the experiment after fetching the pinned public artifact, then executes notebooks and tests with experiment connections denied.

## External checks not performed

Native Beryllium compilation, vendor gate-angle semantics, exact deployed program identity/capacity, and the HTTP method/schema/status contract are unverified. Candidate source uses standard `RY(θ)=exp(−iθY/2)` and `globalphase(θ)=exp(iθ)` conventions. Their validity in the vendor compiler must be established before deployment. Protocol v2 requires a new deployment.

No live Triple Alpha job, credential validation, native hardware measurement, complete ten-endpoint/top-100 study, predictive model evaluation, or time-ordered QSVT annealing circuit forms part of acceptance. The new time-dependent experiment is a bounded classical simulation, separately calibrated against exact spectral and numerical references. Native compiler and service conclusions are not inferred from these simulations, source hashes, or structural checks.

## Primary references

- [RDKit fingerprint generator API](https://www.rdkit.org/docs/source/rdkit.Chem.rdFingerprintGenerator.html): hashed Morgan count construction.
- [scikit-learn mutual_info_classif](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_classif.html): discrete feature selection and nats.
- [Kraskov, Stögbauer and Grassberger, 2004](https://arxiv.org/abs/cond-mat/0305641): nearest-neighbor information estimation.
- [Berry, 2009](https://doi.org/10.1088/1751-8113/42/36/365303): transitionless driving and exact spectral reference.
- [Sels and Polkovnikov, 2017](https://arxiv.org/abs/1607.05687) and [Claeys et al., 2019](https://arxiv.org/abs/1904.03209): variational AGP and nested commutators.
- [Gilyén et al., 2019](https://arxiv.org/abs/1806.01838) and [Martyn et al., 2021](https://arxiv.org/abs/2105.02859): QSVT and phase conventions.
- [pyQSP source](https://github.com/ichuang/pyqsp) and [PennyLane phase conversion](https://docs.pennylane.ai/en/stable/_modules/pennylane/templates/subroutines/qsvt.html#transform_angles): synthesis and convention cross-checks.
- [NIST DLMF 10.14.4](https://dlmf.nist.gov/10.14.E4): Bessel tail majorant.
- [Therapeutics Data Commons](https://tdcommons.ai/single_pred_tasks/overview/) and [Beryllium overview](https://www.horizonquantum.com/resources/blog/an-introduction-to-beryllium): external data/language context, without implied service-contract validation.
