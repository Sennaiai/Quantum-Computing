# Verification scope

The acceptance command is `python run_checks.py` in the pinned Python 3.11 environment. Reproduce the complete control/data experiments with `python run_benchmarks.py --download-bbb` or supply the pinned artifact using `--bbb-file`. The complete benchmark also records the supervised all-molecule summary and matched classical results in `benchmark_summary.json`. The standalone supervised acceptance command is `python run_learning_pipeline.py prepare --bbb-file /path/to/bbb_martins.tab`; it prepares every valid canonical structure and fits matched classical baselines without quantum API calls. The optional installed-SDK acceptance is `python verify_sdk_contract.py` in the isolated `requirements-sdk.txt` environment. `--skip-notebooks` runs tests only and is not the complete acceptance check. Executed notebooks and a source-hashed `validation_summary.json` are retained under `artifacts/validation/`; they are generated evidence, excluded from Git.

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
| Supervised cohort | Concordant duplicate collapse, conflicting-label quarantine, canonical identity/source-row coverage, strict scaffold isolation, binary labels, deterministic split and source/label identity |
| Frozen encoding and batch | Training-only MI/interactions/scaling; all selected local fields; label/hash binding; out-of-range clipping/flags; every canonical structure represented; immutable physics reuse only for identical fitted theta/settings |
| Results and learning | Job/task/molecule/profile binding; explicit native output mapping; complete shot budget, bit order, normalization and heralding checks; histogram-derived raw Z moments; source isolation; train/validation/test separation; portable JSON model round trip and matched count baselines |
| API adapter / installed SDK | Socket-blocked Triple Alpha 0.2.3 request/outer-response model comparison and four mocked POST operations; lifecycle, late create receipt persistence, terminal failures, deadlines, invalid responses, redirects, credential redaction and native verification guards |
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

Top-4 and top-8 are fixed bounded profiles, distinct from the original top-100 configuration. Seeds are estimator-jitter replicates, not independent molecular cohorts. Exact discrete-count information values provide a method-sensitivity baseline. Top-8 pair-support Jaccard spans 0.643–0.857 across jitter seeds, and 20/140 pair-screen decisions differ from the exact discrete baseline; this limitation is retained rather than described as statistically validated interactions. Label permutations are descriptive diagnostics, not calibrated p-values. Information fitting/scaling in this legacy experiment is exploratory on the supplied table; no held-out predictive performance is inferred. The supervised workflow below fits a separate frozen schema.

`results/benchmark_summary.json` contains source/environment/data identity and complete measured records. `results/manifest.json` binds the published JSON and figures by checksum. Published-evidence checks reject missing files, stale source hashes, policy overclaims, or inconsistent numerical acceptance. CI independently regenerates the experiment after fetching the pinned public artifact, then executes notebooks and tests with experiment connections denied.

## Supervised cohort, all-molecule preparation and readout

The supervised branch is separate from the preserved exploratory benchmark. Canonical isomeric identity removes atom-map annotations but preserves salts, charge, isotopes and stereochemistry. Concordant duplicate observations collapse to one labelled identity. All conflicting-label identities are quarantined from training rather than assigned majority labels; valid structures remain available for unlabelled prediction. Invalid/missing labels cannot enter fitting. The pinned BBB audit yields 1,965 labelled identities and 1,975 inference identities: 54 concordant duplicate rows collapse and 20 rows across 10 conflicting identities are quarantined.

A deterministic Murcko-scaffold grouping produces disjoint train/validation/test sets; exact sizes are recorded in the preparation summary. Scaffold grouping ignores stereochemistry while molecular identity preserves it. All acyclic molecules share the empty-scaffold group; no identity/random split fallback is substituted. Both classes are required in each partition. These are one declared split, not independent statistical replicates or a universal generalization estimate.

Only training labels/counts determine MI selection, KSG interactions and scaling. The frozen schema binds exact training IDs and labels, RDKit/fingerprint settings, count ranges, feature-to-wire map and readout supports. Every selected count receives a local field, including features with no screened pair/triad. Transformations do not read labels. Counts outside training ranges are clipped to [0,1] and flagged; this policy avoids silent refitting but does not establish applicability to a new chemical domain.

All 1,975 valid BBB identities are prepared, including quarantined structures without training labels. A task binds molecular identity and original rows to its CD Hamiltonian, normalization, QSVT phase settings, runtime stream, deployment family and requested shots. Immutable physics bundles may be reused only for identical frozen feature vectors/settings. Every unique bounded instance must pass the independent direct-exponential comparison. Resource failures stop preparation; no prefix sampling or term truncation is substituted.

Measurement records require an explicit job receipt and a deployment-specific output mapping. The normalized order is branch, real-extraction, lambda, index bits (MSB first), then system bits (wire 0 first). All measurement bits, lambda, source identity and total shot count are checked. Successful heralding requires zero branch, extraction and index bits. Raw Z singles/pairs/triads are recomputed from the accepted histogram; connected moments are diagnostic plug-in estimates. Their finite-shot unbiasedness is not claimed. The recorded 1/√N accepted-shot bound applies to each raw ±1 moment's standard error, not a simultaneous confidence interval or connected-correlation uncertainty.

The portable model standardizes on training only and fits logistic regression. Validation selects C from the declared grid (0.1, 1, 10 by default); a separate evaluation receives test labels after fitting. Full-count and selected-count classical baselines use identical partitions and the same tuning grid. Exact training-label hashes must match the encoding, and prediction rejects different feature/profile identities or measurement sources. ROC AUC, average precision, Brier score, log loss, balanced accuracy and a fixed 0.5-threshold confusion matrix are reported. Average precision is not labelled trapezoidal PR AUC. No calibration or uncertainty claim follows merely from a logistic probability.

When no API output records are supplied to the public workflow, the readout specification is `awaiting_api_results`, coefficients are absent and quantum prediction metrics are null. Synthetic dense-QSVT shot tests exercise extraction, fitting, evaluation and portable prediction, with `synthetic_test` provenance. They are software acceptance tests, not quantum ADMET results. The current readout supports binary endpoints; regression endpoints require a separately specified model and validation policy.

## SDK verification and remaining native checks

The project author confirms successful private compilation and end-to-end hardware execution using Horizon Quantum's Triple Alpha workflow. Proprietary deployment artifacts and results remain confidential. This author-confirmed execution is distinct from the independently reproducible public checks below.

The optional dependency set pins Triple Alpha 0.2.3 in a separate environment. `verify_sdk_contract.py` blocks sockets and compares the installed SDK request models and outer response models with four mocked operations. All operations use POST. Create/status responses use `job_status`, while results use `status`. Input values are uniquely named numeric strings; the pinned SDK rejects numeric floats in its update model. URL normalization follows the SDK rather than inventing an operation path. These checks establish compatibility with this installed SDK version, not acceptance by a live service.

The logical flat runtime protocol is an internal scientific serialization. It is not submitted as a vendor named-input list. Each dimension family receives separately generated candidate Beryllium source with unique static input/output names, copied dependency sources and a hash-bound mapping. Compilation, capacities and gate/output semantics must be validated for each actual deployment. Inner SDK result data are untyped; no output-field path is guessed. A verified explicit output mapping is required for hardware/simulator adaptation.

The public checks do not independently verify deployment-specific Beryllium compilation, vendor `RY`/global-phase conventions, deployed program identity/capacity or actual result semantics. Candidate source assumes `RY(θ)=exp(−iθY/2)` and `globalphase(θ)=exp(iθ)`. Protocol v2 must be bound to its corresponding verified deployment. All compiler, gate, deployment, service/input/output verification guards and explicit live opt-in must pass before execution; mocked attestations are test values only.

No live Triple Alpha job, credential validation, hardware ADMET performance, complete ten-endpoint/top-100 study or time-ordered QSVT anneal forms part of public acceptance. The transport experiment is a bounded classical simulation calibrated against independent spectral/numerical references. SDK/static/source checks and passing CI do not establish native or production readiness.

## Primary references

- [RDKit fingerprint generator API](https://www.rdkit.org/docs/source/rdkit.Chem.rdFingerprintGenerator.html): hashed Morgan count construction.
- [scikit-learn mutual_info_classif](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_classif.html): discrete feature selection and nats.
- [Kraskov, Stögbauer and Grassberger, 2004](https://arxiv.org/abs/cond-mat/0305641): nearest-neighbor information estimation.
- [Berry, 2009](https://doi.org/10.1088/1751-8113/42/36/365303): transitionless driving and exact spectral reference.
- [Sels and Polkovnikov, 2017](https://arxiv.org/abs/1607.05687) and [Claeys et al., 2019](https://arxiv.org/abs/1904.03209): variational AGP and nested commutators.
- [Gilyén et al., 2019](https://arxiv.org/abs/1806.01838) and [Martyn et al., 2021](https://arxiv.org/abs/2105.02859): QSVT and phase conventions.
- [pyQSP source](https://github.com/ichuang/pyqsp) and [PennyLane phase conversion](https://docs.pennylane.ai/en/stable/_modules/pennylane/templates/subroutines/qsvt.html#transform_angles): synthesis and convention cross-checks.
- [NIST DLMF 10.14.4](https://dlmf.nist.gov/10.14.E4): Bessel tail majorant.
- [Triple Alpha 0.2.3](https://pypi.org/project/triple-alpha/0.2.3/): installed SDK request/response contract; mocked verification does not validate a deployed program.
- [Therapeutics Data Commons](https://tdcommons.ai/single_pred_tasks/overview/) and [Beryllium overview](https://www.horizonquantum.com/resources/blog/an-introduction-to-beryllium): external data/language context, without implied service-contract validation.
