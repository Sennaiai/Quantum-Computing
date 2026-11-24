# ADMET Counterdiabatic QSVT Pipeline

This project implements an end-to-end workflow for constructing and executing an ADMET counterdiabatic Hamiltonian with LCU qubitization and QSVT on TripleAlpha. Molecular data is curated classically, converted into a runtime Hamiltonian, validated, serialized for the compiled Beryllium circuit, and sent to the selected quantum backend through the Triple Alpha API.

## Run the end-to-end workflow

1. From this project directory, create and activate the Conda environment:

   ```bash
   conda env create -f environment.yml
   conda activate admet-triplealpha
   ```
2. Set `TRIPLE_ALPHA_API_KEY` and `TRIPLE_ALPHA_RUN_URL` in your local environment.
3. Open `ADMET_Triple_Alpha_End_to_End.ipynb` with Jupyter from this directory.
4. Select the dataset, molecule, physical time, QSVT tolerance, and shot count in the configuration cell.
5. Run the notebook from top to bottom.

If an API key was previously embedded in a notebook, revoke or rotate it before reuse.

The notebook performs preprocessing, AGP/CD construction, QSVT phase synthesis, runtime-input validation, job creation, input submission, status polling, and result retrieval.

## Classical pipeline

The canonical derivation notebooks are in `pipeline/`:

- `ADMET_future_curation_complete.ipynb` loads the ten committed local `.tab` datasets from `ADMET_data/data`,
  generates Morgan count fingerprints, performs MI feature selection, computes
  pairwise and conditional mutual information, identifies triadic interactions, and constructs the parameterized Stage-6 Hamiltonian.
- `Counter-diabatic hamiltonian_complete.ipynb` constructs the adiabatic Hamiltonian, nested commutators, variational AGP coefficient, first-order AGP, counterdiabatic Hamiltonian, and resource summary.

When run independently, execute the curation notebook first. It writes the portable handoff artifact `artifacts/curated_admet_hamiltonians.pkl`; the
counterdiabatic notebook loads that artifact and writes
`artifacts/HCD_hamiltonians.pkl` and
`artifacts/HCD_resource_summary.pkl`.

The end-to-end notebook invokes the curation stages directly and performs the molecule-specific AGP/CD calculation without interaction-term truncation.

The repository notebook runner sets `NOTEBOOK_CI=1` for a bounded smoke run
(two datasets, 250 rows each, and 8 selected features). A normal notebook run
uses all ten local datasets and the full 100-feature analysis.

## Quantum runtime

The compiled circuit consists of eight Beryllium modules:

```text
qbe_modules/
├── main.qbe
├── math.qbe
├── prep_mod.qbe
├── sel_mod.qbe
├── block_encode.qbe
├── reflect_mod.qbe
├── walk_mod.qbe
└── qsvt_mod.qbe
```

The modules implement runtime PREPARE, arbitrary controlled Pauli SELECT, inverse PREPARE, LCU block encoding, zero-state reflection, qubitized WALK, and
the complete phased QSVT sequence.

The modules are compiled once. A molecular update changes only the ordered runtime input stream.

## Runtime input

`main.qbe` consumes:

1. Number of Hamiltonian terms.
2. Number of system qubits.
3. Number of index ancillas.
4. Number of QSVT phases.
5. System-state initialization angles.
6. Each (signed) Hamiltonian coefficient followed by its dense Pauli row.
7. The complete QSVT phase sequence.

Pauli rows are flattened into one row-major array. Codes are `0=I`, `1=X`, `2=Y`, and `3=Z`; logical element `(term, qubit)` is addressed by
`(term - 1) * num_qubits + qubit` in Beryllium's one-based indexing.

At runtime, `main.qbe` extracts coefficient signs, calculates `lambda = sum(abs(h_i))`, and forms PREPARE amplitudes `sqrt(abs(h_i) / lambda)` using compiler-safe scalar intermediaries.

## Project layout

```text
Modular_QEML_admet_Pipeline_Senay Kahsay/
├── ADMET_Triple_Alpha_End_to_End.ipynb
├── README.md
├── requirements.txt
├── artifacts/         # generated handoff artifacts
├── pipeline/          # canonical curation and CD derivations
├── src/               # preprocessing, serialization, and validation helpers
├── python_classes/    # validated Hamiltonian and QSVT helpers
├── qbe_modules/       # compiler-validated Beryllium source
└── ADMET_data/data/   # local tab-separated ADMET inputs
```

`runtime_inputs.json` and `runtime_manifest.json` are generated runtime
artifacts. The manifest records dimensions, lambda, QSVT settings, input hashes,
module hashes, AGP metadata, and curation provenance.

## Triple Alpha execution

The API lifecycle is:

```text
run/create
→ run/get_status
→ awaiting_input
→ run/update_input
→ run/get_status
→ run/get_results
```

## Reproducibility

The statistical thresholds, fingerprint configuration, feature-selection seed, Hamiltonian interpolation, `lambda_dot`, physical evolution time, QSVT error
tolerance, and shot count are explicit in the notebooks. The notebooks have cleared outputs and portable paths and do not depend on hidden kernel state.

A native Beryllium compiler is not included. Compile and deploy the QBE modules in Triple Alpha before running the API portion of the end-to-end notebook.