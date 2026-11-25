# QFT Arithmetic and Shor Order Finding

Two executable scientific studies share a verified Fourier-transform implementation. They emphasize coherent phases, register conventions, reversible arithmetic and independently checked phase-estimation results.

| Notebook | Execution evidence |
| --- | --- |
| [QFT arithmetic](QFT_Arithmetic_Validation.ipynb) | Merged Qiskit/PennyLane study; full Fourier matrices, controlled-phase regression, modular adders/multipliers/exponentiation, clean workspace and matched ripple-adder comparison |
| [Shor order finding](Shor_Senay_Kahsay.ipynb) | Gate-built QFT arithmetic factors 15; compiled-oracle comparison factors 21; valid order with failed factor extraction retained as a negative control |

## Reproduce

Python 3.11, from this folder:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_checks.py
```

Acceptance runs scientific tests, measured experiments and both notebooks in fresh kernels. Generated evidence stays in ignored `artifacts/validation/`; Python experiment connections are disabled. CI independently repeats execution. Published [JSON and figures](results/) bind the sources and dependencies by checksum.

## Scientific contracts

`F[y,x]=exp(2πixy/2ⁿ)/√2ⁿ`; qubit 0 is least significant. PennyLane wire ordering is explicitly normalized to this convention. Full complex matrices and independent FFT states check phases and ordering. A forward/inverse round trip alone cannot establish agreement with the ideal QFT.

Fourier arithmetic uses P/CP gates: replacing them with RZ/CRZ can alter controlled-branch phases while leaving isolated basis probabilities unchanged. Modular arithmetic promises residue inputs and zero work registers; in-place multiplication requires coprimality. Composite moduli use a verified modular inverse. Original arithmetic operations are consolidated into tested shared functions, with quantum multiplication expressed as reversible accumulation.

The Shor study uses reversible QFT modular arithmetic for N=15. N=21 uses explicitly compiled small-instance permutation matrices, fixing unused basis codes; this is not a scalable modular-arithmetic implementation. Continued-fraction candidates come from measured phases and undergo modular verification, without a blind classical order-search fallback. Base 5 finds order 6 modulo 21 but fails factor extraction; that outcome remains visible.

## Scope

Ideal simulation of bounded established algorithms; no hardware, novelty or runtime-advantage claim. Approximate-QFT errors and gate savings are measured explicitly. Gate counts use a common u/CX basis, optimization level 0 and all-to-all connectivity. QFT/ripple comparisons implement the same sum modulo 2ⁿ, with workspace included. Ideal shot sampling reuses the validated final state; it does not demonstrate a hardware execution saving.

Next extensions: larger arithmetic with fault-tolerant rotation synthesis, iterative phase estimation, and noise-aware approximation studies.

References: [Draper, Fourier addition](https://arxiv.org/abs/quant-ph/0008033), [Shor](https://arxiv.org/abs/quant-ph/9508027), [IBM QFT](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.circuit.library.QFTGate), [PennyLane phase convention](https://docs.pennylane.ai/en/stable/code/api/pennylane.PhaseShift.html).
