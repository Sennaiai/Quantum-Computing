# Amplitude Amplification for 5-Queens

A reproducible quantum-algorithm demonstration that returns valid 5-Queens boards using a reversible constraint oracle. The oracle evaluates column ranges, shared columns and diagonal conflicts; it does not load a solution list. Classical exhaustive search and independent backtracking supply validation references and the known solution count used to choose the iteration budget.

## Execution evidence

| Configuration | Result |
| --- | --- |
| Search register | 15 board qubits; 32,768 binary assignments, including invalid column codes |
| Reversible predicate | Four counter qubits; count violations, mark zero, uncompute |
| Amplification | 44 phase-oracle calls; exact success ≈0.999746 |
| Returned solutions | All ten valid boards recovered in the seeded 4,096-shot run and independently verified |
| Validation | Complete complex-state reference, all 32,768 oracle inputs, counter arithmetic, clean workspace, overshooting and boundary cases |

For `A|0⟩=|ψ⟩`, the implementation uses `Q=(2|ψ⟩⟨ψ|−I)(I−2Πgood)` and checks `P(k)=sin²((2k+1)asin√p)` with `p=10/32768`. The counter cannot wrap to zero: at most 15 predicates are counted in four bits. Bitstrings show row 0 first, with Qiskit qubit 0 least significant.

[The notebook](Grovers_Algorithm_for_Quantum_Search_and_Acceleration.ipynb) presents execution and results. [Measured JSON](results/summary.json) and its checksum manifest bind the implementation, environment and outcomes. Supporting cases cover entangled nonuniform preparation, 4-Queens, and the original solution-lookup oracle as a cost comparison.

![Measured amplification and oracle cost](results/amplification.svg)

## Reproduce

Python 3.11, from this folder:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_checks.py
```

Acceptance executes tests, five simulator cases and a fresh-kernel notebook. Outputs stay in ignored `artifacts/validation/`. Dedicated CI repeats execution; notebook structure checking alone is insufficient.

## Scope

Ideal Aer simulation; no hardware, novelty or end-to-end speedup claim. Known `M=10` sets the iteration count; unknown-success search is not implemented. Circuit synthesis reports logical `u`/CX costs without device routing. The cheap lookup comparison already knows the answers; structured classical backtracking is a separate baseline. Ideal readout samples the validated final state without repeating coherent evolution. Shot intervals are per-run exact binomial 95% intervals; conservative Hoeffding bounds provide the acceptance check.

Next extensions: unknown-success schedules, fixed-point amplification, and arithmetic constraint comparators with explicit workspace and gate-cost ablations.

Primary sources: [Brassard et al.](https://arxiv.org/abs/quant-ph/0005055), [IBM Grover operator](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.circuit.library.grover_operator).
