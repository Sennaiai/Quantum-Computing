"""Execute the canonical ADMET curation notebook stages without reimplementation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


# Exact code cells that form Stages 0--6C of the cleaned submission notebook.
CURATION_CELL_INDICES = (3, 5, 7, 9, 11, 15, 16, 18, 21, 22, 23)


def run_complete_curation(notebook_path: Path) -> dict[str, Any]:
    """Run the complete notebook-authored statistical curation stages in order."""
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    namespace: dict[str, Any] = {
        "__name__": "admet_complete_curation_runtime",
        # Notebook display calls are presentation-only.
        "display": lambda value: value,
    }
    for index in CURATION_CELL_INDICES:
        cell = notebook["cells"][index]
        if cell.get("cell_type") != "code":
            raise ValueError(f"Expected code cell {index} in {notebook_path}")
        source = "".join(cell.get("source", []))
        exec(compile(source, f"{notebook_path.name}:cell-{index}", "exec"), namespace)

    required = (
        "datasets", "fingerprint_data", "selected_feature_data",
        "pairwise_mi_data", "conditional_pair_data", "triadic_interaction_data",
        "interaction_hypergraph", "global_hamiltonian",
        "parameterized_hamiltonian", "problem_hamiltonian",
    )
    missing = [name for name in required if name not in namespace]
    if missing:
        raise RuntimeError(f"Canonical curation did not produce: {missing}")
    return {name: namespace[name] for name in required}


def molecule_problem_input(curation: dict[str, Any], dataset: str, sample_index: int):
    """Adapt Stage-6 output to the canonical counterdiabatic runtime interface."""
    selected = curation["selected_feature_data"][dataset]
    hypergraph = curation["interaction_hypergraph"][dataset]
    global_h = curation["global_hamiltonian"][dataset]
    active_positions = list(hypergraph["active_features"])
    feature_vector = [
        float(selected["X_selected_scaled"][sample_index, position])
        for position in active_positions
    ]
    selected_features = [int(hypergraph["original_feature_map"][i]) for i in range(len(active_positions))]
    pauli_terms = []
    for term in global_h["terms"]:
        pauli_string = " ".join(
            f"{letter}{qubit}" for letter, qubit in zip(term["pauli"], term["qubits"])
        )
        pauli_terms.append({
            "coefficient": float(term["coefficient"]),
            "pauli_string": pauli_string,
            "locality": int(term["locality"]),
        })
    curated_adapter = {
        "hamiltonian_A_interaction_ising": {
            dataset: {"selected_features": selected_features, "pauli_terms": pauli_terms}
        }
    }
    provenance = {
        "samples": int(selected["X_selected_scaled"].shape[0]),
        "selected_features": int(selected["X_selected_scaled"].shape[1]),
        "active_features": len(active_positions),
        "retained_pairs": len(curation["conditional_pair_data"][dataset]["retained_pairs"]),
        "retained_triads": len(curation["triadic_interaction_data"][dataset]["retained_triads"]),
        "interaction_terms": len(pauli_terms),
    }
    return curated_adapter, feature_vector, provenance
