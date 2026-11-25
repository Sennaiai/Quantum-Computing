"""Reproducible exploratory ADMET Hamiltonian curation.

Retains count-feature MI selection, KSG MI/CMI, positive co-information
screening, and pair/triad Ising construction. KSG's continuous estimator uses
tiny jitter on tied counts; this is an approximation, not an exact discrete
estimator or a significance test. No held-out prediction is claimed.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import numbers
from pathlib import Path
import threading
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem import rdFingerprintGenerator
from sklearn.feature_selection import mutual_info_classif
from npeet_plus import cmi, mi


TDC_ENDPOINTS = {
    "CYP3A4_Substrate_CarbonMangels": "ADME",
    "CYP2D6_Substrate_CarbonMangels": "ADME",
    "CYP2C9_Substrate_CarbonMangels": "ADME",
    "hERG": "Tox", "AMES": "Tox", "BBB_Martins": "ADME",
    "PGP_Broccatelli": "ADME", "Bioavailability_Ma": "ADME",
    "DILI": "Tox", "HIA_Hou": "ADME",
}
FIXTURE_NAME = "Public_SMILES_fixture"
_ESTIMATOR_LOCK = threading.Lock()


@dataclass(frozen=True)
class CurationConfig:
    seed: int = 42
    n_bits: int = 1024
    radius: int = 2
    top_k: int = 100
    n_neighbors: int = 3
    information_base: float = 2.0
    conditional_mi_threshold: float = 0.10
    triad_threshold: float = 0.15
    max_samples: int = 25000
    max_fingerprint_cells: int = 30000000
    max_pairs: int = 5000
    max_candidate_triads: int = 20000
    max_estimator_calls: int = 100000

    def validate(self) -> None:
        for name in ("seed", "n_bits", "radius", "top_k", "n_neighbors",
                     "max_samples", "max_fingerprint_cells", "max_pairs", "max_candidate_triads", "max_estimator_calls"):
            value = getattr(self, name)
            lower = 0 if name in ("seed", "radius") else 1
            if isinstance(value, bool) or not isinstance(value, int) or value < lower:
                raise ValueError(f"{name} must be an integer >= {lower}")
        if self.seed >= 2**32:
            raise ValueError("seed must be < 2**32")
        if self.top_k > self.n_bits:
            raise ValueError("top_k must not exceed n_bits")
        if (isinstance(self.information_base, bool) or not isinstance(self.information_base, numbers.Real)
                or not math.isfinite(self.information_base) or self.information_base <= 1):
            raise ValueError("information_base must be finite and > 1")
        for name in ("conditional_mi_threshold", "triad_threshold"):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, numbers.Real)
                    or not math.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be finite and nonnegative")


def public_smiles_fixture() -> dict[str, pd.DataFrame]:
    """Public structures with invented labels, strictly an execution fixture.

    Repeated observations deliberately exercise tied integer fingerprints.
    Labels are not measurements and cannot support ADMET accuracy claims.
    """
    molecules = [
        ("C", 0), ("CC", 0), ("CCC", 0), ("CCCC", 1),
        ("CCCCC", 0), ("CCCCCC", 0), ("CO", 0), ("CCO", 1),
        ("CCCO", 0), ("CCCCO", 0), ("CN", 0), ("CCN", 0),
        ("c1ccccc1", 0), ("Cc1ccccc1", 1), ("CCc1ccccc1", 1),
        ("Oc1ccccc1", 1), ("Nc1ccccc1", 1), ("Clc1ccccc1", 1),
        ("Brc1ccccc1", 1), ("COc1ccccc1", 1),
        ("CCOc1ccccc1", 0), ("CC(=O)c1ccccc1", 1),
        ("O=C(O)c1ccccc1", 1), ("N#Cc1ccccc1", 1),
    ]
    return {FIXTURE_NAME: pd.DataFrame(molecules * 4, columns=["Drug", "Y"])}


def load_tdc_datasets(names: Sequence[str], cache_path: Path) -> dict[str, pd.DataFrame]:
    """Explicit download/cache mode, never used by the default offline fixture."""
    if isinstance(names, str) or not names or len(set(names)) != len(names):
        raise ValueError("Provide a nonempty sequence of unique TDC dataset names")
    unknown = set(names) - TDC_ENDPOINTS.keys()
    if unknown:
        raise ValueError(f"Unsupported TDC endpoints: {sorted(unknown)}")
    try:
        from tdc.single_pred import ADME, Tox
    except ModuleNotFoundError:
        raise RuntimeError("Optional TDC mode needs a compatible PyTDC installation; use a Drug,Y CSV with the pinned offline environment") from None
    loaders = {"ADME": ADME, "Tox": Tox}
    return {name: loaders[TDC_ENDPOINTS[name]](name=name, path=str(cache_path)).get_data() for name in names}


def smiles_to_morgan_count_fp(smiles: Any, config: CurationConfig) -> tuple[np.ndarray, str] | None:
    """Return hashed integer counts and canonical identity, or reject invalid input."""
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None or molecule.GetNumAtoms() == 0:
        return None
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=config.radius, fpSize=config.n_bits)
    fingerprint = generator.GetCountFingerprint(molecule)
    counts = np.zeros(config.n_bits, dtype=np.int64)
    for index, count in fingerprint.GetNonzeroElements().items():
        counts[index] = count
    return counts, Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def fingerprint_datasets(datasets: Mapping[str, pd.DataFrame], config: CurationConfig) -> dict:
    config.validate()
    if not datasets:
        raise ValueError("At least one dataset is required")
    if any(not isinstance(name, str) or not name for name in datasets):
        raise ValueError("Dataset names must be nonempty strings")
    if any(not isinstance(frame, pd.DataFrame) for frame in datasets.values()):
        raise ValueError("Datasets must contain data frames")
    if sum(len(frame) * config.n_bits for frame in datasets.values()) > config.max_fingerprint_cells:
        raise ValueError("Aggregate fingerprint-cell budget exceeded before fingerprint allocation")
    output = {}
    for name in sorted(datasets):
        frame = datasets[name]
        if not isinstance(frame, pd.DataFrame) or not frame.columns.is_unique or not {"Drug", "Y"} <= set(frame.columns):
            raise ValueError(f"{name}: data frame with unique Drug and Y columns is required")
        if len(frame) > config.max_samples:
            raise ValueError(f"{name}: sample budget exceeded ({len(frame)} > {config.max_samples})")
        raw_labels = frame["Y"].to_numpy()
        if any(not isinstance(value, (numbers.Real, np.bool_)) for value in raw_labels):
            raise ValueError(f"{name}: labels must be finite real binary values 0/1")
        try:
            labels = np.asarray(raw_labels, dtype=float)
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError(f"{name}: labels must be finite binary values 0/1") from exc
        if not np.isfinite(labels).all() or not np.isin(labels, [0, 1]).all():
            raise ValueError(f"{name}: labels must be finite binary values 0/1")
        source_records = [[str(index), type(value).__name__, value if isinstance(value, str) else repr(value), float(labels[position])]
                          for position, (index, value) in enumerate(frame["Drug"].items())]
        source_digest = hashlib.sha256(json.dumps(source_records, separators=(",", ":"), ensure_ascii=True,
                                                  allow_nan=False).encode()).hexdigest()
        rows, y, smiles, canonical, positions, source_indices, rejected = [], [], [], [], [], [], []
        for position, (source_index, value) in enumerate(frame["Drug"].items()):
            converted = smiles_to_morgan_count_fp(value, config)
            if converted is None:
                rejected.append({"source_row_position": position, "source_index": str(source_index),
                                 "reason": "invalid_or_empty_smiles"})
                continue
            counts, identity = converted
            rows.append(counts)
            y.append(int(labels[position]))
            smiles.append(value)
            canonical.append(identity)
            positions.append(position)
            source_indices.append(str(source_index))
        if len(rows) <= config.n_neighbors:
            raise ValueError(f"{name}: need more valid molecules than n_neighbors={config.n_neighbors}")
        if set(y) != {0, 1}:
            raise ValueError(f"{name}: both binary classes are required after SMILES validation")
        identity_labels: dict[str, set[int]] = {}
        for identity, label in zip(canonical, y):
            identity_labels.setdefault(identity, set()).add(label)
        output[name] = {
            "X": np.stack(rows), "y": np.asarray(y, dtype=int), "smiles": smiles,
            "canonical_smiles": canonical, "source_row_positions": positions,
            "source_indices": source_indices, "rejected_rows": rejected,
            "fingerprint_type": "hashed Morgan count fingerprint", "n_bits": config.n_bits,
            "radius": config.radius, "rdkit_version": rdBase.rdkitVersion,
            "duplicate_observations": len(rows) - len(identity_labels),
            "conflicting_label_identities": sum(len(v) > 1 for v in identity_labels.values()),
            "source_table_sha256": source_digest,
            "fingerprint_matrix_sha256": hashlib.sha256(np.stack(rows).astype("<i8").tobytes()).hexdigest(),
        }
    return output


def select_features(fingerprint_data: Mapping[str, dict], config: CurationConfig) -> dict:
    """Fit discrete label MI and min/max on the supplied exploratory table."""
    output = {}
    for name, data in fingerprint_data.items():
        X, y = data["X"], data["y"]
        scores = mutual_info_classif(X, y, discrete_features=True, random_state=config.seed)
        ranking = np.lexsort((np.arange(X.shape[1]), -scores))
        selected = ranking[:config.top_k]
        counts = X[:, selected]
        minima, maxima = counts.min(axis=0), counts.max(axis=0)
        scaled = (counts - minima) / np.where(maxima > minima, maxima - minima, 1)
        output[name] = dict(data, X_selected=counts, X_selected_scaled=scaled,
                            selected_features=selected, mi_scores=scores,
                            selected_mi_scores=scores[selected], mi_units="nats",
                            fit_scope="exploratory_full_table",
                            scaling={"method": "minmax", "feature_min": minima, "feature_max": maxima})
    return output


def _seed(config: CurationConfig, *key: Any) -> int:
    digest = hashlib.sha256(repr((config.seed,) + key).encode()).digest()
    return int.from_bytes(digest[:4], "little")


@contextmanager
def _isolated_estimator_seed(seed: int):
    # npeet_plus consumes NumPy's legacy global RNG for tiny tie-breaking jitter.
    with _ESTIMATOR_LOCK:
        previous = np.random.get_state()
        np.random.seed(seed)
        try:
            yield
        finally:
            np.random.set_state(previous)


def _estimate(config: CurationConfig, key: tuple, x: np.ndarray, y: np.ndarray,
              z: np.ndarray | None = None) -> float:
    if np.unique(x).size < 2 or np.unique(y).size < 2:
        return 0.0
    with _isolated_estimator_seed(_seed(config, *key)):
        value = float(mi(x, y, k=config.n_neighbors, base=config.information_base) if z is None
                      else cmi(x, y, z, k=config.n_neighbors, base=config.information_base))
    if not math.isfinite(value):
        raise ValueError(f"Non-finite KSG estimate for {key}")
    return value


def _pair_budget(name: str, n_features: int, config: CurationConfig) -> None:
    n_pairs = math.comb(n_features, 2)
    if n_pairs > config.max_pairs or 2 * n_pairs > config.max_estimator_calls:
        raise ValueError(f"{name}: pair/estimator budget exceeded before estimation")


def estimate_pairwise(selected_data: Mapping[str, dict], config: CurationConfig) -> dict:
    output = {}
    for name, data in selected_data.items():
        X = data["X_selected_scaled"]
        n_features = X.shape[1]
        _pair_budget(name, n_features, config)
        raw = np.zeros((n_features, n_features))
        for i in range(n_features):
            for j in range(i + 1, n_features):
                raw[i, j] = raw[j, i] = _estimate(config, (name, "mi", i, j), X[:, i:i+1], X[:, j:j+1])
        output[name] = {"interaction_matrix": np.maximum(raw, 0), "mi_matrix": np.maximum(raw, 0),
                        "raw_mi_matrix": raw, "selected_features": data["selected_features"],
                        "information_base": config.information_base}
    return output


def select_conditional_pairs(selected_data: Mapping[str, dict], config: CurationConfig) -> dict:
    output = {}
    for name, data in selected_data.items():
        X, labels = data["X_selected_scaled"], data["y"].reshape(-1, 1)
        n_features = X.shape[1]
        _pair_budget(name, n_features, config)
        matrix, raw, retained = np.zeros((n_features, n_features)), np.zeros((n_features, n_features)), []
        for i in range(n_features):
            for j in range(i + 1, n_features):
                value = _estimate(config, (name, "cmi_label", i, j), X[:, i:i+1], X[:, j:j+1], labels)
                raw[i, j] = raw[j, i] = value
                matrix[i, j] = matrix[j, i] = max(0.0, value)
                if matrix[i, j] > config.conditional_mi_threshold:
                    retained.append({"feature_i": i, "feature_j": j,
                                     "original_feature_i": int(data["selected_features"][i]),
                                     "original_feature_j": int(data["selected_features"][j]),
                                     "conditional_mi": float(matrix[i, j])})
        retained.sort(key=lambda pair: (-pair["conditional_mi"], pair["feature_i"], pair["feature_j"]))
        active = sorted({p[k] for p in retained for k in ("feature_i", "feature_j")})
        output[name] = dict(data, retained_pairs=retained, num_retained_pairs=len(retained),
                            active_feature_positions=active, num_active_features=len(active),
                            active_original_features=[int(data["selected_features"][i]) for i in active],
                            conditional_mi_threshold=config.conditional_mi_threshold,
                            conditional_mi_matrix=matrix, raw_conditional_mi_matrix=raw,
                            information_base=config.information_base)
    return output


def discover_triads(conditional_data: Mapping[str, dict], pairwise_data: Mapping[str, dict],
                    config: CurationConfig) -> dict:
    output = {}
    for name, data in conditional_data.items():
        X, retained_pairs = data["X_selected_scaled"], data["retained_pairs"]
        candidates = set()
        for pair in retained_pairs:
            for k in data["active_feature_positions"]:
                triad = tuple(sorted((pair["feature_i"], pair["feature_j"], k)))
                if len(set(triad)) == 3:
                    candidates.add(triad)
                    if len(candidates) > config.max_candidate_triads:
                        raise ValueError(f"{name}: candidate-triad budget exceeded before estimation")
        estimator_calls = 2 * math.comb(X.shape[1], 2) + 3 * len(candidates)
        if estimator_calls > config.max_estimator_calls:
            raise ValueError(f"{name}: estimator-call budget exceeded before triad estimation")
        retained, diagnostics = [], []
        for i, j, k in sorted(candidates):
            orientations = []
            constant = any(np.unique(X[:, p]).size < 2 for p in (i, j, k))
            for a, b, given in ((i, j, k), (i, k, j), (j, k, i)):
                # Raw MI, including negative estimates, is required for signed co-information.
                value = 0.0 if constant else pairwise_data[name]["raw_mi_matrix"][a, b] - _estimate(
                    config, (name, "cmi_feature", a, b, given), X[:, a:a+1], X[:, b:b+1], X[:, given:given+1])
                orientations.append(float(value))
            score = float(np.mean(orientations))
            record = {"feature_i": i, "feature_j": j, "feature_k": k,
                      "original_feature_i": int(data["selected_features"][i]),
                      "original_feature_j": int(data["selected_features"][j]),
                      "original_feature_k": int(data["selected_features"][k]),
                      "triadic_interaction": score,
                      "orientation_ij_given_k": orientations[0],
                      "orientation_ik_given_j": orientations[1],
                      "orientation_jk_given_i": orientations[2]}
            diagnostics.append(record)
            if score > config.triad_threshold:
                retained.append(record)
        retained.sort(key=lambda t: (-t["triadic_interaction"], t["feature_i"], t["feature_j"], t["feature_k"]))
        output[name] = dict(data, candidate_triads=sorted(candidates), num_candidate_triads=len(candidates),
                            retained_triads=retained, num_retained_triads=len(retained),
                            triad_diagnostics=diagnostics, triad_threshold=config.triad_threshold,
                            estimator_calls_upper_bound=estimator_calls)
    return output


def assemble_hamiltonians(conditional_data: Mapping[str, dict], triad_data: Mapping[str, dict]) -> dict:
    hypergraphs, global_h, local_h, problem_h = {}, {}, {}, {}
    for name, pairs in conditional_data.items():
        triads = triad_data[name]["retained_triads"]
        active = sorted({p[k] for p in pairs["retained_pairs"] for k in ("feature_i", "feature_j")} |
                        {t[k] for t in triads for k in ("feature_i", "feature_j", "feature_k")})
        compact = {position: qubit for qubit, position in enumerate(active)}
        original = {compact[position]: int(pairs["selected_features"][position]) for position in active}
        pair_terms = [{"qubit_i": compact[p["feature_i"]], "qubit_j": compact[p["feature_j"]],
                       "weight": p["conditional_mi"]} for p in pairs["retained_pairs"]]
        triad_terms = [{"qubit_i": compact[t["feature_i"]], "qubit_j": compact[t["feature_j"]],
                        "qubit_k": compact[t["feature_k"]], "weight": t["triadic_interaction"]} for t in triads]
        mapping = {"active_features": active, "compact_index_map": compact,
                   "inverse_index_map": {q: p for p, q in compact.items()}, "original_feature_map": original}
        hypergraphs[name] = dict(mapping, pair_terms=pair_terms, triad_terms=triad_terms)
        terms = [dict(term_type="pair", locality=2, pauli="ZZ", qubits=(p["qubit_i"], p["qubit_j"]),
                      coefficient=float(p["weight"])) for p in pair_terms]
        terms += [dict(term_type="triad", locality=3, pauli="ZZZ",
                       qubits=(t["qubit_i"], t["qubit_j"], t["qubit_k"]), coefficient=float(t["weight"]))
                  for t in triad_terms]
        local = [dict(term_type="local", locality=1, pauli="Z", qubits=(q,), parameter_index=q)
                 for q in range(len(active))]
        global_h[name] = dict(mapping, num_qubits=len(active), terms=terms)
        local_h[name] = dict(mapping, num_qubits=len(active), num_parameters=len(active), terms=local,
                            parameter_vector={"name": "theta", "dimension": len(active), "values": None})
        problem_h[name] = dict(mapping, num_qubits=len(active), num_parameters=len(active),
                              interaction_terms=terms, local_terms=local, terms=terms+local,
                              parameter_vector=local_h[name]["parameter_vector"])
    return {"interaction_hypergraph": hypergraphs, "global_hamiltonian": global_h,
            "parameterized_hamiltonian": local_h, "problem_hamiltonian": problem_h}


def curate_datasets(datasets: Mapping[str, pd.DataFrame], config: CurationConfig) -> dict[str, Any]:
    fingerprints = fingerprint_datasets(datasets, config)
    selected = select_features(fingerprints, config)
    pairwise = estimate_pairwise(selected, config)
    conditional = select_conditional_pairs(selected, config)
    triads = discover_triads(conditional, pairwise, config)
    return {"schema_version": 2, "datasets": dict(datasets), "config": asdict(config),
            "fit_scope": "exploratory_full_table", "fingerprint_data": fingerprints,
            "selected_feature_data": selected, "pairwise_mi_data": pairwise,
            "conditional_pair_data": conditional, "triadic_interaction_data": triads,
            **assemble_hamiltonians(conditional, triads)}


def run_complete_curation(notebook_path: Path | None = None, *, mode: str = "fixture",
                          config: CurationConfig | None = None, datasets: Mapping[str, pd.DataFrame] | None = None,
                          dataset_names: Sequence[str] = ("BBB_Martins",),
                          cache_path: Path = Path("artifacts/tdc")) -> dict[str, Any]:
    """Run every curation stage; the public offline fixture is the default.

    ``notebook_path`` is retained for compatibility, but no notebook is executed.
    ``mode='tdc'`` explicitly permits dataset I/O. Local tables require
    ``mode='provided'``. No mode fits or assesses a predictive model.
    """
    if mode not in {"fixture", "tdc", "provided"}:
        raise ValueError("mode must be fixture, provided, or tdc")
    if datasets is not None and mode != "provided":
        raise ValueError("Injected datasets require mode='provided'")
    settings = config or (CurationConfig(top_k=4) if mode == "fixture" else CurationConfig())
    settings.validate()
    if mode == "fixture":
        tables = public_smiles_fixture()
    elif mode == "tdc":
        tables = load_tdc_datasets(dataset_names, cache_path)
    elif datasets is not None:
        tables = datasets
    else:
        raise ValueError("mode='provided' requires datasets")
    result = curate_datasets(tables, settings)
    result["mode"] = mode
    result["fixture_labels_are_synthetic"] = mode == "fixture"
    return result


def molecule_problem_input(curation: dict[str, Any], dataset: str, sample_index: int):
    """Bind molecular count fields to compact qubits without refitting statistics."""
    if dataset not in curation["selected_feature_data"]:
        raise ValueError(f"Unknown curated dataset {dataset!r}")
    selected = curation["selected_feature_data"][dataset]
    X = np.asarray(selected["X_selected_scaled"], dtype=float)
    if isinstance(sample_index, bool) or not isinstance(sample_index, (int, np.integer)) or not 0 <= sample_index < len(X):
        raise ValueError("sample_index must be a nonnegative in-range integer")
    hypergraph = curation["interaction_hypergraph"][dataset]
    active = list(hypergraph["active_features"])
    if not active:
        raise ValueError("No retained interaction features: no qubits to bind; inspect screening diagnostics")
    if len(set(active)) != len(active) or any(not 0 <= p < X.shape[1] for p in active):
        raise ValueError("Invalid active feature mapping")
    expected = {q: int(selected["selected_features"][p]) for q, p in enumerate(active)}
    if hypergraph["original_feature_map"] != expected:
        raise ValueError("Original fingerprint mapping disagrees with compact qubit mapping")
    feature_vector = X[sample_index, active].tolist()
    if not np.isfinite(feature_vector).all() or any(not 0 <= x <= 1 for x in feature_vector):
        raise ValueError("Molecular feature vector must be finite and scaled to [0, 1]")
    terms, seen = [], set()
    for term in curation["global_hamiltonian"][dataset]["terms"]:
        qubits, locality = tuple(term["qubits"]), term["locality"]
        if (locality not in (2, 3) or len(qubits) != locality or len(set(qubits)) != locality
                or term["pauli"] != "Z" * locality or any(not 0 <= q < len(active) for q in qubits)):
            raise ValueError("Invalid curated interaction support")
        word = " ".join(f"Z{q}" for q in sorted(qubits))
        coefficient = float(term["coefficient"])
        if not math.isfinite(coefficient) or coefficient <= 0 or word in seen:
            raise ValueError("Curated interactions must have unique supports and finite positive weights")
        seen.add(word)
        terms.append({"coefficient": coefficient, "pauli_string": word, "locality": locality})
    adapter = {"hamiltonian_A_interaction_ising": {
        dataset: {"selected_features": list(expected.values()), "pauli_terms": terms}}}
    provenance = {
        "dataset": dataset, "sample_index": int(sample_index), "samples": len(X),
        "source_row_position": selected["source_row_positions"][sample_index],
        "source_index": selected["source_indices"][sample_index],
        "selected_features": X.shape[1], "active_features": len(active),
        "original_fingerprint_indices": list(expected.values()),
        "retained_pairs": len(curation["conditional_pair_data"][dataset]["retained_pairs"]),
        "retained_triads": len(curation["triadic_interaction_data"][dataset]["retained_triads"]),
        "interaction_terms": len(terms), "fit_scope": curation["fit_scope"],
        "source_table_sha256": selected["source_table_sha256"],
        "fingerprint_matrix_sha256": selected["fingerprint_matrix_sha256"],
        "scaling": {"method": "minmax", "fit_scope": curation["fit_scope"],
                    "active_feature_min": selected["scaling"]["feature_min"][active].tolist(),
                    "active_feature_max": selected["scaling"]["feature_max"][active].tolist()},
        "rejected_rows": len(selected["rejected_rows"]),
        "duplicate_observations": selected["duplicate_observations"],
        "conflicting_label_identities": selected["conflicting_label_identities"],
    }
    return adapter, feature_vector, provenance
