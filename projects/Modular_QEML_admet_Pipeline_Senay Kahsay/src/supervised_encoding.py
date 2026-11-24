"""Train-only, portable molecular Hamiltonian family and label-free transforms.

All top-k training-selected count features remain local fields. Screened pairs
and triads contribute H_A, possibly empty. Inference never fits statistics or
reads labels. This supervised family is separate from the legacy exploratory
active-interaction-only benchmark, which remains unchanged.
"""
from __future__ import annotations

from dataclasses import asdict
from itertools import combinations
import math
import numbers

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem import rdFingerprintGenerator

from admet_curation import CurationConfig, run_complete_curation
from data_quality import (IDENTITY_POLICY, SCAFFOLD_POLICY, _validate_cleaned,
                          molecule_identity, sha256_json)

ENCODING_KIND = "training_selected_all_local_fields_v1"


def _schema_digest(schema):
    return sha256_json({key: value for key, value in schema.items() if key not in {"schema_id", "encoding_fit_id"}})


def fit_hamiltonian_schema(train_table, endpoint, config=None):
    """Fit MI, scaling and KSG interaction screening exclusively on training IDs."""
    _validate_cleaned(train_table, endpoint)
    config = CurationConfig(top_k=4) if config is None else config
    if not isinstance(config, CurationConfig):
        raise ValueError("config must be a CurationConfig")
    config.validate()
    # Order must not depend on input row ordering: KSG jitter is assigned to
    # this stable identity order, preserving reproducibility of the fit.
    train = train_table.sort_values("molecule_id").reset_index(drop=True)
    if set(train["Y"]) != {0, 1}:
        raise ValueError("Training requires both binary classes")
    frame = pd.DataFrame({"Drug": train["Drug"].tolist(), "Y": train["Y"].astype(int).tolist()},
                         index=train["molecule_id"].tolist())
    curation = run_complete_curation(mode="provided", datasets={endpoint: frame}, config=config)
    selected = curation["selected_feature_data"][endpoint]
    features = [int(x) for x in selected["selected_features"]]
    pairs = curation["conditional_pair_data"][endpoint]["retained_pairs"]
    triads = curation["triadic_interaction_data"][endpoint]["retained_triads"]
    terms = []
    pair_supports, triad_supports = [], []
    for term in pairs:
        support = sorted([int(term["feature_i"]), int(term["feature_j"])])
        pair_supports.append(support)
        terms.append({"coefficient": float(term["conditional_mi"]),
                      "pauli_string": " ".join(f"Z{i}" for i in support), "locality": 2})
    for term in triads:
        support = sorted([int(term["feature_i"]), int(term["feature_j"]), int(term["feature_k"])])
        triad_supports.append(support)
        terms.append({"coefficient": float(term["triadic_interaction"]),
                      "pauli_string": " ".join(f"Z{i}" for i in support), "locality": 3})
    pair_supports = sorted(pair_supports);triad_supports = sorted(triad_supports)
    readout_pairs = {tuple(pair) for pair in pair_supports}
    for triple in triad_supports:
        readout_pairs.update(combinations(triple, 2))
    schema = {"schema_version": 1, "encoding_kind": ENCODING_KIND, "endpoint": endpoint,
        "fit_scope": "training_only", "identity_policy": IDENTITY_POLICY,
        "scaffold_policy": SCAFFOLD_POLICY, "config": asdict(config),
        "morgan": {"n_bits": config.n_bits, "radius": config.radius, "include_chirality": False,
                   "count_simulation": False, "use_bond_types": True,
                   "rdkit_version": rdBase.rdkitVersion,
                   "interpretation": "hashed Morgan counts; chirality-blind original baseline, not unique substructures"},
        "train_molecule_ids": train["molecule_id"].tolist(),
        "train_molecule_ids_sha256": sha256_json(train["molecule_id"].tolist()),
        "training_class_counts": {str(y): int((train["Y"] == y).sum()) for y in (0, 1)},
        "training_table_sha256": sha256_json([[row.molecule_id, row.Drug, int(row.Y)] for row in train.itertuples()]),
        "training_labels_sha256": sha256_json([[row.molecule_id, int(row.Y)] for row in train.itertuples()]),
        "selected_original_features": features, "active_original_features": features,
        "wire_to_fingerprint_index": {str(i): feature for i, feature in enumerate(features)},
        "feature_label_mi_nats": [float(x) for x in selected["selected_mi_scores"]],
        "scaling": {"feature_min": selected["scaling"]["feature_min"].astype(float).tolist(),
                    "feature_max": selected["scaling"]["feature_max"].astype(float).tolist(),
                    "clip": True, "constant_policy": "zero; flag any count differing from the training constant"},
        "adapter": {"hamiltonian_A_interaction_ising": {endpoint: {
                    "selected_features": features, "pauli_terms": sorted(terms, key=lambda item: item["pauli_string"])}}},
        "hamiltonian_pair_supports": pair_supports, "hamiltonian_triad_supports": triad_supports,
        "readout_supports": {"singles": [[i] for i in range(len(features))],
                             "pairs": [list(pair) for pair in sorted(readout_pairs)], "triads": triad_supports}}
    schema["schema_id"] = schema["encoding_fit_id"] = _schema_digest(schema)
    validate_frozen_schema(schema)
    return schema


def validate_frozen_schema(schema):
    keys = {"schema_version", "encoding_kind", "endpoint", "fit_scope", "identity_policy", "scaffold_policy",
            "config", "morgan", "train_molecule_ids", "train_molecule_ids_sha256", "training_class_counts",
            "training_table_sha256", "training_labels_sha256", "selected_original_features", "active_original_features",
            "wire_to_fingerprint_index", "feature_label_mi_nats", "scaling", "adapter",
            "hamiltonian_pair_supports", "hamiltonian_triad_supports", "readout_supports", "schema_id", "encoding_fit_id"}
    if (not isinstance(schema, dict) or set(schema) != keys or type(schema["schema_version"]) is not int
            or schema["schema_version"] != 1 or schema["encoding_kind"] != ENCODING_KIND):
        raise ValueError("Unsupported frozen encoding schema")
    if schema["schema_id"] != schema["encoding_fit_id"] or schema["schema_id"] != _schema_digest(schema):
        raise ValueError("Frozen encoding checksum mismatch")
    if schema["fit_scope"] != "training_only" or schema["identity_policy"] != IDENTITY_POLICY or schema["scaffold_policy"] != SCAFFOLD_POLICY:
        raise ValueError("Encoding requires the declared training-only identity/scaffold policy")
    config = CurationConfig(**schema["config"]);config.validate()
    features = schema["selected_original_features"]
    if (not isinstance(features, list) or len(features) != config.top_k or len(set(features)) != len(features)
            or any(type(i) is not int or not 0 <= i < config.n_bits for i in features)
            or schema["active_original_features"] != features
            or schema["wire_to_fingerprint_index"] != {str(i): bit for i, bit in enumerate(features)}):
        raise ValueError("Selected/active fingerprint mapping is inconsistent")
    morgan = schema["morgan"]
    if (morgan.get("n_bits") != config.n_bits or morgan.get("radius") != config.radius
            or morgan.get("include_chirality") is not False or morgan.get("count_simulation") is not False
            or morgan.get("use_bond_types") is not True or morgan.get("rdkit_version") != rdBase.rdkitVersion):
        raise ValueError("Frozen Morgan configuration/version differs from the supported generator")
    train_ids = schema["train_molecule_ids"]
    from data_quality import _SHA
    if (not isinstance(train_ids, list) or len(train_ids) <= config.n_neighbors or train_ids != sorted(set(train_ids))
            or any(not isinstance(identity, str) or _SHA.fullmatch(identity) is None for identity in train_ids)
            or schema["train_molecule_ids_sha256"] != sha256_json(train_ids)):
        raise ValueError("Training identity membership is inconsistent")
    if (set(schema["training_class_counts"]) != {"0", "1"}
            or any(type(value) is not int or value < 1 for value in schema["training_class_counts"].values())
            or sum(schema["training_class_counts"].values()) != len(train_ids)):
        raise ValueError("Training class counts are inconsistent")
    for name in ("training_table_sha256", "training_labels_sha256"):
        if not isinstance(schema[name], str) or _SHA.fullmatch(schema[name]) is None:
            raise ValueError("Training-table/label checksum is missing")
    minimum = np.asarray(schema["scaling"].get("feature_min"), dtype=float)
    maximum = np.asarray(schema["scaling"].get("feature_max"), dtype=float)
    if (minimum.shape != (len(features),) or maximum.shape != minimum.shape or not np.isfinite(minimum).all()
            or not np.isfinite(maximum).all() or np.any(minimum < 0) or np.any(maximum < minimum)
            or schema["scaling"].get("clip") is not True):
        raise ValueError("Frozen training scales are invalid")
    if len(schema["feature_label_mi_nats"]) != len(features) or any(not isinstance(x, numbers.Real) or not math.isfinite(x) or x < 0 for x in schema["feature_label_mi_nats"]):
        raise ValueError("Feature-selection diagnostics are invalid")
    endpoint = schema["endpoint"]
    if not isinstance(endpoint, str) or not endpoint.strip():
        raise ValueError("Endpoint is required")
    try:
        family = schema["adapter"]["hamiltonian_A_interaction_ising"][endpoint]
    except (KeyError, TypeError) as exc:
        raise ValueError("Missing frozen interaction family") from exc
    if family.get("selected_features") != features:
        raise ValueError("Interaction family mapping differs from selected features")
    actual = {2: [], 3: []}
    seen = set()
    for term in family.get("pauli_terms", []):
        tokens = term["pauli_string"].split()
        if any(not token.startswith("Z") or not token[1:].isdigit() for token in tokens):
            raise ValueError("Frozen interactions must be Z pairs/triads")
        support = [int(token[1:]) for token in tokens]
        value = term["coefficient"]
        if (len(support) not in (2, 3) or support != sorted(set(support)) or max(support) >= len(features)
                or tuple(support) in seen or isinstance(value, bool) or not isinstance(value, numbers.Real)
                or not math.isfinite(value) or value <= 0 or term.get("locality") != len(support)):
            raise ValueError("Invalid frozen interaction support/coefficient")
        actual[len(support)].append(support);seen.add(tuple(support))
    pairs, triads = sorted(actual[2]), sorted(actual[3])
    if schema["hamiltonian_pair_supports"] != pairs or schema["hamiltonian_triad_supports"] != triads:
        raise ValueError("Interaction support metadata differs from H_A")
    closure = {tuple(pair) for pair in pairs}
    for triple in triads:
        closure.update(combinations(triple, 2))
    expected_readout = {"singles": [[i] for i in range(len(features))],
                        "pairs": [list(pair) for pair in sorted(closure)], "triads": triads}
    if schema["readout_supports"] != expected_readout:
        raise ValueError("Readout supports do not close the retained triad moments")
    return True


def transform_molecules(schema, table):
    """Transform valid molecules with a frozen train-only fit; Y is never read.

    Concordant identity duplicates are combined by identity with source maps.
    Distinct molecules sharing a feature vector remain separate records.
    """
    validate_frozen_schema(schema)
    config = CurationConfig(**schema["config"])
    if (not isinstance(table, pd.DataFrame) or not table.columns.is_unique or "Drug" not in table
            or not 1 <= len(table) <= config.max_samples):
        raise ValueError("A bounded molecule table with a Drug column is required")
    if len(table)*config.n_bits > config.max_fingerprint_cells:
        raise ValueError("Fingerprint allocation budget exceeded")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=config.radius, fpSize=config.n_bits,
        includeChirality=False, countSimulation=False, useBondTypes=True)
    minimum = np.asarray(schema["scaling"]["feature_min"])
    maximum = np.asarray(schema["scaling"]["feature_max"])
    ranges = maximum-minimum
    features = schema["selected_original_features"]
    records = {}
    for position in range(len(table)):
        # Column access is explicit: no iteration over or conversion of Y.
        drug = table["Drug"].iloc[position]
        canonical, molecule_id, scaffold_id = molecule_identity(drug)
        if "molecule_id" in table and table["molecule_id"].iloc[position] != molecule_id:
            raise ValueError("Supplied molecule_id differs from its canonical structure")
        if "endpoint" in table and table["endpoint"].iloc[position] != schema["endpoint"]:
            raise ValueError("Molecule endpoint differs from frozen encoding")
        if "scaffold_id" in table and table["scaffold_id"].iloc[position] != scaffold_id:
            raise ValueError("Supplied scaffold_id differs from its molecule")
        def metadata(name, fallback):
            value = table[name].iloc[position] if name in table else fallback
            if not isinstance(value, (list, tuple)):
                raise ValueError(f"{name} must be a source-map sequence")
            return list(value)
        positions = metadata("source_row_positions", [position])
        source_indices = metadata("source_indices", [str(table.index[position])])
        record_ids = metadata("source_record_ids", [str(table["Drug_ID"].iloc[position]) if "Drug_ID" in table else str(table.index[position])])
        if molecule_id in records:
            for key, values in (("source_row_positions", positions), ("source_indices", source_indices), ("source_record_ids", record_ids)):
                records[molecule_id][key].extend(values)
            continue
        mol = Chem.MolFromSmiles(canonical)
        fp = generator.GetCountFingerprint(mol)
        counts = np.zeros(config.n_bits, dtype=np.int64)
        for index, count in fp.GetNonzeroElements().items():
            counts[index] = count
        selected = counts[features].astype(float)
        scaled = np.divide(selected-minimum, ranges, out=np.zeros_like(selected), where=ranges > 0)
        below = np.flatnonzero(selected < minimum).tolist()
        above = np.flatnonzero(selected > maximum).tolist()
        constant = np.flatnonzero(ranges == 0).tolist()
        outside_constant = np.flatnonzero((ranges == 0) & (selected != minimum)).tolist()
        records[molecule_id] = {"molecule_id": molecule_id, "endpoint": schema["endpoint"],
            "encoding_fit_id": schema["encoding_fit_id"], "schema_id": schema["schema_id"],
            "scaffold_id": scaffold_id, "theta": np.clip(scaled, 0, 1).tolist(),
            "full_counts": counts.tolist(), "selected_counts": selected.astype(int).tolist(),
            "source_row_positions": positions, "source_indices": source_indices, "source_record_ids": record_ids,
            "clipping": {"below_training_range_wires": below, "above_training_range_wires": above,
                         "constant_training_wires": constant, "outside_constant_training_wires": outside_constant,
                         "any_out_of_domain": bool(below or above)}}
    ordered = [records[identity] for identity in sorted(records)]
    return {"schema_id": schema["schema_id"], "encoding_fit_id": schema["encoding_fit_id"],
            "endpoint": schema["endpoint"], "records": ordered,
            "diagnostics": {"input_rows": len(table), "unique_molecules": len(ordered),
                            "identity_duplicates_combined": len(table)-len(ordered),
                            "molecules_outside_training_range": sum(r["clipping"]["any_out_of_domain"] for r in ordered),
                            "fit_scope": "training_only; no inference refitting or label access"}}
