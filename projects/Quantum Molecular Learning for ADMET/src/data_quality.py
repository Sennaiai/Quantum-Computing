"""Endpoint-specific identity cleaning and leakage-resistant scaffold partitions.

No labels are invented: conflicting identities are quarantined, concordant
duplicates collapse, and every source row remains auditable. Stereochemistry,
isotopes, formal charges and disconnected fragments are retained. Atom-map
annotations are removed because they are provenance tags, not chemical identity.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
import numbers
import re

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem import rdMolHash

MAX_ROWS = 25000
SPLITS = ("train", "validation", "test")
IDENTITY_POLICY = "canonical_isomeric_smiles; atom_maps_removed; salts_charges_isotopes_stereo_retained"
SCAFFOLD_POLICY = "rdkit.rdMolHash.HashFunction.MurckoScaffold; stereo_removed_for_scaffold_only; shared_empty_acyclic_group"
_SHA = re.compile(r"[0-9a-f]{64}\Z")


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_json(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def molecule_identity(smiles):
    """Return canonical string, stable identity and strict Murcko group IDs."""
    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError("SMILES must be a nonempty string")
    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError("SMILES is invalid or has no atoms")
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    # Direct documented RDKit scaffold hashing avoids the MurckoDecompose
    # Python wrapper's bad-any-cast failure in the pinned RDKit build.
    # Hash a canonical reparse: stereochemical ring tags emitted by the pinned
    # MolHash build can otherwise depend on the original atom ordering.
    canonical_mol = Chem.MolFromSmiles(canonical)
    # A scaffold split groups a constitutional framework irrespective of
    # stereochemical annotations. Fused-ring stereo can survive MolHash's
    # side-chain removal, so remove it explicitly on a separate molecule.
    # Canonical molecular identity above remains fully stereochemical.
    scaffold_mol = Chem.Mol(canonical_mol)
    Chem.RemoveStereochemistry(scaffold_mol)
    scaffold = rdMolHash.MolHash(scaffold_mol, rdMolHash.HashFunction.MurckoScaffold)
    return canonical, hashlib.sha256(canonical.encode()).hexdigest(), hashlib.sha256(scaffold.encode()).hexdigest()


def _endpoint(endpoint):
    if not isinstance(endpoint, str) or not endpoint.strip() or endpoint != endpoint.strip():
        raise ValueError("endpoint must be a nonempty canonical string")
    return endpoint


def _binary_label(value):
    if not isinstance(value, (numbers.Real, np.bool_)) or not math.isfinite(value) or value not in (0, 1):
        return None
    return int(value)


def clean_endpoint_table(table, endpoint):
    """Return cleaned labelled identities, all valid inference identities, audit.

    The inference table deliberately has no Y column. Conflicting or missing
    labels do not prevent otherwise valid molecules from being sent for future
    measurements, but cannot be used to supervise a model.
    """
    endpoint = _endpoint(endpoint)
    if (not isinstance(table, pd.DataFrame) or not table.columns.is_unique
            or not {"Drug", "Y"} <= set(table.columns)):
        raise ValueError("A table with unique Drug and Y columns is required")
    if not 1 <= len(table) <= MAX_ROWS:
        raise ValueError(f"Table must contain 1..{MAX_ROWS} rows")
    groups, excluded = {}, []
    invalid_smiles, invalid_labels = 0, 0
    source_records = []
    for position, (source_index, row) in enumerate(table.iterrows()):
        drug, raw_label = row["Drug"], row["Y"]
        record_id = str(row["Drug_ID"]) if "Drug_ID" in table.columns else str(source_index)
        source_records.append([position, str(source_index), record_id,
                               drug if isinstance(drug, str) else repr(drug), repr(raw_label)])
        source = {"source_row_position": position, "source_index": str(source_index),
                  "source_record_id": record_id}
        try:
            canonical, molecule_id, scaffold_id = molecule_identity(drug)
        except (ValueError, RuntimeError):
            invalid_smiles += 1
            excluded.append(dict(source, reason="invalid_or_empty_smiles"))
            continue
        label = _binary_label(raw_label)
        if label is None:
            invalid_labels += 1
            excluded.append(dict(source, molecule_id=molecule_id, reason="missing_or_invalid_binary_label"))
        group = groups.setdefault(molecule_id, {"Drug": canonical, "molecule_id": molecule_id,
            "endpoint": endpoint, "scaffold_id": scaffold_id, "sources": [], "labels": []})
        group["sources"].append(source)
        group["labels"].append(label)
    labelled, inference, conflicts = [], [], []
    conflicting_rows, duplicates_collapsed = 0, 0
    def row_metadata(group, sources):
        return {key: group[key] for key in ("Drug", "molecule_id", "endpoint", "scaffold_id")} | {
            "source_row_positions": [s["source_row_position"] for s in sources],
            "source_indices": [s["source_index"] for s in sources],
            "source_record_ids": [s["source_record_id"] for s in sources]}
    for molecule_id in sorted(groups):
        group = groups[molecule_id]
        valid = [(s, y) for s, y in zip(group["sources"], group["labels"]) if y is not None]
        labels = {y for _, y in valid}
        status = "conflicting_labels" if len(labels) > 1 else "unlabelled" if not labels else "concordant_label"
        inference.append(dict(row_metadata(group, group["sources"]), label_status=status))
        if len(labels) > 1:
            conflicting_rows += len(valid)
            conflicts.append({"molecule_id": molecule_id, "label_values": sorted(labels),
                "source_row_positions": [s["source_row_position"] for s, _ in valid],
                "source_record_ids": [s["source_record_id"] for s, _ in valid]})
        elif labels:
            duplicates_collapsed += len(valid)-1
            labelled.append(dict(row_metadata(group, [s for s, _ in valid]), Y=next(iter(labels))))
    columns = ["Drug", "molecule_id", "endpoint", "scaffold_id", "source_row_positions", "source_indices", "source_record_ids"]
    cleaned = pd.DataFrame(labelled, columns=columns+["Y"])
    candidates = pd.DataFrame(inference, columns=columns+["label_status"])
    if len(cleaned):
        cleaned["Y"] = cleaned["Y"].astype(int)
    audit = {"schema_version": 1, "endpoint": endpoint, "raw_rows": len(table),
        "identity_policy": IDENTITY_POLICY, "scaffold_policy": SCAFFOLD_POLICY,
        "rdkit_version": rdBase.rdkitVersion,
        "raw_table_sha256": sha256_json(source_records),
        "valid_inference_identities": len(candidates), "cleaned_labelled_identities": len(cleaned),
        "invalid_smiles_rows": invalid_smiles, "invalid_label_rows": invalid_labels,
        "conflicting_label_identities": len(conflicts), "conflicting_labelled_rows_quarantined": conflicting_rows,
        "concordant_duplicate_rows_collapsed": duplicates_collapsed,
        "class_counts_cleaned": {str(y): int((cleaned["Y"] == y).sum()) for y in (0, 1)},
        "quarantined_identities": conflicts, "excluded_rows": excluded,
        "label_policy": "quarantine conflicts; collapse concordant duplicates; never majority-impute labels",
        "inference_label_policy": "all valid unique identities retained without Y, including quarantined identities"}
    if len(table) != invalid_smiles+invalid_labels+conflicting_rows+duplicates_collapsed+len(cleaned):
        raise ValueError("Cleaning row-accounting invariant failed")
    return {"cleaned": cleaned, "inference": candidates, "audit": audit}


def _validate_cleaned(table, endpoint=None):
    if (not isinstance(table, pd.DataFrame) or not table.columns.is_unique
            or not {"Drug", "Y", "molecule_id", "scaffold_id", "endpoint"} <= set(table.columns)
            or not 1 <= len(table) <= MAX_ROWS):
        raise ValueError("A nonempty cleaned identity table is required")
    endpoints = set(table["endpoint"])
    if len(endpoints) != 1:
        raise ValueError("A supervised task contains exactly one endpoint")
    actual_endpoint = _endpoint(next(iter(endpoints)))
    if endpoint is not None and actual_endpoint != _endpoint(endpoint):
        raise ValueError("Cleaned table endpoint differs from requested endpoint")
    if table["molecule_id"].duplicated().any():
        raise ValueError("Cleaned identities must be unique")
    for _, row in table.iterrows():
        _, identity, scaffold = molecule_identity(row["Drug"])
        if row["molecule_id"] != identity or row["scaffold_id"] != scaffold:
            raise ValueError("Cleaned identity or scaffold does not match its molecule")
        if _binary_label(row["Y"]) is None:
            raise ValueError("Cleaned labels must be binary")
    return actual_endpoint


def split_cleaned_dataset(cleaned, *, seed=42, fractions=(0.7, 0.15, 0.15), endpoint=None):
    """Deterministic class-balanced greedy assignment of indivisible scaffolds.

    Acyclic molecules all share the empty-scaffold group. No random-molecule
    fallback is used when a valid three-way scaffold split cannot be made.
    """
    endpoint = _validate_cleaned(cleaned, endpoint)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    if (not isinstance(fractions, (list, tuple)) or len(fractions) != 3
            or any(isinstance(x, bool) or not isinstance(x, numbers.Real) or not math.isfinite(x) or x <= 0 for x in fractions)
            or not math.isclose(sum(fractions), 1., rel_tol=0, abs_tol=1e-12)):
        raise ValueError("Three positive split fractions must sum to one")
    data = cleaned.sort_values("molecule_id").reset_index(drop=True)
    labels = data["Y"].to_numpy(dtype=int)
    totals = np.bincount(labels, minlength=2)
    if np.any(totals < 3):
        raise ValueError("At least three labelled identities per class are required")
    grouped = []
    for scaffold, block in data.groupby("scaffold_id", sort=True):
        count = np.bincount(block["Y"].to_numpy(dtype=int), minlength=2)
        tie = hashlib.sha256(f"{seed}:{scaffold}".encode()).hexdigest()
        grouped.append((scaffold, block.index.to_list(), count, tie))
    if len(grouped) < 3 or any(sum(count[y] > 0 for _, _, count, _ in grouped) < 3 for y in (0, 1)):
        raise ValueError("Strict scaffold isolation cannot supply both classes in all three partitions")
    grouped.sort(key=lambda item: (-len(item[1]), item[3]))
    desired = np.asarray(fractions)[:, None]*totals[None, :]
    current = np.zeros((3, 2), dtype=float)
    assignments = [[] for _ in SPLITS]
    groups = [[] for _ in SPLITS]
    for scaffold, positions, counts, _ in grouped:
        costs = []
        for split in range(3):
            proposed = current.copy();proposed[split] += counts
            # Class-normalized target deficits balance class counts and sizes.
            cost = float(np.sum(((proposed-desired)/np.maximum(desired, 1.))**2))
            costs.append(cost)
        chosen = min(range(3), key=lambda i: (costs[i], i))
        current[chosen] += counts
        assignments[chosen].extend(positions);groups[chosen].append(scaffold)
    partitions = {name: data.iloc[sorted(positions)].reset_index(drop=True) for name, positions in zip(SPLITS, assignments)}
    if any(set(table["Y"]) != {0, 1} for table in partitions.values()):
        raise ValueError("Greedy strict-scaffold partition lacks a class; supply a predeclared feasible partition rather than random fallback")
    manifest = {"schema_version": 1, "endpoint": endpoint, "seed": seed,
        "fractions": list(map(float, fractions)), "policy": SCAFFOLD_POLICY,
        "identity_policy": IDENTITY_POLICY, "rdkit_version": rdBase.rdkitVersion,
        "label_policy": "cleaned concordant identities only",
        **{name: sorted(table["molecule_id"].tolist()) for name, table in partitions.items()},
        "groups": {name: sorted(values) for name, values in zip(SPLITS, groups)},
        "class_counts": {name: {str(y): int((table["Y"] == y).sum()) for y in (0, 1)} for name, table in partitions.items()},
        "molecule_groups": dict(zip(data["molecule_id"], data["scaffold_id"])),
        "cohort_molecule_ids_sha256": sha256_json(sorted(data["molecule_id"].tolist()))}
    manifest["split_id"] = sha256_json(manifest)
    validate_split_manifest(manifest)
    return {**partitions, "manifest": manifest}


def validate_split_manifest(manifest):
    keys = {"schema_version", "endpoint", "seed", "fractions", "policy", "identity_policy", "rdkit_version",
            "label_policy", *SPLITS, "groups", "class_counts", "molecule_groups", "cohort_molecule_ids_sha256", "split_id"}
    if not isinstance(manifest, dict) or set(manifest) != keys or type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1:
        raise ValueError("Unsupported split manifest schema")
    if manifest["split_id"] != sha256_json({k: v for k, v in manifest.items() if k != "split_id"}):
        raise ValueError("Split manifest checksum mismatch")
    _endpoint(manifest["endpoint"])
    if manifest["policy"] != SCAFFOLD_POLICY or manifest["identity_policy"] != IDENTITY_POLICY:
        raise ValueError("Unsupported molecule/scaffold policy")
    all_ids, all_groups = set(), set()
    for name in SPLITS:
        ids = manifest[name]
        group_ids = manifest["groups"][name]
        if (not isinstance(ids, list) or not ids or ids != sorted(set(ids))
                or any(not isinstance(x, str) or _SHA.fullmatch(x) is None for x in ids)):
            raise ValueError("Split identities must be unique sorted SHA-256 IDs")
        if all_ids.intersection(ids) or all_groups.intersection(group_ids):
            raise ValueError("Molecule or scaffold leakage between partitions")
        if (not isinstance(group_ids, list) or not group_ids or group_ids != sorted(set(group_ids))
                or any(not isinstance(x, str) or _SHA.fullmatch(x) is None for x in group_ids)):
            raise ValueError("Scaffold IDs must be sorted unique SHA-256 IDs")
        if set(group_ids) != {manifest["molecule_groups"].get(identity) for identity in ids}:
            raise ValueError("Scaffold membership differs from split identities")
        counts = manifest["class_counts"][name]
        if set(counts) != {"0", "1"} or any(type(v) is not int or v < 1 for v in counts.values()) or sum(counts.values()) != len(ids):
            raise ValueError("Each split must contain both classes with matching row counts")
        all_ids.update(ids);all_groups.update(group_ids)
    if set(manifest["molecule_groups"]) != all_ids or manifest["cohort_molecule_ids_sha256"] != sha256_json(sorted(all_ids)):
        raise ValueError("Split cohort membership checksum mismatch")
    return True
