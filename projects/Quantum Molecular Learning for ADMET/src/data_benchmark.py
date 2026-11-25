"""Pinned real-data curation and estimator-sensitivity benchmark.

This is an exploratory Hamiltonian/estimator experiment, not a predictive
ADMET benchmark or a hypothesis test. Raw data are cached, never redistributed
by the report. The original KSG screening method and thresholds are preserved.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import io
from itertools import combinations
import math
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
from rdkit import rdBase

from admet_curation import (CurationConfig, fingerprint_datasets, select_features,
    estimate_pairwise, select_conditional_pairs, discover_triads,
    assemble_hamiltonians, molecule_problem_input)
from artifact_paths import validate_output_dir
from classical_preprocessing import AGPSettings, recalculate_counterdiabatic_hamiltonian

BBB_URL = "https://dataverse.harvard.edu/api/access/datafile/4259566"
BBB_SHA256 = "eb32676c7bf6e2bcd98b7625186bbb2ba4e4644c1d1468dfa18b6be665bd5e98"
BBB_BYTES = 137899
BBB_DOCUMENTATION_URL = "https://tdcommons.ai/single_pred_tasks/adme/#bbb-blood-brain-barrier-martins-et-al"
BBB_METADATA_URL = "https://raw.githubusercontent.com/mims-harvard/TDC/main/tdc/metadata.py"
DATASET = "BBB_Martins"
DEFAULT_SEEDS = (42, 43, 44, 45, 46)


def verify_bbb_file(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("BBB source must be a regular nonsymlink file")
    if path.stat().st_size != BBB_BYTES:
        raise ValueError("BBB source size differs from the pinned public artifact")
    if hashlib.sha256(path.read_bytes()).hexdigest() != BBB_SHA256:
        raise ValueError("BBB source checksum differs from the pinned public artifact")
    return path


def download_bbb(path, *, transport=None):
    """Explicit opt-in public download; existing caches are verified offline.

    The fixed SHA, byte limit, timeout, and atomic publication prevent a changed
    server response from silently becoming a different benchmark dataset.
    """
    path = Path(path)
    validate_output_dir(path.parent)
    if path.is_symlink():
        raise ValueError("BBB cache must not be a symlink")
    if path.exists():
        return verify_bbb_file(path)
    if transport is None:
        import requests
        transport = requests.get
    response = transport(BBB_URL, timeout=45, stream=True,
                         headers={"User-Agent": "Mozilla/5.0 ADMET-scientific-validation/1.0"})
    raw = bytearray()
    try:
        response.raise_for_status()
        for chunk in response.iter_content(chunk_size=65536):
            raw.extend(chunk)
            if len(raw) > BBB_BYTES:
                raise ValueError("BBB response exceeds the pinned public artifact size")
    finally:
        response.close()
    if len(raw) != BBB_BYTES or hashlib.sha256(raw).hexdigest() != BBB_SHA256:
        raise ValueError("Downloaded BBB response differs from the pinned public artifact")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".bbb-pending-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if Path(temporary).exists():
            Path(temporary).unlink()
    return verify_bbb_file(path)


def discrete_entropy(*variables, base=2.0):
    """Empirical plug-in entropy of aligned finite discrete columns."""
    if not variables or not math.isfinite(base) or base <= 1:
        raise ValueError("Entropy requires variables and a finite base > 1")
    columns = []
    length = None
    for values in variables:
        values = np.asarray(values)
        if (values.ndim not in (1, 2) or values.size == 0
                or not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all()):
            raise ValueError("Discrete variables must be nonempty finite numeric columns")
        values = values.reshape(len(values), -1)
        if length is not None and length != len(values):
            raise ValueError("Discrete variables must have matching row counts")
        length = len(values)
        columns.append(values)
    _, counts = np.unique(np.column_stack(columns), axis=0, return_counts=True)
    probabilities = counts / counts.sum()
    return float(-np.sum(probabilities*np.log(probabilities))/math.log(base))


def discrete_mi(x, y, z=None, *, base=2.0):
    if z is None:
        return discrete_entropy(x, base=base)+discrete_entropy(y, base=base)-discrete_entropy(x, y, base=base)
    return (discrete_entropy(x, z, base=base)+discrete_entropy(y, z, base=base)
            -discrete_entropy(z, base=base)-discrete_entropy(x, y, z, base=base))


def discrete_co_information(x, y, z, *, base=2.0):
    return discrete_mi(x, y, base=base)-discrete_mi(x, y, z, base=base)


def _jaccard(a, b):
    return len(a & b)/len(a | b) if a | b else 1.0


def _validated_settings(profiles, seeds, permutations):
    for name, values, lower, upper in (("profiles", profiles, 3, 8), ("jitter_seeds", seeds, 0, 2**32-1)):
        if (not isinstance(values, (tuple, list)) or not values
                or any(isinstance(v, bool) or not isinstance(v, int) or not lower <= v <= upper for v in values)
                or len(values) != len(set(values))):
            raise ValueError(f"{name} must contain unique bounded integers")
    if 42 not in seeds:
        raise ValueError("Canonical seed 42 must be included")
    if len(seeds) > 10:
        raise ValueError("At most ten jitter seeds are supported by the bounded benchmark")
    if isinstance(permutations, bool) or not isinstance(permutations, int) or not 1 <= permutations <= 100:
        raise ValueError("permutations must be an integer in 1..100")


def benchmark_table(table, *, output_dir, dataset_name=DATASET, profiles=(4, 8),
                    jitter_seeds=DEFAULT_SEEDS, permutations=20, sample_index=13):
    """Exercise unchanged chemistry/statistics/CD/QSVT for a supplied table.

    Profiles change the explicit feature budget, never screening thresholds.
    Sensitivity checks use exact empirical discrete information as a reference;
    they do not silently replace the intended continuous KSG estimator.
    """
    _validated_settings(profiles, jitter_seeds, permutations)
    output = validate_output_dir(output_dir)
    from admet_submission_entry import prepare_runtime_inputs, verify_runtime_bundle
    from runtime_input_protocol import atomic_write_json
    with rdBase.BlockLogs():
        fingerprints = fingerprint_datasets({dataset_name: table}, CurationConfig(top_k=max(profiles)))
    data = fingerprints[dataset_name]
    valid_rows = len(data["y"])
    if isinstance(sample_index, bool) or not isinstance(sample_index, int) or not 0 <= sample_index < valid_rows:
        raise ValueError("sample_index must index a retained valid molecule")
    report = {"schema_version": 1, "status": "passed", "dataset": dataset_name,
        "interpretation": "exploratory data-derived Ising encoding and estimator sensitivity; not ADMET prediction performance",
        "fit_scope": "exploratory_full_table", "raw_rows": len(table), "valid_rows": valid_rows,
        "unique_canonical_molecules": len(set(data["canonical_smiles"])),
        "duplicate_observations": data["duplicate_observations"],
        "conflicting_label_identities": data["conflicting_label_identities"],
        "rejected_rows": data["rejected_rows"],
        "class_counts_valid": {str(k): int(np.sum(data["y"] == k)) for k in (0, 1)},
        "source_table_sha256": data["source_table_sha256"],
        "fingerprint_matrix_sha256": data["fingerprint_matrix_sha256"],
        "profiles": []}
    for top_k in profiles:
        base = CurationConfig(top_k=top_k)
        selected_all = select_features(fingerprints, base)
        selected = selected_all[dataset_name]
        counts, labels = selected["X_selected"], selected["y"]
        runs, canonical = [], None
        for seed in jitter_seeds:
            config = replace(base, seed=seed)
            pairwise = estimate_pairwise(selected_all, config)
            conditional = select_conditional_pairs(selected_all, config)
            triads = discover_triads(conditional, pairwise, config)
            pair_data, triad_data = conditional[dataset_name], triads[dataset_name]
            pairs = {tuple((p["feature_i"], p["feature_j"])) for p in pair_data["retained_pairs"]}
            triple_set = {tuple((t["feature_i"], t["feature_j"], t["feature_k"])) for t in triad_data["retained_triads"]}
            diagnostics = {tuple(t[k] for k in ("feature_i", "feature_j", "feature_k")):t["triadic_interaction"]
                           for t in triad_data["triad_diagnostics"]}
            runs.append({"seed": seed, "pairs": pairs, "triads": triple_set,
                         "cmi": pair_data["raw_conditional_mi_matrix"], "coinfo": diagnostics})
            if seed == 42:
                canonical = {"schema_version": 2, "datasets": {dataset_name: table}, "config": asdict(config),
                    "fit_scope": "exploratory_full_table", "fingerprint_data": fingerprints,
                    "selected_feature_data": selected_all, "pairwise_mi_data": pairwise,
                    "conditional_pair_data": conditional, "triadic_interaction_data": triads,
                    **assemble_hamiltonians(conditional, triads)}
        pair_comparisons = []
        for i, j in combinations(range(top_k), 2):
            exact = discrete_mi(counts[:, i], counts[:, j], labels)
            estimates = [float(r["cmi"][i, j]) for r in runs]
            pair_comparisons.append({"selected_positions": [i, j], "exact_discrete_cmi_bits": exact,
                "ksg_cmi_bits_by_seed": estimates, "ksg_min": min(estimates), "ksg_max": max(estimates),
                "threshold_disagreements": sum((v > base.conditional_mi_threshold) != (exact > base.conditional_mi_threshold) for v in estimates)})
        triad_comparisons = []
        for triple in sorted(set().union(*(set(r["coinfo"]) for r in runs))):
            exact = discrete_co_information(*(counts[:, i] for i in triple))
            estimates = [r["coinfo"].get(triple) for r in runs]
            present = [value for value in estimates if value is not None]
            triad_comparisons.append({"selected_positions": list(triple), "exact_discrete_co_information_bits": exact,
                "ksg_co_information_bits_by_seed": estimates, "ksg_min": min(present), "ksg_max": max(present),
                "threshold_disagreements": sum((v > base.triad_threshold) != (exact > base.triad_threshold) for v in present)})
        rng = np.random.default_rng(20261124)
        permuted = [float(np.mean([discrete_mi(counts[:, i], rng_labels) for i in range(top_k)]))
                    for rng_labels in (rng.permutation(labels) for _ in range(permutations))]
        profile = {"top_k": top_k, "config": asdict(base), "jitter_seeds": list(jitter_seeds),
            "selected_original_features": selected["selected_features"].tolist(),
            "selected_label_mi_nats": selected["selected_mi_scores"].tolist(),
            "seed_runs": [{"seed":r["seed"], "retained_pairs": len(r["pairs"]), "retained_triads": len(r["triads"]),
                           "estimator_calls": 2*math.comb(top_k, 2)+3*len(r["coinfo"]),
                           "pair_supports": [list(p) for p in sorted(r["pairs"])],
                           "triad_supports": [list(t) for t in sorted(r["triads"])]} for r in runs],
            "pair_jaccard_between_seeds": [_jaccard(a["pairs"], b["pairs"]) for a, b in combinations(runs, 2)],
            "triad_jaccard_between_seeds": [_jaccard(a["triads"], b["triads"]) for a, b in combinations(runs, 2)],
            "pair_comparisons": pair_comparisons, "triad_comparisons": triad_comparisons,
            "pair_threshold_disagreements": sum(p["threshold_disagreements"] for p in pair_comparisons),
            "triad_threshold_disagreements": sum(t["threshold_disagreements"] for t in triad_comparisons),
            "pair_threshold_decisions": len(pair_comparisons)*len(jitter_seeds),
            "triad_threshold_decisions": sum(sum(value is not None for value in t["ksg_co_information_bits_by_seed"]) for t in triad_comparisons),
            "maximum_pair_cmi_absolute_difference_bits": max(abs(value-p["exact_discrete_cmi_bits"])
                for p in pair_comparisons for value in p["ksg_cmi_bits_by_seed"]),
            "label_permutation_diagnostic": {"permutations": permutations, "seed": 20261124,
                "observed_fixed_feature_mean_mi_bits": float(np.mean(selected["selected_mi_scores"])/math.log(2)),
                "permuted_fixed_feature_mean_mi_bits": permuted,
                "interpretation": "descriptive fixed-feature label shuffle; no significance or predictive-generalization claim"}}
        active = canonical["interaction_hypergraph"][dataset_name]["active_features"]
        if not active:
            profile["pipeline"] = {"status": "no_retained_interaction_features", "thresholds_changed": False}
        else:
            adapter, theta, provenance = molecule_problem_input(canonical, dataset_name, sample_index)
            payload = recalculate_counterdiabatic_hamiltonian(adapter, dataset_name, theta, AGPSettings())
            payload.update(sample_index=sample_index, curation_provenance=provenance,
                           curation_config=canonical["config"], data_mode="provided", fixture_labels_are_synthetic=False)
            bundle, settings = prepare_runtime_inputs(payload, output/"runtime"/f"top_{top_k}", physical_time=.1, epsilon=1e-6)
            manifest = verify_runtime_bundle(bundle)
            profile["pipeline"] = {"status": "passed", "sample_index_valid_rows": sample_index,
                "source_row_position": provenance["source_row_position"], "system_qubits": len(theta),
                "retained_pairs": provenance["retained_pairs"], "retained_triads": provenance["retained_triads"],
                "cd_terms": len(payload["pauli_terms"]), "lambda_value": settings["lambda_value"],
                "runtime_input_count": settings["synchronization"]["runtime_document"]["input_count"],
                "bundle_id": bundle.name, "error_bound": settings["recovered_evolution_error_bound"],
                "reference": settings["mathematical_reference"], "live_api_execution_allowed": manifest["live_api_execution_allowed"]}
        report["profiles"].append(profile)
    if not any(p["pipeline"]["status"] == "passed" for p in report["profiles"]):
        report["status"] = "diagnostic_only_no_retained_features"
    atomic_write_json(output/"data_benchmark.json", report)
    return report


def run_data_benchmark(path, *, output_dir):
    """Benchmark the exact pinned public BBB artifact without network access."""
    path = verify_bbb_file(path)
    raw = path.read_bytes()
    if len(raw) != BBB_BYTES or hashlib.sha256(raw).hexdigest() != BBB_SHA256:
        raise ValueError("BBB source changed during loading")
    table = pd.read_csv(io.BytesIO(raw), sep="\t")
    if table.columns.tolist() != ["Drug_ID", "Drug", "Y"] or len(table) != 2039:
        raise ValueError("Pinned BBB table schema unexpectedly differs")
    report = benchmark_table(table, output_dir=output_dir)
    report["source"] = {"url": BBB_URL, "sha256": BBB_SHA256, "bytes": BBB_BYTES,
        "documentation_url": BBB_DOCUMENTATION_URL, "metadata_url": BBB_METADATA_URL,
        "documented_rows": 1975, "raw_artifact_rows": len(table),
        "row_count_note": "The pinned downloaded artifact contains 2039 rows; official TDC prose lists 1975. Valid, unique, and rejected counts are reported separately rather than silently equated.",
        "redistribution": "Raw data not included; documentation says license not specified with a CC BY 4.0 link; no broader license claim is inferred."}
    from runtime_input_protocol import atomic_write_json
    atomic_write_json(Path(output_dir)/"data_benchmark.json", report)
    return report
